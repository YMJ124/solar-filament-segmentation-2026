#!/usr/bin/env python3
"""Evaluate an OOF-trained I1 augmentation policy with immutable T1 parents."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesRegressor

from src.data.coco import CocoAnnotations
from src.instance.p5_refiner import postprocessed_instances_from_rles
from src.instance.p9_incremental import (
    FEATURE_NAMES,
    build_additive_candidates,
    extract_candidate_features,
    marginal_counts_and_utility,
)
from src.instance.yolo_cache import (
    load_cache_manifest,
    load_candidate_rles,
    sha256_file,
    validate_annotation_provenance,
)
from src.instance.yolo_evaluation import YoloPostprocessConfig, truth_batches_for_image
from src.io_utils import atomic_write_json
from src.metrics.pq import aggregate_counts, pairwise_rle_iou, score_iou_matrix
from src.submission.rle import mask_to_rle


def _load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("P9 config must use schema_version=1")
    return payload


def _report(counts, physical_images: int, prediction_instances: int) -> dict:
    result = aggregate_counts(counts, physical_images).to_dict()
    result["prediction_instances"] = int(prediction_instances)
    return result


def _fold_summary(rows: dict[int, dict]) -> dict:
    values = np.asarray([rows[fold]["official_pq"] for fold in sorted(rows)])
    return {
        "fold_pq": {str(fold): rows[fold]["official_pq"] for fold in sorted(rows)},
        "mean_fold_pq": float(values.mean()),
        "fold_std_pq": float(values.std()),
    }


def _score_states(states, accepted: np.ndarray | None = None):
    counts_by_fold: dict[int, list] = {}
    instances_by_fold: Counter = Counter()
    images_by_fold: Counter = Counter()
    all_counts = []
    all_instances = 0
    for state in states:
        fold = int(state["fold"])
        additions = state["candidate_rles"]
        if accepted is not None:
            additions = [
                rle
                for row_index, rle in zip(
                    state["candidate_indices"], additions, strict=True
                )
                if bool(accepted[row_index])
            ]
        predictions = [*state["base_rles"], *additions]
        image_counts = []
        for truth in state["truth_batches"]:
            row = score_iou_matrix(
                pairwise_rle_iou(truth, predictions), len(truth), len(predictions)
            )
            image_counts.append(row)
            all_counts.append(row)
        counts_by_fold.setdefault(fold, []).extend(image_counts)
        instances_by_fold[fold] += len(predictions)
        images_by_fold[fold] += 1
        all_instances += len(predictions)
    fold_reports = {
        fold: _report(
            counts_by_fold[fold], images_by_fold[fold], instances_by_fold[fold]
        )
        for fold in sorted(counts_by_fold)
    }
    global_report = _report(all_counts, sum(images_by_fold.values()), all_instances)
    return fold_reports, global_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/instance/stage_p9/p9_d1_t1_i1_incremental_oof.yaml"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-images-per-fold", type=int)
    parser.add_argument("--no-verify-cache-hashes", action="store_true")
    args = parser.parse_args()
    if args.max_images_per_fold is not None and args.max_images_per_fold < 1:
        raise ValueError("max-images-per-fold must be positive")

    config = _load_config(args.config)
    output = args.output or Path(config["output"])
    annotation = Path(config["data"]["annotation_file"])
    image_dir = Path(config["data"]["image_dir"])
    coco = CocoAnnotations(annotation)
    postprocess = YoloPostprocessConfig(**config["postprocess"])
    augmentation = config["augmentation_policy"]
    model_cfg = config["model"]
    if not bool(augmentation.get("preserve_all_t1_instances")):
        raise ValueError("P9 requires an immutable T1 parent")
    if str(model_cfg["type"]) != "extra_trees_regressor":
        raise ValueError("unsupported P9 scorer type")

    t1_dirs = [Path(path) for path in config["data"]["t1_cache_dirs"]]
    i1_dirs = [Path(path) for path in config["data"]["i1_cache_dirs"]]
    t1_manifests = {
        int((manifest := load_cache_manifest(path))["dataset"]["fold"]): (path, manifest)
        for path in t1_dirs
    }
    i1_manifests = {
        int((manifest := load_cache_manifest(path))["dataset"]["fold"]): (path, manifest)
        for path in i1_dirs
    }
    folds = sorted(t1_manifests)
    if folds != [0, 1, 2] or sorted(i1_manifests) != folds:
        raise ValueError("P9 requires exactly Fold0/1/2 T1 and I1 caches")
    for _, manifest in [*t1_manifests.values(), *i1_manifests.values()]:
        validate_annotation_provenance(manifest, annotation)

    started = perf_counter()
    rows: list[dict] = []
    feature_rows: list[np.ndarray] = []
    targets: list[float] = []
    states = []
    source_audit = Counter()
    reference_pq = 0.3248451791985859

    for fold in folds:
        t1_dir, t1_manifest = t1_manifests[fold]
        i1_dir, i1_manifest = i1_manifests[fold]
        i1_by_name = {str(row["file_name"]): row for row in i1_manifest["images"]}
        entries = list(t1_manifest["images"])
        if args.max_images_per_fold is not None:
            entries = entries[: args.max_images_per_fold]
        for image_number, entry in enumerate(entries, start=1):
            file_name = str(entry["file_name"])
            if file_name not in i1_by_name:
                raise ValueError(f"I1 cache lacks T1 image {file_name}")
            i1_entry = i1_by_name[file_name]
            height, width = int(entry["height"]), int(entry["width"])
            if (height, width) != (
                int(i1_entry["height"]), int(i1_entry["width"])
            ):
                raise ValueError(f"T1/I1 shape mismatch for {file_name}")
            image = cv2.imread(str(image_dir / file_name), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (height, width):
                raise ValueError(f"cannot read expected image {file_name}")

            t1_rles_raw, t1_scores_raw, _, _ = load_candidate_rles(
                t1_dir,
                entry,
                verify_hash=not args.no_verify_cache_hashes,
            )
            i1_rles_raw, i1_scores_raw, _, _ = load_candidate_rles(
                i1_dir,
                i1_entry,
                verify_hash=not args.no_verify_cache_hashes,
            )
            t1_masks, _, _ = postprocessed_instances_from_rles(
                t1_rles_raw,
                t1_scores_raw,
                height=height,
                width=width,
                config=postprocess,
            )
            i1_masks, i1_scores, i1_indices = postprocessed_instances_from_rles(
                i1_rles_raw,
                i1_scores_raw,
                height=height,
                width=width,
                config=postprocess,
            )
            additions, image_audit = build_additive_candidates(
                t1_masks,
                i1_masks,
                i1_scores,
                i1_indices,
                min_area=postprocess.min_area,
                max_instances=postprocess.max_instances,
                duplicate_iou=float(augmentation["duplicate_iou"]),
            )
            source_audit.update(image_audit)
            base_rles = [mask_to_rle(mask) for mask in t1_masks]
            truth_batches = [
                truth
                for _, truth in truth_batches_for_image(
                    coco, file_name, height, width
                )
            ]
            base_matrices = [
                pairwise_rle_iou(truth, base_rles) for truth in truth_batches
            ]
            t1_claimed = np.zeros((height, width), dtype=np.uint8)
            for mask in t1_masks:
                t1_claimed[np.asarray(mask, dtype=bool)] = 1
            if t1_claimed.any():
                distance_to_t1 = cv2.distanceTransform(
                    1 - t1_claimed, cv2.DIST_L2, 3
                )
            else:
                distance_to_t1 = np.full(
                    (height, width), np.hypot(height, width), dtype=np.float32
                )

            candidate_indices = []
            candidate_rles = []
            for candidate in additions:
                candidate_rle = mask_to_rle(candidate.mask)
                vectors = [
                    pairwise_rle_iou(truth, [candidate_rle])[:, 0]
                    for truth in truth_batches
                ]
                marginal = marginal_counts_and_utility(
                    base_matrices, vectors, reference_pq=reference_pq
                )
                features = extract_candidate_features(
                    candidate,
                    image=image,
                    i1_output_count=len(i1_masks),
                    t1_output_count=len(t1_masks),
                    raw_t1_rles=t1_rles_raw,
                    raw_t1_scores=t1_scores_raw,
                    distance_to_t1=distance_to_t1,
                )
                row_index = len(rows)
                rows.append(
                    {
                        "row_index": row_index,
                        "fold": fold,
                        "file_name": file_name,
                        "i1_source_index": candidate.source_index,
                        "i1_rank": candidate.i1_rank,
                        "features": {
                            name: float(value)
                            for name, value in zip(
                                FEATURE_NAMES, features, strict=True
                            )
                        },
                        "target": marginal,
                    }
                )
                feature_rows.append(features)
                targets.append(float(marginal["marginal_utility"]))
                candidate_indices.append(row_index)
                candidate_rles.append(candidate_rle)
            states.append(
                {
                    "fold": fold,
                    "file_name": file_name,
                    "base_rles": base_rles,
                    "truth_batches": truth_batches,
                    "candidate_indices": candidate_indices,
                    "candidate_rles": candidate_rles,
                }
            )
            print(
                f"P9 data Fold{fold} {image_number}/{len(entries)}: "
                f"{file_name} additions={len(additions)}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("P9 produced no additive candidates")
    x = np.asarray(feature_rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    row_folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    feature_importances = []
    fit_audit = []
    for heldout in folds:
        train = row_folds != heldout
        valid = row_folds == heldout
        training_folds = sorted(set(row_folds[train].tolist()))
        if heldout in training_folds or not train.any() or not valid.any():
            raise RuntimeError("invalid P9 OOF split")
        model = ExtraTreesRegressor(
            n_estimators=int(model_cfg["n_estimators"]),
            min_samples_leaf=int(model_cfg["min_samples_leaf"]),
            max_features=float(model_cfg["max_features"]),
            random_state=int(config["experiment"]["seed"]),
            n_jobs=1,
        )
        model.fit(x[train], y[train])
        predictions[valid] = model.predict(x[valid])
        feature_importances.append(model.feature_importances_)
        fit_audit.append(
            {
                "heldout_fold": heldout,
                "training_folds": training_folds,
                "training_candidates": int(train.sum()),
                "heldout_candidates": int(valid.sum()),
            }
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("P9 OOF predictions are incomplete")
    threshold = float(model_cfg["threshold"])
    accepted = predictions > threshold
    for index, row in enumerate(rows):
        row["oof_predicted_marginal_utility"] = float(predictions[index])
        row["accepted"] = bool(accepted[index])
        row["scorer_training_folds"] = [
            fold for fold in folds if fold != int(row["fold"])
        ]

    base_fold, base_global = _score_states(states, np.zeros(len(rows), dtype=bool))
    union_fold, union_global = _score_states(states, None)
    candidate_fold, candidate_global = _score_states(states, accepted)
    base_summary = _fold_summary(base_fold)
    union_summary = _fold_summary(union_fold)
    candidate_summary = _fold_summary(candidate_fold)
    fold_deltas = {
        str(fold): candidate_fold[fold]["official_pq"]
        - base_fold[fold]["official_pq"]
        for fold in folds
    }
    gate = config["gate"]
    checks = {
        "minimum_mean_fold_pq_delta": candidate_summary["mean_fold_pq"]
        - base_summary["mean_fold_pq"]
        >= float(gate["minimum_mean_fold_pq_delta"]),
        "minimum_positive_folds": sum(value > 0 for value in fold_deltas.values())
        >= int(gate["minimum_positive_folds"]),
        "minimum_fold_delta": min(fold_deltas.values())
        >= float(gate["minimum_fold_delta"]),
        "maximum_fold_std_increase": candidate_summary["fold_std_pq"]
        - base_summary["fold_std_pq"]
        <= float(gate["maximum_fold_std_increase"]),
    }
    passed = all(checks.values())
    importance = np.mean(np.stack(feature_importances), axis=0)
    importance_order = np.argsort(-importance)
    complete = args.max_images_per_fold is None
    payload = {
        "report_type": "p9_t1_i1_incremental_oof_v1",
        "complete": complete,
        "status": (
            "passed_detector_gate_attach_c2_next"
            if complete and passed
            else "failed_detector_gate"
            if complete
            else "smoke_only"
        ),
        "config": config,
        "config_sha256": sha256_file(args.config),
        "annotation_sha256": sha256_file(annotation),
        "cache_manifest_sha256": {
            family: {
                str(fold): sha256_file(manifests[fold][0] / "manifest.json")
                for fold in folds
            }
            for family, manifests in (("t1", t1_manifests), ("i1", i1_manifests))
        },
        "cache_payload_hashes_verified": not args.no_verify_cache_hashes,
        "reference_t1_global_pq": reference_pq,
        "feature_names": list(FEATURE_NAMES),
        "fit_audit": fit_audit,
        "source_candidate_audit": dict(source_audit),
        "target_summary": {
            "candidates": len(rows),
            "positive_utility": int(np.count_nonzero(y > 0)),
            "nonpositive_utility": int(np.count_nonzero(y <= 0)),
            "utility_mean": float(y.mean()),
            "utility_median": float(np.median(y)),
        },
        "selection_summary": {
            "accepted": int(accepted.sum()),
            "rejected": int((~accepted).sum()),
            "accepted_by_fold": {
                str(fold): int(np.count_nonzero(accepted & (row_folds == fold)))
                for fold in folds
            },
        },
        "feature_importance": [
            {
                "feature": FEATURE_NAMES[int(index)],
                "mean_importance": float(importance[int(index)]),
            }
            for index in importance_order
        ],
        "base_t1": {"folds": base_fold, "global": base_global, **base_summary},
        "unfiltered_union_control": {
            "folds": union_fold,
            "global": union_global,
            **union_summary,
        },
        "oof_scorer_candidate": {
            "folds": candidate_fold,
            "global": candidate_global,
            **candidate_summary,
        },
        "comparison": {
            "fold_pq_delta": fold_deltas,
            "mean_fold_pq_delta": candidate_summary["mean_fold_pq"]
            - base_summary["mean_fold_pq"],
            "global_pq_delta": candidate_global["official_pq"]
            - base_global["official_pq"],
            "fold_std_delta": candidate_summary["fold_std_pq"]
            - base_summary["fold_std_pq"],
            "hit_pairs_delta": candidate_global["hit_pairs"]
            - base_global["hit_pairs"],
            "false_positives_delta": candidate_global["false_positives"]
            - base_global["false_positives"],
            "false_negatives_delta": candidate_global["false_negatives"]
            - base_global["false_negatives"],
            "prediction_instances_delta": candidate_global["prediction_instances"]
            - base_global["prediction_instances"],
            "gate": gate,
            "checks": checks,
            "pass": bool(complete and passed),
        },
        "candidate_rows": rows,
        "elapsed_seconds": perf_counter() - started,
        "next_action": (
            "apply_existing_fold_specific_c2_refiners_to_selected_additions"
            if complete and passed
            else "stop_p9_d1_without_c2_or_submission"
            if complete
            else "run_complete_oof_evaluation"
        ),
    }
    atomic_write_json(output, payload)
    print(f"wrote P9 OOF report to {output}", flush=True)
    print(
        f"P9 mean PQ {base_summary['mean_fold_pq']:.9f} -> "
        f"{candidate_summary['mean_fold_pq']:.9f}; pass={complete and passed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
