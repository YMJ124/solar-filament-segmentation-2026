#!/usr/bin/env python3
"""Apply frozen fold-specific P5-C2 refiners to the registered P9-D1 OOF set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
import yaml

from src.data.coco import CocoAnnotations
from src.instance.p5_refiner import (
    build_refiner_model,
    combine_support_preserving_instances,
    crop_bounds,
    fuse_refinement_in_boundary_band,
    postprocessed_instances_from_rles,
    resize_refiner_crop,
    restore_refined_mask,
)
from src.instance.p9_incremental import build_additive_candidates
from src.instance.p9_refinement import (
    aggregate_image_reports,
    compare_refined_folds,
    paired_image_bootstrap,
    registered_candidate_keys,
    select_registered_additions,
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


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("P9-D2 config must use schema_version=1")
    return payload


def _verify_sha256(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != str(expected):
        raise ValueError(f"{label} SHA256 mismatch: {observed} != {expected}")
    return observed


def _device(raw: str) -> torch.device:
    if raw == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {raw!r} requested but CUDA is unavailable; refusing CPU fallback"
        )
    return torch.device(f"cuda:{int(raw)}")


def _report(counts, physical_images: int, prediction_instances: int) -> dict:
    result = aggregate_counts(counts, physical_images).to_dict()
    result["prediction_instances"] = int(prediction_instances)
    return result


def _score_image(truth_batches, prediction_rles):
    rows = [
        score_iou_matrix(
            pairwise_rle_iou(truth, prediction_rles),
            len(truth),
            len(prediction_rles),
        )
        for truth in truth_batches
    ]
    return rows, aggregate_counts(rows, 1).to_dict()


def _assert_exact_report(observed: dict, expected: dict, label: str) -> None:
    exact = (
        "physical_images",
        "annotator_records",
        "hit_pairs",
        "false_positives",
        "false_negatives",
        "prediction_instances",
    )
    for field in exact:
        if int(observed[field]) != int(expected[field]):
            raise RuntimeError(f"{label} differs at {field}")
    for field in ("hit_iou_sum", "official_pq"):
        if abs(float(observed[field]) - float(expected[field])) > 1e-12:
            raise RuntimeError(f"{label} differs at {field}")


def _load_p5_reference(config: dict):
    reference_path = Path(config["parents"]["p5_c2_report"])
    _verify_sha256(
        reference_path,
        config["parents"]["p5_c2_report_sha256"],
        "P5-C2 report",
    )
    reference = json.loads(reference_path.read_text())
    if reference.get("report_type") != "p5_refiner_policy_multifold_v1":
        raise ValueError("P9-D2 requires a P5-C2 multifold report")
    if reference.get("status") != "promoted" or not bool(reference.get("pass")):
        raise ValueError("P9-D2 requires the promoted P5-C2 parent")
    fold_reports = {}
    per_image = {}
    for row in reference["candidate_fold_reports"]:
        fold = int(row["fold"])
        path = Path(row["path"])
        _verify_sha256(path, row["sha256"], f"P5-C2 Fold{fold} report")
        payload = json.loads(path.read_text())
        if int(payload["heldout_fold"]) != fold or not bool(payload["complete"]):
            raise ValueError(f"invalid P5-C2 Fold{fold} report")
        fold_reports[fold] = payload["candidate"]
        for image_row in payload["per_image"]:
            key = (fold, str(image_row["file_name"]))
            if key in per_image:
                raise ValueError(f"duplicate P5-C2 per-image row: {key}")
            per_image[key] = image_row
    return reference_path, reference, fold_reports, per_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/instance/stage_p9/p9_d2_selected_additions_c2_oof.yaml"),
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-images-per-fold", type=int)
    parser.add_argument("--no-verify-cache-hashes", action="store_true")
    args = parser.parse_args()
    if args.max_images_per_fold is not None and args.max_images_per_fold < 1:
        raise ValueError("max-images-per-fold must be positive")
    if args.max_images_per_fold is not None and args.output is None:
        raise ValueError("smoke evaluation requires --output to protect the formal report")

    config = _load_yaml(args.config)
    output = args.output or Path(config["output"])
    complete = args.max_images_per_fold is None
    device = _device(args.device)
    started = perf_counter()

    d1_path = Path(config["parents"]["p9_d1_report"])
    _verify_sha256(d1_path, config["parents"]["p9_d1_report_sha256"], "P9-D1 report")
    d1 = json.loads(d1_path.read_text())
    all_keys, accepted_keys = registered_candidate_keys(d1)
    d1_config = d1["config"]
    annotation = Path(d1_config["data"]["annotation_file"])
    image_dir = Path(d1_config["data"]["image_dir"])
    coco = CocoAnnotations(annotation)
    postprocess = YoloPostprocessConfig(**d1_config["postprocess"])
    augmentation = d1_config["augmentation_policy"]
    policy = config["refinement_policy"]
    if str(policy.get("name")) != "boundary_band_preserve_support":
        raise ValueError("P9-D2 requires the promoted C2 boundary policy")
    if not bool(policy.get("preserve_all_registered_sources")) or not bool(
        policy.get("reserve_other_source_coarse_support")
    ):
        raise ValueError("P9-D2 source-preservation invariants are disabled")
    if int(policy["per_source_min_area_fallback"]) != postprocess.min_area:
        raise ValueError("P9-D2 fallback area differs from D1 postprocess")

    reference_path, p5_reference, reference_folds, reference_images = (
        _load_p5_reference(config)
    )
    t1_manifests = {
        int((manifest := load_cache_manifest(path))["dataset"]["fold"]): (
            Path(path),
            manifest,
        )
        for path in d1_config["data"]["t1_cache_dirs"]
    }
    i1_manifests = {
        int((manifest := load_cache_manifest(path))["dataset"]["fold"]): (
            Path(path),
            manifest,
        )
        for path in d1_config["data"]["i1_cache_dirs"]
    }
    folds = sorted(t1_manifests)
    if folds != [0, 1, 2] or sorted(i1_manifests) != folds:
        raise ValueError("P9-D2 requires exactly Fold0/1/2 T1 and I1 caches")
    for _, manifest in [*t1_manifests.values(), *i1_manifests.values()]:
        validate_annotation_provenance(manifest, annotation)

    refiner_rows = {int(row["fold"]): row for row in config["refiners"]}
    if sorted(refiner_rows) != folds:
        raise ValueError("P9-D2 requires one registered refiner per fold")

    fold_control_counts = {}
    fold_candidate_counts = {}
    fold_control_instances = Counter()
    fold_candidate_instances = Counter()
    images_by_fold = Counter()
    all_control_counts = []
    all_candidate_counts = []
    per_image = []
    total_base_sources = 0
    total_addition_sources = 0
    total_fallbacks = 0
    selected_reconstructed = 0
    cache_hashes_verified = not args.no_verify_cache_hashes

    with torch.inference_mode():
        for fold in folds:
            refiner_row = refiner_rows[fold]
            refiner_config_path = Path(refiner_row["config"])
            checkpoint_path = Path(refiner_row["checkpoint"])
            _verify_sha256(
                refiner_config_path,
                refiner_row["config_sha256"],
                f"Fold{fold} refiner config",
            )
            _verify_sha256(
                checkpoint_path,
                refiner_row["checkpoint_sha256"],
                f"Fold{fold} refiner checkpoint",
            )
            refiner_config = yaml.safe_load(refiner_config_path.read_text())
            if int(refiner_config["experiment"]["heldout_fold"]) != fold:
                raise ValueError(f"Fold{fold} refiner config has wrong heldout fold")
            if YoloPostprocessConfig(**refiner_config["postprocess"]) != postprocess:
                raise ValueError(f"Fold{fold} refiner postprocess differs from D1")
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            if checkpoint.get("report_type") != "p5_refiner_checkpoint_v1":
                raise ValueError(f"Fold{fold} refiner checkpoint has wrong type")
            if checkpoint.get("config_sha256") != sha256_file(refiner_config_path):
                raise ValueError(f"Fold{fold} checkpoint/config provenance mismatch")
            if int(checkpoint["heldout_fold"]) != fold or fold in set(
                checkpoint["training_folds"]
            ):
                raise ValueError(f"Fold{fold} refiner violates OOF isolation")
            if checkpoint.get("max_steps_smoke_override") is not None:
                raise ValueError(f"Fold{fold} smoke checkpoint is not allowed")
            if int(checkpoint["epoch"]) != int(refiner_config["training"]["epochs"]):
                raise ValueError(f"Fold{fold} refiner did not finish fixed epochs")
            model = build_refiner_model(refiner_config["model"]).to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()

            crop_cfg = refiner_config["crop"]
            boundary_kernel = int(refiner_config["loss"]["boundary_kernel"])
            threshold = float(refiner_config["evaluation"]["mask_threshold"])
            if boundary_kernel != int(policy["boundary_kernel"]):
                raise ValueError("registered C2 boundary kernel mismatch")
            if threshold != float(policy["mask_threshold"]):
                raise ValueError("registered C2 mask threshold mismatch")

            t1_dir, t1_manifest = t1_manifests[fold]
            i1_dir, i1_manifest = i1_manifests[fold]
            i1_by_name = {str(row["file_name"]): row for row in i1_manifest["images"]}
            entries = list(t1_manifest["images"])
            if args.max_images_per_fold is not None:
                entries = entries[: args.max_images_per_fold]
            fold_control_counts[fold] = []
            fold_candidate_counts[fold] = []

            for image_number, entry in enumerate(entries, start=1):
                file_name = str(entry["file_name"])
                if file_name not in i1_by_name:
                    raise ValueError(f"I1 cache lacks T1 image {file_name}")
                i1_entry = i1_by_name[file_name]
                height, width = int(entry["height"]), int(entry["width"])
                if (height, width) != (
                    int(i1_entry["height"]),
                    int(i1_entry["width"]),
                ):
                    raise ValueError(f"T1/I1 shape mismatch for {file_name}")
                image = cv2.imread(str(image_dir / file_name), cv2.IMREAD_GRAYSCALE)
                if image is None or image.shape != (height, width):
                    raise ValueError(f"cannot read expected image {file_name}")

                t1_rles_raw, t1_scores_raw, _, _ = load_candidate_rles(
                    t1_dir,
                    entry,
                    verify_hash=cache_hashes_verified,
                )
                i1_rles_raw, i1_scores_raw, _, _ = load_candidate_rles(
                    i1_dir,
                    i1_entry,
                    verify_hash=cache_hashes_verified,
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
                additions, _ = build_additive_candidates(
                    t1_masks,
                    i1_masks,
                    i1_scores,
                    i1_indices,
                    min_area=postprocess.min_area,
                    max_instances=postprocess.max_instances,
                    duplicate_iou=float(augmentation["duplicate_iou"]),
                )
                image_key = (fold, file_name)
                selected = select_registered_additions(
                    additions,
                    expected_all=all_keys.get(image_key, set()),
                    expected_accepted=accepted_keys.get(image_key, set()),
                )
                selected_reconstructed += len(selected)
                coarse_masks = [*t1_masks, *(candidate.mask for candidate in selected)]
                base_count = len(t1_masks)
                addition_count = len(selected)
                total_base_sources += base_count
                total_addition_sources += addition_count

                coarse_rles = [mask_to_rle(mask) for mask in coarse_masks]
                truth_batches = [
                    truth
                    for _, truth in truth_batches_for_image(
                        coco, file_name, height, width
                    )
                ]
                image_control_rows, image_control = _score_image(
                    truth_batches, coarse_rles
                )

                crops = []
                coarse_crops = []
                bounds_by_instance = []
                zero_target = np.zeros((height, width), dtype=bool)
                for coarse in coarse_masks:
                    bounds = crop_bounds(
                        coarse,
                        context_scale=float(crop_cfg["context_scale"]),
                        minimum_side=int(crop_cfg["minimum_side"]),
                    )
                    image_crop, coarse_crop, _ = resize_refiner_crop(
                        image,
                        coarse,
                        zero_target,
                        bounds,
                        output_size=int(crop_cfg["output_size"]),
                    )
                    crops.append(
                        np.stack(
                            [
                                image_crop.astype(np.float32) / 255.0,
                                coarse_crop.astype(np.float32),
                            ]
                        )
                    )
                    coarse_crops.append(coarse_crop)
                    bounds_by_instance.append(bounds)
                refined_masks = []
                if crops:
                    inputs = torch.from_numpy(np.stack(crops)).to(device)
                    probabilities = torch.sigmoid(model(inputs)).cpu().numpy()[:, 0]
                    refined_masks = [
                        restore_refined_mask(
                            probabilities[index],
                            bounds,
                            height=height,
                            width=width,
                            threshold=threshold,
                        )
                        for index, bounds in enumerate(bounds_by_instance)
                    ]
                    refined_masks = [
                        fuse_refinement_in_boundary_band(
                            coarse_masks[index],
                            refined_masks[index],
                            coarse_crops[index],
                            bounds_by_instance[index],
                            boundary_kernel=boundary_kernel,
                        )
                        for index in range(len(refined_masks))
                    ]
                _, candidate_rles, image_fallbacks = (
                    combine_support_preserving_instances(
                        refined_masks,
                        coarse_masks,
                        height=height,
                        width=width,
                        min_area=postprocess.min_area,
                    )
                )
                if len(candidate_rles) != len(coarse_masks):
                    raise RuntimeError("P9-D2 changed the registered source count")
                total_fallbacks += image_fallbacks
                image_candidate_rows, image_candidate = _score_image(
                    truth_batches, candidate_rles
                )

                reference_row = reference_images.get(image_key)
                if reference_row is None:
                    raise ValueError(f"P5-C2 reference lacks image {image_key}")
                reference_counts = reference_row["candidate"]
                image_control["prediction_instances"] = len(coarse_rles)
                image_candidate["prediction_instances"] = len(candidate_rles)
                per_image.append(
                    {
                        "fold": fold,
                        "file_name": file_name,
                        "base_t1_sources": base_count,
                        "selected_addition_sources": addition_count,
                        "output_instances": len(candidate_rles),
                        "support_fallback_instances": image_fallbacks,
                        "reference_prediction_instances": int(
                            reference_row["candidate_instances"]
                        ),
                        "control": image_control,
                        "reference": reference_counts,
                        "candidate": image_candidate,
                        "delta_vs_p5_c2": {
                            "official_pq": float(image_candidate["official_pq"])
                            - float(reference_counts["official_pq"]),
                            "hit_pairs": int(image_candidate["hit_pairs"])
                            - int(reference_counts["hit_pairs"]),
                            "false_positives": int(image_candidate["false_positives"])
                            - int(reference_counts["false_positives"]),
                            "false_negatives": int(image_candidate["false_negatives"])
                            - int(reference_counts["false_negatives"]),
                        },
                    }
                )
                fold_control_counts[fold].extend(image_control_rows)
                fold_candidate_counts[fold].extend(image_candidate_rows)
                all_control_counts.extend(image_control_rows)
                all_candidate_counts.extend(image_candidate_rows)
                fold_control_instances[fold] += len(coarse_rles)
                fold_candidate_instances[fold] += len(candidate_rles)
                images_by_fold[fold] += 1
                print(
                    f"P9-D2 Fold{fold} {image_number}/{len(entries)}: {file_name} "
                    f"T1={base_count} additions={addition_count} fallback={image_fallbacks}",
                    flush=True,
                )
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()

    expected_selected = int(d1["selection_summary"]["accepted"])
    if complete and selected_reconstructed != expected_selected:
        raise RuntimeError("P9-D2 did not reconstruct all registered D1 selections")
    control_folds = {
        fold: _report(
            fold_control_counts[fold],
            images_by_fold[fold],
            fold_control_instances[fold],
        )
        for fold in folds
    }
    candidate_folds = {
        fold: _report(
            fold_candidate_counts[fold],
            images_by_fold[fold],
            fold_candidate_instances[fold],
        )
        for fold in folds
    }
    control_global = _report(
        all_control_counts,
        sum(images_by_fold.values()),
        sum(fold_control_instances.values()),
    )
    candidate_global = _report(
        all_candidate_counts,
        sum(images_by_fold.values()),
        sum(fold_candidate_instances.values()),
    )
    active_reference_folds = {
        fold: aggregate_image_reports(
            [
                row["reference"]
                for row in per_image
                if int(row["fold"]) == fold
            ],
            prediction_instances=sum(
                int(row["reference_prediction_instances"])
                for row in per_image
                if int(row["fold"]) == fold
            ),
        )
        for fold in folds
    }
    active_reference_global = aggregate_image_reports(
        [row["reference"] for row in per_image],
        prediction_instances=sum(
            int(row["reference_prediction_instances"]) for row in per_image
        ),
    )
    if complete:
        for fold in folds:
            _assert_exact_report(
                control_folds[fold],
                d1["oof_scorer_candidate"]["folds"][str(fold)],
                f"Fold{fold} D1 control",
            )
        _assert_exact_report(
            control_global,
            d1["oof_scorer_candidate"]["global"],
            "global D1 control",
        )

        for fold in folds:
            _assert_exact_report(
                active_reference_folds[fold],
                reference_folds[fold],
                f"Fold{fold} P5-C2 reference",
            )
        _assert_exact_report(
            active_reference_global,
            p5_reference["candidate_vs_t1"]["candidate"],
            "global P5-C2 reference",
        )

    comparison = compare_refined_folds(
        active_reference_folds, candidate_folds, config["gate"]
    )
    reference_global = active_reference_global
    comparison.update(
        {
            "reference_global_pq": float(reference_global["official_pq"]),
            "candidate_global_pq": float(candidate_global["official_pq"]),
            "global_pq_delta": float(candidate_global["official_pq"])
            - float(reference_global["official_pq"]),
            "hit_pairs_delta": int(candidate_global["hit_pairs"])
            - int(reference_global["hit_pairs"]),
            "false_positives_delta": int(candidate_global["false_positives"])
            - int(reference_global["false_positives"]),
            "false_negatives_delta": int(candidate_global["false_negatives"])
            - int(reference_global["false_negatives"]),
            "prediction_instances_delta": int(candidate_global["prediction_instances"])
            - int(reference_global["prediction_instances"]),
        }
    )
    source_count_preserved = (
        int(candidate_global["prediction_instances"])
        == total_base_sources + total_addition_sources
    )
    comparison["checks"]["registered_source_count_preserved"] = source_count_preserved
    comparison["pass"] = bool(
        complete and all(comparison["checks"].values())
    )

    image_risk = {
        "images_improved_vs_p5_c2": sum(
            row["delta_vs_p5_c2"]["official_pq"] > 0 for row in per_image
        ),
        "images_unchanged_vs_p5_c2": sum(
            row["delta_vs_p5_c2"]["official_pq"] == 0 for row in per_image
        ),
        "images_worse_vs_p5_c2": sum(
            row["delta_vs_p5_c2"]["official_pq"] < 0 for row in per_image
        ),
        "images_losing_reference_hits": sum(
            row["delta_vs_p5_c2"]["hit_pairs"] < 0 for row in per_image
        ),
        "images_with_source_removal": sum(
            row["output_instances"]
            != row["base_t1_sources"] + row["selected_addition_sources"]
            for row in per_image
        ),
    }
    bootstrap = paired_image_bootstrap(
        per_image,
        iterations=int(config["diagnostics"]["bootstrap_iterations"]),
        seed=int(config["diagnostics"]["bootstrap_seed"]),
    )
    payload = {
        "report_type": "p9_selected_additions_c2_oof_v1",
        "complete": complete,
        "status": (
            "promoted"
            if comparison["pass"]
            else "failed_c2_gate"
            if complete
            else "smoke_only"
        ),
        "config": config,
        "config_sha256": sha256_file(args.config),
        "parents": {
            "p9_d1_report": str(d1_path),
            "p9_d1_report_sha256": sha256_file(d1_path),
            "p5_c2_report": str(reference_path),
            "p5_c2_report_sha256": sha256_file(reference_path),
        },
        "annotation_sha256": sha256_file(annotation),
        "cache_payload_hashes_verified": cache_hashes_verified,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "selection_audit": {
            "registered_d1_candidates": int(d1["target_summary"]["candidates"]),
            "registered_d1_accepted": expected_selected,
            "selected_reconstructed": selected_reconstructed,
            "base_t1_sources": total_base_sources,
            "addition_sources": total_addition_sources,
            "output_instances": int(candidate_global["prediction_instances"]),
            "support_fallback_instances": total_fallbacks,
            "source_count_preserved": source_count_preserved,
        },
        "d1_control": {
            "folds": control_folds,
            "global": control_global,
        },
        "p5_c2_reference": {
            "folds": active_reference_folds,
            "global": reference_global,
        },
        "d2_candidate": {
            "folds": candidate_folds,
            "global": candidate_global,
        },
        "comparison_vs_p5_c2": comparison,
        "paired_image_bootstrap": bootstrap,
        "risk": image_risk,
        "per_image": per_image,
        "elapsed_seconds": perf_counter() - started,
        "next_action": (
            "freeze_d2_and_define_leakage_safe_test_inference"
            if comparison["pass"]
            else "keep_p5_c2_and_adapt_refiner_only_for_incremental_i1_sources"
            if complete
            else "run_complete_d2_oof"
        ),
    }
    atomic_write_json(output, payload)
    print(f"wrote P9-D2 report to {output}", flush=True)
    print(
        f"P9-D2 mean PQ {comparison['reference_mean_fold_pq']:.9f} -> "
        f"{comparison['candidate_mean_fold_pq']:.9f}; pass={comparison['pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
