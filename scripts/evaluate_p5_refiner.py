#!/usr/bin/env python3
"""Evaluate frozen T1 plus a registered P5 refiner inference policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
import yaml

from src.data.coco import CocoAnnotations
from src.instance.p5_refiner import (
    build_refiner_model,
    combine_refined_instances,
    combine_support_preserving_instances,
    crop_bounds,
    fuse_refinement_in_boundary_band,
    instances_to_label_map,
    postprocessed_instances_from_rles,
    resize_refiner_crop,
    restore_refined_mask,
)
from src.instance.p5_multifold import (
    formal_protocol_sha256,
    refinement_policy_payload,
    refinement_policy_sha256,
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
from src.submission.rle import label_map_to_rles


def _device(raw: str) -> torch.device:
    if raw == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {raw!r} requested but CUDA is unavailable; refusing CPU fallback"
        )
    return torch.device(f"cuda:{int(raw)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/instance/stage_p5/p5_c1_crop_refiner_fold0.yaml",
        type=Path,
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--refinement-policy",
        choices=("full_crop", "boundary_band_preserve_support"),
        default="full_crop",
    )
    parser.add_argument("--no-verify-cache-hashes", action="store_true")
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("max-images must be positive")
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != 1:
        raise ValueError("P5 refiner config must use schema_version=1")
    heldout = int(config["experiment"]["heldout_fold"])
    evaluation = config["evaluation"]
    scope = str(evaluation.get("scope", "fixed_panel"))
    if scope not in {"fixed_panel", "full_fold"}:
        raise ValueError("evaluation.scope must be fixed_panel or full_fold")
    if scope == "fixed_panel" and heldout != 0:
        raise ValueError("P5-C1 fixed panel is preregistered for heldout Fold0")
    output = args.output or Path(config["evaluation"]["output"])
    if (
        args.refinement_policy != "full_crop"
        and args.output is None
    ):
        raise ValueError(
            "non-default policy requires --output so the formal C1 report is not overwritten"
        )
    annotation = Path(config["data"]["annotation_file"])
    image_dir = Path(config["data"]["image_dir"])

    cache_dir = None
    manifest = None
    for raw in config["data"]["cache_dirs"]:
        candidate_manifest = load_cache_manifest(raw)
        if int(candidate_manifest["dataset"]["fold"]) == heldout:
            cache_dir = Path(raw)
            manifest = candidate_manifest
            break
    if cache_dir is None or manifest is None:
        raise ValueError("heldout cache is absent from refiner config")
    validate_annotation_provenance(manifest, annotation)
    by_name = {str(row["file_name"]): row for row in manifest["images"]}
    panel_file = None
    if scope == "fixed_panel":
        panel_file = Path(config["data"]["panel_file"])
        selected_names = list(json.loads(panel_file.read_text()))
    else:
        selected_names = [str(row["file_name"]) for row in manifest["images"]]
    if len(selected_names) != len(set(selected_names)):
        raise ValueError(f"{scope} image selection contains duplicate names")
    missing = sorted(set(selected_names) - set(by_name))
    if missing:
        raise ValueError(f"selected images missing from heldout cache: {missing[:5]}")
    complete = args.max_images is None
    if args.max_images is not None:
        selected_names = selected_names[: args.max_images]

    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("report_type") != "p5_refiner_checkpoint_v1":
        raise ValueError("checkpoint is not a P5 refiner checkpoint")
    if checkpoint.get("config_sha256") != sha256_file(args.config):
        raise ValueError("checkpoint config hash differs from evaluation config")
    if int(checkpoint["heldout_fold"]) != heldout or heldout in set(
        checkpoint["training_folds"]
    ):
        raise ValueError("checkpoint violates heldout-fold isolation")
    if checkpoint.get("max_steps_smoke_override") is not None and complete:
        raise ValueError("smoke-test checkpoint cannot be used for formal evaluation")
    if complete and int(checkpoint["epoch"]) != int(config["training"]["epochs"]):
        raise ValueError("checkpoint did not complete the fixed epoch budget")
    model = build_refiner_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    coco = CocoAnnotations(annotation)
    postprocess = YoloPostprocessConfig(**config["postprocess"])
    crop_cfg = config["crop"]
    threshold = float(config["evaluation"]["mask_threshold"])
    control_counts = []
    candidate_counts = []
    control_instances = 0
    candidate_instances = 0
    transition_rows = []
    support_fallback_instances = 0
    started = perf_counter()
    with torch.inference_mode():
        for image_index, file_name in enumerate(selected_names, start=1):
            entry = by_name[file_name]
            height, width = int(entry["height"]), int(entry["width"])
            image = cv2.imread(str(image_dir / file_name), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (height, width):
                raise ValueError(f"cannot read expected grayscale image: {image_dir / file_name}")
            rles, scores, _, _ = load_candidate_rles(
                cache_dir,
                entry,
                verify_hash=not args.no_verify_cache_hashes,
            )
            coarse_masks, _, _ = postprocessed_instances_from_rles(
                rles,
                scores,
                height=height,
                width=width,
                config=postprocess,
            )
            control_labels = instances_to_label_map(
                coarse_masks, height=height, width=width
            )
            control_rles = label_map_to_rles(control_labels)
            control_instances += len(control_rles)

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
                probability = torch.sigmoid(model(inputs)).cpu().numpy()[:, 0]
                refined_masks = [
                    restore_refined_mask(
                        probability[index],
                        bounds,
                        height=height,
                        width=width,
                        threshold=threshold,
                    )
                    for index, bounds in enumerate(bounds_by_instance)
                ]
            image_fallback_instances = 0
            if args.refinement_policy == "boundary_band_preserve_support":
                refined_masks = [
                    fuse_refinement_in_boundary_band(
                        coarse_masks[index],
                        refined_masks[index],
                        coarse_crops[index],
                        bounds_by_instance[index],
                        boundary_kernel=int(config["loss"]["boundary_kernel"]),
                    )
                    for index in range(len(refined_masks))
                ]
                _, candidate_rles, image_fallback_instances = (
                    combine_support_preserving_instances(
                        refined_masks,
                        coarse_masks,
                        height=height,
                        width=width,
                        min_area=postprocess.min_area,
                    )
                )
                if len(candidate_rles) != len(control_rles):
                    raise RuntimeError(
                        "support-preserving policy changed the source instance count"
                    )
            else:
                _, candidate_rles = combine_refined_instances(
                    refined_masks,
                    height=height,
                    width=width,
                    min_area=postprocess.min_area,
                )
            support_fallback_instances += image_fallback_instances
            candidate_instances += len(candidate_rles)
            image_control = []
            image_candidate = []
            for _, truth in truth_batches_for_image(
                coco, file_name, height, width
            ):
                control_matrix = pairwise_rle_iou(truth, control_rles)
                candidate_matrix = pairwise_rle_iou(truth, candidate_rles)
                control_row = score_iou_matrix(
                    control_matrix, len(truth), len(control_rles)
                )
                candidate_row = score_iou_matrix(
                    candidate_matrix, len(truth), len(candidate_rles)
                )
                control_counts.append(control_row)
                candidate_counts.append(candidate_row)
                image_control.append(control_row)
                image_candidate.append(candidate_row)
            image_control_report = aggregate_counts(image_control, 1).to_dict()
            image_candidate_report = aggregate_counts(image_candidate, 1).to_dict()
            transition_rows.append(
                {
                    "file_name": file_name,
                    "control_instances": len(control_rles),
                    "candidate_instances": len(candidate_rles),
                    "support_fallback_instances": image_fallback_instances,
                    "control_mean_batch_pq": float(
                        np.mean([row.pq for row in image_control])
                    ),
                    "candidate_mean_batch_pq": float(
                        np.mean([row.pq for row in image_candidate])
                    ),
                    "control": image_control_report,
                    "candidate": image_candidate_report,
                }
            )
            print(
                f"P5-refiner[{args.refinement_policy}] {scope} "
                f"{image_index}/{len(selected_names)}: {file_name}",
                flush=True,
            )

    control = aggregate_counts(control_counts, len(selected_names)).to_dict()
    control["prediction_instances"] = control_instances
    candidate = aggregate_counts(candidate_counts, len(selected_names)).to_dict()
    candidate["prediction_instances"] = candidate_instances
    if scope == "fixed_panel":
        baseline_report_path = Path(evaluation["baseline_panel_report"])
        baseline = json.loads(baseline_report_path.read_text())["best"]
    else:
        baseline_report_path = Path(evaluation["baseline_multifold_report"])
        baseline = json.loads(baseline_report_path.read_text())["best"][
            "fold_reports"
        ][str(heldout)]
    if complete:
        exact_fields = (
            "physical_images",
            "annotator_records",
            "hit_pairs",
            "false_positives",
            "false_negatives",
            "prediction_instances",
        )
        for field in exact_fields:
            if int(control[field]) != int(baseline[field]):
                raise RuntimeError(
                    f"P5-C1 control differs from fixed T1 {scope} field {field}"
                )
        for field in ("hit_iou_sum", "official_pq"):
            if abs(float(control[field]) - float(baseline[field])) > 1e-12:
                raise RuntimeError(
                    f"P5-C1 control differs from fixed T1 {scope} field {field}"
                )
    delta = candidate["official_pq"] - control["official_pq"]
    gate = (
        float(evaluation["minimum_panel_pq_delta"])
        if scope == "fixed_panel"
        else None
    )
    if scope == "fixed_panel":
        status = (
            "passed_panel" if complete and delta >= gate else
            "failed_panel" if complete else "partial_smoke"
        )
    else:
        status = "complete_fold" if complete else "partial_smoke"
    payload = {
        "report_type": (
            "p5_crop_refiner_panel_v1"
            if scope == "fixed_panel"
            else "p5_crop_refiner_fold_v1"
        ),
        "evaluation_scope": scope,
        "complete": complete,
        "status": status,
        "config": config,
        "config_sha256": sha256_file(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_training_folds": checkpoint["training_folds"],
        "heldout_fold": heldout,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
        "source_training_signature_sha256": manifest["experiment"][
            "training_signature_sha256"
        ],
        "annotation_sha256": sha256_file(annotation),
        "selection_manifest_sha256": (
            sha256_file(panel_file)
            if panel_file is not None
            else manifest["dataset"]["fold_manifest_sha256"]
        ),
        "baseline_report": str(baseline_report_path),
        "baseline_report_sha256": sha256_file(baseline_report_path),
        "formal_protocol_sha256": (
            formal_protocol_sha256(config) if scope == "full_fold" else None
        ),
        "refinement_policy": refinement_policy_payload(
            config, args.refinement_policy
        ),
        "refinement_policy_sha256": refinement_policy_sha256(
            config, args.refinement_policy
        ),
        "reversibility": {
            "source_cache_immutable": True,
            "checkpoint_immutable": True,
            "branch_output_only": True,
            "source_instance_count_preserved": bool(
                args.refinement_policy == "boundary_band_preserve_support"
                and candidate_instances == control_instances
            ),
            "support_fallback_instances": support_fallback_instances,
        },
        "control": control,
        "candidate": candidate,
        "comparison": {
            "official_pq_delta": delta,
            "hit_pairs_delta": candidate["hit_pairs"] - control["hit_pairs"],
            "false_positives_delta": candidate["false_positives"] - control["false_positives"],
            "false_negatives_delta": candidate["false_negatives"] - control["false_negatives"],
            "prediction_instances_delta": candidate_instances - control_instances,
            "minimum_panel_pq_delta": gate,
            "pass": bool(complete and delta >= gate) if gate is not None else None,
        },
        "per_image": transition_rows,
        "elapsed_seconds": perf_counter() - started,
    }
    atomic_write_json(output, payload)
    print(json.dumps(payload["comparison"], indent=2), flush=True)
    print(
        f"wrote P5 refiner {args.refinement_policy} {scope} report to {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
