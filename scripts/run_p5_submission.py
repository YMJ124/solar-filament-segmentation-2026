#!/usr/bin/env python3
"""Generate and audit a P5-C2 competition submission from one paired fold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
import yaml

from src.instance.p5_multifold import refinement_policy_sha256
from src.instance.p5_refiner import (
    build_refiner_model,
    combine_support_preserving_instances,
    crop_bounds,
    fuse_refinement_in_boundary_band,
    postprocessed_instances_from_rles,
    resize_refiner_crop,
    restore_refined_mask,
)
from src.instance.yolo_cache import sha256_file
from src.instance.yolo_evaluation import YoloPostprocessConfig
from src.instance.yolo_inference import prediction_arrays
from src.io_utils import atomic_write_json
from src.submission.rle import (
    label_map_to_rles,
    mask_to_rle,
    rle_counts,
    write_submission,
)
from src.submission.validate import audit_image_labels, audit_submission_rows
from src.ultralytics_runtime import prepare_ultralytics_runtime


def _file_list_sha256(paths: list[Path]) -> str:
    payload = json.dumps([path.name for path in paths], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}")


def _load_coverage_fallback(
    config: dict,
    expected_image_ids: set[str],
    *,
    allow_extra_images: bool = False,
) -> dict[str, list[dict[str, str]]]:
    """Load an explicit, immutable fallback for images with no P5 candidates.

    The fallback is deliberately used only when the detector returns zero raw
    candidates.  It is not mixed into candidate ranking or refinement, so the
    P5-C2 branch remains the source of truth wherever it has a source mask.
    """
    fallback_cfg = config.get("coverage_fallback")
    if not fallback_cfg or not bool(fallback_cfg.get("enabled", False)):
        return {}
    path = Path(fallback_cfg["submission"])
    _assert_hash(
        path,
        str(fallback_cfg["submission_sha256"]),
        "coverage fallback submission",
    )
    by_image: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("coverage fallback has unexpected submission columns")
        for row in reader:
            filament_id = str(row.get("filament_id", ""))
            encoded = str(row.get("segmentation_rle", ""))
            parts = filament_id.rsplit("_", 1)
            if len(parts) != 2 or not parts[0] or not parts[1] or not encoded:
                raise ValueError("coverage fallback contains an invalid row")
            image_id = parts[0]
            if image_id not in expected_image_ids:
                if allow_extra_images:
                    continue
                raise ValueError(
                    f"coverage fallback contains unknown image id: {image_id}"
                )
            by_image.setdefault(image_id, []).append(
                {"filament_id": filament_id, "segmentation_rle": encoded}
            )
    fallback_summary = audit_submission_rows(
        [row for image_rows in by_image.values() for row in image_rows],
        expected_image_ids,
        validate_rles=True,
    )
    if fallback_summary["missing_images"] != 0:
        raise ValueError("coverage fallback does not cover every test image")
    return by_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/submission/p5_c2_t1_fold0_1536.yaml"),
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-report", type=Path)
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("max-images must be positive")

    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != 1:
        raise ValueError("submission config must use schema_version=1")
    detector_cfg = config["detector"]
    refiner_cfg = config["refiner"]
    detector_checkpoint = Path(detector_cfg["checkpoint"])
    detector_model_config = Path(detector_cfg["model_config"])
    refiner_checkpoint = Path(refiner_cfg["checkpoint"])
    refiner_training_config_path = Path(refiner_cfg["training_config"])
    _assert_hash(
        detector_checkpoint,
        str(detector_cfg["checkpoint_sha256"]),
        "detector checkpoint",
    )
    _assert_hash(
        detector_model_config,
        str(detector_cfg["model_config_sha256"]),
        "detector model config",
    )
    _assert_hash(
        refiner_checkpoint,
        str(refiner_cfg["checkpoint_sha256"]),
        "refiner checkpoint",
    )
    _assert_hash(
        refiner_training_config_path,
        str(refiner_cfg["training_config_sha256"]),
        "refiner training config",
    )

    if args.device == "cpu":
        raise ValueError("P5 submission generation requires CUDA; CPU fallback is disabled")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} requested but unavailable")
    device = torch.device(f"cuda:{int(args.device)}")
    refiner_training_config = yaml.safe_load(
        refiner_training_config_path.read_text()
    )
    policy_name = str(refiner_cfg["refinement_policy"])
    policy_hash = refinement_policy_sha256(refiner_training_config, policy_name)
    if policy_hash != str(refiner_cfg["refinement_policy_sha256"]):
        raise ValueError("refinement policy SHA256 differs from submission config")

    image_dir = Path(config["data"]["test_image_dir"])
    image_paths = sorted(image_dir.glob("*.jpeg"))
    if len(image_paths) != int(config["data"]["expected_images"]):
        raise ValueError(
            f"test image count differs: {len(image_paths)} != "
            f"{config['data']['expected_images']}"
        )
    complete = args.max_images is None
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    output = args.output or Path(config["output"]["submission"])
    audit_path = args.audit_report or Path(config["output"]["audit"])
    if not complete and args.output is None:
        raise ValueError("partial smoke requires --output to protect the final submission")
    if not complete and args.audit_report is None:
        raise ValueError("partial smoke requires --audit-report")

    prepare_ultralytics_runtime()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is unavailable in solar-filament") from exc
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    detector = YOLO(str(detector_checkpoint))
    checkpoint = torch.load(
        refiner_checkpoint, map_location=device, weights_only=False
    )
    if checkpoint.get("report_type") != "p5_refiner_checkpoint_v1":
        raise ValueError("refiner checkpoint has the wrong report type")
    if checkpoint.get("config_sha256") != sha256_file(
        refiner_training_config_path
    ):
        raise ValueError("refiner checkpoint/config provenance mismatch")
    if checkpoint.get("max_steps_smoke_override") is not None:
        raise ValueError("smoke refiner checkpoint cannot generate a submission")
    refiner = build_refiner_model(refiner_training_config["model"]).to(device)
    refiner.load_state_dict(checkpoint["model"])
    refiner.eval()

    postprocess = YoloPostprocessConfig(**config["postprocess"])
    postprocess.validate()
    crop_cfg = refiner_training_config["crop"]
    boundary_kernel = int(refiner_training_config["loss"]["boundary_kernel"])
    threshold = float(refiner_cfg["mask_threshold"])
    expected_shape = (
        int(config["data"]["expected_height"]),
        int(config["data"]["expected_width"]),
    )
    expected_image_ids = {path.stem for path in image_paths}
    coverage_fallback = _load_coverage_fallback(
        config,
        expected_image_ids,
        allow_extra_images=not complete,
    )
    rows = []
    image_reports = []
    raw_candidate_instances = 0
    source_instances = 0
    support_fallback_instances = 0
    coverage_fallback_images = []
    coverage_fallback_instances = 0
    p5_output_instances = 0
    detector_seconds = 0.0
    refiner_seconds = 0.0
    started = perf_counter()
    with torch.inference_mode():
        for image_index, image_path in enumerate(image_paths, start=1):
            detector_started = perf_counter()
            result = detector.predict(
                source=str(image_path),
                imgsz=int(detector_cfg["imgsz"]),
                conf=float(detector_cfg["candidate_confidence"]),
                max_det=int(detector_cfg["max_candidates"]),
                retina_masks=bool(detector_cfg["retina_masks"]),
                device=args.device,
                quantize=int(detector_cfg["quantize"]),
                batch=1,
                verbose=False,
            )[0]
            shape = tuple(int(value) for value in result.orig_shape)
            if shape != expected_shape:
                raise ValueError(f"unexpected test image shape {shape}: {image_path}")
            masks, scores = prediction_arrays(result, shape)
            detector_seconds += perf_counter() - detector_started
            raw_candidate_instances += len(masks)
            coarse_masks, _, _ = postprocessed_instances_from_rles(
                [mask_to_rle(mask) for mask in masks],
                scores,
                height=shape[0],
                width=shape[1],
                config=postprocess,
            )
            source_instances += len(coarse_masks)

            if not coarse_masks:
                fallback_rows = coverage_fallback.get(image_path.stem)
                if not fallback_rows:
                    raise RuntimeError(
                        "P5 detector returned no candidates and no coverage "
                        f"fallback is configured for {image_path.stem}"
                    )
                fallback_summary = audit_submission_rows(
                    fallback_rows, {image_path.stem}, validate_rles=True
                )
                image_reports.append(
                    {
                        "image_id": image_path.stem,
                        "instances": fallback_summary["rows"],
                        "min_area": fallback_summary["min_instance_area"],
                        "max_area": fallback_summary["max_instance_area"],
                        "foreground_pixels": fallback_summary[
                            "total_foreground_pixels"
                        ],
                        "raw_candidates": len(masks),
                        "source_instances": 0,
                        "support_fallback_instances": 0,
                        "coverage_fallback": True,
                    }
                )
                rows.extend(fallback_rows)
                coverage_fallback_images.append(image_path.stem)
                coverage_fallback_instances += len(fallback_rows)
                print(
                    f"P5-C2 submission {image_index}/{len(image_paths)}: "
                    f"{image_path.name} raw=0 source=0 "
                    f"coverage_fallback={len(fallback_rows)}",
                    flush=True,
                )
                del result, masks, scores, coarse_masks
                continue

            refine_started = perf_counter()
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != shape:
                raise ValueError(f"cannot read expected grayscale image: {image_path}")
            zero_target = np.zeros(shape, dtype=bool)
            crops = []
            coarse_crops = []
            bounds_by_instance = []
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
            fused_masks = []
            if crops:
                inputs = torch.from_numpy(np.stack(crops)).to(device)
                probability = torch.sigmoid(refiner(inputs)).cpu().numpy()[:, 0]
                restored = [
                    restore_refined_mask(
                        probability[index],
                        bounds,
                        height=shape[0],
                        width=shape[1],
                        threshold=threshold,
                    )
                    for index, bounds in enumerate(bounds_by_instance)
                ]
                fused_masks = [
                    fuse_refinement_in_boundary_band(
                        coarse_masks[index],
                        restored[index],
                        coarse_crops[index],
                        bounds_by_instance[index],
                        boundary_kernel=boundary_kernel,
                    )
                    for index in range(len(restored))
                ]
            labels, candidate_rles, fallback_count = (
                combine_support_preserving_instances(
                    fused_masks,
                    coarse_masks,
                    height=shape[0],
                    width=shape[1],
                    min_area=postprocess.min_area,
                )
            )
            refiner_seconds += perf_counter() - refine_started
            support_fallback_instances += fallback_count
            if len(candidate_rles) != len(coarse_masks):
                raise RuntimeError("C2 changed source instance count")
            image_report = audit_image_labels(image_path.stem, labels, shape)
            image_report.update(
                {
                    "raw_candidates": len(masks),
                    "source_instances": len(coarse_masks),
                    "support_fallback_instances": fallback_count,
                }
            )
            image_reports.append(image_report)
            for instance_id, rle in enumerate(candidate_rles, start=1):
                rows.append(
                    {
                        "filament_id": f"{image_path.stem}_{instance_id}",
                        "segmentation_rle": rle_counts(rle),
                    }
                )
            p5_output_instances += len(candidate_rles)
            print(
                f"P5-C2 submission {image_index}/{len(image_paths)}: "
                f"{image_path.name} raw={len(masks)} source={len(coarse_masks)} "
                f"fallback={fallback_count}",
                flush=True,
            )
            del result, masks, scores, coarse_masks, fused_masks, labels

    summary = audit_submission_rows(rows, expected_image_ids, validate_rles=True)
    if summary["missing_images"] != 0:
        raise RuntimeError("submission does not cover every selected test image")
    if p5_output_instances != source_instances:
        raise RuntimeError("submission instance total differs from T1 source total")
    elapsed = perf_counter() - started
    summary.update(
        {
            "complete": complete,
            "run_id": str(config["run"]["run_id"]),
            "inference_and_audit_seconds": elapsed,
            "seconds_per_test_image": elapsed / len(image_paths),
            "detector_seconds": detector_seconds,
            "refiner_seconds": refiner_seconds,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "rss_peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "raw_candidate_instances": raw_candidate_instances,
            "source_instances": source_instances,
            "support_fallback_instances": support_fallback_instances,
            "p5_output_instances": p5_output_instances,
            "coverage_fallback_images": coverage_fallback_images,
            "coverage_fallback_instances": coverage_fallback_instances,
            "source_instance_count_preserved": p5_output_instances == source_instances,
        }
    )
    output = write_submission(rows, output)
    payload = {
        "report_type": "p5_c2_submission_audit_v1",
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "submission": str(output),
        "submission_sha256": sha256_file(output),
        "test_file_list_sha256": _file_list_sha256(image_paths),
        "summary": summary,
        "provenance": {
            "detector_checkpoint": str(detector_checkpoint),
            "detector_checkpoint_sha256": sha256_file(detector_checkpoint),
            "detector_model_config": str(detector_model_config),
            "detector_model_config_sha256": sha256_file(detector_model_config),
            "refiner_checkpoint": str(refiner_checkpoint),
            "refiner_checkpoint_sha256": sha256_file(refiner_checkpoint),
            "refiner_training_config": str(refiner_training_config_path),
            "refiner_training_config_sha256": sha256_file(
                refiner_training_config_path
            ),
            "refinement_policy_sha256": policy_hash,
            "coverage_fallback_submission": (
                str(config["coverage_fallback"]["submission"])
                if coverage_fallback
                else None
            ),
            "coverage_fallback_submission_sha256": (
                sha256_file(Path(config["coverage_fallback"]["submission"]))
                if coverage_fallback
                else None
            ),
        },
        "images": image_reports,
    }
    atomic_write_json(audit_path, payload)
    print(f"wrote {len(rows)} instances to {output}", flush=True)
    print(f"submission SHA256: {payload['submission_sha256']}", flush=True)
    print(f"wrote audit report to {audit_path}", flush=True)


if __name__ == "__main__":
    main()
