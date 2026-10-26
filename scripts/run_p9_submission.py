#!/usr/bin/env python3
"""Generate the frozen P9-D3 test challenger with resumable per-image caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
from collections import Counter
from pathlib import Path
from time import perf_counter

import cv2
import joblib
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
from src.instance.p9_deployment import (
    deployment_fingerprint,
    selected_additions_from_features,
)
from src.instance.p9_incremental import (
    FEATURE_NAMES,
    build_additive_candidates,
    extract_candidate_features,
)
from src.instance.yolo_cache import sha256_file
from src.instance.yolo_evaluation import YoloPostprocessConfig
from src.instance.yolo_inference import prediction_arrays
from src.io_utils import atomic_write_json
from src.submission.rle import mask_to_rle, rle_counts, write_submission
from src.submission.validate import audit_image_labels, audit_submission_rows
from src.ultralytics_runtime import prepare_ultralytics_runtime


def _assert_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise ValueError(f"{label} SHA256 mismatch: {observed} != {expected}")
    return observed


def _file_list_sha256(paths: list[Path]) -> str:
    payload = json.dumps([path.name for path in paths], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _submission_by_image(
    path: Path, expected_image_ids: set[str]
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("P5 parent submission has unexpected columns")
        for row in reader:
            filament_id = str(row.get("filament_id", ""))
            encoded = str(row.get("segmentation_rle", ""))
            parts = filament_id.rsplit("_", 1)
            if len(parts) != 2 or not parts[0] or not parts[1].isdigit() or not encoded:
                raise ValueError("P5 parent submission contains an invalid row")
            grouped.setdefault(parts[0], []).append(
                {"filament_id": filament_id, "segmentation_rle": encoded}
            )
    audit = audit_submission_rows(
        [row for rows in grouped.values() for row in rows],
        expected_image_ids,
        validate_rles=True,
    )
    if audit["missing_images"] != 0 or set(grouped) != expected_image_ids:
        raise ValueError("P5 parent submission does not exactly cover the test images")
    for image_id, rows in grouped.items():
        expected_ids = [f"{image_id}_{index}" for index in range(1, len(rows) + 1)]
        if [row["filament_id"] for row in rows] != expected_ids:
            raise ValueError(f"P5 parent instance order is invalid for {image_id}")
    return grouped


def _predict(detector, image_path: Path, detector_config: dict, device: str):
    result = detector.predict(
        source=str(image_path),
        imgsz=int(detector_config["imgsz"]),
        conf=float(detector_config["candidate_confidence"]),
        max_det=int(detector_config["max_candidates"]),
        retina_masks=bool(detector_config["retina_masks"]),
        device=device,
        quantize=int(detector_config["quantize"]),
        batch=1,
        verbose=False,
    )[0]
    shape = tuple(int(value) for value in result.orig_shape)
    masks, scores = prediction_arrays(result, shape)
    return result, shape, masks, scores


def _load_valid_cache(
    path: Path,
    *,
    fingerprint: str,
    image_name: str,
    image_sha256: str,
) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if payload.get("report_type") != "p9_d3_test_image_cache_v1":
        return None
    if payload.get("deployment_fingerprint") != fingerprint:
        return None
    if payload.get("image_name") != image_name:
        return None
    if payload.get("image_sha256") != image_sha256:
        return None
    if not bool(payload.get("complete")):
        return None
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/submission/p9_d3_p5_fold0_incremental_1536.yaml"),
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("max-images must be positive")

    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != 1:
        raise ValueError("P9-D3 submission config must use schema_version=1")
    if args.device == "cpu" or not bool(config["invariants"]["forbid_cpu_fallback"]):
        raise ValueError("P9-D3 submission generation requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} requested but unavailable")
    if not all(
        bool(config["invariants"].get(key))
        for key in (
            "reproduce_p5_base_only_rles_exactly",
            "preserve_all_t1_source_ids",
            "reserve_other_source_coarse_support",
            "skip_incremental_on_p5_coverage_fallback",
        )
    ):
        raise ValueError("P9-D3 deployment invariants are not fully enabled")

    complete = args.max_images is None
    output = args.output or Path(config["output"]["submission"])
    audit_path = args.audit_report or Path(config["output"]["audit"])
    cache_dir = args.cache_dir or Path(config["resume"]["cache_dir"])
    if not complete and (args.output is None or args.audit_report is None):
        raise ValueError("partial smoke requires --output and --audit-report")

    parents = config["parents"]
    d2_path = Path(parents["p9_d2_report"])
    selector_report_path = Path(parents["selector_report"])
    selector_model_path = Path(parents["selector_model"])
    p5_submission_path = Path(parents["p5_submission"])
    p5_audit_path = Path(parents["p5_audit"])
    _assert_hash(d2_path, parents["p9_d2_report_sha256"], "P9-D2 report")
    _assert_hash(
        selector_report_path,
        parents["selector_report_sha256"],
        "P9-D3 selector report",
    )
    _assert_hash(
        selector_model_path,
        parents["selector_model_sha256"],
        "P9-D3 selector model",
    )
    _assert_hash(
        p5_submission_path,
        parents["p5_submission_sha256"],
        "P5 parent submission",
    )
    _assert_hash(p5_audit_path, parents["p5_audit_sha256"], "P5 parent audit")

    d2 = json.loads(d2_path.read_text())
    selector_report = json.loads(selector_report_path.read_text())
    p5_audit = json.loads(p5_audit_path.read_text())
    if d2.get("status") != "promoted" or not bool(
        d2.get("comparison_vs_p5_c2", {}).get("pass")
    ):
        raise ValueError("P9-D3 requires the promoted P9-D2 parent")
    if selector_report.get("status") != "ready_for_test_inference":
        raise ValueError("P9-D3 final selector is not deployment-ready")
    if selector_report.get("model_sha256") != sha256_file(selector_model_path):
        raise ValueError("selector report/model provenance mismatch")
    if tuple(selector_report.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("selector report feature schema mismatch")
    threshold = float(config["selector"]["threshold"])
    if not bool(config["selector"].get("strict_greater")):
        raise ValueError("P9-D3 requires the registered strict-greater selector")
    if threshold != float(selector_report["model_config"]["threshold"]):
        raise ValueError("selector threshold differs from the trained final selector")

    d1_path = Path(selector_report["parents"]["p9_d1_report"])
    _assert_hash(
        d1_path,
        selector_report["parents"]["p9_d1_report_sha256"],
        "selector P9-D1 parent",
    )
    d1 = json.loads(d1_path.read_text())
    postprocess = YoloPostprocessConfig(**config["postprocess"])
    postprocess.validate()
    if config["postprocess"] != d1["config"]["postprocess"]:
        raise ValueError("P9-D3 postprocess differs from P9-D1")
    if config["augmentation_policy"] != d1["config"]["augmentation_policy"]:
        raise ValueError("P9-D3 augmentation policy differs from P9-D1")

    detector_paths = {}
    for name, detector_config in config["detectors"].items():
        checkpoint = Path(detector_config["checkpoint"])
        model_config = Path(detector_config["model_config"])
        _assert_hash(
            checkpoint, detector_config["checkpoint_sha256"], f"{name} checkpoint"
        )
        _assert_hash(
            model_config,
            detector_config["model_config_sha256"],
            f"{name} model config",
        )
        detector_paths[name] = checkpoint

    refiner_config_path = Path(config["refiner"]["training_config"])
    refiner_checkpoint_path = Path(config["refiner"]["checkpoint"])
    _assert_hash(
        refiner_config_path,
        config["refiner"]["training_config_sha256"],
        "C2 training config",
    )
    _assert_hash(
        refiner_checkpoint_path,
        config["refiner"]["checkpoint_sha256"],
        "C2 checkpoint",
    )
    refiner_config = yaml.safe_load(refiner_config_path.read_text())
    policy_hash = refinement_policy_sha256(
        refiner_config, str(config["refiner"]["refinement_policy"])
    )
    if policy_hash != str(config["refiner"]["refinement_policy_sha256"]):
        raise ValueError("C2 refinement policy SHA256 mismatch")
    if int(refiner_config["loss"]["boundary_kernel"]) != int(
        config["refiner"]["boundary_kernel"]
    ):
        raise ValueError("C2 boundary kernel differs from P9-D2")
    if float(refiner_config["evaluation"]["mask_threshold"]) != float(
        config["refiner"]["mask_threshold"]
    ):
        raise ValueError("C2 threshold differs from P9-D2")

    image_dir = Path(config["data"]["test_image_dir"])
    all_image_paths = sorted(image_dir.glob("*.jpeg"))
    if len(all_image_paths) != int(config["data"]["expected_images"]):
        raise ValueError("test image count differs from the registered P9-D3 config")
    all_image_ids = {path.stem for path in all_image_paths}
    parent_rows = _submission_by_image(p5_submission_path, all_image_ids)
    image_paths = (
        all_image_paths
        if args.max_images is None
        else all_image_paths[: args.max_images]
    )
    selected_image_ids = {path.stem for path in image_paths}
    coverage_fallback_ids = set(
        p5_audit.get("summary", {}).get("coverage_fallback_images", [])
    )
    if not coverage_fallback_ids <= all_image_ids:
        raise ValueError("P5 audit coverage fallback refers to an unknown test image")

    script_path = Path(__file__)
    fingerprint_payload = {
        "config_sha256": sha256_file(args.config),
        "script_sha256": sha256_file(script_path),
        "selector_model_sha256": sha256_file(selector_model_path),
        "t1_checkpoint_sha256": config["detectors"]["t1"]["checkpoint_sha256"],
        "i1_checkpoint_sha256": config["detectors"]["i1"]["checkpoint_sha256"],
        "refiner_checkpoint_sha256": config["refiner"]["checkpoint_sha256"],
        "p5_submission_sha256": sha256_file(p5_submission_path),
    }
    fingerprint = deployment_fingerprint(fingerprint_payload)
    cache_dir.mkdir(parents=True, exist_ok=True)

    selector = joblib.load(selector_model_path)
    prepare_ultralytics_runtime()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is unavailable in solar-filament") from exc
    device = torch.device(f"cuda:{int(args.device)}")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t1_detector = YOLO(str(detector_paths["t1"]))
    i1_detector = YOLO(str(detector_paths["i1"]))
    checkpoint = torch.load(
        refiner_checkpoint_path, map_location=device, weights_only=False
    )
    if checkpoint.get("report_type") != "p5_refiner_checkpoint_v1":
        raise ValueError("C2 checkpoint has the wrong report type")
    if checkpoint.get("config_sha256") != sha256_file(refiner_config_path):
        raise ValueError("C2 checkpoint/config provenance mismatch")
    if checkpoint.get("max_steps_smoke_override") is not None:
        raise ValueError("smoke C2 checkpoint cannot generate P9-D3")
    refiner = build_refiner_model(refiner_config["model"]).to(device)
    refiner.load_state_dict(checkpoint["model"])
    refiner.eval()

    expected_shape = (
        int(config["data"]["expected_height"]),
        int(config["data"]["expected_width"]),
    )
    crop_config = refiner_config["crop"]
    boundary_kernel = int(config["refiner"]["boundary_kernel"])
    mask_threshold = float(config["refiner"]["mask_threshold"])
    rows: list[dict[str, str]] = []
    image_reports: list[dict] = []
    source_audit: Counter = Counter()
    cache_hits = 0
    detector_seconds = 0.0
    refiner_seconds = 0.0
    started = perf_counter()

    with torch.inference_mode():
        for image_number, image_path in enumerate(image_paths, start=1):
            image_hash = sha256_file(image_path)
            cache_path = cache_dir / f"{image_path.stem}.json"
            cached = _load_valid_cache(
                cache_path,
                fingerprint=fingerprint,
                image_name=image_path.name,
                image_sha256=image_hash,
            )
            if cached is not None:
                rows.extend(cached["submission_rows"])
                image_reports.append(cached["image_report"])
                source_audit.update(cached["image_report"].get("candidate_audit", {}))
                cache_hits += 1
                print(
                    f"P9-D3 {image_number}/{len(image_paths)}: {image_path.name} cache=hit",
                    flush=True,
                )
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != expected_shape:
                raise ValueError(f"unexpected test image: {image_path}")
            detector_started = perf_counter()
            t1_result, shape, t1_raw_masks, t1_raw_scores = _predict(
                t1_detector, image_path, config["detectors"]["t1"], args.device
            )
            detector_seconds += perf_counter() - detector_started
            if shape != expected_shape:
                raise ValueError(f"unexpected T1 image shape {shape}: {image_path}")
            t1_masks, _, _ = postprocessed_instances_from_rles(
                [mask_to_rle(mask) for mask in t1_raw_masks],
                t1_raw_scores,
                height=shape[0],
                width=shape[1],
                config=postprocess,
            )

            if not t1_masks:
                if image_path.stem not in coverage_fallback_ids:
                    raise RuntimeError(
                        f"T1 returned no source outside registered fallback: {image_path.stem}"
                    )
                fallback_rows = parent_rows[image_path.stem]
                image_report = {
                    "image_id": image_path.stem,
                    "raw_t1_candidates": len(t1_raw_masks),
                    "raw_i1_candidates": None,
                    "base_t1_sources": 0,
                    "unfiltered_additions": 0,
                    "selected_additions": 0,
                    "output_instances": len(fallback_rows),
                    "support_fallback_instances": 0,
                    "coverage_fallback": True,
                    "incremental_skipped": True,
                    "p5_base_only_rles_exact": True,
                    "candidate_audit": {},
                    "selected_sources": [],
                }
                image_rows = fallback_rows
                del t1_result, t1_raw_masks, t1_raw_scores, t1_masks
            else:
                detector_started = perf_counter()
                i1_result, i1_shape, i1_raw_masks, i1_raw_scores = _predict(
                    i1_detector, image_path, config["detectors"]["i1"], args.device
                )
                detector_seconds += perf_counter() - detector_started
                if i1_shape != shape:
                    raise ValueError(f"T1/I1 test shape mismatch: {image_path}")
                i1_masks, i1_scores, i1_indices = postprocessed_instances_from_rles(
                    [mask_to_rle(mask) for mask in i1_raw_masks],
                    i1_raw_scores,
                    height=shape[0],
                    width=shape[1],
                    config=postprocess,
                )
                additions, candidate_audit = build_additive_candidates(
                    t1_masks,
                    i1_masks,
                    i1_scores,
                    i1_indices,
                    min_area=postprocess.min_area,
                    max_instances=postprocess.max_instances,
                    duplicate_iou=float(config["augmentation_policy"]["duplicate_iou"]),
                )
                source_audit.update(candidate_audit)
                t1_claimed = np.zeros(shape, dtype=np.uint8)
                for mask in t1_masks:
                    t1_claimed[np.asarray(mask, dtype=bool)] = 1
                distance_to_t1 = cv2.distanceTransform(
                    1 - t1_claimed, cv2.DIST_L2, 3
                )
                raw_t1_rles = [mask_to_rle(mask) for mask in t1_raw_masks]
                feature_rows = [
                    extract_candidate_features(
                        candidate,
                        image=image,
                        i1_output_count=len(i1_masks),
                        t1_output_count=len(t1_masks),
                        raw_t1_rles=raw_t1_rles,
                        raw_t1_scores=t1_raw_scores,
                        distance_to_t1=distance_to_t1,
                    )
                    for candidate in additions
                ]
                selected, predictions = selected_additions_from_features(
                    selector,
                    additions,
                    feature_rows,
                    threshold=threshold,
                )
                coarse_masks = [
                    *t1_masks,
                    *(candidate.mask for candidate in selected),
                ]

                refine_started = perf_counter()
                zero_target = np.zeros(shape, dtype=bool)
                crops = []
                coarse_crops = []
                bounds_by_instance = []
                for coarse in coarse_masks:
                    bounds = crop_bounds(
                        coarse,
                        context_scale=float(crop_config["context_scale"]),
                        minimum_side=int(crop_config["minimum_side"]),
                    )
                    image_crop, coarse_crop, _ = resize_refiner_crop(
                        image,
                        coarse,
                        zero_target,
                        bounds,
                        output_size=int(crop_config["output_size"]),
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
                inputs = torch.from_numpy(np.stack(crops)).to(device)
                probabilities = torch.sigmoid(refiner(inputs)).cpu().numpy()[:, 0]
                restored = [
                    restore_refined_mask(
                        probabilities[index],
                        bounds,
                        height=shape[0],
                        width=shape[1],
                        threshold=mask_threshold,
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
                base_count = len(t1_masks)
                _, base_only_rles, _ = combine_support_preserving_instances(
                    fused_masks[:base_count],
                    coarse_masks[:base_count],
                    height=shape[0],
                    width=shape[1],
                    min_area=postprocess.min_area,
                )
                parent_encoded = [
                    row["segmentation_rle"] for row in parent_rows[image_path.stem]
                ]
                if [rle_counts(rle) for rle in base_only_rles] != parent_encoded:
                    raise RuntimeError(
                        f"P9-D3 cannot exactly reproduce P5 base RLEs: {image_path.stem}"
                    )
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
                if len(candidate_rles) != len(coarse_masks):
                    raise RuntimeError("P9-D3 changed the registered source count")
                image_rows = [
                    {
                        "filament_id": f"{image_path.stem}_{index}",
                        "segmentation_rle": rle_counts(rle),
                    }
                    for index, rle in enumerate(candidate_rles, start=1)
                ]
                image_report = audit_image_labels(image_path.stem, labels, shape)
                image_report.update(
                    {
                        "raw_t1_candidates": len(t1_raw_masks),
                        "raw_i1_candidates": len(i1_raw_masks),
                        "base_t1_sources": base_count,
                        "unfiltered_additions": len(additions),
                        "selected_additions": len(selected),
                        "output_instances": len(candidate_rles),
                        "support_fallback_instances": fallback_count,
                        "coverage_fallback": False,
                        "incremental_skipped": False,
                        "p5_base_only_rles_exact": True,
                        "candidate_audit": candidate_audit,
                        "selected_sources": [
                            {
                                "i1_source_index": candidate.source_index,
                                "i1_rank": candidate.i1_rank,
                                "predicted_marginal_utility": float(prediction),
                            }
                            for candidate, prediction in zip(
                                additions, predictions, strict=True
                            )
                            if float(prediction) > threshold
                        ],
                    }
                )
                del (
                    t1_result,
                    i1_result,
                    t1_raw_masks,
                    t1_raw_scores,
                    i1_raw_masks,
                    i1_raw_scores,
                    t1_masks,
                    i1_masks,
                    coarse_masks,
                    fused_masks,
                    labels,
                    inputs,
                    probabilities,
                )

            cache_payload = {
                "report_type": "p9_d3_test_image_cache_v1",
                "complete": True,
                "deployment_fingerprint": fingerprint,
                "image_name": image_path.name,
                "image_sha256": image_hash,
                "submission_rows": image_rows,
                "image_report": image_report,
            }
            atomic_write_json(cache_path, cache_payload)
            rows.extend(image_rows)
            image_reports.append(image_report)
            print(
                f"P9-D3 {image_number}/{len(image_paths)}: {image_path.name} "
                f"T1={image_report['base_t1_sources']} "
                f"add={image_report['selected_additions']} "
                f"fallback={image_report['support_fallback_instances']}",
                flush=True,
            )

    summary = audit_submission_rows(rows, selected_image_ids, validate_rles=True)
    if summary["missing_images"] != 0:
        raise RuntimeError("P9-D3 does not cover every selected test image")
    base_sources = sum(int(row["base_t1_sources"]) for row in image_reports)
    selected_additions = sum(int(row["selected_additions"]) for row in image_reports)
    coverage_instances = sum(
        int(row["output_instances"])
        for row in image_reports
        if bool(row["coverage_fallback"])
    )
    output_instances = sum(int(row["output_instances"]) for row in image_reports)
    if output_instances != base_sources + selected_additions + coverage_instances:
        raise RuntimeError("P9-D3 output count differs from registered sources")
    if not all(bool(row["p5_base_only_rles_exact"]) for row in image_reports):
        raise RuntimeError("P9-D3 failed P5 base-only reproduction")
    if complete and base_sources != int(p5_audit["summary"]["source_instances"]):
        raise RuntimeError("P9-D3 T1 source count differs from P5")
    if complete and coverage_instances != int(
        p5_audit["summary"]["coverage_fallback_instances"]
    ):
        raise RuntimeError("P9-D3 coverage fallback count differs from P5")

    elapsed = perf_counter() - started
    summary.update(
        {
            "complete": complete,
            "run_id": str(config["run"]["run_id"]),
            "inference_and_audit_seconds": elapsed,
            "seconds_per_test_image": elapsed / len(image_paths),
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "rss_peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "cache_hits": cache_hits,
            "cache_misses": len(image_paths) - cache_hits,
            "detector_seconds_for_cache_misses": detector_seconds,
            "refiner_seconds_for_cache_misses": refiner_seconds,
            "base_t1_sources": base_sources,
            "unfiltered_additions": sum(
                int(row["unfiltered_additions"]) for row in image_reports
            ),
            "selected_additions": selected_additions,
            "output_instances": output_instances,
            "support_fallback_instances": sum(
                int(row["support_fallback_instances"]) for row in image_reports
            ),
            "coverage_fallback_images": [
                row["image_id"] for row in image_reports if row["coverage_fallback"]
            ],
            "coverage_fallback_instances": coverage_instances,
            "images_with_additions": sum(
                int(row["selected_additions"]) > 0 for row in image_reports
            ),
            "source_instance_count_preserved": (
                output_instances == base_sources + selected_additions + coverage_instances
            ),
            "p5_base_only_reproduced_exactly": True,
        }
    )
    output = write_submission(rows, output)
    report = {
        "report_type": "p9_d3_submission_audit_v1",
        "status": "ready_for_single_challenger_submission" if complete else "smoke_only",
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "deployment_fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "submission": str(output),
        "submission_sha256": sha256_file(output),
        "test_file_list_sha256": _file_list_sha256(image_paths),
        "summary": summary,
        "candidate_audit": dict(source_audit),
        "parents": {
            key: {"path": str(parents[key]), "sha256": str(parents[f"{key}_sha256"])}
            for key in (
                "p9_d2_report",
                "selector_report",
                "selector_model",
                "p5_submission",
                "p5_audit",
            )
        },
        "images": image_reports,
        "next_action": (
            "submit_once_without_posthoc_threshold_or_count_tuning"
            if complete
            else "run_complete_resumable_test_inference"
        ),
    }
    atomic_write_json(audit_path, report)
    print(f"wrote {len(rows)} instances to {output}", flush=True)
    print(f"submission SHA256: {report['submission_sha256']}", flush=True)
    print(f"wrote audit report to {audit_path}", flush=True)


if __name__ == "__main__":
    main()
