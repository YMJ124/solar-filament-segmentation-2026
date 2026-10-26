#!/usr/bin/env python3
"""Generate a reversible three-fold, four-view P5-C2 consensus submission."""

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
import numpy as np
import torch
import yaml

from src.instance.p5_refiner import (
    build_refiner_model,
    combine_support_preserving_instances,
    crop_bounds,
    fuse_refinement_in_boundary_band,
    postprocessed_instances_from_rles,
    resize_refiner_crop,
    restore_refined_mask,
)
from src.instance.p8_consensus import (
    TTA_NAMES,
    apply_tta,
    cluster_instance_records,
    compose_exclusive_consensus,
    invert_tta_masks,
    select_consensus_records,
)
from src.instance.yolo_cache import candidate_file_name, sha256_file
from src.instance.yolo_evaluation import YoloPostprocessConfig
from src.instance.yolo_inference import prediction_arrays
from src.io_utils import atomic_write_json
from src.submission.rle import mask_to_rle, rle_counts, write_submission
from src.submission.validate import audit_image_labels, audit_submission_rows
from src.ultralytics_runtime import prepare_ultralytics_runtime


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}")


def _file_list_sha256(paths: list[Path]) -> str:
    payload = json.dumps([path.name for path in paths], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _runtime_sha256(config_path: Path) -> str:
    """Fingerprint the config and both P8 implementation files for safe resume."""
    digest = hashlib.sha256()
    for path in (
        config_path,
        Path(__file__),
        Path("src/instance/p8_consensus.py"),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_fallback_rows(
    path: Path,
    expected_sha256: str,
    image_ids: set[str],
) -> dict[str, list[dict[str, str]]]:
    _assert_hash(path, expected_sha256, "fallback submission")
    by_image: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("fallback submission has unexpected columns")
        for row in reader:
            image_id = str(row["filament_id"]).rsplit("_", 1)[0]
            if image_id in image_ids:
                by_image.setdefault(image_id, []).append(dict(row))
    missing = sorted(image_ids - set(by_image))
    if missing:
        raise ValueError(f"fallback submission misses selected images: {missing[:5]}")
    audit_submission_rows(
        [row for rows in by_image.values() for row in rows],
        image_ids,
        validate_rles=True,
    )
    return by_image


def _cache_is_reusable(
    path: Path,
    *,
    config_sha256: str,
    runtime_sha256: str,
    image_name: str,
    fold: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("complete") is True
        and payload.get("config_sha256") == config_sha256
        and payload.get("runtime_sha256") == runtime_sha256
        and payload.get("file_name") == image_name
        and int(payload.get("fold", -1)) == fold
    )


def _refine_instances(
    *,
    image: np.ndarray,
    coarse_masks: list[np.ndarray],
    coarse_scores: np.ndarray,
    source_indices: np.ndarray,
    refiner: torch.nn.Module,
    refiner_config: dict,
    threshold: float,
    device: torch.device,
    fold: int,
    transform: str,
    min_area: int,
) -> tuple[list[dict], int]:
    if not coarse_masks:
        return [], 0
    height, width = image.shape
    zero_target = np.zeros((height, width), dtype=bool)
    crop_cfg = refiner_config["crop"]
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
    inputs = torch.from_numpy(np.stack(crops)).to(device)
    probability = torch.sigmoid(refiner(inputs)).cpu().numpy()[:, 0]
    restored = [
        restore_refined_mask(
            probability[index],
            bounds,
            height=height,
            width=width,
            threshold=threshold,
        )
        for index, bounds in enumerate(bounds_by_instance)
    ]
    boundary_kernel = int(refiner_config["loss"]["boundary_kernel"])
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
    _, refined_rles, fallback_count = combine_support_preserving_instances(
        fused_masks,
        coarse_masks,
        height=height,
        width=width,
        min_area=min_area,
    )
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pycocotools is required for P8 inference") from exc
    records = []
    for index, rle in enumerate(refined_rles):
        records.append(
            {
                "fold": int(fold),
                "transform": transform,
                "source_index": int(source_indices[index]),
                "score": float(coarse_scores[index]),
                "area": int(mask_utils.area(rle)),
                "segmentation_rle": rle_counts(rle),
            }
        )
    return records, int(fallback_count)


def _member_cache_payload(
    *,
    config_sha256: str,
    runtime_sha256: str,
    image_path: Path,
    shape: tuple[int, int],
    member: dict,
    transform_reports: list[dict],
    candidates: list[dict],
) -> dict:
    return {
        "report_type": "p8_member_candidate_cache_v1",
        "complete": True,
        "config_sha256": config_sha256,
        "runtime_sha256": runtime_sha256,
        "file_name": image_path.name,
        "image_id": image_path.stem,
        "height": int(shape[0]),
        "width": int(shape[1]),
        "fold": int(member["fold"]),
        "detector_checkpoint": str(member["detector_checkpoint"]),
        "detector_checkpoint_sha256": str(member["detector_checkpoint_sha256"]),
        "refiner_checkpoint": str(member["refiner_checkpoint"]),
        "refiner_checkpoint_sha256": str(member["refiner_checkpoint_sha256"]),
        "transform_reports": transform_reports,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/submission/p8_e1_threefold_4tta_consensus.yaml"),
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("max-images must be positive")
    if args.device == "cpu":
        raise ValueError("P8 inference requires CUDA; CPU fallback is disabled")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} requested but unavailable")

    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != 1:
        raise ValueError("P8 config must use schema_version=1")
    config_hash = sha256_file(args.config)
    runtime_hash = _runtime_sha256(args.config)
    members = list(config["members"])
    folds = [int(member["fold"]) for member in members]
    if len(members) < 2 or len(folds) != len(set(folds)):
        raise ValueError("P8 requires at least two uniquely numbered fold members")
    transforms = [str(value) for value in config["tta"]["transforms"]]
    if not transforms or any(value not in TTA_NAMES for value in transforms):
        raise ValueError("P8 config contains an unsupported TTA transform")
    if len(transforms) != len(set(transforms)):
        raise ValueError("P8 TTA transforms must be unique")
    for member in members:
        _assert_hash(
            Path(member["detector_checkpoint"]),
            str(member["detector_checkpoint_sha256"]),
            f"fold{member['fold']} detector checkpoint",
        )
        _assert_hash(
            Path(member["detector_config"]),
            str(member["detector_config_sha256"]),
            f"fold{member['fold']} detector config",
        )
        _assert_hash(
            Path(member["refiner_checkpoint"]),
            str(member["refiner_checkpoint_sha256"]),
            f"fold{member['fold']} refiner checkpoint",
        )
        _assert_hash(
            Path(member["refiner_config"]),
            str(member["refiner_config_sha256"]),
            f"fold{member['fold']} refiner config",
        )

    image_dir = Path(config["data"]["test_image_dir"])
    all_image_paths = sorted(image_dir.glob("*.jpeg"))
    if len(all_image_paths) != int(config["data"]["expected_images"]):
        raise ValueError("test image count differs from P8 config")
    complete = args.max_images is None
    image_paths = (
        all_image_paths if complete else all_image_paths[: int(args.max_images)]
    )
    cache_root = args.cache_root or Path(config["output"]["cache_root"])
    output = args.output or Path(config["output"]["submission"])
    audit_path = args.audit_report or Path(config["output"]["audit"])
    if not complete and (args.cache_root is None or args.output is None or args.audit_report is None):
        raise ValueError("partial smoke requires cache, output and audit overrides")
    expected_shape = (
        int(config["data"]["expected_height"]),
        int(config["data"]["expected_width"]),
    )
    expected_ids = {path.stem for path in image_paths}
    fallback_cfg = config["fallback"]
    fallback_rows = _load_fallback_rows(
        Path(fallback_cfg["submission"]),
        str(fallback_cfg["submission_sha256"]),
        expected_ids,
    )
    postprocess = YoloPostprocessConfig(**config["postprocess"])
    postprocess.validate()

    prepare_ultralytics_runtime()
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Ultralytics is unavailable in solar-filament") from exc
    device = torch.device(f"cuda:{int(args.device)}")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    generated_cache_files = 0
    reused_cache_files = 0

    inference_cfg = config["detector_inference"]
    with torch.inference_mode():
        for member_index, member in enumerate(members, start=1):
            fold = int(member["fold"])
            member_dir = cache_root / f"fold{fold}"
            detector = YOLO(str(member["detector_checkpoint"]))
            refiner_config = yaml.safe_load(Path(member["refiner_config"]).read_text())
            checkpoint = torch.load(
                Path(member["refiner_checkpoint"]),
                map_location=device,
                weights_only=False,
            )
            if checkpoint.get("report_type") != "p5_refiner_checkpoint_v1":
                raise ValueError(f"fold{fold} refiner checkpoint has wrong type")
            if checkpoint.get("max_steps_smoke_override") is not None:
                raise ValueError(f"fold{fold} smoke refiner cannot generate a submission")
            if checkpoint.get("config_sha256") != str(member["refiner_config_sha256"]):
                raise ValueError(f"fold{fold} refiner checkpoint/config mismatch")
            if int(checkpoint["heldout_fold"]) != fold:
                raise ValueError(f"fold{fold} refiner heldout fold mismatch")
            refiner = build_refiner_model(refiner_config["model"]).to(device)
            refiner.load_state_dict(checkpoint["model"])
            refiner.eval()
            threshold = float(refiner_config["evaluation"]["mask_threshold"])

            for image_index, image_path in enumerate(image_paths, start=1):
                cache_path = member_dir / candidate_file_name(image_path.name)
                if not args.no_resume and _cache_is_reusable(
                    cache_path,
                    config_sha256=config_hash,
                    runtime_sha256=runtime_hash,
                    image_name=image_path.name,
                    fold=fold,
                ):
                    reused_cache_files += 1
                    print(
                        f"P8 cache fold {member_index}/{len(members)} image "
                        f"{image_index}/{len(image_paths)}: {image_path.name} resume",
                        flush=True,
                    )
                    continue
                color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if color is None or gray is None or gray.shape != expected_shape:
                    raise ValueError(f"cannot read expected test image: {image_path}")
                transformed_images = [apply_tta(color, name) for name in transforms]
                results = detector.predict(
                    source=transformed_images,
                    imgsz=int(inference_cfg["imgsz"]),
                    conf=float(inference_cfg["candidate_confidence"]),
                    max_det=int(inference_cfg["max_candidates"]),
                    retina_masks=bool(inference_cfg["retina_masks"]),
                    device=args.device,
                    quantize=int(inference_cfg["quantize"]),
                    batch=1,
                    verbose=False,
                )
                if len(results) != len(transforms):
                    raise RuntimeError("detector did not return one result per transform")
                candidates: list[dict] = []
                transform_reports = []
                for transform, result in zip(transforms, results):
                    shape = tuple(int(value) for value in result.orig_shape)
                    if shape != expected_shape:
                        raise ValueError(f"unexpected transformed image shape: {shape}")
                    masks, scores = prediction_arrays(result, shape)
                    masks = invert_tta_masks(masks, transform)
                    coarse_masks, coarse_scores, source_indices = (
                        postprocessed_instances_from_rles(
                            [mask_to_rle(mask) for mask in masks],
                            scores,
                            height=shape[0],
                            width=shape[1],
                            config=postprocess,
                        )
                    )
                    refined, fallback_count = _refine_instances(
                        image=gray,
                        coarse_masks=coarse_masks,
                        coarse_scores=coarse_scores,
                        source_indices=source_indices,
                        refiner=refiner,
                        refiner_config=refiner_config,
                        threshold=threshold,
                        device=device,
                        fold=fold,
                        transform=transform,
                        min_area=postprocess.min_area,
                    )
                    candidates.extend(refined)
                    transform_reports.append(
                        {
                            "transform": transform,
                            "raw_candidates": int(len(masks)),
                            "source_instances": int(len(coarse_masks)),
                            "refined_instances": int(len(refined)),
                            "support_fallback_instances": int(fallback_count),
                        }
                    )
                payload = _member_cache_payload(
                    config_sha256=config_hash,
                    runtime_sha256=runtime_hash,
                    image_path=image_path,
                    shape=expected_shape,
                    member=member,
                    transform_reports=transform_reports,
                    candidates=candidates,
                )
                atomic_write_json(cache_path, payload)
                generated_cache_files += 1
                print(
                    f"P8 cache fold {member_index}/{len(members)} image "
                    f"{image_index}/{len(image_paths)}: {image_path.name} "
                    f"candidates={len(candidates)}",
                    flush=True,
                )
                del results, transformed_images
            del detector, refiner, checkpoint
            torch.cuda.empty_cache()

    rows: list[dict[str, str]] = []
    image_reports = []
    cache_entries = []
    fallback_images = []
    fold_support_histogram: Counter[int] = Counter()
    consensus_cfg = config["consensus"]
    for image_index, image_path in enumerate(image_paths, start=1):
        records = []
        member_counts = {}
        for member in members:
            fold = int(member["fold"])
            cache_path = cache_root / f"fold{fold}" / candidate_file_name(image_path.name)
            payload = json.loads(cache_path.read_text())
            if not _cache_is_reusable(
                cache_path,
                config_sha256=config_hash,
                runtime_sha256=runtime_hash,
                image_name=image_path.name,
                fold=fold,
            ):
                raise RuntimeError(f"invalid or incomplete P8 cache: {cache_path}")
            candidates = list(payload["candidates"])
            records.extend(candidates)
            member_counts[str(fold)] = len(candidates)
            cache_entries.append(
                {
                    "file_name": image_path.name,
                    "fold": fold,
                    "path": str(cache_path),
                    "sha256": sha256_file(cache_path),
                    "candidate_count": len(candidates),
                }
            )
        clusters = cluster_instance_records(
            records,
            height=expected_shape[0],
            width=expected_shape[1],
            iou_threshold=float(consensus_cfg["iou_threshold"]),
        )
        selected, cluster_audit = select_consensus_records(
            records,
            clusters,
            minimum_fold_support=int(consensus_cfg["minimum_fold_support"]),
            representative=str(consensus_cfg["representative"]),
        )
        for cluster in cluster_audit:
            fold_support_histogram[int(cluster["fold_support"])] += 1
        labels, final_rles, kept_meta = compose_exclusive_consensus(
            selected,
            height=expected_shape[0],
            width=expected_shape[1],
            min_area=int(consensus_cfg["min_area_after_deoverlap"]),
            max_instances=int(consensus_cfg["max_instances"]),
        )
        used_fallback = not final_rles
        if used_fallback:
            image_rows = fallback_rows[image_path.stem]
            rows.extend(image_rows)
            fallback_images.append(image_path.stem)
            image_audit = audit_submission_rows(
                image_rows, {image_path.stem}, validate_rles=True
            )
            final_instances = int(image_audit["rows"])
        else:
            image_audit = audit_image_labels(image_path.stem, labels, expected_shape)
            final_instances = len(final_rles)
            for instance_id, rle in enumerate(final_rles, start=1):
                rows.append(
                    {
                        "filament_id": f"{image_path.stem}_{instance_id}",
                        "segmentation_rle": rle_counts(rle),
                    }
                )
        image_reports.append(
            {
                "image_id": image_path.stem,
                "member_candidate_counts": member_counts,
                "raw_consensus_records": len(records),
                "clusters": len(clusters),
                "accepted_clusters": len(selected),
                "final_instances": final_instances,
                "fallback": used_fallback,
                "kept": kept_meta,
                "cluster_audit": cluster_audit,
                "output_audit": image_audit,
            }
        )
        print(
            f"P8 consensus {image_index}/{len(image_paths)}: {image_path.name} "
            f"records={len(records)} accepted={len(selected)} final={final_instances} "
            f"fallback={used_fallback}",
            flush=True,
        )

    submission_summary = audit_submission_rows(rows, expected_ids, validate_rles=True)
    if submission_summary["missing_images"]:
        raise RuntimeError("P8 submission does not cover every selected image")
    output = write_submission(rows, output)
    elapsed = perf_counter() - started
    cache_manifest = {
        "report_type": "p8_member_cache_manifest_v1",
        "complete": complete,
        "config": str(args.config),
        "config_sha256": config_hash,
        "runtime_sha256": runtime_hash,
        "test_file_list_sha256": _file_list_sha256(image_paths),
        "images": len(image_paths),
        "members": folds,
        "transforms": transforms,
        "entries": cache_entries,
    }
    atomic_write_json(cache_root / "manifest.json", cache_manifest)
    submission_summary.update(
        {
            "complete": complete,
            "run_id": str(config["run"]["run_id"]),
            "elapsed_seconds": elapsed,
            "seconds_per_image_all_members": elapsed / len(image_paths),
            "generated_cache_files": generated_cache_files,
            "reused_cache_files": reused_cache_files,
            "fallback_images": fallback_images,
            "fold_support_histogram": {
                str(key): int(value) for key, value in sorted(fold_support_histogram.items())
            },
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "rss_peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
    )
    audit_payload = {
        "report_type": "p8_consensus_submission_audit_v1",
        "config": str(args.config),
        "config_sha256": config_hash,
        "runtime_sha256": runtime_hash,
        "validation_status": str(config["run"]["validation_status"]),
        "submission": str(output),
        "submission_sha256": sha256_file(output),
        "cache_manifest": str(cache_root / "manifest.json"),
        "cache_manifest_sha256": sha256_file(cache_root / "manifest.json"),
        "summary": submission_summary,
        "images": image_reports,
        "reversibility": {
            "member_outputs_cached_before_consensus": True,
            "member_fold_and_transform_preserved": True,
            "single_model_submission_unchanged": True,
            "consensus_is_final_output_branch_only": True,
        },
    }
    atomic_write_json(audit_path, audit_payload)
    print(json.dumps(submission_summary, indent=2), flush=True)
    print(f"wrote P8 submission to {output}", flush=True)
    print(f"wrote P8 audit to {audit_path}", flush=True)


if __name__ == "__main__":
    main()
