#!/usr/bin/env python3
"""Run YOLO once on validation images and cache raw instance candidates."""

from __future__ import annotations

import argparse
import json
import re
import resource
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import yaml

from src.data.coco import CocoAnnotations
from src.data.folds import load_fold_manifest
from src.instance.yolo_cache import (
    CACHE_FORMAT,
    CACHE_SCHEMA_VERSION,
    candidate_file_name,
    config_sha256,
    load_cache_manifest,
    sha256_file,
    write_candidate_record,
)
from src.instance.yolo_inference import prediction_arrays
from src.io_utils import atomic_write_json
from src.ultralytics_runtime import prepare_ultralytics_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026", type=Path)
    parser.add_argument(
        "--fold-manifest", default="data/manifests/train_folds.csv", type=Path
    )
    parser.add_argument("--validation-file-list", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--quantize", type=int, choices=(16, 32), default=16)
    parser.add_argument(
        "--retina-masks", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--candidate-confidence",
        type=float,
        default=0.01,
        help="lowest confidence retained in the reusable cache",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=50,
        help="maximum raw candidates retained per image",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse valid per-image records from an interrupted identical run",
    )
    return parser.parse_args()


def _existing_entry(path: Path, run_hash: str, record: dict) -> dict:
    payload = json.loads(path.read_text())
    expected = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "format": CACHE_FORMAT,
        "run_config_sha256": run_hash,
        "file_name": str(record["file_name"]),
        "height": int(record["height"]),
        "width": int(record["width"]),
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"cannot resume incompatible candidate record {path}: {mismatches}")
    candidates = payload.get("candidates", [])
    if len(candidates) != int(payload.get("candidate_count", -1)):
        raise ValueError(f"cannot resume corrupt candidate record: {path}")
    return {
        "run_config_sha256": payload["run_config_sha256"],
        "file_name": payload["file_name"],
        "image_ids": payload.get("image_ids", []),
        "selected_image_id": payload.get("selected_image_id"),
        "fold": payload.get("fold"),
        "height": payload["height"],
        "width": payload["width"],
        "candidate_count": payload["candidate_count"],
        "candidate_file": path.name,
        "candidate_file_sha256": sha256_file(path),
    }


def _experiment_metadata(model_config: Path | None) -> dict:
    """Recover model, training resolution, and target provenance from args.yaml."""
    if model_config is None:
        return {}
    payload = yaml.safe_load(model_config.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"model config must contain a mapping: {model_config}")
    if {"run", "data", "model", "training"} <= set(payload):
        payload = {
            **dict(payload["training"]),
            "name": payload["run"].get("name"),
            "seed": payload["run"].get("seed"),
            "deterministic": payload["run"].get("deterministic", True),
            "model": payload["model"].get("weights"),
            "data": payload["data"].get("dataset_yaml"),
        }
    run_name = payload.get("name")
    result = {
        "run_name": run_name,
        "experiment_family": (
            re.sub(r"_fold\d+$", "", str(run_name)) if run_name else None
        ),
        "model": payload.get("model"),
        "training_imgsz": payload.get("imgsz"),
        "seed": payload.get("seed"),
    }
    dataset_value = payload.get("data")
    if not dataset_value:
        return result
    dataset_path = Path(str(dataset_value))
    if not dataset_path.is_absolute():
        candidates = [Path.cwd() / dataset_path, model_config.parent / dataset_path]
        dataset_path = next((path for path in candidates if path.is_file()), candidates[0])
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"training dataset YAML referenced by {model_config} is missing: {dataset_path}"
        )
    dataset_payload = yaml.safe_load(dataset_path.read_text()) or {}
    result.update(
        {
            "dataset_yaml": str(dataset_path.resolve()),
            "dataset_yaml_sha256": sha256_file(dataset_path),
            "target_policy": dataset_payload.get(
                "target_policy", "medoid_annotator_v1"
            ),
            "validation_target_policy": dataset_payload.get(
                "validation_target_policy", "medoid_annotator_v1"
            ),
        }
    )
    signature_fields = (
        "model",
        "imgsz",
        "batch",
        "epochs",
        "workers",
        "cache",
        "mask_ratio",
        "degrees",
        "fliplr",
        "flipud",
        "mosaic",
        "translate",
        "scale",
        "shear",
        "perspective",
        "seed",
        "deterministic",
    )
    signature = {key: payload.get(key) for key in signature_fields}
    signature["target_policy"] = result["target_policy"]
    result["training_signature"] = signature
    result["training_signature_sha256"] = config_sha256(signature)
    return result


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not 0 < args.candidate_confidence < 1:
        raise ValueError("candidate-confidence must be inside (0, 1)")
    if args.max_candidates < 1 or args.imgsz < 1:
        raise ValueError("max-candidates and imgsz must be positive")

    root = args.data_root
    annotation = root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = CocoAnnotations(annotation)
    records = [
        row
        for row in load_fold_manifest(args.fold_manifest, annotation)
        if int(row["fold"]) == args.fold
    ]
    if args.validation_file_list:
        names = set(json.loads(args.validation_file_list.read_text()))
        records = [row for row in records if str(row["file_name"]) in names]
    if not records:
        raise ValueError("validation selection is empty")
    for record in records:
        record["image_ids"] = [
            str(batch["id"])
            for batch in coco.batches_for_file(str(record["file_name"]))
        ]

    model_config = args.model_config
    if model_config is None:
        inferred = checkpoint.parent.parent / "args.yaml"
        model_config = inferred if inferred.is_file() else None
    if model_config is not None and not model_config.is_file():
        raise FileNotFoundError(f"model config not found: {model_config}")
    experiment = _experiment_metadata(model_config)
    if experiment.get("training_imgsz") is not None and int(
        experiment["training_imgsz"]
    ) != int(args.imgsz):
        raise ValueError(
            "cache inference resolution must match training resolution for this stage: "
            f"{args.imgsz} != {experiment['training_imgsz']}"
        )

    inference = {
        "imgsz": args.imgsz,
        "quantize": args.quantize,
        "retina_masks": args.retina_masks,
        "candidate_confidence": args.candidate_confidence,
        "max_candidates": args.max_candidates,
    }
    provenance = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_config_sha256": sha256_file(model_config) if model_config else None,
        "dataset_config_sha256": experiment.get("dataset_yaml_sha256"),
        "inference_config_sha256": config_sha256(inference),
        "annotation_sha256": sha256_file(annotation),
        "fold_manifest_sha256": sha256_file(args.fold_manifest),
        "fold": args.fold,
        "file_names": [str(row["file_name"]) for row in records],
    }
    run_hash = config_sha256(provenance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        existing = load_cache_manifest(args.output_dir)
        if existing.get("run_config_sha256") == run_hash:
            print(f"cache is already complete: {manifest_path}")
            return
        raise FileExistsError(
            f"output contains a different completed cache: {manifest_path}"
        )

    prepare_ultralytics_runtime()
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("install requirements-yolo.txt before inference") from exc

    cuda = args.device != "cpu" and torch.cuda.is_available()
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(checkpoint))
    image_dir = root / "train/train_images"
    entries = []
    resumed = 0
    started = perf_counter()
    for index, record in enumerate(records, start=1):
        image_path = image_dir / str(record["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(f"validation image not found: {image_path}")
        candidate_path = args.output_dir / candidate_file_name(str(record["file_name"]))
        if candidate_path.exists():
            if not args.resume:
                raise FileExistsError(
                    f"partial cache record exists: {candidate_path}; pass --resume "
                    "to validate and reuse it"
                )
            entry = _existing_entry(candidate_path, run_hash, record)
            resumed += 1
        else:
            result = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.candidate_confidence,
                max_det=args.max_candidates,
                retina_masks=args.retina_masks,
                device=args.device,
                quantize=args.quantize,
                batch=1,
                verbose=False,
            )[0]
            shape = (int(record["height"]), int(record["width"]))
            masks, scores = prediction_arrays(result, shape)
            entry = write_candidate_record(
                candidate_path,
                record,
                masks,
                scores,
                run_config_sha256=run_hash,
            )
            del result, masks, scores
        entries.append(entry)
        print(
            f"YOLO candidate cache {index}/{len(records)}: "
            f"{record['file_name']} candidates={entry['candidate_count']}",
            flush=True,
        )

    elapsed = perf_counter() - started
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "format": CACHE_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_config_sha256": run_hash,
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": provenance["checkpoint_sha256"],
        },
        "model_config": (
            {
                "path": str(model_config.resolve()),
                "sha256": provenance["model_config_sha256"],
            }
            if model_config
            else None
        ),
        "experiment": experiment,
        "inference": {
            **inference,
            "config_sha256": provenance["inference_config_sha256"],
            "device_used": args.device,
        },
        "dataset": {
            "data_root": str(root.resolve()),
            "annotation_path": str(annotation.resolve()),
            "annotation_sha256": provenance["annotation_sha256"],
            "fold_manifest_path": str(args.fold_manifest.resolve()),
            "fold_manifest_sha256": provenance["fold_manifest_sha256"],
            "fold": args.fold,
            "validation_file_list": (
                str(args.validation_file_list.resolve())
                if args.validation_file_list
                else None
            ),
        },
        "physical_images": len(entries),
        "candidate_instances": sum(row["candidate_count"] for row in entries),
        "cache_seconds": elapsed,
        "seconds_per_image": elapsed / len(entries),
        "resumed_images": resumed,
        "gpu_peak_mb": (
            torch.cuda.max_memory_allocated() / 1024**2 if cuda else 0.0
        ),
        "rss_peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "images": entries,
    }
    atomic_write_json(manifest_path, manifest)
    print(f"wrote completed candidate cache to {manifest_path}")


if __name__ == "__main__":
    main()
