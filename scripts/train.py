#!/usr/bin/env python3
"""Configuration-driven training for the single-model core."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import resource
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.coco import CocoAnnotations
from src.data.cache import sha256_file
from src.data.dataset import FilamentDataset, collate_filaments
from src.data.folds import load_fold_manifest
from src.engine import (
    config_fingerprint,
    restore_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate_loss,
)
from src.full_image_validation import evaluate_full_images
from src.io_utils import atomic_write_json
from src.models import build_model
from src.models.factory import normalize_model_config
from src.metrics.pq import OFFICIAL_METRIC_NAME, OFFICIAL_METRIC_SOURCE
from src.training_config import atomic_write_yaml, load_training_config

DIRECT_DEPENDENCIES = (
    "numpy", "scipy", "scikit-image", "Pillow", "opencv-python-headless",
    "pycocotools", "tqdm", "torch", "torchvision",
    "segmentation-models-pytorch", "PyYAML",
)


def _set_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _build_scheduler(optimizer, epochs: int, warmup_epochs: int):
    warmup_epochs = min(max(0, int(warmup_epochs)), max(0, epochs - 1))

    def multiplier(epoch_index: int) -> float:
        if warmup_epochs and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / warmup_epochs
        cosine_epochs = max(1, epochs - warmup_epochs)
        progress = min(1.0, max(0.0, (epoch_index - warmup_epochs) / cosine_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _dependency_versions() -> dict[str, str]:
    versions = {}
    for package in DIRECT_DEPENDENCIES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _run_manifest(config, annotation_file: Path, fold_manifest: Path,
                  cache_dir: str | Path | None, device, parameter_count: int,
                  train_images: int, valid_images: int) -> dict:
    cache_manifest = Path(cache_dir) / "manifest.json" if cache_dir else None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition": {
            "slug": "filament-segmentation-2026",
            "metric": OFFICIAL_METRIC_NAME,
            "metric_source": OFFICIAL_METRIC_SOURCE,
            "metric_source_version": 6,
        },
        "config_fingerprint": config_fingerprint(config),
        "command": sys.argv,
        "data": {
            "annotation_file": str(annotation_file),
            "annotation_sha256": sha256_file(annotation_file),
            "fold_manifest": str(fold_manifest),
            "fold_manifest_sha256": sha256_file(fold_manifest),
            "validation_file_list": config["data"].get("validation_file_list"),
            "validation_file_list_sha256": (
                sha256_file(config["data"]["validation_file_list"])
                if config["data"].get("validation_file_list") else None
            ),
            "cache_manifest": (
                json.loads(cache_manifest.read_text())
                if cache_manifest and cache_manifest.exists() else None
            ),
            "ground_truth_policy": "competition_train_annotations_only",
            "train_physical_images": train_images,
            "validation_physical_images": valid_images,
        },
        "model": {
            "parameters": parameter_count,
            "architecture": config["model"]["architecture"],
            "encoder_name": config["model"].get("encoder_name"),
            "pretrained_weights": config["model"].get("encoder_weights"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "dependencies": _dependency_versions(),
        },
    }


def _replace_generated_directory(candidate: Path, destination: Path, run_dir: Path) -> None:
    """Promote generated OOF data while refusing paths outside this run."""
    run_resolved = run_dir.resolve()
    for path in (candidate, destination):
        if run_resolved not in path.resolve().parents:
            raise ValueError(f"refusing to modify generated directory outside run: {path}")
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(candidate, destination)


def _remove_generated_directory(path: Path, run_dir: Path) -> None:
    if path.exists():
        if run_dir.resolve() not in path.resolve().parents:
            raise ValueError(f"refusing to remove directory outside run: {path}")
        shutil.rmtree(path)


def _cli_overrides(args) -> dict:
    overrides: dict = {}

    def put(section: str, key: str, value) -> None:
        if value is not None:
            overrides.setdefault(section, {})[key] = value

    put("data", "data_root", args.data_root)
    put("data", "fold_manifest", args.fold_manifest)
    put("data", "mask_cache_dir", args.mask_cache_dir)
    put("data", "validation_file_list", args.validation_file_list)
    put("data", "fold", args.fold)
    put("data", "image_size", args.image_size)
    put("data", "num_workers", args.num_workers)
    if args.encoder_weights is not None:
        overrides.setdefault("model", {})["encoder_weights"] = (
            None if args.encoder_weights == "none" else args.encoder_weights
        )
    put("training", "epochs", args.epochs)
    put("training", "batch_size", args.batch_size)
    put("training", "learning_rate", args.lr)
    put("training", "max_steps", args.max_steps)
    put("training", "max_validation_batches", args.max_validation_batches)
    put("validation", "max_images", args.full_pq_max_images)
    if args.skip_full_pq:
        put("validation", "full_pq_enabled", False)
    put("artifacts", "run_dir", args.run_dir)
    return overrides


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/global_resnet34_fold0.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--data-root")
    parser.add_argument("--fold-manifest")
    parser.add_argument("--mask-cache-dir")
    parser.add_argument("--validation-file-list")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--encoder-weights", choices=["imagenet", "none"])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--full-pq-max-images", type=int)
    parser.add_argument("--skip-full-pq", action="store_true")
    parser.add_argument("--run-dir")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_training_config(args.config, _cli_overrides(args))
    run_cfg, data_cfg = config["run"], config["data"]
    model_cfg, train_cfg = config["model"], config["training"]
    valid_cfg, artifact_cfg = config["validation"], config["artifacts"]
    run_dir = Path(artifact_cfg["run_dir"])
    best_output = Path(artifact_cfg.get("best_checkpoint", run_dir / "checkpoints/best.pt"))
    last_output = run_dir / "checkpoints/last.pt"
    log_output = run_dir / "metrics/epochs.jsonl"
    collision_paths = [
        path for path in (best_output, last_output, log_output, run_dir / "oof_best")
        if path.exists()
    ]
    if collision_paths and not args.resume:
        raise FileExistsError(
            "run already contains training artifacts; use --resume or a new --run-dir: "
            + ", ".join(str(path) for path in collision_paths)
        )
    if args.resume and not Path(args.resume).is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {args.resume}")
    seed = int(run_cfg["seed"])
    _set_reproducibility(seed, bool(run_cfg.get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(data_cfg["data_root"])
    annotation_file = data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    records = load_fold_manifest(data_cfg["fold_manifest"], annotation_file)
    fold = int(data_cfg["fold"])
    train_records = [row for row in records if int(row["fold"]) != fold]
    valid_records = [row for row in records if int(row["fold"]) == fold]
    validation_file_list = data_cfg.get("validation_file_list")
    if validation_file_list:
        selected_names = set(json.loads(Path(validation_file_list).read_text()))
        unknown = selected_names - {str(row["file_name"]) for row in valid_records}
        if unknown:
            raise ValueError(
                "validation_file_list contains files outside the selected fold: "
                f"{sorted(unknown)[:5]}"
            )
        valid_records = [
            row for row in valid_records if str(row["file_name"]) in selected_names
        ]
    if not train_records or not valid_records:
        raise ValueError(f"fold {fold} is absent or leaves an empty training split")

    dataset_options = {
        "data_root": data_root,
        "fold_manifest": data_cfg["fold_manifest"],
        "seed": seed,
        "image_size": int(data_cfg["image_size"]),
        "positive_crop_probability": float(data_cfg["positive_crop_probability"]),
        "cache_dir": data_cfg.get("mask_cache_dir"),
    }
    train_set = FilamentDataset(records=train_records, train=True, **dataset_options)
    valid_set = FilamentDataset(records=valid_records, train=False, **dataset_options)
    loader_options = {
        "batch_size": int(train_cfg["batch_size"]),
        "num_workers": int(data_cfg["num_workers"]),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_filaments,
    }
    loader = DataLoader(train_set, shuffle=True, **loader_options)
    valid_loader = DataLoader(valid_set, shuffle=False, **loader_options)

    kind = str(model_cfg["kind"])
    normalized_model_cfg = normalize_model_config(kind, model_cfg)
    model = build_model(kind, normalized_model_cfg, initialize_pretrained=not bool(args.resume)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and bool(train_cfg.get("amp", True))
    )
    epochs = int(train_cfg["epochs"])
    scheduler = _build_scheduler(optimizer, epochs, int(train_cfg["warmup_epochs"]))

    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_yaml(run_dir / "config.resolved.yaml", config)

    start_epoch = 1
    global_step = 0
    best_metric = None
    best_metric_name = (
        "official_pq"
        if valid_cfg["full_pq_enabled"]
        else "negative_valid_loss"
    )
    if args.resume:
        state = restore_checkpoint(
            args.resume,
            model,
            optimizer,
            scaler,
            scheduler,
            device,
            expected_config=config,
        )
        start_epoch = state.epoch + 1
        global_step = state.global_step
        best_metric = state.best_metric
        if state.best_metric_name and state.best_metric_name != best_metric_name:
            raise ValueError(
                f"resume metric {state.best_metric_name!r} != configured {best_metric_name!r}"
            )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    manifest_path = run_dir / "run_manifest.json"
    manifest = _run_manifest(
        config,
        annotation_file,
        Path(data_cfg["fold_manifest"]),
        data_cfg.get("mask_cache_dir"),
        device,
        parameter_count,
        len(train_set),
        len(valid_set),
    )
    if manifest_path.exists() and args.resume:
        existing_manifest = json.loads(manifest_path.read_text())
        if existing_manifest.get("config_fingerprint") != manifest["config_fingerprint"]:
            raise ValueError("run manifest does not match resume configuration")
    else:
        atomic_write_json(manifest_path, manifest)
    print({
        "run": run_cfg["name"], "device": str(device), "kind": kind,
        "architecture": normalized_model_cfg["architecture"],
        "parameters": parameter_count, "train_images": len(train_set),
        "valid_images": len(valid_set), "start_epoch": start_epoch,
    })

    coco = CocoAnnotations(annotation_file)
    for epoch in range(start_epoch, epochs + 1):
        train_set.set_epoch(epoch)
        train_stats = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            kind=kind,
            scaler=scaler,
            max_steps=train_cfg.get("max_steps"),
            gradient_accumulation=int(train_cfg["gradient_accumulation"]),
            gradient_clip_norm=float(train_cfg["gradient_clip_norm"]),
        )
        global_step += int(train_stats["optimizer_steps"])
        valid_stats = validate_loss(
            model,
            valid_loader,
            device,
            kind,
            train_cfg.get("max_validation_batches"),
        )
        lr_used = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        metrics = {
            "train_loss": train_stats["loss"],
            "valid_loss": valid_stats["loss"],
            "train_batches": train_stats["batches"],
            "valid_batches": valid_stats["batches"],
            "optimizer_steps": train_stats["optimizer_steps"],
            "gpu_peak_mb": train_stats["gpu_peak_mb"],
            "kind": kind,
            "fold": fold,
        }

        selection_metric = None
        candidate_oof = run_dir / "oof_candidate"
        full_pq_due = (
            bool(valid_cfg["full_pq_enabled"])
            and (epoch % int(valid_cfg["full_pq_interval"]) == 0 or epoch == epochs)
        )
        if full_pq_due:
            _remove_generated_directory(candidate_oof, run_dir)
            oof_path = candidate_oof if valid_cfg.get("save_best_oof", True) else None
            full_report = evaluate_full_images(
                model,
                valid_records,
                coco,
                data_root / "train/train_images",
                device,
                valid_cfg,
                oof_dir=oof_path,
            )
            full_report.update({"epoch": epoch, "fold": fold})
            atomic_write_json(
                run_dir / f"metrics/full_pq_epoch{epoch:03d}.json", full_report
            )
            metrics.update({
                "official_pq": full_report["official_pq"],
                "dice_annotator_macro": full_report["dice_annotator_macro"],
                "iou_annotator_macro": full_report["iou_annotator_macro"],
                "pq_hit_pairs": full_report["hit_pairs"],
                "pq_false_positives": full_report["false_positives"],
                "pq_false_negatives": full_report["false_negatives"],
                "pq_physical_images": full_report["physical_images"],
            })
            selection_metric = float(full_report["official_pq"])
        elif best_metric_name == "negative_valid_loss":
            selection_metric = -float(valid_stats["loss"])

        is_best = selection_metric is not None and (
            best_metric is None or selection_metric > float(best_metric)
        )
        if is_best:
            best_metric = selection_metric
            if full_pq_due and candidate_oof.exists():
                atomic_write_json(candidate_oof / "manifest.json", {
                    **full_report,
                    "checkpoint": str(best_output),
                    "selection_metric": selection_metric,
                })
                _replace_generated_directory(candidate_oof, run_dir / "oof_best", run_dir)
            save_checkpoint(
                best_output, model, None, epoch, metrics, None, None,
                config, kind, normalized_model_cfg, global_step, best_metric,
                best_metric_name,
            )
        elif candidate_oof.exists():
            _remove_generated_directory(candidate_oof, run_dir)

        save_checkpoint(
            last_output, model, optimizer, epoch, metrics, scaler, scheduler,
            config, kind, normalized_model_cfg, global_step, best_metric,
            best_metric_name,
        )
        record = {
            "epoch": epoch,
            **metrics,
            "train_seconds": train_stats["seconds"],
            "learning_rate": lr_used,
            "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "global_step": global_step,
            "best_metric": best_metric,
            "best_metric_name": best_metric_name,
            "is_best": is_best,
            "rss_peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "device": str(device),
        }
        print(record)
        _append_jsonl(log_output, record)

    print(f"best checkpoint: {best_output}")
    print(f"last checkpoint: {last_output}")


if __name__ == "__main__":
    main()
