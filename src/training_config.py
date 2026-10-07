"""Loading and validation for reproducible training configurations."""

from __future__ import annotations

import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def _merge(target: dict, updates: Mapping[str, Any]) -> dict:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def default_training_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run": {"name": "global_resnet34_fold0", "seed": 2026, "deterministic": True},
        "data": {
            "data_root": "MAGFiLO_1.0_Kaggle_2026",
            "fold_manifest": "data/manifests/train_folds.csv",
            "mask_cache_dir": "data/cache/medoid_v1",
            "fold": 0,
            "image_size": 256,
            "positive_crop_probability": 0.7,
            "num_workers": 2,
        },
        "model": {
            "kind": "global",
            "architecture": "resnet34_unet",
            "encoder_weights": "imagenet",
            "input_mean": 0.326732,
            "input_std": 0.223022,
        },
        "training": {
            "epochs": 50,
            "batch_size": 4,
            "gradient_accumulation": 2,
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-4,
            "warmup_epochs": 3,
            "amp": True,
            "gradient_clip_norm": 1.0,
            "max_steps": None,
            "max_validation_batches": None,
            "teacher_checkpoint": None,
            "teacher_probability_augmentation": {
                "dropout": 0.0,
                "blur_probability": 0.0,
                "blur_kernel": 3,
            },
        },
        "validation": {
            "full_pq_enabled": True,
            "full_pq_interval": 5,
            "tile_size": 256,
            "tile_batch_size": 8,
            "overlap": 0.25,
            "threshold": 0.5,
            "min_area": 80,
            "closing_iterations": 1,
            "boundary_threshold": 0.5,
            "marker_threshold": 0.15,
            "marker_min_distance": 8,
            "max_instances": 128,
            "max_images": None,
            "save_best_oof": True,
            "save_components": False,
        },
        "artifacts": {"run_dir": "artifacts/runs/global_resnet34_fold0"},
    }


def validate_training_config(config: Mapping[str, Any]) -> None:
    kind = config["model"]["kind"]
    if kind not in {"global", "detail", "instance"}:
        raise ValueError(f"unknown model kind: {kind}")
    if int(config["training"]["epochs"]) < 1:
        raise ValueError("training.epochs must be >= 1")
    if int(config["training"]["batch_size"]) < 1:
        raise ValueError("training.batch_size must be >= 1")
    if int(config["training"]["gradient_accumulation"]) < 1:
        raise ValueError("training.gradient_accumulation must be >= 1")
    if float(config["training"]["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate must be positive")
    image_size = int(config["data"]["image_size"])
    if image_size < 32 or image_size % 32:
        raise ValueError("data.image_size must be >= 32 and divisible by 32")
    positive_probability = float(config["data"]["positive_crop_probability"])
    if not 0 <= positive_probability <= 1:
        raise ValueError("data.positive_crop_probability must be in [0, 1]")
    tile_size = int(config["validation"]["tile_size"])
    if tile_size < 32 or tile_size % 32:
        raise ValueError("validation.tile_size must be >= 32 and divisible by 32")
    if not 0 <= float(config["validation"]["overlap"]) < 1:
        raise ValueError("validation.overlap must be in [0, 1)")
    if not 0 < float(config["validation"]["threshold"]) < 1:
        raise ValueError("validation.threshold must be in (0, 1)")
    if int(config["validation"]["min_area"]) < 1:
        raise ValueError("validation.min_area must be >= 1")
    if int(config["validation"]["max_instances"]) < 1:
        raise ValueError("validation.max_instances must be >= 1")
    if int(config["validation"]["full_pq_interval"]) < 1:
        raise ValueError("validation.full_pq_interval must be >= 1")
    if int(config["validation"]["tile_batch_size"]) < 1:
        raise ValueError("validation.tile_batch_size must be >= 1")
    if kind == "detail" and not config["training"].get("teacher_checkpoint"):
        raise ValueError("detail training requires training.teacher_checkpoint")
    augmentation = config["training"].get("teacher_probability_augmentation") or {}
    dropout = float(augmentation.get("dropout", 0.0))
    blur_probability = float(augmentation.get("blur_probability", 0.0))
    blur_kernel = int(augmentation.get("blur_kernel", 3))
    if not 0 <= dropout < 1:
        raise ValueError("teacher probability dropout must be in [0, 1)")
    if not 0 <= blur_probability <= 1:
        raise ValueError("teacher blur_probability must be in [0, 1]")
    if blur_kernel < 1 or blur_kernel % 2 == 0:
        raise ValueError("teacher blur_kernel must be a positive odd integer")


def load_training_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> dict:
    config = default_training_config()
    path = Path(path)
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"training config must be a mapping: {path}")
    _merge(config, loaded)
    if overrides:
        _merge(config, overrides)
    validate_training_config(config)
    return config


def atomic_write_yaml(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            yaml.safe_dump(dict(payload), handle, sort_keys=False, allow_unicode=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
