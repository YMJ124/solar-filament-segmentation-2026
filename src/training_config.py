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
        "schema_version": 3,
        "run": {"name": "global_resnet34_fold0", "seed": 2026, "deterministic": True},
        "data": {
            "data_root": "MAGFiLO_1.0_Kaggle_2026",
            "fold_manifest": "data/manifests/train_folds.csv",
            "validation_file_list": None,
            "mask_cache_dir": "data/cache/medoid_semantic_v2",
            "sample_mode": "crop",
            "fold": 0,
            "image_size": 256,
            "positive_crop_probability": 0.7,
            "num_workers": 2,
        },
        "model": {
            "kind": "global",
            "architecture": "smp_unet",
            "encoder_name": "resnet34",
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
            "optimizer": "adamw",
            "loss": {"name": "bce_dice", "max_positive_weight": 20.0},
            "warmup_epochs": 3,
            "amp": True,
            "gradient_clip_norm": 1.0,
            "max_steps": None,
            "max_validation_batches": None,
        },
        "validation": {
            "full_pq_enabled": True,
            "inference_mode": "sliding_window",
            "full_pq_interval": 5,
            "tile_size": 256,
            "tile_batch_size": 8,
            "overlap": 0.25,
            "threshold": 0.5,
            "min_area": 80,
            "closing_iterations": 1,
            "max_instances": 128,
            "max_images": None,
            "save_best_oof": True,
        },
        "artifacts": {"run_dir": "artifacts/runs/global_resnet34_fold0"},
    }


def validate_training_config(config: Mapping[str, Any]) -> None:
    kind = config["model"]["kind"]
    if kind != "global":
        raise ValueError(f"unknown model kind: {kind}")
    if config["model"]["architecture"] not in {
        "compact_unet", "smp_unet", "resnet34_unet", "flat_unet"
    }:
        raise ValueError("unsupported model.architecture")
    if config["model"]["architecture"] == "smp_unet" and not str(
        config["model"].get("encoder_name", "")
    ).strip():
        raise ValueError("smp_unet requires model.encoder_name")
    if config["model"]["architecture"] == "flat_unet":
        if int(config["model"].get("channels", 0)) < 1:
            raise ValueError("flat_unet requires positive model.channels")
        if len(config["model"].get("simplified_blocks", [])) != 5:
            raise ValueError("flat_unet requires five model.simplified_blocks values")
    if int(config["training"]["epochs"]) < 1:
        raise ValueError("training.epochs must be >= 1")
    if int(config["training"]["batch_size"]) < 1:
        raise ValueError("training.batch_size must be >= 1")
    if int(config["training"]["gradient_accumulation"]) < 1:
        raise ValueError("training.gradient_accumulation must be >= 1")
    if float(config["training"]["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if config["training"]["optimizer"] not in {"adam", "adamw"}:
        raise ValueError("training.optimizer must be adam or adamw")
    loss = config["training"]["loss"]
    if loss["name"] not in {"bce_dice", "balanced_bce_dice"}:
        raise ValueError("training.loss.name is unsupported")
    if float(loss["max_positive_weight"]) < 1:
        raise ValueError("training.loss.max_positive_weight must be >= 1")
    image_size = int(config["data"]["image_size"])
    if image_size < 32 or image_size % 32:
        raise ValueError("data.image_size must be >= 32 and divisible by 32")
    if config["data"]["sample_mode"] not in {"crop", "full_resize"}:
        raise ValueError("data.sample_mode must be crop or full_resize")
    positive_probability = float(config["data"]["positive_crop_probability"])
    if not 0 <= positive_probability <= 1:
        raise ValueError("data.positive_crop_probability must be in [0, 1]")
    validation_file_list = config["data"].get("validation_file_list")
    if validation_file_list is not None and not str(validation_file_list).strip():
        raise ValueError("data.validation_file_list must be a path or null")
    tile_size = int(config["validation"]["tile_size"])
    if tile_size < 32 or tile_size % 32:
        raise ValueError("validation.tile_size must be >= 32 and divisible by 32")
    if config["validation"]["inference_mode"] not in {
        "sliding_window", "full_resize"
    }:
        raise ValueError("validation.inference_mode must be sliding_window or full_resize")
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
