#!/usr/bin/env python3
"""Run one configuration-recorded Ultralytics instance-segmentation training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    required = {"run", "data", "model", "training"}
    if not isinstance(config, dict) or required - set(config):
        raise ValueError(f"invalid YOLO config; required sections: {sorted(required)}")
    dataset = Path(config["data"]["dataset_yaml"])
    if not dataset.is_file():
        raise FileNotFoundError(
            f"YOLO dataset is missing: {dataset}; run scripts.build_yolo_dataset first"
        )
    return config


def training_arguments(config: dict) -> dict:
    run, training = config["run"], dict(config["training"])
    return {
        "data": config["data"]["dataset_yaml"],
        "project": run["project"],
        "name": run["name"],
        "seed": int(run["seed"]),
        "deterministic": bool(run.get("deterministic", True)),
        **training,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    train_args = training_arguments(config)
    output = Path(train_args["project"]) / str(train_args["name"])
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"training output already exists: {output}")
    if args.dry_run:
        print(json.dumps({"model": config["model"]["weights"], **train_args}, indent=2))
        return
    config_dir = Path("artifacts/.ultralytics").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is optional; install requirements-yolo.txt before training"
        ) from exc
    expected = str(config["model"].get("ultralytics_version", ""))
    if expected and ultralytics.__version__ != expected:
        raise RuntimeError(
            f"ultralytics {ultralytics.__version__} != pinned {expected}"
        )
    model = YOLO(config["model"]["weights"])
    model.train(**train_args)


if __name__ == "__main__":
    main()
