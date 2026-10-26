#!/usr/bin/env python3
"""Run one configuration-recorded Ultralytics instance-segmentation training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.instance.annotator_schedule import (
    MEDOID_TARGET_POLICY,
    RANDOM_ANNOTATOR_TARGET_POLICY,
    SUPPORTED_TARGET_POLICIES,
)
from src.ultralytics_runtime import prepare_ultralytics_runtime
from src.instance.yolo_sampling import (
    STATION_PRESERVING_MEDIUM_SAMPLING_POLICY,
    SUPPORTED_SAMPLING_POLICIES,
    UNIFORM_SAMPLING_POLICY,
    validate_sampling_manifest,
)


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
    dataset_config = yaml.safe_load(dataset.read_text())
    target_policy = str(
        config["data"].get("target_policy", MEDOID_TARGET_POLICY)
    )
    if target_policy not in SUPPORTED_TARGET_POLICIES:
        raise ValueError(f"unsupported YOLO target policy: {target_policy}")
    if dataset_config.get("target_policy", MEDOID_TARGET_POLICY) != target_policy:
        raise ValueError(
            "training config target policy does not match dataset YAML: "
            f"{target_policy} != {dataset_config.get('target_policy')}"
        )
    if (
        target_policy == RANDOM_ANNOTATOR_TARGET_POLICY
        and int(config["training"].get("workers", -1)) != 0
    ):
        raise ValueError("random-annotator training requires workers=0")
    sampling_policy = str(
        config["data"].get("sampling_policy", UNIFORM_SAMPLING_POLICY)
    )
    if sampling_policy not in SUPPORTED_SAMPLING_POLICIES:
        raise ValueError(f"unsupported YOLO sampling policy: {sampling_policy}")
    if sampling_policy != UNIFORM_SAMPLING_POLICY:
        if target_policy != MEDOID_TARGET_POLICY:
            raise ValueError("P4 sampling requires the medoid target policy")
        if int(config["training"].get("workers", -1)) != 0:
            raise ValueError("P4 deterministic sampling requires workers=0")
        manifest_value = config["data"].get("sampling_manifest")
        if not manifest_value:
            raise ValueError("P4 sampling requires data.sampling_manifest")
        manifest_path = Path(str(manifest_value))
        if not manifest_path.is_file():
            raise FileNotFoundError(f"sampling manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        validate_sampling_manifest(manifest)
        if manifest.get("sampling_policy") != sampling_policy:
            raise ValueError("training config sampling policy does not match manifest")
    return config


def training_arguments(config: dict) -> dict:
    run, training = config["run"], dict(config["training"])
    return {
        "data": str(Path(config["data"]["dataset_yaml"]).resolve()),
        # Ultralytics nests relative projects below its global runs directory.
        "project": str(Path(run["project"]).resolve()),
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
        print(
            json.dumps(
                {
                    "model": config["model"]["weights"],
                    "sampling_policy": config["data"].get(
                        "sampling_policy", UNIFORM_SAMPLING_POLICY
                    ),
                    "sampling_manifest": config["data"].get("sampling_manifest"),
                    **train_args,
                },
                indent=2,
            )
        )
        return
    prepare_ultralytics_runtime()
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
    trainer = None
    if config["data"].get("target_policy") == RANDOM_ANNOTATOR_TARGET_POLICY:
        from src.instance.yolo_random_annotator import random_annotator_trainer_class

        trainer = random_annotator_trainer_class()
    sampling_policy = config["data"].get(
        "sampling_policy", UNIFORM_SAMPLING_POLICY
    )
    if sampling_policy == STATION_PRESERVING_MEDIUM_SAMPLING_POLICY:
        from src.instance.yolo_sampling import (
            station_preserving_sampler_trainer_class,
        )

        trainer = station_preserving_sampler_trainer_class(
            config["data"]["sampling_manifest"],
            selection_seed=int(config["run"]["seed"]),
        )
    if trainer is None:
        model.train(**train_args)
    else:
        model.train(trainer=trainer, **train_args)


if __name__ == "__main__":
    main()
