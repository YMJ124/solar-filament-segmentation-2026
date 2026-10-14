#!/usr/bin/env python3
"""Run a tiny one-epoch CPU smoke test of the random-annotator trainer."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.instance.annotator_schedule import RANDOM_ANNOTATOR_TARGET_POLICY
from src.instance.yolo_random_annotator import random_annotator_trainer_class
from src.ultralytics_runtime import prepare_ultralytics_runtime


def _label(path: Path, offset: float) -> None:
    low, high = 0.2 + offset, 0.7 + offset
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"0 {low:.3f} {low:.3f} {high:.3f} {low:.3f} "
        f"{high:.3f} {high:.3f} {low:.3f} {high:.3f}\n"
    )


def _fixture(root: Path) -> Path:
    train_targets = {}
    for split, names in {"train": ("train_a.jpg", "train_b.jpg"), "val": ("val.jpg",)}.items():
        for index, name in enumerate(names):
            image_path = root / "images" / split / name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.full((64, 64, 3), 40 + index * 40, dtype=np.uint8)
            Image.fromarray(pixels).save(image_path)
            _label(root / "labels" / split / f"{Path(name).stem}.txt", 0.0)
            if split == "train":
                rows = []
                for annotator, offset in enumerate((0.0, 0.05)):
                    alternative = (
                        root / "annotator_labels" / Path(name).stem / f"{annotator:02d}.txt"
                    )
                    _label(alternative, offset)
                    rows.append(
                        {
                            "image_id": f"{Path(name).stem}-annotator-{annotator}",
                            "label_file": str(alternative.resolve()),
                        }
                    )
                train_targets[name] = rows
    annotator_manifest = root / "annotator_manifest.json"
    annotator_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target_policy": RANDOM_ANNOTATOR_TARGET_POLICY,
                "selection_seed": 2026,
                "physical_train_images": 2,
                "train_targets": train_targets,
            }
        )
    )
    dataset = root / "dataset.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
                "path": str(root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "filament"},
                "target_policy": RANDOM_ANNOTATOR_TARGET_POLICY,
                "annotator_manifest": str(annotator_manifest.resolve()),
                "selection_seed": 2026,
                "validation_target_policy": "medoid_annotator_v1",
            },
            sort_keys=False,
        )
    )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo26n-seg.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    prepare_ultralytics_runtime()
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory(prefix="random-annotator-smoke-") as temporary:
        root = Path(temporary)
        dataset = _fixture(root / "dataset")
        YOLO(args.weights).train(
            trainer=random_annotator_trainer_class(),
            data=str(dataset),
            project=str(root / "runs"),
            name="smoke",
            epochs=1,
            imgsz=64,
            batch=1,
            device=args.device,
            workers=0,
            cache=False,
            amp=False,
            val=False,
            plots=False,
            save=False,
            seed=2026,
            deterministic=True,
            mosaic=0.0,
            degrees=0.0,
            fliplr=0.0,
            flipud=0.0,
            verbose=False,
        )
    print("random-annotator one-epoch smoke test passed")


if __name__ == "__main__":
    main()
