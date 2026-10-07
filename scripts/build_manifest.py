#!/usr/bin/env python3
"""Build a CSV manifest keyed by physical observation file name."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from PIL import Image

from src.data.coco import CocoAnnotations
from src.io_utils import atomic_write_csv


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--output", default="data/manifests/physical_images.csv")
    args = parser.parse_args()
    root = Path(args.data_root)
    coco = CocoAnnotations(root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    train_dir = root / "train/train_images"
    rows = []
    for record in coco.iter_physical_records():
        path = train_dir / record["file_name"]
        with Image.open(path) as image:
            mode = image.mode
            width, height = image.size
        rows.append({
            "file_name": record["file_name"], "split": "train", "path": str(path),
            "width": width, "height": height, "mode": mode, "sha256": file_hash(path),
            "n_batches": record["n_batches"], "n_annotations": record["n_annotations"],
            "category_counts": str(record["category_counts"]),
        })
    test_dir = root / "test/test_images"
    for path in sorted(test_dir.glob("*")):
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        rows.append({
            "file_name": path.name, "split": "test", "path": str(path),
            "width": width, "height": height, "mode": mode, "sha256": file_hash(path),
            "n_batches": 0, "n_annotations": 0, "category_counts": "{}",
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    atomic_write_csv(output, fields, rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
