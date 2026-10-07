#!/usr/bin/env python3
"""Audit raw files and COCO annotations without modifying raw data."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from PIL import Image

from src.data.coco import CocoAnnotations


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026")
    args = parser.parse_args()
    root = Path(args.data_root)
    coco = CocoAnnotations(root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    train = sorted((root / "train/train_images").glob("*"))
    test = sorted((root / "test/test_images").glob("*"))
    dimensions = collections.Counter()
    byte_hashes = collections.Counter()
    for path in train + test:
        with Image.open(path) as image:
            dimensions[(image.size, image.mode, image.format)] += 1
        byte_hashes[sha256(path)] += 1
    report = {
        "data_root": str(root),
        "train_files": len(train),
        "test_files": len(test),
        "json_image_records": len(coco.images),
        "unique_physical_files_in_json": len(coco.physical_files),
        "annotation_records": len(coco.annotations),
        "categories": coco.categories,
        "duplicate_byte_groups": sum(value > 1 for value in byte_hashes.values()),
        "dimensions": {str(key): value for key, value in dimensions.items()},
        "multi_annotator_physical_files": sum(len(coco.batches_for_file(name)) > 1 for name in coco.physical_files),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
