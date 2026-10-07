#!/usr/bin/env python3
"""Compute exact grayscale normalization statistics without loading all images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from src.data.folds import load_fold_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--fold-manifest", default="data/manifests/train_folds.csv")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.data_root)
    annotation_file = root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    records = load_fold_manifest(args.fold_manifest, annotation_file)
    records = [row for row in records if int(row["fold"]) != args.fold]
    total = square_total = 0.0
    pixel_count = 0
    for index, row in enumerate(records, start=1):
        with Image.open(root / "train/train_images" / row["file_name"]) as image:
            array = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
        total += float(array.sum())
        square_total += float(np.square(array).sum())
        pixel_count += int(array.size)
        if index % 100 == 0:
            print(f"processed {index}/{len(records)}")
    mean = total / pixel_count
    standard_deviation = max(0.0, square_total / pixel_count - mean * mean) ** 0.5
    report = {
        "fold": args.fold,
        "training_images": len(records),
        "pixel_count": pixel_count,
        "input_mean": mean,
        "input_std": standard_deviation,
    }
    payload = json.dumps(report, indent=2) + "\n"
    print(payload, end="")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)


if __name__ == "__main__":
    main()
