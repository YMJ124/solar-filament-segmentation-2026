#!/usr/bin/env python3
"""Build the date-folded, medoid-supervised YOLO segmentation dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.coco import CocoAnnotations
from src.data.folds import load_fold_manifest
from src.instance import build_yolo_segmentation_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--fold-manifest", default="data/manifests/train_folds.csv")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", default="data/processed/yolo_fold0")
    args = parser.parse_args()
    root = Path(args.data_root)
    annotation = root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = CocoAnnotations(annotation)
    records = load_fold_manifest(args.fold_manifest, annotation)
    manifest = build_yolo_segmentation_dataset(
        records,
        coco,
        root / "train/train_images",
        args.output,
        validation_fold=args.fold,
        fold_manifest=args.fold_manifest,
    )
    summary = {key: value for key, value in manifest.items() if key != "selected_image_ids"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
