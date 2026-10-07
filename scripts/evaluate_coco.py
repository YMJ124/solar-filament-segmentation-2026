#!/usr/bin/env python3
"""Score a competition-format validation CSV with the official PQ v6 logic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.coco import CocoAnnotations
from src.data.folds import load_fold_manifest
from src.metrics.official_evaluator import evaluate_predictions, load_prediction_csv


def _selected_files(coco, fold, fold_manifest, annotation_file, max_images):
    if fold is None:
        files = coco.physical_files
    else:
        records = load_fold_manifest(fold_manifest, annotation_file)
        files = [row["file_name"] for row in records if row["fold"] == fold]
    return files[:max_images]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--predictions", required=True, help="competition-format CSV")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--fold-manifest", default="data/manifests/train_folds.csv")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.data_root)
    annotation_file = root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = CocoAnnotations(annotation_file)
    predictions = load_prediction_csv(args.predictions, coco)
    files = _selected_files(
        coco, args.fold, args.fold_manifest, annotation_file, args.max_images
    )
    report = evaluate_predictions(coco, predictions, files).to_dict()
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
