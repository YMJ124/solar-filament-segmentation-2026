#!/usr/bin/env python3
"""Build the fixed-budget P4 station-preserving YOLO sampling manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.coco import CocoAnnotations
from src.data.folds import load_fold_manifest
from src.instance.yolo_sampling import build_station_preserving_sampling_manifest
from src.io_utils import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026", type=Path)
    parser.add_argument(
        "--fold-manifest", default="data/manifests/train_folds.csv", type=Path
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument(
        "--all-data",
        action="store_true",
        help="Use every fold for a final-fit sampling manifest.",
    )
    parser.add_argument("--dataset-yaml", required=True, type=Path)
    parser.add_argument("--selection-seed", type=int, default=2026)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.all_data == (args.fold is not None):
        raise ValueError("provide exactly one of --fold or --all-data")

    annotation = args.data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = CocoAnnotations(annotation)
    records = load_fold_manifest(args.fold_manifest, annotation)
    payload = build_station_preserving_sampling_manifest(
        records,
        coco,
        validation_fold=None if args.all_data else args.fold,
        fold_manifest=args.fold_manifest,
        dataset_yaml=args.dataset_yaml,
        selection_seed=args.selection_seed,
    )
    atomic_write_json(args.output, payload)
    summary = {key: value for key, value in payload.items() if key != "images"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
