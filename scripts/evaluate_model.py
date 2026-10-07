#!/usr/bin/env python3
"""Evaluate checkpoints on full images with the official annotator-aware PQ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.coco import CocoAnnotations
from src.data.folds import load_fold_manifest
from src.full_image_validation import evaluate_full_images
from src.io_utils import atomic_write_json
from src.models import load_model_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--fold-manifest", default="data/manifests/train_folds.csv")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-batch-size", type=int, default=8)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--closing-iterations", type=int, default=1)
    parser.add_argument("--max-instances", type=int, default=128)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.data_root)
    annotation_file = root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = CocoAnnotations(annotation_file)
    records = load_fold_manifest(args.fold_manifest, annotation_file)
    records = [row for row in records if row["fold"] == args.fold]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.checkpoint, device, expected_kind="global")
    report = evaluate_full_images(
        model,
        records,
        coco,
        root / "train/train_images",
        device,
        {
            "tile_size": args.tile_size,
            "tile_batch_size": args.tile_batch_size,
            "overlap": args.overlap,
            "threshold": args.threshold,
            "min_area": args.min_area,
            "closing_iterations": args.closing_iterations,
            "max_instances": args.max_instances,
            "max_images": args.max_images,
        },
    )
    report["fold"] = args.fold
    report["checkpoint"] = args.checkpoint
    print(json.dumps(report, indent=2))
    if args.output:
        atomic_write_json(args.output, report)


if __name__ == "__main__":
    main()
