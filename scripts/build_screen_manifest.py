#!/usr/bin/env python3
"""Select a small deterministic validation panel spanning easy and hard images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.folds import load_fold_manifest
from src.io_utils import atomic_write_json


def select_panel(records: list[dict], size: int) -> list[str]:
    """Take evenly spaced ranks after sorting by annotation complexity."""
    if size < 1:
        raise ValueError("size must be positive")
    ordered = sorted(
        records,
        key=lambda row: (
            int(row["selected_annotations"]),
            int(row["n_batches"]),
            float(row.get("medoid_mean_iou", 1.0)),
            str(row["file_name"]),
        ),
    )
    if size >= len(ordered):
        return [str(row["file_name"]) for row in ordered]
    indices = np.linspace(0, len(ordered) - 1, size, dtype=int)
    return [str(ordered[index]["file_name"]) for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-manifest", default="data/manifests/train_folds.csv")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--output", default="data/manifests/screen_fold0_32.json")
    args = parser.parse_args()

    records = load_fold_manifest(args.fold_manifest)
    fold_records = [row for row in records if int(row["fold"]) == args.fold]
    if not fold_records:
        raise ValueError(f"fold {args.fold} has no records")
    selected = select_panel(fold_records, args.size)
    atomic_write_json(args.output, selected)
    print(f"wrote {len(selected)} deterministic fold-{args.fold} files to {args.output}")


if __name__ == "__main__":
    main()
