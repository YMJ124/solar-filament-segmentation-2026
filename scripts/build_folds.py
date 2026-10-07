#!/usr/bin/env python3
"""Create deterministic grouped folds for training physical observations."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from src.data.coco import CocoAnnotations
from src.data.folds import (
    GROUP_STRATEGY,
    MANIFEST_SCHEMA_VERSION,
    TARGET_POLICY,
    assign_date_grouped_folds,
)
from src.data.medoid import select_medoid_batch
from src.io_utils import atomic_write_csv, atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--output", default="data/manifests/train_folds.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    root = Path(args.data_root)
    annotation_file = root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = CocoAnnotations(annotation_file)
    rows = assign_date_grouped_folds(
        list(coco.iter_physical_records()), args.n_splits, args.seed
    )
    for index, row in enumerate(rows, start=1):
        selection = select_medoid_batch(coco, str(row["file_name"]))
        row["selected_image_id"] = selection.selected_image_id
        row["selected_annotations"] = selection.selected_annotations
        row["medoid_mean_iou"] = selection.medoid_mean_iou
        if index % 50 == 0 or index == len(rows):
            print(f"selected medoids {index}/{len(rows)}")
    rows.sort(key=lambda row: str(row["file_name"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file_name",
        "fold",
        "group",
        "width",
        "height",
        "n_batches",
        "n_annotations",
        "selected_image_id",
        "selected_annotations",
        "medoid_mean_iou",
        "category_counts",
    ]
    atomic_write_csv(
        output,
        fields,
        ({key: row[key] for key in fields} for row in rows),
    )
    metadata = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "target_policy": TARGET_POLICY,
        "group_strategy": GROUP_STRATEGY,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "rows": len(rows),
        "annotation_sha256": hashlib.sha256(annotation_file.read_bytes()).hexdigest(),
    }
    metadata_path = output.with_suffix(".meta.json")
    atomic_write_json(metadata_path, metadata)
    print(f"wrote {len(rows)} grouped rows to {output}")
    print(f"wrote manifest metadata to {metadata_path}")


if __name__ == "__main__":
    main()
