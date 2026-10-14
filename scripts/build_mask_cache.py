#!/usr/bin/env python3
"""Precompute versioned medoid semantic targets and instance boxes."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.cache import CACHE_MANIFEST_NAME, expected_cache_metadata
from src.data.coco import CocoAnnotations
from src.data.dataset import _instance_boxes, _semantic_target
from src.data.folds import load_fold_manifest
from src.data.masks import annotations_to_masks
from src.io_utils import atomic_write_json
from src.legacy_semantic import add_legacy_semantic_flag, require_legacy_semantic_opt_in


def _prepare_manifest(output, annotation_file, fold_manifest, expected_images):
    expected = expected_cache_metadata(
        annotation_file, fold_manifest, expected_images
    )
    path = output / CACHE_MANIFEST_NAME
    if path.exists():
        current = json.loads(path.read_text())
        if current != expected:
            raise ValueError(
                f"cache manifest mismatch; use a new cache directory: {current} != {expected}"
            )
    else:
        if any(output.glob("*.npz")):
            raise ValueError("cache entries exist without a compatible manifest")
        atomic_write_json(path, expected)
    return expected


def _write_cache_entry(path, selected_image_id, semantic, boxes):
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}-", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle,
            selected_image_id=np.asarray(selected_image_id),
            semantic=semantic.astype(np.uint8),
            instance_boxes=boxes.astype(np.int32),
        )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_legacy_semantic_flag(parser)
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026")
    parser.add_argument("--fold-manifest", default="data/manifests/train_folds.csv")
    parser.add_argument("--output", default="data/cache/medoid_semantic_v2")
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()
    require_legacy_semantic_opt_in(args.allow_legacy_semantic)

    root = Path(args.data_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    annotation_file = root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    fold_manifest = Path(args.fold_manifest)
    records = load_fold_manifest(fold_manifest, annotation_file)
    _prepare_manifest(output, annotation_file, fold_manifest, len(records))
    coco = CocoAnnotations(annotation_file)
    selected_records = records[:args.max_images]
    for index, record in enumerate(selected_records, start=1):
        name = str(record["file_name"])
        path = output / f"{Path(name).stem}.npz"
        selected_image_id = str(record["selected_image_id"])
        if path.exists():
            with np.load(path) as cached:
                cached_image_id = str(cached["selected_image_id"].item())
            if cached_image_id != selected_image_id:
                raise ValueError(
                    f"stale cache entry for {name}: {cached_image_id} != {selected_image_id}"
                )
        else:
            instances = annotations_to_masks(
                coco.annotations_for_batch(selected_image_id),
                int(record["height"]),
                int(record["width"]),
            )
            semantic = _semantic_target(
                instances, (int(record["height"]), int(record["width"]))
            )
            _write_cache_entry(
                path,
                selected_image_id,
                semantic,
                _instance_boxes(instances),
            )
        if index % 25 == 0 or index == len(selected_records):
            print(f"cached {index}/{len(selected_records)}")


if __name__ == "__main__":
    main()
