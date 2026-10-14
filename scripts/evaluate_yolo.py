#!/usr/bin/env python3
"""Search YOLO post-processing on a raw candidate cache using official PQ."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from time import perf_counter

from src.data.coco import CocoAnnotations
from src.instance.yolo_cache import (
    load_cache_manifest,
    sha256_file,
    validate_annotation_provenance,
    validate_search_space,
)
from src.instance.yolo_evaluation import YoloPostprocessConfig, evaluate_cached_grid
from src.io_utils import atomic_write_json


def _values(raw: list[str], cast) -> list:
    tokens = [token for value in raw for token in value.split(",") if token]
    return sorted(set(cast(token) for token in tokens))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026", type=Path)
    parser.add_argument("--validation-file-list", type=Path)
    parser.add_argument("--confidences", nargs="+", default=["0.05", "0.10", "0.20"])
    parser.add_argument("--min-areas", nargs="+", default=["200", "400", "600"])
    parser.add_argument("--max-instances", nargs="+", default=["5", "10", "20"])
    parser.add_argument(
        "--force-one", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--no-verify-cache-hashes",
        action="store_true",
        help="skip per-image SHA256 verification for faster repeated local searches",
    )
    parser.add_argument(
        "--output", default="artifacts/instance/yolo_postprocess_search.json", type=Path
    )
    args = parser.parse_args()

    confidences = _values(args.confidences, float)
    min_areas = _values(args.min_areas, int)
    instance_limits = _values(args.max_instances, int)
    if not confidences or min(confidences) < 0 or max(confidences) > 1:
        raise ValueError("confidences must be within [0, 1]")
    if not min_areas or min(min_areas) < 1:
        raise ValueError("min-areas must be positive")
    if not instance_limits or min(instance_limits) < 1:
        raise ValueError("max-instances must be positive")

    manifest = load_cache_manifest(args.cache_dir)
    validate_search_space(
        manifest, confidences=confidences, max_instances=instance_limits
    )
    selected_names = None
    if args.validation_file_list:
        selected_names = set(json.loads(args.validation_file_list.read_text()))
        if not selected_names:
            raise ValueError("validation file list is empty")

    annotation = (
        args.data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    validate_annotation_provenance(manifest, annotation)
    coco = CocoAnnotations(annotation)
    configs = [
        YoloPostprocessConfig(confidence, min_area, max_instances, args.force_one)
        for confidence, min_area, max_instances in itertools.product(
            confidences, min_areas, instance_limits
        )
    ]
    started = perf_counter()
    rows = evaluate_cached_grid(
        args.cache_dir,
        manifest,
        coco,
        configs,
        selected_names=selected_names,
        verify_cache_hashes=not args.no_verify_cache_hashes,
    )
    selected_count = (
        len(selected_names) if selected_names is not None else len(manifest["images"])
    )
    payload = {
        "cache_dir": str(args.cache_dir.resolve()),
        "cache_manifest_sha256": sha256_file(args.cache_dir / "manifest.json"),
        "cache_run_config_sha256": manifest["run_config_sha256"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "model_config_sha256": (
            manifest.get("model_config") or {}
        ).get("sha256"),
        "experiment": manifest.get("experiment", {}),
        "inference": manifest["inference"],
        "fold": manifest["dataset"].get("fold"),
        "physical_images": selected_count,
        "grid_size": len(configs),
        "search_seconds": perf_counter() - started,
        "best": rows[0],
        "candidates": rows,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["best"], indent=2))
    print(f"wrote {len(rows)} offline candidates to {args.output}")


if __name__ == "__main__":
    main()
