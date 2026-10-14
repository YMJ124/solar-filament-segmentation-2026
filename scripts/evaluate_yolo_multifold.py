#!/usr/bin/env python3
"""Search one shared YOLO post-process grid across multiple OOF fold caches."""

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
from src.instance.yolo_multifold import (
    aggregate_multifold_candidates,
    validate_multifold_manifests,
)
from src.io_utils import atomic_write_json


def _values(raw: list[str], cast) -> list:
    tokens = [token for value in raw for token in value.split(",") if token]
    return sorted(set(cast(token) for token in tokens))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", action="append", required=True, type=Path)
    parser.add_argument("--folds", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--data-root", default="MAGFiLO_1.0_Kaggle_2026", type=Path)
    parser.add_argument(
        "--confidences", nargs="+", default=["0.05", "0.10", "0.15", "0.20", "0.25"]
    )
    parser.add_argument("--min-areas", nargs="+", default=["200", "400", "600", "800"])
    parser.add_argument(
        "--max-instances", nargs="+", default=["8", "10", "12", "15", "20"]
    )
    parser.add_argument(
        "--force-one", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--no-verify-cache-hashes", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    expected_folds = set(_values(args.folds, int))
    confidences = _values(args.confidences, float)
    min_areas = _values(args.min_areas, int)
    instance_limits = _values(args.max_instances, int)
    if not confidences or min(confidences) < 0 or max(confidences) > 1:
        raise ValueError("confidences must be within [0, 1]")
    if not min_areas or min(min_areas) < 1:
        raise ValueError("min-areas must be positive")
    if not instance_limits or min(instance_limits) < 1:
        raise ValueError("max-instances must be positive")

    annotation = (
        args.data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    coco = CocoAnnotations(annotation)
    configs = [
        YoloPostprocessConfig(confidence, min_area, max_instances, args.force_one)
        for confidence, min_area, max_instances in itertools.product(
            confidences, min_areas, instance_limits
        )
    ]
    manifests: dict[int, tuple[Path, dict]] = {}
    for cache_dir in args.cache_dir:
        manifest = load_cache_manifest(cache_dir)
        fold = int(manifest["dataset"]["fold"])
        if fold in manifests:
            raise ValueError(f"duplicate cache for fold {fold}")
        validate_annotation_provenance(manifest, annotation)
        validate_search_space(
            manifest, confidences=confidences, max_instances=instance_limits
        )
        manifests[fold] = (cache_dir, manifest)
    provenance = validate_multifold_manifests(
        {fold: manifest for fold, (_, manifest) in manifests.items()}, expected_folds
    )

    started = perf_counter()
    fold_candidates = {}
    cache_rows = []
    for fold in sorted(manifests):
        cache_dir, manifest = manifests[fold]
        fold_candidates[fold] = evaluate_cached_grid(
            cache_dir,
            manifest,
            coco,
            configs,
            verify_cache_hashes=not args.no_verify_cache_hashes,
        )
        cache_rows.append(
            {
                "fold": fold,
                "cache_dir": str(cache_dir.resolve()),
                "cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
                "cache_run_config_sha256": manifest["run_config_sha256"],
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                "model_config_sha256": (
                    manifest.get("model_config") or {}
                ).get("sha256"),
                "experiment": manifest.get("experiment", {}),
                "physical_images": manifest["physical_images"],
            }
        )
    rows = aggregate_multifold_candidates(fold_candidates)
    payload = {
        "report_type": "yolo_multifold_postprocess_v1",
        "folds": sorted(manifests),
        "grid": {
            "confidences": confidences,
            "min_areas": min_areas,
            "max_instances": instance_limits,
            "force_one": args.force_one,
            "size": len(configs),
        },
        **provenance,
        "caches": cache_rows,
        "search_seconds": perf_counter() - started,
        "best": rows[0],
        "candidates": rows,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["best"], indent=2))
    print(f"wrote {len(rows)} shared-grid candidates to {args.output}")


if __name__ == "__main__":
    main()
