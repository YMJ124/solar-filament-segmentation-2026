#!/usr/bin/env python3
"""Tune connected-component post-processing on one complete OOF fold."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.data.coco import CocoAnnotations
from src.data.folds import load_fold_manifest
from src.io_utils import atomic_write_csv, atomic_write_json
from src.legacy_semantic import add_legacy_semantic_flag, require_legacy_semantic_opt_in
from src.oof_postprocess import (
    PostprocessConfig,
    evaluate_oof_configuration,
    search_oof_postprocess,
    validate_oof_directory,
)


DEFAULT_CONFIG = PostprocessConfig()
DEFAULT_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
DEFAULT_MIN_AREAS = [20, 50, 80, 120, 160]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_legacy_semantic_flag(parser)
    parser.add_argument(
        "--oof-dir",
        default="artifacts/runs/global_resnet34_fold0/oof_best",
        type=Path,
    )
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        default=[str(value) for value in DEFAULT_THRESHOLDS],
        help="Space- or comma-separated coarse thresholds.",
    )
    parser.add_argument(
        "--min-areas",
        nargs="+",
        default=[str(value) for value in DEFAULT_MIN_AREAS],
        help="Space- or comma-separated coarse minimum component areas.",
    )
    parser.add_argument(
        "--closing-iterations",
        nargs="+",
        default=["1"],
        help="Space- or comma-separated coarse closing iteration counts.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/postprocess/fold0",
        type=Path,
    )
    parser.add_argument(
        "--data-root", default="MAGFiLO_1.0_Kaggle_2026", type=Path
    )
    parser.add_argument(
        "--fold-manifest", default="data/manifests/train_folds.csv", type=Path
    )
    parser.add_argument("--max-instances", default=128, type=int)
    parser.add_argument("--max-fine-candidates", default=40, type=int)
    parser.add_argument("--minimum-refine-gain", default=0.003, type=float)
    parser.add_argument("--minimum-acceptance-pq", default=0.095, type=float)
    parser.add_argument("--minimum-acceptance-tp", default=235, type=int)
    parser.add_argument(
        "--refine", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--repeat-best-check", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def parse_values(values: Iterable[str], cast, name: str) -> list[Any]:
    tokens = []
    for value in values:
        tokens.extend(part.strip() for part in str(value).split(",") if part.strip())
    try:
        parsed = [cast(token) for token in tokens]
    except ValueError as error:
        raise ValueError(f"invalid {name}: {tokens}") from error
    if not parsed:
        raise ValueError(f"{name} cannot be empty")
    return sorted(set(parsed))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def make_configs(
    thresholds: Iterable[float],
    min_areas: Iterable[int],
    closing_iterations: Iterable[int],
    max_instances: int,
) -> set[PostprocessConfig]:
    return {
        PostprocessConfig(
            threshold=float(threshold),
            min_area=int(min_area),
            closing_iterations=int(closing),
            max_instances=max_instances,
        )
        for threshold, min_area, closing in itertools.product(
            thresholds, min_areas, closing_iterations
        )
    }


def row_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(row["official_pq"]),
        int(row["hit_pairs"]),
        -int(row["false_positives"]),
        -int(row["false_negatives"]),
        -int(row["images_at_safety_cap"]),
        -abs(float(row["threshold"]) - DEFAULT_CONFIG.threshold),
        -abs(int(row["min_area"]) - DEFAULT_CONFIG.min_area),
        -int(row["closing_iterations"]),
    )


def config_from_row(row: dict[str, Any]) -> PostprocessConfig:
    return PostprocessConfig(
        threshold=float(row["threshold"]),
        min_area=int(row["min_area"]),
        closing_iterations=int(row["closing_iterations"]),
        max_instances=int(row["max_instances"]),
        max_area_fraction=float(row["max_area_fraction"]),
    )


def fine_configs(
    coarse_best: dict[str, Any],
    coarse_min_areas: list[int],
    max_instances: int,
    limit: int,
) -> set[PostprocessConfig]:
    best_threshold = float(coarse_best["threshold"])
    thresholds = sorted(
        {
            round(value, 6)
            for value in (
                best_threshold - 0.025,
                best_threshold,
                best_threshold + 0.025,
            )
            if 0 <= value <= 1
        }
    )
    best_area = int(coarse_best["min_area"])
    index = coarse_min_areas.index(best_area)
    min_areas = coarse_min_areas[max(0, index - 1) : index + 2]
    configs = make_configs(thresholds, min_areas, [0, 1, 2, 3], max_instances)
    if len(configs) > limit:
        raise ValueError(
            f"fine search produced {len(configs)} candidates, exceeding cap {limit}"
        )
    return configs


def repeatability_signature(report: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "hit_pairs",
        "false_positives",
        "false_negatives",
        "hit_iou_sum",
        "official_pq",
        "dice_annotator_macro",
        "iou_annotator_macro",
        "one_to_many_gt_distribution",
        "many_to_one_prediction_distribution",
        "prediction_count_summary",
    ]
    return {key: report[key] for key in keys}


def main() -> None:
    args = parse_args()
    require_legacy_semantic_opt_in(args.allow_legacy_semantic)
    thresholds = parse_values(args.thresholds, float, "thresholds")
    min_areas = parse_values(args.min_areas, int, "min_areas")
    closings = parse_values(args.closing_iterations, int, "closing_iterations")
    if args.max_fine_candidates < 1:
        raise ValueError("max_fine_candidates must be positive")

    annotation_file = (
        args.data_root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    coco = CocoAnnotations(annotation_file)
    all_records = load_fold_manifest(args.fold_manifest, annotation_file)
    records = [row for row in all_records if int(row["fold"]) == args.fold]
    reference = validate_oof_directory(args.oof_dir, records, args.fold)

    coarse_configs = make_configs(
        thresholds, min_areas, closings, args.max_instances
    )
    coarse_configs.add(
        PostprocessConfig(
            threshold=DEFAULT_CONFIG.threshold,
            min_area=DEFAULT_CONFIG.min_area,
            closing_iterations=DEFAULT_CONFIG.closing_iterations,
            max_instances=args.max_instances,
        )
    )
    coarse_rows = search_oof_postprocess(
        args.oof_dir,
        records,
        coco,
        coarse_configs,
        fold=args.fold,
        progress=args.progress,
    )
    for row in coarse_rows:
        row["stage"] = "coarse"

    default_row = next(
        row
        for row in coarse_rows
        if config_from_row(row)
        == PostprocessConfig(max_instances=args.max_instances)
    )
    default_reproduction = {
        "reference_probability_dtype": reference.get("oof_probability_dtype"),
        "reference_official_pq_float32_before_oof_save": reference["official_pq"],
        "rescored_official_pq_float16_oof": default_row["official_pq"],
        "absolute_pq_drift": abs(
            float(reference["official_pq"]) - float(default_row["official_pq"])
        ),
        "tolerance": 1e-5,
        "hit_pairs_match": default_row["hit_pairs"] == reference["hit_pairs"],
        "false_positives_match": (
            default_row["false_positives"] == reference["false_positives"]
        ),
        "false_negatives_match": (
            default_row["false_negatives"] == reference["false_negatives"]
        ),
        "note": (
            "The baseline was scored on float32 probabilities before OOF persistence; "
            "the archived OOF maps are float16. Exact discrete counts plus <=1e-5 PQ "
            "drift is the reproduction gate."
        ),
    }
    default_reproduction["passed"] = bool(
        default_reproduction["hit_pairs_match"]
        and default_reproduction["false_positives_match"]
        and default_reproduction["false_negatives_match"]
        and default_reproduction["absolute_pq_drift"]
        <= default_reproduction["tolerance"]
    )
    if not default_reproduction["passed"]:
        raise RuntimeError(f"default OOF reproduction failed: {default_reproduction}")

    coarse_best = max(coarse_rows, key=row_rank)
    coarse_gain = float(coarse_best["official_pq"]) - float(
        default_row["official_pq"]
    )
    combined_rows = list(coarse_rows)
    fine_search_ran = bool(args.refine and coarse_gain >= args.minimum_refine_gain)
    stop_reason = None
    if fine_search_ran:
        candidates = fine_configs(
            coarse_best,
            min_areas,
            args.max_instances,
            args.max_fine_candidates,
        )
        candidates -= coarse_configs
        if candidates:
            fine_rows = search_oof_postprocess(
                args.oof_dir,
                records,
                coco,
                candidates,
                fold=args.fold,
                progress=args.progress,
            )
            for row in fine_rows:
                row["stage"] = "fine"
            combined_rows.extend(fine_rows)
    elif not args.refine:
        stop_reason = "fine search disabled by command line"
    else:
        stop_reason = (
            f"coarse gain {coarse_gain:.12f} is below the "
            f"{args.minimum_refine_gain:.3f} expansion threshold; retained default config"
        )

    if fine_search_ran:
        selected_row = max(combined_rows, key=row_rank)
    else:
        selected_row = default_row
    selected_config = config_from_row(selected_row)
    detailed_report = evaluate_oof_configuration(
        args.oof_dir,
        records,
        coco,
        selected_config,
        fold=args.fold,
        progress=args.progress,
    )

    repeatability = {
        "enabled": bool(args.repeat_best_check),
        "passed": None,
        "first_signature": repeatability_signature(detailed_report),
    }
    if args.repeat_best_check:
        repeated_report = evaluate_oof_configuration(
            args.oof_dir,
            records,
            coco,
            selected_config,
            fold=args.fold,
            progress=args.progress,
        )
        repeatability["second_signature"] = repeatability_signature(repeated_report)
        repeatability["passed"] = (
            repeatability["first_signature"] == repeatability["second_signature"]
        )
        if not repeatability["passed"]:
            raise RuntimeError("fixed-input repeatability check failed")

    acceptance = {
        "default_reproduction": default_reproduction["passed"],
        "official_pq_at_least_0_095": (
            detailed_report["official_pq"] >= args.minimum_acceptance_pq
        ),
        "true_positives_at_least_235": (
            detailed_report["hit_pairs"] >= args.minimum_acceptance_tp
        ),
        "images_at_safety_cap_is_zero": (
            detailed_report["prediction_count_summary"]["images_at_safety_cap"] == 0
        ),
        "fixed_input_repeatable": repeatability["passed"],
    }
    acceptance["passed"] = all(value is True for value in acceptance.values())

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "candidate_id",
        "threshold",
        "min_area",
        "closing_iterations",
        "max_instances",
        "max_area_fraction",
        "official_pq",
        "hit_pairs",
        "false_positives",
        "false_negatives",
        "hit_iou_sum",
        "prediction_instances",
        "mean_prediction_instances",
        "max_prediction_instances",
        "images_at_safety_cap",
        "physical_images",
        "annotator_records",
        "search_elapsed_seconds",
    ]
    sorted_rows = sorted(combined_rows, key=row_rank, reverse=True)
    atomic_write_csv(output_dir / "grid_results.csv", fieldnames, sorted_rows)
    config_payload = {
        "schema_version": 1,
        "selection_metric": "official_pq_v6",
        "fold": args.fold,
        "selected_config": selected_config.to_dict(),
        "selected_metrics": {
            "official_pq": detailed_report["official_pq"],
            "dice_annotator_macro": detailed_report["dice_annotator_macro"],
            "iou_annotator_macro": detailed_report["iou_annotator_macro"],
            "hit_pairs": detailed_report["hit_pairs"],
            "false_positives": detailed_report["false_positives"],
            "false_negatives": detailed_report["false_negatives"],
        },
    }
    atomic_write_yaml(output_dir / "best_config.yaml", config_payload)
    report_payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "metric": {
            "name": "official_pq_v6",
            "iou_rule": "strictly_greater_than_0.5",
            "annotator_policy": "score every annotator batch; never union or medoid",
        },
        "inputs": {
            "oof_dir": str(args.oof_dir),
            "oof_manifest_sha256": sha256_file(args.oof_dir / "manifest.json"),
            "fold_manifest": str(args.fold_manifest),
            "fold_manifest_sha256": sha256_file(args.fold_manifest),
            "annotation_file": str(annotation_file),
            "annotation_sha256": sha256_file(annotation_file),
        },
        "search": {
            "coarse_candidates": len(coarse_rows),
            "fine_candidates": len(combined_rows) - len(coarse_rows),
            "fine_search_ran": fine_search_ran,
            "max_fine_candidates": args.max_fine_candidates,
            "coarse_best": coarse_best,
            "coarse_gain_over_default": coarse_gain,
            "minimum_gain_to_expand": args.minimum_refine_gain,
            "stop_reason": stop_reason,
            "selected_candidate": selected_row,
        },
        "default_reproduction": default_reproduction,
        "best_report": detailed_report,
        "repeatability": repeatability,
        "acceptance": acceptance,
    }
    atomic_write_json(output_dir / "best_report.json", report_payload)
    print(json.dumps({
        "output_dir": str(output_dir),
        "selected_config": selected_config.to_dict(),
        "official_pq": detailed_report["official_pq"],
        "hit_pairs": detailed_report["hit_pairs"],
        "false_positives": detailed_report["false_positives"],
        "false_negatives": detailed_report["false_negatives"],
        "acceptance": acceptance,
    }, indent=2))


if __name__ == "__main__":
    main()
