"""Validated cross-fold summaries for cached YOLO error reports."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _derived_rates(aggregate: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    prediction_annotator_opportunities = sum(
        int(row["prediction_instances"]) * int(row["annotator_batches"])
        for row in rows
    )
    multi_annotator_rows = [row for row in rows if int(row["annotator_batches"]) > 1]
    unstable_rows = [
        row for row in multi_annotator_rows if float(row["annotator_pq_range"]) >= 0.1
    ]
    return {
        "false_positives_per_physical_image": _safe_ratio(
            aggregate["false_positives"], aggregate["physical_images"]
        ),
        "false_positives_per_annotator_record": _safe_ratio(
            aggregate["false_positives"], aggregate["annotator_records"]
        ),
        "false_negatives_per_annotator_record": _safe_ratio(
            aggregate["false_negatives"], aggregate["annotator_records"]
        ),
        "small_miss_share_of_false_negatives": _safe_ratio(
            aggregate["small_instance_misses"], aggregate["false_negatives"]
        ),
        "large_fp_share_of_consensus_false_positives": _safe_ratio(
            aggregate["large_false_positive_instances"],
            aggregate["consensus_false_positive_instances"],
        ),
        "fragmented_share_of_truth_instances": _safe_ratio(
            aggregate["fragmented_gt_instances"], aggregate["truth_instances_total"]
        ),
        "merged_share_of_prediction_annotator_opportunities": _safe_ratio(
            aggregate["merged_prediction_instances"], prediction_annotator_opportunities
        ),
        "multi_annotator_image_share": _safe_ratio(
            len(multi_annotator_rows), aggregate["physical_images"]
        ),
        "unstable_multi_annotator_image_share": _safe_ratio(
            len(unstable_rows), len(multi_annotator_rows)
        ),
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hit_pairs = sum(int(row["hit_pairs"]) for row in rows)
    false_positives = sum(int(row["false_positives"]) for row in rows)
    false_negatives = sum(int(row["false_negatives"]) for row in rows)
    hit_iou_sum = float(sum(float(row["hit_iou_sum"]) for row in rows))
    denominator = hit_pairs + 0.5 * false_positives + 0.5 * false_negatives
    return {
        "physical_images": len(rows),
        "annotator_records": sum(int(row["annotator_batches"]) for row in rows),
        "official_pq": hit_iou_sum / denominator if denominator else 0.0,
        "hit_pairs": hit_pairs,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "hit_iou_sum": hit_iou_sum,
        "prediction_instances": sum(int(row["prediction_instances"]) for row in rows),
        "truth_instances_total": sum(int(row["truth_instances_total"]) for row in rows),
        "consensus_false_positive_instances": sum(
            int(row["consensus_false_positive_instances"]) for row in rows
        ),
        "large_false_positive_instances": sum(
            int(row["large_false_positive_instances"]) for row in rows
        ),
        "small_instance_misses": sum(int(row["small_instance_misses"]) for row in rows),
        "fragmented_gt_instances": sum(
            int(row["fragmented_gt_instances"]) for row in rows
        ),
        "merged_prediction_instances": sum(
            int(row["merged_prediction_instances"]) for row in rows
        ),
        "mean_annotator_pq_std": float(
            np.mean([float(row["annotator_pq_std"]) for row in rows])
        ),
        "max_annotator_pq_range": max(float(row["annotator_pq_range"]) for row in rows),
    }


def _selection(rows: Sequence[Mapping[str, Any]], *, best: bool, limit: int = 15) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            (-1 if best else 1) * float(row["official_pq"]),
            int(row["false_negatives"]),
            int(row["false_positives"]),
            str(row["file_name"]),
        ),
    )
    return [
        {
            "file_name": str(row["file_name"]),
            "fold": int(row["fold"]),
            "official_pq": float(row["official_pq"]),
            "hit_pairs": int(row["hit_pairs"]),
            "false_positives": int(row["false_positives"]),
            "false_negatives": int(row["false_negatives"]),
        }
        for row in ordered[:limit]
    ]


def summarize_multifold_error_reports(
    reports: Sequence[Mapping[str, Any]], *, expected_folds: set[int] = {0, 1, 2}
) -> dict[str, Any]:
    """Combine compatible single-fold reports and derive P2 decision evidence."""
    if not reports:
        raise ValueError("no error reports supplied")
    folds = [int(report["fold"]) for report in reports]
    if len(folds) != len(set(folds)):
        raise ValueError("error reports contain duplicate folds")
    if set(folds) != expected_folds:
        raise ValueError(f"error reports use folds {sorted(folds)}, expected {sorted(expected_folds)}")

    reference = reports[0]
    for report in reports[1:]:
        if report.get("postprocess") != reference.get("postprocess"):
            raise ValueError("error reports use different post-processing configurations")
        if report.get("diagnostics") != reference.get("diagnostics"):
            raise ValueError("error reports use different diagnostic thresholds")
        for key in ("annotation_sha256", "fold_manifest_sha256", "training_signature_sha256"):
            if report.get(key) != reference.get(key):
                raise ValueError(f"error reports mix incompatible {key} values")

    rows = [dict(row) for report in reports for row in report["per_image"]]
    names = [str(row["file_name"]) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("physical images occur in more than one fold report")

    fold_summaries: dict[str, dict[str, Any]] = {}
    fold_pq = []
    for report in sorted(reports, key=lambda item: int(item["fold"])):
        fold = int(report["fold"])
        fold_rows = [dict(row) for row in report["per_image"]]
        aggregate = _aggregate_rows(fold_rows)
        fold_pq.append(aggregate["official_pq"])
        fold_summaries[str(fold)] = {
            **aggregate,
            "rates": _derived_rates(aggregate, fold_rows),
            "cache_dir": report.get("cache_dir"),
            "cache_run_config_sha256": report.get("cache_run_config_sha256"),
            "checkpoint_sha256": report.get("checkpoint_sha256"),
        }

    overall = _aggregate_rows(rows)
    rates = _derived_rates(overall, rows)
    category_counts = Counter(category for row in rows for category in row["categories"])
    fold2 = fold_summaries["2"]
    comparison_folds = [fold_summaries["0"], fold_summaries["1"]]
    comparison_fp_rate = float(
        np.mean([row["rates"]["false_positives_per_annotator_record"] for row in comparison_folds])
    )
    fold2_large_density = _safe_ratio(
        fold2["large_false_positive_instances"], fold2["physical_images"]
    )
    comparison_large_density = float(
        np.mean(
            [
                _safe_ratio(row["large_false_positive_instances"], row["physical_images"])
                for row in comparison_folds
            ]
        )
    )
    fold2_diagnostic = {
        "fp_per_annotator_record": fold2["rates"]["false_positives_per_annotator_record"],
        "fold01_mean_fp_per_annotator_record": comparison_fp_rate,
        "fp_rate_excess": fold2["rates"]["false_positives_per_annotator_record"] - comparison_fp_rate,
        "prediction_instances_per_image": _safe_ratio(
            fold2["prediction_instances"], fold2["physical_images"]
        ),
        "fold01_mean_prediction_instances_per_image": float(
            np.mean(
                [
                    _safe_ratio(row["prediction_instances"], row["physical_images"])
                    for row in comparison_folds
                ]
            )
        ),
        "large_consensus_fp_per_image": fold2_large_density,
        "fold01_mean_large_consensus_fp_per_image": comparison_large_density,
        "fold2_large_consensus_fp_count": fold2["large_false_positive_instances"],
        "global_large_fp_share_of_consensus_false_positives": rates[
            "large_fp_share_of_consensus_false_positives"
        ],
        "large_fp_explains_excess": (
            fold2_large_density > comparison_large_density * 1.2
            and fold2["large_false_positive_instances"] >= 10
            and rates["large_fp_share_of_consensus_false_positives"] >= 0.10
        ),
    }

    small_is_primary = rates["small_miss_share_of_false_negatives"] >= 0.30
    priorities = [
        {
            "rank": 1,
            "candidate": "expanded_multi_annotator_equal_physical_weight",
            "recommended": True,
            "reason": (
                "uses all available target variants without union while preserving equal "
                "total weight per physical image"
            ),
            "evidence": {
                "multi_annotator_image_share": rates["multi_annotator_image_share"],
                "unstable_multi_annotator_image_share": rates[
                    "unstable_multi_annotator_image_share"
                ],
                "mean_annotator_pq_std": overall["mean_annotator_pq_std"],
            },
        },
        {
            "rank": 2,
            "candidate": "small_instance_targeted_training",
            "recommended": small_is_primary,
            "reason": (
                "small-instance misses meet the predeclared >=30% share-of-FN threshold"
                if small_is_primary
                else "small-instance misses do not meet the predeclared >=30% share-of-FN threshold"
            ),
            "evidence": {
                "small_instance_misses": overall["small_instance_misses"],
                "false_negatives": overall["false_negatives"],
                "small_miss_share_of_false_negatives": rates[
                    "small_miss_share_of_false_negatives"
                ],
                "decision_threshold": 0.30,
            },
        },
        {
            "rank": 3,
            "candidate": "fold2_large_fp_filtering",
            "recommended": bool(fold2_diagnostic["large_fp_explains_excess"]),
            "reason": (
                "Fold2 has >=10 large FPs, >=10% global large-FP share, and >20% excess density"
                if fold2_diagnostic["large_fp_explains_excess"]
                else "large FPs lack the count and share needed to explain Fold2 excess FP"
            ),
            "evidence": fold2_diagnostic,
        },
    ]

    return {
        "report_type": "yolo_multifold_error_analysis_v1",
        "folds": sorted(folds),
        "postprocess": reference["postprocess"],
        "diagnostics": reference["diagnostics"],
        "annotation_sha256": reference.get("annotation_sha256"),
        "fold_manifest_sha256": reference.get("fold_manifest_sha256"),
        "training_signature_sha256": reference.get("training_signature_sha256"),
        "fold_summaries": fold_summaries,
        "fold_mean_pq": float(np.mean(fold_pq)),
        "fold_std_pq": float(np.std(fold_pq)),
        "global": {**overall, "rates": rates},
        "category_image_counts_nonexclusive": dict(sorted(category_counts.items())),
        "fold2_diagnostic": fold2_diagnostic,
        "candidate_priorities": priorities,
        "worst_images": _selection(rows, best=False),
        "best_images": _selection(rows, best=True),
        "per_image": rows,
    }
