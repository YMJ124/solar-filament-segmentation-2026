"""Registered-selection and comparison helpers for P9-D2 refinement."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.instance.p9_incremental import AdditiveCandidate


CandidateKey = tuple[int, int]
ImageKey = tuple[int, str]


def aggregate_image_reports(
    rows: Sequence[Mapping[str, Any]], *, prediction_instances: int
) -> dict[str, Any]:
    """Aggregate per-physical-image reports without losing annotator counts."""
    if not rows:
        raise ValueError("image-report aggregation requires at least one row")
    metadata_fields = (
        "metric_name",
        "source_version",
        "source_url",
        "iou_threshold",
        "matching",
    )
    first = rows[0]
    for row in rows[1:]:
        for field in metadata_fields:
            if row[field] != first[field]:
                raise ValueError(f"image reports differ at metric metadata {field}")
    hit_pairs = sum(int(row["hit_pairs"]) for row in rows)
    false_positives = sum(int(row["false_positives"]) for row in rows)
    false_negatives = sum(int(row["false_negatives"]) for row in rows)
    hit_iou_sum = float(sum(float(row["hit_iou_sum"]) for row in rows))
    denominator = hit_pairs + 0.5 * false_positives + 0.5 * false_negatives
    return {
        "physical_images": sum(int(row["physical_images"]) for row in rows),
        "annotator_records": sum(int(row["annotator_records"]) for row in rows),
        "hit_pairs": hit_pairs,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "hit_iou_sum": hit_iou_sum,
        "official_pq": float(hit_iou_sum / denominator) if denominator else 0.0,
        **{field: first[field] for field in metadata_fields},
        "prediction_instances": int(prediction_instances),
    }


def registered_candidate_keys(
    report: Mapping[str, Any],
) -> tuple[dict[ImageKey, set[CandidateKey]], dict[ImageKey, set[CandidateKey]]]:
    """Return all/accepted D1 candidate keys after validating the formal report."""
    if report.get("report_type") != "p9_t1_i1_incremental_oof_v1":
        raise ValueError("D2 requires a P9-D1 OOF report")
    if not bool(report.get("complete")) or not bool(
        report.get("comparison", {}).get("pass")
    ):
        raise ValueError("D2 requires a complete, promoted P9-D1 report")

    all_keys: dict[ImageKey, set[CandidateKey]] = defaultdict(set)
    accepted_keys: dict[ImageKey, set[CandidateKey]] = defaultdict(set)
    row_indices = []
    for row in report.get("candidate_rows", []):
        image_key = (int(row["fold"]), str(row["file_name"]))
        candidate_key = (int(row["i1_source_index"]), int(row["i1_rank"]))
        if candidate_key in all_keys[image_key]:
            raise ValueError(f"duplicate D1 candidate key: {image_key} {candidate_key}")
        all_keys[image_key].add(candidate_key)
        if bool(row["accepted"]):
            accepted_keys[image_key].add(candidate_key)
        row_indices.append(int(row["row_index"]))

    if row_indices != list(range(len(row_indices))):
        raise ValueError("D1 candidate rows are not in registered row-index order")
    expected = int(report["target_summary"]["candidates"])
    if len(row_indices) != expected:
        raise ValueError("D1 candidate row count differs from target summary")
    accepted = sum(len(keys) for keys in accepted_keys.values())
    if accepted != int(report["selection_summary"]["accepted"]):
        raise ValueError("D1 accepted keys differ from selection summary")
    return dict(all_keys), dict(accepted_keys)


def select_registered_additions(
    additions: Sequence[AdditiveCandidate],
    *,
    expected_all: set[CandidateKey],
    expected_accepted: set[CandidateKey],
) -> list[AdditiveCandidate]:
    """Select exactly the registered D1 additions and reject reconstruction drift."""
    by_key: dict[CandidateKey, AdditiveCandidate] = {}
    for candidate in additions:
        key = (int(candidate.source_index), int(candidate.i1_rank))
        if key in by_key:
            raise ValueError(f"reconstructed duplicate candidate key: {key}")
        by_key[key] = candidate
    if set(by_key) != set(expected_all):
        missing = sorted(set(expected_all) - set(by_key))
        extra = sorted(set(by_key) - set(expected_all))
        raise ValueError(
            f"D1 candidate reconstruction drift; missing={missing[:5]} extra={extra[:5]}"
        )
    if not set(expected_accepted) <= set(expected_all):
        raise ValueError("accepted D1 keys are not a subset of all candidate keys")
    return [
        candidate
        for candidate in additions
        if (int(candidate.source_index), int(candidate.i1_rank))
        in expected_accepted
    ]


def compare_refined_folds(
    reference_folds: Mapping[int, Mapping[str, Any]],
    candidate_folds: Mapping[int, Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered D2 gate relative to the promoted P5-C2 parent."""
    folds = sorted(int(fold) for fold in reference_folds)
    if folds != sorted(int(fold) for fold in candidate_folds):
        raise ValueError("reference and candidate fold sets differ")
    reference = np.asarray(
        [float(reference_folds[fold]["official_pq"]) for fold in folds]
    )
    candidate = np.asarray(
        [float(candidate_folds[fold]["official_pq"]) for fold in folds]
    )
    deltas = candidate - reference
    reference_std = float(reference.std())
    candidate_std = float(candidate.std())
    checks = {
        "minimum_mean_fold_pq_delta": float(deltas.mean())
        >= float(gate["minimum_mean_fold_pq_delta"]),
        "minimum_positive_folds": int(np.count_nonzero(deltas > 0))
        >= int(gate["minimum_positive_folds"]),
        "minimum_fold_delta": float(deltas.min())
        >= float(gate["minimum_fold_delta"]),
        "maximum_fold_std_increase": candidate_std - reference_std
        <= float(gate["maximum_fold_std_increase"]),
    }
    return {
        "fold_pq_delta": {
            str(fold): float(delta) for fold, delta in zip(folds, deltas, strict=True)
        },
        "reference_mean_fold_pq": float(reference.mean()),
        "candidate_mean_fold_pq": float(candidate.mean()),
        "mean_fold_pq_delta": float(deltas.mean()),
        "reference_fold_std_pq": reference_std,
        "candidate_fold_std_pq": candidate_std,
        "fold_std_delta": candidate_std - reference_std,
        "positive_folds": int(np.count_nonzero(deltas > 0)),
        "worst_fold_delta": float(deltas.min()),
        "gate": dict(gate),
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def _pq_from_totals(values: np.ndarray) -> float:
    hits, false_positives, false_negatives, hit_iou_sum = values
    denominator = hits + 0.5 * false_positives + 0.5 * false_negatives
    return float(hit_iou_sum / denominator) if denominator else 0.0


def paired_image_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap pooled-PQ deltas with physical images as the resampling unit."""
    if iterations < 1 or not rows:
        raise ValueError("paired bootstrap requires rows and positive iterations")
    fields = ("hit_pairs", "false_positives", "false_negatives", "hit_iou_sum")
    reference = np.asarray(
        [[float(row["reference"][field]) for field in fields] for row in rows],
        dtype=np.float64,
    )
    candidate = np.asarray(
        [[float(row["candidate"][field]) for field in fields] for row in rows],
        dtype=np.float64,
    )
    observed = _pq_from_totals(candidate.sum(axis=0)) - _pq_from_totals(
        reference.sum(axis=0)
    )
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled = rng.integers(0, len(rows), size=len(rows))
        deltas[index] = _pq_from_totals(candidate[sampled].sum(axis=0)) - (
            _pq_from_totals(reference[sampled].sum(axis=0))
        )
    return {
        "unit": "physical_image_with_all_annotator_records",
        "iterations": int(iterations),
        "seed": int(seed),
        "observed_global_pq_delta": float(observed),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "ci95": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "probability_delta_positive": float(np.mean(deltas > 0)),
    }
