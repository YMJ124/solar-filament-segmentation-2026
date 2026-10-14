"""Deterministic promotion gates for the three-fold YOLO stage."""

from __future__ import annotations

from typing import Any, Mapping


def panel_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    minimum_delta: float = 0.010,
) -> dict[str, Any]:
    if int(baseline.get("physical_images", -1)) != 32 or int(
        candidate.get("physical_images", -1)
    ) != 32:
        raise ValueError("panel gate requires two 32-image reports")
    baseline_pq = float(baseline["best"]["official_pq"])
    candidate_pq = float(candidate["best"]["official_pq"])
    delta = candidate_pq - baseline_pq
    passed = delta >= minimum_delta
    return {
        "passed": passed,
        "baseline_pq": baseline_pq,
        "candidate_pq": candidate_pq,
        "pq_delta": delta,
        "minimum_delta": minimum_delta,
        "reason": (
            "panel_delta_passed"
            if passed
            else f"panel_delta_{delta:+.6f}_below_{minimum_delta:+.6f}"
        ),
    }


def multifold_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    minimum_mean_delta: float = 0.008,
    minimum_positive_folds: int = 2,
    minimum_fold_delta: float = -0.005,
    maximum_std_increase: float = 0.010,
) -> dict[str, Any]:
    baseline_best = baseline["best"]
    candidate_best = candidate["best"]
    baseline_folds = baseline_best["fold_reports"]
    candidate_folds = candidate_best["fold_reports"]
    if set(baseline_folds) != set(candidate_folds):
        raise ValueError("baseline and candidate multi-fold reports use different folds")
    deltas = {
        fold: float(candidate_folds[fold]["official_pq"])
        - float(baseline_folds[fold]["official_pq"])
        for fold in sorted(baseline_folds, key=int)
    }
    mean_delta = float(candidate_best["fold_mean_pq"]) - float(
        baseline_best["fold_mean_pq"]
    )
    positive_folds = sum(value > 0 for value in deltas.values())
    worst_delta = min(deltas.values())
    std_increase = float(candidate_best["fold_std_pq"]) - float(
        baseline_best["fold_std_pq"]
    )
    criteria = {
        "mean_delta": mean_delta >= minimum_mean_delta,
        "positive_folds": positive_folds >= minimum_positive_folds,
        "worst_fold": worst_delta >= minimum_fold_delta,
        "stability": std_increase <= maximum_std_increase,
    }
    failed = [name for name, passed in criteria.items() if not passed]
    return {
        "passed": not failed,
        "baseline_mean_pq": float(baseline_best["fold_mean_pq"]),
        "candidate_mean_pq": float(candidate_best["fold_mean_pq"]),
        "mean_pq_delta": mean_delta,
        "baseline_std_pq": float(baseline_best["fold_std_pq"]),
        "candidate_std_pq": float(candidate_best["fold_std_pq"]),
        "std_increase": std_increase,
        "fold_pq_deltas": deltas,
        "positive_folds": positive_folds,
        "worst_fold_delta": worst_delta,
        "criteria": criteria,
        "thresholds": {
            "minimum_mean_delta": minimum_mean_delta,
            "minimum_positive_folds": minimum_positive_folds,
            "minimum_fold_delta": minimum_fold_delta,
            "maximum_std_increase": maximum_std_increase,
        },
        "reason": "multifold_gate_passed" if not failed else "failed:" + ",".join(failed),
    }
