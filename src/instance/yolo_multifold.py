"""Fold-level aggregation for shared YOLO post-processing grids."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


CONFIG_FIELDS = ("confidence", "min_area", "max_instances", "force_one")


def validate_multifold_manifests(
    manifests: Mapping[int, Mapping[str, Any]], expected_folds: set[int]
) -> dict[str, Any]:
    """Reject incomplete folds or caches mixed across experiments/provenance."""
    actual_folds = {int(fold) for fold in manifests}
    if actual_folds != expected_folds:
        raise ValueError(
            f"cache folds {sorted(actual_folds)} do not match expected "
            f"{sorted(expected_folds)}"
        )
    inference_signatures = set()
    dataset_signatures = set()
    experiment_signatures = set()
    experiment_families = set()
    for manifest in manifests.values():
        inference = manifest["inference"]
        inference_signatures.add(
            (
                int(inference["imgsz"]),
                int(inference["quantize"]),
                bool(inference["retina_masks"]),
                float(inference["candidate_confidence"]),
                int(inference["max_candidates"]),
            )
        )
        dataset = manifest["dataset"]
        dataset_signatures.add(
            (dataset["annotation_sha256"], dataset["fold_manifest_sha256"])
        )
        experiment = manifest.get("experiment") or {}
        signature = experiment.get("training_signature_sha256")
        family = experiment.get("experiment_family")
        if not signature or not family:
            raise ValueError(
                "cache lacks stage experiment identity; regenerate it with the current "
                "cache_yolo_candidates implementation and its training args.yaml"
            )
        experiment_signatures.add(signature)
        experiment_families.add(family)
    if len(inference_signatures) != 1:
        raise ValueError("all fold caches must use the same inference configuration")
    if len(dataset_signatures) != 1:
        raise ValueError("all fold caches must use the same annotation and fold manifest")
    if len(experiment_signatures) != 1 or len(experiment_families) != 1:
        raise ValueError("fold caches mix different model/training experiment families")
    return {
        "inference_signature": list(next(iter(inference_signatures))),
        "experiment_family": next(iter(experiment_families)),
        "training_signature_sha256": next(iter(experiment_signatures)),
    }


def config_key(row: Mapping[str, Any]) -> tuple[float, int, int, bool]:
    return (
        float(row["confidence"]),
        int(row["min_area"]),
        int(row["max_instances"]),
        bool(row["force_one"]),
    )


def aggregate_multifold_candidates(
    fold_candidates: Mapping[int, list[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Combine the same grid across folds while preserving each fold report."""
    if len(fold_candidates) < 2:
        raise ValueError("multi-fold evaluation requires at least two folds")
    indexed = {
        int(fold): {config_key(row): dict(row) for row in rows}
        for fold, rows in fold_candidates.items()
    }
    expected: set[tuple[float, int, int, bool]] | None = None
    for fold, rows in indexed.items():
        keys = set(rows)
        if len(keys) != len(fold_candidates[fold]):
            raise ValueError(f"fold {fold} contains duplicate post-process configs")
        if expected is None:
            expected = keys
        elif keys != expected:
            raise ValueError(f"fold {fold} does not contain the shared config grid")
    if not expected:
        raise ValueError("multi-fold post-process grid is empty")

    results = []
    for key in sorted(expected):
        confidence, min_area, max_instances, force_one = key
        reports = {fold: rows[key] for fold, rows in indexed.items()}
        fold_pq = np.asarray(
            [reports[fold]["official_pq"] for fold in sorted(reports)],
            dtype=np.float64,
        )
        hit_pairs = sum(int(row["hit_pairs"]) for row in reports.values())
        false_positives = sum(
            int(row["false_positives"]) for row in reports.values()
        )
        false_negatives = sum(
            int(row["false_negatives"]) for row in reports.values()
        )
        hit_iou_sum = float(sum(row["hit_iou_sum"] for row in reports.values()))
        denominator = hit_pairs + 0.5 * false_positives + 0.5 * false_negatives
        results.append(
            {
                "confidence": confidence,
                "min_area": min_area,
                "max_instances": max_instances,
                "force_one": force_one,
                "fold_mean_pq": float(fold_pq.mean()),
                "fold_std_pq": float(fold_pq.std()),
                "global_official_pq": (
                    hit_iou_sum / denominator if denominator else 0.0
                ),
                "physical_images": sum(
                    int(row["physical_images"]) for row in reports.values()
                ),
                "annotator_records": sum(
                    int(row["annotator_records"]) for row in reports.values()
                ),
                "hit_pairs": hit_pairs,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "hit_iou_sum": hit_iou_sum,
                "prediction_instances": sum(
                    int(row["prediction_instances"]) for row in reports.values()
                ),
                "fold_reports": {
                    str(fold): reports[fold] for fold in sorted(reports)
                },
            }
        )
    results.sort(
        key=lambda row: (
            row["fold_mean_pq"],
            -row["fold_std_pq"],
            -row["false_positives"],
            row["hit_pairs"],
        ),
        reverse=True,
    )
    return results
