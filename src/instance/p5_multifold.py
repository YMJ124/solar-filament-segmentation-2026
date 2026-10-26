"""Leakage-safe aggregation helpers for the formal P5 crop-refiner protocol."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


def refinement_policy_payload(
    config: Mapping[str, Any], policy_name: str
) -> dict[str, Any]:
    """Describe an inference-only refinement policy without changing training."""
    if policy_name == "full_crop":
        return {
            "name": "full_crop",
            "support_preserving": False,
        }
    if policy_name == "boundary_band_preserve_support":
        return {
            "name": policy_name,
            "boundary_kernel": int(config["loss"]["boundary_kernel"]),
            "band_coordinate_space": "resized_refiner_crop",
            "preserve_deep_coarse_interior": True,
            "reserve_other_source_coarse_support": True,
            "per_source_min_area_fallback": int(config["postprocess"]["min_area"]),
            "support_preserving": True,
        }
    raise ValueError(f"unknown refinement policy: {policy_name}")


def refinement_policy_sha256(
    config: Mapping[str, Any], policy_name: str
) -> str:
    encoded = json.dumps(
        refinement_policy_payload(config, policy_name),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def formal_protocol_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields that must be identical across formal held-out folds."""
    experiment = config["experiment"]
    data = config["data"]
    training = config["training"]
    evaluation = config["evaluation"]
    if evaluation.get("scope") != "full_fold":
        raise ValueError("formal P5 protocol requires evaluation.scope=full_fold")
    return {
        "schema_version": int(config["schema_version"]),
        "protocol_name": str(experiment["protocol_name"]),
        "data": {
            "annotation_file": str(data["annotation_file"]),
            "image_dir": str(data["image_dir"]),
            "cache_dirs": [str(path) for path in data["cache_dirs"]],
        },
        "postprocess": dict(config["postprocess"]),
        "crop": dict(config["crop"]),
        "model": dict(config["model"]),
        "loss": dict(config["loss"]),
        "training": {
            key: value for key, value in training.items() if key != "output_dir"
        },
        "evaluation": {
            "scope": str(evaluation["scope"]),
            "mask_threshold": float(evaluation["mask_threshold"]),
            "baseline_multifold_report": str(
                evaluation["baseline_multifold_report"]
            ),
            "multifold_gate": dict(evaluation["multifold_gate"]),
        },
    }


def formal_protocol_sha256(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        formal_protocol_payload(config), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _aggregate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
        "official_pq": hit_iou_sum / denominator if denominator else 0.0,
        "prediction_instances": sum(int(row["prediction_instances"]) for row in rows),
    }


def aggregate_refiner_folds(
    fold_reports: Mapping[int, Mapping[str, Any]],
    baseline_best: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate fixed fold reports and apply the preregistered T1-relative gate."""
    if set(fold_reports) != {0, 1, 2}:
        raise ValueError("formal P5 aggregation requires exactly folds 0, 1 and 2")
    baseline_by_fold = baseline_best["fold_reports"]
    controls = []
    candidates = []
    fold_deltas = {}
    per_image = []
    for fold in sorted(fold_reports):
        report = fold_reports[fold]
        if not report.get("complete") or report.get("status") != "complete_fold":
            raise ValueError(f"fold {fold} report is not a complete formal fold")
        if int(report["heldout_fold"]) != fold:
            raise ValueError(f"fold {fold} report has a different heldout fold")
        control = report["control"]
        candidate = report["candidate"]
        expected = baseline_by_fold[str(fold)]
        if abs(float(control["official_pq"]) - float(expected["official_pq"])) > 1e-12:
            raise ValueError(f"fold {fold} control differs from the T1 baseline")
        controls.append(control)
        candidates.append(candidate)
        fold_deltas[str(fold)] = float(candidate["official_pq"]) - float(
            control["official_pq"]
        )
        per_image.extend(report["per_image"])

    control_pq = np.asarray(
        [float(row["official_pq"]) for row in controls], dtype=np.float64
    )
    candidate_pq = np.asarray(
        [float(row["official_pq"]) for row in candidates], dtype=np.float64
    )
    control = _aggregate_metric_rows(controls)
    candidate = _aggregate_metric_rows(candidates)
    control["fold_mean_pq"] = float(control_pq.mean())
    control["fold_std_pq"] = float(control_pq.std())
    candidate["fold_mean_pq"] = float(candidate_pq.mean())
    candidate["fold_std_pq"] = float(candidate_pq.std())

    image_deltas = []
    for row in per_image:
        image_deltas.append(
            {
                "pq": float(row["candidate"]["official_pq"])
                - float(row["control"]["official_pq"]),
                "instances": int(row["candidate_instances"])
                - int(row["control_instances"]),
                "hits": int(row["candidate"]["hit_pairs"])
                - int(row["control"]["hit_pairs"]),
            }
        )
    tolerance = 1e-12
    risk = {
        "improved_images": sum(row["pq"] > tolerance for row in image_deltas),
        "unchanged_images": sum(abs(row["pq"]) <= tolerance for row in image_deltas),
        "worse_images": sum(row["pq"] < -tolerance for row in image_deltas),
        "images_with_instance_removal": sum(
            row["instances"] < 0 for row in image_deltas
        ),
        "improved_images_with_instance_removal": sum(
            row["instances"] < 0 and row["pq"] > tolerance for row in image_deltas
        ),
        "worse_images_with_instance_removal": sum(
            row["instances"] < 0 and row["pq"] < -tolerance for row in image_deltas
        ),
        "images_losing_hits_after_instance_removal": sum(
            row["instances"] < 0 and row["hits"] < 0 for row in image_deltas
        ),
    }

    mean_delta = candidate["fold_mean_pq"] - control["fold_mean_pq"]
    std_delta = candidate["fold_std_pq"] - control["fold_std_pq"]
    positive_folds = sum(delta > 0 for delta in fold_deltas.values())
    worst_fold_delta = min(fold_deltas.values())
    checks = {
        "minimum_mean_pq_delta": mean_delta
        >= float(gate["minimum_mean_pq_delta"]),
        "minimum_positive_folds": positive_folds
        >= int(gate["minimum_positive_folds"]),
        "minimum_fold_delta": worst_fold_delta
        >= float(gate["minimum_fold_delta"]),
        "maximum_std_increase": std_delta
        <= float(gate["maximum_std_increase"]),
    }
    return {
        "control": control,
        "candidate": candidate,
        "comparison": {
            "fold_pq_delta": fold_deltas,
            "mean_pq_delta": mean_delta,
            "global_official_pq_delta": candidate["official_pq"]
            - control["official_pq"],
            "std_pq_delta": std_delta,
            "positive_folds": positive_folds,
            "worst_fold_delta": worst_fold_delta,
            "hit_pairs_delta": candidate["hit_pairs"] - control["hit_pairs"],
            "false_positives_delta": candidate["false_positives"]
            - control["false_positives"],
            "false_negatives_delta": candidate["false_negatives"]
            - control["false_negatives"],
            "prediction_instances_delta": candidate["prediction_instances"]
            - control["prediction_instances"],
            "gate": dict(gate),
            "checks": checks,
            "pass": all(checks.values()),
        },
        "risk": risk,
    }


def compare_refiner_policy_folds(
    candidate_reports: Mapping[int, Mapping[str, Any]],
    reference_reports: Mapping[int, Mapping[str, Any]],
    baseline_best: Mapping[str, Any],
    gate_vs_t1: Mapping[str, Any],
    gate_vs_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply T1 score gates plus C1-relative and reversibility gates."""
    candidate = aggregate_refiner_folds(
        candidate_reports, baseline_best, gate_vs_t1
    )
    reference = aggregate_refiner_folds(
        reference_reports, baseline_best, gate_vs_t1
    )
    fold_delta = {}
    instance_count_preserved = True
    report_reversibility_valid = True
    for fold in (0, 1, 2):
        candidate_report = candidate_reports[fold]
        reference_report = reference_reports[fold]
        fold_delta[str(fold)] = float(
            candidate_report["candidate"]["official_pq"]
        ) - float(reference_report["candidate"]["official_pq"])
        instance_count_preserved &= int(
            candidate_report["candidate"]["prediction_instances"]
        ) == int(candidate_report["control"]["prediction_instances"])
        reversibility = candidate_report.get("reversibility") or {}
        report_reversibility_valid &= all(
            bool(reversibility.get(key))
            for key in (
                "source_cache_immutable",
                "checkpoint_immutable",
                "branch_output_only",
                "source_instance_count_preserved",
            )
        )

    mean_delta = (
        candidate["candidate"]["fold_mean_pq"]
        - reference["candidate"]["fold_mean_pq"]
    )
    worst_fold_delta = min(fold_delta.values())
    hit_pairs_delta = (
        candidate["candidate"]["hit_pairs"]
        - reference["candidate"]["hit_pairs"]
    )
    structural_checks = {
        "maximum_mean_pq_drop": mean_delta
        >= -float(gate_vs_reference["maximum_mean_pq_drop"]),
        "maximum_fold_pq_drop": worst_fold_delta
        >= -float(gate_vs_reference["maximum_fold_pq_drop"]),
        "minimum_hit_pairs_delta": hit_pairs_delta
        >= int(gate_vs_reference["minimum_hit_pairs_delta"]),
        "source_instance_count_preserved": instance_count_preserved,
        "report_reversibility_valid": report_reversibility_valid,
        "zero_images_with_instance_removal": candidate["risk"][
            "images_with_instance_removal"
        ] == 0,
        "zero_images_losing_hits_after_instance_removal": candidate["risk"][
            "images_losing_hits_after_instance_removal"
        ] == 0,
    }
    passed = bool(
        candidate["comparison"]["pass"] and all(structural_checks.values())
    )
    return {
        "candidate_vs_t1": candidate,
        "reference_c1": reference["candidate"],
        "comparison_vs_c1": {
            "fold_pq_delta": fold_delta,
            "mean_pq_delta": mean_delta,
            "worst_fold_delta": worst_fold_delta,
            "hit_pairs_delta": hit_pairs_delta,
            "false_positives_delta": candidate["candidate"]["false_positives"]
            - reference["candidate"]["false_positives"],
            "false_negatives_delta": candidate["candidate"]["false_negatives"]
            - reference["candidate"]["false_negatives"],
            "prediction_instances_delta": candidate["candidate"][
                "prediction_instances"
            ] - reference["candidate"]["prediction_instances"],
            "gate": dict(gate_vs_reference),
            "checks": structural_checks,
        },
        "pass": passed,
    }
