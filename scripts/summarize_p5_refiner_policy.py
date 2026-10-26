#!/usr/bin/env python3
"""Compare one reversible P5 refiner policy with fixed T1 and formal C1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.instance.p5_multifold import (
    compare_refiner_policy_folds,
    formal_protocol_sha256,
    refinement_policy_payload,
    refinement_policy_sha256,
)
from src.instance.yolo_cache import sha256_file
from src.io_utils import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/instance/stage_p5/p5_c2_boundary_policy.yaml"
        ),
    )
    args = parser.parse_args()
    policy_config = yaml.safe_load(args.config.read_text())
    if int(policy_config.get("schema_version", 0)) != 1:
        raise ValueError("P5 policy config must use schema_version=1")
    policy_name = str(policy_config["refinement_policy"]["name"])
    parent_protocol = str(
        policy_config["experiment"]["parent_protocol_sha256"]
    )
    candidate_reports = {}
    reference_reports = {}
    candidate_rows = []
    reference_rows = []
    baseline_paths = set()
    expected_policy_hash = None
    for fold in (0, 1, 2):
        row = policy_config["folds"][str(fold)]
        training_config_path = Path(row["training_config"])
        training_config = yaml.safe_load(training_config_path.read_text())
        if formal_protocol_sha256(training_config) != parent_protocol:
            raise ValueError(f"fold {fold} parent protocol differs")
        expected_payload = refinement_policy_payload(training_config, policy_name)
        if int(expected_payload["boundary_kernel"]) != int(
            policy_config["refinement_policy"]["expected_boundary_kernel"]
        ):
            raise ValueError("boundary kernel differs from preregistration")
        if int(expected_payload["per_source_min_area_fallback"]) != int(
            policy_config["refinement_policy"]["per_source_min_area_fallback"]
        ):
            raise ValueError("min-area fallback differs from preregistration")
        policy_hash = refinement_policy_sha256(training_config, policy_name)
        if expected_policy_hash is None:
            expected_policy_hash = policy_hash
        elif policy_hash != expected_policy_hash:
            raise ValueError("fold policy hashes differ")

        candidate_path = Path(row["candidate_report"])
        reference_path = Path(row["reference_report"])
        candidate = json.loads(candidate_path.read_text())
        reference = json.loads(reference_path.read_text())
        for name, report in (("candidate", candidate), ("reference", reference)):
            if report.get("report_type") != "p5_crop_refiner_fold_v1":
                raise ValueError(f"fold {fold} {name} is not a formal report")
            if not report.get("complete") or report.get("status") != "complete_fold":
                raise ValueError(f"fold {fold} {name} report is incomplete")
            if int(report["heldout_fold"]) != fold:
                raise ValueError(f"fold {fold} {name} heldout mismatch")
            if report.get("formal_protocol_sha256") != parent_protocol:
                raise ValueError(f"fold {fold} {name} protocol mismatch")
        if candidate.get("refinement_policy") != expected_payload:
            raise ValueError(f"fold {fold} candidate policy payload mismatch")
        if candidate.get("refinement_policy_sha256") != policy_hash:
            raise ValueError(f"fold {fold} candidate policy hash mismatch")
        if candidate["checkpoint_sha256"] != reference["checkpoint_sha256"]:
            raise ValueError(f"fold {fold} candidate changed checkpoint")
        if candidate["control"] != reference["control"]:
            raise ValueError(f"fold {fold} candidate changed T1 control")
        baseline_paths.add(str(candidate["baseline_report"]))
        candidate_reports[fold] = candidate
        reference_reports[fold] = reference
        candidate_rows.append(
            {"fold": fold, "path": str(candidate_path), "sha256": sha256_file(candidate_path)}
        )
        reference_rows.append(
            {"fold": fold, "path": str(reference_path), "sha256": sha256_file(reference_path)}
        )
    if len(baseline_paths) != 1:
        raise ValueError("policy reports use different T1 baselines")
    baseline_path = Path(next(iter(baseline_paths)))
    baseline = json.loads(baseline_path.read_text())
    result = compare_refiner_policy_folds(
        candidate_reports,
        reference_reports,
        baseline["best"],
        policy_config["gate_vs_t1"],
        policy_config["gate_vs_c1"],
    )
    payload = {
        "report_type": "p5_refiner_policy_multifold_v1",
        "status": "promoted" if result["pass"] else "not_promoted",
        "policy_config": str(args.config),
        "policy_config_sha256": sha256_file(args.config),
        "parent_protocol_sha256": parent_protocol,
        "refinement_policy": policy_config["refinement_policy"],
        "refinement_policy_sha256": expected_policy_hash,
        "baseline_report": str(baseline_path),
        "baseline_report_sha256": sha256_file(baseline_path),
        "candidate_fold_reports": candidate_rows,
        "reference_c1_fold_reports": reference_rows,
        **result,
    }
    output = Path(policy_config["output"])
    atomic_write_json(output, payload)
    print(json.dumps({
        "candidate_vs_t1": payload["candidate_vs_t1"]["comparison"],
        "comparison_vs_c1": payload["comparison_vs_c1"],
        "pass": payload["pass"],
    }, indent=2), flush=True)
    print(f"wrote P5 policy report to {output}", flush=True)


if __name__ == "__main__":
    main()
