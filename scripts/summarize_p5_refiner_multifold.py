#!/usr/bin/env python3
"""Aggregate the three fixed P5-C1 OOF refiner fold reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.instance.p5_multifold import (
    aggregate_refiner_folds,
    formal_protocol_sha256,
)
from src.instance.yolo_cache import sha256_file
from src.io_utils import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-report", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.fold_report) != 3:
        raise ValueError("exactly three --fold-report paths are required")

    reports = {}
    report_rows = []
    protocol_hashes = set()
    baseline_paths = set()
    gates = []
    source_training_signatures = set()
    for path in args.fold_report:
        report = json.loads(path.read_text())
        if report.get("report_type") != "p5_crop_refiner_fold_v1":
            raise ValueError(f"not a formal P5 fold report: {path}")
        fold = int(report["heldout_fold"])
        if fold in reports:
            raise ValueError(f"duplicate formal report for fold {fold}")
        config = report["config"]
        protocol_hash = formal_protocol_sha256(config)
        if report.get("formal_protocol_sha256") != protocol_hash:
            raise ValueError(f"protocol hash mismatch in {path}")
        expected_training = sorted({0, 1, 2} - {fold})
        if sorted(int(value) for value in report["checkpoint_training_folds"]) != expected_training:
            raise ValueError(f"fold {fold} checkpoint violates heldout isolation")
        protocol_hashes.add(protocol_hash)
        baseline_paths.add(str(report["baseline_report"]))
        gates.append(config["evaluation"]["multifold_gate"])
        source_training_signatures.add(report["source_training_signature_sha256"])
        reports[fold] = report
        report_rows.append(
            {
                "fold": fold,
                "path": str(path),
                "sha256": sha256_file(path),
                "config_sha256": report["config_sha256"],
                "checkpoint": report["checkpoint"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "training_folds": report["checkpoint_training_folds"],
                "device": report["device"],
                "cuda_device_name": report["cuda_device_name"],
            }
        )
    if set(reports) != {0, 1, 2}:
        raise ValueError("formal reports must cover folds 0, 1 and 2")
    if len(protocol_hashes) != 1:
        raise ValueError("formal fold reports use different P5 protocols")
    if len(baseline_paths) != 1:
        raise ValueError("formal fold reports use different T1 baselines")
    if any(gate != gates[0] for gate in gates[1:]):
        raise ValueError("formal fold reports use different adoption gates")
    if len(source_training_signatures) != 1:
        raise ValueError("formal fold reports use different T1 experiment signatures")

    baseline_path = Path(next(iter(baseline_paths)))
    baseline = json.loads(baseline_path.read_text())
    result = aggregate_refiner_folds(reports, baseline["best"], gates[0])
    payload = {
        "report_type": "p5_crop_refiner_multifold_v1",
        "status": "promoted" if result["comparison"]["pass"] else "failed",
        "formal_protocol_sha256": next(iter(protocol_hashes)),
        "source_training_signature_sha256": next(
            iter(source_training_signatures)
        ),
        "baseline_report": str(baseline_path),
        "baseline_report_sha256": sha256_file(baseline_path),
        "fold_reports": sorted(report_rows, key=lambda row: row["fold"]),
        **result,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["comparison"], indent=2), flush=True)
    print(f"wrote P5-C1 multifold report to {args.output}", flush=True)


if __name__ == "__main__":
    main()
