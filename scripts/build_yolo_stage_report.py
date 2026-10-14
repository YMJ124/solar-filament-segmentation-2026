#!/usr/bin/env python3
"""Combine panel and multi-fold reports into one deterministic stage decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.instance.yolo_stage_gates import multifold_gate, panel_gate
from src.io_utils import atomic_write_json
from src.instance.yolo_cache import sha256_file


FIXED_PANEL_CONFIG = {
    "confidence": 0.10,
    "min_area": 400,
    "max_instances": 12,
    "force_one": True,
}
FIXED_MULTIFOLD_GRID = {
    "confidences": [0.05, 0.10, 0.15, 0.20, 0.25],
    "min_areas": [200, 400, 600, 800],
    "max_instances": [8, 10, 12, 15, 20],
    "force_one": True,
    "size": 100,
}


def _named_reports(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        if name in result:
            raise ValueError(f"duplicate candidate name: {name}")
        result[name] = Path(raw_path)
    return result


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _validate_panel(report: dict) -> None:
    if int(report.get("grid_size", -1)) != 1:
        raise ValueError("panel report must contain exactly one fixed config")
    actual = {key: report["best"][key] for key in FIXED_PANEL_CONFIG}
    if actual != FIXED_PANEL_CONFIG:
        raise ValueError(f"panel config is not frozen: {actual} != {FIXED_PANEL_CONFIG}")
    if int(report.get("fold", -1)) != 0:
        raise ValueError("fixed panel report must use Fold0")
    experiment = report.get("experiment") or {}
    if not experiment.get("experiment_family") or not experiment.get(
        "training_signature_sha256"
    ):
        raise ValueError("panel report lacks stage experiment provenance")


def _validate_multifold(report: dict, panel_report: dict) -> None:
    if report.get("report_type") != "yolo_multifold_postprocess_v1":
        raise ValueError("invalid multi-fold report type")
    if report.get("folds") != [0, 1, 2]:
        raise ValueError("multi-fold report must contain Fold0/1/2")
    if report.get("grid") != FIXED_MULTIFOLD_GRID:
        raise ValueError("multi-fold report does not use the frozen 100-config grid")
    panel_experiment = panel_report["experiment"]
    if (
        report.get("experiment_family") != panel_experiment["experiment_family"]
        or report.get("training_signature_sha256")
        != panel_experiment["training_signature_sha256"]
    ):
        raise ValueError("panel and multi-fold reports belong to different experiments")


def _source(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _panel_identity(report: dict) -> dict:
    """Keep the hashes and experiment fields needed to audit a stage decision."""
    return {
        "fold": report.get("fold"),
        "experiment": report.get("experiment", {}),
        "cache_manifest_sha256": report.get("cache_manifest_sha256"),
        "cache_run_config_sha256": report.get("cache_run_config_sha256"),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "model_config_sha256": report.get("model_config_sha256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-panel", required=True, type=Path)
    parser.add_argument("--candidate-panel", action="append", default=[])
    parser.add_argument("--baseline-multifold", type=Path)
    parser.add_argument("--candidate-multifold", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    panel_paths = _named_reports(args.candidate_panel)
    multifold_paths = _named_reports(args.candidate_multifold)
    unknown = set(multifold_paths) - set(panel_paths)
    if unknown:
        raise ValueError(f"multi-fold reports have no panel candidate: {sorted(unknown)}")
    baseline_panel = _load(args.baseline_panel)
    _validate_panel(baseline_panel)
    baseline_multifold = (
        _load(args.baseline_multifold) if args.baseline_multifold else None
    )
    if baseline_multifold is not None:
        _validate_multifold(baseline_multifold, baseline_panel)

    candidates = {}
    for name, panel_path in panel_paths.items():
        candidate_panel = _load(panel_path)
        _validate_panel(candidate_panel)
        panel = panel_gate(baseline_panel, candidate_panel)
        row = {
            "experiment_identity": _panel_identity(candidate_panel),
            "panel_gate": panel,
            "panel_report": _source(panel_path),
            "multifold_gate": None,
            "multifold_report": None,
            "status": "rejected_panel" if not panel["passed"] else "pending_multifold",
        }
        if name in multifold_paths:
            if baseline_multifold is None:
                raise ValueError("candidate multi-fold report requires baseline multi-fold report")
            candidate_multifold = _load(multifold_paths[name])
            _validate_multifold(candidate_multifold, candidate_panel)
            multi = multifold_gate(baseline_multifold, candidate_multifold)
            row.update(
                {
                    "multifold_gate": multi,
                    "multifold_report": _source(multifold_paths[name]),
                    "status": (
                        "promoted"
                        if panel["passed"] and multi["passed"]
                        else (
                            "rejected_multifold"
                            if panel["passed"]
                            else "rejected_panel"
                        )
                    ),
                }
            )
        candidates[name] = row

    complete = baseline_multifold is not None and all(
        (not row["panel_gate"]["passed"]) or row["multifold_gate"] is not None
        for row in candidates.values()
    )
    promoted = sorted(
        name for name, row in candidates.items() if row["status"] == "promoted"
    )
    if not complete:
        outcome, next_action = "pending", "complete_required_multifold_runs"
    elif not promoted:
        outcome, next_action = "all_failed", "return_to_target_and_error_review"
    elif len(promoted) == 1:
        outcome, next_action = "one_promoted", "nominate_for_next_long_training_stage"
    else:
        outcome, next_action = (
            "multiple_promoted",
            "test_random_annotator_plus_1280_before_long_training",
        )
    payload = {
        "report_type": "yolo_stage_p1_v1",
        "fixed_panel_config": FIXED_PANEL_CONFIG,
        "fixed_multifold_grid": FIXED_MULTIFOLD_GRID,
        "baseline": {
            "experiment_identity": _panel_identity(baseline_panel),
            "panel_report": _source(args.baseline_panel),
            "multifold_report": (
                _source(args.baseline_multifold) if args.baseline_multifold else None
            ),
        },
        "candidates": candidates,
        "stage_complete": complete,
        "promoted_candidates": promoted,
        "outcome": outcome,
        "next_action": next_action,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
