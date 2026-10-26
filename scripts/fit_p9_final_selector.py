#!/usr/bin/env python3
"""Fit the frozen P9 selector on all registered OOF candidate-utility rows."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import sklearn
import yaml

from src.instance.p9_deployment import (
    fit_final_selector,
    selector_matrix_sha256,
    selector_training_matrix,
)
from src.instance.p9_incremental import FEATURE_NAMES
from src.instance.yolo_cache import sha256_file
from src.io_utils import atomic_write_json


def _verify(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise ValueError(f"{label} SHA256 mismatch: {observed} != {expected}")


def _atomic_joblib_dump(model, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        joblib.dump(model, temporary, compress=3)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/instance/stage_p9/p9_d3_final_selector.yaml"),
    )
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != 1:
        raise ValueError("P9-D3 selector config must use schema_version=1")
    if not bool(config["training"].get("use_all_oof_candidate_rows")):
        raise ValueError("P9-D3 must use all registered OOF candidate rows")
    if bool(config["training"].get("permit_test_features_or_labels")):
        raise ValueError("P9-D3 selector training cannot consume test data")

    parents = config["parents"]
    d1_path = Path(parents["p9_d1_report"])
    d2_path = Path(parents["p9_d2_report"])
    _verify(d1_path, parents["p9_d1_report_sha256"], "P9-D1 report")
    _verify(d2_path, parents["p9_d2_report_sha256"], "P9-D2 report")
    d1 = json.loads(d1_path.read_text())
    d2 = json.loads(d2_path.read_text())
    if d2.get("report_type") != "p9_selected_additions_c2_oof_v1":
        raise ValueError("P9-D3 requires the P9-D2 OOF report")
    if not bool(d2.get("complete")) or d2.get("status") != "promoted" or not bool(
        d2.get("comparison_vs_p5_c2", {}).get("pass")
    ):
        raise ValueError("P9-D3 requires a promoted complete P9-D2 report")
    if d2.get("parents", {}).get("p9_d1_report_sha256") != sha256_file(d1_path):
        raise ValueError("P9-D2 does not descend from the configured P9-D1 report")

    started = perf_counter()
    x, y, folds = selector_training_matrix(d1)
    model = fit_final_selector(
        x,
        y,
        config["model"],
        seed=int(config["experiment"]["seed"]),
    )
    predictions = np.asarray(model.predict(x), dtype=np.float64)
    threshold = float(config["model"]["threshold"])
    model_output = args.model_output or Path(config["outputs"]["model"])
    report_output = args.report_output or Path(config["outputs"]["report"])
    _atomic_joblib_dump(model, model_output)
    report = {
        "report_type": "p9_final_selector_v1",
        "status": "ready_for_test_inference",
        "complete": True,
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "parents": {
            "p9_d1_report": str(d1_path),
            "p9_d1_report_sha256": sha256_file(d1_path),
            "p9_d2_report": str(d2_path),
            "p9_d2_report_sha256": sha256_file(d2_path),
        },
        "feature_names": list(FEATURE_NAMES),
        "model_config": config["model"],
        "training": {
            "candidate_rows": int(len(x)),
            "folds": sorted(set(folds.tolist())),
            "candidates_by_fold": {
                str(fold): int(np.count_nonzero(folds == fold))
                for fold in sorted(set(folds.tolist()))
            },
            "positive_targets": int(np.count_nonzero(y > 0)),
            "nonpositive_targets": int(np.count_nonzero(y <= 0)),
            "matrix_sha256": selector_matrix_sha256(x, y, folds),
            "source": "OOF detector candidates and OOF ground-truth utility only",
            "test_data_used": False,
        },
        "training_fit_diagnostic_not_an_oof_estimate": {
            "selected": int(np.count_nonzero(predictions > threshold)),
            "rejected": int(np.count_nonzero(predictions <= threshold)),
            "prediction_min": float(predictions.min()),
            "prediction_median": float(np.median(predictions)),
            "prediction_max": float(predictions.max()),
        },
        "model": str(model_output),
        "model_sha256": sha256_file(model_output),
        "runtime": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "elapsed_seconds": perf_counter() - started,
        "next_action": "run_p9_d3_test_inference_once_without_threshold_tuning",
    }
    atomic_write_json(report_output, report)
    print(f"wrote P9-D3 selector to {model_output}", flush=True)
    print(f"selector SHA256: {report['model_sha256']}", flush=True)
    print(f"wrote selector report to {report_output}", flush=True)
    print(
        f"training diagnostic selected {report['training_fit_diagnostic_not_an_oof_estimate']['selected']}"
        f"/{len(x)} at frozen threshold {threshold}",
        flush=True,
    )


if __name__ == "__main__":
    main()
