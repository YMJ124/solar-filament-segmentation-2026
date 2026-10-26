"""Leakage-safe final-selector and deployment helpers for P9."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from src.instance.p9_incremental import AdditiveCandidate, FEATURE_NAMES


def selector_training_matrix(
    report: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover the registered D1 feature matrix and OOF-derived targets."""
    if report.get("report_type") != "p9_t1_i1_incremental_oof_v1":
        raise ValueError("final selector requires a P9-D1 OOF report")
    if not bool(report.get("complete")) or not bool(
        report.get("comparison", {}).get("pass")
    ):
        raise ValueError("final selector requires a promoted complete P9-D1 report")
    if tuple(report.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("P9-D1 feature schema differs from deployment schema")

    rows = report.get("candidate_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("P9-D1 report has no candidate rows")
    features: list[list[float]] = []
    targets: list[float] = []
    folds: list[int] = []
    for expected_index, row in enumerate(rows):
        if int(row.get("row_index", -1)) != expected_index:
            raise ValueError("P9-D1 candidate rows are not in registered order")
        feature_map = row.get("features")
        if not isinstance(feature_map, Mapping) or set(feature_map) != set(FEATURE_NAMES):
            raise ValueError("P9-D1 candidate row has an invalid feature schema")
        target = row.get("target", {}).get("marginal_utility")
        if target is None:
            raise ValueError("P9-D1 candidate row lacks the registered target")
        features.append([float(feature_map[name]) for name in FEATURE_NAMES])
        targets.append(float(target))
        folds.append(int(row["fold"]))
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    row_folds = np.asarray(folds, dtype=np.int64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("P9-D1 selector training values are not finite")
    if sorted(set(row_folds.tolist())) != [0, 1, 2]:
        raise ValueError("final selector requires all three registered OOF folds")
    return x, y, row_folds


def fit_final_selector(
    x: np.ndarray,
    y: np.ndarray,
    model_config: Mapping[str, Any],
    *,
    seed: int,
) -> ExtraTreesRegressor:
    """Fit the fixed D1 regressor on all cross-fitted candidate/utility rows."""
    if str(model_config.get("type")) != "extra_trees_regressor":
        raise ValueError("unsupported P9 final selector type")
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or len(x) != len(y):
        raise ValueError("invalid P9 final selector training matrix")
    model = ExtraTreesRegressor(
        n_estimators=int(model_config["n_estimators"]),
        min_samples_leaf=int(model_config["min_samples_leaf"]),
        max_features=float(model_config["max_features"]),
        random_state=int(seed),
        n_jobs=1,
    )
    model.fit(x, y)
    return model


def selector_matrix_sha256(
    x: np.ndarray, y: np.ndarray, row_folds: np.ndarray
) -> str:
    """Hash the ordered final-selector training payload."""
    digest = hashlib.sha256()
    digest.update(json.dumps(FEATURE_NAMES, separators=(",", ":")).encode())
    for values in (x, y, row_folds):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def selected_additions_from_features(
    model: Any,
    additions: Sequence[AdditiveCandidate],
    feature_rows: Sequence[np.ndarray],
    *,
    threshold: float,
) -> tuple[list[AdditiveCandidate], np.ndarray]:
    """Apply the frozen strict-greater utility rule in candidate order."""
    if len(additions) != len(feature_rows):
        raise ValueError("P9 additions and feature rows must align")
    if not additions:
        return [], np.empty(0, dtype=np.float64)
    matrix = np.asarray(feature_rows, dtype=np.float64)
    if matrix.shape != (len(additions), len(FEATURE_NAMES)):
        raise ValueError("P9 deployment feature matrix has the wrong shape")
    if not np.isfinite(matrix).all():
        raise ValueError("P9 deployment features are not finite")
    predictions = np.asarray(model.predict(matrix), dtype=np.float64)
    if predictions.shape != (len(additions),) or not np.isfinite(predictions).all():
        raise ValueError("P9 deployment selector returned invalid predictions")
    selected = [
        candidate
        for candidate, prediction in zip(additions, predictions, strict=True)
        if float(prediction) > float(threshold)
    ]
    return selected, predictions


def deployment_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a stable semantic fingerprint for resumable per-image caches."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
