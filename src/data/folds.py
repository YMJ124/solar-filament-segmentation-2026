"""Persistent date-grouped folds for physical solar observations."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

MANIFEST_SCHEMA_VERSION = 2
TARGET_POLICY = "medoid_annotator_v1"
GROUP_STRATEGY = "acquisition_date_yyyymmdd"


def observation_date(file_name: str) -> str:
    """Extract the YYYYMMDD acquisition date encoded in a MAGFiLO filename."""
    token = Path(file_name).stem[:8]
    if len(token) != 8 or not token.isdigit():
        raise ValueError(f"cannot extract YYYYMMDD date from {file_name!r}")
    return token


def _station(file_name: str) -> str:
    return Path(file_name).stem[-2:]


def _feature_weight(name: str) -> float:
    if name in {"physical_images", "annotations"}:
        return 4.0
    if name == "annotator_batches":
        return 2.0
    return 1.0


def assign_date_grouped_folds(
    records: Iterable[dict], n_splits: int = 5, seed: int = 2026
) -> list[dict]:
    """Assign whole acquisition dates while balancing load and acquisition mix."""
    rows = [dict(row) for row in records]
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    date_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        date_rows[observation_date(str(row["file_name"]))].append(row)
    if len(date_rows) < n_splits:
        raise ValueError("number of date groups is smaller than n_splits")

    group_features: dict[str, dict[str, float]] = {}
    for date, members in date_rows.items():
        features: dict[str, float] = defaultdict(float)
        for row in members:
            file_name = str(row["file_name"])
            features["physical_images"] += 1
            features["annotator_batches"] += int(row.get("n_batches", 1))
            features["annotations"] += int(row.get("n_annotations", 0))
            features[f"year:{date[:4]}"] += 1
            features[f"station:{_station(file_name)}"] += 1
        group_features[date] = dict(features)

    totals: dict[str, float] = defaultdict(float)
    for features in group_features.values():
        for key, value in features.items():
            totals[key] += value
    targets = {key: value / n_splits for key, value in totals.items()}
    fold_features: list[dict[str, float]] = [defaultdict(float) for _ in range(n_splits)]

    rng = random.Random(seed)
    dates = list(date_rows)
    rng.shuffle(dates)
    dates.sort(
        key=lambda date: (
            group_features[date].get("annotations", 0),
            group_features[date].get("annotator_batches", 0),
            group_features[date].get("physical_images", 0),
        ),
        reverse=True,
    )

    date_to_fold: dict[str, int] = {}
    for date in dates:
        features = group_features[date]

        def candidate_score(fold: int) -> tuple[float, float, int]:
            score = 0.0
            for key, target in targets.items():
                if target <= 0:
                    continue
                candidate = fold_features[fold].get(key, 0.0) + features.get(key, 0.0)
                score += _feature_weight(key) * (candidate / target) ** 2
            return score, fold_features[fold].get("physical_images", 0.0), fold

        selected_fold = min(range(n_splits), key=candidate_score)
        date_to_fold[date] = selected_fold
        for key, value in features.items():
            fold_features[selected_fold][key] += value

    for row in rows:
        date = observation_date(str(row["file_name"]))
        row["fold"] = date_to_fold[date]
        row["group"] = date
    return rows


def load_fold_manifest(
    path: str | Path, annotation_file: str | Path | None = None
) -> list[dict]:
    """Load the canonical persisted fold/target manifest with typed fields."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"fold manifest not found: {path}; run scripts.build_folds")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"fold manifest is empty: {path}")
    required = {
        "file_name", "fold", "group", "width", "height", "n_batches",
        "n_annotations", "selected_image_id", "selected_annotations",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"fold manifest is missing columns: {sorted(missing)}")
    int_fields = {
        "fold",
        "width",
        "height",
        "n_batches",
        "n_annotations",
        "selected_annotations",
    }
    float_fields = {"medoid_mean_iou"}
    for row in rows:
        for key in int_fields & set(row):
            row[key] = int(row[key])
        for key in float_fields & set(row):
            row[key] = float(row[key])
    if len({str(row["file_name"]) for row in rows}) != len(rows):
        raise ValueError("fold manifest contains duplicate physical file names")
    for row in rows:
        if str(row["group"]) != observation_date(str(row["file_name"])):
            raise ValueError(
                f"manifest group for {row['file_name']} is not its acquisition date"
            )
    group_folds: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        group_folds[str(row["group"])].add(int(row["fold"]))
    crossed = sorted(group for group, folds in group_folds.items() if len(folds) != 1)
    if crossed:
        raise ValueError(f"fold manifest leaks acquisition dates across folds: {crossed[:5]}")
    if annotation_file is not None:
        metadata_path = path.with_suffix(".meta.json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"fold manifest metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        annotation_path = Path(annotation_file)
        annotation_payload = json.loads(annotation_path.read_text())
        expected_files = {
            str(image["file_name"]) for image in annotation_payload.get("images", [])
        }
        manifest_files = {str(row["file_name"]) for row in rows}
        if manifest_files != expected_files:
            raise ValueError(
                "fold manifest does not exactly cover competition training images; "
                f"missing={sorted(expected_files - manifest_files)[:5]}, "
                f"unexpected={sorted(manifest_files - expected_files)[:5]}"
            )
        n_splits = int(metadata.get("n_splits", 0))
        if n_splits < 2 or metadata.get("seed") is None:
            raise ValueError("fold manifest metadata must record n_splits>=2 and seed")
        used_folds = {int(row["fold"]) for row in rows}
        if used_folds != set(range(n_splits)):
            raise ValueError(
                f"fold manifest uses folds {sorted(used_folds)}, expected 0..{n_splits - 1}"
            )
        expected = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "target_policy": TARGET_POLICY,
            "group_strategy": GROUP_STRATEGY,
            "rows": len(rows),
            "annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"stale or incompatible fold manifest: {mismatches}")
    return rows
