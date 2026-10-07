"""Annotator-aware adapters for the official competition PQ."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from src.data.coco import CocoAnnotations
from src.data.masks import annotations_to_rles
from src.submission.rle import rle_from_counts

from .pq import OfficialPQReport, PQCounts, aggregate_counts, score_rles


def score_physical_image(
    coco: CocoAnnotations,
    file_name: str,
    prediction_rles: list[dict],
) -> list[PQCounts]:
    """Reuse one physical-image prediction against every annotator batch."""
    batches = coco.batches_for_file(file_name)
    if not batches:
        raise KeyError(f"unknown physical image: {file_name}")
    scores = []
    for batch in batches:
        height, width = int(batch["height"]), int(batch["width"])
        truth_rles = annotations_to_rles(
            coco.annotations_for_batch(str(batch["id"])), height, width
        )
        scores.append(score_rles(truth_rles, prediction_rles))
    return scores


def evaluate_predictions(
    coco: CocoAnnotations,
    predictions_by_file: Mapping[str, list[dict]],
    file_names: Iterable[str] | None = None,
) -> OfficialPQReport:
    """Evaluate predictions over selected physical files with global aggregation."""
    selected = list(file_names) if file_names is not None else coco.physical_files
    unknown = sorted(set(selected) - set(coco.physical_files))
    if unknown:
        raise ValueError(f"unknown evaluation image names: {unknown[:5]}")
    prediction_unknown = sorted(set(predictions_by_file) - set(coco.physical_files))
    if prediction_unknown:
        raise ValueError(f"unknown prediction image names: {prediction_unknown[:5]}")

    counts = []
    for file_name in selected:
        counts.extend(
            score_physical_image(
                coco,
                file_name,
                list(predictions_by_file.get(file_name, [])),
            )
        )
    return aggregate_counts(counts, physical_images=len(selected))


def load_prediction_csv(
    path: str | Path,
    coco: CocoAnnotations,
) -> dict[str, list[dict]]:
    """Load and validate a competition-format prediction CSV."""
    path = Path(path)
    stem_to_name = {Path(name).stem: name for name in coco.physical_files}
    predictions: dict[str, list[dict]] = defaultdict(list)
    seen_ids: set[str] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["filament_id", "segmentation_rle"]
        if reader.fieldnames != expected:
            raise ValueError(f"CSV columns must be exactly {expected}, got {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=2):
            filament_id = str(row.get("filament_id", ""))
            stem, separator, suffix = filament_id.partition("_")
            if not separator or not stem or not suffix:
                raise ValueError(f"row {row_number}: invalid filament_id {filament_id!r}")
            if filament_id in seen_ids:
                raise ValueError(f"row {row_number}: duplicate filament_id {filament_id!r}")
            seen_ids.add(filament_id)
            if stem not in stem_to_name:
                raise ValueError(f"row {row_number}: unknown image stem {stem!r}")
            file_name = stem_to_name[stem]
            batch = coco.batches_for_file(file_name)[0]
            predictions[file_name].append(
                rle_from_counts(
                    str(row.get("segmentation_rle", "")),
                    height=int(batch["height"]),
                    width=int(batch["width"]),
                )
            )
    return dict(predictions)
