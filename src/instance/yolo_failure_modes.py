"""Fine-grained false-negative and false-positive diagnostics for YOLO caches."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.coco import CocoAnnotations
from src.data.masks import annotations_to_masks, annotations_to_rles
from src.metrics.pq import aggregate_counts, pairwise_rle_iou, score_iou_matrix

from .yolo_evaluation import YoloPostprocessConfig, predictions_for_config


TRUTH_AREA_EDGES = (400, 1600, 6400)
PREDICTION_AREA_EDGES = (1200, 2400, 4800)


def area_bin(area: int, edges: Sequence[int]) -> str:
    """Return a stable half-open area-bin label."""
    lower = 0
    for upper in edges:
        if area < upper:
            return f"{lower}_{upper - 1}"
        lower = upper
    return f"{lower}_plus"


def iou_miss_band(max_iou: float) -> str:
    """Classify an unmatched instance by its best sub-threshold overlap."""
    if max_iou <= 0:
        return "no_overlap"
    if max_iou <= 0.25:
        return "low_overlap_le_0_25"
    return "near_threshold_gt_0_25_to_0_5"


def analyze_yolo_failure_modes(
    masks: np.ndarray,
    scores: np.ndarray,
    record: Mapping[str, Any],
    coco: CocoAnnotations,
    config: YoloPostprocessConfig,
    *,
    truth_area_edges: Sequence[int] = TRUTH_AREA_EDGES,
    prediction_area_edges: Sequence[int] = PREDICTION_AREA_EDGES,
) -> dict[str, Any]:
    """Decompose misses and false positives for one physical image."""
    file_name = str(record["file_name"])
    height, width = int(record["height"]), int(record["width"])
    labels, prediction_rles = predictions_for_config(masks, scores, config)
    prediction_areas = [
        int((labels == index).sum()) for index in range(1, int(labels.max()) + 1)
    ]
    batches = coco.batches_for_file(file_name)
    if not batches:
        raise KeyError(f"unknown physical image: {file_name}")

    truth_by_area: Counter[str] = Counter()
    fn_by_area: Counter[str] = Counter()
    fn_iou_bands: Counter[str] = Counter()
    official_fp_iou_bands: Counter[str] = Counter()
    counts = []
    prediction_hit_batches = np.zeros(len(prediction_rles), dtype=np.int64)
    prediction_max_iou = np.zeros(len(prediction_rles), dtype=np.float64)

    for batch in batches:
        batch_id = str(batch["id"])
        if (int(batch["height"]), int(batch["width"])) != (height, width):
            raise ValueError(f"annotation shape mismatch for {file_name}")
        annotations = coco.annotations_for_batch(batch_id)
        truth_rles = annotations_to_rles(annotations, height, width)
        truth_masks = annotations_to_masks(annotations, height, width)
        matrix = pairwise_rle_iou(truth_rles, prediction_rles)
        batch_counts = score_iou_matrix(matrix, len(truth_rles), len(prediction_rles))
        counts.append(batch_counts)
        hits = matrix > 0.5

        if len(prediction_rles):
            prediction_hit_batches += (hits.sum(axis=0) > 0).astype(np.int64)
            if len(truth_rles):
                prediction_max_iou = np.maximum(prediction_max_iou, matrix.max(axis=0))
            batch_prediction_max = (
                matrix.max(axis=0)
                if len(truth_rles)
                else np.zeros(len(prediction_rles), dtype=np.float64)
            )
            for index in np.flatnonzero(hits.sum(axis=0) == 0):
                official_fp_iou_bands[iou_miss_band(float(batch_prediction_max[index]))] += 1

        truth_max = (
            matrix.max(axis=1)
            if len(prediction_rles)
            else np.zeros(len(truth_rles), dtype=np.float64)
        )
        for index, truth_mask in enumerate(truth_masks):
            bin_name = area_bin(int(truth_mask.sum()), truth_area_edges)
            truth_by_area[bin_name] += 1
            if not hits[index].any():
                fn_by_area[bin_name] += 1
                fn_iou_bands[iou_miss_band(float(truth_max[index]))] += 1

    report = aggregate_counts(counts, physical_images=1).to_dict()
    n_batches = len(batches)
    consensus_indices = np.flatnonzero(prediction_hit_batches == 0)
    partial_indices = np.flatnonzero(
        (prediction_hit_batches > 0) & (prediction_hit_batches < n_batches)
    )
    consensus_area_bins: Counter[str] = Counter()
    consensus_iou_bands: Counter[str] = Counter()
    for index in consensus_indices:
        consensus_area_bins[area_bin(prediction_areas[index], prediction_area_edges)] += 1
        consensus_iou_bands[iou_miss_band(float(prediction_max_iou[index]))] += 1

    consensus_fp_records = int(len(consensus_indices) * n_batches)
    partial_fp_records = int(
        sum(n_batches - prediction_hit_batches[index] for index in partial_indices)
    )
    if consensus_fp_records + partial_fp_records != report["false_positives"]:
        raise RuntimeError(f"false-positive decomposition mismatch for {file_name}")

    stem = Path(file_name).stem
    return {
        "file_name": file_name,
        "fold": int(record["fold"]),
        "year": stem[:4],
        "station": stem[-2:],
        "annotator_batches": n_batches,
        "prediction_instances": len(prediction_rles),
        "truth_instances": int(sum(truth_by_area.values())),
        "hit_pairs": report["hit_pairs"],
        "false_positives": report["false_positives"],
        "false_negatives": report["false_negatives"],
        "hit_iou_sum": report["hit_iou_sum"],
        "truth_instances_by_area": dict(sorted(truth_by_area.items())),
        "false_negatives_by_area": dict(sorted(fn_by_area.items())),
        "false_negative_iou_bands": dict(sorted(fn_iou_bands.items())),
        "official_false_positive_iou_bands": dict(
            sorted(official_fp_iou_bands.items())
        ),
        "consensus_false_positive_instances": len(consensus_indices),
        "consensus_false_positive_records": consensus_fp_records,
        "partial_annotator_false_positive_instances": len(partial_indices),
        "partial_annotator_false_positive_records": partial_fp_records,
        "consensus_false_positive_area_bins": dict(
            sorted(consensus_area_bins.items())
        ),
        "consensus_false_positive_iou_bands": dict(
            sorted(consensus_iou_bands.items())
        ),
    }


def _sum_counters(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    result: Counter[str] = Counter()
    for row in rows:
        result.update({name: int(value) for name, value in row[key].items()})
    return dict(sorted(result.items()))


def summarize_failure_mode_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate image-level failure modes with fold and station breakdowns."""
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError("no failure-mode rows supplied")

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        physical = len(selected)
        annotators = sum(int(row["annotator_batches"]) for row in selected)
        tp = sum(int(row["hit_pairs"]) for row in selected)
        fp = sum(int(row["false_positives"]) for row in selected)
        fn = sum(int(row["false_negatives"]) for row in selected)
        hit_iou = float(sum(float(row["hit_iou_sum"]) for row in selected))
        denominator = tp + 0.5 * fp + 0.5 * fn
        truth_by_area = _sum_counters(selected, "truth_instances_by_area")
        fn_by_area = _sum_counters(selected, "false_negatives_by_area")
        fn_iou = _sum_counters(selected, "false_negative_iou_bands")
        fp_iou = _sum_counters(selected, "official_false_positive_iou_bands")
        consensus_area = _sum_counters(selected, "consensus_false_positive_area_bins")
        consensus_iou = _sum_counters(selected, "consensus_false_positive_iou_bands")
        return {
            "physical_images": physical,
            "annotator_records": annotators,
            "official_pq": hit_iou / denominator if denominator else 0.0,
            "hit_pairs": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "hit_iou_sum": hit_iou,
            "prediction_instances": sum(int(row["prediction_instances"]) for row in selected),
            "truth_instances_by_area": truth_by_area,
            "false_negatives_by_area": fn_by_area,
            "false_negative_area_miss_rates": {
                key: fn_by_area.get(key, 0) / value if value else 0.0
                for key, value in truth_by_area.items()
            },
            "false_negative_area_shares": {
                key: value / fn if fn else 0.0 for key, value in fn_by_area.items()
            },
            "false_negative_iou_bands": fn_iou,
            "false_negative_iou_band_shares": {
                key: value / fn if fn else 0.0 for key, value in fn_iou.items()
            },
            "official_false_positive_iou_bands": fp_iou,
            "official_false_positive_iou_band_shares": {
                key: value / fp if fp else 0.0 for key, value in fp_iou.items()
            },
            "consensus_false_positive_instances": sum(
                int(row["consensus_false_positive_instances"]) for row in selected
            ),
            "consensus_false_positive_records": sum(
                int(row["consensus_false_positive_records"]) for row in selected
            ),
            "partial_annotator_false_positive_instances": sum(
                int(row["partial_annotator_false_positive_instances"]) for row in selected
            ),
            "partial_annotator_false_positive_records": sum(
                int(row["partial_annotator_false_positive_records"]) for row in selected
            ),
            "consensus_false_positive_area_bins": consensus_area,
            "consensus_false_positive_iou_bands": consensus_iou,
            "false_positives_per_annotator_record": fp / annotators if annotators else 0.0,
            "predictions_per_physical_image": sum(
                int(row["prediction_instances"]) for row in selected
            ) / physical,
        }

    folds = sorted({int(row["fold"]) for row in rows})
    fold_summaries = {
        str(fold): aggregate([row for row in rows if int(row["fold"]) == fold])
        for fold in folds
    }
    station_summaries = {
        str(fold): {
            station: aggregate(
                [
                    row
                    for row in rows
                    if int(row["fold"]) == fold and str(row["station"]) == station
                ]
            )
            for station in sorted(
                {str(row["station"]) for row in rows if int(row["fold"]) == fold}
            )
        }
        for fold in folds
    }
    return {
        "global": aggregate(rows),
        "fold_summaries": fold_summaries,
        "station_summaries": station_summaries,
        "per_image": rows,
    }
