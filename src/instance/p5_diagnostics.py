"""P5 diagnostics for raw-candidate recoverability and experiment routing."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.instance.p4_diagnostics import annotation_area, equivalent_width, representation_band
from src.instance.yolo_evaluation import YoloPostprocessConfig
from src.instance.yolo_failure_modes import area_bin, iou_miss_band
from src.metrics.pq import OFFICIAL_IOU_THRESHOLD, pairwise_rle_iou


RANKING_GLOBAL_MIN_RATE = 0.10
RANKING_EACH_FOLD_MIN_RATE = 0.05
CANDIDATE_ABSENCE_GLOBAL_MIN_RATE = 0.50
CANDIDATE_ABSENCE_FOLD_MIN_RATE = 0.50
REFINER_GLOBAL_MIN_RATE = 0.25
REFINER_POSITIVE_FOLDS = 2
BBOX_SHORT_SIDE_EDGES = (32, 64, 128)


def raw_iou_band(max_iou: float) -> str:
    """Extend the miss bands with an oracle-recoverable ``IoU > 0.5`` band."""
    if max_iou > OFFICIAL_IOU_THRESHOLD:
        return "recoverable_gt_0_5"
    return iou_miss_band(max_iou)


def recoverability_route(
    ious: np.ndarray,
    scores: np.ndarray,
    areas: np.ndarray,
    config: YoloPostprocessConfig,
) -> str:
    """Identify the earliest pipeline stage able to retain any high-IoU mask."""
    ious = np.asarray(ious, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    areas = np.asarray(areas, dtype=np.int64).reshape(-1)
    if not (len(ious) == len(scores) == len(areas)):
        raise ValueError("candidate IoUs, scores and areas must have equal length")
    high_iou = np.flatnonzero(ious > OFFICIAL_IOU_THRESHOLD)
    if not len(high_iou):
        return "not_recoverable_from_cached_candidates"

    ranks = np.arange(1, len(scores) + 1)
    above_conf = scores >= config.confidence
    within_cap = ranks <= config.max_instances
    area_ok = areas >= config.min_area
    if np.any(above_conf[high_iou] & within_cap[high_iou] & area_ok[high_iou]):
        return "postprocess_interaction_after_preselection"
    if np.any(above_conf[high_iou] & within_cap[high_iou]):
        return "raw_area_below_min_area"
    if np.any(above_conf[high_iou]):
        return "beyond_max_instances_cap"
    return "below_confidence_threshold"


def _candidate_maximum(
    ious: np.ndarray,
    scores: np.ndarray,
    areas: np.ndarray,
) -> dict[str, Any]:
    if not len(ious):
        return {
            "raw_max_iou": 0.0,
            "raw_best_candidate_rank": 0,
            "raw_best_candidate_confidence": 0.0,
            "raw_best_candidate_area": 0,
        }
    index = int(np.argmax(ious))
    return {
        "raw_max_iou": float(ious[index]),
        "raw_best_candidate_rank": index + 1,
        "raw_best_candidate_confidence": float(scores[index]),
        "raw_best_candidate_area": int(areas[index]),
    }


def audit_candidate_batch(
    annotations: Sequence[Mapping[str, Any]],
    truth_rles: list[dict[str, Any]],
    residual_by_annotation: Mapping[str, Mapping[str, Any]],
    candidate_rles: list[dict[str, Any]],
    scores: np.ndarray,
    areas: np.ndarray,
    record: Mapping[str, Any],
    config: YoloPostprocessConfig,
    *,
    cache_max_candidates: int,
    inference_size: int,
) -> list[dict[str, Any]]:
    """Audit cached raw masks for every final T1 miss in one annotation batch."""
    if len(annotations) != len(truth_rles):
        raise ValueError("annotation and truth-RLE counts differ")
    if not (len(candidate_rles) == len(scores) == len(areas)):
        raise ValueError("candidate RLE, score and area counts differ")
    matrix = pairwise_rle_iou(truth_rles, candidate_rles)
    file_name = str(record["file_name"])
    height, width = int(record["height"]), int(record["width"])
    scale = min(float(inference_size) / width, float(inference_size) / height)
    rows: list[dict[str, Any]] = []
    for annotation, candidate_ious in zip(annotations, matrix, strict=True):
        annotation_id = str(annotation.get("id"))
        residual = residual_by_annotation.get(annotation_id)
        if residual is None:
            raise ValueError(
                f"residual audit is missing annotation {annotation_id} for {file_name}"
            )
        if str(residual["t1_outcome"]) == "matched_gt_0_5":
            continue
        maximum = _candidate_maximum(candidate_ious, scores, areas)
        width_estimate, spine_length, width_source = equivalent_width(annotation)
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"annotation {annotation_id} has no usable bbox")
        bbox_short = min(float(bbox[2]), float(bbox[3]))
        if not np.isfinite(bbox_short) or bbox_short <= 0:
            raise ValueError(f"annotation {annotation_id} has invalid bbox")
        effective_width = width_estimate * scale
        effective_bbox_short = bbox_short * scale
        rows.append(
            {
                "file_name": file_name,
                "image_id": str(residual["image_id"]),
                "annotation_id": annotation_id,
                "fold": int(record["fold"]),
                "station": Path(file_name).stem[-2:],
                "area": float(annotation_area(annotation)),
                "area_band": area_bin(int(round(annotation_area(annotation))), (400, 1600, 6400)),
                "spine_length": float(spine_length),
                "width_source": width_source,
                "equivalent_width_original": float(width_estimate),
                "equivalent_width_inference": float(effective_width),
                "representation_band": representation_band(effective_width / 4.0),
                "bbox_short_side_original": float(bbox_short),
                "bbox_short_side_inference": float(effective_bbox_short),
                "bbox_short_side_band": area_bin(
                    int(np.floor(effective_bbox_short)), BBOX_SHORT_SIDE_EDGES
                ),
                "final_outcome": str(residual["t1_outcome"]),
                "final_max_iou": float(residual["t1_max_prediction_iou"]),
                **maximum,
                "raw_iou_band": raw_iou_band(float(maximum["raw_max_iou"])),
                "raw_high_iou_candidates": int(
                    np.count_nonzero(candidate_ious > OFFICIAL_IOU_THRESHOLD)
                ),
                "recoverability_route": recoverability_route(
                    candidate_ious, scores, areas, config
                ),
                "cached_candidate_count": len(candidate_rles),
                "cache_max_candidates": int(cache_max_candidates),
                "cache_limit_hit": len(candidate_rles) >= int(cache_max_candidates),
            }
        )
    return rows


def summarize_recoverability_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate candidate recoverability and produce preregistered routing checks."""
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError("no missed-GT recoverability rows supplied")

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        total = len(selected)
        outcomes = Counter(str(row["final_outcome"]) for row in selected)
        raw_bands = Counter(str(row["raw_iou_band"]) for row in selected)
        routes = Counter(str(row["recoverability_route"]) for row in selected)
        no_overlap = [row for row in selected if row["final_outcome"] == "no_overlap"]
        no_overlap_raw = Counter(str(row["raw_iou_band"]) for row in no_overlap)
        no_overlap_routes = Counter(
            str(row["recoverability_route"]) for row in no_overlap
        )
        no_overlap_recoverable = no_overlap_raw.get("recoverable_gt_0_5", 0)
        no_overlap_absent = sum(
            no_overlap_raw.get(name, 0)
            for name in ("no_overlap", "low_overlap_le_0_25")
        )
        near_threshold = outcomes.get("near_threshold_gt_0_25_to_0_5", 0)
        return {
            "miss_instances": total,
            "final_outcomes": dict(sorted(outcomes.items())),
            "raw_iou_bands": dict(sorted(raw_bands.items())),
            "recoverability_routes": dict(sorted(routes.items())),
            "raw_recoverable_gt_0_5": raw_bands.get("recoverable_gt_0_5", 0),
            "raw_recoverable_rate": (
                raw_bands.get("recoverable_gt_0_5", 0) / total if total else 0.0
            ),
            "final_no_overlap_instances": len(no_overlap),
            "final_no_overlap_raw_iou_bands": dict(sorted(no_overlap_raw.items())),
            "final_no_overlap_recoverability_routes": dict(
                sorted(no_overlap_routes.items())
            ),
            "final_no_overlap_raw_recoverable_gt_0_5": no_overlap_recoverable,
            "final_no_overlap_raw_recoverable_rate": (
                no_overlap_recoverable / len(no_overlap) if no_overlap else 0.0
            ),
            "final_no_overlap_raw_le_0_25": no_overlap_absent,
            "final_no_overlap_raw_le_0_25_rate": (
                no_overlap_absent / len(no_overlap) if no_overlap else 0.0
            ),
            "final_near_threshold_instances": near_threshold,
            "final_near_threshold_rate_of_misses": (
                near_threshold / total if total else 0.0
            ),
            "cache_limit_hit_instances": sum(bool(row["cache_limit_hit"]) for row in selected),
            "cache_limit_hit_rate": (
                sum(bool(row["cache_limit_hit"]) for row in selected) / total
                if total
                else 0.0
            ),
        }

    folds = sorted({int(row["fold"]) for row in rows})
    by_fold = {
        str(fold): aggregate([row for row in rows if int(row["fold"]) == fold])
        for fold in folds
    }
    global_summary = aggregate(rows)
    ranking_supported = (
        global_summary["final_no_overlap_raw_recoverable_rate"]
        >= RANKING_GLOBAL_MIN_RATE
        and all(
            row["final_no_overlap_raw_recoverable_rate"]
            >= RANKING_EACH_FOLD_MIN_RATE
            for row in by_fold.values()
        )
    )
    absence_positive_folds = sum(
        row["final_no_overlap_raw_le_0_25_rate"]
        >= CANDIDATE_ABSENCE_FOLD_MIN_RATE
        for row in by_fold.values()
    )
    absence_supported = (
        global_summary["final_no_overlap_raw_le_0_25_rate"]
        >= CANDIDATE_ABSENCE_GLOBAL_MIN_RATE
        and absence_positive_folds >= 2
    )
    refiner_positive_folds = sum(
        row["final_near_threshold_rate_of_misses"] >= REFINER_GLOBAL_MIN_RATE
        for row in by_fold.values()
    )
    refiner_supported = (
        global_summary["final_near_threshold_rate_of_misses"]
        >= REFINER_GLOBAL_MIN_RATE
        and refiner_positive_folds >= REFINER_POSITIVE_FOLDS
    )
    return {
        "global": global_summary,
        "by_fold": by_fold,
        "by_area": {
            name: aggregate([row for row in rows if str(row["area_band"]) == name])
            for name in sorted({str(row["area_band"]) for row in rows})
        },
        "by_representation_band": {
            name: aggregate(
                [row for row in rows if str(row["representation_band"]) == name]
            )
            for name in sorted({str(row["representation_band"]) for row in rows})
        },
        "by_bbox_short_side_band": {
            name: aggregate(
                [row for row in rows if str(row["bbox_short_side_band"]) == name]
            )
            for name in sorted({str(row["bbox_short_side_band"]) for row in rows})
        },
        "routing_thresholds": {
            "ranking_global_min_rate": RANKING_GLOBAL_MIN_RATE,
            "ranking_each_fold_min_rate": RANKING_EACH_FOLD_MIN_RATE,
            "candidate_absence_global_min_rate": CANDIDATE_ABSENCE_GLOBAL_MIN_RATE,
            "candidate_absence_fold_min_rate": CANDIDATE_ABSENCE_FOLD_MIN_RATE,
            "candidate_absence_minimum_positive_folds": 2,
            "refiner_global_min_rate": REFINER_GLOBAL_MIN_RATE,
            "refiner_minimum_positive_folds": REFINER_POSITIVE_FOLDS,
        },
        "routing_checks": {
            "mask_quality_reranking_supported": ranking_supported,
            "candidate_generation_supported": absence_supported,
            "candidate_absence_positive_folds": absence_positive_folds,
            "crop_refiner_supported": refiner_supported,
            "crop_refiner_positive_folds": refiner_positive_folds,
        },
        "recommended_recall_route": (
            "mask_quality_reranking"
            if ranking_supported
            else "candidate_generation"
            if absence_supported
            else "mixed_manual_review"
        ),
        "recommended_mask_route": (
            "crop_refiner" if refiner_supported else "retain_current_mask_head"
        ),
        "cache_censoring_note": (
            "Raw-candidate absence is a lower bound whenever cache_limit_hit is true."
        ),
    }
