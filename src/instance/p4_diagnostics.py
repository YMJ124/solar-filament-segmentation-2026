"""P4 diagnostics for thin-mask representation and P3 sampling shift."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.data.coco import CocoAnnotations
from src.data.masks import annotations_to_rles
from src.metrics.pq import pairwise_rle_iou

from .yolo_evaluation import YoloPostprocessConfig, predictions_for_config
from .yolo_failure_modes import iou_miss_band


DEFAULT_INFERENCE_SIZES = (1024, 1536)
MEDIUM_AREA_MIN = 400
MEDIUM_AREA_MAX = 1600


def _instance_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["file_name"]),
        str(row["image_id"]),
        str(row["annotation_id"]),
    )


def _paired_transition(control_outcome: str, candidate_outcome: str) -> str:
    control_matched = control_outcome == "matched_gt_0_5"
    candidate_matched = candidate_outcome == "matched_gt_0_5"
    if control_matched and candidate_matched:
        return "both_matched"
    if not control_matched and candidate_matched:
        return "recovered_by_t1"
    if control_matched and not candidate_matched:
        return "regressed_in_t1"
    return "both_missed"


def pair_residual_instance_rows(
    control_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    reference_size: int = 1536,
) -> list[dict[str, Any]]:
    """Join the same GT instances and label I1-to-T1 match transitions."""
    control = {_instance_key(row): row for row in control_rows}
    candidate = {_instance_key(row): row for row in candidate_rows}
    if len(control) != len(control_rows) or len(candidate) != len(candidate_rows):
        raise ValueError("duplicate GT identity in residual rows")
    if set(control) != set(candidate):
        raise ValueError("control and candidate residual rows contain different GT instances")

    band_key = f"representation_band_{reference_size}"
    paired = []
    for key in sorted(control):
        left, right = control[key], candidate[key]
        for field in ("fold", "station", "area", band_key):
            if left[field] != right[field]:
                raise ValueError(f"paired GT metadata mismatch for {key}: {field}")
        i1_outcome = str(left["outcome"])
        t1_outcome = str(right["outcome"])
        paired.append(
            {
                "file_name": key[0],
                "image_id": key[1],
                "annotation_id": key[2],
                "fold": int(left["fold"]),
                "station": str(left["station"]),
                "area": float(left["area"]),
                "medium_area": bool(left["medium_area"]),
                "representation_band": str(left[band_key]),
                "i1_outcome": i1_outcome,
                "t1_outcome": t1_outcome,
                "transition": _paired_transition(i1_outcome, t1_outcome),
                "i1_max_prediction_iou": float(left["max_prediction_iou"]),
                "t1_max_prediction_iou": float(right["max_prediction_iou"]),
                "max_prediction_iou_delta": float(right["max_prediction_iou"])
                - float(left["max_prediction_iou"]),
            }
        )
    return paired


def summarize_paired_residual_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize paired I1/T1 GT transitions by width, area, Fold and station."""
    if not rows:
        raise ValueError("no paired residual rows supplied")

    def area_band(area: float) -> str:
        if area < 400:
            return "0_399"
        if area < 1600:
            return "400_1599"
        if area < 6400:
            return "1600_6399"
        return "6400_plus"

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        transitions = Counter(str(row["transition"]) for row in selected)
        i1_misses = sum(row["i1_outcome"] != "matched_gt_0_5" for row in selected)
        t1_misses = sum(row["t1_outcome"] != "matched_gt_0_5" for row in selected)
        recovered = transitions.get("recovered_by_t1", 0)
        regressed = transitions.get("regressed_in_t1", 0)
        i1_matched = len(selected) - i1_misses
        deltas = np.asarray(
            [float(row["max_prediction_iou_delta"]) for row in selected],
            dtype=np.float64,
        )
        remaining = Counter(
            str(row["t1_outcome"])
            for row in selected
            if row["t1_outcome"] != "matched_gt_0_5"
        )
        return {
            "instances": len(selected),
            "i1_misses": i1_misses,
            "t1_misses": t1_misses,
            "miss_delta": t1_misses - i1_misses,
            "transitions": dict(sorted(transitions.items())),
            "net_recovered": recovered - regressed,
            "recovery_rate_of_i1_misses": recovered / i1_misses if i1_misses else 0.0,
            "regression_rate_of_i1_matches": regressed / i1_matched if i1_matched else 0.0,
            "mean_max_iou_delta": float(deltas.mean()),
            "median_max_iou_delta": float(np.median(deltas)),
            "remaining_t1_miss_outcomes": dict(sorted(remaining.items())),
        }

    def grouped(key: str) -> dict[str, Any]:
        values = sorted({str(row[key]) for row in rows})
        return {
            value: aggregate([row for row in rows if str(row[key]) == value])
            for value in values
        }

    enriched = [dict(row, area_band=area_band(float(row["area"]))) for row in rows]
    return {
        "global": aggregate(enriched),
        "by_representation_band_1536": {
            value: aggregate(
                [row for row in enriched if str(row["representation_band"]) == value]
            )
            for value in sorted({str(row["representation_band"]) for row in enriched})
        },
        "by_area": {
            value: aggregate([row for row in enriched if row["area_band"] == value])
            for value in sorted({str(row["area_band"]) for row in enriched})
        },
        "by_fold": grouped("fold"),
        "by_station": grouped("station"),
    }


def summarize_cap_pressure(
    rows: Sequence[Mapping[str, Any]], *, max_instances: int
) -> dict[str, Any]:
    """Summarize whether the postprocess or raw cache instance limits bind."""
    if not rows:
        raise ValueError("no cap-pressure rows supplied")

    def aggregate(selected: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
        total = len(selected)
        above_key = f"{prefix}_candidates_above_confidence"
        beyond_key = f"{prefix}_raw_area_eligible_beyond_cap"
        selected_key = f"{prefix}_prediction_instances"
        cache_key = f"{prefix}_candidate_count"
        cache_limit_key = f"{prefix}_cache_max_candidates"
        truncated = sum(int(row[above_key]) > max_instances for row in selected)
        at_or_above = sum(int(row[above_key]) >= max_instances for row in selected)
        area_beyond = sum(int(row[beyond_key]) > 0 for row in selected)
        selected_at_cap = sum(int(row[selected_key]) >= max_instances for row in selected)
        cache_limit_hits = sum(
            int(row[cache_key]) >= int(row[cache_limit_key]) for row in selected
        )
        return {
            "physical_images": total,
            "confidence_pool_at_or_above_cap_images": at_or_above,
            "confidence_pool_truncated_images": truncated,
            "confidence_pool_truncated_rate": truncated / total,
            "raw_area_eligible_beyond_cap_images": area_beyond,
            "raw_area_eligible_beyond_cap_total": sum(
                int(row[beyond_key]) for row in selected
            ),
            "output_at_cap_images": selected_at_cap,
            "cache_max_candidates_hit_images": cache_limit_hits,
        }

    return {
        "i1": aggregate(rows, "i1"),
        "t1": aggregate(rows, "t1"),
        "by_fold": {
            str(fold): {
                "i1": aggregate(
                    [row for row in rows if int(row["fold"]) == fold], "i1"
                ),
                "t1": aggregate(
                    [row for row in rows if int(row["fold"]) == fold], "t1"
                ),
            }
            for fold in sorted({int(row["fold"]) for row in rows})
        },
    }


def annotation_area(annotation: Mapping[str, Any]) -> float:
    """Return a finite positive annotation area, with a polygon fallback."""
    value = annotation.get("area")
    if value is not None and np.isfinite(float(value)) and float(value) > 0:
        return float(value)
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list) or len(segmentation) != 1:
        raise ValueError(f"annotation {annotation.get('id')} has no usable area")
    points = np.asarray(segmentation[0], dtype=np.float64).reshape(-1, 2)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError(f"annotation {annotation.get('id')} has an invalid polygon")
    x, y = points[:, 0], points[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    if area <= 0:
        raise ValueError(f"annotation {annotation.get('id')} has zero polygon area")
    return area


def spine_length(annotation: Mapping[str, Any]) -> float:
    """Return polyline length for an annotation spine, or zero when unavailable."""
    values = annotation.get("spine")
    if not isinstance(values, list) or len(values) < 4 or len(values) % 2:
        return 0.0
    points = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(points).all():
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def equivalent_width(annotation: Mapping[str, Any]) -> tuple[float, float, str]:
    """Estimate mean filament width as area divided by annotated spine length."""
    area = annotation_area(annotation)
    length = spine_length(annotation)
    if length > 0:
        return area / length, length, "area_over_spine_length"
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"annotation {annotation.get('id')} has no usable spine or bbox")
    width = min(float(bbox[2]), float(bbox[3]))
    if not np.isfinite(width) or width <= 0:
        raise ValueError(f"annotation {annotation.get('id')} has invalid bbox dimensions")
    return width, 0.0, "bbox_minor_side_fallback"


def representation_band(proto_cells: float) -> str:
    """Bucket effective width by the number of stride-4 mask-prototype cells."""
    if proto_cells < 1:
        return "lt_1_proto_cell"
    if proto_cells < 2:
        return "1_to_lt_2_proto_cells"
    return "2_plus_proto_cells"


def _outcome(max_iou: float) -> str:
    return "matched_gt_0_5" if max_iou > 0.5 else iou_miss_band(max_iou)


def analyze_representation_image(
    masks: np.ndarray,
    scores: np.ndarray,
    record: Mapping[str, Any],
    coco: CocoAnnotations,
    config: YoloPostprocessConfig,
    *,
    inference_sizes: Sequence[int] = DEFAULT_INFERENCE_SIZES,
) -> list[dict[str, Any]]:
    """Return one representation/error row per annotator GT instance."""
    file_name = str(record["file_name"])
    height, width = int(record["height"]), int(record["width"])
    _, prediction_rles = predictions_for_config(masks, scores, config)
    batches = coco.batches_for_file(file_name)
    rows: list[dict[str, Any]] = []
    for batch in batches:
        batch_id = str(batch["id"])
        annotations = coco.annotations_for_batch(batch_id)
        truth_rles = annotations_to_rles(annotations, height, width)
        matrix = pairwise_rle_iou(truth_rles, prediction_rles)
        maxima = (
            matrix.max(axis=1)
            if len(prediction_rles)
            else np.zeros(len(annotations), dtype=np.float64)
        )
        for annotation, max_iou in zip(annotations, maxima, strict=True):
            area = annotation_area(annotation)
            width_estimate, length, width_source = equivalent_width(annotation)
            row: dict[str, Any] = {
                "file_name": file_name,
                "fold": int(record["fold"]),
                "year": Path(file_name).stem[:4],
                "station": Path(file_name).stem[-2:],
                "annotator_batches": len(batches),
                "image_id": batch_id,
                "annotation_id": str(annotation.get("id")),
                "area": area,
                "medium_area": MEDIUM_AREA_MIN <= area < MEDIUM_AREA_MAX,
                "spine_length": length,
                "equivalent_width_original": width_estimate,
                "width_source": width_source,
                "max_prediction_iou": float(max_iou),
                "outcome": _outcome(float(max_iou)),
            }
            for image_size in inference_sizes:
                scale = min(float(image_size) / width, float(image_size) / height)
                effective = width_estimate * scale
                row[f"equivalent_width_{image_size}"] = effective
                row[f"proto_cells_{image_size}"] = effective / 4.0
                row[f"representation_band_{image_size}"] = representation_band(
                    effective / 4.0
                )
            rows.append(row)
    return rows


def summarize_representation_rows(
    rows: Sequence[Mapping[str, Any]], *, reference_size: int = 1024
) -> dict[str, Any]:
    """Summarize match outcomes by width band, fold, and station."""
    if not rows:
        raise ValueError("no representation rows supplied")

    band_key = f"representation_band_{reference_size}"

    def aggregate(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        outcomes = Counter(str(row["outcome"]) for row in selected)
        total = len(selected)
        misses = total - outcomes.get("matched_gt_0_5", 0)
        return {
            "instances": total,
            "matched": outcomes.get("matched_gt_0_5", 0),
            "misses": misses,
            "miss_rate": misses / total if total else 0.0,
            "outcomes": dict(sorted(outcomes.items())),
            "medium_instances": sum(bool(row["medium_area"]) for row in selected),
        }

    bands = sorted({str(row[band_key]) for row in rows})
    folds = sorted({int(row["fold"]) for row in rows})
    stations = sorted({str(row["station"]) for row in rows})
    return {
        "reference_size": int(reference_size),
        "global": aggregate(rows),
        "by_representation_band": {
            band: aggregate([row for row in rows if str(row[band_key]) == band])
            for band in bands
        },
        "by_fold": {
            str(fold): aggregate([row for row in rows if int(row["fold"]) == fold])
            for fold in folds
        },
        "by_station": {
            station: aggregate(
                [row for row in rows if str(row["station"]) == station]
            )
            for station in stations
        },
    }


def physical_feature_rows(
    records: Iterable[Mapping[str, Any]], coco: CocoAnnotations
) -> list[dict[str, Any]]:
    """Build medoid-only physical-image features used by P3/P4 sampling audits."""
    rows = []
    for record in records:
        file_name = str(record["file_name"])
        image_id = str(record["selected_image_id"])
        annotations = coco.annotations_for_batch(image_id)
        areas = [annotation_area(annotation) for annotation in annotations]
        medium_count = sum(MEDIUM_AREA_MIN <= area < MEDIUM_AREA_MAX for area in areas)
        rows.append(
            {
                "file_name": file_name,
                "fold": int(record["fold"]),
                "year": Path(file_name).stem[:4],
                "station": Path(file_name).stem[-2:],
                "annotator_batches": int(record.get("n_batches", 1)),
                "medoid_mean_iou": float(record.get("medoid_mean_iou", 1.0)),
                "selected_image_id": image_id,
                "instances": len(annotations),
                "medium_instances": medium_count,
                "medium_bin": (
                    "0_2" if medium_count <= 2 else "3_5" if medium_count <= 5 else "6_plus"
                ),
            }
        )
    return rows


def _distribution(
    rows: Sequence[Mapping[str, Any]], exposures: Mapping[str, int], key: str
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    weighted: Counter[str] = Counter()
    for row in rows:
        value = str(row[key])
        name = str(row["file_name"])
        counts[value] += 1
        weighted[value] += int(exposures[name])
    base_total = sum(counts.values())
    exposure_total = sum(weighted.values())
    return {
        value: {
            "physical_images": counts[value],
            "p3_exposures": weighted[value],
            "base_share": counts[value] / base_total if base_total else 0.0,
            "p3_share": weighted[value] / exposure_total if exposure_total else 0.0,
            "share_delta": (
                weighted[value] / exposure_total - counts[value] / base_total
                if base_total and exposure_total
                else 0.0
            ),
        }
        for value in sorted(counts)
    }


def summarize_p3_sampling_shift(
    feature_rows: Sequence[Mapping[str, Any]],
    manifests_by_validation_fold: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare P3 repeated-sample exposure with the physical-image baseline."""
    result: dict[str, Any] = {}
    for fold, manifest in sorted(manifests_by_validation_fold.items()):
        train_rows = [row for row in feature_rows if int(row["fold"]) != int(fold)]
        names = {str(row["file_name"]) for row in train_rows}
        exposures = {name: 1 for name in names}
        repeated = manifest.get("repeated_train_images", [])
        for row in repeated:
            name = str(row["file_name"])
            if name not in exposures:
                raise ValueError(f"P3 manifest fold {fold} repeats non-train image: {name}")
            exposures[name] += len(row.get("sample_names", []))
        expected_samples = int(manifest.get("train_samples", -1))
        if sum(exposures.values()) != expected_samples:
            raise ValueError(f"P3 manifest fold {fold} exposure count is inconsistent")
        result[str(fold)] = {
            "physical_train_images": len(train_rows),
            "p3_train_samples": sum(exposures.values()),
            "optimizer_step_multiplier": sum(exposures.values()) / len(train_rows),
            "repeated_physical_images": sum(value > 1 for value in exposures.values()),
            "station": _distribution(train_rows, exposures, "station"),
            "year": _distribution(train_rows, exposures, "year"),
            "medium_bin": _distribution(train_rows, exposures, "medium_bin"),
        }
    return result
