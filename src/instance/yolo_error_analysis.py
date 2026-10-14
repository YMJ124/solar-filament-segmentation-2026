"""Annotator-aware per-image diagnostics for cached YOLO predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from src.data.coco import CocoAnnotations
from src.data.masks import annotations_to_masks, annotations_to_rles
from src.metrics.pq import aggregate_counts, pairwise_rle_iou, score_iou_matrix

from .yolo_evaluation import YoloPostprocessConfig, predictions_for_config


def analyze_yolo_image(
    masks: np.ndarray,
    scores: np.ndarray,
    record: Mapping[str, Any],
    coco: CocoAnnotations,
    config: YoloPostprocessConfig,
    *,
    small_instance_area: int = 400,
    large_false_positive_area: int = 10_000,
    diagnostic_iou: float = 0.0,
) -> tuple[dict[str, Any], np.ndarray, list[np.ndarray], np.ndarray]:
    """Score one image and derive explicit miss/split/merge diagnostics.

    Official TP/FP/FN use IoU > 0.5. Fragmentation and merging are diagnostic
    relations based on IoU > ``diagnostic_iou`` and are intentionally separate.
    """
    if small_instance_area < 1 or large_false_positive_area < 1:
        raise ValueError("diagnostic area thresholds must be positive")
    if not 0 <= diagnostic_iou < 1:
        raise ValueError("diagnostic_iou must be within [0, 1)")
    file_name = str(record["file_name"])
    height, width = int(record["height"]), int(record["width"])
    labels, prediction_rles = predictions_for_config(masks, scores, config)
    prediction_areas = np.asarray(
        [int((labels == index).sum()) for index in range(1, int(labels.max()) + 1)],
        dtype=np.int64,
    )

    counts = []
    annotators = []
    small_missed = 0
    fragmented = 0
    merged = 0
    prediction_hit_any = np.zeros(len(prediction_rles), dtype=bool)
    selected_masks: list[np.ndarray] | None = None
    selected_matrix: np.ndarray | None = None
    selected_id = record.get("selected_image_id")
    selected_id = str(selected_id) if selected_id is not None else None
    batches = coco.batches_for_file(file_name)
    if not batches:
        raise KeyError(f"unknown physical image: {file_name}")

    for batch_index, batch in enumerate(batches):
        batch_id = str(batch["id"])
        batch_shape = (int(batch["height"]), int(batch["width"]))
        if batch_shape != (height, width):
            raise ValueError(
                f"annotation shape {batch_shape} does not match cached image "
                f"shape {(height, width)} for {file_name}"
            )
        annotations = coco.annotations_for_batch(batch_id)
        truth_rles = annotations_to_rles(annotations, height, width)
        truth_masks = annotations_to_masks(annotations, height, width)
        matrix = pairwise_rle_iou(truth_rles, prediction_rles)
        batch_counts = score_iou_matrix(matrix, len(truth_rles), len(prediction_rles))
        counts.append(batch_counts)
        hits = matrix > 0.5
        if len(prediction_rles):
            prediction_hit_any |= hits.sum(axis=0) > 0
        missed = hits.sum(axis=1) == 0
        truth_areas = [int(mask.sum()) for mask in truth_masks]
        small_batch_misses = sum(
            bool(is_missed) and area < small_instance_area
            for is_missed, area in zip(missed, truth_areas, strict=True)
        )
        relations = matrix > diagnostic_iou
        batch_fragmented = int((relations.sum(axis=1) > 1).sum())
        batch_merged = int((relations.sum(axis=0) > 1).sum())
        small_missed += int(small_batch_misses)
        fragmented += batch_fragmented
        merged += batch_merged
        annotators.append(
            {
                "image_id": batch_id,
                "truth_instances": len(truth_rles),
                "official_pq": batch_counts.pq,
                "hit_pairs": batch_counts.hit_pairs,
                "false_positives": batch_counts.false_positives,
                "false_negatives": batch_counts.false_negatives,
                "small_instance_misses": int(small_batch_misses),
                "fragmented_gt_instances": batch_fragmented,
                "merged_prediction_instances": batch_merged,
            }
        )
        if batch_id == selected_id or (selected_id is None and batch_index == 0):
            selected_id = batch_id
            selected_masks = truth_masks
            selected_matrix = matrix

    if selected_masks is None or selected_matrix is None:
        raise ValueError(
            f"selected annotator batch {selected_id!r} is absent for {file_name}"
        )
    report = aggregate_counts(counts, physical_images=1).to_dict()
    consensus_fp_indices = np.flatnonzero(~prediction_hit_any)
    consensus_fp_areas = [int(prediction_areas[index]) for index in consensus_fp_indices]
    large_fp_areas = [
        area for area in consensus_fp_areas if area >= large_false_positive_area
    ]
    annotator_pq = np.asarray(
        [row["official_pq"] for row in annotators], dtype=np.float64
    )
    truth_counts = np.asarray(
        [row["truth_instances"] for row in annotators], dtype=np.int64
    )
    categories = []
    if large_fp_areas:
        categories.append("large_false_positive")
    if small_missed:
        categories.append("small_instance_miss")
    if fragmented:
        categories.append("fragmented_filament")
    if merged:
        categories.append("merged_filaments")
    if len(annotator_pq) > 1 and float(np.ptp(annotator_pq)) >= 0.1:
        categories.append("annotator_instability")

    row: dict[str, Any] = {
        "file_name": file_name,
        "fold": record.get("fold"),
        "height": height,
        "width": width,
        "annotator_batches": len(annotators),
        "official_pq": report["official_pq"],
        "hit_pairs": report["hit_pairs"],
        "false_positives": report["false_positives"],
        "false_negatives": report["false_negatives"],
        "hit_iou_sum": report["hit_iou_sum"],
        "prediction_instances": len(prediction_rles),
        "truth_instances_total": int(truth_counts.sum()),
        "truth_instances_mean": float(truth_counts.mean()),
        "truth_instances_min": int(truth_counts.min()),
        "truth_instances_max": int(truth_counts.max()),
        "consensus_false_positive_instances": len(consensus_fp_indices),
        "large_false_positive_instances": len(large_fp_areas),
        "largest_consensus_false_positive_area": (
            max(consensus_fp_areas) if consensus_fp_areas else None
        ),
        "small_instance_misses": small_missed,
        "fragmented_gt_instances": fragmented,
        "merged_prediction_instances": merged,
        "annotator_pq_mean": float(annotator_pq.mean()),
        "annotator_pq_std": float(annotator_pq.std()),
        "annotator_pq_min": float(annotator_pq.min()),
        "annotator_pq_max": float(annotator_pq.max()),
        "annotator_pq_range": float(np.ptp(annotator_pq)),
        "selected_annotator_image_id": selected_id,
        "small_instance_area_threshold": small_instance_area,
        "large_false_positive_area_threshold": large_false_positive_area,
        "diagnostic_iou_threshold": diagnostic_iou,
        "categories": categories,
        "annotators": annotators,
    }
    return row, labels, selected_masks, selected_matrix


def select_yolo_error_groups(
    rows: list[Mapping[str, Any]], top_k: int = 10
) -> dict[str, list[str]]:
    """Select deterministic best/worst and error-type image lists."""
    if top_k < 1:
        raise ValueError("top_k must be positive")

    def names(key, count: int | None = None) -> list[str]:
        return [str(row["file_name"]) for row in sorted(rows, key=key)[: count or top_k]]

    return {
        "worst_pq": names(
            lambda row: (
                row["official_pq"],
                -row["false_negatives"],
                -row["false_positives"],
                row["file_name"],
            )
        ),
        "best_pq": names(
            lambda row: (
                -row["official_pq"],
                row["false_positives"],
                row["false_negatives"],
                row["file_name"],
            ),
            min(5, top_k),
        ),
        "most_large_false_positives": names(
            lambda row: (
                -row["large_false_positive_instances"],
                -(row["largest_consensus_false_positive_area"] or 0),
                row["official_pq"],
                row["file_name"],
            )
        ),
        "most_small_instance_misses": names(
            lambda row: (
                -row["small_instance_misses"],
                row["official_pq"],
                row["file_name"],
            )
        ),
        "most_fragmented": names(
            lambda row: (
                -row["fragmented_gt_instances"],
                row["official_pq"],
                row["file_name"],
            )
        ),
        "most_merged": names(
            lambda row: (
                -row["merged_prediction_instances"],
                row["official_pq"],
                row["file_name"],
            )
        ),
        "largest_annotator_variation": names(
            lambda row: (
                -row["annotator_pq_range"],
                row["official_pq"],
                row["file_name"],
            )
        ),
    }


def _palette(size: int) -> np.ndarray:
    colors = np.zeros((max(size + 1, 2), 3), dtype=np.uint8)
    for index in range(1, len(colors)):
        hue = int((index * 47) % 180)
        hsv = np.uint8([[[hue, 210, 255]]])
        colors[index] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return colors


def _blend_labels(base: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = base.copy()
    colors = _palette(int(labels.max()))
    foreground = labels > 0
    if foreground.any():
        result[foreground] = (
            0.45 * result[foreground] + 0.55 * colors[labels[foreground]]
        ).astype(np.uint8)
        boundary = np.zeros(labels.shape, dtype=bool)
        for index in range(1, int(labels.max()) + 1):
            mask = labels == index
            boundary |= mask & ~ndimage.binary_erosion(mask)
        result[boundary] = 255
    return result


def _gt_panel(base: np.ndarray, truth_masks: list[np.ndarray]) -> np.ndarray:
    result = base.copy()
    colors = _palette(len(truth_masks))
    for index, mask in enumerate(truth_masks, start=1):
        boundary = mask & ~ndimage.binary_erosion(mask)
        result[ndimage.binary_dilation(boundary, iterations=2)] = colors[index]
    return result


def _error_panel(
    base: np.ndarray,
    labels: np.ndarray,
    truth_masks: list[np.ndarray],
    matrix: np.ndarray,
) -> np.ndarray:
    result = base.copy()
    hits = matrix > 0.5
    prediction_hit = hits.sum(axis=0) > 0
    truth_hit = hits.sum(axis=1) > 0
    for index in range(1, int(labels.max()) + 1):
        mask = labels == index
        color = np.array(
            [0, 220, 80] if prediction_hit[index - 1] else [245, 45, 45]
        )
        result[mask] = (0.45 * result[mask] + 0.55 * color).astype(np.uint8)
    for index, mask in enumerate(truth_masks):
        if truth_hit[index]:
            continue
        boundary = mask & ~ndimage.binary_erosion(mask)
        result[ndimage.binary_dilation(boundary, iterations=3)] = np.array(
            [40, 120, 255], dtype=np.uint8
        )
    return result


def render_yolo_error_canvas(
    image: np.ndarray,
    labels: np.ndarray,
    truth_masks: list[np.ndarray],
    selected_iou_matrix: np.ndarray,
    metrics: Mapping[str, Any],
    output_path: str | Path,
    *,
    display_size: int = 512,
) -> None:
    """Render original, selected GT, predictions, and TP/FP/FN panels."""
    if display_size < 128:
        raise ValueError("display_size must be at least 128")
    image = np.asarray(image)
    if image.ndim == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.float32)
        if float(gray.max(initial=0)) <= 1.0:
            gray = gray * 255
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    base = np.repeat(gray[..., None], 3, axis=2)
    panels = [
        base,
        _gt_panel(base, truth_masks),
        _blend_labels(base, labels),
        _error_panel(base, labels, truth_masks, selected_iou_matrix),
    ]
    titles = [
        "Original",
        "GT boundaries (selected annotator)",
        "Predicted instances",
        "TP green | FP red | FN blue",
    ]
    header_height, title_height = 48, 30
    canvas = Image.new(
        "RGB",
        (display_size * len(panels), header_height + title_height + display_size),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    header = (
        f"{metrics['file_name']}  official(all annotators): "
        f"PQ={metrics['official_pq']:.4f} TP/FP/FN="
        f"{metrics['hit_pairs']}/{metrics['false_positives']}/"
        f"{metrics['false_negatives']} pred/GTmean="
        f"{metrics['prediction_instances']}/{metrics['truth_instances_mean']:.1f}"
    )
    draw.text((8, 15), header, fill="white", font=font)
    for index, (panel, title) in enumerate(zip(panels, titles, strict=True)):
        resized = Image.fromarray(panel).resize(
            (display_size, display_size), Image.Resampling.LANCZOS
        )
        x = index * display_size
        canvas.paste(resized, (x, header_height + title_height))
        draw.text((x + 8, header_height + 9), title, fill="white", font=font)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
