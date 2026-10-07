"""Per-image OOF error diagnostics and dependency-light visualization helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from src.data.coco import CocoAnnotations
from src.data.masks import annotations_to_masks, annotations_to_rles
from src.metrics.pq import aggregate_counts, pairwise_rle_iou, score_iou_matrix
from src.metrics.segmentation import segmentation_metrics
from src.oof_postprocess import PostprocessConfig
from src.postprocess.instances import probability_to_instance_labels
from src.submission.rle import label_map_to_rles


CATEGORY_LABELS = {
    "low_contrast_miss": "低对比度漏检",
    "small_target_miss": "小目标漏检",
    "structure_middle_break": "结构中部断裂",
    "dark_background_false_positive": "背景暗区误检",
    "tile_seam": "tile 接缝",
    "instance_adhesion": "实例粘连",
    "annotator_disagreement": "annotator 分歧",
}


def _mask_contrast(image: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    ring = ndimage.binary_dilation(mask, iterations=5) & ~mask
    if not ring.any():
        return None
    return float(abs(float(image[mask].mean()) - float(image[ring].mean())))


def _semantic_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    if not union:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def _seam_mask(
    shape: tuple[int, int], tile_size: int, overlap: float, tolerance: int = 6
) -> np.ndarray:
    height, width = shape
    step = max(1, int(tile_size * (1.0 - overlap)))

    def starts(length: int) -> list[int]:
        if tile_size >= length:
            return [0]
        values = list(range(0, length - tile_size + 1, step))
        last = length - tile_size
        if values[-1] != last:
            values.append(last)
        return values

    mask = np.zeros(shape, dtype=bool)
    for value in starts(height)[1:]:
        mask[max(0, value - tolerance) : min(height, value + tolerance + 1), :] = True
    for value in starts(width)[1:]:
        mask[:, max(0, value - tolerance) : min(width, value + tolerance + 1)] = True
    return mask


def classify_error_categories(metrics: Mapping[str, Any]) -> list[str]:
    """Assign deterministic, explicitly thresholded diagnostic categories."""
    categories = []
    if (
        int(metrics["false_negatives"]) > 0
        and metrics.get("minimum_missed_gt_contrast") is not None
        and float(metrics["minimum_missed_gt_contrast"]) <= 0.08
    ):
        categories.append("low_contrast_miss")
    if (
        metrics.get("minimum_missed_gt_area") is not None
        and int(metrics["minimum_missed_gt_area"]) < int(metrics["min_area_threshold"])
    ):
        categories.append("small_target_miss")
    if int(metrics["fragmented_gt_instances"]) > 0:
        categories.append("structure_middle_break")
    if (
        int(metrics["false_positives"]) > 0
        and metrics.get("minimum_medoid_fp_mean_intensity") is not None
        and float(metrics["minimum_medoid_fp_mean_intensity"])
        <= float(metrics["image_intensity_p25"])
    ):
        categories.append("dark_background_false_positive")
    if (
        int(metrics["medoid_error_pixels"]) >= 20
        and float(metrics["tile_seam_error_fraction"]) >= 0.25
    ):
        categories.append("tile_seam")
    if int(metrics["merged_prediction_instances"]) > 0:
        categories.append("instance_adhesion")
    if (
        int(metrics["annotator_batches"]) > 1
        and metrics.get("annotator_mean_pairwise_semantic_iou") is not None
        and float(metrics["annotator_mean_pairwise_semantic_iou"]) < 0.5
    ):
        categories.append("annotator_disagreement")
    return categories


def analyze_oof_image(
    probability: np.ndarray,
    image: np.ndarray,
    record: Mapping[str, Any],
    coco: CocoAnnotations,
    config: PostprocessConfig,
    tile_size: int = 256,
    overlap: float = 0.25,
) -> tuple[dict[str, Any], np.ndarray, list[np.ndarray], np.ndarray]:
    """Score one physical image across all annotators and derive error signals."""
    file_name = str(record["file_name"])
    labels = probability_to_instance_labels(
        probability,
        threshold=config.threshold,
        min_area=config.min_area,
        closing_iterations=config.closing_iterations,
        use_watershed=False,
        max_area_fraction=config.max_area_fraction,
        max_instances=config.max_instances,
    )
    prediction_rles = label_map_to_rles(labels)
    counts = []
    pixel_rows = []
    gt_degrees: list[int] = []
    prediction_degrees: list[int] = []
    missed_areas: list[int] = []
    missed_contrasts: list[float] = []
    semantic_masks: list[np.ndarray] = []
    selected_masks: list[np.ndarray] | None = None
    selected_matrix: np.ndarray | None = None
    selected_id = str(record["selected_image_id"])

    for batch in coco.batches_for_file(file_name):
        annotations = coco.annotations_for_batch(str(batch["id"]))
        truth_rles = annotations_to_rles(
            annotations, int(batch["height"]), int(batch["width"])
        )
        truth_masks = annotations_to_masks(
            annotations, int(batch["height"]), int(batch["width"])
        )
        matrix = pairwise_rle_iou(truth_rles, prediction_rles)
        counts.append(
            score_iou_matrix(
                matrix,
                n_truth=len(truth_rles),
                n_prediction=len(prediction_rles),
            )
        )
        overlap_matrix = matrix > 0
        gt_degrees.extend(overlap_matrix.sum(axis=1).astype(int).tolist())
        prediction_degrees.extend(overlap_matrix.sum(axis=0).astype(int).tolist())
        hits = matrix > 0.5
        missed_indices = np.flatnonzero(hits.sum(axis=1) == 0)
        for index in missed_indices:
            mask = truth_masks[int(index)]
            missed_areas.append(int(mask.sum()))
            contrast = _mask_contrast(image, mask)
            if contrast is not None:
                missed_contrasts.append(contrast)
        semantic = (
            np.logical_or.reduce(truth_masks)
            if truth_masks
            else np.zeros(image.shape, dtype=bool)
        )
        semantic_masks.append(semantic)
        pixel_rows.append(segmentation_metrics(semantic, labels > 0))
        if str(batch["id"]) == selected_id:
            selected_masks = truth_masks
            selected_matrix = matrix

    if selected_masks is None or selected_matrix is None:
        raise ValueError(f"selected medoid batch {selected_id!r} missing for {file_name}")

    report = aggregate_counts(counts, physical_images=1).to_dict()
    medoid_hits = selected_matrix > 0.5
    medoid_fp_indices = np.flatnonzero(medoid_hits.sum(axis=0) == 0) + 1
    medoid_fn_indices = np.flatnonzero(medoid_hits.sum(axis=1) == 0)
    fp_intensities = [
        float(image[labels == int(index)].mean())
        for index in medoid_fp_indices
        if np.any(labels == int(index))
    ]
    error_mask = np.zeros(image.shape, dtype=bool)
    for index in medoid_fp_indices:
        error_mask |= labels == int(index)
    for index in medoid_fn_indices:
        error_mask |= selected_masks[int(index)]
    seam = _seam_mask(image.shape, tile_size=tile_size, overlap=overlap)
    error_pixels = int(error_mask.sum())
    seam_fraction = (
        float(np.logical_and(error_mask, seam).sum() / error_pixels)
        if error_pixels
        else 0.0
    )
    pairwise_semantic_ious = [
        _semantic_iou(semantic_masks[first], semantic_masks[second])
        for first in range(len(semantic_masks))
        for second in range(first + 1, len(semantic_masks))
    ]
    fragmented_gt = sum(value > 1 for value in gt_degrees)
    merged_predictions = sum(value > 1 for value in prediction_degrees)
    row: dict[str, Any] = {
        "file_name": file_name,
        "fold": int(record["fold"]),
        "annotator_batches": len(counts),
        "official_pq": report["official_pq"],
        "hit_pairs": report["hit_pairs"],
        "false_positives": report["false_positives"],
        "false_negatives": report["false_negatives"],
        "hit_iou_sum": report["hit_iou_sum"],
        "dice_annotator_macro": float(np.mean([value["dice"] for value in pixel_rows])),
        "iou_annotator_macro": float(np.mean([value["iou"] for value in pixel_rows])),
        "prediction_instances": len(prediction_rles),
        "fragmented_gt_instances": int(fragmented_gt),
        "merged_prediction_instances": int(merged_predictions),
        "medoid_hit_pairs": int(medoid_hits.sum()),
        "medoid_false_positives": int(len(medoid_fp_indices)),
        "medoid_false_negatives": int(len(medoid_fn_indices)),
        "minimum_missed_gt_area": min(missed_areas) if missed_areas else None,
        "minimum_missed_gt_contrast": (
            min(missed_contrasts) if missed_contrasts else None
        ),
        "minimum_medoid_fp_mean_intensity": (
            min(fp_intensities) if fp_intensities else None
        ),
        "image_intensity_p25": float(np.quantile(image, 0.25)),
        "medoid_error_pixels": error_pixels,
        "tile_seam_error_fraction": seam_fraction,
        "annotator_mean_pairwise_semantic_iou": (
            float(np.mean(pairwise_semantic_ious))
            if pairwise_semantic_ious
            else None
        ),
        "min_area_threshold": config.min_area,
        "selected_medoid_image_id": selected_id,
    }
    row["categories"] = classify_error_categories(row)
    return row, labels, selected_masks, selected_matrix


def select_error_groups(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Select deterministic extreme examples for the five requested groups."""
    return {
        "lowest_pq": [
            row["file_name"]
            for row in sorted(
                rows,
                key=lambda row: (
                    row["official_pq"],
                    -row["false_negatives"],
                    -row["false_positives"],
                    row["file_name"],
                ),
            )[:10]
        ],
        "most_false_positives": [
            row["file_name"]
            for row in sorted(
                rows,
                key=lambda row: (
                    -row["false_positives"],
                    row["official_pq"],
                    row["file_name"],
                ),
            )[:10]
        ],
        "most_false_negatives": [
            row["file_name"]
            for row in sorted(
                rows,
                key=lambda row: (
                    -row["false_negatives"],
                    row["official_pq"],
                    row["file_name"],
                ),
            )[:10]
        ],
        "most_fragmented": [
            row["file_name"]
            for row in sorted(
                rows,
                key=lambda row: (
                    -row["fragmented_gt_instances"],
                    row["official_pq"],
                    row["file_name"],
                ),
            )[:10]
        ],
        "best_pq": [
            row["file_name"]
            for row in sorted(
                rows,
                key=lambda row: (
                    -row["official_pq"],
                    row["false_positives"],
                    row["false_negatives"],
                    row["file_name"],
                ),
            )[:5]
        ],
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


def _gt_boundary_panel(base: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    result = base.copy()
    colors = _palette(len(masks))
    for index, mask in enumerate(masks, start=1):
        boundary = mask & ~ndimage.binary_erosion(mask)
        thick = ndimage.binary_dilation(boundary, iterations=2)
        result[thick] = colors[index]
    return result


def _error_panel(
    base: np.ndarray,
    labels: np.ndarray,
    truth_masks: list[np.ndarray],
    iou_matrix: np.ndarray,
) -> np.ndarray:
    result = base.copy()
    hits = iou_matrix > 0.5
    prediction_hit = hits.sum(axis=0) > 0
    truth_hit = hits.sum(axis=1) > 0
    for index in range(1, int(labels.max()) + 1):
        mask = labels == index
        color = np.array([0, 220, 80] if prediction_hit[index - 1] else [245, 45, 45])
        result[mask] = (0.45 * result[mask] + 0.55 * color).astype(np.uint8)
    for index, mask in enumerate(truth_masks):
        if truth_hit[index]:
            continue
        boundary = mask & ~ndimage.binary_erosion(mask)
        thick = ndimage.binary_dilation(boundary, iterations=3)
        result[thick] = np.array([40, 120, 255], dtype=np.uint8)
    return result


def render_error_canvas(
    image: np.ndarray,
    probability: np.ndarray,
    labels: np.ndarray,
    truth_masks: list[np.ndarray],
    medoid_iou_matrix: np.ndarray,
    metrics: Mapping[str, Any],
    output_path: str | Path,
    display_size: int = 512,
) -> None:
    """Render original, GT, heatmap, instances, and TP/FP/FN without plotting libs."""
    if display_size < 128:
        raise ValueError("display_size must be at least 128")
    gray = np.clip(image * 255, 0, 255).astype(np.uint8)
    base = np.repeat(gray[..., None], 3, axis=2)
    heat = cv2.applyColorMap(
        np.clip(probability * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = (0.35 * base + 0.65 * heat).astype(np.uint8)
    panels = [
        base,
        _gt_boundary_panel(base, truth_masks),
        heat,
        _blend_labels(base, labels),
        _error_panel(base, labels, truth_masks, medoid_iou_matrix),
    ]
    titles = [
        "Original",
        "GT boundaries (medoid)",
        "Probability heatmap",
        "Predicted instances",
        "TP green | FP red | FN blue",
    ]
    title_height = 30
    header_height = 42
    canvas = Image.new(
        "RGB", (display_size * len(panels), header_height + title_height + display_size), "black"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    header = (
        f"{metrics['file_name']}  official(all annotators): "
        f"PQ={metrics['official_pq']:.4f} TP/FP/FN="
        f"{metrics['hit_pairs']}/{metrics['false_positives']}/{metrics['false_negatives']}"
    )
    draw.text((8, 12), header, fill="white", font=font)
    for index, (panel, title) in enumerate(zip(panels, titles)):
        resized = Image.fromarray(panel).resize(
            (display_size, display_size), Image.Resampling.LANCZOS
        )
        x = index * display_size
        canvas.paste(resized, (x, header_height + title_height))
        draw.text((x + 8, header_height + 9), title, fill="white", font=font)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def category_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(
        category for row in rows for category in row.get("categories", [])
    )
    return {key: int(counts.get(key, 0)) for key in CATEGORY_LABELS}
