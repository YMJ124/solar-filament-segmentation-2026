"""Immutable-parent candidate augmentation primitives for P9."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from src.metrics.pq import pairwise_rle_iou
from src.submission.rle import mask_to_rle


@dataclass(frozen=True)
class AdditiveCandidate:
    mask: np.ndarray
    original_mask: np.ndarray
    score: float
    source_index: int
    i1_rank: int
    maximum_t1_iou: float
    retained_area_fraction: float


def build_additive_candidates(
    t1_masks: Sequence[np.ndarray],
    i1_masks: Sequence[np.ndarray],
    i1_scores: Sequence[float],
    i1_source_indices: Sequence[int],
    *,
    min_area: int,
    max_instances: int,
    duplicate_iou: float,
) -> tuple[list[AdditiveCandidate], dict[str, int]]:
    """Return I1-only additions without modifying any T1 source instance."""
    if not (len(i1_masks) == len(i1_scores) == len(i1_source_indices)):
        raise ValueError("I1 masks, scores and source indices must align")
    if min_area < 1 or max_instances < 1 or not 0 <= duplicate_iou <= 1:
        raise ValueError("invalid additive-candidate policy")
    if not t1_masks and not i1_masks:
        return [], {"duplicates": 0, "area_rejects": 0, "cap_rejects": 0}
    shape = np.asarray((t1_masks or i1_masks)[0]).shape
    if len(shape) != 2:
        raise ValueError("candidate masks must be 2D")
    for mask in [*t1_masks, *i1_masks]:
        if np.asarray(mask).shape != shape:
            raise ValueError("all candidate masks must share a shape")

    t1_boolean = [np.asarray(mask, dtype=bool) for mask in t1_masks]
    t1_rles = [mask_to_rle(mask) for mask in t1_boolean]
    claimed = np.zeros(shape, dtype=bool)
    for mask in t1_boolean:
        claimed |= mask

    output: list[AdditiveCandidate] = []
    audit = {"duplicates": 0, "area_rejects": 0, "cap_rejects": 0}
    for rank, (raw_mask, score, source_index) in enumerate(
        zip(i1_masks, i1_scores, i1_source_indices, strict=True), start=1
    ):
        original = np.asarray(raw_mask, dtype=bool)
        if t1_rles:
            maximum_iou = float(
                pairwise_rle_iou(t1_rles, [mask_to_rle(original)]).max()
            )
        else:
            maximum_iou = 0.0
        if maximum_iou > duplicate_iou:
            audit["duplicates"] += 1
            continue
        if len(t1_boolean) + len(output) >= max_instances:
            audit["cap_rejects"] += 1
            continue
        available = original & ~claimed
        available_area = int(available.sum())
        if available_area < min_area:
            audit["area_rejects"] += 1
            continue
        original_area = int(original.sum())
        output.append(
            AdditiveCandidate(
                mask=available,
                original_mask=original,
                score=float(score),
                source_index=int(source_index),
                i1_rank=rank,
                maximum_t1_iou=maximum_iou,
                retained_area_fraction=available_area / original_area,
            )
        )
        claimed |= available
    return output, audit


def marginal_counts_and_utility(
    base_iou_matrices: Sequence[np.ndarray],
    candidate_iou_vectors: Sequence[np.ndarray],
    *,
    reference_pq: float,
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Measure one candidate's exact marginal terms against an immutable base."""
    if len(base_iou_matrices) != len(candidate_iou_vectors):
        raise ValueError("base matrices and candidate vectors must align")
    delta_hits = 0
    delta_fp = 0
    delta_fn = 0
    delta_iou = 0.0
    for base, vector in zip(
        base_iou_matrices, candidate_iou_vectors, strict=True
    ):
        base = np.asarray(base, dtype=np.float64)
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        if base.ndim != 2 or base.shape[0] != len(vector):
            raise ValueError("candidate IoUs do not match the truth dimension")
        hits = vector > iou_threshold
        hit_count = int(hits.sum())
        delta_hits += hit_count
        delta_iou += float(vector[hits].sum())
        if not hit_count:
            delta_fp += 1
        base_matched = (base > iou_threshold).any(axis=1)
        delta_fn -= int((hits & ~base_matched).sum())
    delta_denominator = delta_hits + 0.5 * delta_fp + 0.5 * delta_fn
    utility = delta_iou - float(reference_pq) * delta_denominator
    return {
        "delta_hit_pairs": delta_hits,
        "delta_false_positives": delta_fp,
        "delta_false_negatives": delta_fn,
        "delta_hit_iou_sum": delta_iou,
        "delta_denominator": delta_denominator,
        "marginal_utility": utility,
        "positive_utility": bool(utility > 0),
    }


FEATURE_NAMES = (
    "i1_confidence",
    "i1_rank_fraction",
    "log_area",
    "area_fraction_image",
    "bbox_width_fraction",
    "bbox_height_fraction",
    "log_bbox_aspect",
    "bbox_fill_fraction",
    "perimeter_over_sqrt_area",
    "component_count",
    "centroid_x_fraction",
    "centroid_y_fraction",
    "centroid_radius",
    "maximum_t1_output_iou",
    "retained_area_fraction",
    "maximum_raw_t1_iou",
    "raw_t1_best_confidence",
    "raw_t1_support_iou_gt_0_25",
    "raw_t1_support_iou_gt_0_50",
    "t1_output_count",
    "i1_output_count",
    "inside_intensity_mean",
    "inside_intensity_std",
    "inside_intensity_q10",
    "inside_intensity_q50",
    "inside_intensity_q90",
    "inside_minus_image_mean",
    "minimum_distance_to_t1",
    "median_distance_to_t1",
)


def extract_candidate_features(
    candidate: AdditiveCandidate,
    *,
    image: np.ndarray,
    i1_output_count: int,
    t1_output_count: int,
    raw_t1_rles: Sequence[dict[str, Any]],
    raw_t1_scores: Sequence[float],
    distance_to_t1: np.ndarray,
) -> np.ndarray:
    """Compute inference-only mask, image and cross-model support features."""
    mask = np.asarray(candidate.mask, dtype=bool)
    image = np.asarray(image)
    if mask.ndim != 2 or image.shape != mask.shape:
        raise ValueError("candidate mask and grayscale image must share a 2D shape")
    if np.asarray(distance_to_t1).shape != mask.shape:
        raise ValueError("distance map shape differs from candidate mask")
    if not mask.any():
        raise ValueError("candidate mask must be non-empty")
    height, width = mask.shape
    area = int(mask.sum())
    ys, xs = np.nonzero(mask)
    box_width = int(xs.max() - xs.min() + 1)
    box_height = int(ys.max() - ys.min() + 1)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    component_count = int(
        cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1
    )
    center_x = float(xs.mean() / max(width - 1, 1))
    center_y = float(ys.mean() / max(height - 1, 1))
    radius = float(np.hypot(center_x - 0.5, center_y - 0.5))

    candidate_rle = mask_to_rle(mask)
    if raw_t1_rles:
        raw_ious = pairwise_rle_iou(list(raw_t1_rles), [candidate_rle])[:, 0]
        best_index = int(np.argmax(raw_ious))
        maximum_raw_iou = float(raw_ious[best_index])
        best_raw_confidence = float(raw_t1_scores[best_index])
        support_025 = int(np.count_nonzero(raw_ious > 0.25))
        support_050 = int(np.count_nonzero(raw_ious > 0.50))
    else:
        maximum_raw_iou = best_raw_confidence = 0.0
        support_025 = support_050 = 0

    pixels = image[mask].astype(np.float64) / 255.0
    distances = np.asarray(distance_to_t1, dtype=np.float64)[mask]
    values = np.asarray(
        [
            candidate.score,
            candidate.i1_rank / max(i1_output_count, 1),
            np.log1p(area),
            area / (height * width),
            box_width / width,
            box_height / height,
            np.log((box_width + 1.0) / (box_height + 1.0)),
            area / (box_width * box_height),
            perimeter / np.sqrt(area),
            component_count,
            center_x,
            center_y,
            radius,
            candidate.maximum_t1_iou,
            candidate.retained_area_fraction,
            maximum_raw_iou,
            best_raw_confidence,
            support_025,
            support_050,
            t1_output_count,
            i1_output_count,
            pixels.mean(),
            pixels.std(),
            *np.quantile(pixels, [0.1, 0.5, 0.9]),
            pixels.mean() - float(image.mean() / 255.0),
            distances.min(),
            np.median(distances),
        ],
        dtype=np.float64,
    )
    if len(values) != len(FEATURE_NAMES) or not np.isfinite(values).all():
        raise ValueError("candidate feature vector is invalid")
    return values
