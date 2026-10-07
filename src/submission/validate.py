"""Validation helpers for predicted masks and submission rows."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .rle import rle_from_counts


def audit_image_masks(image_id: str, masks: list[np.ndarray], shape=(2048, 2048)) -> dict:
    occupied = np.zeros(shape, dtype=bool)
    areas, overlaps = [], 0
    for index, raw in enumerate(masks, start=1):
        mask = np.asarray(raw, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"{image_id}_{index}: mask shape {mask.shape} != {shape}")
        area = int(mask.sum())
        if area == 0:
            raise ValueError(f"{image_id}_{index}: empty mask")
        overlaps += int(np.logical_and(occupied, mask).sum())
        occupied |= mask
        areas.append(area)
    if overlaps:
        raise ValueError(f"{image_id}: predicted instances overlap by {overlaps} pixels")
    return {"image_id": image_id, "instances": len(masks), "min_area": min(areas) if areas else 0,
            "max_area": max(areas) if areas else 0, "foreground_pixels": int(occupied.sum())}


def audit_image_labels(
    image_id: str, labels: np.ndarray, shape=(2048, 2048)
) -> dict:
    """Audit a mutually exclusive label map without expanding every instance."""
    labels = np.asarray(labels)
    if labels.shape != shape:
        raise ValueError(f"{image_id}: label shape {labels.shape} != {shape}")
    if labels.ndim != 2 or np.any(labels < 0):
        raise ValueError(f"{image_id}: labels must be a non-negative 2D array")
    unique = np.unique(labels)
    positive = unique[unique > 0]
    expected = np.arange(1, len(positive) + 1)
    if not np.array_equal(positive, expected):
        raise ValueError(f"{image_id}: positive labels must be contiguous from 1")
    areas = np.bincount(labels.ravel())[1:]
    if np.any(areas == 0):
        raise ValueError(f"{image_id}: label map contains empty instance IDs")
    return {
        "image_id": image_id,
        "instances": int(len(areas)),
        "min_area": int(areas.min()) if len(areas) else 0,
        "max_area": int(areas.max()) if len(areas) else 0,
        "foreground_pixels": int((labels > 0).sum()),
    }


def audit_submission_rows(rows: list[dict], expected_image_ids: set[str],
                          shape=(2048, 2048), validate_rles: bool = True) -> dict:
    """Validate competition IDs and, by default, every compressed COCO RLE."""
    ids = [str(row.get("filament_id", "")) for row in rows]
    if any(not value for value in ids):
        raise ValueError("filament_id values must be non-empty")
    if len(ids) != len(set(ids)):
        raise ValueError("filament_id values must be unique")
    if any(not row.get("segmentation_rle") for row in rows):
        raise ValueError("segmentation_rle values must be non-empty")
    parsed = [value.rsplit("_", 1) for value in ids]
    if any(len(parts) != 2 or not parts[0] or not parts[1] for parts in parsed):
        raise ValueError("filament_id must be '<test-image-stem>_<unique-tail>'")
    image_ids = [parts[0] for parts in parsed]
    unknown = sorted(set(image_ids) - expected_image_ids)
    if unknown:
        raise ValueError(f"unknown image ids: {unknown[:5]}")
    counts = Counter(image_ids)
    areas = []
    total_foreground = 0
    if validate_rles:
        from pycocotools import mask as mask_utils

        by_image = defaultdict(list)
        for image_id, row in zip(image_ids, rows):
            by_image[image_id].append(str(row["segmentation_rle"]))
        for image_id, encoded_masks in by_image.items():
            rles = []
            image_areas = []
            for index, encoded in enumerate(encoded_masks, start=1):
                try:
                    rle = rle_from_counts(
                        encoded, height=shape[0], width=shape[1]
                    )
                except Exception as exc:
                    raise ValueError(f"{image_id}_{index}: invalid RLE") from exc
                area = int(mask_utils.area(rle))
                if area == 0:
                    raise ValueError(f"{image_id}_{index}: empty RLE mask")
                rles.append(rle)
                image_areas.append(area)
            areas.extend(image_areas)
            image_foreground = sum(image_areas)
            total_foreground += image_foreground
            union_area = int(mask_utils.area(mask_utils.merge(rles)))
            overlap = image_foreground - union_area
            if overlap:
                raise ValueError(
                    f"{image_id}: submitted instances overlap by {overlap} pixels"
                )
    return {
        "rows": len(rows),
        "expected_images": len(expected_image_ids),
        "covered_images": len(counts),
        "missing_images": len(expected_image_ids - set(image_ids)),
        "max_instances_per_image": max(counts.values(), default=0),
        "min_instance_area": min(areas) if areas else None,
        "max_instance_area": max(areas) if areas else None,
        "total_foreground_pixels": total_foreground if validate_rles else None,
        "rles_validated": bool(validate_rles),
    }
