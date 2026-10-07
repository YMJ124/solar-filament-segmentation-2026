"""COCO compressed RLE helpers and submission CSV writer."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np


def _mask_utils():
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError(
            "pycocotools is required for RLE encoding; install requirements.txt first"
        ) from exc
    return mask_utils


def mask_to_rle(mask: np.ndarray) -> dict:
    """Encode a binary mask as an in-memory COCO RLE object."""
    mask_utils = _mask_utils()
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    return mask_utils.encode(np.asfortranarray(mask))


def label_map_to_rles(labels: np.ndarray) -> list[dict]:
    """Encode a compact label map while holding only one dense mask at a time."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or np.any(labels < 0):
        raise ValueError("labels must be a non-negative 2D array")
    return [
        mask_to_rle(labels == index)
        for index in range(1, int(labels.max()) + 1)
    ]


def rle_counts(rle: dict) -> str:
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)


def rle_from_counts(counts: str, height: int = 2048, width: int = 2048) -> dict:
    """Build and validate the shape-independent RLE stored in a submission row."""
    if not isinstance(counts, str) or not counts:
        raise ValueError("RLE counts must be a non-empty string")
    try:
        encoded = {"size": [int(height), int(width)], "counts": counts.encode("ascii")}
    except UnicodeEncodeError as exc:
        raise ValueError("RLE counts must contain ASCII characters only") from exc
    try:
        _mask_utils().area(encoded)
    except Exception as exc:
        raise ValueError("invalid compressed COCO RLE counts") from exc
    return encoded


def encode_mask(mask: np.ndarray) -> str:
    return rle_counts(mask_to_rle(mask))


def decode_rle(counts: str, height: int = 2048, width: int = 2048) -> np.ndarray:
    mask_utils = _mask_utils()
    encoded = rle_from_counts(counts, height, width)
    return np.asarray(mask_utils.decode(encoded), dtype=bool)


def write_submission(rows: Iterable[dict], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    ids = [row["filament_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("filament_id values must be unique")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["filament_id", "segmentation_rle"]
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return output_path
