"""P5 crop-refiner primitives with fixed T1 candidate semantics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.instance.yolo_evaluation import YoloPostprocessConfig
from src.models import RefinerUNet
from src.submission.rle import label_map_to_rles


def postprocessed_instances_from_rles(
    rles: Sequence[dict[str, Any]],
    scores: np.ndarray,
    *,
    height: int,
    width: int,
    config: YoloPostprocessConfig,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Reproduce T1 postprocess while retaining instance scores/source indices."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pycocotools is required for P5 refiner data") from exc
    config.validate()
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(rles) != len(scores):
        raise ValueError("candidate RLE and score counts differ")
    if not len(rles):
        return [], np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    retained = order[scores[order] >= config.confidence]
    if not len(retained) and config.force_one:
        retained = order[:1]
    retained = retained[: config.max_instances]
    if not len(retained):
        return [], np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    decoded = np.asarray(mask_utils.decode([rles[int(index)] for index in retained]))
    if decoded.ndim == 2:
        decoded = decoded[:, :, None]
    masks = np.moveaxis(decoded, -1, 0).astype(bool, copy=False)
    if masks.shape[1:] != (height, width):
        raise ValueError("decoded candidate shape differs from cache metadata")
    claimed = np.zeros((height, width), dtype=bool)
    output_masks: list[np.ndarray] = []
    output_scores = []
    output_indices = []
    for local_index, source_index in enumerate(retained):
        available = masks[local_index] & ~claimed
        if int(available.sum()) < config.min_area:
            continue
        output_masks.append(available)
        output_scores.append(float(scores[source_index]))
        output_indices.append(int(source_index))
        claimed |= available
    return (
        output_masks,
        np.asarray(output_scores, dtype=np.float64),
        np.asarray(output_indices, dtype=np.int64),
    )


def instances_to_label_map(
    masks: Sequence[np.ndarray], *, height: int, width: int
) -> np.ndarray:
    labels = np.zeros((height, width), dtype=np.int32)
    for index, mask in enumerate(masks, start=1):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != labels.shape:
            raise ValueError("instance mask shape differs from label map")
        labels[mask & (labels == 0)] = index
    return labels


def crop_bounds(
    mask: np.ndarray,
    *,
    context_scale: float,
    minimum_side: int,
) -> tuple[int, int, int, int]:
    """Return clipped ``x0,y0,x1,y1`` bounds around one non-empty mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("crop mask must be a non-empty 2D array")
    if context_scale < 1 or minimum_side < 1:
        raise ValueError("context_scale and minimum_side must be >= 1")
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    box_width = int(xs.max() - xs.min() + 1)
    box_height = int(ys.max() - ys.min() + 1)
    crop_width = min(width, max(minimum_side, int(np.ceil(box_width * context_scale))))
    crop_height = min(height, max(minimum_side, int(np.ceil(box_height * context_scale))))
    center_x = 0.5 * (int(xs.min()) + int(xs.max()) + 1)
    center_y = 0.5 * (int(ys.min()) + int(ys.max()) + 1)
    x0 = int(round(center_x - crop_width / 2))
    y0 = int(round(center_y - crop_height / 2))
    x0 = min(max(0, x0), width - crop_width)
    y0 = min(max(0, y0), height - crop_height)
    return x0, y0, x0 + crop_width, y0 + crop_height


def resize_refiner_crop(
    image: np.ndarray,
    coarse_mask: np.ndarray,
    target_mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    *,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop and anisotropically resize image/coarse/target to the model size."""
    if output_size < 32 or output_size % 16:
        raise ValueError("output_size must be >=32 and divisible by 16")
    x0, y0, x1, y1 = bounds
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError("crop bounds lie outside image")
    if image.shape != coarse_mask.shape or image.shape != target_mask.shape:
        raise ValueError("image and masks must share shape")
    image_crop = image[y0:y1, x0:x1]
    coarse_crop = np.asarray(coarse_mask[y0:y1, x0:x1], dtype=np.uint8)
    target_crop = np.asarray(target_mask[y0:y1, x0:x1], dtype=np.uint8)
    size = (output_size, output_size)
    return (
        cv2.resize(image_crop, size, interpolation=cv2.INTER_LINEAR),
        cv2.resize(coarse_crop, size, interpolation=cv2.INTER_NEAREST),
        cv2.resize(target_crop, size, interpolation=cv2.INTER_NEAREST),
    )


def restore_refined_mask(
    probability: np.ndarray,
    bounds: tuple[int, int, int, int],
    *,
    height: int,
    width: int,
    threshold: float,
) -> np.ndarray:
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim != 2:
        raise ValueError("refiner probability must be 2D")
    x0, y0, x1, y1 = bounds
    restored = cv2.resize(
        probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR
    )
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = restored >= threshold
    return mask


def fuse_refinement_in_boundary_band(
    coarse_mask: np.ndarray,
    refined_mask: np.ndarray,
    resized_coarse_crop: np.ndarray,
    bounds: tuple[int, int, int, int],
    *,
    boundary_kernel: int,
) -> np.ndarray:
    """Use refinement only near the coarse boundary and preserve deep interior.

    The band is defined in the refiner's resized crop coordinates so the fixed
    kernel has exactly the same coordinate meaning as the training loss.  The
    band is mapped back with nearest-neighbour interpolation; the refined mask
    itself keeps the existing bilinear-restore/threshold semantics.
    """
    coarse_mask = np.asarray(coarse_mask, dtype=bool)
    refined_mask = np.asarray(refined_mask, dtype=bool)
    resized_coarse_crop = np.asarray(resized_coarse_crop, dtype=np.uint8)
    if coarse_mask.ndim != 2 or refined_mask.shape != coarse_mask.shape:
        raise ValueError("coarse and refined masks must share one 2D shape")
    if resized_coarse_crop.ndim != 2:
        raise ValueError("resized coarse crop must be 2D")
    if boundary_kernel < 1 or boundary_kernel % 2 != 1:
        raise ValueError("boundary_kernel must be a positive odd integer")
    x0, y0, x1, y1 = bounds
    height, width = coarse_mask.shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("crop bounds lie outside mask")
    kernel = np.ones((boundary_kernel, boundary_kernel), dtype=np.uint8)
    binary_crop = (resized_coarse_crop > 0).astype(np.uint8, copy=False)
    dilated = cv2.dilate(
        binary_crop,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    eroded = cv2.erode(
        binary_crop,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    band_crop = (dilated != eroded).astype(np.uint8)
    band = np.zeros_like(coarse_mask)
    band[y0:y1, x0:x1] = cv2.resize(
        band_crop,
        (x1 - x0, y1 - y0),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    fused = coarse_mask.copy()
    fused[band] = refined_mask[band]
    return fused


def combine_refined_instances(
    masks: Sequence[np.ndarray],
    *,
    height: int,
    width: int,
    min_area: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply original score order, exclusivity and minimum area after refinement."""
    if min_area < 1:
        raise ValueError("min_area must be positive")
    claimed = np.zeros((height, width), dtype=bool)
    kept = []
    for mask in masks:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != claimed.shape:
            raise ValueError("refined mask shape differs from image")
        available = mask & ~claimed
        if int(available.sum()) < min_area:
            continue
        kept.append(available)
        claimed |= available
    labels = instances_to_label_map(kept, height=height, width=width)
    return labels, label_map_to_rles(labels)


def combine_support_preserving_instances(
    masks: Sequence[np.ndarray],
    coarse_masks: Sequence[np.ndarray],
    *,
    height: int,
    width: int,
    min_area: int,
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    """Combine a reversible refinement branch without deleting source slots.

    Other instances' coarse support is reserved before score-order composition.
    If a fused source becomes too small, only that source falls back to its
    immutable coarse mask.  This makes output instance count preservation an
    invariant rather than a tuned score heuristic.
    """
    if len(masks) != len(coarse_masks):
        raise ValueError("refined and coarse instance counts differ")
    if min_area < 1:
        raise ValueError("min_area must be positive")
    shape = (height, width)
    normalized_coarse = []
    for mask in coarse_masks:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError("coarse mask shape differs from image")
        if int(mask.sum()) < min_area:
            raise ValueError("source coarse mask is smaller than min_area")
        normalized_coarse.append(mask)
    protected = np.zeros(shape, dtype=bool)
    for coarse in normalized_coarse:
        if np.any(protected & coarse):
            raise ValueError("source coarse masks must be mutually exclusive")
        protected |= coarse

    claimed = np.zeros(shape, dtype=bool)
    kept = []
    fallback_instances = 0
    for mask, coarse in zip(masks, normalized_coarse):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError("refined mask shape differs from image")
        other_source_support = protected & ~coarse
        available = mask & ~other_source_support & ~claimed
        if int(available.sum()) < min_area:
            available = coarse & ~claimed
            fallback_instances += 1
        if int(available.sum()) < min_area:
            raise RuntimeError("source-support invariant could not be restored")
        kept.append(available)
        claimed |= available
    labels = instances_to_label_map(kept, height=height, width=width)
    rles = label_map_to_rles(labels)
    if len(rles) != len(coarse_masks):
        raise RuntimeError("source-support invariant changed instance count")
    return labels, rles, fallback_instances


class BoundaryWeightedDiceLoss(nn.Module):
    """BCE + Dice with a fixed target-boundary BCE weight map."""

    def __init__(self, boundary_weight: float = 4.0, boundary_kernel: int = 7):
        super().__init__()
        if boundary_weight < 0:
            raise ValueError("boundary_weight must be non-negative")
        if boundary_kernel < 1 or boundary_kernel % 2 != 1:
            raise ValueError("boundary_kernel must be a positive odd integer")
        self.boundary_weight = float(boundary_weight)
        self.boundary_kernel = int(boundary_kernel)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(dtype=logits.dtype)
        if logits.shape != target.shape:
            raise ValueError("logits and target must share shape")
        pad = self.boundary_kernel // 2
        dilated = F.max_pool2d(target, self.boundary_kernel, stride=1, padding=pad)
        eroded = -F.max_pool2d(-target, self.boundary_kernel, stride=1, padding=pad)
        boundary = (dilated - eroded).clamp(0, 1)
        weights = 1.0 + self.boundary_weight * boundary
        bce = F.binary_cross_entropy_with_logits(
            logits, target, weight=weights, reduction="none"
        ).flatten(1).mean(1)
        probability = torch.sigmoid(logits)
        intersection = (probability * target).flatten(1).sum(1)
        denominator = probability.flatten(1).sum(1) + target.flatten(1).sum(1)
        dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        return (bce + dice).mean()


class RefinerCropDataset(Dataset):
    """Dataset backed by compact uint8 NPZ crops built from OOF T1 candidates."""

    def __init__(self, dataset_dir: str | Path, *, augment: bool):
        self.dataset_dir = Path(dataset_dir)
        self.index = json.loads((self.dataset_dir / "index.json").read_text())
        self.records = list(self.index.get("samples") or [])
        if not self.records:
            raise ValueError("refiner crop dataset contains no samples")
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with np.load(self.dataset_dir / record["sample_file"]) as payload:
            image = np.asarray(payload["image"], dtype=np.float32) / 255.0
            coarse = np.asarray(payload["coarse"], dtype=np.float32)
            target = np.asarray(payload["target"], dtype=np.float32)
        if self.augment:
            if bool(torch.rand(()) < 0.5):
                image, coarse, target = (
                    np.flip(array, axis=1).copy() for array in (image, coarse, target)
                )
            if bool(torch.rand(()) < 0.5):
                image, coarse, target = (
                    np.flip(array, axis=0).copy() for array in (image, coarse, target)
                )
        inputs = np.stack([image, coarse]).astype(np.float32, copy=False)
        return {
            "input": torch.from_numpy(inputs),
            "target": torch.from_numpy(target[None]),
            "file_name": str(record["file_name"]),
        }


def physical_image_sampling_weights(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Give every physical image equal total sampling mass."""
    counts = Counter(str(record["file_name"]) for record in records)
    weights = np.asarray(
        [1.0 / counts[str(record["file_name"])] for record in records],
        dtype=np.float64,
    )
    return weights / weights.sum()


def build_refiner_model(config: Mapping[str, Any]) -> RefinerUNet:
    return RefinerUNet(
        base=int(config["base_channels"]),
        input_mean=float(config["input_mean"]),
        input_std=float(config["input_std"]),
    )
