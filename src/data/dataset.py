"""PyTorch datasets and target construction for filament segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset

from .coco import CocoAnnotations
from .cache import validate_cache_manifest
from .folds import load_fold_manifest
from .masks import annotations_to_masks


def _semantic_target(
    instances: list[np.ndarray], shape: tuple[int, int] | None = None
) -> np.ndarray:
    """Build the binary semantic target used by the single-model core."""
    if not instances:
        shape = shape or (1, 1)
        return np.zeros(shape, np.float32)
    h, w = instances[0].shape
    semantic = np.zeros((h, w), dtype=bool)
    for mask in instances:
        semantic |= np.asarray(mask, dtype=bool)
    return semantic.astype(np.float32)


def _instance_boxes(instances: list[np.ndarray]) -> np.ndarray:
    """Return [y0, x0, y1, x1] boxes for non-empty instance masks."""
    boxes = []
    for mask in instances:
        points = np.argwhere(np.asarray(mask, dtype=bool))
        if len(points):
            y0, x0 = points.min(axis=0)
            y1, x1 = points.max(axis=0) + 1
            boxes.append((int(y0), int(x0), int(y1), int(x1)))
    return np.asarray(boxes, dtype=np.int32).reshape(-1, 4)


def _crop(arrays: list[np.ndarray], size: int, rng: np.random.Generator, train: bool,
          positive_probability: float = 0.7,
          instance_boxes: np.ndarray | None = None) -> list[np.ndarray]:
    h, w = arrays[0].shape[-2:]
    if size >= h and size >= w:
        return arrays
    if train:
        foreground = arrays[1][0] > 0
        if foreground.any() and rng.random() < positive_probability:
            points = None
            boxes = np.asarray(instance_boxes, dtype=np.int32).reshape(-1, 4) if instance_boxes is not None else []
            if len(boxes):
                y0, x0, y1, x1 = boxes[int(rng.integers(0, len(boxes)))]
                local = np.argwhere(foreground[y0:y1, x0:x1])
                if len(local):
                    points = local + np.array([y0, x0])
            if points is None:
                points = np.argwhere(foreground)
            cy, cx = points[int(rng.integers(0, len(points)))]
            jitter = max(1, size // 4)
            cy += int(rng.integers(-jitter, jitter + 1))
            cx += int(rng.integers(-jitter, jitter + 1))
            y = int(np.clip(cy - size // 2, 0, h - size))
            x = int(np.clip(cx - size // 2, 0, w - size))
        else:
            y = int(rng.integers(0, max(1, h - size + 1)))
            x = int(rng.integers(0, max(1, w - size + 1)))
    else:
        foreground = arrays[1][0] > 0
        if foreground.any():
            points = None
            boxes = np.asarray(instance_boxes, dtype=np.int32).reshape(-1, 4) if instance_boxes is not None else []
            if len(boxes):
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                y0, x0, y1, x1 = boxes[int(np.argmax(areas))]
                local = np.argwhere(foreground[y0:y1, x0:x1])
                if len(local):
                    points = local + np.array([y0, x0])
            if points is None:
                points = np.argwhere(foreground)
            cy, cx = points.mean(axis=0)
            y = int(np.clip(round(cy) - size // 2, 0, h - size))
            x = int(np.clip(round(cx) - size // 2, 0, w - size))
        else:
            y, x = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return [a[..., y:y + size, x:x + size] for a in arrays]


def _resize_full(arrays: list[np.ndarray], size: int) -> list[np.ndarray]:
    """Resize a complete physical image and its semantic target."""
    image, semantic = arrays
    resized_image = cv2.resize(
        image[0], (size, size), interpolation=cv2.INTER_LINEAR
    )[None]
    resized_semantic = cv2.resize(
        semantic[0], (size, size), interpolation=cv2.INTER_NEAREST
    )[None]
    return [
        np.asarray(resized_image, dtype=np.float32),
        np.asarray(resized_semantic > 0, dtype=np.float32),
    ]


class FilamentDataset(Dataset):
    """Lazy PyTorch dataset for cropped filament-segmentation targets."""

    def __init__(self, data_root: str | Path, annotation_file: str | Path | None = None,
                 fold: int | None = None, seed: int = 2026,
                 image_size: int = 512, train: bool = True, records: list[dict[str, Any]] | None = None,
                 positive_crop_probability: float = 0.7, cache_dir: str | Path | None = None,
                 fold_manifest: str | Path = "data/manifests/train_folds.csv",
                 sample_mode: str = "crop"):
        self.root = Path(data_root)
        self.train_dir = self.root / "train/train_images"
        self.image_size, self.train = image_size, train
        self.positive_crop_probability = positive_crop_probability
        if sample_mode not in {"crop", "full_resize"}:
            raise ValueError("sample_mode must be 'crop' or 'full_resize'")
        self.sample_mode = sample_mode
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.fold_manifest = Path(fold_manifest)
        self.seed = int(seed + (fold or 0))
        self.epoch = 0
        annotation_file = Path(annotation_file or self.root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
        self.annotation_file = annotation_file
        self.coco = CocoAnnotations(annotation_file) if annotation_file.exists() else None
        if records is None:
            if self.coco is None:
                raise FileNotFoundError(annotation_file)
            records = load_fold_manifest(self.fold_manifest, self.annotation_file)
        self.records = [dict(x) for x in records if fold is None or int(x.get("fold", -1)) == fold]
        if any(not row.get("selected_image_id") for row in self.records):
            raise ValueError("records must contain selected_image_id from the medoid manifest")
        if self.coco:
            for row in self.records:
                valid_ids = {
                    str(batch["id"])
                    for batch in self.coco.batches_for_file(str(row["file_name"]))
                }
                if str(row["selected_image_id"]) not in valid_ids:
                    raise ValueError(
                        f"selected_image_id is invalid for {row['file_name']}: "
                        f"{row['selected_image_id']}"
                    )
        if self.cache_dir:
            validate_cache_manifest(
                self.cache_dir, self.annotation_file, self.fold_manifest
            )

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Select deterministic augmentations for an epoch.

        Randomness is derived from ``(seed, epoch, sample index)`` instead of a
        mutable process-local generator. This keeps results stable with multiple
        workers and makes a resumed epoch reproducible.
        """
        self.epoch = int(epoch)

    def _read(self, index: int) -> dict[str, Any]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        record = self.records[index]
        name = str(record["file_name"])
        with Image.open(self.train_dir / name) as image:
            image_array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        cache_path = self.cache_dir / f"{Path(name).stem}.npz" if self.cache_dir else None
        selected_image_id = str(record["selected_image_id"])
        if cache_path:
            if not cache_path.exists():
                raise FileNotFoundError(f"target cache entry not found: {cache_path}")
            with np.load(cache_path) as cached:
                cached_image_id = str(cached["selected_image_id"].item())
                if cached_image_id != selected_image_id:
                    raise ValueError(
                        f"stale target cache for {name}: {cached_image_id} != {selected_image_id}"
                    )
                semantic = cached["semantic"].astype(np.float32)
                instance_boxes = cached["instance_boxes"].astype(np.int32)
        else:
            instances = annotations_to_masks(
                self.coco.annotations_for_batch(selected_image_id), *image_array.shape
            ) if self.coco else []
            semantic = _semantic_target(instances, image_array.shape)
            instance_boxes = _instance_boxes(instances)
        arrays = [image_array[None], semantic[None]]
        if self.sample_mode == "full_resize":
            arrays = _resize_full(arrays, self.image_size)
        else:
            arrays = _crop(
                arrays, self.image_size, rng, self.train,
                self.positive_crop_probability, instance_boxes,
            )
        image_array, semantic = [x[0] for x in arrays]
        if self.train and rng.random() < 0.5:
            image_array, semantic = [np.flip(x, axis=1).copy() for x in (image_array, semantic)]
        if self.train and rng.random() < 0.5:
            image_array, semantic = [np.flip(x, axis=0).copy() for x in (image_array, semantic)]
        if self.train:
            rotations = int(rng.integers(0, 4))
            if rotations:
                image_array, semantic = [
                    np.rot90(x, rotations).copy()
                    for x in (image_array, semantic)
                ]
        if self.train and rng.random() < 0.3:
            gamma = float(rng.uniform(0.8, 1.2))
            image_array = np.clip(image_array ** gamma, 0, 1)
        return {"image": image_array[None].astype(np.float32),
                "semantic_mask": semantic[None].astype(np.float32),
                "file_name": name, "fold": int(record.get("fold", -1)),
                "selected_image_id": selected_image_id}

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._read(index)
        for key in ("image", "semantic_mask"):
            item[key] = torch.from_numpy(item[key])
        return item


def collate_filaments(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {"image": torch.stack([x["image"] for x in batch]),
            "semantic_mask": torch.stack([x["semantic_mask"] for x in batch]),
            "file_name": [x["file_name"] for x in batch], "fold": [x["fold"] for x in batch],
            "selected_image_id": [x["selected_image_id"] for x in batch]}
