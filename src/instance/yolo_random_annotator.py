"""Ultralytics adapters for deterministic random-annotator training."""

from __future__ import annotations

import json
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np

from .annotator_schedule import RANDOM_ANNOTATOR_TARGET_POLICY, annotator_for_epoch


def parse_yolo_segment_label(
    path: str | Path, *, im_file: str, shape: tuple[int, int]
) -> dict[str, Any]:
    """Parse a one-class YOLO segment file into an Ultralytics label dictionary."""
    classes: list[float] = []
    segments: list[np.ndarray] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        tokens = line.split()
        if not tokens:
            continue
        try:
            values = np.asarray([float(token) for token in tokens], dtype=np.float32)
        except ValueError as exc:
            raise ValueError(f"invalid numeric label at {path}:{line_number}") from exc
        if len(values) < 7 or (len(values) - 1) % 2:
            raise ValueError(f"invalid segment label at {path}:{line_number}")
        if values[0] != 0:
            raise ValueError(f"expected one-class label 0 at {path}:{line_number}")
        points = values[1:].reshape(-1, 2)
        if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1):
            raise ValueError(f"non-finite or unnormalized segment at {path}:{line_number}")
        classes.append(0.0)
        segments.append(points)

    boxes = []
    for points in segments:
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        center = (minimum + maximum) / 2
        boxes.append([center[0], center[1], *(maximum - minimum)])
    return {
        "im_file": str(im_file),
        "shape": tuple(int(value) for value in shape),
        "cls": np.asarray(classes, dtype=np.float32).reshape(-1, 1),
        "bboxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        "segments": segments,
        "keypoints": None,
        "normalized": True,
        "bbox_format": "xywh",
    }


def _ultralytics_components():
    try:
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.models.yolo.segment import SegmentationTrainer
        from ultralytics.utils import colorstr
        from ultralytics.utils.torch_utils import unwrap_model
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("install requirements-yolo.txt for random-annotator training") from exc
    return YOLODataset, SegmentationTrainer, colorstr, unwrap_model


def random_annotator_trainer_class():
    """Create a trainer class lazily so core repository imports stay optional."""
    YOLODataset, SegmentationTrainer, colorstr, unwrap_model = _ultralytics_components()

    class RandomAnnotatorYOLODataset(YOLODataset):
        """One physical image per epoch with a deterministic rotating target."""

        def __init__(self, *args, annotator_manifest: str | Path, selection_seed: int, **kwargs):
            self.annotator_manifest_path = Path(annotator_manifest)
            self.selection_seed = int(selection_seed)
            super().__init__(*args, **kwargs)
            payload = json.loads(self.annotator_manifest_path.read_text())
            if payload.get("target_policy") != RANDOM_ANNOTATOR_TARGET_POLICY:
                raise ValueError("annotator manifest target policy is incompatible")
            if int(payload.get("selection_seed", -1)) != self.selection_seed:
                raise ValueError("annotator manifest selection seed does not match dataset")
            targets = payload.get("train_targets", {})
            expected = {Path(path).name for path in self.im_files}
            if int(payload.get("physical_train_images", -1)) != len(expected):
                raise ValueError("annotator manifest physical train-image count is stale")
            if set(targets) != expected:
                raise ValueError(
                    "random-annotator manifest does not exactly cover train images: "
                    f"missing={sorted(expected - set(targets))[:5]}, "
                    f"unexpected={sorted(set(targets) - expected)[:5]}"
                )
            self._alternatives: list[dict[str, dict[str, Any]]] = []
            self._image_ids: list[tuple[str, ...]] = []
            for base_label in self.labels:
                file_name = Path(base_label["im_file"]).name
                rows = targets[file_name]
                image_ids = [str(row["image_id"]) for row in rows]
                if not image_ids or len(image_ids) != len(set(image_ids)):
                    raise ValueError(
                        f"annotator alternatives must be non-empty and unique: {file_name}"
                    )
                alternatives = {
                    str(row["image_id"]): parse_yolo_segment_label(
                        row["label_file"],
                        im_file=base_label["im_file"],
                        shape=tuple(base_label["shape"]),
                    )
                    for row in rows
                }
                self._alternatives.append(alternatives)
                self._image_ids.append(tuple(image_ids))
            self.selected_image_ids: dict[str, str] = {}
            self.current_epoch = 0
            self.set_epoch(0)

        def set_epoch(self, epoch: int) -> None:
            if epoch < 0:
                raise ValueError("epoch must be non-negative")
            selections = {}
            for index, im_file in enumerate(self.im_files):
                file_name = Path(im_file).name
                selected = annotator_for_epoch(
                    self._image_ids[index],
                    seed=self.selection_seed,
                    file_name=file_name,
                    epoch=epoch,
                )
                self.labels[index] = self._alternatives[index][selected]
                selections[file_name] = selected
            self.current_epoch = int(epoch)
            self.selected_image_ids = selections

    class RandomAnnotatorSegmentationTrainer(SegmentationTrainer):
        """Segmentation trainer that advances annotator targets each epoch."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback("on_train_epoch_start", self._select_epoch_targets)

        @staticmethod
        def _select_epoch_targets(trainer) -> None:
            dataset = trainer.train_loader.dataset
            if not isinstance(dataset, RandomAnnotatorYOLODataset):
                raise TypeError("random-annotator trainer received an incompatible dataset")
            dataset.set_epoch(int(trainer.epoch))

        def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
            if mode != "train":
                return super().build_dataset(img_path, mode, batch)
            if int(self.args.workers) != 0:
                raise ValueError("random-annotator training requires workers=0")
            manifest = self.data.get("annotator_manifest")
            if not manifest:
                raise ValueError("dataset YAML is missing annotator_manifest")
            stride = max(int(unwrap_model(self.model).stride.max()), 32)
            return RandomAnnotatorYOLODataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=True,
                hyp=copy(self.args),
                rect=bool(self.args.rect),
                cache=self.args.cache or None,
                single_cls=self.args.single_cls or False,
                stride=stride,
                pad=0.0,
                prefix=colorstr("train: "),
                task=self.args.task,
                classes=self.args.classes,
                data=self.data,
                fraction=self.args.fraction,
                annotator_manifest=manifest,
                selection_seed=int(self.data.get("selection_seed", self.args.seed)),
            )

    return RandomAnnotatorSegmentationTrainer
