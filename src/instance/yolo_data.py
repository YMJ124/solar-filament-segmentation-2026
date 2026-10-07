"""Build leakage-safe YOLO segmentation data from the selected annotator targets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from src.data.coco import CocoAnnotations
from src.io_utils import atomic_write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def yolo_segment_lines(
    annotations: Iterable[Mapping[str, Any]], width: int, height: int
) -> list[str]:
    """Convert one-instance-one-polygon COCO rows to one-class YOLO segments."""
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    lines = []
    for annotation in annotations:
        polygons = annotation.get("segmentation")
        if not isinstance(polygons, list) or len(polygons) != 1:
            raise ValueError(
                f"annotation {annotation.get('id')} must contain exactly one polygon"
            )
        coordinates = np.asarray(polygons[0], dtype=np.float64)
        if coordinates.ndim != 1 or len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError(f"invalid polygon in annotation {annotation.get('id')}")
        points = coordinates.reshape(-1, 2)
        if not np.isfinite(points).all():
            raise ValueError(f"non-finite polygon in annotation {annotation.get('id')}")
        points[:, 0] = np.clip(points[:, 0] / width, 0.0, 1.0)
        points[:, 1] = np.clip(points[:, 1] / height, 0.0, 1.0)
        flattened = " ".join(f"{value:.8f}" for value in points.ravel())
        lines.append(f"0 {flattened}")
    return lines


def _replace_symlink(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def build_yolo_segmentation_dataset(
    records: Iterable[Mapping[str, Any]],
    coco: CocoAnnotations,
    image_dir: str | Path,
    output_dir: str | Path,
    *,
    validation_fold: int,
    fold_manifest: str | Path,
) -> dict[str, Any]:
    """Create labels and image symlinks using one medoid annotation per JPEG."""
    selected = [dict(record) for record in records]
    if not selected:
        raise ValueError("no fold records supplied")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    image_dir = Path(image_dir)
    fold_manifest = Path(fold_manifest)
    split_counts = {"train": 0, "val": 0}
    instance_counts = {"train": 0, "val": 0}
    empty_counts = {"train": 0, "val": 0}
    selected_ids: dict[str, str] = {}
    for split in split_counts:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    for record in selected:
        file_name = str(record["file_name"])
        image_id = str(record["selected_image_id"])
        split = "val" if int(record["fold"]) == validation_fold else "train"
        source = image_dir / file_name
        if not source.is_file():
            raise FileNotFoundError(source)
        batches = {str(row["id"]): row for row in coco.batches_for_file(file_name)}
        if image_id not in batches:
            raise ValueError(f"selected image id {image_id} is invalid for {file_name}")
        batch = batches[image_id]
        width, height = int(batch["width"]), int(batch["height"])
        annotations = coco.annotations_for_batch(image_id)
        lines = yolo_segment_lines(annotations, width, height)
        _replace_symlink(source, output / "images" / split / file_name)
        label_path = output / "labels" / split / f"{Path(file_name).stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        split_counts[split] += 1
        instance_counts[split] += len(lines)
        empty_counts[split] += int(not lines)
        selected_ids[file_name] = image_id

    dataset_yaml = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "filament"},
    }
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True)
    )
    manifest = {
        "schema_version": 1,
        "target_policy": "medoid_annotator_v1",
        "validation_fold": int(validation_fold),
        "physical_images": split_counts,
        "instances": instance_counts,
        "empty_images": empty_counts,
        "fold_manifest": str(fold_manifest),
        "fold_manifest_sha256": _sha256(fold_manifest),
        "annotation_file": str(coco.path),
        "annotation_sha256": _sha256(coco.path),
        "image_link_type": "symlink",
        "selected_image_ids": selected_ids,
    }
    atomic_write_json(output / "manifest.json", manifest)
    return manifest
