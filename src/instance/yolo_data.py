"""Build leakage-safe YOLO segmentation data from the selected annotator targets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from src.data.coco import CocoAnnotations
from src.io_utils import atomic_write_json

from .annotator_schedule import (
    MEDOID_TARGET_POLICY,
    RANDOM_ANNOTATOR_TARGET_POLICY,
    SUPPORTED_TARGET_POLICIES,
)


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
    target_policy: str = MEDOID_TARGET_POLICY,
    selection_seed: int = 2026,
) -> dict[str, Any]:
    """Create one-image-per-observation YOLO data with a versioned target policy."""
    selected = [dict(record) for record in records]
    if not selected:
        raise ValueError("no fold records supplied")
    if target_policy not in SUPPORTED_TARGET_POLICIES:
        raise ValueError(f"unsupported target policy: {target_policy}")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    image_dir = Path(image_dir)
    fold_manifest = Path(fold_manifest)
    split_counts = {"train": 0, "val": 0}
    instance_counts = {"train": 0, "val": 0}
    empty_counts = {"train": 0, "val": 0}
    selected_ids: dict[str, str] = {}
    train_targets: dict[str, list[dict[str, str]]] = {}
    annotator_label_count = 0
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
        if split == "train" and target_policy == RANDOM_ANNOTATOR_TARGET_POLICY:
            alternatives = []
            alternative_dir = output / "annotator_labels" / Path(file_name).stem
            alternative_dir.mkdir(parents=True, exist_ok=True)
            for annotator_index, annotator_batch in enumerate(
                coco.batches_for_file(file_name)
            ):
                annotator_id = str(annotator_batch["id"])
                annotator_shape = (
                    int(annotator_batch["width"]),
                    int(annotator_batch["height"]),
                )
                if annotator_shape != (width, height):
                    raise ValueError(
                        f"annotator image dimensions disagree for {file_name}: "
                        f"{annotator_shape} != {(width, height)}"
                    )
                annotator_lines = yolo_segment_lines(
                    coco.annotations_for_batch(annotator_id), width, height
                )
                annotator_path = alternative_dir / f"{annotator_index:02d}.txt"
                annotator_path.write_text(
                    "\n".join(annotator_lines) + ("\n" if annotator_lines else "")
                )
                alternatives.append(
                    {
                        "image_id": annotator_id,
                        "label_file": str(annotator_path.resolve()),
                    }
                )
                annotator_label_count += 1
            train_targets[file_name] = alternatives
        split_counts[split] += 1
        instance_counts[split] += len(lines)
        empty_counts[split] += int(not lines)
        selected_ids[file_name] = image_id

    dataset_yaml: dict[str, Any] = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "filament"},
        "target_policy": target_policy,
    }
    annotator_manifest_path = output / "annotator_manifest.json"
    if target_policy == RANDOM_ANNOTATOR_TARGET_POLICY:
        annotator_payload = {
            "schema_version": 1,
            "target_policy": RANDOM_ANNOTATOR_TARGET_POLICY,
            "selection_algorithm": "sha256_seeded_permutation_then_epoch_cycle_v1",
            "selection_seed": int(selection_seed),
            "validation_policy": MEDOID_TARGET_POLICY,
            "physical_train_images": split_counts["train"],
            "annotator_label_files": annotator_label_count,
            "train_targets": train_targets,
        }
        atomic_write_json(annotator_manifest_path, annotator_payload)
        dataset_yaml.update(
            {
                "annotator_manifest": str(annotator_manifest_path.resolve()),
                "selection_seed": int(selection_seed),
                "validation_target_policy": MEDOID_TARGET_POLICY,
            }
        )
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True)
    )
    manifest = {
        "schema_version": 1,
        "target_policy": target_policy,
        "validation_target_policy": MEDOID_TARGET_POLICY,
        "selection_seed": (
            int(selection_seed)
            if target_policy == RANDOM_ANNOTATOR_TARGET_POLICY
            else None
        ),
        "annotator_manifest": (
            str(annotator_manifest_path.resolve())
            if target_policy == RANDOM_ANNOTATOR_TARGET_POLICY
            else None
        ),
        "annotator_label_files": annotator_label_count,
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
