"""COCO reader with physical-observation grouping for MAGFiLO."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PhysicalImage:
    file_name: str
    width: int
    height: int
    image_ids: tuple[str, ...]


class CocoAnnotations:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, Any] = json.loads(self.path.read_text())
        self.images = self.data.get("images", [])
        self.annotations = self.data.get("annotations", [])
        self.categories = self.data.get("categories", [])
        self._anns_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._images_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for image in self.images:
            self._images_by_file[image["file_name"]].append(image)
        for ann in self.annotations:
            self._anns_by_image[str(ann["image_id"])].append(ann)

    @property
    def physical_files(self) -> list[str]:
        return sorted(self._images_by_file)

    def physical_images(self) -> list[PhysicalImage]:
        return [
            PhysicalImage(
                file_name=file_name,
                width=int(self._images_by_file[file_name][0]["width"]),
                height=int(self._images_by_file[file_name][0]["height"]),
                image_ids=tuple(str(row["id"]) for row in self._images_by_file[file_name]),
            )
            for file_name in self.physical_files
        ]

    def annotations_for_batch(self, image_id: str) -> list[dict[str, Any]]:
        return list(self._anns_by_image.get(str(image_id), []))

    def annotations_for_file(self, file_name: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for image in self._images_by_file[file_name]:
            rows.extend(self.annotations_for_batch(str(image["id"])))
        return rows

    def batches_for_file(self, file_name: str) -> list[dict[str, Any]]:
        return list(self._images_by_file[file_name])

    def iter_physical_records(self) -> Iterable[dict[str, Any]]:
        for image in self.physical_images():
            anns = self.annotations_for_file(image.file_name)
            yield {
                "file_name": image.file_name,
                "width": image.width,
                "height": image.height,
                "image_ids": list(image.image_ids),
                "n_batches": len(image.image_ids),
                "n_annotations": len(anns),
                "category_counts": _category_counts(anns),
            }


def _category_counts(annotations: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for ann in annotations:
        counts[str(ann["category_id"])] += 1
    return dict(sorted(counts.items()))
