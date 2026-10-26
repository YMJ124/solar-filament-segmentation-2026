"""Deterministic, fixed-budget sampling for P4 YOLO training."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

from src.data.coco import CocoAnnotations
from src.io_utils import atomic_write_json

from .p4_diagnostics import MEDIUM_AREA_MAX, MEDIUM_AREA_MIN, annotation_area
from .yolo_cache import sha256_file


UNIFORM_SAMPLING_POLICY = "uniform_physical_v1"
STATION_PRESERVING_MEDIUM_SAMPLING_POLICY = (
    "station_preserving_medium_density_400_1600_cap1p5_v1"
)
SUPPORTED_SAMPLING_POLICIES = {
    UNIFORM_SAMPLING_POLICY,
    STATION_PRESERVING_MEDIUM_SAMPLING_POLICY,
}


def build_station_preserving_sampling_manifest(
    records: Iterable[Mapping[str, Any]],
    coco: CocoAnnotations,
    *,
    validation_fold: int | None,
    fold_manifest: str | Path,
    dataset_yaml: str | Path,
    selection_seed: int = 2026,
) -> dict[str, Any]:
    """Build capped medium-density weights while preserving each station's mass."""
    selected = [
        dict(record)
        for record in records
        if validation_fold is None or int(record["fold"]) != validation_fold
    ]
    if not selected:
        raise ValueError("sampling manifest has no training physical images")
    names = [str(record["file_name"]) for record in selected]
    if len(names) != len(set(names)):
        raise ValueError("sampling manifest records contain duplicate physical images")

    rows = []
    station_raw_totals: defaultdict[str, float] = defaultdict(float)
    station_counts: Counter[str] = Counter()
    for record in selected:
        file_name = str(record["file_name"])
        image_id = str(record["selected_image_id"])
        annotations = coco.annotations_for_batch(image_id)
        medium_count = sum(
            MEDIUM_AREA_MIN <= annotation_area(annotation) < MEDIUM_AREA_MAX
            for annotation in annotations
        )
        raw_weight = 1.0 + 0.5 * min(medium_count / 6.0, 1.0)
        station = Path(file_name).stem[-2:]
        row = {
            "file_name": file_name,
            "selected_image_id": image_id,
            "station": station,
            "year": Path(file_name).stem[:4],
            "medium_instances": medium_count,
            "raw_weight": raw_weight,
        }
        rows.append(row)
        station_raw_totals[station] += raw_weight
        station_counts[station] += 1

    for row in rows:
        station = str(row["station"])
        row["sampling_weight"] = (
            float(row["raw_weight"])
            * station_counts[station]
            / station_raw_totals[station]
        )
    total_weight = sum(float(row["sampling_weight"]) for row in rows)
    if not np.isclose(total_weight, len(rows), rtol=0.0, atol=1e-9):
        raise RuntimeError("station-preserving weights do not sum to the sample budget")

    fold_manifest = Path(fold_manifest)
    dataset_yaml = Path(dataset_yaml)
    return {
        "schema_version": 1,
        "sampling_policy": STATION_PRESERVING_MEDIUM_SAMPLING_POLICY,
        "target_policy": "medoid_annotator_v1",
        "validation_policy": (
            "medoid_annotator_v1" if validation_fold is not None else None
        ),
        "validation_fold": (
            int(validation_fold) if validation_fold is not None else None
        ),
        "final_fit": validation_fold is None,
        "selection_seed": int(selection_seed),
        "replacement": True,
        "physical_train_images": len(rows),
        "samples_per_epoch": len(rows),
        "weight_formula": "1+0.5*min(medium_count/6,1), station-mass normalized",
        "medium_area_interval": {
            "minimum_inclusive": MEDIUM_AREA_MIN,
            "maximum_exclusive": MEDIUM_AREA_MAX,
        },
        "weight_bounds": {
            "minimum": min(float(row["sampling_weight"]) for row in rows),
            "maximum": max(float(row["sampling_weight"]) for row in rows),
        },
        "station_expected_exposures": dict(sorted(station_counts.items())),
        "fold_manifest": str(fold_manifest.resolve()),
        "fold_manifest_sha256": sha256_file(fold_manifest),
        "annotation_file": str(coco.path.resolve()),
        "annotation_sha256": sha256_file(coco.path),
        "dataset_yaml": str(dataset_yaml.resolve()),
        "dataset_yaml_sha256": sha256_file(dataset_yaml),
        "images": sorted(rows, key=lambda row: str(row["file_name"])),
    }


def validate_sampling_manifest(
    payload: Mapping[str, Any], *, expected_file_names: Sequence[str] | None = None
) -> None:
    """Validate coverage, budget, and station-mass invariants."""
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported sampling manifest schema")
    if payload.get("sampling_policy") != STATION_PRESERVING_MEDIUM_SAMPLING_POLICY:
        raise ValueError("sampling manifest policy is incompatible")
    rows = payload.get("images")
    if not isinstance(rows, list) or not rows:
        raise ValueError("sampling manifest contains no images")
    names = [str(row.get("file_name")) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("sampling manifest contains duplicate images")
    if expected_file_names is not None and set(names) != set(expected_file_names):
        expected = set(expected_file_names)
        actual = set(names)
        raise ValueError(
            "sampling manifest does not exactly cover the training dataset: "
            f"missing={sorted(expected - actual)[:5]}, "
            f"unexpected={sorted(actual - expected)[:5]}"
        )
    count = int(payload.get("physical_train_images", -1))
    samples = int(payload.get("samples_per_epoch", -1))
    if count != len(rows) or samples != len(rows):
        raise ValueError("sampling manifest does not preserve the baseline epoch budget")
    station_weights: defaultdict[str, float] = defaultdict(float)
    station_counts: Counter[str] = Counter()
    for row in rows:
        weight = float(row.get("sampling_weight", 0.0))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("sampling weights must be finite and positive")
        station = str(row.get("station"))
        station_weights[station] += weight
        station_counts[station] += 1
    for station, station_count in station_counts.items():
        if not np.isclose(
            station_weights[station], station_count, rtol=0.0, atol=1e-8
        ):
            raise ValueError(f"sampling weights do not preserve station {station}")


class EpochWeightedSampler(Sampler[int]):
    """Weighted sampler whose sequence is a pure function of seed and epoch."""

    def __init__(self, weights: Sequence[float], *, num_samples: int, seed: int):
        values = torch.as_tensor(weights, dtype=torch.double)
        if values.ndim != 1 or len(values) == 0 or not torch.isfinite(values).all():
            raise ValueError("sampler weights must be a finite non-empty vector")
        if torch.any(values <= 0) or num_samples < 1:
            raise ValueError("sampler weights and num_samples must be positive")
        self.weights = values
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        self.last_indices: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _epoch_seed(self) -> int:
        digest = hashlib.sha256(f"{self.seed}:{self.epoch}".encode()).digest()
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self._epoch_seed())
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True,
            generator=generator,
        ).tolist()
        self.last_indices = [int(index) for index in indices]
        return iter(self.last_indices)

    def __len__(self) -> int:
        return self.num_samples


def station_preserving_sampler_trainer_class(
    sampling_manifest: str | Path, *, selection_seed: int
):
    """Create an Ultralytics trainer using the deterministic P4 sampler."""
    try:
        from ultralytics.data.build import InfiniteDataLoader
        from ultralytics.models.yolo.segment import SegmentationTrainer
        from ultralytics.utils.torch_utils import torch_distributed_zero_first
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("install requirements-yolo.txt for P4 sampling") from exc

    manifest_path = Path(sampling_manifest).resolve()
    payload = json.loads(manifest_path.read_text())
    validate_sampling_manifest(payload)
    manifest_hash = sha256_file(manifest_path)
    by_name = {str(row["file_name"]): dict(row) for row in payload["images"]}

    class StationPreservingSegmentationTrainer(SegmentationTrainer):
        """Single-GPU segmentation trainer with a fixed per-epoch draw budget."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._p4_sampler: EpochWeightedSampler | None = None
            self._p4_names: list[str] = []
            self.add_callback("on_train_epoch_start", self._set_sampling_epoch)
            self.add_callback("on_train_epoch_end", self._write_sampling_trace)

        def get_dataloader(
            self,
            dataset_path: str,
            batch_size: int = 16,
            rank: int = 0,
            mode: str = "train",
        ):
            if mode != "train":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            if int(self.args.workers) != 0:
                raise ValueError("P4 deterministic sampling requires workers=0")
            if rank != -1:
                raise ValueError("P4 deterministic sampling supports single-GPU training only")
            with torch_distributed_zero_first(rank):
                dataset = self.build_dataset(dataset_path, mode, batch_size)
            names = [Path(path).name for path in dataset.im_files]
            validate_sampling_manifest(payload, expected_file_names=names)
            weights = [float(by_name[name]["sampling_weight"]) for name in names]
            sampler = EpochWeightedSampler(
                weights,
                num_samples=int(payload["samples_per_epoch"]),
                seed=int(selection_seed),
            )
            self._p4_sampler = sampler
            self._p4_names = names
            return InfiniteDataLoader(
                dataset=dataset,
                batch_size=min(int(batch_size), len(dataset)),
                shuffle=False,
                sampler=sampler,
                num_workers=0,
                pin_memory=getattr(self.device, "type", str(self.device)) == "cuda",
                collate_fn=getattr(dataset, "collate_fn", None),
                drop_last=bool(self.args.compile),
            )

        @staticmethod
        def _set_sampling_epoch(trainer) -> None:
            if trainer._p4_sampler is None:
                raise RuntimeError("P4 sampler was not initialized")
            trainer._p4_sampler.set_epoch(int(trainer.epoch))

        @staticmethod
        def _write_sampling_trace(trainer) -> None:
            sampler = trainer._p4_sampler
            if sampler is None or len(sampler.last_indices) != len(sampler):
                raise RuntimeError("P4 sampler did not produce a complete epoch")
            counts = Counter(trainer._p4_names[index] for index in sampler.last_indices)
            station_counts: Counter[str] = Counter()
            medium_bin_counts: Counter[str] = Counter()
            for name, count in counts.items():
                row = by_name[name]
                station_counts[str(row["station"])] += count
                medium = int(row["medium_instances"])
                medium_bin = "0_2" if medium <= 2 else "3_5" if medium <= 5 else "6_plus"
                medium_bin_counts[medium_bin] += count
            path = Path(trainer.save_dir) / "sampling_trace.json"
            trace = (
                json.loads(path.read_text())
                if path.is_file()
                else {
                    "schema_version": 1,
                    "sampling_policy": STATION_PRESERVING_MEDIUM_SAMPLING_POLICY,
                    "sampling_manifest": str(manifest_path),
                    "sampling_manifest_sha256": manifest_hash,
                    "selection_seed": int(selection_seed),
                    "samples_per_epoch": len(sampler),
                    "epochs": [],
                }
            )
            row = {
                "epoch": int(trainer.epoch),
                "unique_physical_images": len(counts),
                "station_draws": dict(sorted(station_counts.items())),
                "medium_bin_draws": dict(sorted(medium_bin_counts.items())),
                "draw_counts": dict(sorted(counts.items())),
            }
            epochs = [entry for entry in trace["epochs"] if int(entry["epoch"]) != int(trainer.epoch)]
            trace["epochs"] = sorted([*epochs, row], key=lambda entry: int(entry["epoch"]))
            atomic_write_json(path, trace)

    return StationPreservingSegmentationTrainer
