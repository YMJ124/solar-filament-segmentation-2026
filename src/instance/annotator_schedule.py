"""Deterministic per-epoch annotator schedules for physical images."""

from __future__ import annotations

import hashlib
import random
from typing import Sequence


MEDOID_TARGET_POLICY = "medoid_annotator_v1"
RANDOM_ANNOTATOR_TARGET_POLICY = "random_annotator_per_epoch_v1"
SUPPORTED_TARGET_POLICIES = {
    MEDOID_TARGET_POLICY,
    RANDOM_ANNOTATOR_TARGET_POLICY,
}


def annotator_order(
    image_ids: Sequence[str], *, seed: int, file_name: str
) -> tuple[str, ...]:
    """Return a stable randomized cycle containing every annotator exactly once."""
    values = [str(value) for value in image_ids]
    if not values:
        raise ValueError(f"no annotator image IDs for {file_name}")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate annotator image IDs for {file_name}")
    material = f"{int(seed)}\0{file_name}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    rng = random.Random(local_seed)
    rng.shuffle(values)
    return tuple(values)


def annotator_for_epoch(
    image_ids: Sequence[str], *, seed: int, file_name: str, epoch: int
) -> str:
    """Select one annotator from the stable cycle for a zero-based epoch."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    order = annotator_order(image_ids, seed=seed, file_name=file_name)
    return order[epoch % len(order)]
