"""Versioned, portable cache for raw YOLO instance candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.io_utils import atomic_write_json
from src.submission.rle import decode_rle, mask_to_rle, rle_counts, rle_from_counts


CACHE_SCHEMA_VERSION = 1
CACHE_FORMAT = "yolo_candidate_rle_v1"


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(payload: Mapping[str, Any]) -> str:
    """Return a stable hash for a JSON-compatible configuration."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def candidate_file_name(file_name: str) -> str:
    """Map an image name to a readable collision-resistant cache file name."""
    suffix = hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:12]
    return f"{Path(file_name).stem}-{suffix}.json"


def candidate_record_payload(
    record: Mapping[str, Any],
    masks: np.ndarray,
    confidences: np.ndarray,
    *,
    run_config_sha256: str,
) -> dict[str, Any]:
    """Serialize one physical image's pre-postprocess candidates."""
    height, width = int(record["height"]), int(record["width"])
    masks = np.asarray(masks, dtype=bool)
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    if masks.ndim != 3 or masks.shape[1:] != (height, width):
        raise ValueError(
            f"candidate masks must have shape [N, {height}, {width}], got {masks.shape}"
        )
    if len(masks) != len(confidences):
        raise ValueError("mask and confidence counts differ")
    if not np.all(np.isfinite(confidences)) or np.any(confidences < 0) or np.any(
        confidences > 1
    ):
        raise ValueError("candidate confidences must be finite and within [0, 1]")

    order = np.argsort(-confidences, kind="stable")
    candidates = []
    for index in order:
        rle = mask_to_rle(masks[int(index)])
        candidates.append(
            {
                "confidence": float(confidences[int(index)]),
                "area": int(np.asarray(masks[int(index)]).sum()),
                "segmentation_rle": rle_counts(rle),
            }
        )
    image_ids = record.get("image_ids")
    if image_ids is None and record.get("selected_image_id") is not None:
        image_ids = [record["selected_image_id"]]
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "format": CACHE_FORMAT,
        "run_config_sha256": run_config_sha256,
        "file_name": str(record["file_name"]),
        "image_ids": [str(value) for value in (image_ids or [])],
        "selected_image_id": (
            str(record["selected_image_id"])
            if record.get("selected_image_id") is not None
            else None
        ),
        "fold": int(record["fold"]) if record.get("fold") is not None else None,
        "height": height,
        "width": width,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def write_candidate_record(
    path: str | Path,
    record: Mapping[str, Any],
    masks: np.ndarray,
    confidences: np.ndarray,
    *,
    run_config_sha256: str,
) -> dict[str, Any]:
    """Atomically write one candidate record and return its manifest entry."""
    path = Path(path)
    payload = candidate_record_payload(
        record, masks, confidences, run_config_sha256=run_config_sha256
    )
    atomic_write_json(path, payload)
    return {
        "run_config_sha256": payload["run_config_sha256"],
        "file_name": payload["file_name"],
        "image_ids": payload["image_ids"],
        "selected_image_id": payload["selected_image_id"],
        "fold": payload["fold"],
        "height": payload["height"],
        "width": payload["width"],
        "candidate_count": payload["candidate_count"],
        "candidate_file": path.name,
        "candidate_file_sha256": sha256_file(path),
    }


def load_cache_manifest(cache_dir: str | Path) -> dict[str, Any]:
    """Load and structurally validate a completed candidate cache."""
    cache_dir = Path(cache_dir)
    path = cache_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"completed cache manifest not found: {path}; run cache_yolo_candidates first"
        )
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported YOLO cache schema: {payload.get('schema_version')!r}"
        )
    if payload.get("format") != CACHE_FORMAT:
        raise ValueError(f"unsupported YOLO cache format: {payload.get('format')!r}")
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("cache manifest contains no images")
    names = [str(row.get("file_name")) for row in images]
    if len(names) != len(set(names)):
        raise ValueError("cache manifest contains duplicate image names")
    if int(payload.get("physical_images", -1)) != len(images):
        raise ValueError("cache manifest physical_images does not match image entries")
    run_hash = payload.get("run_config_sha256")
    if not isinstance(run_hash, str) or not run_hash:
        raise ValueError("cache manifest has no run configuration hash")
    if any(row.get("run_config_sha256") != run_hash for row in images):
        raise ValueError("cache manifest mixes image records from different runs")
    return payload


def _load_candidate_payload(
    cache_dir: str | Path,
    image_entry: Mapping[str, Any],
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    """Load and structurally validate one cached candidate JSON payload."""
    cache_dir = Path(cache_dir)
    path = cache_dir / str(image_entry["candidate_file"])
    if not path.is_file():
        raise FileNotFoundError(f"candidate cache file missing: {path}")
    if verify_hash:
        expected = image_entry.get("candidate_file_sha256")
        if expected and sha256_file(path) != expected:
            raise ValueError(f"candidate cache hash mismatch: {path}")
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported candidate record schema in {path}")
    for key in (
        "run_config_sha256",
        "file_name",
        "height",
        "width",
        "candidate_count",
    ):
        if payload.get(key) != image_entry.get(key):
            raise ValueError(f"candidate record {key} does not match manifest: {path}")
    candidates = payload.get("candidates", [])
    if len(candidates) != int(payload["candidate_count"]):
        raise ValueError(f"candidate count mismatch in {path}")
    return payload


def load_candidate_rles(
    cache_dir: str | Path,
    image_entry: Mapping[str, Any],
    *,
    verify_hash: bool = True,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    """Load candidates as compressed RLEs without allocating dense masks."""
    payload = _load_candidate_payload(
        cache_dir, image_entry, verify_hash=verify_hash
    )
    candidates = payload.get("candidates", [])
    height, width = int(payload["height"]), int(payload["width"])
    rles = [
        rle_from_counts(str(row["segmentation_rle"]), height, width)
        for row in candidates
    ]
    scores = np.asarray(
        [float(row["confidence"]) for row in candidates], dtype=np.float32
    )
    if len(scores) and (
        not np.all(np.isfinite(scores)) or np.any(scores < 0) or np.any(scores > 1)
    ):
        raise ValueError(
            f"invalid confidence in {Path(cache_dir) / str(image_entry['candidate_file'])}"
        )
    areas = np.asarray([int(row["area"]) for row in candidates], dtype=np.int64)
    if np.any(areas < 0):
        raise ValueError("candidate areas must be non-negative")
    if len(scores) > 1 and np.any(scores[:-1] < scores[1:]):
        raise ValueError("cached candidates must be sorted by descending confidence")
    return rles, scores, areas, payload


def load_candidate_record(
    cache_dir: str | Path,
    image_entry: Mapping[str, Any],
    *,
    verify_hash: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one cached image as dense boolean masks plus confidences."""
    rles, scores, _, payload = load_candidate_rles(
        cache_dir, image_entry, verify_hash=verify_hash
    )
    height, width = int(payload["height"]), int(payload["width"])
    if rles:
        masks = np.stack(
            [
                decode_rle(str(row["counts"].decode("ascii")), height, width)
                for row in rles
            ]
        )
    else:
        masks = np.zeros((0, height, width), dtype=bool)
    return masks, scores, payload


def validate_search_space(
    manifest: Mapping[str, Any],
    *,
    confidences: list[float],
    max_instances: list[int],
) -> None:
    """Reject grids that request candidates discarded during GPU inference."""
    inference = manifest.get("inference", {})
    cached_confidence = float(inference.get("candidate_confidence", 1.0))
    cached_limit = int(inference.get("max_candidates", 0))
    if min(confidences) < cached_confidence:
        raise ValueError(
            f"search confidence {min(confidences)} is below cache floor "
            f"{cached_confidence}; regenerate the cache with a lower candidate confidence"
        )
    if max(max_instances) > cached_limit:
        raise ValueError(
            f"search max_instances {max(max_instances)} exceeds cached max_candidates "
            f"{cached_limit}; regenerate the cache with a larger limit"
        )


def validate_annotation_provenance(
    manifest: Mapping[str, Any], annotation_path: str | Path
) -> None:
    """Ensure offline scoring uses the exact annotations recorded by the cache."""
    expected = manifest.get("dataset", {}).get("annotation_sha256")
    if not expected:
        raise ValueError("cache manifest has no annotation SHA256")
    actual = sha256_file(annotation_path)
    if actual != expected:
        raise ValueError(
            "annotation file does not match candidate cache provenance: "
            f"{actual} != {expected}"
        )
