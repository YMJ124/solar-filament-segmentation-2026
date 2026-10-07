"""Versioned target-cache metadata and validation."""

from __future__ import annotations

import hashlib
import csv
import json
from pathlib import Path

from .folds import TARGET_POLICY

CACHE_SCHEMA_VERSION = 2
CACHE_MANIFEST_NAME = "manifest.json"


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_cache_metadata(
    annotation_file: str | Path,
    fold_manifest: str | Path,
    expected_images: int,
) -> dict:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "target_policy": TARGET_POLICY,
        "annotation_sha256": sha256_file(annotation_file),
        "fold_manifest_sha256": sha256_file(fold_manifest),
        "expected_images": int(expected_images),
    }


def validate_cache_manifest(
    cache_dir: str | Path,
    annotation_file: str | Path,
    fold_manifest: str | Path,
) -> dict:
    """Reject caches built from a different annotation file or target policy."""
    path = Path(cache_dir) / CACHE_MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"target cache manifest not found: {path}")
    metadata = json.loads(path.read_text())
    expected = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "target_policy": TARGET_POLICY,
        "annotation_sha256": sha256_file(annotation_file),
        "fold_manifest_sha256": sha256_file(fold_manifest),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"stale or incompatible target cache: {mismatches}")
    with Path(fold_manifest).open(newline="") as handle:
        expected_entries = {
            f"{Path(row['file_name']).stem}.npz" for row in csv.DictReader(handle)
        }
    if metadata.get("expected_images") != len(expected_entries):
        raise ValueError(
            "target cache expected_images does not match the fold manifest: "
            f"{metadata.get('expected_images')} != {len(expected_entries)}"
        )
    actual_entries = {
        path.name
        for path in Path(cache_dir).glob("*.npz")
        if not path.name.startswith(".")
    }
    missing = sorted(expected_entries - actual_entries)
    unexpected = sorted(actual_entries - expected_entries)
    if missing or unexpected:
        raise ValueError(
            "target cache entries do not exactly match the fold manifest; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return metadata
