import json

import pytest

from src.data.cache import (
    CACHE_MANIFEST_NAME,
    expected_cache_metadata,
    validate_cache_manifest,
)


def test_cache_manifest_detects_changed_fold_manifest(tmp_path):
    annotation = tmp_path / "annotations.json"
    folds = tmp_path / "folds.csv"
    cache = tmp_path / "cache"
    cache.mkdir()
    annotation.write_text('{"images": [], "annotations": []}')
    folds.write_text("file_name,fold,group,selected_image_id\n")
    metadata = expected_cache_metadata(annotation, folds, expected_images=0)
    (cache / CACHE_MANIFEST_NAME).write_text(json.dumps(metadata))
    validated = validate_cache_manifest(cache, annotation, folds)
    assert validated["target_policy"] == "medoid_annotator_v1"
    folds.write_text("file_name,fold,group,selected_image_id\nchanged.jpeg,0,20260101,id\n")
    with pytest.raises(ValueError, match="stale or incompatible"):
        validate_cache_manifest(cache, annotation, folds)


def test_cache_manifest_rejects_missing_entries_before_training(tmp_path):
    annotation = tmp_path / "annotations.json"
    folds = tmp_path / "folds.csv"
    cache = tmp_path / "cache"
    cache.mkdir()
    annotation.write_text('{"images": [], "annotations": []}')
    folds.write_text(
        "file_name,fold,group,selected_image_id\n"
        "20260101000000Bh.jpeg,0,20260101,id\n"
    )
    metadata = expected_cache_metadata(annotation, folds, expected_images=1)
    (cache / CACHE_MANIFEST_NAME).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="do not exactly match"):
        validate_cache_manifest(cache, annotation, folds)
