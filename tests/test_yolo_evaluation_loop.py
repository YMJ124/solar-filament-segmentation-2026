import json

import numpy as np
import pytest

from src.data.coco import CocoAnnotations
from src.data.masks import annotations_to_masks
from src.instance.yolo_cache import (
    CACHE_FORMAT,
    CACHE_SCHEMA_VERSION,
    load_cache_manifest,
    load_candidate_record,
    sha256_file,
    validate_annotation_provenance,
    validate_search_space,
    write_candidate_record,
)
from src.instance.yolo_error_analysis import (
    analyze_yolo_image,
    select_yolo_error_groups,
)
from src.instance.yolo_evaluation import (
    YoloPostprocessConfig,
    evaluate_cached_grid,
)
from src.io_utils import atomic_write_json


@pytest.fixture
def diagnostic_coco(tmp_path):
    path = tmp_path / "annotations.json"
    data = {
        "images": [
            {
                "id": "annotator-1",
                "file_name": "image.jpeg",
                "height": 16,
                "width": 16,
            }
        ],
        "annotations": [
            {
                "id": "large",
                "image_id": "annotator-1",
                "category_id": 1,
                "segmentation": [[1, 1, 1, 7, 7, 7, 7, 1]],
            },
            {
                "id": "small",
                "image_id": "annotator-1",
                "category_id": 1,
                "segmentation": [[10, 10, 10, 12, 12, 12, 12, 10]],
            },
        ],
        "categories": [{"id": 1, "name": "filament"}],
    }
    path.write_text(json.dumps(data))
    return CocoAnnotations(path)


def test_candidate_cache_round_trip_and_offline_grid(tmp_path, diagnostic_coco):
    cache = tmp_path / "cache"
    cache.mkdir()
    truth = annotations_to_masks(
        diagnostic_coco.annotations_for_batch("annotator-1"), 16, 16
    )[0]
    masks = np.stack([truth, np.eye(16, dtype=bool)])
    scores = np.asarray([0.8, 0.2], dtype=np.float32)
    record = {
        "file_name": "image.jpeg",
        "image_ids": ["annotator-1"],
        "selected_image_id": "annotator-1",
        "fold": 0,
        "height": 16,
        "width": 16,
    }
    entry = write_candidate_record(
        cache / "image.json", record, masks, scores, run_config_sha256="run-hash"
    )
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "format": CACHE_FORMAT,
        "run_config_sha256": "run-hash",
        "checkpoint": {"sha256": "checkpoint-hash"},
        "inference": {"candidate_confidence": 0.01, "max_candidates": 20},
        "dataset": {
            "fold": 0,
            "annotation_sha256": sha256_file(diagnostic_coco.path),
        },
        "physical_images": 1,
        "images": [entry],
    }
    atomic_write_json(cache / "manifest.json", manifest)

    loaded_manifest = load_cache_manifest(cache)
    loaded_masks, loaded_scores, payload = load_candidate_record(cache, entry)
    np.testing.assert_array_equal(loaded_masks, masks)
    np.testing.assert_allclose(loaded_scores, scores)
    assert payload["image_ids"] == ["annotator-1"]

    rows = evaluate_cached_grid(
        cache,
        loaded_manifest,
        diagnostic_coco,
        [YoloPostprocessConfig(0.5, 1, 10, False)],
    )
    assert rows[0]["official_pq"] == pytest.approx(2 / 3)
    assert rows[0]["hit_pairs"] == 1
    assert rows[0]["false_negatives"] == 1

    validate_search_space(
        loaded_manifest, confidences=[0.1], max_instances=[10]
    )
    validate_annotation_provenance(loaded_manifest, diagnostic_coco.path)
    wrong_annotation = tmp_path / "wrong.json"
    wrong_annotation.write_text("{}")
    with pytest.raises(ValueError, match="does not match"):
        validate_annotation_provenance(loaded_manifest, wrong_annotation)
    with pytest.raises(ValueError, match="below cache floor"):
        validate_search_space(
            loaded_manifest, confidences=[0.001], max_instances=[10]
        )
    with pytest.raises(ValueError, match="exceeds cached"):
        validate_search_space(
            loaded_manifest, confidences=[0.1], max_instances=[30]
        )


def test_yolo_error_analysis_finds_split_small_miss_and_large_fp(diagnostic_coco):
    truth_masks = annotations_to_masks(
        diagnostic_coco.annotations_for_batch("annotator-1"), 16, 16
    )
    large, _ = truth_masks
    columns = np.indices(large.shape)[1]
    first_half = large & (columns <= 3)
    second_half = large & (columns > 3)
    unrelated = np.zeros((16, 16), dtype=bool)
    unrelated[8:16, 0:8] = True
    masks = np.stack([first_half, second_half, unrelated])
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    record = {
        "file_name": "image.jpeg",
        "selected_image_id": "annotator-1",
        "fold": 0,
        "height": 16,
        "width": 16,
    }
    row, labels, selected_truth, matrix = analyze_yolo_image(
        masks,
        scores,
        record,
        diagnostic_coco,
        YoloPostprocessConfig(0.1, 1, 10, False),
        small_instance_area=10,
        large_false_positive_area=40,
        diagnostic_iou=0.0,
    )

    assert labels.max() == 3
    assert len(selected_truth) == 2
    assert matrix.shape == (2, 3)
    assert row["prediction_instances"] == 3
    assert row["small_instance_misses"] == 1
    assert row["large_false_positive_instances"] == 1
    assert row["fragmented_gt_instances"] == 1
    assert row["merged_prediction_instances"] == 0
    assert {"large_false_positive", "small_instance_miss", "fragmented_filament"} <= set(
        row["categories"]
    )


def test_yolo_error_group_selection_includes_requested_views():
    rows = [
        {
            "file_name": f"image-{index}.jpeg",
            "official_pq": index / 10,
            "false_positives": 10 - index,
            "false_negatives": index,
            "large_false_positive_instances": index % 3,
            "largest_consensus_false_positive_area": index * 100,
            "small_instance_misses": index,
            "fragmented_gt_instances": index,
            "merged_prediction_instances": 10 - index,
            "annotator_pq_range": index / 20,
        }
        for index in range(10)
    ]
    groups = select_yolo_error_groups(rows, top_k=4)
    assert groups["worst_pq"][0] == "image-0.jpeg"
    assert groups["best_pq"][0] == "image-9.jpeg"
    assert groups["most_small_instance_misses"][0] == "image-9.jpeg"
    assert groups["largest_annotator_variation"][0] == "image-9.jpeg"
