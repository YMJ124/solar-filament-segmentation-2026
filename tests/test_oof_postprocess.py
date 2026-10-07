import json
from pathlib import Path

import numpy as np
import pytest

from src.data.coco import CocoAnnotations
from src.data.masks import annotations_to_masks
from src.oof_postprocess import (
    PostprocessConfig,
    evaluate_oof_configuration,
    search_oof_postprocess,
    validate_oof_directory,
)
from src.postprocess.instances import probability_to_instance_labels


@pytest.fixture
def oof_case(tmp_path):
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": "annotator-1",
                        "file_name": "image.jpeg",
                        "height": 16,
                        "width": 16,
                    },
                    {
                        "id": "annotator-2",
                        "file_name": "image.jpeg",
                        "height": 16,
                        "width": 16,
                    },
                ],
                "annotations": [
                    {
                        "id": "left",
                        "image_id": "annotator-1",
                        "category_id": 1,
                        "segmentation": [[1, 1, 1, 6, 6, 6, 6, 1]],
                    },
                    {
                        "id": "right",
                        "image_id": "annotator-2",
                        "category_id": 1,
                        "segmentation": [[9, 9, 9, 14, 14, 14, 14, 9]],
                    },
                ],
                "categories": [{"id": 1, "name": "filament"}],
            }
        )
    )
    coco = CocoAnnotations(annotation_path)
    probability = annotations_to_masks(
        coco.annotations_for_batch("annotator-1"), 16, 16
    )[0].astype(np.float32)
    oof_dir = tmp_path / "oof"
    oof_dir.mkdir()
    np.savez_compressed(
        oof_dir / "image.npz",
        probability=probability.astype(np.float16),
        file_name=np.asarray("image.jpeg"),
    )
    (oof_dir / "manifest.json").write_text(
        json.dumps(
            {
                "physical_images": 1,
                "oof_probability_files": ["image.npz"],
                "evaluated_file_names": ["image.jpeg"],
            }
        )
    )
    records = [
        {
            "file_name": "image.jpeg",
            "fold": 0,
            "height": 16,
            "width": 16,
        }
    ]
    return coco, oof_dir, records


def test_closing_iterations_zero_through_three_are_legal():
    probability = np.zeros((32, 32), dtype=np.float32)
    probability[8:12, 4:9] = 1
    probability[8:12, 10:15] = 1
    counts = []
    for iterations in range(4):
        labels = probability_to_instance_labels(
            probability,
            threshold=0.5,
            min_area=1,
            closing_iterations=iterations,
            use_watershed=False,
        )
        assert labels.shape == probability.shape
        assert labels.dtype == np.int32
        assert np.all(labels >= 0)
        counts.append(int(labels.max()))
    assert counts[0] == 2
    assert counts[1] == 1


def test_closing_iterations_rejects_invalid_values():
    probability = np.zeros((8, 8), dtype=np.float32)
    for invalid in (-1, 0.5, True):
        with pytest.raises(ValueError, match="closing_iterations"):
            probability_to_instance_labels(
                probability,
                min_area=1,
                closing_iterations=invalid,
                use_watershed=False,
            )


def test_single_config_search_scores_every_annotator(oof_case):
    coco, oof_dir, records = oof_case
    config = PostprocessConfig(
        threshold=0.5, min_area=1, closing_iterations=0, max_instances=8
    )
    row = search_oof_postprocess(
        oof_dir, records, coco, [config], fold=0, progress=False
    )[0]
    assert row["physical_images"] == 1
    assert row["annotator_records"] == 2
    assert row["hit_pairs"] == 1
    assert row["false_positives"] == 1
    assert row["false_negatives"] == 1
    assert row["official_pq"] == pytest.approx(0.5)

    detailed = evaluate_oof_configuration(
        oof_dir, records, coco, config, fold=0, progress=False
    )
    assert detailed["official_pq"] == pytest.approx(row["official_pq"])
    assert detailed["hit_pairs"] == row["hit_pairs"]


def test_oof_validation_rejects_missing_extra_and_wrong_fold(oof_case):
    _, oof_dir, records = oof_case
    validate_oof_directory(oof_dir, records, fold=0)

    extra = oof_dir / "extra.npz"
    np.savez_compressed(
        extra,
        probability=np.zeros((16, 16), dtype=np.float16),
        file_name=np.asarray("extra.jpeg"),
    )
    with pytest.raises(ValueError, match="unexpected"):
        validate_oof_directory(oof_dir, records, fold=0)
    extra.unlink()

    expected = oof_dir / "image.npz"
    hidden = oof_dir / "image.hidden"
    expected.rename(hidden)
    with pytest.raises(ValueError, match="missing"):
        validate_oof_directory(oof_dir, records, fold=0)
    hidden.rename(expected)

    wrong_records = [{**records[0], "fold": 1}]
    with pytest.raises(ValueError, match="outside fold"):
        validate_oof_directory(oof_dir, wrong_records, fold=0)


def test_oof_payload_rejects_wrong_stored_file_name(oof_case):
    coco, oof_dir, records = oof_case
    np.savez_compressed(
        oof_dir / "image.npz",
        probability=np.zeros((16, 16), dtype=np.float16),
        file_name=np.asarray("wrong.jpeg"),
    )
    with pytest.raises(ValueError, match="expected 'image.jpeg'"):
        search_oof_postprocess(
            oof_dir,
            records,
            coco,
            [PostprocessConfig(min_area=1, closing_iterations=0)],
            fold=0,
            progress=False,
        )
