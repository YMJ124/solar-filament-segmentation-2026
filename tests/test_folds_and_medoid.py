import json
from collections import defaultdict

import pytest

from src.data.coco import CocoAnnotations
from src.data.folds import assign_date_grouped_folds, load_fold_manifest, observation_date
from src.data.medoid import select_medoid_batch


def test_date_groups_never_cross_folds_and_assignment_is_deterministic():
    records = []
    for index in range(12):
        date = f"202601{index // 2 + 1:02d}"
        records.append(
            {
                "file_name": f"{date}{index:06d}Bh.jpeg",
                "n_batches": 1 + index % 3,
                "n_annotations": 2 + index,
            }
        )
    first = assign_date_grouped_folds(records, n_splits=3, seed=7)
    second = assign_date_grouped_folds(records, n_splits=3, seed=7)
    assert [(row["file_name"], row["fold"]) for row in first] == [
        (row["file_name"], row["fold"]) for row in second
    ]
    date_folds = defaultdict(set)
    for row in first:
        date_folds[observation_date(row["file_name"])].add(row["fold"])
        assert row["group"] == observation_date(row["file_name"])
    assert all(len(folds) == 1 for folds in date_folds.values())
    assert {row["fold"] for row in first} == {0, 1, 2}


def test_medoid_selects_most_agreed_annotation_with_stable_tie_break(tmp_path):
    annotation_path = tmp_path / "annotations.json"
    common = [1, 1, 1, 6, 6, 6, 6, 1]
    outlier = [9, 9, 9, 14, 14, 14, 14, 9]
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": "ann-a-image", "file_name": "image.jpeg", "height": 16, "width": 16},
                    {"id": "ann-b-image", "file_name": "image.jpeg", "height": 16, "width": 16},
                    {"id": "ann-c-image", "file_name": "image.jpeg", "height": 16, "width": 16},
                ],
                "annotations": [
                    {"id": "a", "image_id": "ann-a-image", "category_id": 1, "segmentation": [common]},
                    {"id": "b", "image_id": "ann-b-image", "category_id": 1, "segmentation": [common]},
                    {"id": "c", "image_id": "ann-c-image", "category_id": 1, "segmentation": [outlier]},
                ],
                "categories": [{"id": 1, "name": "filament"}],
            }
        )
    )
    selection = select_medoid_batch(CocoAnnotations(annotation_path), "image.jpeg")
    assert selection.selected_image_id == "ann-a-image"
    assert selection.selected_annotations == 1
    assert selection.medoid_mean_iou == 0.5


def test_repository_fold_manifest_passes_complete_leakage_check():
    rows = load_fold_manifest(
        "data/manifests/train_folds.csv",
        "MAGFiLO_1.0_Kaggle_2026/train/"
        "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    )
    assert len(rows) == 707
    assert {row["fold"] for row in rows} == {0, 1, 2, 3, 4}


def test_fold_loader_rejects_same_date_in_multiple_folds(tmp_path):
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps({
        "images": [
            {"id": "a", "file_name": "20260101000000Bh.jpeg"},
            {"id": "b", "file_name": "20260101120000Ch.jpeg"},
        ],
        "annotations": [],
    }))
    manifest_path = tmp_path / "folds.csv"
    header = (
        "file_name,fold,group,width,height,n_batches,n_annotations,"
        "selected_image_id,selected_annotations\n"
    )
    manifest_path.write_text(
        header
        + "20260101000000Bh.jpeg,0,20260101,16,16,1,0,a,0\n"
        + "20260101120000Ch.jpeg,1,20260101,16,16,1,0,b,0\n"
    )
    (tmp_path / "folds.meta.json").write_text(json.dumps({
        "schema_version": 2,
        "target_policy": "medoid_annotator_v1",
        "group_strategy": "acquisition_date_yyyymmdd",
        "n_splits": 2,
        "seed": 7,
        "rows": 2,
    }))
    with pytest.raises(ValueError, match="leaks acquisition dates"):
        load_fold_manifest(manifest_path, annotation_path)
