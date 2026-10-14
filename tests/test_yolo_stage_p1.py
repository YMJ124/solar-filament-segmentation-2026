import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data.coco import CocoAnnotations
from src.instance.annotator_schedule import (
    MEDOID_TARGET_POLICY,
    RANDOM_ANNOTATOR_TARGET_POLICY,
    annotator_for_epoch,
    annotator_order,
)
from src.instance.yolo_data import build_yolo_segmentation_dataset
from src.instance.yolo_multifold import (
    aggregate_multifold_candidates,
    validate_multifold_manifests,
)
from src.instance.yolo_random_annotator import parse_yolo_segment_label
from src.instance.yolo_stage_gates import multifold_gate, panel_gate
from scripts.build_yolo_stage_report import (
    FIXED_MULTIFOLD_GRID,
    FIXED_PANEL_CONFIG,
    _validate_multifold,
    _validate_panel,
)
from scripts.cache_yolo_candidates import _experiment_metadata


def test_annotator_schedule_is_deterministic_complete_and_resume_safe():
    image_ids = ["a", "b", "c"]
    first = annotator_order(image_ids, seed=2026, file_name="sample.jpeg")
    second = annotator_order(image_ids, seed=2026, file_name="sample.jpeg")
    assert first == second
    assert set(first) == set(image_ids)
    assert len(first) == len(set(first))
    assert [
        annotator_for_epoch(
            image_ids, seed=2026, file_name="sample.jpeg", epoch=epoch
        )
        for epoch in range(3)
    ] == list(first)
    assert annotator_for_epoch(
        image_ids, seed=2026, file_name="sample.jpeg", epoch=7
    ) == first[1]


def test_single_annotator_schedule_is_stable():
    for epoch in (0, 1, 99):
        assert annotator_for_epoch(
            ["only"], seed=2026, file_name="one.jpeg", epoch=epoch
        ) == "only"


@pytest.fixture
def tiny_annotator_data(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for file_name in ("20260101_a.jpeg", "20260102_b.jpeg"):
        (images / file_name).write_bytes(b"image-placeholder")
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": "a1",
                        "file_name": "20260101_a.jpeg",
                        "width": 16,
                        "height": 16,
                    },
                    {
                        "id": "a2",
                        "file_name": "20260101_a.jpeg",
                        "width": 16,
                        "height": 16,
                    },
                    {
                        "id": "b1",
                        "file_name": "20260102_b.jpeg",
                        "width": 16,
                        "height": 16,
                    },
                ],
                "annotations": [
                    {
                        "id": "ann-a1",
                        "image_id": "a1",
                        "category_id": 1,
                        "segmentation": [[1, 1, 8, 1, 8, 8, 1, 8]],
                    },
                    {
                        "id": "ann-a2",
                        "image_id": "a2",
                        "category_id": 1,
                        "segmentation": [[2, 2, 9, 2, 9, 9, 2, 9]],
                    },
                    {
                        "id": "ann-b1",
                        "image_id": "b1",
                        "category_id": 1,
                        "segmentation": [[3, 3, 10, 3, 10, 10, 3, 10]],
                    },
                ],
                "categories": [{"id": 1, "name": "filament"}],
            }
        )
    )
    fold_manifest = tmp_path / "folds.csv"
    fold_manifest.write_text("fixture")
    records = [
        {
            "file_name": "20260101_a.jpeg",
            "selected_image_id": "a2",
            "fold": 1,
        },
        {
            "file_name": "20260102_b.jpeg",
            "selected_image_id": "b1",
            "fold": 0,
        },
    ]
    return CocoAnnotations(annotation_path), images, fold_manifest, records


def test_random_dataset_keeps_one_physical_sample_and_medoid_validation(
    tmp_path, tiny_annotator_data
):
    coco, image_dir, fold_manifest, records = tiny_annotator_data
    medoid_dir = tmp_path / "medoid"
    random_dir = tmp_path / "random"
    medoid = build_yolo_segmentation_dataset(
        records,
        coco,
        image_dir,
        medoid_dir,
        validation_fold=0,
        fold_manifest=fold_manifest,
        target_policy=MEDOID_TARGET_POLICY,
    )
    random = build_yolo_segmentation_dataset(
        records,
        coco,
        image_dir,
        random_dir,
        validation_fold=0,
        fold_manifest=fold_manifest,
        target_policy=RANDOM_ANNOTATOR_TARGET_POLICY,
        selection_seed=2026,
    )

    assert random["physical_images"] == medoid["physical_images"] == {
        "train": 1,
        "val": 1,
    }
    assert len(list((random_dir / "images/train").iterdir())) == 1
    assert len(list((random_dir / "images/val").iterdir())) == 1
    assert (random_dir / "labels/val/20260102_b.txt").read_text() == (
        medoid_dir / "labels/val/20260102_b.txt"
    ).read_text()
    annotators = json.loads((random_dir / "annotator_manifest.json").read_text())
    assert annotators["physical_train_images"] == 1
    rows = annotators["train_targets"]["20260101_a.jpeg"]
    assert {row["image_id"] for row in rows} == {"a1", "a2"}
    assert all(Path(row["label_file"]).is_file() for row in rows)
    dataset = yaml.safe_load((random_dir / "dataset.yaml").read_text())
    assert dataset["target_policy"] == RANDOM_ANNOTATOR_TARGET_POLICY
    assert dataset["validation_target_policy"] == MEDOID_TARGET_POLICY


def test_parse_yolo_segment_label_builds_ultralytics_fields(tmp_path):
    label = tmp_path / "label.txt"
    label.write_text("0 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8\n")
    parsed = parse_yolo_segment_label(label, im_file="image.jpeg", shape=(16, 20))
    np.testing.assert_allclose(parsed["cls"], [[0]])
    np.testing.assert_allclose(parsed["bboxes"], [[0.3, 0.5, 0.4, 0.6]])
    assert len(parsed["segments"]) == 1
    assert parsed["normalized"] is True
    assert parsed["bbox_format"] == "xywh"


def test_experiment_metadata_hashes_training_policy(tmp_path):
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
                "target_policy": RANDOM_ANNOTATOR_TARGET_POLICY,
                "validation_target_policy": MEDOID_TARGET_POLICY,
            }
        )
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "name": "c1_random_annotator_1024_fold2",
                    "seed": 2026,
                    "deterministic": True,
                },
                "data": {"dataset_yaml": str(dataset)},
                "model": {"weights": "yolo26n-seg.pt"},
                "training": {
                    "epochs": 5,
                    "imgsz": 1024,
                    "batch": 2,
                    "workers": 0,
                    "cache": False,
                },
            }
        )
    )
    metadata = _experiment_metadata(config)
    assert metadata["experiment_family"] == "c1_random_annotator_1024"
    assert metadata["training_imgsz"] == 1024
    assert metadata["target_policy"] == RANDOM_ANNOTATOR_TARGET_POLICY
    assert len(metadata["training_signature_sha256"]) == 64


def _fold_row(pq, tp, fp, fn):
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "confidence": 0.1,
        "min_area": 400,
        "max_instances": 12,
        "force_one": True,
        "official_pq": pq,
        "physical_images": 10,
        "annotator_records": 12,
        "hit_pairs": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "hit_iou_sum": pq * denominator,
        "prediction_instances": tp + fp,
    }


def test_multifold_aggregation_computes_mean_std_and_global_counts():
    rows = aggregate_multifold_candidates(
        {
            0: [_fold_row(0.30, 10, 2, 4)],
            1: [_fold_row(0.40, 12, 3, 5)],
            2: [_fold_row(0.50, 14, 4, 6)],
        }
    )
    best = rows[0]
    assert best["fold_mean_pq"] == pytest.approx(0.4)
    assert best["fold_std_pq"] == pytest.approx(np.std([0.3, 0.4, 0.5]))
    assert best["hit_pairs"] == 36
    assert best["false_positives"] == 9
    assert best["false_negatives"] == 15
    expected_iou = sum(
        row["hit_iou_sum"]
        for row in [
            _fold_row(0.30, 10, 2, 4),
            _fold_row(0.40, 12, 3, 5),
            _fold_row(0.50, 14, 4, 6),
        ]
    )
    assert best["global_official_pq"] == pytest.approx(
        expected_iou / (36 + 0.5 * 9 + 0.5 * 15)
    )


def test_multifold_manifest_validation_rejects_mixed_experiments():
    def manifest(fold, family="b0", signature="same"):
        return {
            "inference": {
                "imgsz": 1024,
                "quantize": 16,
                "retina_masks": False,
                "candidate_confidence": 0.01,
                "max_candidates": 50,
            },
            "dataset": {
                "fold": fold,
                "annotation_sha256": "annotation",
                "fold_manifest_sha256": "folds",
            },
            "experiment": {
                "experiment_family": family,
                "training_signature_sha256": signature,
            },
        }

    manifests = {fold: manifest(fold) for fold in range(3)}
    result = validate_multifold_manifests(manifests, {0, 1, 2})
    assert result["experiment_family"] == "b0"
    manifests[2] = manifest(2, family="c1", signature="different")
    with pytest.raises(ValueError, match="mix different"):
        validate_multifold_manifests(manifests, {0, 1, 2})


def test_panel_and_multifold_promotion_gates():
    baseline_panel = {"physical_images": 32, "best": {"official_pq": 0.30}}
    candidate_panel = {"physical_images": 32, "best": {"official_pq": 0.31}}
    assert panel_gate(baseline_panel, candidate_panel)["passed"]

    def report(mean, std, values):
        return {
            "best": {
                "fold_mean_pq": mean,
                "fold_std_pq": std,
                "fold_reports": {
                    str(index): {"official_pq": value}
                    for index, value in enumerate(values)
                },
            }
        }

    baseline = report(0.300, 0.020, [0.29, 0.30, 0.31])
    candidate = report(0.309, 0.025, [0.300, 0.307, 0.320])
    result = multifold_gate(baseline, candidate)
    assert result["passed"]
    assert result["positive_folds"] == 3

    failed = multifold_gate(
        baseline,
        report(0.309, 0.025, [0.284, 0.315, 0.328]),
    )
    assert not failed["passed"]
    assert failed["criteria"]["worst_fold"] is False


def test_stage_report_requires_frozen_panel_and_multifold_grid():
    experiment = {
        "experiment_family": "b0_medoid_1024",
        "training_signature_sha256": "signature",
    }
    panel = {
        "fold": 0,
        "physical_images": 32,
        "grid_size": 1,
        "best": dict(FIXED_PANEL_CONFIG, official_pq=0.3),
        "experiment": experiment,
    }
    multifold = {
        "report_type": "yolo_multifold_postprocess_v1",
        "folds": [0, 1, 2],
        "grid": FIXED_MULTIFOLD_GRID,
        **experiment,
    }
    _validate_panel(panel)
    _validate_multifold(multifold, panel)
    multifold["grid"] = dict(FIXED_MULTIFOLD_GRID, size=99)
    with pytest.raises(ValueError, match="frozen"):
        _validate_multifold(multifold, panel)
