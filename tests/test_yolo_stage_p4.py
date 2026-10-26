import json
from pathlib import Path

import pytest
import yaml

from scripts.cache_yolo_candidates import (
    _experiment_metadata,
    inference_resolution_provenance,
)
from scripts.train_yolo import load_config
from src.data.coco import CocoAnnotations
from src.instance.p4_diagnostics import (
    equivalent_width,
    pair_residual_instance_rows,
    physical_feature_rows,
    representation_band,
    summarize_cap_pressure,
    summarize_paired_residual_rows,
    summarize_p3_sampling_shift,
)
from src.instance.yolo_sampling import (
    EpochWeightedSampler,
    STATION_PRESERVING_MEDIUM_SAMPLING_POLICY,
    build_station_preserving_sampling_manifest,
    validate_sampling_manifest,
)


@pytest.fixture
def p4_fixture(tmp_path):
    annotation = tmp_path / "annotations.json"
    images = []
    annotations = []
    records = []
    specs = [
        ("20260101000000Bh.jpeg", 0, "a", [500, 700, 1800]),
        ("20260102000000Bh.jpeg", 1, "b", [450] * 6),
        ("20260103000000Ch.jpeg", 2, "c", [100, 900, 2000]),
        ("20260104000000Ch.jpeg", 3, "d", [600] * 4),
    ]
    for file_name, fold, image_id, areas in specs:
        images.append(
            {"id": image_id, "file_name": file_name, "width": 2048, "height": 2048}
        )
        for index, area in enumerate(areas):
            annotations.append(
                {
                    "id": f"{image_id}-{index}",
                    "image_id": image_id,
                    "category_id": 1,
                    "area": area,
                    "bbox": [0, 0, 20, 10],
                    "segmentation": [[0, 0, 20, 0, 20, 10, 0, 10]],
                    "spine": [0, 5, 10, 5, 20, 5],
                }
            )
        records.append(
            {
                "file_name": file_name,
                "fold": fold,
                "selected_image_id": image_id,
                "n_batches": 1,
                "medoid_mean_iou": 1.0,
            }
        )
    annotation.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "filament"}],
            }
        )
    )
    folds = tmp_path / "folds.csv"
    folds.write_text("fixture")
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("target_policy: medoid_annotator_v1\n")
    return CocoAnnotations(annotation), records, folds, dataset


def test_equivalent_width_and_proto_bands():
    annotation = {
        "id": "x",
        "area": 80,
        "spine": [0, 0, 10, 0, 20, 0],
        "bbox": [0, 0, 20, 5],
    }
    width, length, source = equivalent_width(annotation)
    assert width == pytest.approx(4.0)
    assert length == pytest.approx(20.0)
    assert source == "area_over_spine_length"
    assert representation_band(0.5) == "lt_1_proto_cell"
    assert representation_band(1.5) == "1_to_lt_2_proto_cells"
    assert representation_band(2.0) == "2_plus_proto_cells"


def test_paired_residual_summary_and_cap_pressure():
    common = {
        "file_name": "x.jpeg",
        "image_id": "batch",
        "fold": 0,
        "station": "Bh",
        "area": 500.0,
        "medium_area": True,
        "representation_band_1536": "lt_1_proto_cell",
    }
    control = [
        dict(common, annotation_id="a", outcome="no_overlap", max_prediction_iou=0.0),
        dict(common, annotation_id="b", outcome="matched_gt_0_5", max_prediction_iou=0.7),
    ]
    candidate = [
        dict(common, annotation_id="a", outcome="matched_gt_0_5", max_prediction_iou=0.6),
        dict(
            common,
            annotation_id="b",
            outcome="near_threshold_gt_0_25_to_0_5",
            max_prediction_iou=0.4,
        ),
    ]
    paired = pair_residual_instance_rows(control, candidate)
    summary = summarize_paired_residual_rows(paired)["global"]
    assert summary["transitions"] == {
        "recovered_by_t1": 1,
        "regressed_in_t1": 1,
    }
    assert summary["net_recovered"] == 0
    cap = summarize_cap_pressure(
        [
            {
                "fold": 0,
                "i1_candidates_above_confidence": 21,
                "i1_raw_area_eligible_beyond_cap": 1,
                "i1_prediction_instances": 19,
                "i1_candidate_count": 50,
                "i1_cache_max_candidates": 50,
                "t1_candidates_above_confidence": 20,
                "t1_raw_area_eligible_beyond_cap": 0,
                "t1_prediction_instances": 20,
                "t1_candidate_count": 30,
                "t1_cache_max_candidates": 50,
            }
        ],
        max_instances=20,
    )
    assert cap["i1"]["confidence_pool_truncated_images"] == 1
    assert cap["i1"]["raw_area_eligible_beyond_cap_images"] == 1
    assert cap["t1"]["confidence_pool_truncated_images"] == 0
    assert cap["t1"]["output_at_cap_images"] == 1


def test_sampling_manifest_preserves_budget_and_station_mass(p4_fixture):
    coco, records, folds, dataset = p4_fixture
    payload = build_station_preserving_sampling_manifest(
        records,
        coco,
        validation_fold=0,
        fold_manifest=folds,
        dataset_yaml=dataset,
        selection_seed=2026,
    )
    validate_sampling_manifest(
        payload,
        expected_file_names=[row["file_name"] for row in records if row["fold"] != 0],
    )
    assert payload["physical_train_images"] == 3
    assert payload["samples_per_epoch"] == 3
    by_station = {}
    for row in payload["images"]:
        by_station.setdefault(row["station"], 0.0)
        by_station[row["station"]] += row["sampling_weight"]
    assert by_station == pytest.approx({"Bh": 1.0, "Ch": 2.0})
    rows = {row["file_name"]: row for row in payload["images"]}
    assert rows["20260104000000Ch.jpeg"]["sampling_weight"] > rows[
        "20260103000000Ch.jpeg"
    ]["sampling_weight"]


def test_full_fit_sampling_manifest_uses_every_fold(p4_fixture):
    coco, records, folds, dataset = p4_fixture
    payload = build_station_preserving_sampling_manifest(
        records,
        coco,
        validation_fold=None,
        fold_manifest=folds,
        dataset_yaml=dataset,
        selection_seed=2026,
    )
    validate_sampling_manifest(
        payload, expected_file_names=[row["file_name"] for row in records]
    )
    assert payload["final_fit"] is True
    assert payload["validation_fold"] is None
    assert payload["physical_train_images"] == len(records)
    assert payload["samples_per_epoch"] == len(records)


def test_epoch_weighted_sampler_is_epoch_deterministic():
    sampler = EpochWeightedSampler([1.0, 2.0, 1.0], num_samples=12, seed=2026)
    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)
    assert list(sampler) == first
    sampler.set_epoch(4)
    assert list(sampler) != first
    assert len(first) == 12


def test_p3_shift_reports_extra_steps(p4_fixture):
    coco, records, _, _ = p4_fixture
    features = physical_feature_rows(records, coco)
    manifest = {
        "train_samples": 4,
        "repeated_train_images": [
            {
                "file_name": "20260102000000Bh.jpeg",
                "sample_names": ["repeat.jpeg"],
            }
        ],
    }
    result = summarize_p3_sampling_shift(features, {0: manifest})["0"]
    assert result["physical_train_images"] == 3
    assert result["p3_train_samples"] == 4
    assert result["optimizer_step_multiplier"] == pytest.approx(4 / 3)


def test_inference_resolution_requires_explicit_mismatch():
    experiment = {"training_imgsz": 1024}
    with pytest.raises(ValueError, match="explicitly provided"):
        inference_resolution_provenance(experiment, 1536, allow_mismatch=False)
    result = inference_resolution_provenance(
        experiment, 1536, allow_mismatch=True
    )
    assert result["train_inference_imgsz_mismatch"] is True
    assert result["inference_to_training_ratio"] == pytest.approx(1.5)


def test_sampling_provenance_preserves_historical_b0_signature():
    metadata = _experiment_metadata(
        Path("configs/instance/stage_p1/b0_medoid_1024_fold0.yaml")
    )
    assert metadata["training_signature_sha256"] == (
        "73d12b41f3a3090552035ffbbcdfbf5251800bc554b78d63fc836560f2f62465"
    )
    assert "sampling_policy" not in metadata["training_signature"]
    assert "sampling_manifest_sha256" not in metadata["training_signature"]


def test_p4_training_config_and_metadata_include_sampling_hash(
    tmp_path, p4_fixture
):
    coco, records, folds, dataset = p4_fixture
    sampling = build_station_preserving_sampling_manifest(
        records,
        coco,
        validation_fold=0,
        fold_manifest=folds,
        dataset_yaml=dataset,
    )
    sampling_path = tmp_path / "sampling.json"
    sampling_path.write_text(json.dumps(sampling))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "name": "p4_t1_fold0",
                    "project": str(tmp_path / "runs"),
                    "seed": 2026,
                },
                "data": {
                    "dataset_yaml": str(dataset),
                    "target_policy": "medoid_annotator_v1",
                    "sampling_policy": STATION_PRESERVING_MEDIUM_SAMPLING_POLICY,
                    "sampling_manifest": str(sampling_path),
                },
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
    loaded = load_config(config_path)
    assert loaded["data"]["sampling_policy"] == STATION_PRESERVING_MEDIUM_SAMPLING_POLICY
    metadata = _experiment_metadata(config_path)
    assert metadata["sampling_policy"] == STATION_PRESERVING_MEDIUM_SAMPLING_POLICY
    assert len(metadata["sampling_manifest_sha256"]) == 64
    assert metadata["training_signature"]["sampling_policy"] == (
        STATION_PRESERVING_MEDIUM_SAMPLING_POLICY
    )
    assert "sampling_manifest_sha256" not in metadata["training_signature"]
