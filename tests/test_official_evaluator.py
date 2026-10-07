import csv
import json

import pytest

from src.data.coco import CocoAnnotations
from src.data.masks import annotations_to_rles
from src.metrics.official_evaluator import evaluate_predictions, load_prediction_csv


@pytest.fixture
def two_annotator_coco(tmp_path):
    path = tmp_path / "annotations.json"
    data = {
        "images": [
            {
                "id": "ann1-image",
                "file_name": "image.jpeg",
                "height": 16,
                "width": 16,
            },
            {
                "id": "ann2-image",
                "file_name": "image.jpeg",
                "height": 16,
                "width": 16,
            },
        ],
        "annotations": [
            {
                "id": "a1",
                "image_id": "ann1-image",
                "category_id": 1,
                "segmentation": [[1, 1, 1, 5, 5, 5, 5, 1]],
            },
            {
                "id": "a2",
                "image_id": "ann2-image",
                "category_id": 1,
                "segmentation": [[9, 9, 9, 13, 13, 13, 13, 9]],
            },
        ],
        "categories": [{"id": 1, "name": "filament"}],
    }
    path.write_text(json.dumps(data))
    return CocoAnnotations(path)


def test_prediction_is_reused_for_every_annotator(two_annotator_coco):
    first_batch = two_annotator_coco.annotations_for_batch("ann1-image")
    prediction = annotations_to_rles(first_batch, 16, 16)
    report = evaluate_predictions(
        two_annotator_coco, {"image.jpeg": prediction}, ["image.jpeg"]
    )
    assert report.physical_images == 1
    assert report.annotator_records == 2
    assert report.hit_pairs == 1
    assert report.false_positives == 1
    assert report.false_negatives == 1
    assert report.official_pq == 0.5


def test_competition_csv_loader(two_annotator_coco, tmp_path):
    prediction = annotations_to_rles(
        two_annotator_coco.annotations_for_batch("ann1-image"), 16, 16
    )[0]
    counts = prediction["counts"].decode("ascii")
    path = tmp_path / "predictions.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["filament_id", "segmentation_rle"]
        )
        writer.writeheader()
        writer.writerow(
            {"filament_id": "image_1", "segmentation_rle": counts}
        )
    loaded = load_prediction_csv(path, two_annotator_coco)
    report = evaluate_predictions(two_annotator_coco, loaded)
    assert list(loaded) == ["image.jpeg"]
    assert report.official_pq == 0.5


def test_competition_csv_rejects_duplicate_ids(two_annotator_coco, tmp_path):
    prediction = annotations_to_rles(
        two_annotator_coco.annotations_for_batch("ann1-image"), 16, 16
    )[0]
    counts = prediction["counts"].decode("ascii")
    path = tmp_path / "duplicates.csv"
    path.write_text(
        "filament_id,segmentation_rle\n"
        f"image_1,{counts}\n"
        f"image_1,{counts}\n"
    )
    with pytest.raises(ValueError, match="duplicate filament_id"):
        load_prediction_csv(path, two_annotator_coco)
