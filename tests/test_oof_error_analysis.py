import numpy as np
from PIL import Image

from src.oof_error_analysis import (
    classify_error_categories,
    render_error_canvas,
    select_error_groups,
)


def test_error_category_rules_are_explicit_and_multilabel():
    metrics = {
        "false_negatives": 2,
        "minimum_missed_gt_contrast": 0.03,
        "minimum_missed_gt_area": 20,
        "min_area_threshold": 160,
        "fragmented_gt_instances": 1,
        "false_positives": 3,
        "minimum_medoid_fp_mean_intensity": 0.1,
        "image_intensity_p25": 0.2,
        "medoid_error_pixels": 100,
        "tile_seam_error_fraction": 0.4,
        "merged_prediction_instances": 1,
        "annotator_batches": 2,
        "annotator_mean_pairwise_semantic_iou": 0.3,
    }
    assert classify_error_categories(metrics) == [
        "low_contrast_miss",
        "small_target_miss",
        "structure_middle_break",
        "dark_background_false_positive",
        "tile_seam",
        "instance_adhesion",
        "annotator_disagreement",
    ]


def test_error_group_selection_sizes_and_order():
    rows = [
        {
            "file_name": f"image-{index:02d}.jpeg",
            "official_pq": index / 20,
            "false_positives": 20 - index,
            "false_negatives": index,
            "fragmented_gt_instances": index % 4,
        }
        for index in range(15)
    ]
    groups = select_error_groups(rows)
    assert {key: len(value) for key, value in groups.items()} == {
        "lowest_pq": 10,
        "most_false_positives": 10,
        "most_false_negatives": 10,
        "most_fragmented": 10,
        "best_pq": 5,
    }
    assert groups["lowest_pq"][0] == "image-00.jpeg"
    assert groups["best_pq"][0] == "image-14.jpeg"
    assert groups["most_false_positives"][0] == "image-00.jpeg"
    assert groups["most_false_negatives"][0] == "image-14.jpeg"


def test_five_panel_render_has_expected_layout(tmp_path):
    image = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    probability = np.zeros((32, 32), dtype=np.float32)
    probability[4:12, 4:12] = 0.9
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[4:12, 4:12] = 1
    truth = np.zeros((32, 32), dtype=bool)
    truth[5:13, 5:13] = True
    path = tmp_path / "panel.png"
    render_error_canvas(
        image,
        probability,
        labels,
        [truth],
        np.array([[0.62]], dtype=np.float64),
        {
            "file_name": "image.jpeg",
            "official_pq": 0.62,
            "hit_pairs": 1,
            "false_positives": 0,
            "false_negatives": 0,
        },
        path,
        display_size=128,
    )
    with Image.open(path) as rendered:
        assert rendered.size == (640, 200)
        assert rendered.mode == "RGB"
