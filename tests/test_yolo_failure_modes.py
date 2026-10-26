import json

import numpy as np
import pytest

from src.data.coco import CocoAnnotations
from src.instance.yolo_evaluation import YoloPostprocessConfig
from src.instance.yolo_failure_modes import (
    analyze_yolo_failure_modes,
    summarize_failure_mode_rows,
)


def test_failure_modes_decompose_fn_and_fp(tmp_path):
    annotation = tmp_path / "annotations.json"
    annotation.write_text(
        json.dumps(
            {
                "images": [
                    {"id": "a", "file_name": "20260101000000Bh.jpeg", "height": 16, "width": 16},
                    {"id": "b", "file_name": "20260101000000Bh.jpeg", "height": 16, "width": 16},
                ],
                "annotations": [
                    {"id": "a1", "image_id": "a", "category_id": 1, "segmentation": [[1, 1, 1, 7, 7, 7, 7, 1]]},
                    {"id": "b1", "image_id": "b", "category_id": 1, "segmentation": [[1, 1, 1, 7, 7, 7, 7, 1]]},
                    {"id": "b2", "image_id": "b", "category_id": 1, "segmentation": [[10, 10, 10, 12, 12, 12, 12, 10]]},
                ],
                "categories": [{"id": 1, "name": "filament"}],
            }
        )
    )
    masks = np.zeros((2, 16, 16), dtype=bool)
    masks[0, 1:8, 1:8] = True
    masks[1, 8:16, 0:8] = True
    row = analyze_yolo_failure_modes(
        masks,
        np.asarray([0.9, 0.8]),
        {"file_name": "20260101000000Bh.jpeg", "fold": 2, "height": 16, "width": 16},
        CocoAnnotations(annotation),
        YoloPostprocessConfig(0.1, 1, 10, False),
        truth_area_edges=(10, 40),
        prediction_area_edges=(40,),
    )
    assert row["false_negatives"] == 1
    assert row["false_negative_iou_bands"] == {"no_overlap": 1}
    assert row["consensus_false_positive_instances"] == 1
    assert row["consensus_false_positive_records"] == 2
    assert row["partial_annotator_false_positive_records"] == 0
    summary = summarize_failure_mode_rows([row])
    assert summary["global"]["false_positives"] == 2
    assert summary["global"]["false_negative_area_shares"]["0_9"] == pytest.approx(1.0)
