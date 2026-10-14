import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.instance import exclusive_label_map, prediction_arrays, yolo_segment_lines


def test_yolo_segment_lines_collapse_categories_and_normalize():
    rows = yolo_segment_lines([
        {"id": "a", "category_id": 3, "segmentation": [[0, 0, 20, 0, 20, 10]]}
    ], width=20, height=10)
    values = rows[0].split()
    assert values[0] == "0"
    np.testing.assert_allclose([float(value) for value in values[1:]], [0, 0, 1, 0, 1, 1])


def test_yolo_segment_lines_reject_multiple_polygons_per_instance():
    with pytest.raises(ValueError, match="exactly one polygon"):
        yolo_segment_lines([
            {"id": "a", "segmentation": [[0, 0, 2, 0, 2, 2], [3, 3, 4, 3, 4, 4]]}
        ], width=8, height=8)


def test_exclusive_masks_assign_overlap_to_higher_confidence():
    masks = np.zeros((2, 5, 5), dtype=bool)
    masks[0, 1:4, 1:4] = True
    masks[1, 2:5, 2:5] = True
    labels = exclusive_label_map(
        masks, np.array([0.4, 0.9]), confidence_threshold=0.3,
        max_instances=2,
    )
    assert labels[2, 2] == 1
    assert labels[1, 1] == 2
    assert (labels == 1).sum() == 9
    assert (labels == 2).sum() == 5


def test_exclusive_masks_force_one_and_area_filter():
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, :2, :2] = True
    masks[1, 3, 3] = True
    labels = exclusive_label_map(
        masks, np.array([0.2, 0.1]), confidence_threshold=0.8,
        max_instances=2, min_area=2, force_one=True,
    )
    assert labels.max() == 1
    assert (labels == 1).sum() == 4


def test_yolo_prediction_arrays_resize_masks_and_copy_scores():
    class Value:
        def __init__(self, value):
            self.value = torch.as_tensor(value)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value.numpy()

    class Boxes:
        def __init__(self):
            self.conf = Value([0.75])

        def __len__(self):
            return 1

    result = type(
        "Result",
        (),
        {
            "masks": type("Masks", (), {"data": Value([[[0, 1], [1, 0]]])})(),
            "boxes": Boxes(),
        },
    )()

    masks, scores = prediction_arrays(result, (4, 4))

    assert masks.shape == (1, 4, 4)
    assert masks.dtype == np.bool_
    np.testing.assert_allclose(scores, [0.75])
