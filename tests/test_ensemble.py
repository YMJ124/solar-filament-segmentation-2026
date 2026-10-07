import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models import GlobalUNet
from src.inference import predict_sliding_window, sliding_starts
from src.metrics.segmentation import segmentation_metrics
from src.postprocess.instances import (
    probability_to_instance_labels,
    probability_to_instances,
)
from src.submission.validate import (
    audit_image_labels,
    audit_image_masks,
    audit_submission_rows,
)
from src.submission.rle import encode_mask, label_map_to_rles


def test_model_shape():
    image = torch.randn(1, 1, 64, 64)
    assert GlobalUNet(base=4)(image).shape == (1, 1, 64, 64)


def test_connected_component_postprocess():
    probability = np.zeros((32, 32), dtype=np.float32)
    probability[2:8, 2:8] = 1
    probability[20:28, 20:28] = 1
    masks = probability_to_instances(probability, threshold=0.5, min_area=5)
    assert len(masks) == 2


def test_label_postprocess_keeps_largest_instances_under_safety_cap():
    probability = np.zeros((32, 32), dtype=np.float32)
    probability[2:4, 2:4] = 1
    probability[10:13, 10:13] = 1
    probability[20:24, 20:24] = 1
    labels = probability_to_instance_labels(
        probability,
        threshold=0.5,
        min_area=1,
        max_instances=2,
    )
    assert labels.max() == 2
    assert int((labels > 0).sum()) == 25
    assert len(label_map_to_rles(labels)) == 2
    assert audit_image_labels("image", labels, shape=(32, 32))["instances"] == 2


def test_sliding_window_covers_last_pixel():
    starts = sliding_starts(2048, 256, 192)
    assert starts[0] == 0
    assert starts[-1] == 2048 - 256


def test_tile_batching_does_not_change_sliding_prediction():
    class ConstantModel(torch.nn.Module):
        def forward(self, image):
            return torch.zeros_like(image) + 0.25

    image = np.random.default_rng(7).random((32, 32), dtype=np.float32)
    model = ConstantModel().eval()
    first = predict_sliding_window(
        model, image, torch.device("cpu"), tile_size=16,
        overlap=0.25, tile_batch_size=1,
    )
    batched = predict_sliding_window(
        model, image, torch.device("cpu"), tile_size=16,
        overlap=0.25, tile_batch_size=4,
    )
    np.testing.assert_allclose(first, batched, rtol=0, atol=0)

    with pytest.raises(ValueError, match="no larger than the image"):
        predict_sliding_window(
            model, image, torch.device("cpu"), tile_size=64
        )


def test_metrics_and_submission_audit():
    truth = np.zeros((8, 8), dtype=bool)
    truth[1:4, 1:4] = True
    assert segmentation_metrics(truth, truth)["dice"] == 1.0
    report = audit_image_masks("image", [truth], shape=(8, 8))
    assert report["instances"] == 1
    summary = audit_submission_rows(
        [{"filament_id": "image_1", "segmentation_rle": encode_mask(truth)}],
        {"image"},
        shape=(8, 8),
    )
    assert summary["covered_images"] == 1
    assert summary["rles_validated"] is True


def test_submission_audit_rejects_overlapping_instances():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:4, 1:4] = True
    rows = [
        {"filament_id": "image_1", "segmentation_rle": encode_mask(mask)},
        {"filament_id": "image_2", "segmentation_rle": encode_mask(mask)},
    ]
    with pytest.raises(ValueError, match="overlap"):
        audit_submission_rows(rows, {"image"}, shape=(8, 8))
