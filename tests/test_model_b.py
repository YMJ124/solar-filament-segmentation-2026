import numpy as np
import pytest
import sys

torch = pytest.importorskip("torch")

from src.engine import augment_teacher_probability, train_one_epoch
from src.full_image_validation import _atomic_save_probability
from src.models import DetailUNet, EfficientNetB0DetailUNet, GlobalUNet, build_model
from src.models.ensemble import predict_batch, predict_sliding_window
from src.training_config import load_training_config
from scripts.train import _cli_overrides, _parse_args


class _Capture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = None

    def forward(self, value):
        self.value = value.detach().clone()
        return value[:, :1]


def test_efficientnet_detail_shape_and_channel_normalization():
    model = EfficientNetB0DetailUNet(
        encoder_weights=None, input_mean=0.25, input_std=0.25
    )
    capture = _Capture()
    model.network = capture
    inputs = torch.cat([
        torch.full((1, 1, 32, 32), 0.50),
        torch.full((1, 1, 32, 32), 0.75),
    ], dim=1)
    output = model(inputs)
    assert output.shape == (1, 1, 32, 32)
    assert torch.allclose(capture.value[:, :1], torch.ones_like(output))
    assert torch.allclose(capture.value[:, 1:], torch.full_like(output, 0.5))
    with pytest.raises(ValueError, match="two channels"):
        model(torch.zeros(1, 1, 32, 32))


def test_factory_builds_formal_model_b_without_downloading_weights():
    model = build_model(
        "detail",
        {"architecture": "efficientnet_b0_unet", "encoder_weights": None},
        initialize_pretrained=False,
    ).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 2, 64, 64))
    assert output.shape == (1, 1, 64, 64)


def test_detail_training_keeps_teacher_frozen():
    teacher = GlobalUNet(base=2).eval()
    teacher.requires_grad_(False)
    model = DetailUNet(base=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    before = [parameter.detach().clone() for parameter in teacher.parameters()]
    batch = {
        "image": torch.zeros(1, 1, 32, 32),
        "semantic_mask": torch.zeros(1, 1, 32, 32),
    }
    stats = train_one_epoch(
        model, [batch], optimizer, torch.device("cpu"), "detail",
        teacher_model=teacher,
        teacher_augmentation={
            "dropout": 0.05, "blur_probability": 1.0, "blur_kernel": 3,
        },
    )
    assert stats["batches"] == 1
    assert not teacher.training
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert all(torch.equal(old, new) for old, new in zip(before, teacher.parameters()))


def test_teacher_augmentation_validation_and_bounds():
    probability = torch.ones(2, 1, 8, 8)
    augmented = augment_teacher_probability(
        probability, {"dropout": 0.25, "blur_probability": 1.0, "blur_kernel": 3}
    )
    assert augmented.shape == probability.shape
    assert float(augmented.min()) >= 0.0
    assert float(augmented.max()) <= 1.0
    with pytest.raises(ValueError, match="odd"):
        augment_teacher_probability(probability, {"blur_kernel": 2})


def test_detail_inference_requires_teacher_and_saves_components(tmp_path):
    image = torch.zeros(1, 1, 16, 16)
    with pytest.raises(ValueError, match="requires a global"):
        predict_batch({"detail": DetailUNet(base=2)}, image, torch.device("cpu"))

    models = {
        "global": GlobalUNet(base=2).eval(),
        "detail": DetailUNet(base=2).eval(),
    }
    prediction = predict_sliding_window(
        models, np.zeros((32, 32), dtype=np.float32), torch.device("cpu"),
        tile_size=32, weights=(0.0, 1.0, 0.0), return_components=True,
    )
    probability, boundary, distance, components = prediction
    assert probability.shape == (32, 32)
    assert boundary is None and distance is None
    assert set(components) == {"global_probability", "detail_probability"}

    output = tmp_path / "prediction.npz"
    _atomic_save_probability(output, probability, "sample.jpeg", components)
    with np.load(output) as payload:
        assert set(payload.files) == {
            "probability", "file_name", "global_probability", "detail_probability"
        }
        assert payload["probability"].dtype == np.float16


def test_formal_model_b_config_is_training_ready():
    config = load_training_config("configs/detail_efficientnet_b0_fold0.yaml")
    assert config["model"]["kind"] == "detail"
    assert config["training"]["epochs"] == 40
    assert config["training"]["teacher_checkpoint"]
    assert config["validation"]["save_components"] is True


def test_cli_can_explicitly_disable_encoder_weights(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--encoder-weights", "none"])
    overrides = _cli_overrides(_parse_args())
    assert "encoder_weights" in overrides["model"]
    assert overrides["model"]["encoder_weights"] is None
