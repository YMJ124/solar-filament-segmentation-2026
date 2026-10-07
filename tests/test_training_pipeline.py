import numpy as np
import pytest
import random

torch = pytest.importorskip("torch")

from src.data.dataset import _crop, _semantic_target
from src.engine import restore_checkpoint, save_checkpoint, train_one_epoch
from src.full_image_validation import _degree_distribution, _positive_score_distribution
from src.models import GlobalUNet, build_model
from src.training_config import load_training_config


def test_foreground_aware_crop_contains_foreground():
    image = np.zeros((1, 64, 64), dtype=np.float32)
    mask = np.zeros((1, 64, 64), dtype=np.float32)
    mask[:, 50:55, 50:55] = 1
    arrays = [image, mask, mask.copy(), mask.copy()]
    cropped = _crop(arrays, 16, np.random.default_rng(1), True, positive_probability=1.0)
    assert cropped[1].sum() > 0


def test_instance_boxes_sample_small_and_large_instances_uniformly():
    image = np.zeros((1, 64, 64), dtype=np.float32)
    image[:, 32:] = 1
    mask = np.zeros((1, 64, 64), dtype=np.float32)
    mask[:, 4:8, 4:8] = 1
    mask[:, 48:58, 48:58] = 1
    arrays = [image, mask, mask.copy(), mask.copy()]
    boxes = np.array([[4, 4, 8, 8], [48, 48, 58, 58]], dtype=np.int32)
    image_means = []
    for seed in range(30):
        cropped = _crop(
            arrays,
            16,
            np.random.default_rng(seed),
            True,
            positive_probability=1.0,
            instance_boxes=boxes,
        )
        assert cropped[1].sum() > 0
        image_means.append(float(cropped[0].mean()))
    assert min(image_means) < 0.2
    assert max(image_means) > 0.8


def test_validation_crop_uses_an_instance_instead_of_global_centroid():
    image = np.zeros((1, 64, 64), dtype=np.float32)
    mask = np.zeros((1, 64, 64), dtype=np.float32)
    mask[:, 2:8, 2:8] = 1
    mask[:, 54:62, 54:62] = 1
    arrays = [image, mask, mask.copy(), mask.copy()]
    boxes = np.array([[2, 2, 8, 8], [54, 54, 62, 62]], dtype=np.int32)
    cropped = _crop(
        arrays,
        16,
        np.random.default_rng(1),
        False,
        instance_boxes=boxes,
    )
    assert cropped[1].sum() > 0


def test_semantic_target_is_union_of_instances():
    first = np.zeros((32, 32), dtype=bool)
    second = np.zeros((32, 32), dtype=bool)
    first[0:7, 2:9] = True
    second[18:30, 20:31] = True
    semantic = _semantic_target([first, second])
    np.testing.assert_array_equal(semantic.astype(bool), first | second)


def test_max_steps_uses_processed_batch_count():
    model = GlobalUNet(base=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    batch = {"image": torch.zeros(1, 1, 32, 32),
             "semantic_mask": torch.zeros(1, 1, 32, 32)}
    stats = train_one_epoch(model, [batch, batch, batch, batch], optimizer,
                            torch.device("cpu"), "global", max_steps=2)
    assert stats["batches"] == 2
    assert stats["loss"] > 0.1


def test_gradient_accumulation_counts_optimizer_steps():
    model = GlobalUNet(base=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    batch = {"image": torch.zeros(1, 1, 32, 32),
             "semantic_mask": torch.zeros(1, 1, 32, 32)}
    stats = train_one_epoch(
        model, [batch] * 5, optimizer, torch.device("cpu"), "global",
        gradient_accumulation=2,
    )
    assert stats["batches"] == 5
    assert stats["optimizer_steps"] == 3


def test_atomic_checkpoint_restores_full_state_and_rng(tmp_path):
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    model = GlobalUNet(base=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    config = {"model": {"kind": "global", "architecture": "compact_unet", "base": 2}}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, model, optimizer, 3, {"official_pq": 0.2}, scheduler=scheduler,
        config=config, model_kind="global", model_config=config["model"],
        global_step=11, best_metric=0.2, best_metric_name="official_pq",
    )
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    restored = restore_checkpoint(
        path, model, optimizer, scheduler=scheduler, expected_config=config
    )
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert restored.epoch == 3
    assert restored.global_step == 11
    assert restored.best_metric == 0.2
    assert actual == pytest.approx(expected)
    assert not list(tmp_path.glob("*.tmp"))


def test_inference_only_checkpoint_cannot_resume(tmp_path):
    model = GlobalUNet(base=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    path = tmp_path / "best.pt"
    save_checkpoint(
        path,
        model,
        None,
        1,
        {"official_pq": 0.1},
        config={},
        model_kind="global",
        model_config={"kind": "global", "architecture": "compact_unet", "base": 2},
    )
    with pytest.raises(ValueError, match="inference-only"):
        restore_checkpoint(path, model, optimizer, scheduler=scheduler)


def test_resnet34_unet_and_yaml_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n  encoder_weights: null\n"
        "training:\n  epochs: 1\n"
        "validation:\n  full_pq_enabled: false\n"
    )
    config = load_training_config(config_path)
    model = build_model("global", config["model"])
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 1, 64, 64))
    assert output.shape == (1, 1, 64, 64)


def test_official_diagnostic_distributions_exclude_zero_pairs():
    scores = _positive_score_distribution(
        [np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float64)]
    )
    assert scores["count"] == 3
    assert scores["mean"] == pytest.approx(7 / 12)
    assert sum(scores["histogram_counts"]) == 3

    degrees = _degree_distribution([0, 1, 2, 2, 3])
    assert degrees["degree_counts"] == {"0": 1, "1": 1, "2": 2, "3": 1}
    assert degrees["misses"] == 1
    assert degrees["single_partner"] == 1
    assert degrees["multiple_partners"] == 3
