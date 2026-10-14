import os
from pathlib import Path

import pytest
import yaml

from scripts.train_yolo import training_arguments
from src.ultralytics_runtime import prepare_ultralytics_runtime


def test_prepare_ultralytics_runtime_installs_offline_font(tmp_path, monkeypatch):
    source = tmp_path / "source.ttf"
    source.write_bytes(b"test-font")
    config_root = tmp_path / "cache"
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)

    user_dir = prepare_ultralytics_runtime(
        config_root, font_candidates=(source,)
    )

    assert user_dir == config_root.resolve() / "Ultralytics"
    assert (user_dir / "Arial.ttf").read_bytes() == b"test-font"
    assert Path(os.environ["YOLO_CONFIG_DIR"]) == config_root.resolve()
    assert os.environ["YOLO_OFFLINE"] == "true"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert Path(os.environ["MPLCONFIGDIR"]) == config_root.resolve() / "matplotlib"


def test_prepare_ultralytics_runtime_fails_without_font(tmp_path):
    with pytest.raises(RuntimeError, match="requires a local plotting font"):
        prepare_ultralytics_runtime(
            tmp_path / "cache", font_candidates=(tmp_path / "missing.ttf",)
        )


def test_training_paths_are_absolute():
    config = {
        "run": {"project": "artifacts/instance", "name": "probe", "seed": 7},
        "data": {"dataset_yaml": "data/dataset.yaml"},
        "training": {"epochs": 1},
    }

    arguments = training_arguments(config)

    assert Path(arguments["project"]).is_absolute()
    assert Path(arguments["data"]).is_absolute()


def test_safe_yolo_configs_preserve_training_math():
    proxy = yaml.safe_load(
        Path("configs/instance/yolo26n_fold0_proxy.yaml").read_text()
    )["training"]
    candidate = yaml.safe_load(
        Path("configs/instance/yolo26n_fold0_maskratio2_proxy.yaml").read_text()
    )["training"]

    assert candidate["mask_ratio"] == 2
    assert candidate["workers"] == 0
    assert candidate["cache"] is False
    for key in (
        "epochs", "imgsz", "batch", "amp", "degrees", "fliplr", "flipud",
        "mosaic", "translate", "scale", "shear", "perspective", "hsv_h",
        "hsv_s", "hsv_v",
    ):
        assert candidate[key] == proxy[key]
