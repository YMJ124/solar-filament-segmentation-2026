#!/usr/bin/env python3
"""Gate-0 shape, parameter-count and memory checks for model configs."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.losses import GlobalLoss
from src.models import build_model
from src.training_config import load_training_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports = []
    for path in args.configs:
        config = load_training_config(path)
        model_config = {**config["model"], "encoder_weights": None}
        model = build_model("global", model_config).to(device).train()
        size = int(config["data"]["image_size"])
        batch_size = int(config["training"]["batch_size"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        image = torch.zeros(batch_size, 1, size, size, device=device)
        target = torch.zeros_like(image)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(image)
            loss = GlobalLoss()(output, target)
        loss.backward()
        report = {
            "config": path,
            "encoder": model_config.get("encoder_name"),
            "image_size": size,
            "batch_size": batch_size,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "output_shape": list(output.shape),
            "gpu_peak_mb": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda" else 0.0
            ),
            "device": str(device),
        }
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False))
        del image, target, output, loss, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if any(row["output_shape"][-2:] != [row["image_size"], row["image_size"]] for row in reports):
        raise RuntimeError("one or more candidates changed the spatial output shape")


if __name__ == "__main__":
    main()
