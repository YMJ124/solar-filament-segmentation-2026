#!/usr/bin/env python3
"""Fast Model-A sanity check: repeatedly optimize one fixed cached batch."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.dataset import FilamentDataset, collate_filaments
from src.data.folds import load_fold_manifest
from src.io_utils import atomic_write_json
from src.losses import GlobalLoss
from src.models import build_model
from src.training_config import load_training_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/global_resnet34_fold0.yaml")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output")
    parser.add_argument(
        "--require-decrease", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    config = load_training_config(args.config)
    if config["model"]["kind"] != "global":
        parser.error("overfit_batch currently supports Model A/global only")
    if args.steps < 2 or args.batch_size < 1:
        parser.error("steps must be >= 2 and batch-size must be >= 1")

    seed = int(config["run"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    root = Path(data_cfg["data_root"])
    annotation_file = root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    records = load_fold_manifest(data_cfg["fold_manifest"], annotation_file)
    records = [row for row in records if int(row["fold"]) != int(data_cfg["fold"])]
    dataset = FilamentDataset(
        root,
        records=records,
        fold_manifest=data_cfg["fold_manifest"],
        seed=seed,
        image_size=int(data_cfg["image_size"]),
        train=True,
        positive_crop_probability=float(data_cfg["positive_crop_probability"]),
        cache_dir=data_cfg.get("mask_cache_dir"),
    )
    dataset.set_epoch(0)
    batch = collate_filaments([
        dataset[index] for index in range(min(args.batch_size, len(dataset)))
    ])
    image = batch["image"].to(device)
    target = batch["semantic_mask"].to(device)
    model = build_model("global", config["model"]).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = GlobalLoss()
    losses = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = criterion(model(image), target)
        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach()))
        print(f"step {step + 1}/{args.steps}: loss={losses[-1]:.6f}")
    tail_size = min(3, len(losses) - 1)
    tail_mean = float(np.mean(losses[-tail_size:]))
    report = {
        "steps": args.steps,
        "batch_size": int(image.shape[0]),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "final_tail_mean": tail_mean,
        "loss_decreased": tail_mean < losses[0],
        "gpu_peak_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0
        ),
        "device": str(device),
    }
    print(report)
    if args.output:
        atomic_write_json(args.output, report)
    if args.require_decrease and not report["loss_decreased"]:
        raise RuntimeError("fixed-batch loss did not decrease")


if __name__ == "__main__":
    main()
