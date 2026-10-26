#!/usr/bin/env python3
"""Train the fixed-budget P5-C1 crop refiner on non-heldout OOF crops."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.instance.p5_refiner import (
    BoundaryWeightedDiceLoss,
    RefinerCropDataset,
    build_refiner_model,
    physical_image_sampling_weights,
)
from src.instance.yolo_cache import config_sha256, sha256_file
from src.io_utils import atomic_write_json


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _device(raw: str) -> torch.device:
    if raw == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {raw!r} requested but CUDA is unavailable; refusing CPU fallback"
        )
    device = torch.device(f"cuda:{int(raw)}")
    torch.cuda.set_device(device)
    return device


def _seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/instance/stage_p5/p5_c1_crop_refiner_fold0.yaml",
        type=Path,
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="0", help="CUDA index or 'cpu' for smoke tests")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict) or int(config.get("schema_version", 0)) != 1:
        raise ValueError("P5 refiner config must use schema_version=1")
    train_cfg = config["training"]
    dataset_dir = args.dataset_dir or Path(config["data"]["training_crop_dir"])
    output_dir = args.output_dir or Path(train_cfg["output_dir"])
    epochs = int(args.epochs or train_cfg["epochs"])
    if epochs < 1 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError("epochs and max-steps must be positive")
    final_path = output_dir / "final.pt"
    last_path = output_dir / "last.pt"
    log_path = output_dir / "epochs.jsonl"
    if not args.resume and any(path.exists() for path in (final_path, last_path, log_path)):
        raise FileExistsError(
            f"training output already exists in {output_dir}; use --resume or a new path"
        )
    if args.resume and not args.resume.is_file():
        raise FileNotFoundError(args.resume)

    index_path = dataset_dir / "index.json"
    index = json.loads(index_path.read_text())
    current_config_sha256 = sha256_file(args.config)
    if index.get("config_sha256") != current_config_sha256:
        raise ValueError(
            "training dataset was built from a different refiner config; rebuild it"
        )
    expected_folds = sorted(int(v) for v in config["experiment"]["training_folds"])
    if index.get("training_folds") != expected_folds:
        raise ValueError("training dataset uses different folds")
    if not index.get("complete") and args.max_steps is None:
        raise ValueError("partial crop datasets are allowed only with --max-steps smoke runs")
    expected_heldout = config["experiment"].get("heldout_fold")
    if index.get("heldout_fold") != expected_heldout:
        raise ValueError("training dataset heldout fold differs from config")
    if bool(index.get("final_fit", False)) != bool(
        config["experiment"].get("final_fit", False)
    ):
        raise ValueError("training dataset final-fit mode differs from config")

    seed = int(train_cfg["seed"])
    _seed(seed, bool(train_cfg["deterministic"]))
    device = _device(args.device)
    dataset = RefinerCropDataset(dataset_dir, augment=True)
    weights = physical_image_sampling_weights(dataset.records)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        sampler=sampler,
        num_workers=int(train_cfg["workers"]),
        pin_memory=device.type == "cuda",
    )
    model = build_refiner_model(config["model"]).to(device)
    criterion = BoundaryWeightedDiceLoss(
        boundary_weight=float(config["loss"]["boundary_weight"]),
        boundary_kernel=int(config["loss"]["boundary_kernel"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and bool(train_cfg["amp"])
    )
    start_epoch = 1
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("config_sha256") != sha256_file(args.config):
            raise ValueError("resume checkpoint config hash differs")
        if checkpoint.get("dataset_index_sha256") != sha256_file(index_path):
            raise ValueError("resume checkpoint dataset hash differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "report_type": "p5_refiner_training_manifest_v1",
        "config": config,
        "config_sha256": current_config_sha256,
        "config_fingerprint": config_sha256(config),
        "dataset_dir": str(dataset_dir),
        "dataset_index_sha256": sha256_file(index_path),
        "heldout_fold": index.get("heldout_fold"),
        "final_fit": bool(index.get("final_fit", False)),
        "training_folds": index["training_folds"],
        "samples": len(dataset),
        "physical_images": int(index["physical_images"]),
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fixed_epoch_budget": epochs,
        "max_steps_smoke_override": args.max_steps,
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    print(manifest, flush=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        started = perf_counter()
        progress = tqdm(loader, desc=f"P5-C1 epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress, start=1):
            if args.max_steps is not None and step > args.max_steps:
                break
            inputs = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=scaler.is_enabled(),
            ):
                loss = criterion(model(inputs), target)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += float(loss.detach())
            batches += 1
            global_step += 1
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": total / max(1, batches),
            "batches": batches,
            "global_step": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": perf_counter() - started,
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        checkpoint = {
            "schema_version": 1,
            "report_type": "p5_refiner_checkpoint_v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "metrics": row,
            "config": config,
            "config_sha256": current_config_sha256,
            "config_fingerprint": config_sha256(config),
            "dataset_index_sha256": sha256_file(index_path),
            "heldout_fold": index.get("heldout_fold"),
            "final_fit": bool(index.get("final_fit", False)),
            "training_folds": index["training_folds"],
            "dataset_samples": len(dataset),
            "max_steps_smoke_override": args.max_steps,
        }
        _atomic_torch_save(last_path, checkpoint)
        print(row, flush=True)
    _atomic_torch_save(final_path, checkpoint)
    print(f"wrote fixed-budget final checkpoint to {final_path}", flush=True)


if __name__ == "__main__":
    main()
