"""Reliable training, validation and checkpoint primitives."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch
from .losses import GlobalLoss

CHECKPOINT_SCHEMA_VERSION = 2


def _loss_for(kind, loss_config=None):
    if kind != "global":
        raise ValueError(f"single-model core only supports kind='global', got {kind!r}")
    loss_config = dict(loss_config or {})
    return GlobalLoss(
        mode=str(loss_config.get("name", "bce_dice")),
        max_positive_weight=float(loss_config.get("max_positive_weight", 20.0)),
    )


def _batch_loss(model, batch, device, criterion):
    image = batch["image"].to(device, non_blocking=True)
    semantic = batch["semantic_mask"].to(device, non_blocking=True)
    return criterion(model(image), semantic)


def _optimizer_step(model, optimizer, scaler, use_amp: bool, gradient_clip_norm: float) -> None:
    if use_amp:
        scaler.unscale_(optimizer)
    if gradient_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    if use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def train_one_epoch(model, loader, optimizer, device, kind="global", scaler=None,
                    max_steps=None, gradient_accumulation=1,
                    gradient_clip_norm=1.0, loss_config=None):
    """Train one epoch with bounded AMP and true gradient accumulation."""
    from tqdm import tqdm

    if gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be >= 1")
    model.train()
    criterion = _loss_for(kind, loss_config)
    total = 0.0
    processed = 0
    optimizer_steps = 0
    use_amp = scaler is not None and device.type == "cuda"
    planned_batches = len(loader)
    if max_steps is not None:
        planned_batches = min(planned_batches, int(max_steps))
    progress = tqdm(loader, desc=f"train:{kind}", leave=False)
    started = perf_counter()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step, batch in enumerate(progress):
        if step >= planned_batches:
            break
        group_start = (processed // gradient_accumulation) * gradient_accumulation
        group_size = min(gradient_accumulation, planned_batches - group_start)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = _batch_loss(model, batch, device, criterion)
            scaled_loss = loss / group_size
        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        total += float(loss.detach())
        processed += 1
        if processed % gradient_accumulation == 0 or processed == planned_batches:
            _optimizer_step(
                model, optimizer, scaler, use_amp, float(gradient_clip_norm)
            )
            optimizer_steps += 1
        memory = (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0
        )
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}", gpu_mb=f"{memory:.0f}")
    return {
        "loss": total / max(1, processed),
        "batches": processed,
        "optimizer_steps": optimizer_steps,
        "seconds": perf_counter() - started,
        "gpu_peak_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0
        ),
    }


@torch.no_grad()
def validate_loss(model, loader, device, kind="global", max_batches=None,
                  loss_config=None):
    model.eval()
    criterion = _loss_for(kind, loss_config)
    total = 0.0
    processed = 0
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        loss = _batch_loss(model, batch, device, criterion)
        total += float(loss)
        processed += 1
    return {"loss": total / max(1, processed), "batches": processed}


def config_fingerprint(config: Mapping[str, Any] | None) -> str:
    payload = json.dumps(config or {}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # A checkpoint loaded with map_location="cuda" moves every tensor,
    # including the CPU generator state. PyTorch requires RNG state tensors on
    # CPU even when the model itself is restored on a GPU.
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
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


def save_checkpoint(path: str | Path, model, optimizer, epoch, metrics, scaler=None,
                    scheduler=None, config=None, model_kind=None, model_config=None,
                    global_step=0, best_metric=None, best_metric_name=None):
    """Atomically save all state required to resume the next epoch."""
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
        "metrics": dict(metrics),
        "config": dict(config or {}),
        "config_fingerprint": config_fingerprint(config),
        "model_kind": model_kind,
        "model_config": dict(model_config or {}),
        "rng_state": capture_rng_state(),
    }
    _atomic_torch_save(payload, path)


@dataclass(frozen=True)
class CheckpointState:
    epoch: int
    global_step: int
    best_metric: float | None
    best_metric_name: str | None
    metrics: dict[str, Any]


def restore_checkpoint(path: str | Path, model, optimizer=None, scaler=None, scheduler=None,
                       device="cpu", expected_config=None, restore_rng=True,
                       require_training_state=True) -> CheckpointState:
    """Restore a trusted checkpoint and reject incompatible run configuration."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise ValueError("resume requires a full training checkpoint")
    if int(checkpoint.get("schema_version", 0)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema: {checkpoint.get('schema_version')}"
        )
    if require_training_state:
        missing_state = []
        if optimizer is not None and checkpoint.get("optimizer") is None:
            missing_state.append("optimizer")
        if scheduler is not None and checkpoint.get("scheduler") is None:
            missing_state.append("scheduler")
        if scaler is not None and scaler.is_enabled() and checkpoint.get("scaler") is None:
            missing_state.append("AMP scaler")
        if missing_state:
            raise ValueError(
                "checkpoint is inference-only and cannot resume training; missing "
                + ", ".join(missing_state)
            )
    if expected_config is not None and checkpoint.get("config_fingerprint"):
        expected = config_fingerprint(expected_config)
        if checkpoint["config_fingerprint"] != expected:
            raise ValueError(
                "resume configuration differs from checkpoint: "
                f"{expected} != {checkpoint['config_fingerprint']}"
            )
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if restore_rng:
        restore_rng_state(checkpoint.get("rng_state"))
    return CheckpointState(
        epoch=int(checkpoint.get("epoch", 0)),
        global_step=int(checkpoint.get("global_step", 0)),
        best_metric=checkpoint.get("best_metric"),
        best_metric_name=checkpoint.get("best_metric_name"),
        metrics=dict(checkpoint.get("metrics") or {}),
    )
