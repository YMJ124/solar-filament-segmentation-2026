"""Configuration-driven model construction and checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .unet import GlobalUNet, SMPUNet


def normalize_model_config(kind: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a complete, serializable model configuration."""
    config = dict(config or {})
    config["kind"] = kind
    if kind != "global":
        raise ValueError(f"single-model core only supports kind='global', got {kind!r}")
    config.setdefault("architecture", "compact_unet")
    if config["architecture"] == "resnet34_unet":
        config["architecture"] = "smp_unet"
        config.setdefault("encoder_name", "resnet34")
    return config


def build_model(kind: str, config: Mapping[str, Any] | None = None,
                initialize_pretrained: bool = True):
    """Build the single semantic model from its recorded architecture config."""
    cfg = normalize_model_config(kind, config)
    architecture = str(cfg["architecture"])
    if kind == "global" and architecture == "compact_unet":
        return GlobalUNet(base=int(cfg.get("base", 32)))
    if kind == "global" and architecture == "smp_unet":
        weights = cfg.get("encoder_weights", "imagenet") if initialize_pretrained else None
        return SMPUNet(
            encoder_name=str(cfg.get("encoder_name", "resnet34")),
            encoder_weights=weights,
            input_mean=float(cfg.get("input_mean", 0.326732)),
            input_std=float(cfg.get("input_std", 0.223022)),
        )
    raise ValueError(f"unsupported model configuration: kind={kind!r}, architecture={architecture!r}")


def checkpoint_model_config(checkpoint: Mapping[str, Any], expected_kind: str | None = None) -> tuple[str, dict]:
    """Extract model metadata, including backward compatibility for stage-2 files."""
    config = checkpoint.get("config") or {}
    model_config = checkpoint.get("model_config") or config.get("model") or {}
    kind = checkpoint.get("model_kind") or model_config.get("kind") or config.get("kind") or expected_kind or "global"
    if not kind:
        raise ValueError("checkpoint does not record model kind; pass expected_kind")
    if expected_kind and kind != expected_kind:
        raise ValueError(f"checkpoint kind {kind!r} does not match expected {expected_kind!r}")
    return str(kind), normalize_model_config(str(kind), model_config)


def load_model_from_checkpoint(path: str | Path, device,
                               expected_kind: str | None = None):
    """Construct and load the exact architecture recorded in a trusted checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        if expected_kind is None:
            raise ValueError("raw state-dict checkpoints require expected_kind")
        checkpoint = {"model": checkpoint, "model_kind": expected_kind}
    kind, model_config = checkpoint_model_config(checkpoint, expected_kind)
    model = build_model(kind, model_config, initialize_pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()
