"""Model loading and probability-level ensemble helpers."""

from __future__ import annotations

import numpy as np
import torch

from src.postprocess.instances import fuse_probabilities


MODEL_ORDER = ("global", "detail", "instance")


def _semantic_tensor(output):
    return output["semantic"] if isinstance(output, dict) else output


def _batch_output_maps(outputs: dict, batch_index: int = 0):
    components = {
        key: _semantic_tensor(output)[batch_index, 0].detach().cpu().numpy()
        for key, output in outputs.items()
    }
    boundary = (
        outputs["instance"]["boundary"][batch_index, 0].detach().cpu().numpy()
        if "instance" in outputs else None
    )
    distance = (
        outputs["instance"]["distance"][batch_index, 0].detach().cpu().numpy()
        if "instance" in outputs else None
    )
    return components, boundary, distance


def _fuse_component_maps(components: dict, weights) -> np.ndarray:
    probabilities, selected_weights = [], []
    for key, weight in zip(MODEL_ORDER, weights):
        if key in components:
            probabilities.append(components[key])
            selected_weights.append(weight)
    return fuse_probabilities(probabilities, selected_weights)


@torch.no_grad()
def predict_batch(models: dict, image, device):
    image = image.to(device, non_blocking=True)
    outputs = {}
    if "global" in models:
        outputs["global"] = torch.sigmoid(models["global"](image))
    if "detail" in models:
        if "global" not in outputs:
            raise ValueError("detail inference requires a global Model-A teacher")
        global_probability = outputs["global"]
        outputs["detail"] = torch.sigmoid(models["detail"](torch.cat([image, global_probability], dim=1)))
    if "instance" in models:
        instance_output = models["instance"](image)
        outputs["instance"] = {key: torch.sigmoid(value) for key, value in instance_output.items()}
    return outputs


def fuse_outputs(outputs: dict, weights=(0.45, 0.30, 0.25), batch_index: int = 0) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    components, boundary, distance = _batch_output_maps(outputs, batch_index)
    return _fuse_component_maps(components, weights), boundary, distance


def sliding_starts(length: int, tile_size: int, step: int) -> list[int]:
    if tile_size >= length:
        return [0]
    starts = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _blend_window(tile_size: int) -> np.ndarray:
    axis = np.hanning(tile_size).astype(np.float32)
    window = np.outer(axis, axis)
    return np.clip(window, 1e-3, None)


@torch.inference_mode()
def predict_sliding_window(models: dict, image: np.ndarray, device, tile_size: int = 512,
                           overlap: float = 0.25, weights=(0.45, 0.30, 0.25),
                           tile_batch_size: int = 1, return_components: bool = False):
    """Predict a full image while keeping model memory bounded."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("image must be a 2D grayscale array")
    if tile_batch_size < 1:
        raise ValueError("tile_batch_size must be >= 1")
    height, width = image.shape
    if tile_size < 1 or tile_size > min(height, width):
        raise ValueError("tile_size must be positive and no larger than the image")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    step = max(1, int(tile_size * (1.0 - overlap)))
    fused_sum = np.zeros_like(image, dtype=np.float32)
    fused_count = np.zeros_like(image, dtype=np.float32)
    boundary_sum = np.zeros_like(image, dtype=np.float32) if "instance" in models else None
    distance_sum = np.zeros_like(image, dtype=np.float32) if "instance" in models else None
    component_sums = (
        {key: np.zeros_like(image, dtype=np.float32) for key in models}
        if return_components else None
    )
    blend = _blend_window(tile_size)
    coordinates = [
        (y0, x0)
        for y0 in sliding_starts(height, tile_size, step)
        for x0 in sliding_starts(width, tile_size, step)
    ]
    for start in range(0, len(coordinates), tile_batch_size):
        batch_coordinates = coordinates[start:start + tile_batch_size]
        tiles = np.stack([
            image[y0:y0 + tile_size, x0:x0 + tile_size]
            for y0, x0 in batch_coordinates
        ])
        tensor = torch.from_numpy(tiles[:, None]).to(device)
        outputs = predict_batch(models, tensor, device)
        for batch_index, (y0, x0) in enumerate(batch_coordinates):
            output_maps, boundary, distance = _batch_output_maps(outputs, batch_index)
            fused = _fuse_component_maps(output_maps, weights)
            fused_sum[y0:y0 + tile_size, x0:x0 + tile_size] += fused * blend
            fused_count[y0:y0 + tile_size, x0:x0 + tile_size] += blend
            if boundary_sum is not None:
                boundary_sum[y0:y0 + tile_size, x0:x0 + tile_size] += boundary * blend
                distance_sum[y0:y0 + tile_size, x0:x0 + tile_size] += distance * blend
            if component_sums is not None:
                for key in models:
                    component_sums[key][
                        y0:y0 + tile_size, x0:x0 + tile_size
                    ] += output_maps[key] * blend
    fused = fused_sum / np.maximum(fused_count, 1)
    boundary = boundary_sum / np.maximum(fused_count, 1) if boundary_sum is not None else None
    distance = distance_sum / np.maximum(fused_count, 1) if distance_sum is not None else None
    if not return_components:
        return fused, boundary, distance
    components = {
        f"{key}_probability": value / np.maximum(fused_count, 1)
        for key, value in component_sums.items()
    }
    return fused, boundary, distance, components
