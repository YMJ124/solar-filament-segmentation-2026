"""Memory-bounded inference for one semantic segmentation model."""

from __future__ import annotations

import numpy as np
import torch


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
    return np.clip(np.outer(axis, axis), 1e-3, None)


@torch.inference_mode()
def predict_sliding_window(model, image: np.ndarray, device,
                           tile_size: int = 512, overlap: float = 0.25,
                           tile_batch_size: int = 1) -> np.ndarray:
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
    probability_sum = np.zeros_like(image, dtype=np.float32)
    probability_count = np.zeros_like(image, dtype=np.float32)
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
        probability = torch.sigmoid(model(tensor))[:, 0].detach().cpu().numpy()
        for index, (y0, x0) in enumerate(batch_coordinates):
            probability_sum[y0:y0 + tile_size, x0:x0 + tile_size] += (
                probability[index] * blend
            )
            probability_count[y0:y0 + tile_size, x0:x0 + tile_size] += blend
    return probability_sum / np.maximum(probability_count, 1e-6)
