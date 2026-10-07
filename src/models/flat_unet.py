"""Flat U-Net adapted for the single-model competition pipeline.

Derived from Flat U-Net by gaofei zhu (Copyright 2024), distributed under
the MIT License. See ``THIRD_PARTY_NOTICES.md`` for provenance.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ChannelSelfAttentionBlock(nn.Module):
    """Full channel self-attention block used only at low resolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.key = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.value = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = self.query(inputs).unsqueeze(2)
        key = self.key(inputs).unsqueeze(1).permute(0, 1, 2, 4, 3)
        value = self.value(inputs)
        similarity = torch.matmul(query, key) / (query.size(-1) ** 0.5)
        scores = similarity.sum(dim=2).sum(dim=(-2, -1)) / (inputs.size(2) ** 2)
        scores = torch.softmax(scores, dim=1).unsqueeze(-1).unsqueeze(-1)
        return self.activation(self.norm(scores * value + inputs))


class SimplifiedChannelAttentionBlock(nn.Module):
    """Memory-efficient per-channel spatial attention block."""

    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.key = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.value = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)
        similarity = torch.matmul(query, key.transpose(2, 3)) / (query.size(-1) ** 0.5)
        scores = torch.softmax(similarity.sum(dim=(2, 3)), dim=1)
        scores = scores.unsqueeze(-1).unsqueeze(-1)
        return self.activation(self.norm(scores * value + inputs))


def _attention_block(channels: int, simplified: bool) -> nn.Module:
    return (
        SimplifiedChannelAttentionBlock(channels)
        if simplified else ChannelSelfAttentionBlock(channels)
    )


class _EncoderBlock(nn.Module):
    def __init__(self, channels: int, simplified: bool):
        super().__init__()
        self.block = _attention_block(channels, simplified)
        self.pool = nn.MaxPool2d(2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(inputs)
        return skip, self.pool(skip)


class _DecoderBlock(nn.Module):
    def __init__(self, channels: int, simplified: bool):
        super().__init__()
        self.reduce = nn.Conv2d(2 * channels, channels, 1)
        self.block = _attention_block(channels, simplified)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = nn.functional.interpolate(
            inputs, size=skip.shape[-2:], mode="bilinear", align_corners=True
        )
        return self.block(self.reduce(torch.cat([inputs, skip], dim=1)))


class FlatUNet(nn.Module):
    """Four-level U-Net retaining a constant channel width at every scale."""

    def __init__(self, channels: int = 32,
                 simplified_blocks: Sequence[bool | int] = (1, 1, 1, 1, 0)):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if len(simplified_blocks) != 5:
            raise ValueError("simplified_blocks must contain four levels and one bottleneck")
        flags = [bool(value) for value in simplified_blocks]
        self.stem = nn.Conv2d(1, channels, 3, padding=1)
        self.encoders = nn.ModuleList([
            _EncoderBlock(channels, flags[index]) for index in range(4)
        ])
        self.bottleneck = _attention_block(channels, flags[4])
        self.decoders = nn.ModuleList([
            _DecoderBlock(channels, flags[index]) for index in reversed(range(4))
        ])
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        skips = []
        for encoder in self.encoders:
            skip, features = encoder(features)
            skips.append(skip)
        features = self.bottleneck(features)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            features = decoder(features, skip)
        return self.head(features)
