"""Semantic, boundary and distance multi-task network."""

from __future__ import annotations

from torch import nn

from .unet import _UNet


class InstanceSeparationNet(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.backbone = _UNet(1, 3, base)

    def forward(self, x):
        output = self.backbone(x)
        return {"semantic": output[:, 0:1], "boundary": output[:, 1:2], "distance": output[:, 2:3]}
