"""U-Net models used by the segmentation pipeline.

The compact models are retained for CPU tests and legacy checkpoints. Formal
training uses ImageNet-pretrained encoders for Model A and Model B.
"""

from __future__ import annotations

import torch
from torch import nn


def _build_smp_unet(encoder_name: str, encoder_weights: str | None,
                    in_channels: int) -> nn.Module:
    """Construct an SMP U-Net while keeping the dependency error actionable."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "segmentation-models-pytorch is required for pretrained U-Nets"
        ) from exc
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=1,
        activation=None,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class _UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=32):
        super().__init__()
        widths = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([ConvBlock(in_channels, widths[0]), ConvBlock(widths[0], widths[1]),
                                  ConvBlock(widths[1], widths[2]), ConvBlock(widths[2], widths[3])])
        self.pool = nn.MaxPool2d(2)
        self.bridge = ConvBlock(widths[3], widths[3] * 2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[3] * 2, widths[3], 2, 2),
                                 nn.ConvTranspose2d(widths[3], widths[2], 2, 2),
                                 nn.ConvTranspose2d(widths[2], widths[1], 2, 2),
                                 nn.ConvTranspose2d(widths[1], widths[0], 2, 2)])
        self.dec = nn.ModuleList([ConvBlock(widths[3] * 2, widths[3]), ConvBlock(widths[2] * 2, widths[2]),
                                  ConvBlock(widths[1] * 2, widths[1]), ConvBlock(widths[0] * 2, widths[0])])
        self.head = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for block in self.enc:
            x = block(x); skips.append(x); x = self.pool(x)
        x = self.bridge(x)
        for up, block, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


class GlobalUNet(_UNet):
    def __init__(self, base=32):
        super().__init__(1, 1, base)


class DetailUNet(_UNet):
    def __init__(self, base=24):
        super().__init__(2, 1, base)


class ResNet34UNet(nn.Module):
    """Single-channel U-Net with a ResNet34 encoder.

    Input normalization is part of the model, rather than the Dataset, so tiled
    training, validation and submission inference cannot silently diverge.
    ``encoder_weights=None`` is useful for offline tests; formal training should
    use ``"imagenet"``.
    """

    def __init__(self, encoder_weights: str | None = "imagenet",
                 input_mean: float = 0.326732, input_std: float = 0.223022):
        super().__init__()
        if input_std <= 0:
            raise ValueError("input_std must be positive")
        self.network = _build_smp_unet("resnet34", encoder_weights, 1)
        self.register_buffer(
            "input_mean", torch.tensor(float(input_mean), dtype=torch.float32)
        )
        self.register_buffer(
            "input_std", torch.tensor(float(input_std), dtype=torch.float32)
        )

    def forward(self, x):
        return self.network((x - self.input_mean) / self.input_std)


class EfficientNetB0DetailUNet(nn.Module):
    """Two-channel Model B that refines a frozen Model-A probability map.

    The first channel is the raw grayscale image in ``[0, 1]`` and uses the
    Fold-specific Model-A normalization. The second channel is the sigmoid
    probability from Model A and is mapped from ``[0, 1]`` to ``[-1, 1]``.
    Keeping both transforms inside the model prevents training, full-image
    validation, and submission inference from silently using different input
    conventions.
    """

    def __init__(self, encoder_weights: str | None = "imagenet",
                 input_mean: float = 0.326732, input_std: float = 0.223022):
        super().__init__()
        if input_std <= 0:
            raise ValueError("input_std must be positive")
        self.network = _build_smp_unet("efficientnet-b0", encoder_weights, 2)
        self.register_buffer(
            "input_mean", torch.tensor(float(input_mean), dtype=torch.float32)
        )
        self.register_buffer(
            "input_std", torch.tensor(float(input_std), dtype=torch.float32)
        )

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("Model B expects BCHW input with two channels")
        image = (x[:, :1] - self.input_mean) / self.input_std
        global_probability = x[:, 1:2] * 2.0 - 1.0
        return self.network(torch.cat([image, global_probability], dim=1))
