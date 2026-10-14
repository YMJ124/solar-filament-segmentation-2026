"""Offline-safe runtime setup for Ultralytics commands."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


DEFAULT_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def prepare_ultralytics_runtime(
    config_root: str | Path = "artifacts/.ultralytics",
    *,
    font_candidates: tuple[Path, ...] = DEFAULT_FONT_CANDIDATES,
) -> Path:
    """Configure a project-local cache and provide the plotting font offline.

    Ultralytics appends ``Ultralytics`` to ``YOLO_CONFIG_DIR``. Its dataset
    validation downloads ``Arial.ttf`` when that exact filename is absent, so
    install a compatible system font under that name before importing it.
    """
    root = Path(config_root).resolve()
    user_config_dir = root / "Ultralytics"
    user_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(root)
    # All required competition assets are local. This also makes the optional
    # AMP reference-model check skip immediately instead of retrying downloads.
    os.environ["YOLO_OFFLINE"] = "true"
    # Reduce CUDA allocator fragmentation during high-resolution mask inference.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    matplotlib_dir = root / "matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))

    target = user_config_dir / "Arial.ttf"
    if target.is_file() and target.stat().st_size > 0:
        return user_config_dir

    for candidate in font_candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            shutil.copyfile(candidate, target)
            return user_config_dir

    raise RuntimeError(
        "Ultralytics requires a local plotting font, but no supported system "
        f"font was found. Place a valid TTF file at: {target}"
    )
