"""Wrist-camera frame loading — the M7 load-side counterpart to the render side
(`step_callbacks.EpisodeLogger`, `docs/data-schema.md`).

Rendered frames live at ``<episode_dir>/imgs/step_NNNNN.jpg``, decimated by the
generator's ``render_every`` cadence — so a frame stream is normally a strict
subset of an episode's step range, not one frame per step. This module decodes
those frames **once each** into a compact ``(F, 3, 224, 224)`` tensor (``F`` =
number of rendered frames) plus a per-step ``(T,)`` index that forward-fills
every step to its most recent frame. A naive dense ``(T, 3, 224, 224)`` tensor
would be ~900 MB for a single undecimated 6000-step episode — the compact form
is what makes ``load_images=True`` practical.

Forward-fill (not interpolation) is the correct alignment, not an
approximation: `docs/design/policy-model.md` specs the image branch as running
"once per new frame" and holding the same embedding between frames, so a
per-step index into the same decoded frame is exactly how the policy consumes
vision at inference.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

_FRAME_NAME_RE = re.compile(r"step_(\d+)\.jpg")

# ImageNet channel stats — the pretrained CNN backbone (docs/design/policy-model.md
# Decision B) was trained on this normalization; matching it at fine-tune time is what
# makes the pretrained weights a useful starting point rather than noise.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def discover_frames(imgs_dir: Path) -> list[tuple[int, Path]]:
    """Rendered frames in ``imgs_dir`` as ``(step, path)`` pairs, sorted by step."""
    frames = []
    for path in imgs_dir.glob("step_*.jpg"):
        match = _FRAME_NAME_RE.fullmatch(path.name)
        if match is not None:
            frames.append((int(match.group(1)), path))
    frames.sort(key=lambda item: item[0])
    return frames


def _load_and_normalize(path: Path) -> Tensor:
    """Decode one JPEG frame to a normalized ``(3, 224, 224)`` float32 tensor."""
    from PIL import Image

    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    frame = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
    return (frame - _IMAGENET_MEAN) / _IMAGENET_STD


def load_frame_stream(imgs_dir: Path, n_steps: int) -> tuple[Tensor, Tensor]:
    """Load an episode's rendered wrist-cam frames as a compact ``(images, frame_index)`` pair.

    ``images`` is ``(F, 3, 224, 224)`` — the ``F`` unique decoded frames, in step order.
    ``frame_index`` is ``(n_steps,)`` long — for each step, the index into ``images`` of the
    most recent frame at or before it (steps before the first rendered frame use frame 0).

    Raises ``FileNotFoundError`` if ``imgs_dir`` has no rendered frames — the caller asked for
    images from an episode generated without ``--record all`` / ``--record images``.
    """
    frames = discover_frames(imgs_dir)
    if not frames:
        raise FileNotFoundError(
            f"no rendered frames found under {imgs_dir}; regenerate this dataset with "
            "--record all (or --record images) to populate imgs/"
        )

    images = torch.stack([_load_and_normalize(path) for _, path in frames])  # (F, 3, 224, 224)

    frame_steps = np.array([step for step, _ in frames])
    step_range = np.arange(n_steps)
    # Index of the last frame_step <= step; clamp steps before the first frame to frame 0.
    frame_index = np.clip(np.searchsorted(frame_steps, step_range, side="right") - 1, 0, None)
    return images, torch.from_numpy(frame_index).long()
