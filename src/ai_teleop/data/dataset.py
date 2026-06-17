"""Offline residual-BC dataset loader (LAB-32) — turns the M4 corpus into
training samples for the Phase-1 residual policy.

This module owns the **loader-facing** contracts; the on-disk schema and the
per-episode-folder layout live in ``trajectory.py``. The dataset's only required
input is a dataset directory containing ``metadata.json`` (the manifest written
by the M4 generator): from it the loader discovers every episode, verifies the
``runs/episode_NNNNN/episode.npz`` files exist, and — when ``download=True`` —
regenerates any that are missing before training reads them.

Note for the type contracts below: a TypeScript ``interface`` over JSON maps to a
``TypedDict`` (see ``ResBCDatasetMetadata`` / ``EpisodeMetadata`` /
``EpisodeSummary`` in ``schema.py``); an ``interface`` describing an object you
*construct and return* maps to a ``@dataclass`` — that's ``Episode`` here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ai_teleop.common.log import get_logger
from ai_teleop.common.utils.rotations import quat_to_6d
from ai_teleop.data.generate import regenerate_from_metadata
from ai_teleop.data.schema import (
    EpisodeColumns,
    EpisodeMetadata,
    EpisodeSummary,
    ResBCDatasetMetadata,
)
from ai_teleop.data.trajectory import load_episode

log = get_logger("dataset")


@dataclass
class Episode:
    """One loaded episode-sequence — the item ``__getitem__`` returns.

    ``images`` is populated only when the dataset is
    built with ``load_images=True`` *and* the episode has rendered frames on disk
    (vision is M7 — normally ``None``).

    The exact training-sample shape (full episode-sequence vs. windowed per-step,
    early-fused vs. separate streams) is still under revision — see the M5 spec
    banner — so treat these fields as a starting contract you can reshape.
    """

    episode_index: int
    command: Tensor  # (T, 9)  cmd_position(3) + cmd_orientation→6D(6)       — input
    force_torque: Tensor  # (T, 6)  wrist_ft, bias-subtracted               — input
    proprioception: Tensor  # (T, 24) ee_pos(3)+ee_6D(6)+joints(7)+joint_vel(7)+grip(1) — input
    delta: Tensor  # (T, 7)  delta_position(3)+orientation(3)+grip(1)       — BC target
    images: Tensor | None = None  # (T, H, W, 3) uint8 wrist-cam frames, or None


INPUT_STREAMS: tuple[str, ...] = ("command", "force_torque", "proprioception")


@dataclass(frozen=True)
class NormStats:
    """Per-channel mean/std for each input stream, computed on the **train** split
    and reused unchanged for val and at inference (stash in the checkpoint). The
    target (``delta``) is intentionally left raw — the BC loss owns its per-channel
    weighting.
    """

    mean: dict[str, Tensor]
    std: dict[str, Tensor]


def split_episodes(
    episodes: list[EpisodeSummary],
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[EpisodeSummary], list[EpisodeSummary]]:
    """Deterministic **episode-level** train/val split; returns ``(train, val)``."""

    ordered = sorted(episodes, key=lambda summary: summary["episode_index"])
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    n_val = int(len(ordered) * val_fraction)
    val = [ordered[i] for i in permutation[:n_val]]
    train = [ordered[i] for i in permutation[n_val:]]
    return train, val


def missing_episode_indices(metadata: ResBCDatasetMetadata, dataset_dir: str | Path) -> list[int]:
    """Episode indices listed in ``metadata`` whose ``episode.npz`` is absent on disk.

    ``summary["file"]`` is **dataset-relative**, so it is resolved against
    ``dataset_dir``. Empty list ⇒ the corpus is complete; feed a non-empty result
    to ``regenerate_from_metadata`` to fill the gaps when ``download=True``.
    """
    root = Path(dataset_dir)
    return [
        summary["episode_index"]
        for summary in metadata["episodes"]
        if not (root / summary["file"]).exists()
    ]


def _quaternions_to_6d(quaternions: np.ndarray) -> np.ndarray:
    """(T, 4) w-first quaternions → (T, 6) continuous rotations (per-step ``quat_to_6d``).

    ``quat_to_6d`` is single-quaternion (``mju_quat2Mat`` takes one quat), so map
    it over the time axis. This is the one-time eager cost at dataset construction.
    """
    return np.stack([quat_to_6d(quaternion) for quaternion in quaternions])


def extract_training_episode(full_episode: tuple[EpisodeColumns, EpisodeMetadata]) -> Episode:
    """Assemble one **raw** (un-normalized) ``Episode`` from loaded npz columns.

    Orientations (command + the EE pose in proprio) are mapped to the continuous
    6D rep; scalar columns (gripper width, Δgrip) get a trailing axis before the
    per-step feature concat. Normalization is applied later, by the dataset.
    """
    columns, metadata = full_episode

    command = np.concatenate(
        [columns["cmd_position"], _quaternions_to_6d(columns["cmd_quaternion"])], axis=1
    )  # (T, 9)
    force_torque = columns["wrist_ft"]  # (T, 6)
    proprioception = np.concatenate(
        [
            columns["ee_pose"][:, :3],
            _quaternions_to_6d(columns["ee_pose"][:, 3:7]),
            columns["joint_positions"],
            columns["joint_velocities"],
            columns["gripper_width"][:, None],
        ],
        axis=1,
    )  # (T, 24)
    delta = np.concatenate(
        [columns["delta_position"], columns["delta_orientation"], columns["delta_grip"][:, None]],
        axis=1,
    )  # (T, 7)

    return Episode(
        episode_index=metadata["episode_index"],
        command=torch.tensor(command, dtype=torch.float32),
        force_torque=torch.tensor(force_torque, dtype=torch.float32),
        proprioception=torch.tensor(proprioception, dtype=torch.float32),
        delta=torch.tensor(delta, dtype=torch.float32),
    )


def compute_norm_stats(episodes: list[Episode], *, eps: float = 1e-6) -> NormStats:
    """Per-channel mean/std over all steps of all episodes, per input stream.

    Call on the **train** split only. ``std`` is floored at ``eps`` so constant
    channels (e.g. an unused DoF) don't divide by zero.
    """
    mean: dict[str, Tensor] = {}
    std: dict[str, Tensor] = {}
    for stream in INPUT_STREAMS:
        all_steps = torch.cat([getattr(episode, stream) for episode in episodes], dim=0)
        mean[stream] = all_steps.mean(dim=0)
        std[stream] = all_steps.std(dim=0).clamp_min(eps)
    return NormStats(mean=mean, std=std)


def normalize_episode(episode: Episode, stats: NormStats) -> Episode:
    """Z-score each input stream with ``stats``; leave the Δ target (and images) raw."""
    normalized = {
        stream: (getattr(episode, stream) - stats.mean[stream]) / stats.std[stream]
        for stream in INPUT_STREAMS
    }
    return replace(episode, **normalized)


# currently only implements offline
class OfflineResidualBCDataset(Dataset):
    """BC training set over an M4 dataset directory.

    From the dataset manifest (``metadata.json``) it discovers every episode,
    regenerates any missing ``episode.npz`` (when ``download``), takes the
    episode-level train/val split, loads each assigned episode, assembles the
    ``Episode`` streams (quat→6D), and z-scores the input streams. Normalization
    stats are computed on the **train** split only and exposed as ``norm_stats``;
    build the val split with ``norm_stats=<train_dataset>.norm_stats`` so both
    splits — and, later, inference — share one normalization.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        load_images: bool = False,
        download: bool = True,
        offline: bool = True,
        train: bool = True,
        norm_stats: NormStats | None = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.metadata_path = self.dataset_dir / "metadata.json"
        self.load_images = load_images
        self.download = download

        with open(self.metadata_path, encoding="utf-8") as metadata_file:
            self.metadata: ResBCDatasetMetadata = json.load(metadata_file)

        missing_episodes = missing_episode_indices(self.metadata, self.dataset_dir)
        if len(missing_episodes) > 0:
            if not download:
                log.error(
                    f"Missing {len(missing_episodes)} episodes, but the download flag is set to False."
                )
                raise FileNotFoundError(
                    f"Missing {len(missing_episodes)} episodes. Please make sure the dataset exists or set the download option to True."
                )
            else:
                log.info(f"Missing {len(missing_episodes)} episodes. Regenerating from metadata...")
                regenerate_from_metadata(self.metadata_path)

        if offline and load_images:
            log.warning("load_images=True and offline=True are incompatible.")

        train_episodes, val_episodes = split_episodes(self.metadata["episodes"])
        episodes_summary = train_episodes if train else val_episodes
        raw_episodes = [
            extract_training_episode(load_episode(self.dataset_dir / summary["file"]))
            for summary in episodes_summary
        ]

        if norm_stats is None:
            if not train:
                raise ValueError("the val split requires norm_stats computed on the train split")
            norm_stats = compute_norm_stats(raw_episodes)
        self.norm_stats = norm_stats
        self.episodes = [normalize_episode(episode, norm_stats) for episode in raw_episodes]
        self.length = len(self.episodes)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Episode:
        return self.episodes[index]
