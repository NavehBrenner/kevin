"""Offline residual-BC dataset loader (LAB-32) — turns the M4 corpus into
training samples for the Phase-1 residual policy.

This module owns the **loader-facing** contracts; the on-disk schema and the
per-episode-folder layout live in ``trajectory.py``. The dataset's only required
input is a dataset directory containing ``metadata.json`` (the manifest written
by the M4 generator): from it the loader discovers every episode, verifies the
``runs/episode_NNNNN/episode.npz`` files exist, and — when ``download=True`` —
regenerates any that are missing before training reads them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from ai_teleop.common.geometry import quat_to_6d
from ai_teleop.common.log import get_logger
from ai_teleop.data.generate import regenerate_from_metadata
from ai_teleop.data.images import (
    decode_frames,
    discover_frames,
    frame_index_for_steps,
    load_frame_stream,
)
from ai_teleop.data.schema import (
    EpisodeColumns,
    EpisodeMetadata,
    EpisodeSummary,
    ResBCDatasetMetadata,
)
from ai_teleop.data.trajectory import episode_imgs_dir, load_episode

log = get_logger("dataset")


@dataclass
class Episode:
    """One loaded episode-sequence — the item ``__getitem__`` returns.

    ``images``/``image_frame_index`` are populated only when the dataset is built with
    ``load_images=True`` (otherwise both are ``None``). They are a **compact** pair, not a
    dense per-step tensor: ``images`` holds only the episode's rendered (decimated) frames,
    and ``image_frame_index`` maps each of the ``T`` steps to the frame it should use. A
    dense ``(T, 3, 224, 224)`` tensor would be ~900 MB for a single undecimated 6000-step
    episode — see ``ai_teleop.data.images.load_frame_stream``.
    """

    episode_index: int
    command: Tensor  # (T, 9)  cmd_position(3) + cmd_orientation→6D(6)       — input
    force_torque: Tensor  # (T, 6)  wrist_ft, bias-subtracted               — input
    proprioception: Tensor  # (T, 24) ee_pos(3)+ee_6D(6)+joints(7)+joint_vel(7)+grip(1) — input
    delta: Tensor  # (T, 7)  delta_position(3)+orientation(3)+grip(1)       — BC target
    images: Tensor | None = None  # (F, 3, 224, 224) normalized wrist-cam frames, or None
    image_frame_index: Tensor | None = None  # (T,) long, step -> images index, or None


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


@dataclass
class EpisodeBatch:
    """A padded batch of episode-sequences — what ``collate_episodes`` returns.

    Streams are zero-padded to the batch's longest episode ``T_max`` (batch-first);
    ``lengths`` holds each episode's true step count so the train loop can pack or
    mask the padded tail. ``images``/``image_frame_index`` are ``None`` unless the
    dataset was built with ``load_images=True`` — see ``Episode`` for their compact
    (frames, per-step index) shape.
    """

    command: Tensor  # (B, T_max, 9)
    force_torque: Tensor  # (B, T_max, 6)
    proprioception: Tensor  # (B, T_max, 24)
    delta: Tensor  # (B, T_max, 7)
    lengths: Tensor  # (B,) long — true episode lengths
    images: Tensor | None = None  # (B, F_max, 3, 224, 224)
    image_frame_index: Tensor | None = None  # (B, T_max) long


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
    ``dataset_dir``. Empty list ⇒ the corpus is complete. The count drives the
    loader's log messages; the actual gap-filling is cache-scoped inside
    ``regenerate_from_metadata`` (it skips episodes already on disk), so the
    indices are not passed there.
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


def extract_training_episode(
    full_episode: tuple[EpisodeColumns, EpisodeMetadata],
    *,
    imgs_dir: Path | None = None,
    command_ee_delta: bool = False,
) -> Episode:
    """Assemble one **raw** (un-normalized) ``Episode`` from loaded npz columns.

    Orientations (command + the EE pose in proprio) are mapped to the continuous
    6D rep; scalar columns (gripper width, Δgrip) get a trailing axis before the
    per-step feature concat. Normalization is applied later, by the dataset.

    When ``imgs_dir`` is given, the episode's rendered wrist-cam frames are loaded
    alongside the trajectory (see ``ai_teleop.data.images.load_frame_stream``).

    ``command_ee_delta`` (LAB-106): append the raw ``cmd_position − ee_position``
    tracking-error vector to the proprioception stream, so the GRU is handed the
    quantity the residual is ∝ to (and that → 0 in free space) as its own channel,
    normalized as a unit rather than reconstructed from two separately-z-scored
    streams. Mirror this in ``LearnedResidual._assemble_streams`` for deployment.
    """
    columns, metadata = full_episode
    images, image_frame_index = (
        load_frame_stream(imgs_dir, n_steps=len(columns["step"]))
        if imgs_dir is not None
        else (None, None)
    )

    command = np.concatenate(
        [columns["cmd_position"], _quaternions_to_6d(columns["cmd_quaternion"])], axis=1
    )  # (T, 9)
    force_torque = columns["wrist_ft"]  # (T, 6)
    proprioception_streams = [
        columns["ee_pose"][:, :3],
        _quaternions_to_6d(columns["ee_pose"][:, 3:7]),
        columns["joint_positions"],
        columns["joint_velocities"],
        columns["gripper_width"][:, None],
    ]
    if command_ee_delta:
        proprioception_streams.append(columns["cmd_position"] - columns["ee_pose"][:, :3])  # (T, 3)
    proprioception = np.concatenate(proprioception_streams, axis=1)  # (T, 24) or (T, 27)
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
        images=images,
        image_frame_index=image_frame_index,
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
        train: bool = True,
        val_fraction: float = 0.2,
        seed: int = 0,
        norm_stats: NormStats | None = None,
        command_ee_delta: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.metadata_path = self.dataset_dir / "metadata.json"
        self.load_images = load_images
        self.download = download
        self.command_ee_delta = command_ee_delta

        with open(self.metadata_path, encoding="utf-8") as metadata_file:
            self.metadata: ResBCDatasetMetadata = json.load(metadata_file)

        missing_episodes = missing_episode_indices(self.metadata, self.dataset_dir)
        if missing_episodes:
            if not download:
                log.error(
                    "Missing %d episodes, but the download flag is set to False.",
                    len(missing_episodes),
                )
                raise FileNotFoundError(
                    f"Missing {len(missing_episodes)} episodes. Please make sure the dataset exists or set the download option to True."
                )
            log.info("Missing %d episodes. Regenerating from metadata...", len(missing_episodes))
            regenerate_from_metadata(self.metadata_path)

        train_episodes, val_episodes = split_episodes(
            self.metadata["episodes"], val_fraction=val_fraction, seed=seed
        )
        episode_summaries = train_episodes if train else val_episodes
        # Load the cheap vector streams eagerly; when load_images, store frame *paths* only and
        # the (cheap) per-step frame index — pixels decode lazily in __getitem__ (LAB-103). Holding
        # the whole decoded corpus resident OOMs before epoch 1 at the ~5k-episode LAB-82 scale.
        raw_episodes: list[Episode] = []
        self._frame_paths: list[list[Path] | None] = []
        for summary in episode_summaries:
            episode = extract_training_episode(
                load_episode(self.dataset_dir / summary["file"]),
                command_ee_delta=self.command_ee_delta,
            )
            paths: list[Path] | None = None
            if load_images:
                imgs_dir = episode_imgs_dir(self.dataset_dir / "runs", summary["episode_index"])
                frames = discover_frames(imgs_dir)
                if not frames:
                    raise FileNotFoundError(
                        f"no rendered frames found under {imgs_dir}; regenerate this dataset "
                        "with --record all (or --record images) to populate imgs/"
                    )
                paths = [path for _, path in frames]
                episode = replace(
                    episode,
                    image_frame_index=frame_index_for_steps(
                        [step for step, _ in frames], n_steps=episode.command.shape[0]
                    ),
                )
            raw_episodes.append(episode)
            self._frame_paths.append(paths)

        if norm_stats is None:
            if not train:
                raise ValueError("the val split requires norm_stats computed on the train split")
            # Vector streams only — extract_training_episode above never decoded images, so
            # norm-stats never hold the corpus resident (LAB-103).
            norm_stats = compute_norm_stats(raw_episodes)
        self.norm_stats = norm_stats
        self.episodes = [normalize_episode(episode, norm_stats) for episode in raw_episodes]

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> Episode:
        episode = self.episodes[index]
        paths = self._frame_paths[index]
        if paths is None:
            return episode
        # Lazy decode — runs in a DataLoader worker when num_workers>0, so resident RAM is
        # bounded to ~one batch of frames rather than the whole corpus.
        return replace(episode, images=decode_frames(paths))


def collate_episodes(batch: list[Episode]) -> EpisodeBatch:
    """Pad a list of variable-length episodes to ``T_max`` (batch-first) for a DataLoader.

    Each stream is zero-padded; ``lengths`` records the true per-episode step count
    so the train loop can pack/mask the padded tail. ``images``/``image_frame_index``
    are only assembled when every episode in the batch has them (``load_images=True``);
    otherwise both stay ``None``, matching the F/T-only (Phase-1) path.
    """
    has_images = all(episode.images is not None for episode in batch)
    images: Tensor | None = None
    image_frame_index: Tensor | None = None
    if has_images:
        images = pad_sequence([cast(Tensor, e.images) for e in batch], batch_first=True)
        image_frame_index = pad_sequence(
            [cast(Tensor, e.image_frame_index) for e in batch], batch_first=True
        )
    return EpisodeBatch(
        command=pad_sequence([episode.command for episode in batch], batch_first=True),
        force_torque=pad_sequence([episode.force_torque for episode in batch], batch_first=True),
        proprioception=pad_sequence(
            [episode.proprioception for episode in batch], batch_first=True
        ),
        delta=pad_sequence([episode.delta for episode in batch], batch_first=True),
        lengths=torch.tensor([episode.command.shape[0] for episode in batch], dtype=torch.long),
        images=images,
        image_frame_index=image_frame_index,
    )


def build_dataloaders(
    dataset_dir: str | Path,
    *,
    batch_size: int = 8,
    val_fraction: float = 0.2,
    seed: int = 0,
    download: bool = True,
    load_images: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    command_ee_delta: bool = False,
) -> tuple[DataLoader, DataLoader, NormStats]:
    """Build train + val ``DataLoader``s over an M4 dataset directory.

    The episode-level split is taken once with the given ``val_fraction``/``seed``,
    and the val dataset reuses the train-computed ``norm_stats`` so both splits —
    and, later, inference — share one normalization. The train loader is shuffled
    (episodes are the i.i.d. unit), the val loader is not. ``load_images`` requires
    the dataset was generated with ``--record all``/``--record images``; frames decode
    lazily per item, so ``num_workers>0`` parallelizes the decode into worker processes
    and keeps resident RAM bounded to ~one batch (LAB-103). Returns
    ``(train_loader, val_loader, norm_stats)``.
    """
    train_dataset = OfflineResidualBCDataset(
        dataset_dir,
        download=download,
        train=True,
        val_fraction=val_fraction,
        seed=seed,
        load_images=load_images,
        command_ee_delta=command_ee_delta,
    )
    val_dataset = OfflineResidualBCDataset(
        dataset_dir,
        download=download,
        train=False,
        val_fraction=val_fraction,
        seed=seed,
        load_images=load_images,
        norm_stats=train_dataset.norm_stats,
        command_ee_delta=command_ee_delta,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_episodes,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_episodes,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, train_dataset.norm_stats
