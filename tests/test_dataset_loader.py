"""Tests for the M5 offline BC dataset loader (LAB-32).

Self-contained: a fixture builds a tiny synthetic dataset on disk (a few short
episodes written with ``EpisodeRecorder`` + a ``metadata.json`` manifest) in a
temp dir, so the suite needs no committed corpus and runs anywhere — the real
episode ``.npz`` files are gitignored. Everything stays on CPU.

Exercises the loader-facing contracts: the deterministic episode-level split, the
assembled per-episode streams + their train-only normalization, episode-edge
padding in ``collate_episodes``, and the ``build_dataloaders`` factory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ai_teleop.data.dataset import (
    INPUT_STREAMS,
    Episode,
    EpisodeBatch,
    NormStats,
    OfflineResidualBCDataset,
    build_dataloaders,
    collate_episodes,
    split_episodes,
)
from ai_teleop.data.images import load_frame_stream
from ai_teleop.data.trajectory import EpisodeRecorder, episode_imgs_dir, episode_npz_path

# Per-stream channel widths — the (T, C) feature contract from the Episode dataclass.
STREAM_WIDTHS: dict[str, int] = {"command": 9, "force_torque": 6, "proprioception": 24}
DELTA_WIDTH = 7
N_EPISODES = 16  # enough that a 0.2 val split (→ 3) differs across seeds non-flakily


def _unit_quaternion(rng: np.random.Generator) -> np.ndarray:
    quaternion = rng.standard_normal(4)
    return quaternion / np.linalg.norm(quaternion)


def _random_row(rng: np.random.Generator, step: int) -> dict[str, object]:
    """One schema-valid per-step row with variance on every channel (so the
    normalization check sees a non-zero std) and valid unit quaternions."""
    return {
        "step": step,
        "sim_time": step * 0.002,
        "wrist_ft": rng.standard_normal(6),
        "joint_positions": rng.standard_normal(7),
        "joint_velocities": rng.standard_normal(7),
        "ee_pose": np.concatenate([rng.standard_normal(3), _unit_quaternion(rng)]),
        "gripper_width": rng.random(),
        "cmd_position": rng.standard_normal(3),
        "cmd_quaternion": _unit_quaternion(rng),
        "cmd_grip": rng.standard_normal(),
        "delta_position": rng.standard_normal(3),
        "delta_orientation": rng.standard_normal(3),
        "delta_grip": rng.standard_normal(),
        "peg_pose": np.concatenate([rng.standard_normal(3), _unit_quaternion(rng)]),
        "target_hole_pose": np.concatenate([rng.standard_normal(3), _unit_quaternion(rng)]),
        "distance": rng.random(),
        "step_success": False,
    }


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a small synthetic dataset (manifest + per-episode npz) on disk."""
    root = tmp_path_factory.mktemp("dataset_tiny")
    runs = root / "runs"
    rng = np.random.default_rng(0)
    episodes = []
    for index in range(N_EPISODES):
        length = 4 + int(rng.integers(0, 6))  # 4..9 steps, varied for padding
        recorder = EpisodeRecorder()
        for step in range(length):
            recorder.add(**_random_row(rng, step))
        path = episode_npz_path(runs, index)
        # Minimal by design: the loader must cope with sparse per-episode metadata.
        recorder.save(path, metadata={"episode_index": index})  # type: ignore[typeddict-item]
        episodes.append({
            "episode_index": index,
            "file": f"runs/{path.parent.name}/{path.name}",
            "n_steps": length,
        })
    (root / "metadata.json").write_text(json.dumps({"schema_version": "2.0", "episodes": episodes}))
    return root


def _summaries(dataset_dir: Path) -> list:
    return json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))["episodes"]


def _make_episode(episode_index: int, length: int) -> Episode:
    """A tiny synthetic Episode with constant per-step values, for collate tests."""
    return Episode(
        episode_index=episode_index,
        command=torch.ones(length, STREAM_WIDTHS["command"]),
        force_torque=torch.ones(length, STREAM_WIDTHS["force_torque"]),
        proprioception=torch.ones(length, STREAM_WIDTHS["proprioception"]),
        delta=torch.ones(length, DELTA_WIDTH),
    )


# ---------------------------------------------------------------------------
# split_episodes — deterministic episode-level split
# ---------------------------------------------------------------------------


def test_split_episodes_disjoint_and_complete(tiny_dataset: Path):
    episodes = _summaries(tiny_dataset)
    train, val = split_episodes(episodes, val_fraction=0.2, seed=0)

    train_indices = {summary["episode_index"] for summary in train}
    val_indices = {summary["episode_index"] for summary in val}
    all_indices = {summary["episode_index"] for summary in episodes}

    assert train_indices.isdisjoint(val_indices)
    assert train_indices | val_indices == all_indices
    assert len(train) + len(val) == len(episodes)


def test_split_episodes_reproducible_same_seed(tiny_dataset: Path):
    episodes = _summaries(tiny_dataset)
    train_a, val_a = split_episodes(episodes, val_fraction=0.2, seed=0)
    train_b, val_b = split_episodes(episodes, val_fraction=0.2, seed=0)

    assert [s["episode_index"] for s in train_a] == [s["episode_index"] for s in train_b]
    assert [s["episode_index"] for s in val_a] == [s["episode_index"] for s in val_b]


def test_split_episodes_different_seed_differs(tiny_dataset: Path):
    episodes = _summaries(tiny_dataset)
    _, val_a = split_episodes(episodes, val_fraction=0.2, seed=0)
    _, val_b = split_episodes(episodes, val_fraction=0.2, seed=1)

    assert [s["episode_index"] for s in val_a] != [s["episode_index"] for s in val_b]


def test_split_episodes_val_fraction(tiny_dataset: Path):
    episodes = _summaries(tiny_dataset)
    _, val = split_episodes(episodes, val_fraction=0.2, seed=0)
    assert len(val) == int(len(episodes) * 0.2)


# ---------------------------------------------------------------------------
# Dataset construction — streams + shapes + dtypes
# ---------------------------------------------------------------------------


def test_dataset_construct_len_matches_train_split(tiny_dataset: Path):
    train_split, _ = split_episodes(_summaries(tiny_dataset), val_fraction=0.2, seed=0)
    dataset = OfflineResidualBCDataset(tiny_dataset, download=False, train=True)
    assert len(dataset) == len(train_split)


def test_dataset_sample_shapes_and_dtype(tiny_dataset: Path):
    dataset = OfflineResidualBCDataset(tiny_dataset, download=False, train=True)
    episode = dataset[0]

    length = episode.command.shape[0]
    for stream, width in STREAM_WIDTHS.items():
        tensor = getattr(episode, stream)
        assert tensor.shape == (length, width)
        assert tensor.dtype == torch.float32
    assert episode.delta.shape == (length, DELTA_WIDTH)
    assert episode.delta.dtype == torch.float32


# ---------------------------------------------------------------------------
# Normalization — pooled stats and shared norm_stats contract
# ---------------------------------------------------------------------------


def test_train_streams_are_normalized(tiny_dataset: Path):
    dataset = OfflineResidualBCDataset(tiny_dataset, download=False, train=True)
    for stream in INPUT_STREAMS:
        pooled = torch.cat([getattr(episode, stream) for episode in dataset.episodes], dim=0)
        assert torch.allclose(pooled.mean(dim=0), torch.zeros(pooled.shape[1]), atol=1e-3)
        assert torch.allclose(pooled.std(dim=0), torch.ones(pooled.shape[1]), atol=1e-3)


def test_val_split_shares_train_norm_stats(tiny_dataset: Path):
    train = OfflineResidualBCDataset(tiny_dataset, download=False, train=True)
    val = OfflineResidualBCDataset(
        tiny_dataset, download=False, train=False, norm_stats=train.norm_stats
    )
    assert val.norm_stats is train.norm_stats


def test_val_split_without_norm_stats_raises(tiny_dataset: Path):
    with pytest.raises(ValueError):
        OfflineResidualBCDataset(tiny_dataset, download=False, train=False, norm_stats=None)


# ---------------------------------------------------------------------------
# collate_episodes — episode-edge padding
# ---------------------------------------------------------------------------


def test_collate_episodes_shapes_and_lengths():
    episodes = [_make_episode(0, 5), _make_episode(1, 3), _make_episode(2, 8)]
    t_max = max(episode.command.shape[0] for episode in episodes)
    batch_size = len(episodes)

    batch = collate_episodes(episodes)
    assert isinstance(batch, EpisodeBatch)

    for stream, width in STREAM_WIDTHS.items():
        assert getattr(batch, stream).shape == (batch_size, t_max, width)
        assert getattr(batch, stream).dtype == torch.float32
    assert batch.delta.shape == (batch_size, t_max, DELTA_WIDTH)
    assert batch.delta.dtype == torch.float32

    assert batch.lengths.dtype == torch.long
    expected_lengths = torch.tensor([5, 3, 8], dtype=torch.long)
    assert torch.equal(batch.lengths, expected_lengths)


def test_collate_episodes_pads_tail_with_zeros():
    short = _make_episode(0, 3)
    long = _make_episode(1, 8)
    batch = collate_episodes([short, long])

    # The shorter episode (row 0, true length 3) has a zero-padded tail in every
    # stream past its true length — the episode-edge padding check.
    short_length = short.command.shape[0]
    for stream in (*STREAM_WIDTHS, "delta"):
        padded = getattr(batch, stream)[0]
        tail = padded[short_length:]
        assert torch.count_nonzero(tail) == 0


# ---------------------------------------------------------------------------
# Wrist-image loading (LAB-80) — load_frame_stream + load_images=True
# ---------------------------------------------------------------------------

RENDER_EVERY = 3
FRAME_SIZE = 8  # small synthetic frames; real frames are 224x224 (render_wrist_camera)


def _write_frame(imgs_dir: Path, step: int, rng: np.random.Generator) -> None:
    from PIL import Image

    imgs_dir.mkdir(parents=True, exist_ok=True)
    pixels = rng.integers(0, 256, size=(FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(imgs_dir / f"step_{step:05d}.jpg", quality=90)


@pytest.fixture(scope="session")
def tiny_dataset_with_images(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Like ``tiny_dataset``, but every episode also has rendered frames every
    ``RENDER_EVERY`` steps under ``imgs/`` — for exercising ``load_images=True``."""
    root = tmp_path_factory.mktemp("dataset_tiny_images")
    runs = root / "runs"
    rng = np.random.default_rng(1)
    episodes = []
    for index in range(N_EPISODES):
        length = 4 + int(rng.integers(0, 6))
        recorder = EpisodeRecorder()
        for step in range(length):
            recorder.add(**_random_row(rng, step))
        path = episode_npz_path(runs, index)
        # Minimal by design (see the F/T fixture above).
        recorder.save(path, metadata={"episode_index": index})  # type: ignore[typeddict-item]
        imgs_dir = episode_imgs_dir(runs, index)
        for step in range(0, length, RENDER_EVERY):
            _write_frame(imgs_dir, step, rng)
        episodes.append({
            "episode_index": index,
            "file": f"runs/{path.parent.name}/{path.name}",
            "n_steps": length,
        })
    (root / "metadata.json").write_text(json.dumps({"schema_version": "2.0", "episodes": episodes}))
    return root


def test_load_frame_stream_shapes_and_forward_fill(tiny_dataset_with_images: Path):
    summary = _summaries(tiny_dataset_with_images)[0]
    n_steps = summary["n_steps"]
    imgs_dir = episode_imgs_dir(tiny_dataset_with_images / "runs", summary["episode_index"])

    images, frame_index = load_frame_stream(imgs_dir, n_steps=n_steps)

    n_frames = len(range(0, n_steps, RENDER_EVERY))
    assert images.shape == (n_frames, 3, FRAME_SIZE, FRAME_SIZE)
    assert images.dtype == torch.float32
    assert frame_index.shape == (n_steps,)
    assert frame_index.dtype == torch.long

    # Forward-fill: every step maps to the most recent rendered frame at/before it.
    for step in range(n_steps):
        assert frame_index[step].item() == step // RENDER_EVERY


def test_load_frame_stream_no_frames_raises(tmp_path: Path):
    empty_imgs_dir = tmp_path / "imgs"
    empty_imgs_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_frame_stream(empty_imgs_dir, n_steps=5)


def test_dataset_load_images_true_populates_episode_images(tiny_dataset_with_images: Path):
    dataset = OfflineResidualBCDataset(
        tiny_dataset_with_images, download=False, train=True, load_images=True
    )
    episode = dataset[0]
    length = episode.command.shape[0]

    assert episode.images is not None
    assert episode.image_frame_index is not None
    assert episode.images.shape[1:] == (3, FRAME_SIZE, FRAME_SIZE)
    assert episode.image_frame_index.shape == (length,)


def test_dataset_load_images_true_keeps_frames_lazy(tiny_dataset_with_images: Path):
    """LAB-103: decoded frames must not be resident in ``self.episodes`` — they decode
    per ``__getitem__``. This is the invariant that keeps RAM batch-scaled, not corpus-scaled."""
    dataset = OfflineResidualBCDataset(
        tiny_dataset_with_images, download=False, train=True, load_images=True
    )
    # Nothing decoded up front — only frame paths + the cheap per-step index are held.
    assert all(episode.images is None for episode in dataset.episodes)
    assert all(episode.image_frame_index is not None for episode in dataset.episodes)
    # __getitem__ decodes on demand.
    assert dataset[0].images is not None


def test_dataset_load_images_false_leaves_images_none(tiny_dataset_with_images: Path):
    dataset = OfflineResidualBCDataset(
        tiny_dataset_with_images, download=False, train=True, load_images=False
    )
    assert all(episode.images is None for episode in dataset.episodes)


def test_collate_episodes_pads_images_for_mixed_length_batch():
    short = Episode(
        episode_index=0,
        command=torch.ones(3, STREAM_WIDTHS["command"]),
        force_torque=torch.ones(3, STREAM_WIDTHS["force_torque"]),
        proprioception=torch.ones(3, STREAM_WIDTHS["proprioception"]),
        delta=torch.ones(3, DELTA_WIDTH),
        images=torch.ones(1, 3, FRAME_SIZE, FRAME_SIZE),
        image_frame_index=torch.zeros(3, dtype=torch.long),
    )
    long = Episode(
        episode_index=1,
        command=torch.ones(8, STREAM_WIDTHS["command"]),
        force_torque=torch.ones(8, STREAM_WIDTHS["force_torque"]),
        proprioception=torch.ones(8, STREAM_WIDTHS["proprioception"]),
        delta=torch.ones(8, DELTA_WIDTH),
        images=torch.ones(3, 3, FRAME_SIZE, FRAME_SIZE),
        image_frame_index=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.long),
    )
    batch = collate_episodes([short, long])

    assert batch.images is not None
    assert batch.image_frame_index is not None
    assert batch.images.shape == (2, 3, 3, FRAME_SIZE, FRAME_SIZE)
    assert batch.image_frame_index.shape == (2, 8)
    # Short episode's true frame_index (length 3) is preserved, untouched by padding.
    assert short.image_frame_index is not None
    assert torch.equal(batch.image_frame_index[0, :3], short.image_frame_index)


def test_collate_episodes_images_none_when_not_all_episodes_have_them():
    with_images = _make_episode(0, 5)
    with_images.images = torch.ones(2, 3, FRAME_SIZE, FRAME_SIZE)
    with_images.image_frame_index = torch.zeros(5, dtype=torch.long)
    without_images = _make_episode(1, 5)

    batch = collate_episodes([with_images, without_images])
    assert batch.images is None
    assert batch.image_frame_index is None


# ---------------------------------------------------------------------------
# build_dataloaders — end-to-end factory
# ---------------------------------------------------------------------------


def test_build_dataloaders_returns_loaders_and_stats(tiny_dataset: Path):
    train_loader, val_loader, norm_stats = build_dataloaders(
        tiny_dataset, batch_size=4, download=False, num_workers=0
    )

    assert isinstance(norm_stats, NormStats)
    assert isinstance(train_loader.dataset, OfflineResidualBCDataset)
    assert isinstance(val_loader.dataset, OfflineResidualBCDataset)
    # Both splits share the train normalization.
    assert val_loader.dataset.norm_stats is norm_stats


def test_build_dataloaders_train_batch_shapes(tiny_dataset: Path):
    batch_size = 4
    train_loader, _, _ = build_dataloaders(
        tiny_dataset, batch_size=batch_size, download=False, num_workers=0
    )

    batch = next(iter(train_loader))
    assert isinstance(batch, EpisodeBatch)

    actual_batch_size = batch.lengths.shape[0]
    assert actual_batch_size <= batch_size
    t_max = batch.command.shape[1]
    for stream, width in STREAM_WIDTHS.items():
        assert getattr(batch, stream).shape == (actual_batch_size, t_max, width)
    assert batch.delta.shape == (actual_batch_size, t_max, DELTA_WIDTH)
    assert batch.lengths.shape == (actual_batch_size,)
