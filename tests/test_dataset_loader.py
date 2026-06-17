"""Tests for the M5 offline BC dataset loader (LAB-32).

Exercises the loader-facing contracts: the deterministic episode-level split,
the assembled per-episode streams + their normalization, episode-edge padding in
``collate_episodes``, and the end-to-end ``build_dataloaders`` factory.

These run against the small committed corpus at ``data/dataset_0`` (20 episodes).
We always pass ``download=False`` so nothing regenerates, and everything stays on
CPU. ``collate_episodes`` / ``build_dataloaders`` / ``EpisodeBatch`` are written
to the teammate's not-yet-landed contract, so those tests are expected to be red
until that code lands.
"""

from __future__ import annotations

from pathlib import Path

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

DATASET_DIR = Path(__file__).resolve().parents[1] / "data" / "dataset_0"

requires_corpus = pytest.mark.skipif(
    not DATASET_DIR.exists(), reason="data/dataset_0 corpus not found"
)

# Per-stream channel widths, the (T, C) feature contract from the Episode dataclass.
STREAM_WIDTHS: dict[str, int] = {"command": 9, "force_torque": 6, "proprioception": 24}
DELTA_WIDTH = 7


def _summaries() -> list:
    """All episode summaries from the corpus manifest, via a constructed dataset's
    metadata (avoids re-reading metadata.json by hand)."""
    dataset = OfflineResidualBCDataset(DATASET_DIR, download=False, train=True)
    return dataset.metadata["episodes"]


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


@requires_corpus
def test_split_episodes_disjoint_and_complete():
    episodes = _summaries()
    train, val = split_episodes(episodes, val_fraction=0.2, seed=0)

    train_indices = {summary["episode_index"] for summary in train}
    val_indices = {summary["episode_index"] for summary in val}
    all_indices = {summary["episode_index"] for summary in episodes}

    assert train_indices.isdisjoint(val_indices)
    assert train_indices | val_indices == all_indices
    assert len(train) + len(val) == len(episodes)


@requires_corpus
def test_split_episodes_reproducible_same_seed():
    episodes = _summaries()
    train_a, val_a = split_episodes(episodes, val_fraction=0.2, seed=0)
    train_b, val_b = split_episodes(episodes, val_fraction=0.2, seed=0)

    assert [s["episode_index"] for s in train_a] == [s["episode_index"] for s in train_b]
    assert [s["episode_index"] for s in val_a] == [s["episode_index"] for s in val_b]


@requires_corpus
def test_split_episodes_different_seed_differs():
    episodes = _summaries()
    _, val_a = split_episodes(episodes, val_fraction=0.2, seed=0)
    _, val_b = split_episodes(episodes, val_fraction=0.2, seed=1)

    assert [s["episode_index"] for s in val_a] != [s["episode_index"] for s in val_b]


@requires_corpus
def test_split_episodes_val_fraction():
    episodes = _summaries()
    _, val = split_episodes(episodes, val_fraction=0.2, seed=0)
    assert len(val) == int(len(episodes) * 0.2)


# ---------------------------------------------------------------------------
# Dataset construction — streams + shapes + dtypes
# ---------------------------------------------------------------------------


@requires_corpus
def test_dataset_construct_len_matches_train_split():
    train_split, _ = split_episodes(_summaries(), val_fraction=0.2, seed=0)
    dataset = OfflineResidualBCDataset(DATASET_DIR, download=False, train=True)
    assert len(dataset) == len(train_split)


@requires_corpus
def test_dataset_sample_shapes_and_dtype():
    dataset = OfflineResidualBCDataset(DATASET_DIR, download=False, train=True)
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


@requires_corpus
def test_train_streams_are_normalized():
    dataset = OfflineResidualBCDataset(DATASET_DIR, download=False, train=True)
    for stream in INPUT_STREAMS:
        pooled = torch.cat([getattr(episode, stream) for episode in dataset.episodes], dim=0)
        assert torch.allclose(pooled.mean(dim=0), torch.zeros(pooled.shape[1]), atol=1e-3)
        assert torch.allclose(pooled.std(dim=0), torch.ones(pooled.shape[1]), atol=1e-3)


@requires_corpus
def test_val_split_shares_train_norm_stats():
    train = OfflineResidualBCDataset(DATASET_DIR, download=False, train=True)
    val = OfflineResidualBCDataset(
        DATASET_DIR, download=False, train=False, norm_stats=train.norm_stats
    )
    assert val.norm_stats is train.norm_stats


@requires_corpus
def test_val_split_without_norm_stats_raises():
    with pytest.raises(ValueError):
        OfflineResidualBCDataset(DATASET_DIR, download=False, train=False, norm_stats=None)


# ---------------------------------------------------------------------------
# collate_episodes — episode-edge padding (NOT-YET-IMPLEMENTED contract)
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
    # stream past its true length — this is the episode-edge padding check.
    short_length = short.command.shape[0]
    for stream in (*STREAM_WIDTHS, "delta"):
        padded = getattr(batch, stream)[0]
        tail = padded[short_length:]
        assert torch.count_nonzero(tail) == 0


# ---------------------------------------------------------------------------
# build_dataloaders — end-to-end factory (NOT-YET-IMPLEMENTED contract)
# ---------------------------------------------------------------------------


@requires_corpus
def test_build_dataloaders_returns_loaders_and_stats():
    train_loader, val_loader, norm_stats = build_dataloaders(
        DATASET_DIR, batch_size=4, download=False, num_workers=0
    )

    assert isinstance(norm_stats, NormStats)
    assert isinstance(train_loader.dataset, OfflineResidualBCDataset)
    assert isinstance(val_loader.dataset, OfflineResidualBCDataset)
    # Both splits share the train normalization.
    assert val_loader.dataset.norm_stats is norm_stats


@requires_corpus
def test_build_dataloaders_train_batch_shapes():
    batch_size = 4
    train_loader, _, _ = build_dataloaders(
        DATASET_DIR, batch_size=batch_size, download=False, num_workers=0
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
