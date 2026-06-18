"""Tests for the Phase-1 residual policy model (LAB-33).

Self-contained and CPU-only: the model is exercised on synthetic batches built
through the *real* loader collate (``collate_episodes``), so the test proves the
network consumes exactly what the M5 dataset emits (padded ``EpisodeBatch``
streams + ``lengths``) without needing a corpus on disk.

Covers the acceptance criteria: the sequence forward returns a per-step
``(B, T, 7)`` Δ; the single-step ``step`` path returns ``(B, 7)``; both are
float32 and NaN-free; the recurrent hidden state has the documented shape; and
the model is deterministic in eval mode.
"""

from __future__ import annotations

import torch

from ai_teleop.data.dataset import Episode, collate_episodes
from ai_teleop.policy.config import PolicyConfig
from ai_teleop.policy.model import ResidualPolicy

# Per-stream channel widths — the (T, C) feature contract from the Episode dataclass.
STREAM_WIDTHS: dict[str, int] = {"command": 9, "force_torque": 6, "proprioception": 24}
DELTA_WIDTH = 7


def _make_episode(episode_index: int, length: int, *, seed: int = 0) -> Episode:
    """A synthetic Episode of the given length with random per-step values."""
    generator = torch.Generator().manual_seed(seed + episode_index)
    return Episode(
        episode_index=episode_index,
        command=torch.randn(length, STREAM_WIDTHS["command"], generator=generator),
        force_torque=torch.randn(length, STREAM_WIDTHS["force_torque"], generator=generator),
        proprioception=torch.randn(length, STREAM_WIDTHS["proprioception"], generator=generator),
        delta=torch.randn(length, DELTA_WIDTH, generator=generator),
    )


def _model(**overrides: object) -> ResidualPolicy:
    """A small, deterministically-initialized policy for fast CPU tests."""
    torch.manual_seed(0)
    config = PolicyConfig(hidden_size=32, num_layers=2, head_hidden=(32,), **overrides)  # type: ignore[arg-type]
    return ResidualPolicy(config).eval()


# ---------------------------------------------------------------------------
# Sequence forward — the training path
# ---------------------------------------------------------------------------


def test_forward_returns_per_step_delta_sequence():
    episodes = [_make_episode(0, 5), _make_episode(1, 3), _make_episode(2, 8)]
    batch = collate_episodes(episodes)
    batch_size = len(episodes)
    t_max = max(episode.command.shape[0] for episode in episodes)

    model = _model()
    delta, hidden = model.forward(
        batch.command, batch.force_torque, batch.proprioception, lengths=batch.lengths
    )

    assert delta.shape == (batch_size, t_max, DELTA_WIDTH)
    assert delta.dtype == torch.float32
    assert not torch.isnan(delta).any()
    # GRU final hidden: (num_layers, B, hidden_size).
    assert hidden.shape == (model.config.num_layers, batch_size, model.config.hidden_size)


def test_forward_without_lengths_runs_on_padded_tensor():
    """``lengths`` is optional — the forward must still work on the raw padded batch."""
    episodes = [_make_episode(0, 6), _make_episode(1, 4)]
    batch = collate_episodes(episodes)

    model = _model()
    delta, _ = model.forward(batch.command, batch.force_torque, batch.proprioception)

    assert delta.shape == (2, batch.command.shape[1], DELTA_WIDTH)
    assert not torch.isnan(delta).any()


# ---------------------------------------------------------------------------
# Single-step path — the O(1) deployment / latency path
# ---------------------------------------------------------------------------


def test_step_returns_single_delta_and_advances_hidden():
    batch_size = 4
    model = _model()

    command = torch.randn(batch_size, STREAM_WIDTHS["command"])
    force_torque = torch.randn(batch_size, STREAM_WIDTHS["force_torque"])
    proprioception = torch.randn(batch_size, STREAM_WIDTHS["proprioception"])

    delta, hidden = model.step(command, force_torque, proprioception)

    assert delta.shape == (batch_size, DELTA_WIDTH)
    assert delta.dtype == torch.float32
    assert not torch.isnan(delta).any()
    assert hidden.shape == (model.config.num_layers, batch_size, model.config.hidden_size)

    # The hidden state actually carries: a second step from the returned hidden
    # differs from a fresh cold-start step on the same input.
    next_cold, _ = model.step(command, force_torque, proprioception)
    next_warm, _ = model.step(command, force_torque, proprioception, hidden=hidden)
    assert not torch.allclose(next_cold, next_warm)


def test_step_matches_first_timestep_of_forward():
    """A single ``step`` from a cold start equals the first step of the sequence
    forward on the same inputs — the two paths share one core + head."""
    model = _model()
    episode = _make_episode(0, 1)
    batch = collate_episodes([episode])

    seq_delta, _ = model.forward(batch.command, batch.force_torque, batch.proprioception)
    step_delta, _ = model.step(
        episode.command[:1], episode.force_torque[:1], episode.proprioception[:1]
    )

    assert torch.allclose(seq_delta[:, 0, :], step_delta, atol=1e-5)


# ---------------------------------------------------------------------------
# Determinism + shape derivation
# ---------------------------------------------------------------------------


def test_eval_forward_is_deterministic():
    episodes = [_make_episode(0, 5), _make_episode(1, 7)]
    batch = collate_episodes(episodes)

    model = _model()
    with torch.no_grad():
        first, _ = model.forward(batch.command, batch.force_torque, batch.proprioception)
        second, _ = model.forward(batch.command, batch.force_torque, batch.proprioception)

    assert torch.equal(first, second)


def test_input_dim_is_derived_not_hardcoded():
    config = PolicyConfig()
    assert (
        config.input_dim == config.command_dim + config.force_torque_dim + config.proprioception_dim
    )
    assert config.input_dim == 39
