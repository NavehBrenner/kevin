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


# Small image geometry for fast CPU tests. The backbone has adaptive pooling, so a
# tiny frame is valid input — the loader's real frames are 224×224 (data/images.py).
_IMG_HW = 32
_N_FRAMES = 3


def _make_episode(
    episode_index: int, length: int, *, seed: int = 0, with_images: bool = False
) -> Episode:
    """A synthetic Episode of the given length with random per-step values.

    With ``with_images``, also attaches a compact ``(F, 3, H, W)`` frame stack and a
    monotonic per-step ``(T,)`` frame index — the shape the loader emits for the
    Phase-2 vision path.
    """
    generator = torch.Generator().manual_seed(seed + episode_index)
    images = image_frame_index = None
    if with_images:
        images = torch.randn(_N_FRAMES, 3, _IMG_HW, _IMG_HW, generator=generator)
        # Forward-fill index: spread T steps across the F frames, monotonic in [0, F).
        image_frame_index = (torch.arange(length) * _N_FRAMES // max(length, 1)).clamp(
            max=_N_FRAMES - 1
        )
    return Episode(
        episode_index=episode_index,
        command=torch.randn(length, STREAM_WIDTHS["command"], generator=generator),
        force_torque=torch.randn(length, STREAM_WIDTHS["force_torque"], generator=generator),
        proprioception=torch.randn(length, STREAM_WIDTHS["proprioception"], generator=generator),
        delta=torch.randn(length, DELTA_WIDTH, generator=generator),
        images=images,
        image_frame_index=image_frame_index,
    )


def _model(**overrides: object) -> ResidualPolicy:
    """A small, deterministically-initialized policy for fast CPU tests."""
    torch.manual_seed(0)
    config = PolicyConfig(hidden_size=32, num_layers=2, head_hidden=(32,), **overrides)  # type: ignore[arg-type]
    return ResidualPolicy(config).eval()


def _vision_model(**overrides: object) -> ResidualPolicy:
    """A small vision-conditioned policy; from-scratch backbone so tests stay offline."""
    torch.manual_seed(0)
    config = PolicyConfig(
        hidden_size=32,
        num_layers=2,
        head_hidden=(32,),
        use_vision=True,
        image_pretrained=False,  # no weight download in CI
        image_embed_dim=16,
        **overrides,  # type: ignore[arg-type]
    )
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


# ---------------------------------------------------------------------------
# Phase-2 vision stream — the image-conditioned path (LAB-81)
# ---------------------------------------------------------------------------


def test_gru_input_widens_with_vision_only():
    """Vision leaves the vector width (input_dim) alone but widens the GRU input."""
    ft_only = PolicyConfig()
    vision = PolicyConfig(use_vision=True, image_embed_dim=16)
    assert ft_only.input_dim == vision.input_dim == 39  # vector streams unchanged
    assert ft_only.gru_input_dim == 39
    assert vision.gru_input_dim == 39 + 16


def test_vision_forward_returns_per_step_delta_sequence():
    episodes = [
        _make_episode(0, 5, with_images=True),
        _make_episode(1, 3, with_images=True),
        _make_episode(2, 8, with_images=True),
    ]
    batch = collate_episodes(episodes)
    t_max = max(episode.command.shape[0] for episode in episodes)

    model = _vision_model()
    delta, hidden = model.forward(
        batch.command,
        batch.force_torque,
        batch.proprioception,
        images=batch.images,
        image_frame_index=batch.image_frame_index,
        lengths=batch.lengths,
    )

    assert delta.shape == (len(episodes), t_max, DELTA_WIDTH)
    assert not torch.isnan(delta).any()
    assert hidden.shape == (model.config.num_layers, len(episodes), model.config.hidden_size)


def test_vision_forward_requires_images():
    """A vision model must be given frames — silently running F/T-only would be a
    train/deploy mismatch, so the omission is a hard error."""
    batch = collate_episodes([_make_episode(0, 4, with_images=True)])
    model = _vision_model()
    try:
        model.forward(batch.command, batch.force_torque, batch.proprioception)
    except ValueError:
        return
    raise AssertionError("forward without images should raise ValueError under use_vision")


def test_vision_step_returns_single_delta():
    batch_size = 2
    model = _vision_model()

    command = torch.randn(batch_size, STREAM_WIDTHS["command"])
    force_torque = torch.randn(batch_size, STREAM_WIDTHS["force_torque"])
    proprioception = torch.randn(batch_size, STREAM_WIDTHS["proprioception"])
    image = torch.randn(batch_size, 3, _IMG_HW, _IMG_HW)

    delta, hidden = model.step(command, force_torque, proprioception, image=image)

    assert delta.shape == (batch_size, DELTA_WIDTH)
    assert not torch.isnan(delta).any()
    assert hidden.shape == (model.config.num_layers, batch_size, model.config.hidden_size)


def test_vision_changes_the_output():
    """The image stream must actually reach the head: two batches identical except
    for their frames produce different corrections.

    Asserted in **train** mode — the regime training actually runs in. An
    untrained-from-scratch backbone in eval mode collapses to a near-constant
    embedding on tiny synthetic frames (default BatchNorm running stats +
    Hardswish/squeeze-excite saturation), which is an artifact of this offline
    test setup, not of the real pretrained-224×224 path.
    """
    episode = _make_episode(0, 6, with_images=True)
    batch = collate_episodes([episode])
    model = _vision_model().train()

    with torch.no_grad():
        first, _ = model.forward(
            batch.command,
            batch.force_torque,
            batch.proprioception,
            images=batch.images,
            image_frame_index=batch.image_frame_index,
        )
        second, _ = model.forward(
            batch.command,
            batch.force_torque,
            batch.proprioception,
            images=torch.randn_like(batch.images),  # type: ignore[arg-type]
            image_frame_index=batch.image_frame_index,
        )

    assert not torch.allclose(first, second)


# ---------------------------------------------------------------------------
# Stage-C memory levers (LAB-105) — must not change results, must still learn
# ---------------------------------------------------------------------------


def test_encode_frames_chunking_matches_unchunked():
    """Frame chunking is a pure memory optimization: encoding the F frames in
    sub-batches must give the same embeddings as one forward (LAB-105 Stage C)."""
    model = _vision_model()  # eval mode ⇒ deterministic, checkpoint branch inactive
    encoder = model.image_encoder
    assert encoder is not None
    frames = torch.randn(2, 5, 3, _IMG_HW, _IMG_HW, generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        encoder.encode_chunk_size = 0  # whole B·F=10 batch
        whole = encoder.encode_frames(frames)
        encoder.encode_chunk_size = 2  # forces >1 chunk
        chunked = encoder.encode_frames(frames)

    assert torch.allclose(whole, chunked, atol=1e-6)


def test_gradient_checkpointing_reaches_the_unfrozen_backbone():
    """Stage C unfreezes the backbone; with checkpointing + chunking on, backward
    must still populate backbone gradients — otherwise the encoder can't learn to
    localize the hole (the whole point of Stage C)."""
    model = _vision_model().train()
    encoder = model.image_encoder
    assert encoder is not None
    encoder.use_gradient_checkpoint = True
    encoder.encode_chunk_size = 2
    # B·F=8 with chunk 2 ⇒ even [2,2,2,2] chunks: batch ≥ 2 keeps train-mode BatchNorm
    # valid even though these tiny 32² frames pool to 1×1 (real 224² frames stay 7×7).
    frames = torch.randn(2, 4, 3, _IMG_HW, _IMG_HW, generator=torch.Generator().manual_seed(2))

    encoder.encode_frames(frames).sum().backward()

    backbone_grads = [p.grad for p in encoder.backbone.parameters() if p.requires_grad]
    assert backbone_grads, "backbone should be trainable (unfrozen) under Stage C"
    assert any(g is not None and float(g.abs().sum()) > 0 for g in backbone_grads), (
        "checkpointed backward must reach the backbone weights"
    )
