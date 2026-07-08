"""Tests for the Phase-2 vision deploy path (LAB-83).

The Step-0 wiring that lets the trained vision checkpoint run closed-loop:

* ``SimEnv.enable_wrist_capture`` — the env is the frame-rate limiter (renders every
  ``render_every`` ticks, holds the frame between, clears on reset).
* ``LearnedResidual`` vision branch — reads ``Observation.wrist_image`` and feeds it
  through ``model.step``; raises a clear error when a vision policy gets no frame.
* ``normalize_frame`` — the single ImageNet-normalization shared by the training
  loader and the live path (the covariate-shift guard).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ai_teleop.common.command import Command
from ai_teleop.common.observation import Observation
from ai_teleop.data.dataset import INPUT_STREAMS, NormStats
from ai_teleop.data.images import normalize_frame
from ai_teleop.domain import Delta
from ai_teleop.policy import LearnedResidual, PolicyConfig, ResidualPolicy

_STREAM_DIMS = {"command": 9, "force_torque": 6, "proprioception": 24}
_SCENE_PATH = Path(__file__).resolve().parents[1] / "assets" / "mjcf" / "full_scene.xml"


def _identity_stats() -> NormStats:
    return NormStats(
        mean={stream: torch.zeros(_STREAM_DIMS[stream]) for stream in INPUT_STREAMS},
        std={stream: torch.ones(_STREAM_DIMS[stream]) for stream in INPUT_STREAMS},
    )


def _vision_provider() -> LearnedResidual:
    """A small vision policy (random backbone, no ImageNet download) for wiring tests."""
    torch.manual_seed(0)
    config = PolicyConfig(use_vision=True, image_pretrained=False, hidden_size=16, num_layers=1)
    return LearnedResidual(ResidualPolicy(config).eval(), _identity_stats())


def _frame(value: int = 128) -> np.ndarray:
    return np.full((224, 224, 3), value, dtype=np.uint8)


def _observation(*, wrist_image: np.ndarray | None, sim_time: float = 0.0) -> Observation:
    return Observation(
        joint_positions=np.linspace(-0.3, 0.3, 7),
        joint_velocities=np.linspace(0.0, 0.1, 7),
        ee_pose=np.array([0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
        wrist_ft=np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
        gripper_width=0.05,
        peg_pose=np.array([0.5, 0.0, 0.45, 1.0, 0.0, 0.0, 0.0]),
        hole_poses=np.array([[0.6, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]]),
        sim_time=sim_time,
        wrist_image=wrist_image,
    )


def _command() -> Command:
    return Command(np.array([0.55, 0.05, 0.48]), np.array([1.0, 0.0, 0.0, 0.0]), 0.0)


# ---------------------------------------------------------------------------
# normalize_frame — the shared covariate-shift-safe normalization
# ---------------------------------------------------------------------------


def test_normalize_frame_shape_and_imagenet_stats():
    tensor = normalize_frame(_frame(255))  # all-white frame
    assert tensor.shape == (3, 224, 224)
    # White (1.0 after /255) normalized by ImageNet per-channel mean/std.
    expected = (1.0 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    assert torch.allclose(tensor[:, 0, 0], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# LearnedResidual vision branch
# ---------------------------------------------------------------------------


def test_vision_provider_reports_use_vision():
    assert _vision_provider().use_vision is True


def test_vision_provider_consumes_wrist_image():
    """A vision policy produces a clamped Δ when the observation carries a frame."""
    provider = _vision_provider()
    delta = provider.get_delta(_observation(wrist_image=_frame()), _command())
    assert isinstance(delta, Delta)
    assert np.linalg.norm(delta.delta_position) <= 0.03 + 1e-9


def test_vision_provider_passes_image_into_model_step():
    """The wrist frame actually reaches ``model.step`` (not silently dropped)."""
    provider = _vision_provider()
    seen: dict[str, torch.Tensor | None] = {}
    original_step = provider._model.step

    def _spy(command, force_torque, proprioception, image=None, hidden=None):  # noqa: ANN001
        seen["image"] = image
        return original_step(command, force_torque, proprioception, image=image, hidden=hidden)

    provider._model.step = _spy  # type: ignore[method-assign]
    provider.get_delta(_observation(wrist_image=_frame()), _command())
    assert seen["image"] is not None
    assert seen["image"].shape == (1, 3, 224, 224)


def test_vision_provider_without_frame_raises():
    """A vision policy given no frame fails loudly (env capture not enabled)."""
    provider = _vision_provider()
    with pytest.raises(ValueError, match="wrist_image"):
        provider.get_delta(_observation(wrist_image=None), _command())


# ---------------------------------------------------------------------------
# SimEnv wrist capture — the env as frame-rate limiter
# ---------------------------------------------------------------------------


def _env_with_stubbed_render():
    """A real SimEnv with ``render_wrist_camera`` stubbed to a distinct-frame counter."""
    from ai_teleop.sim.scene import SimEnv

    env = SimEnv(str(_SCENE_PATH), render_mode="headless")
    calls = {"n": 0}

    def _stub() -> np.ndarray:
        calls["n"] += 1
        return _frame(calls["n"])  # a distinct frame each render

    env.render_wrist_camera = _stub  # type: ignore[method-assign]
    return env, calls


def test_wrist_capture_disabled_by_default():
    if not _SCENE_PATH.exists():
        pytest.skip(f"scene file not found: {_SCENE_PATH}")
    env, calls = _env_with_stubbed_render()
    try:
        observation = env.reset()
        assert observation.wrist_image is None
        assert calls["n"] == 0  # nothing rendered when capture is off
    finally:
        env.close()


def test_wrist_capture_rate_limits_and_holds_between_renders():
    if not _SCENE_PATH.exists():
        pytest.skip(f"scene file not found: {_SCENE_PATH}")
    env, calls = _env_with_stubbed_render()
    try:
        env.enable_wrist_capture(render_every=3)
        first = env.reset()  # tick 0 → renders
        assert first.wrist_image is not None
        assert calls["n"] == 1

        held = [env.get_observation().wrist_image for _ in range(2)]  # ticks 1,2 → hold
        assert calls["n"] == 1
        # Held frame is the same object the first render produced.
        assert all(frame is first.wrist_image for frame in held)

        rerendered = env.get_observation().wrist_image  # tick 3 → renders again
        assert calls["n"] == 2
        assert rerendered is not first.wrist_image
    finally:
        env.close()


def test_wrist_capture_cache_clears_on_reset():
    if not _SCENE_PATH.exists():
        pytest.skip(f"scene file not found: {_SCENE_PATH}")
    env, calls = _env_with_stubbed_render()
    try:
        env.enable_wrist_capture(render_every=100)
        env.reset()
        assert calls["n"] == 1
        env.get_observation()  # within the interval → held, no new render
        assert calls["n"] == 1
        env.reset()  # new episode → cache cleared → first obs renders fresh
        assert calls["n"] == 2
    finally:
        env.close()


def test_enable_wrist_capture_rejects_bad_cadence():
    if not _SCENE_PATH.exists():
        pytest.skip(f"scene file not found: {_SCENE_PATH}")
    env, _ = _env_with_stubbed_render()
    try:
        with pytest.raises(ValueError, match="render_every"):
            env.enable_wrist_capture(render_every=0)
    finally:
        env.close()
