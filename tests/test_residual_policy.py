"""Tests for ``LearnedResidual`` — the trained residual as an AssistProvider.

Covers the M5 seam-integration acceptance: protocol conformance, the exact
input-assembly contract with the M4 loader (the silent-covariate-shift guard),
the F/T-bias capture and per-episode auto-reset, checkpoint round-tripping, the
per-step latency budget, and a real ``run_episode`` swap-in against the live sim
(``NoAssist`` → ``LearnedResidual`` with no other change).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch

from ai_teleop.common.command import Command
from ai_teleop.common.observation import Observation
from ai_teleop.data.dataset import INPUT_STREAMS, NormStats, extract_training_episode
from ai_teleop.domain import AssistProvider, Delta, apply_delta
from ai_teleop.policy import LearnedResidual, PolicyConfig, ResidualPolicy, save_checkpoint
from ai_teleop.policy.losses import LossConfig
from ai_teleop.policy.residual_policy import load_checkpoint

_STREAM_DIMS = {"command": 9, "force_torque": 6, "proprioception": 24}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _identity_stats() -> NormStats:
    """Zero-mean / unit-std stats — a valid no-op normalization for assembly tests."""
    return NormStats(
        mean={stream: torch.zeros(_STREAM_DIMS[stream]) for stream in INPUT_STREAMS},
        std={stream: torch.ones(_STREAM_DIMS[stream]) for stream in INPUT_STREAMS},
    )


def _model(*, seed: int = 0, hidden_size: int = 16, num_layers: int = 1) -> ResidualPolicy:
    torch.manual_seed(seed)
    return ResidualPolicy(PolicyConfig(hidden_size=hidden_size, num_layers=num_layers)).eval()


def _provider(*, hidden_size: int = 16, num_layers: int = 1) -> LearnedResidual:
    return LearnedResidual(
        _model(hidden_size=hidden_size, num_layers=num_layers), _identity_stats()
    )


def _observation(*, sim_time: float = 0.0, wrist_ft: np.ndarray | None = None) -> Observation:
    return Observation(
        joint_positions=np.linspace(-0.3, 0.3, 7),
        joint_velocities=np.linspace(0.0, 0.1, 7),
        ee_pose=np.array([0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
        wrist_ft=np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]) if wrist_ft is None else wrist_ft,
        gripper_width=0.05,
        peg_pose=np.array([0.5, 0.0, 0.45, 1.0, 0.0, 0.0, 0.0]),
        hole_poses=np.array([[0.6, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]]),
        sim_time=sim_time,
    )


def _command() -> Command:
    quaternion = np.array([0.92388, 0.0, 0.0, 0.38268])  # 45° about z, unit norm
    return Command(np.array([0.55, 0.05, 0.48]), quaternion, 0.0)


# ---------------------------------------------------------------------------
# Conformance + output contract
# ---------------------------------------------------------------------------


def test_learned_residual_satisfies_assist_provider_protocol():
    assert isinstance(_provider(), AssistProvider)


def test_get_delta_returns_clamped_delta():
    delta = _provider().get_delta(_observation(), _command())
    assert isinstance(delta, Delta)
    assert delta.delta_position.shape == (3,)
    assert delta.delta_orientation.shape == (3,)
    # Within the seam's per-step Δ bounds (the wrapper clamps on the way out).
    assert np.linalg.norm(delta.delta_position) <= 0.03 + 1e-9
    assert np.linalg.norm(delta.delta_orientation) <= np.deg2rad(10.0) + 1e-9
    assert abs(delta.delta_grip_force) <= 5.0 + 1e-9


# ---------------------------------------------------------------------------
# Input assembly matches the loader exactly (covariate-shift guard)
# ---------------------------------------------------------------------------


def test_inference_assembly_matches_loader():
    """The wrapper's per-step streams must equal ``extract_training_episode``'s.

    Build a one-step episode's columns, run them through the real loader path, and
    compare to the wrapper assembling the same raw values — with ``ft_bias = 0`` so
    the bias-subtracted F/T equals the logged column. Any drift in column order or
    the quat→6D map (which would silently shift the policy's inputs) fails here.
    """
    command = _command()
    observation = _observation()

    columns = {
        "cmd_position": command.target_position[None, :],
        "cmd_quaternion": command.target_quaternion[None, :],
        "wrist_ft": observation.wrist_ft[None, :],
        "ee_pose": observation.ee_pose[None, :],
        "joint_positions": observation.joint_positions[None, :],
        "joint_velocities": observation.joint_velocities[None, :],
        "gripper_width": np.array([observation.gripper_width]),
        "delta_position": np.zeros((1, 3)),
        "delta_orientation": np.zeros((1, 3)),
        "delta_grip": np.zeros(1),
    }
    # Only the columns matter here; the metadata is a stub.
    loader_episode = extract_training_episode(
        (columns, {"episode_index": 0})  # type: ignore[arg-type,typeddict-item]
    )

    provider = _provider()
    provider._ft_bias = np.zeros(6)  # match the loader (no bias subtraction)
    command_vec, force_torque_vec, proprioception_vec = provider._assemble_streams(
        observation, command
    )

    np.testing.assert_allclose(command_vec, loader_episode.command[0].numpy(), atol=1e-6)
    np.testing.assert_allclose(force_torque_vec, loader_episode.force_torque[0].numpy(), atol=1e-6)
    np.testing.assert_allclose(
        proprioception_vec, loader_episode.proprioception[0].numpy(), atol=1e-6
    )


# ---------------------------------------------------------------------------
# F/T bias capture + per-episode auto-reset
# ---------------------------------------------------------------------------


def test_ft_bias_captured_on_first_step_of_episode():
    provider = _provider()
    first = _observation(sim_time=0.0, wrist_ft=np.array([5.0, -2.0, 1.0, 0.0, 0.0, 0.0]))
    provider.get_delta(first, _command())
    assert provider._ft_bias is not None
    np.testing.assert_allclose(provider._ft_bias, first.wrist_ft)

    # A later step in the same episode (sim_time increased) keeps the same bias.
    provider.get_delta(_observation(sim_time=0.5, wrist_ft=np.full(6, 9.0)), _command())
    assert provider._ft_bias is not None
    np.testing.assert_allclose(provider._ft_bias, first.wrist_ft)


def test_sim_time_restart_recaptures_bias_and_resets_hidden():
    provider = _provider()
    provider.get_delta(_observation(sim_time=0.0), _command())
    provider.get_delta(_observation(sim_time=0.5), _command())
    assert provider._hidden is not None

    # New episode: sim_time jumps back → bias re-captured to the new episode's F/T.
    new_episode_ft = np.array([7.0, 7.0, 7.0, 0.0, 0.0, 0.0])
    provider.get_delta(_observation(sim_time=0.0, wrist_ft=new_episode_ft), _command())
    assert provider._ft_bias is not None
    np.testing.assert_allclose(provider._ft_bias, new_episode_ft)


def test_explicit_reset_clears_state():
    provider = _provider()
    provider.get_delta(_observation(sim_time=0.1), _command())
    provider.reset()
    assert provider._hidden is None
    assert provider._ft_bias is None
    assert provider._last_sim_time is None


def test_hidden_state_advances_across_steps():
    provider = _provider()
    provider.get_delta(_observation(sim_time=0.1), _command())
    assert provider._hidden is not None
    hidden_after_one = provider._hidden.clone()
    provider.get_delta(_observation(sim_time=0.2), _command())
    hidden_after_two = provider._hidden
    assert hidden_after_two is not None
    assert not torch.allclose(hidden_after_one, hidden_after_two)


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_round_trips_outputs(tmp_path: Path):
    config = PolicyConfig(hidden_size=16, num_layers=1)
    torch.manual_seed(3)
    model = ResidualPolicy(config).eval()
    stats = _identity_stats()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, config=config, norm_stats=stats, loss_config=LossConfig())

    loaded = load_checkpoint(path)
    assert loaded.config == config
    assert loaded.data_schema_version != "unknown"

    original = LearnedResidual(model, stats)
    restored = LearnedResidual.from_checkpoint(path)
    observation, command = _observation(), _command()
    delta_original = original.get_delta(observation, command)
    delta_restored = restored.get_delta(observation, command)
    np.testing.assert_allclose(delta_original.delta_position, delta_restored.delta_position)
    np.testing.assert_allclose(delta_original.delta_orientation, delta_restored.delta_orientation)
    assert delta_original.delta_grip_force == pytest.approx(delta_restored.delta_grip_force)


def test_checkpoint_with_retired_config_key_still_loads(tmp_path: Path):
    """A checkpoint carrying a config key the current PolicyConfig no longer defines
    must still load (finding A-3).

    Retiring a knob — ``use_tanh_head`` was the first — otherwise strands every
    checkpoint trained before the removal, since ``PolicyConfig(**payload)`` raises on
    an unexpected keyword. All 12 of the project's trained runs carried that key.
    """
    config = PolicyConfig(hidden_size=16, num_layers=1)
    path = tmp_path / "legacy.pt"
    save_checkpoint(
        path, model=ResidualPolicy(config).eval(), config=config, norm_stats=_identity_stats()
    )

    # Re-write the payload with a key no current field matches, as an old run would have.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["config"]["use_tanh_head"] = False
    torch.save(payload, path)

    assert load_checkpoint(path).config == config


# ---------------------------------------------------------------------------
# Real-time latency budget
# ---------------------------------------------------------------------------


def test_per_step_inference_within_latency_budget():
    """Mean ``get_delta`` cost must fit the control budget (design nominal ~10 ms)."""
    provider = _provider(hidden_size=128, num_layers=2)  # the deployment default core
    observation, command = _observation(sim_time=0.1), _command()
    provider.get_delta(observation, command)  # warm up

    iterations = 100
    start = time.perf_counter()
    for step in range(iterations):
        provider.get_delta(_observation(sim_time=0.1 + 0.002 * step), command)
    mean_ms = (time.perf_counter() - start) / iterations * 1e3
    assert mean_ms < 10.0, f"mean get_delta {mean_ms:.2f} ms exceeds the 10 ms budget"


# ---------------------------------------------------------------------------
# Seam swap-in through the real run_episode (integration)
# ---------------------------------------------------------------------------

_SCENE_PATH = Path(__file__).resolve().parents[1] / "assets" / "mjcf" / "full_scene.xml"


def test_learned_residual_runs_in_run_episode_unchanged():
    """``LearnedResidual`` slots into ``run_episode`` where ``NoAssist`` was, with
    no runner/input/controller edit, and every applied Δ stays within bounds."""
    if not _SCENE_PATH.exists():
        pytest.skip(f"scene file not found: {_SCENE_PATH}")

    from ai_teleop.control import Controller
    from ai_teleop.input import ScriptedNoisyHuman
    from ai_teleop.sim.runner import run_episode
    from ai_teleop.sim.scene import SimEnv

    env = SimEnv(str(_SCENE_PATH), render_mode="headless")
    try:
        observation = env.reset()
        controller = Controller(env)
        target_position = observation.hole_poses[0][:3].copy()  # task goal: hole_0
        target_pose = np.concatenate([target_position, controller.home_pose[3:]])
        human = ScriptedNoisyHuman(target_pose, seed=0)
        assist = LearnedResidual(_model(), _identity_stats())

        applied_deltas: list[Delta] = []

        def _spy(step, obs, base_command, delta, command):  # noqa: ANN001, ARG001
            applied_deltas.append(delta)
            return False

        result = run_episode(env, controller, human, assist, max_steps=60, step_callback=_spy)
    finally:
        env.close()

    assert result.n_steps == 60
    assert len(applied_deltas) == 60
    for delta in applied_deltas:
        assert np.linalg.norm(delta.delta_position) <= 0.03 + 1e-9
        assert np.linalg.norm(delta.delta_orientation) <= np.deg2rad(10.0) + 1e-9
        # apply_delta must accept it without error (the seam combine step).
        apply_delta(human.get_command(observation), delta)


# ---------------------------------------------------------------------------
# Vision embedding cache (deployment tick budget)
# ---------------------------------------------------------------------------


def _vision_provider(*, seed: int = 0) -> LearnedResidual:
    torch.manual_seed(seed)
    model = ResidualPolicy(PolicyConfig(hidden_size=16, num_layers=1, use_vision=True)).eval()
    return LearnedResidual(model, _identity_stats())


def _vision_observation(frame: np.ndarray, *, sim_time: float = 0.0) -> Observation:
    observation = _observation(sim_time=sim_time)
    object.__setattr__(observation, "wrist_image", frame)
    return observation


def test_embedding_cache_does_not_change_the_policy_output():
    """The cache must be a pure speedup — same deltas, to the bit.

    Drives two identical policies over the same held-frame sequence. One gets the very
    same array object each tick (cache hits); the other gets a fresh copy with identical
    contents (cache misses, i.e. the old re-encode-every-tick path). Any divergence means
    the cache changed behaviour rather than just cost.
    """
    # `arange(..., dtype=uint8)` already wraps at 256; an explicit `% 256` is redundant
    # and, under numpy 2's NEP 50 rules, an OverflowError (256 does not fit in uint8).
    frame = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    cached, recomputed = _vision_provider(), _vision_provider()

    for tick in range(6):
        sim_time = tick * 0.002
        held = cached.get_delta(_vision_observation(frame, sim_time=sim_time), _command())
        fresh = recomputed.get_delta(
            _vision_observation(frame.copy(), sim_time=sim_time), _command()
        )
        assert np.array_equal(held.delta_position, fresh.delta_position), f"tick {tick}"
        assert np.array_equal(held.delta_orientation, fresh.delta_orientation), f"tick {tick}"
        assert held.delta_grip_force == fresh.delta_grip_force, f"tick {tick}"


def test_embedding_is_reused_only_while_the_frame_object_is_held():
    """The whole speedup rests on identity: same object reuses, new object re-encodes."""
    provider = _vision_provider()
    encoder = provider._model.image_encoder
    assert encoder is not None
    encoder_calls = 0

    def count(module, args, output):
        nonlocal encoder_calls
        encoder_calls += 1

    encoder.register_forward_hook(count)

    held = np.zeros((224, 224, 3), dtype=np.uint8)
    for tick in range(20):  # one render interval of held frames
        provider.get_delta(_vision_observation(held, sim_time=tick * 0.002), _command())
    assert encoder_calls == 1, "held frame must be encoded once, not once per tick"

    provider.get_delta(_vision_observation(held.copy(), sim_time=0.040), _command())
    assert encoder_calls == 2, "a newly rendered frame must be encoded"


def test_reset_drops_the_cached_embedding():
    """A new episode must not inherit the previous one's embedding via a reused address."""
    provider = _vision_provider()
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    provider.get_delta(_vision_observation(frame, sim_time=0.0), _command())
    assert provider._cached_embedding is not None
    provider.reset()
    assert provider._cached_embedding is None
    assert provider._cached_frame is None


# ---------------------------------------------------------------------------
# Vision fast path — accelerating only the CNN (CUDA graph)
# ---------------------------------------------------------------------------


def _vision_provider_on(device: str | None, *, seed: int = 0) -> LearnedResidual:
    torch.manual_seed(seed)
    model = ResidualPolicy(PolicyConfig(hidden_size=16, num_layers=1, use_vision=True)).eval()
    return LearnedResidual(model, _identity_stats(), image_encoder_device=device)


def test_default_keeps_the_encoder_with_the_rest_of_the_policy():
    """Opting out must be the no-op it looks like: one device, plain encoder, no capture."""
    provider = _vision_provider_on(None)
    assert provider._image_device == provider._device
    assert provider._encode_image is provider._model.image_encoder


def test_an_ft_only_policy_ignores_the_image_encoder_device():
    """No CNN, nothing to place — asking for one must not build or break anything."""
    provider = LearnedResidual(
        ResidualPolicy(PolicyConfig(hidden_size=16, num_layers=1)).eval(),
        _identity_stats(),
        image_encoder_device="cuda",
    )
    assert provider._encode_image is None
    assert provider.use_vision is False
    assert isinstance(provider.get_delta(_observation(), _command()), Delta)


def test_an_unusable_encoder_device_falls_back_instead_of_raising():
    """A missing GPU is a performance problem, not a reason to lose the run."""
    provider = _vision_provider_on("cuda:99")
    assert provider._image_device == provider._device
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    assert isinstance(provider.get_delta(_vision_observation(frame), _command()), Delta)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_graph_encoder_matches_the_cpu_encoder():
    """The fast path must change only where the CNN runs, not what the policy does.

    Tolerance rather than bit-equality: cuDNN picks different convolution algorithms than
    the CPU kernels, so the two agree to float32 accumulation noise. This bound is tight
    enough to catch the regression that matters — leaving TF32 on (PyTorch's default for
    convolutions) lands ~7e-5 out on the Δ, comfortably failing here.

    Errors compound across ticks because the GRU carries the difference forward, so this
    walks several ticks rather than checking one.
    """
    from ai_teleop.policy.residual_policy import _GraphedImageEncoder

    on_cpu, on_gpu = _vision_provider_on(None), _vision_provider_on("cuda")
    assert isinstance(on_gpu._encode_image, _GraphedImageEncoder)

    rng = np.random.default_rng(0)
    for tick in range(4):
        frame = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        sim_time = tick * 0.002
        expected = on_cpu.get_delta(_vision_observation(frame, sim_time=sim_time), _command())
        actual = on_gpu.get_delta(_vision_observation(frame, sim_time=sim_time), _command())
        assert np.allclose(expected.delta_position, actual.delta_position, atol=1e-6)
        assert np.allclose(expected.delta_orientation, actual.delta_orientation, atol=1e-6)
        assert actual.delta_position.dtype == np.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_graphed_embedding_survives_the_next_replay():
    """A graph replays into fixed memory, so the cached embedding must be a copy of it.

    Without the clone, the embedding held across a render interval would be silently
    rewritten by the next frame's replay — the cache would return the *newest* frame's
    features while believing it held the old one's.
    """
    provider = _vision_provider_on("cuda")
    first = provider._image_embedding(_vision_observation(np.zeros((224, 224, 3), np.uint8)))
    before = first.clone()
    provider._image_embedding(_vision_observation(np.full((224, 224, 3), 200, np.uint8)))
    assert torch.equal(first, before), "the first embedding was aliased to the graph's buffer"
