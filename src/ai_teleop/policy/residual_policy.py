"""``LearnedResidual`` — the trained Phase-1 residual as an ``AssistProvider`` (LAB-34).

This is the deployment side of M5: it wraps a trained checkpoint so the learned
policy slots into the M3 seam exactly where ``NoAssist`` / ``Expert`` do, with **no
edit to the runner, input strategy, or controller** (the dependency-inversion
property the seam exists to provide). ``run_episode`` calls ``get_delta`` each tick;
the wrapper advances the GRU hidden state by one ``model.step`` and returns a
``clamp_delta``'d ``Delta``.

Two correctness details that must mirror the M4 training pipeline exactly, or the
policy sees a different input distribution than it trained on (silent covariate
shift):

1. **F/T bias subtraction.** ``data.generate`` logs ``wrist_ft - ft_bias`` where
   ``ft_bias`` is the *raw* wrist F/T captured at the episode's reset
   (``generate.py``). The runtime ``Observation.wrist_ft`` is raw, so the wrapper
   re-captures that bias on the first observation of each episode and subtracts it.
2. **Stream assembly.** The per-step command / F/T / proprioception vectors are
   built identically to ``data.dataset.extract_training_episode`` (same column
   order, same quaternion→6D map) and then z-scored with the checkpoint's stored
   train-set normalization. ``tests/test_residual_policy.py`` asserts this matches
   the loader to guard against drift.

**Per-episode reset.** The GRU hidden state and the F/T bias must reset between
episodes, but ``AssistProvider.get_delta`` has no reset signal. Rather than add a
reset hook to the shared runner (which would weaken the "drop-in, no runner edit"
guarantee), the wrapper is **self-resetting**: it watches ``observation.sim_time``
(monotonic within an episode, reset toward 0 at episode start) and clears its state
when it sees the clock jump backwards. An explicit :meth:`reset` is also exposed for
callers that prefer to be explicit (e.g. the M6 ablation orchestration).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ai_teleop.common.command import Command
from ai_teleop.common.observation import Observation
from ai_teleop.common.utils.rotations import quat_to_6d
from ai_teleop.data.dataset import INPUT_STREAMS, NormStats
from ai_teleop.data.trajectory import SCHEMA_VERSION as DATA_SCHEMA_VERSION
from ai_teleop.domain import Delta, clamp_delta
from ai_teleop.policy.config import PolicyConfig
from ai_teleop.policy.losses import LossConfig
from ai_teleop.policy.model import ResidualPolicy

# Bumped when the checkpoint payload layout changes (independent of the data schema).
POLICY_CHECKPOINT_VERSION = "1.0"

# A backward jump in sim_time larger than this signals a new episode (the clock is
# monotonic within an episode and resets toward 0 at reset).
_EPISODE_RESET_SIM_TIME_DROP = 1e-6


@dataclass
class LoadedCheckpoint:
    """A checkpoint reconstructed from disk: an eval-mode model + its provenance."""

    model: ResidualPolicy
    config: PolicyConfig
    norm_stats: NormStats
    loss_config: LossConfig | None
    train_history: dict[str, list[float]] | None
    policy_checkpoint_version: str
    data_schema_version: str


def save_checkpoint(
    path: str | Path,
    *,
    model: ResidualPolicy,
    config: PolicyConfig,
    norm_stats: NormStats,
    loss_config: LossConfig | None = None,
    train_history: dict[str, list[float]] | None = None,
) -> None:
    """Serialize weights + normalization stats + hyperparameters + schema versions.

    Everything needed to rebuild a deployable policy from disk: the model weights,
    the train-set normalization (so inference normalizes identically), the model
    and loss hyperparameters, and both schema versions (to flag a stale checkpoint
    against a changed corpus / payload).
    """
    payload = {
        "policy_checkpoint_version": POLICY_CHECKPOINT_VERSION,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "config": asdict(config),
        "model_state_dict": model.state_dict(),
        "norm_stats": {
            "mean": {stream: norm_stats.mean[stream] for stream in INPUT_STREAMS},
            "std": {stream: norm_stats.std[stream] for stream in INPUT_STREAMS},
        },
        "loss_config": asdict(loss_config) if loss_config is not None else None,
        "train_history": train_history,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> LoadedCheckpoint:
    """Rebuild a :class:`LoadedCheckpoint` (eval-mode model + provenance) from disk."""
    # weights_only=False: the payload carries our own config/stats dicts, not just
    # tensors. The file is a first-party training artifact, so this is trusted.
    payload = torch.load(path, map_location=map_location, weights_only=False)

    config = PolicyConfig(**payload["config"])
    norm_stats = NormStats(mean=payload["norm_stats"]["mean"], std=payload["norm_stats"]["std"])

    model = ResidualPolicy(config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    loss_config_payload = payload.get("loss_config")
    loss_config = LossConfig(**loss_config_payload) if loss_config_payload is not None else None

    return LoadedCheckpoint(
        model=model,
        config=config,
        norm_stats=norm_stats,
        loss_config=loss_config,
        train_history=payload.get("train_history"),
        policy_checkpoint_version=payload.get("policy_checkpoint_version", "unknown"),
        data_schema_version=payload.get("data_schema_version", "unknown"),
    )


class LearnedResidual:
    """Trained Phase-1 residual as a stateful, real-time ``AssistProvider``.

    Build from a checkpoint with :meth:`from_checkpoint`, or pass a model +
    normalization stats directly (handy in tests). Reused across episodes safely:
    state auto-resets on a detected ``sim_time`` restart (or call :meth:`reset`).
    """

    def __init__(
        self,
        model: ResidualPolicy,
        norm_stats: NormStats,
        *,
        device: str = "cpu",
    ) -> None:
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()
        # Stash normalization as device tensors so per-step z-scoring is allocation-free.
        self._mean = {stream: norm_stats.mean[stream].to(self._device) for stream in INPUT_STREAMS}
        self._std = {stream: norm_stats.std[stream].to(self._device) for stream in INPUT_STREAMS}

        self._hidden: Tensor | None = None
        self._ft_bias: np.ndarray | None = None
        self._last_sim_time: float | None = None

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str = "cpu") -> LearnedResidual:
        """Load a trained checkpoint into a deployable provider."""
        loaded = load_checkpoint(path, map_location=device)
        return cls(loaded.model, loaded.norm_stats, device=device)

    def reset(self) -> None:
        """Clear the GRU hidden state and F/T bias — call at episode start.

        A no-op-safe equivalent happens automatically on a detected ``sim_time``
        restart, so callers driving a single ``run_episode`` need not call this.
        """
        self._hidden = None
        self._ft_bias = None
        self._last_sim_time = None

    def _is_new_episode(self, observation: Observation) -> bool:
        """True on the first call ever, or when ``sim_time`` jumps backward (reset)."""
        if self._last_sim_time is None:
            return True
        return observation.sim_time < self._last_sim_time - _EPISODE_RESET_SIM_TIME_DROP

    def _assemble_streams(
        self, observation: Observation, command: Command
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the raw per-step (command, F/T, proprioception) vectors.

        Mirrors ``data.dataset.extract_training_episode`` for a single step: same
        column order, same quaternion→6D map, and the **bias-subtracted** F/T
        (``self._ft_bias`` is captured on the episode's first observation).
        """
        assert self._ft_bias is not None  # set by get_delta before this is called

        command_vector = np.concatenate(
            [command.target_position, quat_to_6d(command.target_quaternion)]
        )  # (9,)
        force_torque_vector = observation.wrist_ft - self._ft_bias  # (6,) bias-subtracted
        proprioception_vector = np.concatenate(
            [
                observation.ee_pose[:3],
                quat_to_6d(observation.ee_pose[3:7]),
                observation.joint_positions,
                observation.joint_velocities,
                [observation.gripper_width],
            ]
        )  # (24,)
        return command_vector, force_torque_vector, proprioception_vector

    def _normalized_step_tensor(self, stream: str, vector: np.ndarray) -> Tensor:
        """Z-score one stream and shape it ``(1, dim)`` for ``model.step``."""
        raw = torch.as_tensor(vector, dtype=torch.float32, device=self._device)
        return ((raw - self._mean[stream]) / self._std[stream]).unsqueeze(0)

    def get_delta(self, observation: Observation, command: Command) -> Delta:
        """Advance the policy one step and return the clamped correction Δ.

        ``command`` is the **base** operator command (pre-Δ) the seam hands in —
        exactly the ``cmd_*`` the training corpus logged.
        """
        if self._is_new_episode(observation):
            self.reset()
        if self._ft_bias is None:
            self._ft_bias = np.asarray(observation.wrist_ft, dtype=np.float64).copy()
        self._last_sim_time = observation.sim_time

        command_vector, force_torque_vector, proprioception_vector = self._assemble_streams(
            observation, command
        )
        command_tensor = self._normalized_step_tensor("command", command_vector)
        force_torque_tensor = self._normalized_step_tensor("force_torque", force_torque_vector)
        proprioception_tensor = self._normalized_step_tensor(
            "proprioception", proprioception_vector
        )

        with torch.no_grad():
            raw_delta, self._hidden = self._model.step(
                command_tensor, force_torque_tensor, proprioception_tensor, hidden=self._hidden
            )
        delta = raw_delta.squeeze(0).cpu().numpy()  # (7,)

        return clamp_delta(
            Delta(
                delta_position=delta[0:3].astype(np.float64),
                delta_orientation=delta[3:6].astype(np.float64),
                delta_grip_force=float(delta[6]),
            )
        )
