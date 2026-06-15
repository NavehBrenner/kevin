"""ScriptedNoisyHuman — deterministic stub InputStrategy for data generation.

Each tick emits a Command whose target is the constructor-supplied goal pose
plus per-axis Gaussian noise on position and a small random orientation jitter.
The noise model is intentionally simple: no drift, no intent phases, no
release/withdraw behaviour — those are deferred to M4.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ai_teleop.common.command import Command
from ai_teleop.common.observation import Observation


class ScriptedNoisyHuman:
    """Scripted actor that tracks a fixed target EE pose with additive noise.

    Parameters
    ----------
    target_pose:
        Shape (7,) array — (px, py, pz, qw, qx, qy, qz) in world frame.
        Typically the hole-vicinity pose for the active trial.
    position_noise_std:
        Per-axis Gaussian σ for position noise, in metres.
    orientation_noise_std:
        Gaussian σ for each component of the axis-angle jitter, in radians.
    seed:
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        target_pose: np.ndarray,
        position_noise_std: float = 0.005,
        orientation_noise_std: float = float(np.deg2rad(2.0)),
        seed: int = 0,
    ) -> None:
        if target_pose.shape != (7,):
            raise ValueError(f"target_pose must have shape (7,), got {target_pose.shape}")
        self._target_position = target_pose[:3].copy()
        self._target_quaternion = target_pose[3:].copy()
        mujoco.mju_normalize4(self._target_quaternion)
        self._position_noise_std = position_noise_std
        self._orientation_noise_std = orientation_noise_std
        self._rng = np.random.default_rng(seed)

    def get_command(self, observation: Observation) -> Command:  # noqa: ARG002
        noisy_position = self._target_position + self._rng.normal(
            0.0, self._position_noise_std, size=3
        )

        # Build a small random orientation jitter as axis-angle, then compose
        # with the target quaternion (world-frame left-multiply, same convention
        # as apply_delta in domain/).
        axis_angle = self._rng.normal(0.0, self._orientation_noise_std, size=3)
        angle = float(np.linalg.norm(axis_angle))
        jitter_quat = np.zeros(4)
        if angle > 0.0:
            mujoco.mju_axisAngle2Quat(jitter_quat, axis_angle / angle, angle)
        else:
            jitter_quat[0] = 1.0  # identity

        noisy_quat = np.zeros(4)
        mujoco.mju_mulQuat(noisy_quat, jitter_quat, self._target_quaternion)
        mujoco.mju_normalize4(noisy_quat)

        return Command(noisy_position, noisy_quat)
