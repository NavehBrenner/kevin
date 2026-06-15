"""Delta dataclass, apply_delta, and NoAssist — the seam's core primitives.

Per-tick combine step:

    base_command = input_strategy.get_command(observation)
    delta        = assist.get_delta(observation, base_command)
    command      = apply_delta(base_command, delta)   # → Controller.compute

Per-step Δ bounds (distinct from the controller's command clamp added in M2):
    |Δposition| ≤ 2 cm, |Δorientation| ≤ 10°, |Δgrip| ≤ 5 N.

Quaternion composition uses MuJoCo helpers for consistency with the rest of
the stack. Δorientation is a world-frame rotation (left-multiply).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ai_teleop.common.command import Command
from ai_teleop.common.observation import Observation

_MAX_DELTA_POSITION: float = 0.02
_MAX_DELTA_ORIENTATION: float = float(np.deg2rad(10.0))
_MAX_DELTA_GRIP_FORCE: float = 5.0


@dataclass(frozen=True)
class Delta:
    delta_position: np.ndarray  # (3,) world frame, metres
    delta_orientation: np.ndarray  # (3,) axis-angle, radians
    delta_grip_force: float = 0.0  # newtons


ZERO_DELTA: Delta = Delta(np.zeros(3), np.zeros(3), 0.0)


def clamp_delta(delta: Delta) -> Delta:
    """Clamp delta to the per-step Δ bounds from the residual-policy interface."""
    position = delta.delta_position
    position_norm = float(np.linalg.norm(position))
    if position_norm > _MAX_DELTA_POSITION:
        position = position * (_MAX_DELTA_POSITION / position_norm)

    orientation = delta.delta_orientation
    orientation_norm = float(np.linalg.norm(orientation))
    if orientation_norm > _MAX_DELTA_ORIENTATION:
        orientation = orientation * (_MAX_DELTA_ORIENTATION / orientation_norm)

    grip_force = float(
        np.clip(
            delta.delta_grip_force,
            -_MAX_DELTA_GRIP_FORCE,
            _MAX_DELTA_GRIP_FORCE,
        )
    )

    return Delta(position, orientation, grip_force)


def apply_delta(command: Command, delta: Delta) -> Command:
    """Clamp delta, then combine with command to produce the controller's input."""
    clamped = clamp_delta(delta)

    new_position = command.target_position + clamped.delta_position

    # Convert axis-angle to quaternion; compose as a world-frame left-multiply
    # (q_new = delta_quat * q_base) so Δorientation is expressed in world coords,
    # consistent with Command.target_quaternion's world-frame convention.
    orientation_norm = float(np.linalg.norm(clamped.delta_orientation))
    delta_quaternion = np.zeros(4)
    if orientation_norm > 0.0:
        axis = clamped.delta_orientation / orientation_norm
        mujoco.mju_axisAngle2Quat(delta_quaternion, axis, orientation_norm)
    else:
        delta_quaternion[0] = 1.0  # identity: w=1, xyz=0

    new_quaternion = np.zeros(4)
    mujoco.mju_mulQuat(new_quaternion, delta_quaternion, command.target_quaternion)
    mujoco.mju_normalize4(new_quaternion)

    new_grip_force = command.delta_grip_force + clamped.delta_grip_force

    return Command(new_position, new_quaternion, new_grip_force)


class NoAssist:
    """Zero-Δ AssistProvider — recovers no-assist mode without special-casing."""

    def get_delta(self, observation: Observation, command: Command) -> Delta:
        return ZERO_DELTA
