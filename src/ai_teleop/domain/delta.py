"""Delta dataclass, apply_delta, and NoAssist — the seam's core primitives.

Per-tick combine step:

    base_command = input_strategy.get_command(observation)
    delta        = assist.get_delta(observation, base_command)
    command      = apply_delta(base_command, delta)   # → Controller.compute

Per-step Δ bounds (distinct from the controller's command clamp added in M2):
    |Δposition| ≤ 3 cm, |Δorientation| ≤ 10°, |Δgrip| ≤ 5 N.

Quaternion composition uses MuJoCo helpers for consistency with the rest of
the stack. Δorientation is a world-frame rotation (left-multiply).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ai_teleop.common.command import Command
from ai_teleop.common.observation import Observation

# Raised 0.02 → 0.03 by LAB-100: under the deployment controller config the
# ±2 cm bound was the binding constraint on the expert's approach-speed brake
# (LAB-98's saturated residual aborts). 3 cm is the smallest bound that stops
# the clamp saturating on success episodes and matches 4 cm's measured ceiling
# on both sweep seed families. Corpora record their own bound (`delta_clamp`
# in metadata), so pre-LAB-100 datasets regenerate at their original ±2 cm.
_MAX_DELTA_POSITION: float = 0.03
_MAX_DELTA_ORIENTATION: float = float(np.deg2rad(10.0))
_MAX_DELTA_GRIP_FORCE: float = 5.0


@dataclass(frozen=True)
class Delta:
    delta_position: np.ndarray  # (3,) world frame, metres
    delta_orientation: np.ndarray  # (3,) axis-angle, radians
    delta_grip_force: float = 0.0  # newtons


ZERO_DELTA: Delta = Delta(np.zeros(3), np.zeros(3), 0.0)


def clamp_delta(delta: Delta, *, max_delta_position: float | None = None) -> Delta:
    """Clamp delta to the per-step Δ bounds from the residual-policy interface.

    ``max_delta_position`` overrides the position bound for callers that carry a
    per-corpus bound (data generation regenerating a legacy dataset must clamp
    the expert at the bound the corpus was recorded under — see
    ``data.generate``). ``None`` (the default, and the deployed-policy path)
    uses the module bound.
    """
    position_bound = _MAX_DELTA_POSITION if max_delta_position is None else max_delta_position
    position = delta.delta_position
    position_norm = float(np.linalg.norm(position))
    if position_norm > position_bound:
        position = position * (position_bound / position_norm)

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
