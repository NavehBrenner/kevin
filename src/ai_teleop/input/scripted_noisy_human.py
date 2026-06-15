"""ScriptedNoisyHuman — the realistic structured-noise operator (M4).

A seedable, deterministic source of *realistically-wrong* coarse commands. It
is **not** a model of human cognition; it is a controllable actor so the expert
has something to correct and KPI runs are reproducible.

The design constraint (see `docs/design/human-generation.md`): the noise must be
**structured and low-frequency, not per-step white noise**. Per-step i.i.d.
noise (the M3 stub) collapses the expert's optimal correction to "negate the
injected noise" — a trivial, unphysical denoising task. Instead this actor
commits to a *biased, drifting, coarse* trajectory that is internally consistent
over time, so the correction problem stays a geometric/contact-reasoning one.

Composition of three layers, fully determined by the constructor `seed`:

    c_t = held(goal ⊕ drift_t)  ⊕  tremor_t
    goal = (p_hole + bias,  R_hole · ΔR_bias)        # fixed for the whole episode

- **Intent / bias** — `bias` (position + orientation) drawn **once** at
  construction. A systematic, consistent misjudgement of where the hole is; it
  does not resample per step. This consistency is what makes the correction
  non-trivial.
- **Drift** — a low-frequency Ornstein–Uhlenbeck process on position and
  orientation, refreshed at ``refresh_hz`` (~5–10 Hz) and **held** between
  refreshes. Models a human issuing discrete coarse intents while the controller
  runs fast; the drift is correlated over hundreds of ms, not white.
- **Tremor** — optional small per-tick high-frequency jitter (off by default),
  kept well below the magnitude that would make denoising the whole game.

The actor is deliberately **contact-unaware**: it always commands toward the
(biased) goal regardless of what the peg is touching — it will keep pushing into
flat wall if its goal sits off the hole, exactly the situation the assist must
rescue. The M2 controller's 2 cm/step command clamp turns the full-goal command
into a smooth bounded approach (the division of labour M3 established).

Noise *magnitudes* here are placeholders to be calibrated post-baseline; the
*form* (biased + drifting + coarse) is what is fixed now.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ai_teleop.common.command import Command
from ai_teleop.common.observation import Observation


def _axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert a (3,) axis-angle vector to a (w,x,y,z) unit quaternion."""
    quat = np.zeros(4)
    angle = float(np.linalg.norm(axis_angle))
    if angle > 0.0:
        mujoco.mju_axisAngle2Quat(quat, axis_angle / angle, angle)
    else:
        quat[0] = 1.0  # identity
    return quat


class ScriptedNoisyHuman:
    """Structured-noise scripted operator implementing :class:`InputStrategy`.

    Parameters
    ----------
    target_pose:
        Shape (7,) array — (px, py, pz, qw, qx, qy, qz) in world frame. The
        actor's *intended* goal (typically the active trial's hole-vicinity
        pose); the actor adds its own per-episode bias on top.
    position_bias_std:
        Per-episode constant position-bias σ, in metres (drawn once).
    orientation_bias_std:
        Per-episode constant angular-bias σ per axis-angle component, in radians.
    drift_position_std:
        Stationary σ of the position OU drift, in metres.
    drift_orientation_std:
        Stationary σ of the orientation OU drift, in radians.
    drift_tau:
        OU time constant, in seconds (larger ⇒ slower, more correlated drift).
    tremor_std:
        Per-tick high-frequency position-tremor σ, in metres. 0 disables tremor.
    refresh_hz:
        Rate at which the commanded target refreshes (~5–10 Hz); held between.
    control_hz:
        Control-loop rate (one ``get_command`` call per tick). Sets the
        refresh-hold length ``round(control_hz / refresh_hz)``.
    seed:
        RNG seed. The data-gen driver passes a per-episode seed so that
        ``(master_seed, episode_index)`` fully determines the command stream.
    """

    def __init__(
        self,
        target_pose: np.ndarray,
        *,
        position_bias_std: float = 0.01,
        orientation_bias_std: float = float(np.deg2rad(3.0)),
        drift_position_std: float = 0.005,
        drift_orientation_std: float = float(np.deg2rad(1.0)),
        drift_tau: float = 0.3,
        tremor_std: float = 0.0,
        refresh_hz: float = 8.0,
        control_hz: float = 500.0,
        seed: int = 0,
    ) -> None:
        if target_pose.shape != (7,):
            raise ValueError(f"target_pose must have shape (7,), got {target_pose.shape}")

        self._target_position = target_pose[:3].copy()
        self._target_quaternion = target_pose[3:].copy()
        mujoco.mju_normalize4(self._target_quaternion)

        self._drift_position_std = drift_position_std
        self._drift_orientation_std = drift_orientation_std
        self._tremor_std = tremor_std
        self._hold_steps = max(1, round(control_hz / refresh_hz))

        # OU decay over one refresh interval: stationary std preserved via the
        # sqrt(1 - beta^2) innovation scaling.
        refresh_dt = 1.0 / refresh_hz
        self._ou_beta = float(np.exp(-refresh_dt / drift_tau))
        self._ou_innovation = float(np.sqrt(1.0 - self._ou_beta**2))

        self._rng = np.random.default_rng(seed)

        # Per-episode constant bias — drawn ONCE, never resampled.
        self.position_bias: np.ndarray = self._rng.normal(0.0, position_bias_std, size=3)
        self.orientation_bias: np.ndarray = self._rng.normal(0.0, orientation_bias_std, size=3)
        self._goal_position = self._target_position + self.position_bias
        bias_quat = _axis_angle_to_quat(self.orientation_bias)
        self._goal_quaternion = np.zeros(4)
        mujoco.mju_mulQuat(self._goal_quaternion, bias_quat, self._target_quaternion)
        mujoco.mju_normalize4(self._goal_quaternion)

        # OU drift state (axis-angle for orientation), updated at each refresh.
        self._drift_position = np.zeros(3)
        self._drift_orientation = np.zeros(3)

        self._tick = 0
        self._held_position = self._goal_position.copy()
        self._held_quaternion = self._goal_quaternion.copy()

    def _refresh_held_target(self) -> None:
        """Advance the OU drift one refresh step and recompute the held target."""
        self._drift_position = (
            self._ou_beta * self._drift_position
            + self._ou_innovation * self._drift_position_std * self._rng.normal(size=3)
        )
        self._drift_orientation = (
            self._ou_beta * self._drift_orientation
            + self._ou_innovation * self._drift_orientation_std * self._rng.normal(size=3)
        )

        self._held_position = self._goal_position + self._drift_position

        drift_quat = _axis_angle_to_quat(self._drift_orientation)
        held_quaternion = np.zeros(4)
        mujoco.mju_mulQuat(held_quaternion, drift_quat, self._goal_quaternion)
        mujoco.mju_normalize4(held_quaternion)
        self._held_quaternion = held_quaternion

    def get_command(self, observation: Observation) -> Command:  # noqa: ARG002
        # Refresh the held target at the start of each hold window (tick 0 too,
        # so the first command already carries one drift step).
        if self._tick % self._hold_steps == 0:
            self._refresh_held_target()
        self._tick += 1

        position = self._held_position
        if self._tremor_std > 0.0:
            position = position + self._rng.normal(0.0, self._tremor_std, size=3)

        return Command(position.copy(), self._held_quaternion.copy())
