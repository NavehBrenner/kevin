"""Analytical privileged-info expert — the behavioral-cloning teacher (M4).

A closed-form, geometry-driven :class:`~ai_teleop.domain.AssistProvider`. It
reads the **privileged** true peg/hole poses out of the ``Observation`` and the
operator's noisy ``Command``, and returns a clamped correction ``Δ*`` with the
*same signature* as the future learned policy. It is allowed to "cheat" (it sees
true geometry); the policy will have to reproduce its output from non-privileged
observation alone. See ``docs/design/expert-corrections.md``.

Per-step law (align-then-advance), all in the world frame:

1. Split the tip→hole error ``e = p_hole − p_tip`` into the component along the
   insertion axis ``n`` and the lateral remainder ``e_lat``.
2. **Lateral alignment** — command a shift of ``e_lat`` to bring the tip onto the
   hole axis.
3. **Angular alignment** — rotate the peg long-axis ``a`` onto ``n`` (smallest
   rotation), expressed as a world-frame axis-angle the seam's ``apply_delta``
   composes by left-multiply.
4. **Axial advance — gated by alignment** — only once lateral + angular error are
   within tolerance do we advance the tip along ``+n`` into the hole.
5. **Approach-speed braking (LAB-98)** — retract the command's axial *lead*
   (how far past the arm the operator's command sits along the bore) down to a
   distance-proportional allowance, so the arm decelerates before contact
   instead of slamming the wall at the operator's sweep speed. Under the
   deployment controller config (``joint_damping=1.5``) the arm tracks the
   command tightly, so a hasty operator's contact speed is set by the command
   lead — the one thing the expert can shrink. Off by default
   (``brake_gain=0``): the kd=4 data-gen config suppressed impact transients
   in the controller itself, so pre-LAB-98 corpora never needed it. Note the
   brake reads only ``command − ee_pose`` — *non-privileged* streams — so this
   correction component is fully inferable by the deployed policy.
6. **Grip** — reduce grip on a detected jam (lateral contact force) so a
   slightly-wedged peg can slip free.

The whole correction is multiplied by a smooth **distance gate** ``g(d)`` that is
**zero by construction** for ``d ≥ d_far`` — far from the hole the expert is a
no-op, matching what the deployed policy can support (F/T ≈ 0 in free space, no
exteroception in Phase 1). The final clamp uses the shared residual-interface
bounds via :func:`~ai_teleop.domain.clamp_delta`.

Scene conventions (current MJCF):
- Peg long axis ``a`` is the peg body's local +z; the tip is ``peg_half_length``
  along ``+a`` from the body origin (the ``peg_tip`` site at z=0.030).
- The hole-site local +x axis is the bore / insertion axis ``n`` (the wall
  normal, pointing into the wall). The sites are world-aligned, so this is
  world +x in practice.
"""

from __future__ import annotations

import numpy as np

from ai_teleop.common.command import Command
from ai_teleop.common.geometry import axis_from_quat, quat_to_matrix
from ai_teleop.common.observation import Observation
from ai_teleop.domain import ZERO_DELTA, Delta, clamp_delta


def _smoothstep_gate(distance: float, d_near: float, d_far: float) -> float:
    """C¹ gate: 1 at/under d_near, 0 at/over d_far, Hermite ramp between.

    Zero **by construction** for ``distance >= d_far`` — the property that makes
    the expert's far-field correction structurally (not approximately) zero.
    """
    if distance >= d_far:
        return 0.0
    if distance <= d_near:
        return 1.0
    t = (d_far - distance) / (d_far - d_near)  # 0 at d_far → 1 at d_near
    return float(t * t * (3.0 - 2.0 * t))


class Expert:
    """Closed-form align-then-advance expert (an ``AssistProvider``).

    Parameters
    ----------
    target_hole_index:
        Which hole (index into ``observation.hole_poses``) the expert servos
        toward. Supplied by the episode/task layer — the observation no longer
        carries the goal. Generated walls put the target at ``hole_0``.
    peg_half_length:
        Distance from the peg body origin to its tip along the long axis (m).
    d_near, d_far:
        Distance-gate band (m): full authority at/under ``d_near``, zero at/over
        ``d_far``. ``d_far`` sets where the expert starts engaging on approach.
    epsilon_lateral, epsilon_angular:
        Alignment tolerances (m, rad) that gate the axial advance.
    advance_per_step:
        Capped axial advance toward the hole, per step (m), once aligned.
    brake_gain, brake_lead_floor:
        Approach-speed brake (LAB-98). When ``brake_gain > 0``, the command's
        axial lead beyond ``brake_gain * distance + brake_lead_floor`` is
        subtracted from the correction, so the effective target the impedance
        law chases stays a controlled "carrot" ahead of the arm and the
        approach decelerates toward contact. ``brake_gain = 0`` (default)
        disables the brake entirely — pre-LAB-98 behavior, bit-exact.
    jam_force_threshold:
        Lateral wrist-force magnitude (N) above which a jam is declared.
    grip_reduce_force:
        Grip-force reduction applied on a detected jam (N). Placeholder default;
        calibrated against logged jam episodes.
    max_delta_position:
        Position bound (m) for the final clamp — the expert's per-step
        authority, and the label bound BC clones. ``None`` (default) uses the
        shared module bound in ``domain.delta``; data generation passes its
        fingerprinted per-corpus value so regenerating a legacy dataset clamps
        at the bound that corpus was recorded under (LAB-100).
    """

    def __init__(
        self,
        *,
        target_hole_index: int = 0,
        peg_half_length: float = 0.030,
        d_near: float = 0.01,
        d_far: float = 0.10,
        epsilon_lateral: float = 0.003,
        epsilon_angular: float = float(np.deg2rad(8.0)),
        advance_per_step: float = 0.01,
        brake_gain: float = 0.0,
        brake_lead_floor: float = 0.008,
        jam_force_threshold: float = 8.0,
        grip_reduce_force: float = 1.0,
        max_delta_position: float | None = None,
    ) -> None:
        self._target_hole_index = target_hole_index
        self._peg_half_length = peg_half_length
        self._d_near = d_near
        self._d_far = d_far
        self._epsilon_lateral = epsilon_lateral
        self._epsilon_angular = epsilon_angular
        self._advance_per_step = advance_per_step
        self._brake_gain = brake_gain
        self._brake_lead_floor = brake_lead_floor
        self._jam_force_threshold = jam_force_threshold
        self._grip_reduce_force = grip_reduce_force
        self._max_delta_position = max_delta_position

    def get_delta(self, observation: Observation, command: Command) -> Delta:
        # --- Privileged geometry -----------------------------------------
        peg_axis = axis_from_quat(observation.peg_pose[3:], 2)  # long axis = local +z
        peg_tip = observation.peg_pose[:3] + self._peg_half_length * peg_axis

        hole_pose = observation.hole_poses[self._target_hole_index]
        hole_position = hole_pose[:3]
        insertion_axis = quat_to_matrix(hole_pose[3:])[:, 0]  # bore = hole local +x

        error = hole_position - peg_tip
        distance = float(np.linalg.norm(error))

        gate = _smoothstep_gate(distance, self._d_near, self._d_far)
        if gate == 0.0:
            return ZERO_DELTA  # far-field: structurally a no-op

        axial_error = float(error @ insertion_axis)
        lateral_error = error - axial_error * insertion_axis

        # --- Angular alignment: rotate peg axis a onto n -----------------
        align_axis = np.cross(peg_axis, insertion_axis)
        align_axis_norm = float(np.linalg.norm(align_axis))
        angular_error = float(np.arccos(np.clip(peg_axis @ insertion_axis, -1.0, 1.0)))
        align_rotation = (
            align_axis / align_axis_norm * angular_error if align_axis_norm > 1e-09 else np.zeros(3)
        )

        # --- Axial advance, gated by lateral + angular alignment ---------
        aligned = (
            float(np.linalg.norm(lateral_error)) < self._epsilon_lateral
            and angular_error < self._epsilon_angular
        )
        advance = np.zeros(3)
        if aligned and axial_error > 0.0:
            advance = min(self._advance_per_step, axial_error) * insertion_axis

        # --- Approach-speed brake (LAB-98) --------------------------------
        # Under the deployment controller config the arm tracks the command
        # tightly, so contact speed ≈ how far the command leads the arm along
        # the bore. Retract any lead beyond a distance-proportional allowance:
        # the effective target becomes a short carrot ahead of the arm, and the
        # approach decelerates (v ∝ lead) instead of slamming the wall.
        brake = np.zeros(3)
        if self._brake_gain > 0.0:
            command_lead = float(
                (command.target_position - observation.ee_pose[:3]) @ insertion_axis
            )
            allowed_lead = self._brake_gain * distance + self._brake_lead_floor
            brake = max(0.0, command_lead - allowed_lead) * insertion_axis

        # --- Grip modulation on a detected jam ---------------------------
        # A jam shows up as wrist force perpendicular to the bore while the peg
        # is still mis-aligned (catching on the rim) — reduce grip so it slips.
        wrist_force = observation.wrist_ft[:3]
        lateral_wrist_force = float(
            np.linalg.norm(wrist_force - (wrist_force @ insertion_axis) * insertion_axis)
        )
        jammed = (not aligned) and lateral_wrist_force > self._jam_force_threshold
        delta_grip = -self._grip_reduce_force if jammed else 0.0

        delta = Delta(
            delta_position=gate * (lateral_error + advance - brake),
            delta_orientation=gate * align_rotation,
            delta_grip_force=gate * delta_grip,
        )
        return clamp_delta(delta, max_delta_position=self._max_delta_position)
