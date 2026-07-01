"""Peg-vs-hole seating geometry — the one shared definition of "how seated".

Both data generation (scoring the BC corpus as it is recorded) and evaluation
(scoring trials in the M6 harness) need the *same* physical answer to "where is
the peg tip relative to the target hole": penetration along the hole's insertion
axis and the lateral tip error. Defining that twice invites silent drift — if
the peg half-length or an axis convention changed in one place but not the
other, generation and evaluation would disagree on what "seated" means. So the
geometry lives here once, as a pure function of an :class:`Observation`.

This module is a leaf of the dependency DAG: it imports only numpy and
``common`` and carries no behaviour beyond the geometry (no thresholds, no
success/termination policy — those belong to whoever consumes the numbers).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_teleop.common.observation import Observation
from ai_teleop.common.utils.rotations import axis_from_quat

# The peg tip sits half a peg-length down the peg body's local +z axis. This is
# a fixed property of the peg body in the MJCF, identical for generation and
# evaluation — the constant that must not be duplicated.
PEG_HALF_LENGTH = 0.030


@dataclass(frozen=True)
class SeatingGeometry:
    """Peg tip vs. the target hole, computed once from one observation.

    * ``penetration`` — the tip's advance past the hole entry along the hole's
      insertion axis (its local +x); positive once the tip is inside.
    * ``lateral_error`` — the perpendicular tip offset from the hole axis.
    * ``distance`` — the raw tip→hole-centre gap.
    * ``target_hole_pose`` — the selected hole's pose, passed through so callers
      that log it need not re-index ``hole_poses``.
    """

    penetration: float
    lateral_error: float
    distance: float
    target_hole_pose: np.ndarray

    @classmethod
    def from_observation(cls, observation: Observation, target_hole_index: int) -> SeatingGeometry:
        """Seating geometry against ``hole_poses[target_hole_index]``.

        The target index is supplied by the caller (the episode/task layer), not
        read off the observation — the env does not know which hole is the goal.
        """
        peg_axis = axis_from_quat(observation.peg_pose[3:], 2)
        tip = observation.peg_pose[:3] + PEG_HALF_LENGTH * peg_axis

        hole_pose = observation.hole_poses[target_hole_index]
        insertion_axis = axis_from_quat(hole_pose[3:], 0)

        error = hole_pose[:3] - tip
        axial_error = float(error @ insertion_axis)
        return cls(
            penetration=-axial_error,
            lateral_error=float(np.linalg.norm(error - axial_error * insertion_axis)),
            distance=float(np.linalg.norm(error)),
            target_hole_pose=hole_pose,
        )
