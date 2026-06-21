"""Observation dataclass shared by SimEnv, controllers, expert, and policy.

Defined in `common/` rather than `sim/` because every downstream consumer
(input strategy, expert, residual policy) reads this object, and we don't
want them to depend on `sim/`.

All pose arrays use the convention (px, py, pz, qw, qx, qy, qz) — position
in metres, unit quaternion as [w, x, y, z] (the same layout MuJoCo uses for
free-joint qpos and body xquat). All quantities are expressed in the world
frame at the robot base, z up — see project-scope.md `World frame at the
robot base, z up.`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Observation:
    # --- Proprioception (always available to the policy) -----------------
    joint_positions: np.ndarray  # shape (7,)    rad,  panda joints 1..7
    joint_velocities: np.ndarray  # shape (7,)    rad/s
    ee_pose: np.ndarray  # shape (7,)    (px,py,pz,qw,qx,qy,qz) of the gripper TCP
    wrist_ft: np.ndarray  # shape (6,)    (Fx,Fy,Fz,Mx,My,Mz) at the wrist site, RAW
    gripper_width: float  # metres,  sum of the two finger joint openings

    # --- Privileged ground truth (training & evaluation only) ------------
    peg_pose: np.ndarray  # shape (7,)    peg body pose in world
    hole_poses: np.ndarray  # shape (N, 7)  every hole's pose in world
    target_hole_index: int  # which hole is the active target this trial

    # --- Timing ----------------------------------------------------------
    sim_time: float  # seconds since reset

    @property
    def target_hole_pose(self) -> np.ndarray:
        """Pose (7,) of the active target hole — ``hole_poses[target_hole_index]``."""
        return self.hole_poses[self.target_hole_index]

    @property
    def target_hole_position(self) -> np.ndarray:
        """Position (3,) of the active target hole (a view; ``.copy()`` to keep)."""
        return self.target_hole_pose[:3]
