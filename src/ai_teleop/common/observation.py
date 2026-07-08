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

    # Which hole is the *goal* is not here: that is an episode/task concept the
    # task layer owns (it picks the index and feeds it to the expert, the
    # operator's target, and the seating/observer scoring). The env reports every
    # hole's pose as privileged sensing and stays agnostic to the objective.

    # --- Timing ----------------------------------------------------------
    sim_time: float  # seconds since reset

    # --- Wrist camera (Phase-2 vision policy; real exteroception) --------
    # A (H, W, 3) uint8 RGB frame from the wrist camera, or None when the env is
    # not capturing images (F/T-only runs, data-gen, every pre-M7 caller). The env
    # is the frame-rate limiter: it holds the last frame and re-renders only on its
    # capture cadence, so the policy reads the most-recent frame and stays stateless
    # (see SimEnv.enable_wrist_capture). Not privileged — a real wrist camera
    # provides this at deploy, unlike the peg/hole ground truth above. Defaulted so
    # every existing construction site (data-gen, tests, trace replay) is unchanged.
    wrist_image: np.ndarray | None = None
