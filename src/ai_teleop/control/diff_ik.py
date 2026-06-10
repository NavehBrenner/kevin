"""Damped-least-squares differential IK with null-space posture cost.

Pure-function module — no controller state, no MuJoCo handles. The caller
(see `ai_teleop.control.impedance`) is responsible for computing the
geometric Jacobian at the TCP site via `mujoco.mj_jacSite` and slicing it
to the seven arm DoFs before passing it here.

Solver
======

We treat the EE pose error as a 6-vector `e = [Δp; ω]` in the world frame
(translation in metres, rotation as axis-angle in radians) and ask: what
joint velocity `qdot ∈ R^7` produces that EE motion in a single timestep
of duration `dt`?

The base solve is damped-least-squares (DLS), which keeps the solution
bounded as the Jacobian approaches rank-deficiency:

    J⁺ = Jᵀ (J Jᵀ + λ² I)⁻¹            (7×6)
    qdot_task = J⁺ · (e / dt)

The Panda is redundant (7 DoF for a 6-DoF task), so any motion in the
null space of J leaves the EE pose unchanged. We use that slack to bias
the configuration toward a nominal posture `q_nominal` (typically the
M1 home pose):

    N = I − J⁺ J                       (7×7 null-space projector)
    qdot_posture = N · k_posture · (q_nominal − q)

    qdot = qdot_task + qdot_posture

Conventions
===========

- Joint angles in radians. EE position in metres. Quaternions
  `(w, x, y, z)`, unit-norm, world frame — same as `Observation` and
  `Command`.
- The orientation error is computed with `mujoco.mju_subQuat(res, qa, qb)`,
  which returns the axis-angle 3-vector `θ·n` such that
  `qa = exp(θ·n / 2) ⊗ qb`. With both quaternions in the world frame, the
  returned vector is the world-frame angular displacement from `qb` to
  `qa`, which is exactly what the rotational rows of the geometric
  Jacobian operate on (`jacr` is world-frame angular velocity).
- The Jacobian must be supplied with translational rows on top:
  `J = [jacp; jacr]`, shape `(6, 7)`.
"""

from __future__ import annotations

import mujoco
import numpy as np


def differential_ik(
    *,
    q: np.ndarray,
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    jacobian: np.ndarray,
    dt: float,
    damping: float,
    q_nominal: np.ndarray,
    posture_gain: float,
) -> np.ndarray:
    """Return a 7-vector joint-velocity command.

    Parameters
    ----------
    q : (7,)
        Current arm joint angles, rad.
    ee_pos, ee_quat : (3,), (4,)
        Current EE pose in the world frame. Quaternion is (w, x, y, z).
    target_pos, target_quat : (3,), (4,)
        Commanded EE pose in the world frame.
    jacobian : (6, 7)
        Geometric Jacobian at the EE site, world-frame, rows = [jacp; jacr].
    dt : float
        Control timestep (seconds). The DLS step solves for the joint
        velocity that closes `e` in one step of duration `dt`.
    damping : float
        DLS regularisation `λ`. Larger λ = smoother solutions near
        singularities at the cost of tracking accuracy.
    q_nominal : (7,)
        Posture the null-space projector biases towards.
    posture_gain : float
        Scalar `k_posture` on `(q_nominal − q)`.

    Returns
    -------
    qdot_des : (7,)
        Joint velocity (rad/s) that nudges the EE toward `target_pose`
        and spends null-space DoF toward `q_nominal`.
    """
    if q.shape != (7,) or q_nominal.shape != (7,):
        raise ValueError(f"q and q_nominal must be (7,), got {q.shape} and {q_nominal.shape}")
    if jacobian.shape != (6, 7):
        raise ValueError(f"jacobian must be (6, 7), got {jacobian.shape}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    pos_err = target_pos - ee_pos
    rot_err = np.zeros(3)
    mujoco.mju_subQuat(rot_err, target_quat, ee_quat)
    e = np.concatenate([pos_err, rot_err])

    J = jacobian
    # Solve (J Jᵀ + λ² I) x = e once instead of forming the inverse explicitly.
    JJT_reg = J @ J.T + (damping**2) * np.eye(6)
    j_pinv = J.T @ np.linalg.solve(JJT_reg, np.eye(6))

    qdot_task = j_pinv @ (e / dt)
    null_space = np.eye(7) - j_pinv @ J
    qdot_posture = null_space @ (posture_gain * (q_nominal - q))

    return qdot_task + qdot_posture
