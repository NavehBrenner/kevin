"""Direction-dependent Cartesian impedance controller — torque output.

The arm behaves like a spring–damper to its commanded EE pose, with
**diagonal stiffness expressed in the TCP frame** so the "stiff along
insertion, soft laterally" design intent rotates with the gripper instead
of staying glued to the world axes.

Law in pseudo-code:

    e_world = [target_pos − ee_pos;  axisangle(target_quat ⊗ ee_quat⁻¹)]
    ẋ_world = J · qdot                 # current EE twist
    e_tcp   = R⁻¹ · e_world             # block-diag(R⁻¹, R⁻¹) ⊗ 6-vector
    ẋ_tcp   = R⁻¹ · ẋ_world
    F_tcp   = K_diag_tcp · e_tcp  −  D_diag_tcp · ẋ_tcp     # (elem-wise)
    F_world = R · F_tcp
    τ_task     = Jᵀ · F_world
    τ_posture  = N · k_posture · (q_nominal − q)             # null-space P term
    τ_bias     = qfrc_bias                                   # gravity + Coriolis
    τ          = τ_task + τ_posture + τ_bias

`R` is the rotation matrix that maps TCP-frame coordinates to world
coordinates (so `R⁻¹ = Rᵀ` maps the other way). For a unit-quaternion
EE orientation this is just `data.site_xmat`.

`N = I − J⁺ J` is the null-space projector for the arm joints, with
`J⁺` the DLS pseudoinverse, computed inline below. `dls_damping`
controls its regularisation λ so the task and posture terms stay
consistent.

The function takes cached MuJoCo IDs (TCP site, per-joint qpos/qvel
addresses) as keyword args — the `Controller` (see `backbone.py`)
performs the name lookups once at construction and passes them in,
matching how `SimEnv` caches indices in M1.
"""

from __future__ import annotations

import mujoco
import numpy as np


def impedance_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    K_diag_tcp: np.ndarray,
    D_diag_tcp: np.ndarray,
    q_nominal: np.ndarray,
    posture_gain: float,
    tcp_site_id: int,
    arm_qpos_adr: np.ndarray,
    arm_dof_adr: np.ndarray,
    dls_damping: float = 0.05,
    joint_damping: float = 0.0,
) -> np.ndarray:
    """Return a 7-vector arm torque command for one control tick.

    Parameters
    ----------
    model, data
        Live MuJoCo model and data. Read-only here — no `mj_step`.
    target_pos, target_quat
        Commanded EE pose in the world frame. Quaternion is (w, x, y, z).
    K_diag_tcp, D_diag_tcp : (6,)
        Diagonal stiffness/damping in the TCP frame, ordered
        [Kx, Ky, Kz, Krx, Kry, Krz] (translation N/m, rotation N·m/rad;
        damping in N·s/m and N·m·s/rad correspondingly). The diagonal
        layout is what enables direction-dependent compliance: e.g.
        Kz = 800 (stiff insertion), Kx = Ky = 200 (soft laterally),
        Krx = Kry = 5 (soft pitch/roll).
    q_nominal : (7,)
        Posture toward which the null-space projector biases joint
        positions. Defaults to the M1 home pose at the `Controller` layer.
    posture_gain : float
        Scalar `k_posture` on `(q_nominal − q)` in joint space.
    tcp_site_id : int
        MuJoCo site id of the TCP frame the impedance law is anchored to.
    arm_qpos_adr, arm_dof_adr : (7,)
        Cached `qpos` / `qvel` addresses for the seven arm hinge joints.
    dls_damping : float
        Regularisation `λ` for the DLS pseudoinverse used to build the
        null-space projector `N = I − J⁺ J`.
    joint_damping : float
        Optional flat joint-space velocity damping `−kd · qdot`. Stabilises
        slow null-space modes the Cartesian D-term can't see. Recommended
        when configurations vary widely (the Panda's reflected inertia at
        the TCP swings by ~3× across the workspace, so a single critically
        damped Cartesian D undershoots somewhere).

    Returns
    -------
    τ : (7,)
        Joint torque command for the arm actuators.
    """
    ee_pos = data.site_xpos[tcp_site_id].copy()
    site_xmat = np.ascontiguousarray(data.site_xmat[tcp_site_id]).reshape(3, 3)
    ee_quat = np.zeros(4)
    mujoco.mju_mat2Quat(ee_quat, site_xmat.ravel())

    # World-frame geometric Jacobian at the TCP site, sliced to arm DoFs.
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, tcp_site_id)
    J = np.vstack([jacp[:, arm_dof_adr], jacr[:, arm_dof_adr]])  # (6, 7)

    q_arm = data.qpos[arm_qpos_adr]
    qdot_arm = data.qvel[arm_dof_adr]

    pos_err_world = target_pos - ee_pos
    # CAREFUL: MuJoCo's `mju_subQuat(res, qa, qb)` returns res such that
    # `qa = qb * quat(res)` (right-multiplication), so `res` is the
    # axis-angle expressed in qb's **body** frame, not world. The TCP
    # frame *is* the body frame for the EE, so the result is already in
    # the TCP frame — no R.T conversion needed.
    rot_err_tcp = np.zeros(3)
    mujoco.mju_subQuat(rot_err_tcp, target_quat, ee_quat)

    twist_world = J @ qdot_arm

    # Rotate world → TCP frame for the position spring. R columns are TCP
    # basis vectors expressed in world, so R.T maps world vectors into the
    # TCP frame. The angular twist comes from `jacr` (which produces
    # world-frame ω) so it also has to be rotated; the rotational error
    # is already in TCP frame from mju_subQuat and goes through unchanged.
    R = site_xmat
    e_tcp = np.concatenate([R.T @ pos_err_world, rot_err_tcp])
    twist_tcp = np.concatenate([R.T @ twist_world[:3], R.T @ twist_world[3:]])

    F_tcp = K_diag_tcp * e_tcp - D_diag_tcp * twist_tcp
    F_world = np.concatenate([R @ F_tcp[:3], R @ F_tcp[3:]])

    tau_task = J.T @ F_world

    # Null-space posture term — DLS pseudoinverse so the projector stays
    # well-conditioned near singularities.
    JJT_reg = J @ J.T + (dls_damping**2) * np.eye(6)
    j_pinv = J.T @ np.linalg.solve(JJT_reg, np.eye(6))
    null_space = np.eye(7) - j_pinv @ J
    tau_posture = null_space @ (posture_gain * (q_nominal - q_arm))

    tau_bias = data.qfrc_bias[arm_dof_adr]
    tau_joint_damp = -joint_damping * qdot_arm

    return tau_task + tau_posture + tau_bias + tau_joint_damp
