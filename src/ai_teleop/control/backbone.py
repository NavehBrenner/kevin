"""`Controller` — the M2 backbone, wired against a `SimEnv`.

Pipeline per control tick (`compute(obs, command)`):

1. Clamp the incoming command relative to the current EE pose
   (≤ 2.5 cm position, ≤ 10° rotation, ≤ 5 N grip-force step).
2. Hand the clamped target to the lock state machine, which may
   override it with a hold pose or the home pose, and may trip the
   force-cap watchdog on the way through.
3. Run the direction-dependent Cartesian impedance law against the
   effective target, producing seven joint torques + gravity comp.
4. Write the torques to `data.ctrl[0:7]` (arm) and a gripper command
   to `data.ctrl[7]`. **Does not step the sim** — caller invokes
   `env.step()`.

Constructor caches every MuJoCo handle the hot loop needs (TCP site id,
arm qpos/qvel addresses, actuator indices, home keyframe id), and reads
the home EE pose via a snapshot/restore round-trip on `mj_resetDataKeyframe`
so the controller doesn't depend on the caller having just reset.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ai_teleop.common.command import Command
from ai_teleop.common.observation import Observation
from ai_teleop.control.impedance import impedance_torque
from ai_teleop.control.lock import LockController, LockStatus

# Names must match the M1 scene contract (see ai_teleop.sim.scene).
_ARM_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
)
_ARM_ACTUATOR_NAMES = (
    "actuator1",
    "actuator2",
    "actuator3",
    "actuator4",
    "actuator5",
    "actuator6",
    "actuator7",
)
_GRIPPER_ACTUATOR_NAME = "actuator8"
_TCP_SITE_NAME = "tcp_site"
_HOME_KEYFRAME = "home"

# Stiffness in the TCP frame, ordered [Kx, Ky, Kz, Krx, Kry, Krz].
# z is the gripper-outward axis (the insertion axis for our peg, see
# project-wiki/entities/franka-panda.md "Hand frame conventions"). K_z >
# K_lateral encodes the "stiff along insertion, soft laterally" design
# contract from project-scope.md *Compliance / contact behavior*. K_z is
# capped so that a 5 cm "intrusion" command into a flat wall produces a
# contact force well below the 30 N watchdog (K_z · 0.05 = 25 N).
_DEFAULT_STIFFNESS_TCP = np.array([400.0, 400.0, 500.0, 3.0, 3.0, 3.0])

# Critically-ish damped against a nominal M_trans ≈ 4 kg, I_rot ≈ 0.3 kg·m².
# The "shape" lives here; per-axis numbers tuned in M2 Step 8. The rotational
# gains are deliberately small — see project-wiki entry for the rationale.
# Too-stiff orientation coupling fights the position dynamics and drives a
# slow limit cycle (observed during tuning: pos err grew to >50 mm at K_rot=25
# even with critically-damped D_rot; dropping K_rot to 3 collapsed it to <1 mm).
_DEFAULT_DAMPING_TCP = np.array(
    [
        2 * np.sqrt(4.0 * 400.0),  # x  ≈ 80
        2 * np.sqrt(4.0 * 400.0),  # y  ≈ 80
        2 * np.sqrt(4.0 * 500.0),  # z  ≈ 89
        4.0,  # rx — slightly overdamped vs nominal
        4.0,  # ry
        4.0,  # rz
    ]
)

_DEFAULT_POSTURE_GAIN = 1.0
_DEFAULT_DLS_DAMPING = 0.05
# Joint-space velocity damping. The Panda's reflected mass at the TCP
# varies by ~3× across the workspace, so the Cartesian D term (tuned for
# a nominal mass) can leave slow joint modes undamped. A flat joint-space
# kd absorbs that without re-tuning per configuration. Value chosen from
# scripts/dev/sweep_joint_damping.py: kd=4 gives hold drift ~1.1 mm at the
# cost of bounding free-space slew to ~0.05 m/s. Higher kd would tighten
# hold further but stalls long-distance approach.
_DEFAULT_JOINT_DAMPING = 4.0

_DEFAULT_MAX_DPOS = 0.025  # m / control step (approach-speed / strictness knob)
_DEFAULT_MAX_DROT = np.deg2rad(10.0)  # rad / control step
_DEFAULT_MAX_DGRIP_FORCE = 5.0  # N  / control step

_GRIPPER_BASELINE_CTRL = 0.0  # squeezing/closed; see panda.xml comment
_GRIPPER_FORCE_TO_CTRL = 255.0 / 4.0  # invert the panda.xml gain=4/255 mapping


def _name2id(model: mujoco.MjModel, objtype: int, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, objtype, name)
    if obj_id == -1:
        raise KeyError(f"MJCF object not found: type={objtype} name={name!r}")
    return obj_id


class Controller:
    """Backbone controller — operational-space impedance + lock state machine."""

    def __init__(
        self,
        env,
        *,
        force_cap_n: float = 30.0,
        stiffness_tcp: np.ndarray | None = None,
        damping_tcp: np.ndarray | None = None,
        posture_gain: float = _DEFAULT_POSTURE_GAIN,
        dls_damping: float = _DEFAULT_DLS_DAMPING,
        joint_damping: float = _DEFAULT_JOINT_DAMPING,
        max_dpos_per_step: float = _DEFAULT_MAX_DPOS,
        max_drot_per_step: float = _DEFAULT_MAX_DROT,
        max_dgrip_force_per_step: float = _DEFAULT_MAX_DGRIP_FORCE,
        home_pose: np.ndarray | None = None,
    ) -> None:
        self._env = env
        m: mujoco.MjModel = env.model
        d: mujoco.MjData = env.data
        self._model = m
        self._data = d

        # MuJoCo handle cache.
        self._arm_qpos_adr = np.array(
            [m.jnt_qposadr[_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in _ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self._arm_dof_adr = np.array(
            [m.jnt_dofadr[_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in _ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self._arm_act_idx = np.array(
            [_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in _ARM_ACTUATOR_NAMES],
            dtype=np.int32,
        )
        self._gripper_act_idx = _name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, _GRIPPER_ACTUATOR_NAME)
        self._tcp_site_id = _name2id(m, mujoco.mjtObj.mjOBJ_SITE, _TCP_SITE_NAME)
        self._home_kid = _name2id(m, mujoco.mjtObj.mjOBJ_KEY, _HOME_KEYFRAME)

        self.force_cap_n = float(force_cap_n)
        self.stiffness_tcp = np.asarray(
            stiffness_tcp if stiffness_tcp is not None else _DEFAULT_STIFFNESS_TCP,
            dtype=np.float64,
        )
        self.damping_tcp = np.asarray(
            damping_tcp if damping_tcp is not None else _DEFAULT_DAMPING_TCP,
            dtype=np.float64,
        )
        if self.stiffness_tcp.shape != (6,) or self.damping_tcp.shape != (6,):
            raise ValueError("stiffness_tcp and damping_tcp must be shape (6,)")
        self.posture_gain = float(posture_gain)
        self.dls_damping = float(dls_damping)
        self.joint_damping = float(joint_damping)

        self.max_dpos_per_step = float(max_dpos_per_step)
        self.max_drot_per_step = float(max_drot_per_step)
        self.max_dgrip_force_per_step = float(max_dgrip_force_per_step)

        # Discover home EE pose + nominal posture via snapshot/restore.
        if home_pose is None:
            home_pose, q_home = self._read_home_state()
        else:
            home_pose = np.asarray(home_pose, dtype=np.float64).copy()
            if home_pose.shape != (7,):
                raise ValueError(f"home_pose must be (7,), got {home_pose.shape}")
            # We still need a nominal joint posture; do the snapshot read for q.
            _, q_home = self._read_home_state()
        self.home_pose = home_pose
        self.q_nominal = q_home

        self._lock = LockController(home_pose=self.home_pose, force_cap_n=self.force_cap_n)

    # ------------------------------------------------------------------
    def _read_home_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Snapshot-restore round-trip to read the home EE pose and arm q."""
        m, d = self._model, self._data
        qpos_save = d.qpos.copy()
        qvel_save = d.qvel.copy()
        ctrl_save = d.ctrl.copy()
        time_save = float(d.time)
        try:
            mujoco.mj_resetDataKeyframe(m, d, self._home_kid)
            mujoco.mj_forward(m, d)
            ee_pos = d.site_xpos[self._tcp_site_id].copy()
            ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat, d.site_xmat[self._tcp_site_id].ravel())
            home_pose = np.concatenate([ee_pos, ee_quat])
            q_home = d.qpos[self._arm_qpos_adr].copy()
        finally:
            d.qpos[:] = qpos_save
            d.qvel[:] = qvel_save
            d.ctrl[:] = ctrl_save
            d.time = time_save
            mujoco.mj_forward(m, d)
        return home_pose, q_home

    # ------------------------------------------------------------------
    # Lock state API.
    # ------------------------------------------------------------------
    def request_hold_lock(self) -> None:
        self._lock.request_hold_lock(float(self._data.time))

    def request_park_lock(self) -> None:
        self._lock.request_park_lock(float(self._data.time))

    def release_lock(self) -> None:
        self._lock.release_lock(float(self._data.time))

    def reset(self) -> None:
        """Clear transient control state for a new episode.

        Releases any latched lock back to ACTIVE. The impedance law itself is
        stateless, so the lock is the only thing that carries over — and it
        MUST be cleared when a `Controller` is reused across episodes (e.g. the
        data-gen loop), or one episode's force-cap → HOLD trip silently freezes
        every episode after it.
        """
        self._lock.release_lock(float(self._data.time))

    @property
    def status(self) -> LockStatus:
        return self._lock.status

    # ------------------------------------------------------------------
    # Hot loop.
    # ------------------------------------------------------------------
    def compute(self, obs: Observation, command: Command) -> None:
        """Resolve lock + clamp + impedance and write `data.ctrl`. No `mj_step`."""
        clamped_pos, clamped_quat = self._clamp_command(command, obs.ee_pose)
        target_pos, target_quat = self._lock.resolve_target(obs, clamped_pos, clamped_quat)

        tau = impedance_torque(
            self._model,
            self._data,
            target_pos=target_pos,
            target_quat=target_quat,
            K_diag_tcp=self.stiffness_tcp,
            D_diag_tcp=self.damping_tcp,
            q_nominal=self.q_nominal,
            posture_gain=self.posture_gain,
            tcp_site_id=self._tcp_site_id,
            arm_qpos_adr=self._arm_qpos_adr,
            arm_dof_adr=self._arm_dof_adr,
            dls_damping=self.dls_damping,
            joint_damping=self.joint_damping,
        )
        self._data.ctrl[self._arm_act_idx] = tau

        dgrip = float(
            np.clip(
                command.delta_grip_force,
                -self.max_dgrip_force_per_step,
                self.max_dgrip_force_per_step,
            )
        )
        self._data.ctrl[self._gripper_act_idx] = (
            _GRIPPER_BASELINE_CTRL + dgrip * _GRIPPER_FORCE_TO_CTRL
        )

    # ------------------------------------------------------------------
    def _clamp_command(
        self, command: Command, ee_pose: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        current_pos = ee_pose[:3]
        current_quat = ee_pose[3:]

        delta_pos = command.target_position - current_pos
        norm = float(np.linalg.norm(delta_pos))
        if norm > self.max_dpos_per_step:
            delta_pos = delta_pos * (self.max_dpos_per_step / norm)
        clamped_pos = current_pos + delta_pos

        rot_err = np.zeros(3)
        mujoco.mju_subQuat(rot_err, command.target_quaternion, current_quat)
        angle = float(np.linalg.norm(rot_err))
        if angle > self.max_drot_per_step:
            rot_err = rot_err * (self.max_drot_per_step / angle)
        clamped_quat = current_quat.copy()
        mujoco.mju_quatIntegrate(clamped_quat, rot_err, 1.0)
        mujoco.mju_normalize4(clamped_quat)
        return clamped_pos, clamped_quat
