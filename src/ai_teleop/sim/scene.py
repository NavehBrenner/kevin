"""SimEnv — thin object wrapper around a MuJoCo scene.

Milestone 1 scope: load the scene, reset to a clean state, step the physics,
read sensors, render the wrist camera, optionally open an interactive viewer.
No controller, no command input, no randomisation beyond a seed plumbed
through for future use. Subsequent milestones add command input on top of
this contract — see `docs/milestone-1-spec.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

from ai_teleop.common.observation import Observation

RenderMode = Literal["viewer", "headless"]

# Hardcoded for M1 — the scene XML names these explicitly. Anything reading
# them by string lives here so a rename in the MJCF is a one-place fix.
_ARM_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
)
_HOLE_SITE_NAMES = ("hole_0", "hole_1", "hole_2")
_TCP_SITE_NAME = "tcp_site"
_PEG_JOINT_NAME = "peg_joint"
_WRIST_FORCE_SENSOR = "wrist_force"
_WRIST_TORQUE_SENSOR = "wrist_torque"
_WRIST_CAMERA_NAME = "wrist_cam"
_HOME_KEYFRAME = "home"


def _name2id(model: mujoco.MjModel, objtype: int, name: str) -> int:
    """Lookup wrapper that fails loudly when the name doesn't exist in the scene."""
    obj_id = mujoco.mj_name2id(model, objtype, name)
    if obj_id == -1:
        raise KeyError(f"MJCF object not found: type={objtype} name={name!r}")
    return obj_id


def _mat3_to_quat(mat_flat: np.ndarray) -> np.ndarray:
    """Convert a flattened 3x3 row-major rotation matrix to a (w,x,y,z) unit quat.

    `data.site_xmat` and `data.xmat` are stored as 9-element row-major flat arrays.
    MuJoCo's mju_mat2Quat expects exactly this layout.
    """
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(mat_flat).reshape(9))
    return quat


class SimEnv:
    """A MuJoCo scene with a clean reset / step / observation / render API.

    Lifecycle:
        env = SimEnv("assets/mjcf/full_scene.xml", render_mode="headless")
        obs = env.reset()
        for _ in range(N):
            env.step()
            obs = env.get_observation()
        frame = env.render_wrist_camera()      # (H, W, 3) uint8
        env.close()
    """

    def __init__(
        self,
        scene_path: str | Path,
        render_mode: RenderMode = "headless",
        *,
        camera_height: int = 128,
        camera_width: int = 128,
        target_hole_index: int = 1,
        seed: int = 0,
    ) -> None:
        # MuJoCo resolves mesh paths relative to the cwd, not the XML file —
        # using an absolute path here avoids surprises when tests / scripts
        # are launched from different working directories.
        abs_path = str(Path(scene_path).expanduser().resolve(strict=True))
        self._model: mujoco.MjModel = mujoco.MjModel.from_xml_path(abs_path)
        self._data: mujoco.MjData = mujoco.MjData(self._model)

        if render_mode not in ("viewer", "headless"):
            raise ValueError(f"render_mode must be 'viewer' or 'headless', got {render_mode!r}")
        self._render_mode: RenderMode = render_mode

        self._rng = np.random.default_rng(seed)
        self._seed = seed

        n_holes = len(_HOLE_SITE_NAMES)
        if not (0 <= target_hole_index < n_holes):
            raise ValueError(f"target_hole_index must be in [0, {n_holes}), got {target_hole_index}")
        self._target_hole_index = target_hole_index

        # Cache MuJoCo IDs once.
        model = self._model
        self._arm_joint_qadr = np.array(
            [model.jnt_qposadr[_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in _ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self._arm_joint_vadr = np.array(
            [model.jnt_dofadr[_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in _ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self._peg_qadr = model.jnt_qposadr[_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _PEG_JOINT_NAME)]
        self._tcp_site_id = _name2id(model, mujoco.mjtObj.mjOBJ_SITE, _TCP_SITE_NAME)
        self._hole_site_ids = np.array(
            [_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in _HOLE_SITE_NAMES],
            dtype=np.int32,
        )
        self._force_sensor_adr = model.sensor_adr[_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, _WRIST_FORCE_SENSOR)]
        self._torque_sensor_adr = model.sensor_adr[_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, _WRIST_TORQUE_SENSOR)]
        self._home_keyframe_id = _name2id(model, mujoco.mjtObj.mjOBJ_KEY, _HOME_KEYFRAME)
        self._wrist_camera_id = _name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, _WRIST_CAMERA_NAME)

        # Renderer for offscreen wrist-camera frames. Created lazily on first
        # render so headless smoke tests on machines without GL fail late
        # rather than at construction.
        self._camera_height = camera_height
        self._camera_width = camera_width
        self._renderer: mujoco.Renderer | None = None

        # Interactive viewer handle, only populated when render_mode="viewer".
        # mujoco.viewer is imported lazily — it pulls in GLFW which we don't
        # want loaded during headless runs.
        self._viewer = None

    # ------------------------------------------------------------------
    # Read-only access for callers (controller, smoke test, expert).
    # ------------------------------------------------------------------
    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def render_mode(self) -> RenderMode:
        return self._render_mode

    @property
    def target_hole_index(self) -> int:
        return self._target_hole_index

    @property
    def viewer(self):
        """The passive viewer handle when render_mode='viewer', else None.

        Caller is responsible for calling `.sync()` between physics steps if
        they want the window to update.
        """
        return self._viewer

    # ------------------------------------------------------------------
    # Episode lifecycle.
    # ------------------------------------------------------------------
    def reset(self) -> Observation:
        """Reset to the home pose: arm at canonical config, peg pre-grasped.

        Reads from the `home` keyframe in `full_scene.xml`, which encodes
        both the panda joint angles and the peg's free-joint pose (computed
        offline so the weld constraint is already satisfied at t=0).
        """
        mujoco.mj_resetDataKeyframe(self._model, self._data, self._home_keyframe_id)
        mujoco.mj_forward(self._model, self._data)
        if self._viewer is not None:
            self._viewer.sync()
        return self.get_observation()

    def step(self) -> None:
        """Advance physics by one timestep.

        Follows the standard MuJoCo split (`mj_step1` → caller writes ctrl
        externally → `mj_step2` advances) collapsed into one call: we use
        `mj_step` then `mj_forward` so that sensors, Jacobians, and
        `qfrc_bias` reflect the **new** state when the next control tick
        reads them. Without the trailing forward pass those derived
        quantities lag by one step, which adds a 2 ms control-loop delay
        and destabilises tightly-tuned impedance gains.
        """
        mujoco.mj_step(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        if self._viewer is not None:
            self._viewer.sync()

    # ------------------------------------------------------------------
    # Sensing.
    # ------------------------------------------------------------------
    def get_observation(self) -> Observation:
        model, data = self._model, self._data

        joint_positions = data.qpos[self._arm_joint_qadr].copy()
        joint_velocities = data.qvel[self._arm_joint_vadr].copy()

        tcp_pos = data.site_xpos[self._tcp_site_id].copy()
        tcp_quat = _mat3_to_quat(data.site_xmat[self._tcp_site_id])
        ee_pose = np.concatenate([tcp_pos, tcp_quat])

        peg_pose = data.qpos[self._peg_qadr : self._peg_qadr + 7].copy()

        hole_poses = np.zeros((len(self._hole_site_ids), 7))
        for i, site_id in enumerate(self._hole_site_ids):
            hole_poses[i, 0:3] = data.site_xpos[site_id]
            hole_poses[i, 3:7] = _mat3_to_quat(data.site_xmat[site_id])

        wrist_ft = np.concatenate(
            [
                data.sensordata[self._force_sensor_adr : self._force_sensor_adr + 3],
                data.sensordata[self._torque_sensor_adr : self._torque_sensor_adr + 3],
            ]
        ).copy()

        return Observation(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            ee_pose=ee_pose,
            wrist_ft=wrist_ft,
            peg_pose=peg_pose,
            hole_poses=hole_poses,
            target_hole_index=self._target_hole_index,
            sim_time=float(data.time),
        )

    # ------------------------------------------------------------------
    # Rendering.
    # ------------------------------------------------------------------
    def render_wrist_camera(self) -> np.ndarray:
        """Return a (H, W, 3) uint8 RGB frame from the wrist camera."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=self._camera_height, width=self._camera_width)
        self._renderer.update_scene(self._data, camera=self._wrist_camera_id)
        return self._renderer.render()

    def launch_viewer(self) -> None:
        """Open the interactive passive viewer (no-op if already open).

        Only valid when render_mode='viewer'. Done lazily so SimEnv can be
        constructed in viewer mode without immediately blocking on a window
        being available.
        """
        if self._render_mode != "viewer":
            raise RuntimeError(f"launch_viewer() requires render_mode='viewer', got {self._render_mode!r}")
        if self._viewer is not None:
            return
        # Imported here so headless tests don't load GLFW.
        import mujoco.viewer as mjv  # noqa: PLC0415

        self._viewer = mjv.launch_passive(self._model, self._data)

    # ------------------------------------------------------------------
    # Teardown.
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
