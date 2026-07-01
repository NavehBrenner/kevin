"""SimEnv — thin object wrapper around a MuJoCo scene.

Scope: load the scene, reset to a clean base state, step the physics, read
sensors, render the wrist camera, optionally open an interactive viewer. The
env owns physics + sensing only — it is built on one wall (recorded in its
`EnvConfig`) and `reset()` always returns it to the same t=0 state. Which hole
is the *goal* is a task concept the caller owns, not the env's. Per-episode
variation comes from building a different env (a different wall), not from a
reset argument. See `docs/milestone-1-spec.md`.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Literal

import mujoco
import mujoco.viewer
import numpy as np

from ai_teleop.common.geometry import mat3_to_quat
from ai_teleop.common.observation import Observation
from ai_teleop.sim.config import EnvConfig

RenderMode = Literal["viewer", "headless"]

# The passive viewer only needs ~30-60 Hz, but step() runs at the 500 Hz sim rate.
# sync_viewer() rate-limits to this fps off a wall-clock deadline — syncing every step
# saturates WSLg's GUI pipe (window freezes, then snaps back).
_VIEWER_FPS = 50.0

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
# Hole sites are discovered by name at load time — any number of `hole_<i>`
# sites — so procedurally generated walls with arbitrary hole counts load
# unchanged. The hand-written full_scene.xml exposes hole_0/1/2; generated
# walls put the target at hole_0.
_HOLE_SITE_PATTERN = re.compile(r"^hole_(\d+)$")
_TCP_SITE_NAME = "tcp_site"
_PEG_JOINT_NAME = "peg_joint"
_FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
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


def _discover_hole_site_ids(model: mujoco.MjModel) -> np.ndarray:
    """Return site IDs for every `hole_<i>` site, ordered by index `i`.

    Lets one SimEnv serve the fixed full_scene wall and any generated wall
    without knowing the hole count ahead of time.
    """
    found: list[tuple[int, int]] = []
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        match = _HOLE_SITE_PATTERN.match(name) if name else None
        if match:
            found.append((int(match.group(1)), site_id))
    if not found:
        raise KeyError("scene contains no `hole_<i>` sites")
    found.sort()
    return np.array([site_id for _, site_id in found], dtype=np.int32)


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
        config: EnvConfig | None = None,
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

        # The env's metadata. The wall it was built on is already baked into the
        # model above; the config records it (and any future per-env knobs) so the
        # env can describe itself. The target hole is *not* here — that is a
        # task-layer concept (see `EnvConfig`).
        self._config = config if config is not None else EnvConfig()

        # Cache MuJoCo IDs once.
        model = self._model
        self._hole_site_ids = _discover_hole_site_ids(model)

        self._arm_joint_qadr = np.array(
            [
                model.jnt_qposadr[_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in _ARM_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self._arm_joint_vadr = np.array(
            [
                model.jnt_dofadr[_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in _ARM_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self._peg_qadr = model.jnt_qposadr[
            _name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _PEG_JOINT_NAME)
        ]
        self._finger_qadr = np.array(
            [
                model.jnt_qposadr[_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in _FINGER_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self._tcp_site_id = _name2id(model, mujoco.mjtObj.mjOBJ_SITE, _TCP_SITE_NAME)
        self._force_sensor_adr = model.sensor_adr[
            _name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, _WRIST_FORCE_SENSOR)
        ]
        self._torque_sensor_adr = model.sensor_adr[
            _name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, _WRIST_TORQUE_SENSOR)
        ]
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
        self._next_frame_time = 0.0  # wall-clock deadline for the next viewer sync

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
    def config(self) -> EnvConfig:
        """The metadata this env was built from (e.g. its wall seed)."""
        return self._config

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
        """Restore the env to its base state: arm at canonical config, peg pre-grasped.

        Reads from the `home` keyframe in `full_scene.xml`, which encodes both the
        panda joint angles and the peg's free-joint pose (computed offline so the
        weld constraint is already satisfied at t=0). A pure restore — it takes no
        arguments and mutates no config: the same env always returns to the same
        t=0 state. Per-episode variation comes from building a *different* env (a
        different `EnvConfig`/wall), not from an argument here.
        """
        mujoco.mj_resetDataKeyframe(self._model, self._data, self._home_keyframe_id)
        mujoco.mj_forward(self._model, self._data)
        self.sync_viewer()  # show the reset pose immediately (no-op when headless)
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

        Pure physics — no GUI. Refresh the viewer separately via `sync_viewer()`
        (the caller decides cadence; it self-throttles to ~50 Hz).
        """
        mujoco.mj_step(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

    # ------------------------------------------------------------------
    # Sensing.
    # ------------------------------------------------------------------
    def get_observation(self) -> Observation:
        data = self._data

        joint_positions = data.qpos[self._arm_joint_qadr].copy()
        joint_velocities = data.qvel[self._arm_joint_vadr].copy()

        tcp_pos = data.site_xpos[self._tcp_site_id].copy()
        tcp_quat = mat3_to_quat(data.site_xmat[self._tcp_site_id])
        ee_pose = np.concatenate([tcp_pos, tcp_quat])

        peg_pose = data.qpos[self._peg_qadr : self._peg_qadr + 7].copy()
        gripper_width = float(data.qpos[self._finger_qadr].sum())

        hole_poses = np.zeros((len(self._hole_site_ids), 7))
        for i, site_id in enumerate(self._hole_site_ids):
            hole_poses[i, 0:3] = data.site_xpos[site_id]
            hole_poses[i, 3:7] = mat3_to_quat(data.site_xmat[site_id])

        wrist_ft = np.concatenate([
            data.sensordata[self._force_sensor_adr : self._force_sensor_adr + 3],
            data.sensordata[self._torque_sensor_adr : self._torque_sensor_adr + 3],
        ]).copy()

        return Observation(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            ee_pose=ee_pose,
            wrist_ft=wrist_ft,
            gripper_width=gripper_width,
            peg_pose=peg_pose,
            hole_poses=hole_poses,
            sim_time=float(data.time),
        )

    # ------------------------------------------------------------------
    # Rendering.
    # ------------------------------------------------------------------
    def render_wrist_camera(self) -> np.ndarray:
        """Return a (H, W, 3) uint8 RGB frame from the wrist camera."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self._model, height=self._camera_height, width=self._camera_width
            )
        self._renderer.update_scene(self._data, camera=self._wrist_camera_id)
        return self._renderer.render()

    def launch_viewer(self, *, wrist_cam: bool = False) -> None:
        """Open the interactive passive viewer (no-op if already open).

        Only valid when render_mode='viewer'. Done lazily so SimEnv can be
        constructed in viewer mode without immediately blocking on a window
        being available. With ``wrist_cam=True`` the view starts locked to the
        Panda's wrist camera (the robot's-eye POV) instead of the free camera;
        the usual viewer keys still switch cameras live.
        """
        if self._render_mode != "viewer":
            raise RuntimeError(
                f"launch_viewer() requires render_mode='viewer', got {self._render_mode!r}"
            )
        if self._viewer is not None:
            return None
        # Imported here so headless tests don't load GLFW.
        import mujoco.viewer as mjv  # noqa: PLC0415

        viewer = mjv.launch_passive(self._model, self._data)
        if wrist_cam:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = self._wrist_camera_id
        self._viewer = viewer

    def highlight_target(
        self,
        position: np.ndarray,
        *,
        radius: float = 0.01,
        rgba: tuple[float, float, float, float] = (1.0, 0.85, 0.0, 0.2),
    ) -> None:
        """Draw a translucent marker at ``position`` in the interactive viewer only.

        Added to the viewer's ``user_scn`` (its user-owned decoration scene), which
        the passive viewer renders but the offscreen wrist-camera ``Renderer`` never
        touches — so the human sees which hole to aim at while the images fed to the
        policy (``render_wrist_camera``) stay unmarked. No-op without a viewer.
        """
        if self._viewer is None:
            return
        scene = self._viewer.user_scn
        geom_index = scene.ngeom
        mujoco.mjv_initGeom(
            scene.geoms[geom_index],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0.0, 0.0]),
            pos=np.asarray(position, dtype=np.float64),
            mat=np.eye(3).flatten(),
            rgba=np.array(rgba, dtype=np.float32),
        )
        scene.ngeom = geom_index + 1

    def sync_viewer(self) -> None:
        """Push physics state to the viewer window, rate-limited to ~_VIEWER_FPS.

        No-op when no viewer is open. Safe (and intended) to call every step from the
        stepping thread — it self-throttles off a wall-clock frame deadline, so callers
        don't track frame timing themselves and syncing never floods WSLg's GUI pipe.
        Must run on the same thread that steps physics (MuJoCo serializes mj_step and
        mj_copyDataVisual via the MjData stack — concurrent calls error).
        """
        if self._viewer is None:
            return
        now = time.monotonic()
        if now >= self._next_frame_time:
            self._viewer.sync()
            self._next_frame_time = now + 1.0 / _VIEWER_FPS

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
