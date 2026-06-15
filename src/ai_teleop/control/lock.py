"""Lock state machine and force-cap watchdog for the backbone controller.

Three runtime states (per `project-scope.md` *Runtime state — two modes only*
plus the M2-spec park variant):

==========  ==========================================  ====================================
State        What it does                                Exits to
==========  ==========================================  ====================================
Active       The external `Command` reaches impedance.   HoldLock (force-cap or request),
                                                          ParkLock (request)
HoldLock     Override target = pose latched at trip.     Active (release)
ParkLock     Override target = home pose; once within    HoldLock (auto, on home reached),
              tolerance, auto-transition to HoldLock.    Active (release)
==========  ==========================================  ====================================

`LockController` is a pure-Python state machine — no MuJoCo handles. The
`Controller` (in `backbone.py`) calls `step()` once per control tick with
the current observation; the lock looks at the wrist force, considers any
pending requests, and returns an *effective target pose* the impedance
law should track this tick.

`LockStatus` is the read-only struct exposed to the (future) eval
harness — see `docs/milestone-2-spec.md` *Lock state machine*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import mujoco
import numpy as np

from ai_teleop.common.observation import Observation


class LockState(Enum):
    ACTIVE = "active"
    HOLD = "hold_lock"
    PARK = "park_lock"


@dataclass(frozen=True)
class LockStatus:
    """Snapshot of the lock controller for external readers.

    The fields are kept narrow on purpose — the eval harness should be
    able to log this without depending on the controller's internals.
    """

    state: LockState
    last_transition_reason: str
    last_transition_sim_time: float


@dataclass
class LockController:
    """Stateful lock manager.

    Parameters
    ----------
    home_pose : (7,)
        Park target — `(px, py, pz, qw, qx, qy, qz)` in world frame.
        Latched at construction; the controller passes it in from
        `SimEnv`'s home keyframe.
    force_cap_n : float
        Magnitude threshold on the wrist force vector. Crossing it
        triggers an automatic ACTIVE → HOLD transition.
    park_pos_tol_m : float
        EE position tolerance for the PARK → HOLD auto-transition.
    park_quat_tol_rad : float
        EE orientation tolerance (axis-angle magnitude) for the
        PARK → HOLD auto-transition.
    """

    home_pose: np.ndarray
    force_cap_n: float = 30.0
    park_pos_tol_m: float = 5e-3
    # Orientation impedance is kept soft (see backbone.py defaults). With
    # K_rot=3 the system can't fully unwind a 25° contact-induced rotation
    # within a sensible park timeout, so we accept "close enough" for the
    # park lock — the next active command will refine orientation. The
    # tighter the spec needs, the higher K_rot must go (and the more the
    # position dynamics get hit, see the soft-rot rationale in backbone.py).
    park_quat_tol_rad: float = np.deg2rad(10.0)

    _state: LockState = field(default=LockState.ACTIVE, init=False)
    _reason: str = field(default="init", init=False)
    _t_transition: float = field(default=0.0, init=False)
    _hold_target: np.ndarray | None = field(default=None, init=False)

    # ------------------------------------------------------------------
    # External requests (called from the `Controller` facade).
    # ------------------------------------------------------------------
    def request_hold_lock(self, sim_time: float, reason: str = "user_request") -> None:
        # No pose available at request time — `resolve_target` latches the
        # current EE pose on the next tick (see fallback below).
        self._state = LockState.HOLD
        self._reason = reason
        self._t_transition = sim_time
        self._hold_target = None

    def request_park_lock(self, sim_time: float) -> None:
        self._state = LockState.PARK
        self._reason = "user_request"
        self._t_transition = sim_time
        self._hold_target = None

    def release_lock(self, sim_time: float) -> None:
        if self._state == LockState.ACTIVE:
            return
        self._state = LockState.ACTIVE
        self._reason = "released"
        self._t_transition = sim_time
        self._hold_target = None

    # ------------------------------------------------------------------
    # Per-tick update from `Controller.compute`.
    # ------------------------------------------------------------------
    def resolve_target(
        self,
        obs: Observation,
        external_target_pos: np.ndarray,
        external_target_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run watchdog + transitions and return (target_pos, target_quat).

        Called once per control tick. Mutates internal state.
        """
        force_mag = float(np.linalg.norm(obs.wrist_ft[:3]))
        if self._state == LockState.ACTIVE and force_mag > self.force_cap_n:
            self._transition_to_hold(
                obs.sim_time, f"force_cap_trip_|F|={force_mag:.2f}N", current_pose=obs.ee_pose
            )

        if self._state == LockState.PARK:
            pos_err = float(np.linalg.norm(obs.ee_pose[:3] - self.home_pose[:3]))
            rot_err_axis = np.zeros(3)
            # Same quaternion-difference convention the impedance law uses.
            mujoco.mju_subQuat(rot_err_axis, self.home_pose[3:], obs.ee_pose[3:])
            rot_err = float(np.linalg.norm(rot_err_axis))
            if pos_err < self.park_pos_tol_m and rot_err < self.park_quat_tol_rad:
                self._transition_to_hold(obs.sim_time, "park_complete", current_pose=obs.ee_pose)

        if self._state == LockState.ACTIVE:
            return external_target_pos, external_target_quat
        if self._state == LockState.HOLD:
            # Latch the current pose on the first tick after the transition
            # — `request_hold_lock()` has no pose available at request time.
            if self._hold_target is None:
                self._hold_target = obs.ee_pose.copy()
            return self._hold_target[:3].copy(), self._hold_target[3:].copy()
        # PARK
        return self.home_pose[:3].copy(), self.home_pose[3:].copy()

    # ------------------------------------------------------------------
    @property
    def status(self) -> LockStatus:
        return LockStatus(
            state=self._state,
            last_transition_reason=self._reason,
            last_transition_sim_time=self._t_transition,
        )

    # ------------------------------------------------------------------
    def _transition_to_hold(
        self, sim_time: float, reason: str, *, current_pose: np.ndarray | None = None
    ) -> None:
        self._state = LockState.HOLD
        self._reason = reason
        self._t_transition = sim_time
        if current_pose is not None:
            self._hold_target = current_pose.copy()
        elif self._hold_target is None:
            # Fallback: hold at home if we got asked to hold without a pose.
            self._hold_target = self.home_pose.copy()
