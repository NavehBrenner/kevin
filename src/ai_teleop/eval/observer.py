"""Passive-observer evaluation harness (LAB-36).

A :class:`TrialObserver` watches a runtime episode through the existing
``run_episode`` per-tick hook and emits one :class:`~ai_teleop.eval.schema.TrialKPIs`
record. It is a **passive observer**: it reads the ``Observation`` stream and
never calls into the controller — the controller stays mode-less and knows
nothing about trials, success, or KPIs. *Trial concepts live only here.* This
Dependency-Inversion split is a pillar from ``project-scope.md``; a structural
test asserts no ``eval/`` ↔ ``control/`` import either way.

Plug it in as the runner's ``step_callback``::

    observer = TrialObserver()
    run_episode(env, controller, human, assist, max_steps=N, step_callback=observer)
    kpis = observer.result()   # TrialKPIs

Because it is a plain per-tick callback that reads only its ``observation``
argument (never the runner), the *same* observer computes the *same* KPIs whether
driven live by ``run_episode`` or replayed offline over a logged trajectory — the
eval-log producer/consumer split LAB-37 adds. The metric math has one home, not
two; "offline" just points this calculator at a recorded ``Observation`` stream.

The observer self-detects a trial boundary from ``sim_time`` resetting toward 0
(the same signal the stateful policy uses for its per-episode reset), so a fresh
instance per trial and a reused instance across episodes both behave correctly.

What it reads off each ``Observation`` (all privileged ground truth — evaluation
only, never fed to a deployed policy):

* ``peg_pose`` + ``hole_poses[target_hole_index]`` → insertion depth along the
  hole axis and lateral error → success/failure classification.
* ``wrist_ft`` → peak contact force and contact-event counts. The wrench is
  **tared at trial start** (bias captured on the first step, exactly as data-gen
  and the deployed policy tare at reset), so "contact force" is the static-offset
  -removed quantity, not raw.
* ``ee_pose`` over ``sim_time`` → trajectory smoothness (∫|jerk| dt) and
  time-to-insert.

Success is the headline decision and is deliberately conservative: the peg must
be **sustained** past the depth threshold (within lateral clearance) for
``sustained_duration_s`` — a transient overshoot that pops back out does not
count.
"""

from __future__ import annotations

import numpy as np

from ai_teleop.common.observation import Observation
from ai_teleop.common.seating import PEG_HALF_LENGTH, SeatingGeometry
from ai_teleop.eval.schema import TrialKPIs, TrialOutcome

# ``PEG_HALF_LENGTH`` and the seating geometry come from ``common.seating`` — the
# one shared definition data-gen and this harness both use, so a "success" here
# means the same thing it meant when the BC corpus was scored.
__all__ = ["PEG_HALF_LENGTH", "TrialObserver"]

# Classification thresholds. Defaults match the data-generation seating
# definition (``data.generate.DEFAULT_*``) so a "success" here means the same
# thing it meant when the BC corpus was scored; LAB-37 calibrates the operating
# point (these are the knobs it sweeps).
DEFAULT_SUCCESS_DEPTH = 0.015  # penetration past the hole entry → seated (m)
DEFAULT_LATERAL_TOLERANCE = 0.006  # max lateral tip error for a seated peg (m)
DEFAULT_FORCE_CAP = 50.0  # contact-force magnitude that aborts the trial (N)
DEFAULT_SUSTAINED_DURATION = 0.05  # seating must hold this long to count (s)

# Contact-event detection — a rising edge above ``CONTACT_FORCE_FLOOR`` counts
# one contact; the force must fall back below ``CONTACT_RELEASE_FLOOR`` before
# the next is counted (hysteresis debounce, so one sustained press is one event).
DEFAULT_CONTACT_FORCE_FLOOR = 5.0  # N
DEFAULT_CONTACT_RELEASE_FLOOR = 2.5  # N


class TrialObserver:
    """``step_callback``-compatible passive observer producing a :class:`TrialKPIs`.

    Construct one per trial (or reuse across episodes — it resets on a detected
    ``sim_time`` boundary), pass it as ``run_episode(..., step_callback=obs)``,
    then read :meth:`result` once the episode returns.
    """

    def __init__(
        self,
        *,
        success_depth: float = DEFAULT_SUCCESS_DEPTH,
        lateral_tolerance: float = DEFAULT_LATERAL_TOLERANCE,
        force_cap: float = DEFAULT_FORCE_CAP,
        sustained_duration_s: float = DEFAULT_SUSTAINED_DURATION,
        contact_force_floor: float = DEFAULT_CONTACT_FORCE_FLOOR,
        contact_release_floor: float = DEFAULT_CONTACT_RELEASE_FLOOR,
        seed: int | None = None,
        config_label: str | None = None,
    ) -> None:
        self._success_depth = success_depth
        self._lateral_tolerance = lateral_tolerance
        self._force_cap = force_cap
        self._sustained_duration_s = sustained_duration_s
        self._contact_force_floor = contact_force_floor
        self._contact_release_floor = contact_release_floor
        self._seed = seed
        self._config_label = config_label
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self._started = False
        self._ft_bias = np.zeros(6)
        self._start_time = 0.0
        self._last_time = 0.0
        self._n_steps = 0
        self._outcome = TrialOutcome.TIMEOUT
        self._peak_contact_force = 0.0
        self._contact_events = 0
        self._in_contact = False
        self._seated_since_time: float | None = None  # when the current seat began
        self._time_to_insert_s: float | None = None
        self._ee_positions: list[np.ndarray] = []
        self._times: list[float] = []

    def __call__(
        self,
        step: int,
        observation: Observation,
        base_command: object,
        delta: object,
        command: object,
    ) -> bool:
        """One tick. Returns True to end the trial (success or force-abort)."""
        # Trial boundary: first call, or sim_time jumping back toward 0.
        if not self._started or observation.sim_time < self._last_time:
            self._reset_accumulators()
            self._started = True
            self._ft_bias = observation.wrist_ft.copy()  # tare, as the policy does
            self._start_time = observation.sim_time

        self._last_time = observation.sim_time
        self._n_steps += 1
        self._ee_positions.append(observation.ee_pose[:3].copy())
        self._times.append(observation.sim_time)

        contact_force = float(np.linalg.norm((observation.wrist_ft - self._ft_bias)[:3]))
        self._peak_contact_force = max(self._peak_contact_force, contact_force)
        self._update_contact_events(contact_force)

        if contact_force > self._force_cap:
            self._outcome = TrialOutcome.FORCE_ABORT
            return True

        geometry = SeatingGeometry.from_observation(observation)
        if (
            geometry.penetration >= self._success_depth
            and geometry.lateral_error < self._lateral_tolerance
        ):
            if self._seated_since_time is None:
                self._seated_since_time = observation.sim_time
            if observation.sim_time - self._seated_since_time >= self._sustained_duration_s:
                self._outcome = TrialOutcome.SUCCESS
                self._time_to_insert_s = self._seated_since_time - self._start_time
                return True
        else:
            self._seated_since_time = None  # popped back out — not sustained

        return False

    def _update_contact_events(self, contact_force: float) -> None:
        """Count rising edges above the contact floor with hysteresis debounce."""
        if not self._in_contact and contact_force > self._contact_force_floor:
            self._in_contact = True
            self._contact_events += 1
        elif self._in_contact and contact_force < self._contact_release_floor:
            self._in_contact = False

    def result(self) -> TrialKPIs:
        """Assemble the trial's KPI record (call once the episode has ended)."""
        duration_s = self._last_time - self._start_time if self._n_steps else 0.0
        return TrialKPIs(
            outcome=self._outcome,
            time_to_insert_s=self._time_to_insert_s,
            peak_contact_force=self._peak_contact_force,
            contact_events=self._contact_events,
            jerk_integral=self._compute_jerk_integral(),
            n_steps=self._n_steps,
            duration_s=duration_s,
            seed=self._seed,
            config_label=self._config_label,
        )

    def _compute_jerk_integral(self) -> float:
        """∫|jerk| dt over the end-effector path via finite differences.

        Jerk is the third time-derivative of position; integrated absolute jerk
        is a standard motion-smoothness proxy (lower = smoother). Needs at least
        four samples to form a third difference — fewer ⇒ 0.0.
        """
        if self._n_steps < 4:
            return 0.0
        positions = np.array(self._ee_positions)  # (T, 3)
        times = np.array(self._times)  # (T,)
        velocity = np.diff(positions, axis=0) / np.diff(times)[:, None]
        midpoint_times = 0.5 * (times[:-1] + times[1:])
        acceleration = np.diff(velocity, axis=0) / np.diff(midpoint_times)[:, None]
        accel_times = 0.5 * (midpoint_times[:-1] + midpoint_times[1:])
        jerk = np.diff(acceleration, axis=0) / np.diff(accel_times)[:, None]
        jerk_dt = np.diff(accel_times)
        return float(np.sum(np.linalg.norm(jerk, axis=1) * jerk_dt))
