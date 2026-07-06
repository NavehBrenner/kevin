"""ScriptedNoisyHuman — the realistic structured-noise operator (M4).

A seedable, deterministic source of *realistically-wrong* coarse commands. It
is **not** a model of human cognition; it is a controllable actor so the expert
has something to correct and KPI runs are reproducible.

The design constraint (see `docs/design/human-generation.md`): the noise must be
**structured and low-frequency, not per-step white noise**. Per-step i.i.d.
noise (the M3 stub) collapses the expert's optimal correction to "negate the
injected noise" — a trivial, unphysical denoising task. Instead this actor
commits to a *biased, drifting, coarse* trajectory that is internally consistent
over time, so the correction problem stays a geometric/contact-reasoning one.

The command stream itself carries an **approach phase** (LAB-78): the actor
integrates a command that *chases* the drifting biased goal at a capped rate,
seeded from where the arm actually starts. So the command sweeps in from ~400 mm
out to the goal — the operator owns the approach, not the controller's command
clamp. (The old model parked the command at the goal from tick 0 and leaned on
the clamp to manufacture the approach; that left the command stream — a live
policy input via the command-history GRU — structurally unlike a real
operator's, which bit M7 vision specifically.)

Near the goal, the approach **decelerates proportionally to distance** (LAB-91)
instead of holding the rate cap until the literal last tick: real operators
visually servo and slow down before contact, which a bang-bang rate cap can't
reproduce (measured: recorded near-field command speed has a heavy tail, p90:
median ratio ~2.5x, vs a flat ~1.1x for the old bang-bang model — see
`project-wiki/synthesis/scripted-vs-real-operator.md`). Within
``decel_radius`` of the (drifting) goal the per-tick step is scaled by
``distance / decel_radius``, a simple proportional/critically-damped approach;
outside it the actor is still rate-capped at ``max_approach_speed`` exactly as
before, so LAB-78's far-field aim/travel calibration is untouched.

A per-episode ``careless_probability`` (LAB-92) draws whether *this* episode
skips that deceleration entirely — modeling an inattentive/overconfident
operator who sweeps in at ``max_approach_speed`` all the way to contact. This
targets a gap LAB-91's continuous per-tick deceleration couldn't close: real
force-abort episodes move ~45% faster/step than real successes (a distinct
"careless fast approach" failure mode), while a continuously-tuned deceleration
profile applies the same near-goal slowdown to every episode alike, so
scripted force-abort and success motion stayed ~3% apart regardless of
``decel_radius``. Default 0 disables the mode, reproducing LAB-91 behavior
exactly (including its RNG stream — see the constructor).

A per-episode **lognormal draw on ``max_approach_speed``** (LAB-95/96,
``speed_lognormal_median``/``speed_lognormal_sigma``) supersedes that discrete
carelessness draw as the episode-level speed lever: the recorded corpus'
realized contact speed spreads ~4x across episodes, and that spread — not any
within-episode profile — is what separates force-abort episodes from successes
(aborts arrive ~+41% faster). Neither LAB-91's deceleration nor LAB-92's
carelessness could express it under the old data-gen controller config
(kd=4 pinned realized speed at ~55 mm/s regardless of the command); under the
deployment config (kd=1.5, the config `data/recorded` was captured under) the
lognormal draw reproduces the recorded force-abort rate, motion-tail ratio,
and contact-speed differential simultaneously — see
`project-wiki/synthesis/scripted-vs-real-operator.md` (LAB-95 section).
Default 0 disables the draw (fixed ``max_approach_speed``, RNG untouched).

Composition, fully determined by the constructor `seed` (plus the arm's
deterministic reset pose, read once from the first observation):

    command_0 = observation.ee_pose[:3]                          # seed at the arm's start
    target_t  = goal ⊕ drift_t                                   # the drifting biased goal
    command_t = command_{t-1} + clamp(target_t - command_{t-1},  max_approach_speed · dt)
    goal      = (p_hole + bias,  R_hole · ΔR_bias)               # fixed for the whole episode

- **Intent / bias** — `bias` (position + orientation) drawn **once** at
  construction. A systematic, consistent misjudgement of where the hole is; it
  does not resample per step. This consistency is what makes the correction
  non-trivial.
- **Drift** — a low-frequency Ornstein–Uhlenbeck process on position and
  orientation, advanced **every control tick** (not held), so the command keeps
  making small moves even after arrival; correlated over hundreds of ms, not
  white.
- **Approach** — the command integrates toward ``target_t`` at up to
  ``max_approach_speed``, decelerating proportionally inside the last step.
- **Tremor** — optional small per-tick high-frequency jitter (off by default),
  kept well below the magnitude that would make denoising the whole game.

The actor is deliberately **contact-unaware**: it always commands toward the
(biased) goal regardless of what the peg is touching — it will keep pushing into
flat wall if its goal sits off the hole, exactly the situation the assist must
rescue.

Noise *magnitudes* here are placeholders to be calibrated post-baseline; the
*form* (biased + drifting + capped-rate approach) is what is fixed now.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ai_teleop.common.command import Command
from ai_teleop.common.observation import Observation

# Default per-episode lateral-error magnitudes (m). Named so the eval harness can
# scale operator difficulty relative to the distribution the M5 corpus was generated
# at (scale 1.0 == these values). Bias is the constant per-episode offset; drift is
# the stationary OU wander on top of it — together they set the lateral error at
# contact, which the difficulty pin (LAB-53) trades against the chamfer capture radius.
DEFAULT_POSITION_BIAS_STD: float = 0.013
DEFAULT_DRIFT_POSITION_STD: float = 0.005

# Cap on how fast the command sweeps toward the (drifting) goal, m/s. Sets the
# approach duration and near-field command speed; the LAB-78 fit target. The
# controller's per-step clamp (2 cm/step) sits well above the per-tick move this
# implies, so the actor — not the clamp — owns the approach.
DEFAULT_MAX_APPROACH_SPEED: float = 0.35

# Radius (m) around the drifting goal within which the approach decelerates
# proportionally to distance instead of holding max_approach_speed — the LAB-91
# fit target for the near-field command-speed distribution (real operators slow
# down before contact; placeholder, calibrated against `data/recorded`).
DEFAULT_DECEL_RADIUS: float = 0.016

# Per-episode probability the operator is "careless" (LAB-92): skips the
# decel_radius deceleration entirely for that episode, sweeping in at
# max_approach_speed all the way through contact. Models operator
# inattention/overconfidence — a discrete, episode-level failure mode distinct
# from decel_radius's continuous per-tick profile. 0 (default) disables it,
# reproducing LAB-91 behavior (and its RNG stream) exactly.
DEFAULT_CARELESS_PROBABILITY: float = 0.0

# Per-episode lognormal draw on max_approach_speed (LAB-95/96): a real operator's
# realized contact speed spreads ~4x across episodes (26-105 mm/s p10-p90 in
# `data/recorded`), and that episode-level careful-vs-hasty spectrum IS the
# force-abort signature (fast episodes abort). A lognormal fitted to the recorded
# near-field command speeds reproduces it: p90/median ~2.7 => sigma = 0.76;
# median 0.09-0.12 m/s brackets the recorded stats under the deployment
# controller config. Median 0 (default) disables the draw — the fixed
# max_approach_speed applies, and the RNG stream is untouched (LAB-92 pattern).
DEFAULT_SPEED_LOGNORMAL_MEDIAN: float = 0.0
DEFAULT_SPEED_LOGNORMAL_SIGMA: float = 0.76


def _axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert a (3,) axis-angle vector to a (w,x,y,z) unit quaternion."""
    quat = np.zeros(4)
    angle = float(np.linalg.norm(axis_angle))
    if angle > 0.0:
        mujoco.mju_axisAngle2Quat(quat, axis_angle / angle, angle)
    else:
        quat[0] = 1.0  # identity
    return quat


class ScriptedNoisyHuman:
    """Structured-noise scripted operator implementing :class:`InputStrategy`.

    Parameters
    ----------
    target_pose:
        Shape (7,) array — (px, py, pz, qw, qx, qy, qz) in world frame. The
        actor's *intended* goal (typically the active trial's hole-vicinity
        pose); the actor adds its own per-episode bias on top.
    position_bias_std:
        Per-episode constant position-bias σ, in metres (drawn once).
    orientation_bias_std:
        Per-episode constant angular-bias σ per axis-angle component, in radians.
    drift_position_std:
        Stationary σ of the position OU drift, in metres.
    drift_orientation_std:
        Stationary σ of the orientation OU drift, in radians.
    drift_tau:
        OU time constant, in seconds (larger ⇒ slower, more correlated drift).
    tremor_std:
        Per-tick high-frequency position-tremor σ, in metres. 0 disables tremor.
    max_approach_speed:
        Cap on the command sweep toward the goal, in m/s. Larger ⇒ faster
        approach. See :data:`DEFAULT_MAX_APPROACH_SPEED`.
    decel_radius:
        Distance (m) to the drifting goal inside which the approach speed
        scales down proportionally to distance, instead of holding
        ``max_approach_speed``. See :data:`DEFAULT_DECEL_RADIUS`.
    careless_probability:
        Per-episode probability the operator skips ``decel_radius``'s
        deceleration entirely, sweeping in at ``max_approach_speed`` all the
        way to contact. Drawn once at construction, alongside the position/
        orientation bias. Only consumes RNG state when > 0, so the default
        (0) reproduces the pre-LAB-92 RNG stream exactly. See
        :data:`DEFAULT_CARELESS_PROBABILITY`.
    speed_lognormal_median:
        Median (m/s) of a per-episode lognormal draw that *replaces* the fixed
        ``max_approach_speed`` — the careful-vs-hasty episode spectrum
        (LAB-95/96). Drawn once at construction; the realized value is exposed
        as :attr:`max_approach_speed`. Only consumes RNG state when > 0, so
        the default (0, disabled) reproduces the pre-LAB-96 RNG stream
        exactly. See :data:`DEFAULT_SPEED_LOGNORMAL_MEDIAN`.
    speed_lognormal_sigma:
        Log-space sigma of that draw (p90/median = exp(1.2817·sigma); the
        default 0.76 gives the recorded corpus' ~2.7). Ignored while the
        median is 0. See :data:`DEFAULT_SPEED_LOGNORMAL_SIGMA`.
    control_hz:
        Control-loop rate (one ``get_command`` call per tick). Sets the per-tick
        drift step ``dt = 1 / control_hz`` and the per-tick approach cap.
    seed:
        RNG seed. The data-gen driver passes a per-episode seed so that
        ``(master_seed, episode_index)`` fully determines the command stream.
    """

    def __init__(
        self,
        target_pose: np.ndarray,
        *,
        position_bias_std: float = DEFAULT_POSITION_BIAS_STD,
        orientation_bias_std: float = float(np.deg2rad(3.0)),
        drift_position_std: float = DEFAULT_DRIFT_POSITION_STD,
        drift_orientation_std: float = float(np.deg2rad(1.0)),
        drift_tau: float = 0.3,
        tremor_std: float = 0.0,
        max_approach_speed: float = DEFAULT_MAX_APPROACH_SPEED,
        decel_radius: float = DEFAULT_DECEL_RADIUS,
        careless_probability: float = DEFAULT_CARELESS_PROBABILITY,
        speed_lognormal_median: float = DEFAULT_SPEED_LOGNORMAL_MEDIAN,
        speed_lognormal_sigma: float = DEFAULT_SPEED_LOGNORMAL_SIGMA,
        control_hz: float = 500.0,
        seed: int = 0,
    ) -> None:
        if target_pose.shape != (7,):
            raise ValueError(f"target_pose must have shape (7,), got {target_pose.shape}")

        self._target_position = target_pose[:3].copy()
        self._target_quaternion = target_pose[3:].copy()
        mujoco.mju_normalize4(self._target_quaternion)

        self._drift_position_std = drift_position_std
        self._drift_orientation_std = drift_orientation_std
        self._tremor_std = tremor_std
        self._dt_control = 1.0 / control_hz
        self._decel_radius = decel_radius

        # Per-tick OU decay; stationary std preserved via the sqrt(1 - beta^2)
        # innovation scaling, so the drift's stationary σ is independent of dt.
        self._ou_beta = float(np.exp(-self._dt_control / drift_tau))
        self._ou_innovation = float(np.sqrt(1.0 - self._ou_beta**2))

        self._rng = np.random.default_rng(seed)

        # Per-episode constant bias — drawn ONCE, never resampled.
        self.position_bias: np.ndarray = self._rng.normal(0.0, position_bias_std, size=3)
        self.orientation_bias: np.ndarray = self._rng.normal(0.0, orientation_bias_std, size=3)

        # Per-episode "carelessness" draw (LAB-92) — only consumes the RNG when enabled,
        # so careless_probability=0.0 (default) leaves the RNG stream identical to
        # pre-LAB-92 code and every existing seeded trajectory stays reproducible.
        self._careless = bool(
            careless_probability > 0.0 and self._rng.random() < careless_probability
        )

        # Per-episode approach-speed draw (LAB-95/96) — a lognormal on
        # max_approach_speed models the careful-vs-hasty episode spectrum the
        # recorded corpus shows. Only consumes the RNG when enabled, so
        # speed_lognormal_median=0.0 (default) leaves the RNG stream identical
        # to pre-LAB-96 code and every existing seeded trajectory stays
        # reproducible.
        if speed_lognormal_median > 0.0:
            max_approach_speed = float(
                speed_lognormal_median * np.exp(speed_lognormal_sigma * self._rng.normal())
            )
        self.max_approach_speed: float = float(max_approach_speed)
        self._max_step = self.max_approach_speed * self._dt_control  # per-tick cap, metres

        self._goal_position = self._target_position + self.position_bias
        bias_quat = _axis_angle_to_quat(self.orientation_bias)
        self._goal_quaternion = np.zeros(4)
        mujoco.mju_mulQuat(self._goal_quaternion, bias_quat, self._target_quaternion)
        mujoco.mju_normalize4(self._goal_quaternion)

        # OU drift state (axis-angle for orientation), advanced every tick.
        self._drift_position = np.zeros(3)
        self._drift_orientation = np.zeros(3)

        # Command-integrator state, seeded lazily from the first observation's
        # ee_pose (the arm's deterministic reset pose) so the approach starts
        # where the arm actually is.
        self._command_position = np.zeros(3)
        self._initialized = False

    def _advance_drift(self) -> None:
        """Advance the position + orientation OU drift one control tick."""
        self._drift_position = (
            self._ou_beta * self._drift_position
            + self._ou_innovation * self._drift_position_std * self._rng.normal(size=3)
        )
        self._drift_orientation = (
            self._ou_beta * self._drift_orientation
            + self._ou_innovation * self._drift_orientation_std * self._rng.normal(size=3)
        )

    def get_command(self, observation: Observation) -> Command:
        if not self._initialized:
            self._command_position = observation.ee_pose[:3].copy()
            self._initialized = True

        self._advance_drift()

        # Capped-rate move of the command toward the drifting biased goal. The rate
        # cap (LAB-91) scales with distance to the *fixed* goal (large-scale "am I
        # still approaching or have I arrived", not the per-tick drift wobble) —
        # using the drifting target itself here would reset the reference every
        # tick, so the tracking error (and therefore speed) would settle at a
        # narrow near-constant value instead of the observed heavy-tailed
        # near-field speed (real operators cruise in from ~400 mm, then slow over
        # the final approach). Outside decel_radius this is the LAB-78 bang-bang
        # cap (unchanged); inside it the cap decays proportionally to distance.
        target = self._goal_position + self._drift_position
        step = target - self._command_position
        distance = float(np.linalg.norm(step))
        distance_to_goal = float(np.linalg.norm(self._command_position - self._goal_position))
        # A careless-drawn episode (LAB-92) never decelerates — it holds the far-field
        # rate cap all the way to contact, unlike the LAB-91 proportional ramp below.
        speed_scale = 1.0 if self._careless else min(1.0, distance_to_goal / self._decel_radius)
        effective_max_step = self._max_step * speed_scale
        self._command_position = self._command_position + step * min(
            1.0, effective_max_step / (distance + 1e-9)
        )

        position = self._command_position
        if self._tremor_std > 0.0:
            position = position + self._rng.normal(0.0, self._tremor_std, size=3)

        drift_quat = _axis_angle_to_quat(self._drift_orientation)
        quaternion = np.zeros(4)
        mujoco.mju_mulQuat(quaternion, drift_quat, self._goal_quaternion)
        mujoco.mju_normalize4(quaternion)

        return Command(position.copy(), quaternion)


def bore_aligned_grasp(home_quaternion: np.ndarray, bore_axis: np.ndarray) -> np.ndarray:
    """Grasp orientation that points the peg long axis along ``bore_axis``.

    The home (upright) grasp points the peg long axis along world +x — the bore
    direction of an *upright* wall. For a tilted wall the bore tilts, so we
    pre-rotate the home grasp by the shortest arc from +x to ``bore_axis``. This
    lets the scripted human issue a *coarse* bore-aimed orientation (its bias and
    drift are layered on top), leaving only a small residual for the expert —
    the coarse-human / fine-assist division of labour. Roll about the bore is
    irrelevant for a round peg, so the minimal-arc choice is fine.
    """
    bore = np.asarray(bore_axis, dtype=np.float64)
    bore = bore / np.linalg.norm(bore)
    x_axis = np.array([1.0, 0.0, 0.0])
    rotation_axis = np.cross(x_axis, bore)
    sin_angle = float(np.linalg.norm(rotation_axis))
    if sin_angle < 1e-9:
        return home_quaternion.copy()  # already aligned with +x
    angle = float(np.arccos(np.clip(x_axis @ bore, -1.0, 1.0)))
    rotation_quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(rotation_quat, rotation_axis / sin_angle, angle)
    grasp = np.zeros(4)
    mujoco.mju_mulQuat(grasp, rotation_quat, home_quaternion)
    mujoco.mju_normalize4(grasp)
    return grasp
