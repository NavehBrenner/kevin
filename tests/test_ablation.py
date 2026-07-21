"""Paired-seed ablation runner tests (LAB-37).

Drives the real M3 stack on the static task scene (like ``test_episode_e2e``), so it
is an end-to-end check that the runner produces well-formed paired records, the
paired design holds (identical operator stream across configs), and a saved trace
replays to the same KPIs.
"""

from __future__ import annotations

import numpy as np

from ai_teleop.common.command import Command
from ai_teleop.common.observation import Observation
from ai_teleop.domain.delta import Delta
from ai_teleop.eval.ablation import (
    HUMAN_ONLY,
    INSERTION_MAX_STEPS,
    Config,
    replay_kpis,
    run_paired,
    run_trial,
)
from ai_teleop.eval.schema import TrialKPIs, TrialOutcome
from ai_teleop.eval.trace import TRACE_NPZ_NAME, load_eval_trace

# A short budget keeps the test fast; neither config seats in this window, so the
# two traces stay the same length and the full base_command stream is comparable.
MAX_STEPS = 80


class _ConstantNudge:
    """An AssistProvider that adds a fixed small Δ — a stand-in for a real policy.

    It changes the *trajectory* (so the two configs diverge) without needing a
    trained checkpoint, which is what makes the paired-design proof meaningful.
    """

    def get_delta(self, observation: Observation, command: Command) -> Delta:
        return Delta(np.array([0.003, 0.0, 0.0]), np.zeros(3), 0.0)


NUDGE = Config(label="nudge", assist_factory=_ConstantNudge)


class _RamIntoWall:
    """An AssistProvider that pushes hard *past* wherever the base command is
    already aiming, overshooting into whatever the operator is approaching.

    Used only to trip the controller's force-cap watchdog reliably (LAB-94
    regression) -- not a stand-in for any real policy. Direction-agnostic (scales
    off the base command's own aim) so it doesn't depend on the static scene's
    world-frame orientation.
    """

    def get_delta(self, observation: Observation, command: Command) -> Delta:
        toward_target = command.target_position - observation.ee_pose[:3]
        norm = np.linalg.norm(toward_target)
        direction = toward_target / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        return Delta(direction * 0.02, np.zeros(3), 0.0)


RAM_INTO_WALL = Config(label="ram_into_wall", assist_factory=_RamIntoWall)


def test_run_trial_produces_wellformed_record():
    kpis = run_trial(0, HUMAN_ONLY, max_steps=MAX_STEPS, generated_walls=False)
    assert isinstance(kpis, TrialKPIs)
    assert kpis.config_label == "human_only"
    assert kpis.seed == 0
    assert kpis.n_steps > 0
    assert kpis.peak_contact_force >= 0.0
    assert kpis.jerk_integral >= 0.0


def test_paired_run_same_seed_identical_operator_stream(tmp_path):
    """The paired-design pillar: same seed ⇒ identical base_command across configs,
    even though the nudge config follows a different trajectory."""
    results = run_paired(
        4, [HUMAN_ONLY, NUDGE], out_dir=tmp_path, max_steps=MAX_STEPS, generated_walls=False
    )
    assert set(results) == {"human_only", "nudge"}

    human_cols, _ = load_eval_trace(tmp_path / "human_only" / "episode_00004" / TRACE_NPZ_NAME)
    nudge_cols, _ = load_eval_trace(tmp_path / "nudge" / "episode_00004" / TRACE_NPZ_NAME)

    n = min(len(human_cols["step"]), len(nudge_cols["step"]))
    # Operator command stream is open-loop ⇒ identical tick-for-tick.
    np.testing.assert_allclose(
        human_cols["base_cmd_position"][:n], nudge_cols["base_cmd_position"][:n]
    )
    np.testing.assert_allclose(human_cols["base_cmd_quat"][:n], nudge_cols["base_cmd_quat"][:n])
    # The nudge actually perturbed the realized trajectory (sanity: configs differ).
    assert not np.allclose(human_cols["ee_pose"][:n], nudge_cols["ee_pose"][:n])


def test_operator_error_scale_changes_the_command_stream(tmp_path):
    """The LAB-53 difficulty knob is wired: at the same seed, scale=0 (no bias/drift)
    yields a different — and closer-to-target — operator stream than the trained σ's."""
    run_trial(
        2,
        HUMAN_ONLY,
        max_steps=MAX_STEPS,
        operator_error_scale=0.0,
        trace_path=tmp_path / "off.npz",
        generated_walls=False,
    )
    run_trial(
        2,
        HUMAN_ONLY,
        max_steps=MAX_STEPS,
        operator_error_scale=1.0,
        trace_path=tmp_path / "on.npz",
        generated_walls=False,
    )
    off_cols, _ = load_eval_trace(tmp_path / "off.npz")
    on_cols, _ = load_eval_trace(tmp_path / "on.npz")

    target = off_cols["target_hole_pose"][0, :3]
    off_err = np.linalg.norm(off_cols["base_cmd_position"] - target, axis=1).mean()
    on_err = np.linalg.norm(on_cols["base_cmd_position"] - target, axis=1).mean()
    assert off_err < on_err  # scaling the σ's down moves the operator toward the hole


def test_saved_trace_replays_to_same_kpis(tmp_path):
    """Offline replay of a real episode's trace reproduces the live KPIs."""
    trace_path = tmp_path / "trace.npz"
    live = run_trial(7, NUDGE, max_steps=MAX_STEPS, trace_path=trace_path, generated_walls=False)
    replayed = replay_kpis(trace_path)
    assert live.to_dict() == replayed.to_dict()


def test_controller_watchdog_trip_is_classified_force_abort():
    """LAB-94 regression: a trial that trips the controller's force-cap watchdog
    must be classified FORCE_ABORT, not TIMEOUT.

    Before the fix, ``run_trial`` wired the controller's watchdog (hardcoded 30N)
    and the observer's FORCE_ABORT threshold (its own default, 50N) independently
    -- the controller always froze the arm at 30N first, so the observer's raw
    force check never saw anything past that and every such trial silently fell
    through to TIMEOUT. Confirmed live on the pre-fix code: with this same setup
    the controller's watchdog visibly trips (`lock active -> HOLD ... 33.21N` in
    the log) yet the outcome was still TIMEOUT. Post-fix, ``force_cap`` couples
    both thresholds and this same scenario correctly classifies FORCE_ABORT.

    A larger ``max_dpos`` raises the steady-state contact-force ceiling (the
    controller's own position clamp otherwise bounds it to ~10-15N here, which
    the default 30N cap can't reliably reach) so the watchdog trips deterministically.
    """
    kpis = run_trial(0, RAM_INTO_WALL, max_steps=3000, max_dpos=0.1, generated_walls=False)
    assert kpis.outcome is TrialOutcome.FORCE_ABORT


def test_ablation_runners_default_to_insertion_budget():
    """LAB-107 regression: the paired-ablation entry points must default to the
    insertion step budget. Callers that omit ``max_steps`` — notably
    ``dagger._reablate`` — rely on this default, and ``scripts/evaluate.py`` anchors
    its own ``--max-steps`` default on the same ``INSERTION_MAX_STEPS`` constant. If
    this default drifts back to the generic sim budget the two eval paths silently
    disagree again (the DAgger path read human-only 25% vs evaluate.py's 35%)."""
    import inspect

    assert inspect.signature(run_paired).parameters["max_steps"].default == INSERTION_MAX_STEPS
    assert inspect.signature(run_trial).parameters["max_steps"].default == INSERTION_MAX_STEPS
