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
from ai_teleop.eval.ablation import HUMAN_ONLY, Config, replay_kpis, run_paired, run_trial
from ai_teleop.eval.schema import TrialKPIs
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


def test_run_trial_produces_wellformed_record():
    kpis = run_trial(0, HUMAN_ONLY, max_steps=MAX_STEPS)
    assert isinstance(kpis, TrialKPIs)
    assert kpis.config_label == "human_only"
    assert kpis.seed == 0
    assert kpis.n_steps > 0
    assert kpis.peak_contact_force >= 0.0
    assert kpis.jerk_integral >= 0.0


def test_paired_run_same_seed_identical_operator_stream(tmp_path):
    """The paired-design pillar: same seed ⇒ identical base_command across configs,
    even though the nudge config follows a different trajectory."""
    results = run_paired(4, [HUMAN_ONLY, NUDGE], out_dir=tmp_path, max_steps=MAX_STEPS)
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


def test_saved_trace_replays_to_same_kpis(tmp_path):
    """Offline replay of a real episode's trace reproduces the live KPIs."""
    trace_path = tmp_path / "trace.npz"
    live = run_trial(7, NUDGE, max_steps=MAX_STEPS, trace_path=trace_path)
    replayed = replay_kpis(trace_path)
    assert live.to_dict() == replayed.to_dict()
