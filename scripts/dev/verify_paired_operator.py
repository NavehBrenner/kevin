"""Is the operator command stream identical across arms? (the paired design's core assumption)

``eval/trace.py`` logs ``base_cmd_*`` explicitly so "the paired-seed design can be *verified*
(identical operator stream across configs)". This is that verification, and it answers a
second question the M7 post-mortem needs: **does the operator react to the policy?**

`ScriptedNoisyHuman` integrates its own command state, seeded once from the first
observation's ``ee_pose`` and thereafter advanced only by its own OU drift — it never reads
the observation again. So it *should* be open-loop with respect to the assist: same seed ⇒
bit-identical command stream in both arms, no matter what the policy does to the arm.

If that holds, two things follow for interpreting the closed-loop nulls:

* The paired Δ really does isolate the assist (zero operator variance), as claimed.
* The online/offline gap is **not** compounded by operator reaction. The only thing that
  differs online is the state distribution the policy induces on the *arm* — which is
  precisely what DAgger addresses, and DAgger was run.

If it does not hold, both of those claims need revisiting, which is why this is worth an
explicit check rather than an appeal to the source.

Read-only. Run from kevin/::

    uv run python scripts/dev/verify_paired_operator.py --run runs/eval_official_ft_s0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.eval.trace import load_eval_trace  # noqa: E402

log = get_logger("verify-operator")


def compare_arms(run_dir: Path, baseline: str, treatment: str) -> int:
    """Compare the logged operator command of both arms, episode by episode."""
    baseline_dir = run_dir / "traces" / baseline
    treatment_dir = run_dir / "traces" / treatment
    episodes = sorted(path.parent.name for path in baseline_dir.glob("episode_*/trace.npz"))
    if not episodes:
        log.error("no traces under %s", baseline_dir)
        return 1

    worst_divergence = 0.0
    worst_episode = ""
    diverged = 0
    compared = 0
    for episode in episodes:
        treatment_trace = treatment_dir / episode / "trace.npz"
        if not treatment_trace.is_file():
            continue
        baseline_columns, _ = load_eval_trace(baseline_dir / episode / "trace.npz")
        treatment_columns, _ = load_eval_trace(treatment_trace)
        # The arms end at different steps (one aborts, one seats), so compare the overlap:
        # divergence, if any, would appear from the first step the policy could influence.
        steps = min(
            baseline_columns["base_cmd_position"].shape[0],
            treatment_columns["base_cmd_position"].shape[0],
        )
        difference = (
            baseline_columns["base_cmd_position"][:steps]
            - treatment_columns["base_cmd_position"][:steps]
        )
        divergence = float(np.abs(difference).max())
        compared += 1
        if divergence > 0.0:
            diverged += 1
        if divergence > worst_divergence:
            worst_divergence, worst_episode = divergence, episode

    log.info(
        "%s: compared %d episode pairs (%s vs %s)", run_dir.name, compared, baseline, treatment
    )
    log.info("  episodes with any divergence: %d/%d", diverged, compared)
    log.info("  worst |Δ base_cmd_position|: %.3e m  (%s)", worst_divergence, worst_episode or "—")
    if worst_divergence == 0.0:
        log.info("  ⇒ operator stream is BIT-IDENTICAL across arms — open-loop w.r.t. the assist.")
    else:
        log.warning("  ⇒ operator stream DIFFERS — the operator reacts to the policy.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/eval_official_ft_s0"))
    parser.add_argument("--baseline", default="human_only")
    parser.add_argument("--treatment", default="residual")
    add_logging_arguments(parser)
    arguments = parser.parse_args()
    configure_from_args(arguments)
    return compare_arms(arguments.run, arguments.baseline, arguments.treatment)


if __name__ == "__main__":
    raise SystemExit(main())
