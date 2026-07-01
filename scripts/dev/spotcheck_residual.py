"""Headless qualitative spot-check: learned residual vs human-only (LAB-34 / M5).

Loads a trained checkpoint and runs **paired** episodes — same scene, same
operator command stream (identical ``(seed, episode_index)``) — once with the
``LearnedResidual`` assisting and once with ``NoAssist`` (human-only). Reports the
final seating depth / lateral error / success per episode and the aggregate
success rates, so you can eyeball whether the policy beats human-only on held-out
seeds. This is the M5 acceptance's *qualitative* check; the rigorous paired KPI
comparison is M6.

Run from the ``kevin/`` directory (point it at a run folder's ``checkpoint.pt``)::

    uv run python scripts/dev/spotcheck_residual.py outputs/policy/runs/<run>/checkpoint.pt \\
        --episodes 20 --seed 99

Reuses the M4 generator's matched-operator + seating definitions
(``_make_human`` / ``_SeatingMetrics``) so "seated" means exactly what data-gen
scored it as.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.control import Controller  # noqa: E402
from ai_teleop.data.generate import (  # noqa: E402
    DEFAULT_LATERAL_TOLERANCE,
    DEFAULT_MAX_STEPS,
    DEFAULT_SUCCESS_DEPTH,
    _make_human,
)
from ai_teleop.data.step_callbacks import _SeatingMetrics  # noqa: E402
from ai_teleop.domain import NoAssist  # noqa: E402
from ai_teleop.policy import LearnedResidual  # noqa: E402
from ai_teleop.sim.config import EnvConfig, episode_wall_seed  # noqa: E402
from ai_teleop.sim.env_setup import make_env  # noqa: E402
from ai_teleop.sim.runner import run_episode  # noqa: E402

log = get_logger("spotcheck")

# Data generation places the goal at hole_0; the spot-check scores against the same.
_TARGET_HOLE_INDEX = 0


def _seated_after(environment, controller, human, assist, *, max_steps):
    """Run one episode; return (penetration_m, lateral_error_m, success)."""
    controller.reset()
    result = run_episode(environment, controller, human, assist, max_steps=max_steps)
    metrics = _SeatingMetrics(result.final_observation, _TARGET_HOLE_INDEX)
    success = metrics.penetration >= DEFAULT_SUCCESS_DEPTH and (
        metrics.lateral_error < DEFAULT_LATERAL_TOLERANCE
    )
    return metrics.penetration, metrics.lateral_error, success


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Trained policy checkpoint (.pt).")
    parser.add_argument("--seed", type=int, default=0, help="Master seed (match training).")
    parser.add_argument("--episodes", type=int, default=20, help="Paired episodes to run.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--device", default="cpu")
    add_logging_arguments(parser)
    args = parser.parse_args(argv)
    configure_from_args(args)

    policy = LearnedResidual.from_checkpoint(args.checkpoint, device=args.device)
    no_assist = NoAssist()

    policy_successes = 0
    human_successes = 0
    for episode_index in range(args.episodes):
        # One clean env per episode, on its own wall (seeded like data-gen), so the
        # paired policy/human runs are compared on an identical scene.
        environment = make_env(
            EnvConfig(wall_seed=episode_wall_seed(args.seed, episode_index)),
            render_mode="headless",
        )
        controller = Controller(environment)
        home_quaternion = controller.home_pose[3:]
        observation = environment.reset()
        target_position = observation.hole_poses[_TARGET_HOLE_INDEX][:3].copy()

        policy.reset()  # explicit per-episode reset (also auto-resets on sim_time restart)
        policy_pen, policy_lat, policy_ok = _seated_after(
            environment,
            controller,
            _make_human(
                target_position, home_quaternion, seed=args.seed, episode_index=episode_index
            ),
            policy,
            max_steps=args.max_steps,
        )
        human_pen, human_lat, human_ok = _seated_after(
            environment,
            controller,
            _make_human(
                target_position, home_quaternion, seed=args.seed, episode_index=episode_index
            ),
            no_assist,
            max_steps=args.max_steps,
        )
        environment.close()
        policy_successes += int(policy_ok)
        human_successes += int(human_ok)
        log.info(
            "ep %3d │ policy %s pen %5.1fmm lat %4.1fmm │ human %s pen %5.1fmm lat %4.1fmm",
            episode_index,
            "✓" if policy_ok else "✗",
            policy_pen * 1e3,
            policy_lat * 1e3,
            "✓" if human_ok else "✗",
            human_pen * 1e3,
            human_lat * 1e3,
        )

    n = args.episodes
    log.info(
        "success: policy %d/%d (%.0f%%) vs human-only %d/%d (%.0f%%)",
        policy_successes,
        n,
        100 * policy_successes / n,
        human_successes,
        n,
        100 * human_successes / n,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
