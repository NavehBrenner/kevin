"""LAB-78: verify a noassist replay reproduces each episode's generation baseline.

Mirrors what `kvn episode --input <ep> --policy noassist` now does: rebuild the
generation scene, *reconstruct the operator* from its seed (not the recorded
commands), and run it under NoAssist for the generation budget with the shared
termination policy. The terminal reason must match metadata.baseline.
Run: uv run python scripts/dev/lab78_verify_replay_matches_baseline.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ai_teleop.common.seating import SeatingGeometry
from ai_teleop.control import Controller
from ai_teleop.control.lock import LockState
from ai_teleop.data.generate import (
    DEFAULT_LATERAL_TOLERANCE,
    DEFAULT_SUCCESS_DEPTH,
    episode_terminal_reason,
    make_episode_operator,
)
from ai_teleop.data.trajectory import TerminalReason, load_episode
from ai_teleop.domain import NoAssist
from ai_teleop.sim.runner import run_episode
from ai_teleop.sim.scene import SimEnv
from ai_teleop.sim.scene_source import STATIC_TASK_SCENE


def replay_reason(meta) -> tuple[str, int]:
    master_seed, episode_index = (int(v) for v in meta["scene_seed"])
    env = SimEnv(str(STATIC_TASK_SCENE), seed=master_seed, randomize=True)
    obs = env.reset(episode_index)
    controller = Controller(env, max_dpos_per_step=float(meta["max_dpos"]))
    operator = make_episode_operator(
        obs.target_hole_position.copy(),
        controller.home_pose[3:],
        seed=master_seed,
        episode_index=episode_index,
        max_approach_speed=float(meta["max_approach_speed"]),
    )
    force_cap = float(meta["force_cap"])
    state = {"reason": TerminalReason.TIMEOUT, "step": -1}

    def cb(step, o, base, delta, command):
        g = SeatingGeometry.from_observation(o)
        r = episode_terminal_reason(
            penetration=g.penetration,
            lateral_error=g.lateral_error,
            force_magnitude=float(np.linalg.norm(o.wrist_ft[:3])),
            locked=controller.status.state is LockState.HOLD,
            success_depth=DEFAULT_SUCCESS_DEPTH,
            lateral_tolerance=DEFAULT_LATERAL_TOLERANCE,
            force_cap=force_cap,
        )
        if r is not None:
            state["reason"], state["step"] = r, step
            return True
        return False

    result = run_episode(
        env,
        controller,
        operator,
        NoAssist(),
        max_steps=int(meta["max_steps"]),
        reset_episode_index=episode_index,
        step_callback=cb,
    )
    return state["reason"].value, (result.n_steps if state["step"] < 0 else state["step"] + 1)


def main() -> None:
    paths = sorted(Path("data/dataset_0/runs").glob("episode_*/episode.npz"))
    mismatches = 0
    print(f"{'ep':>3} {'baseline (gen)':<14} {'replay (noassist)':<18} {'steps':>6}  match")
    for p in paths:
        _, meta = load_episode(p)
        baseline = meta.get("baseline_terminal_reason")
        reason, steps = replay_reason(meta)
        ok = reason == baseline
        mismatches += not ok
        print(
            f"{meta['episode_index']:>3} {str(baseline):<14} {reason:<18} {steps:>6}  {'OK' if ok else 'MISMATCH'}"
        )
    print(
        f"\n{len(paths) - mismatches}/{len(paths)} match"
        + ("" if not mismatches else f"  ({mismatches} MISMATCH)")
    )


if __name__ == "__main__":
    main()
