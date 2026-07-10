"""LAB-105 probe: time a real vision DAgger rollout + confirm the relabel path.

Runs a couple of on-policy rollouts with the actual vision checkpoint and rendering
ON, so we (a) confirm the vision policy acts on captured frames and the expert
relabels, and (b) measure per-episode wall time to size the real round.

    GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \\
    LD_LIBRARY_PATH=/usr/lib/wsl/lib \\
    uv run python scripts/dev/lab105_rollout_probe.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ai_teleop.common.log import configure_from_args, get_logger
from ai_teleop.dagger import expert_from_config, rollout_and_relabel
from ai_teleop.data.trajectory import load_episode
from ai_teleop.policy import LearnedResidual

log = get_logger("lab105-probe")

BASE = Path("data/dataset_vision")
CHECKPOINT = Path("outputs/policy/runs/vision_frozen_ar100/checkpoint.pt")


def main() -> int:
    configure_from_args(type("A", (), {"log_level": "INFO", "quiet": False, "log_file": None})())
    config = json.loads((BASE / "metadata.json").read_text())["config"]
    expert = expert_from_config(config)
    policy = LearnedResidual.from_checkpoint(CHECKPOINT, device="cuda")
    log.info("policy use_vision=%s", policy.use_vision)

    runs_dir = Path("data/_lab105_probe/runs")
    for rollout_index in range(2):
        start = time.time()
        summary = rollout_and_relabel(
            policy=policy,
            expert=expert,
            runs_dir=runs_dir,
            dagger_index=rollout_index,
            master_seed=105,
            rollout_index=rollout_index,
            config=config,
            render_every=20,
        )
        wall = time.time() - start
        columns, _ = load_episode(runs_dir / f"episode_{rollout_index:05d}" / "episode.npz")
        n_frames = len(list((runs_dir / f"episode_{rollout_index:05d}" / "imgs").glob("*.jpg")))
        log.info(
            "rollout %d │ %6d steps │ %s │ %d frames │ %.1fs (%.1f steps/s)",
            rollout_index,
            summary["n_steps"],
            summary["terminal_reason"],
            n_frames,
            wall,
            summary["n_steps"] / wall,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
