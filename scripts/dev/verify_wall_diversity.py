"""Verify per-episode wall diversity (LAB-84).

The new default builds a fresh procedural wall per episode, seeded from
``(master_seed, episode_index)``. This exercises that path end-to-end — the part
the pytest suite skips (it pins ``generated_walls=False`` to stay CadQuery-free):

1. Three envs on consecutive episode wall seeds have *different* hole layouts.
2. A tiny ``generate_dataset(generated_walls=True)`` run writes distinct
   ``wall_seed`` per episode and stamps ``generated_wall=True``.

Run: uv run python scripts/dev/verify_wall_diversity.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ai_teleop.data import generate_dataset, load_episode
from ai_teleop.data.trajectory import episode_npz_path
from ai_teleop.sim.config import EnvConfig, episode_wall_seed
from ai_teleop.sim.env_setup import make_env

print("=== 1) three envs on consecutive episode wall seeds ===")
layouts = []
for episode_index in range(3):
    wall_seed = episode_wall_seed(0, episode_index)
    env = make_env(EnvConfig(wall_seed=wall_seed))
    holes = env.reset().hole_poses
    print(f"  episode {episode_index}: wall_seed={wall_seed}  holes={holes.shape[0]}")
    layouts.append(holes)
    env.close()
distinct = any(
    layouts[0].shape != layouts[i].shape or not np.allclose(layouts[0], layouts[i]) for i in (1, 2)
)
assert distinct, "expected distinct hole layouts across episodes"
print("  → hole layouts differ across episodes ✓")

print("\n=== 2) generate_dataset(generated_walls=True) stamps distinct wall_seed ===")
with tempfile.TemporaryDirectory() as tmp:
    runs = Path(tmp) / "runs"
    generate_dataset(tmp, n_episodes=2, seed=0, max_steps=60, baseline=False)
    seeds = []
    for episode_index in range(2):
        _, meta = load_episode(episode_npz_path(runs, episode_index))
        print(
            f"  episode {episode_index}: generated_wall={meta['generated_wall']}  "
            f"wall_seed={meta['wall_seed']}"
        )
        assert meta["generated_wall"] is True
        seeds.append(meta["wall_seed"])
    assert seeds[0] != seeds[1], "expected distinct wall seeds per episode"
print("  → per-episode wall_seed is distinct and stamped ✓")

print("\nWALL DIVERSITY OK")
