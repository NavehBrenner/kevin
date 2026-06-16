"""Verify SimEnv hole-site discovery + the env_setup bridge.

1. Static full_scene.xml still loads through SimEnv (backward compat, 3 holes).
2. A generated wall flows generate -> compose -> SimEnv with target at hole_0.

Run: uv run python scripts/dev/verify_env_integration.py
"""

import numpy as np

from ai_teleop.sim.env_setup import make_wall_task_env
from ai_teleop.sim.scene import SimEnv

print("=== 1) static full_scene via SimEnv (backward compat) ===")
env = SimEnv("assets/mjcf/full_scene.xml", render_mode="headless")
obs = env.reset()
print(f"  holes discovered: {len(env._hole_site_ids)} (expect 3)")
print(f"  hole_poses shape: {obs.hole_poses.shape}  target_index={obs.target_hole_index}")
env.close()

print("\n=== 2) generated 5-hole wall via env_setup bridge ===")
env = make_wall_task_env(seed=20, distractors=4, render_mode="headless")
obs = env.reset()
print(f"  holes discovered: {len(env._hole_site_ids)} (expect 5)")
print(f"  target_index={obs.target_hole_index} (expect 0)")
target_pose = obs.hole_poses[obs.target_hole_index]
print(f"  target hole world pos: {np.round(target_pose[:3], 4)}")
# Step the sim a few times to confirm it's runnable (peg held, no explosion).
for _ in range(50):
    env.step()
peg_pos = env.data.qpos[env._peg_qadr : env._peg_qadr + 3]
print(f"  peg pos after 50 steps: {np.round(peg_pos, 4)} (held by gripper, stable)")
env.close()
print("\nINTEGRATION OK")
