"""Spot-check a generated episode: privileged ground truth lines up with sensors."""

from pathlib import Path
import numpy as np
from ai_teleop.data import load_episode

succ = Path("/tmp/m4_smoke/episode_00001.npz")  # a success
cols, meta = load_episode(succ)
print(
    "episode 1 meta:",
    {k: meta[k] for k in ("n_steps", "terminal_reason", "target_hole_index", "episode_success")},
)
d = cols["distance"]
ft = cols["wrist_ft"]
print(f"distance: start={d[0] * 1000:.0f}mm  min={d.min() * 1000:.0f}mm  end={d[-1] * 1000:.0f}mm")
print(
    f"|F| bias-subtracted: start={np.linalg.norm(ft[0, :3]):.2f}N  max={np.linalg.norm(ft[:, :3], axis=1).max():.2f}N"
)
# Expert delta should be ~0 early (far field) and nonzero near contact.
dpn = np.linalg.norm(cols["delta_position"], axis=1)
far = d > 0.1
near = d < 0.05
print(
    f"mean |Δpos| far-field (d>10cm): {dpn[far].mean() * 1000:.3f}mm   near (d<5cm): {dpn[near].mean() * 1000:.3f}mm"
)
print(
    f"step_success flips True at step: {int(np.argmax(cols['step_success'])) if cols['step_success'].any() else 'never'}"
)
