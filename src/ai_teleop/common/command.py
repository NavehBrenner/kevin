"""Command dataclass — the EE-pose setpoint a Controller consumes each tick.

Defined alongside `Observation` in `common/` because every command-producing
layer (the M2 dev harness, M3's scripted noisy human, M4+'s expert and
residual policy) hands one of these to the controller. Keeping it free of
sim-specific imports is what lets all those layers stay independent of
`ai_teleop.sim`.

## Conventions

- World frame at the robot base, z up — same as `Observation.ee_pose`.
- Quaternion layout `(w, x, y, z)`, unit norm — same as MuJoCo and `Observation`.
- All quantities in SI: metres, radians (via quaternion), newtons.

## Clamping

The controller (see `ai_teleop.control.backbone.Controller`) enforces the
following per-step clamps before the command reaches the impedance law:

- `|target_position − current_ee_position| ≤ 2 cm`
- angle between `target_quaternion` and current EE quaternion ≤ 10°
- `|delta_grip_force| ≤ 5 N`

These bounds protect the controller from a misbehaving upstream (be it a
scripted human, a stale value, or eventually a learned residual). See
`code/project-scope.md` *Residual policy interface* for the rationale —
the same clamps later protect M5's behavioral-cloning residual.

## Δgrip-force semantics

`delta_grip_force` is **additive** on top of a baseline closing force set
at scene reset, so 0.0 means "grip exactly as held at trial start". The
M2 controller plumbs and clamps the channel but the M2 harness does not
exercise it (see `docs/milestone-2-spec.md` *What's not in M2*).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Command:
    target_position: np.ndarray       # shape (3,)  world frame, metres
    target_quaternion: np.ndarray     # shape (4,)  world frame, (w, x, y, z), unit quat
    delta_grip_force: float = 0.0     # newtons, additive on top of baseline grip
