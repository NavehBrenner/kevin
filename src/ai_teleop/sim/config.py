"""Env-level configuration — the metadata that defines one concrete task env.

A :class:`SimEnv` *is* its :class:`EnvConfig`: build an env from a config and
``reset()`` always returns that same env to t=0. The config is what varies an
episode — today only which procedural wall the env is built on (``wall_seed``),
with room to grow (e.g. simulated medium resistance, visibility) without
touching the reset contract.

Construction-time, not reset-time: the wall geometry is compiled into the
MuJoCo model at ``SimEnv.__init__`` (the seed is resolved to a scene *before*
the env exists), so a different wall means a different env instance, not a
``reset`` argument. ``env_setup.make_env`` turns a config into a runnable env.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnvConfig:
    """All metadata that defines one concrete task environment.

    ``wall_seed=None`` selects the static hand-authored ``full_scene.xml`` wall;
    an integer selects a procedurally generated wall (reproducible + disk-cached
    from that seed). The target hole is **not** here — which hole is the goal is
    an episode/task concept the task layer owns, not an env-physics one.
    """

    wall_seed: int | None = None


def episode_wall_seed(master_seed: int, episode_index: int) -> int:
    """Deterministic per-episode wall seed from ``(master_seed, episode_index)``.

    Mirrors the human-seed derivation so a dataset's walls are reproducible yet
    distinct per episode, and two datasets with different master seeds get
    different walls (no cache collisions on ``wall_<seed>/``).
    """
    return int(np.random.SeedSequence([master_seed, episode_index]).generate_state(1)[0])
