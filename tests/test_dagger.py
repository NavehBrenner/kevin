"""DAgger rollout-relabel + aggregation smoke (LAB-105).

Exercises the *new* mechanism end-to-end on the static scene (no CadQuery, no
render, no torch): that on-policy rollout records the **label provider's** Δ (not
the acting policy's), and that the aggregated corpus loads back through the real
training dataloader. The expensive vision rounds are a deliberate CLI run, not a
test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ai_teleop.dagger import append_summaries, rollout_and_relabel, seed_aggregate
from ai_teleop.data import build_dataloaders
from ai_teleop.data.trajectory import load_episode
from ai_teleop.domain import Delta

# Short static-scene episodes — fast and dependency-light.
_CONFIG = {"max_steps": 80, "success_depth": 0.015, "lateral_tolerance": 0.01, "force_cap": 50.0}


class _ConstantAssist:
    """An ``AssistProvider`` that returns a fixed Δ — lets the test tell the acting
    policy's output apart from the recorded (relabeled) target."""

    def __init__(self, delta: Delta) -> None:
        self._delta = delta

    def get_delta(self, observation, command) -> Delta:  # noqa: ANN001 - test double
        return self._delta

    def reset(self) -> None:
        pass


def test_relabel_records_label_not_acting_delta(tmp_path: Path) -> None:
    acting = Delta(
        delta_position=np.array([0.01, 0.0, 0.0]),
        delta_orientation=np.zeros(3),
        delta_grip_force=0.0,
    )
    label = Delta(
        delta_position=np.array([0.001, -0.002, 0.003]),
        delta_orientation=np.array([0.0, 0.01, 0.0]),
        delta_grip_force=-0.5,
    )
    runs_dir = tmp_path / "runs"

    summary = rollout_and_relabel(
        policy=_ConstantAssist(acting),
        expert=_ConstantAssist(label),  # stand-in label provider
        runs_dir=runs_dir,
        dagger_index=0,
        master_seed=105,
        rollout_index=0,
        config=_CONFIG,
        render_every=None,  # F/T-only: no capture, no frames
        generated_walls=False,
    )

    assert summary["n_steps"] > 0
    columns, metadata = load_episode(runs_dir / "episode_00000" / "episode.npz")
    assert metadata["source"] == "dagger"
    # Every recorded row carries the *label* provider's Δ, not the acting policy's.
    assert np.allclose(columns["delta_position"], label.delta_position)
    assert np.allclose(columns["delta_orientation"], label.delta_orientation)
    assert not np.allclose(columns["delta_position"], acting.delta_position)


def test_aggregate_unions_and_loads(tmp_path: Path) -> None:
    label = Delta(
        delta_position=np.array([0.002, 0.0, 0.001]),
        delta_orientation=np.zeros(3),
        delta_grip_force=0.0,
    )
    policy = _ConstantAssist(label)

    # A tiny "seed corpus": three relabeled episodes + a minimal manifest.
    base = tmp_path / "base"
    base_summaries = [
        rollout_and_relabel(
            policy=policy,
            expert=policy,
            runs_dir=base / "runs",
            dagger_index=i,
            master_seed=105,
            rollout_index=i,
            config=_CONFIG,
            render_every=None,
            generated_walls=False,
        )
        for i in range(3)
    ]
    (base / "metadata.json").write_text(
        json.dumps({"schema_version": "2.0", "n_episodes": 3, "episodes": base_summaries})
    )

    # Seed the aggregate (symlinks the base episodes in), add one DAgger episode.
    aggregate = tmp_path / "agg"
    summaries = seed_aggregate(base, aggregate)
    assert len(summaries) == 3
    assert (aggregate / "runs" / "episode_00000").is_symlink()

    summaries.append(
        rollout_and_relabel(
            policy=policy,
            expert=policy,
            runs_dir=aggregate / "runs",
            dagger_index=1_000_000,
            master_seed=105,
            rollout_index=0,
            config=_CONFIG,
            render_every=None,
            generated_walls=False,
        )
    )
    append_summaries(aggregate, summaries)

    manifest = json.loads((aggregate / "metadata.json").read_text())
    assert manifest["n_episodes"] == 4

    # The real training dataloader reads the aggregated corpus (F/T-only path).
    train_loader, val_loader, _ = build_dataloaders(
        aggregate, batch_size=1, val_fraction=0.25, load_images=False, download=False
    )
    assert len(train_loader.dataset) + len(val_loader.dataset) == 4
