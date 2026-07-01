"""Tests for the M4 trajectory schema + data-generation driver (LAB-28).

The schema round-trip is the contract M5 depends on; the driver smoke test runs
a couple of short episodes end-to-end and confirms the files load with the right
columns and metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ai_teleop.data import (
    COLUMN_SHAPES,
    SCHEMA_VERSION,
    EpisodeRecorder,
    generate_dataset,
    load_episode,
    regenerate_from_metadata,
)

SCENE_PATH = Path(__file__).resolve().parents[1] / "assets" / "mjcf" / "full_scene.xml"


def _synthetic_row(step: int) -> dict[str, object]:
    return {
        "step": step,
        "sim_time": step * 0.002,
        "wrist_ft": np.arange(6, dtype=float),
        "joint_positions": np.zeros(7),
        "joint_velocities": np.zeros(7),
        "ee_pose": np.array([0.4, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
        "gripper_width": 0.08,
        "cmd_position": np.array([0.5, 0.0, 0.45]),
        "cmd_quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
        "cmd_grip": 0.0,
        "delta_position": np.array([0.001, 0.0, 0.0]),
        "delta_orientation": np.zeros(3),
        "delta_grip": 0.0,
        "peg_pose": np.array([0.42, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
        "target_hole_pose": np.array([0.79, 0.0, 0.45, 1.0, 0.0, 0.0, 0.0]),
        "distance": 0.37,
        "step_success": False,
    }


# ---------------------------------------------------------------------------
# Schema round-trip + validation
# ---------------------------------------------------------------------------


def test_recorder_roundtrip(tmp_path):
    recorder = EpisodeRecorder()
    for step in range(10):
        recorder.add(**_synthetic_row(step))
    path = tmp_path / "episode_00000.npz"
    recorder.save(path, metadata={"master_seed": 0, "episode_index": 0})

    columns, metadata = load_episode(path)
    assert set(columns) == set(COLUMN_SHAPES)
    for name, per_step_shape in COLUMN_SHAPES.items():
        assert columns[name].shape == (10, *per_step_shape)
    np.testing.assert_array_equal(columns["step"], np.arange(10))
    assert metadata["schema_version"] == SCHEMA_VERSION
    assert metadata["n_steps"] == 10
    assert metadata["master_seed"] == 0


def test_recorder_rejects_missing_column():
    recorder = EpisodeRecorder()
    row = _synthetic_row(0)
    del row["wrist_ft"]
    with pytest.raises(ValueError, match="missing"):
        recorder.add(**row)


def test_recorder_rejects_wrong_shape():
    recorder = EpisodeRecorder()
    row = _synthetic_row(0)
    row["wrist_ft"] = np.zeros(3)  # should be (6,)
    with pytest.raises(ValueError, match="shape"):
        recorder.add(**row)


def test_recorder_refuses_empty_save(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        EpisodeRecorder().save(tmp_path / "x.npz", metadata={})


# ---------------------------------------------------------------------------
# Driver smoke test — full pipeline, short episodes
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SCENE_PATH.exists(), reason="scene file not found")
def test_generate_dataset_smoke(tmp_path):
    paths = generate_dataset(
        tmp_path, n_episodes=2, seed=0, max_steps=120, baseline=False, generated_walls=False
    )
    assert len(paths) == 2
    assert all(p.exists() for p in paths)
    # Each episode is its own folder runs/episode_NNNNN/{episode.npz, imgs/}.
    assert all(p.name == "episode.npz" for p in paths)
    assert all(p.parent.parent == tmp_path / "runs" for p in paths)
    assert all((p.parent / "imgs").is_dir() for p in paths)

    columns, metadata = load_episode(paths[0])
    assert set(columns) == set(COLUMN_SHAPES)
    assert 0 < metadata["n_steps"] <= 120
    assert metadata["terminal_reason"] in ("success", "force_abort", "timeout")
    assert metadata["episode_index"] == 0
    # Privileged distance is finite and positive.
    assert np.all(np.isfinite(columns["distance"]))


@pytest.mark.skipif(not SCENE_PATH.exists(), reason="scene file not found")
def test_generate_dataset_writes_layout_and_metadata(tmp_path):
    paths = generate_dataset(
        tmp_path, n_episodes=2, seed=0, max_steps=120, baseline=True, generated_walls=False
    )

    meta_path = tmp_path / "metadata.json"
    assert meta_path.exists()
    summary = json.loads(meta_path.read_text())
    assert summary["master_seed"] == 0
    assert summary["n_episodes"] == 2
    assert summary["schema_version"] == SCHEMA_VERSION
    assert set(summary["expert"]["counts"]) <= {"success", "force_abort", "timeout"}
    # Baseline ran, so an aggregate human-only rate is present (a float in [0, 1]).
    assert 0.0 <= summary["baseline_no_assist"]["success_rate"] <= 1.0
    assert "expert_lift" in summary
    assert len(summary["episodes"]) == 2

    # Per-episode trajectory metadata carries the paired baseline outcome...
    _, ep_meta = load_episode(paths[0])
    assert "baseline_terminal_reason" in ep_meta
    assert isinstance(ep_meta["baseline_success"], bool)
    # ...and the seeds the episode was generated with.
    assert ep_meta["scene_seed"] == [0, 0]  # [master_seed, episode_index]
    assert isinstance(ep_meta["human_seed"], int)
    # Seeds are surfaced in the dataset summary too.
    assert summary["episodes"][0]["human_seed"] == ep_meta["human_seed"]
    assert summary["episodes"][0]["scene_seed"] == [0, 0]


@pytest.mark.skipif(not SCENE_PATH.exists(), reason="scene file not found")
def test_generate_dataset_no_baseline_omits_baseline_stats(tmp_path):
    generate_dataset(
        tmp_path, n_episodes=1, seed=0, max_steps=80, baseline=False, generated_walls=False
    )
    summary = json.loads((tmp_path / "metadata.json").read_text())
    assert "baseline_no_assist" not in summary
    assert "expert_lift" not in summary


@pytest.mark.skipif(not SCENE_PATH.exists(), reason="scene file not found")
def test_regenerate_from_metadata_reproduces_episodes(tmp_path):
    # Only metadata.json is committed; regenerating from it must reproduce the
    # exact episodes (byte-identical columns) the original config implies.
    original = generate_dataset(
        tmp_path / "orig",
        n_episodes=2,
        seed=1,
        max_steps=100,
        baseline=False,
        generated_walls=False,
    )
    regenerated = regenerate_from_metadata(
        tmp_path / "orig" / "metadata.json", out_dir=tmp_path / "regen"
    )

    # Same episode folders (the filenames are all "episode.npz"; the per-episode
    # directory name is what distinguishes them).
    assert [p.parent.name for p in regenerated] == [p.parent.name for p in original]
    for orig_path, regen_path in zip(original, regenerated, strict=True):
        cols_orig, _ = load_episode(orig_path)
        cols_regen, _ = load_episode(regen_path)
        for column in COLUMN_SHAPES:
            np.testing.assert_array_equal(cols_orig[column], cols_regen[column])

    # The regenerated dataset carries the same fingerprint as the source metadata.
    src_meta = json.loads((tmp_path / "orig" / "metadata.json").read_text())
    regen_meta = json.loads((tmp_path / "regen" / "metadata.json").read_text())
    assert regen_meta["fingerprint"] == src_meta["fingerprint"]


@pytest.mark.skipif(not SCENE_PATH.exists(), reason="scene file not found")
def test_generate_dataset_is_reproducible(tmp_path):
    paths_a = generate_dataset(
        tmp_path / "a", n_episodes=1, seed=3, max_steps=80, baseline=False, generated_walls=False
    )
    paths_b = generate_dataset(
        tmp_path / "b", n_episodes=1, seed=3, max_steps=80, baseline=False, generated_walls=False
    )
    cols_a, _ = load_episode(paths_a[0])
    cols_b, _ = load_episode(paths_b[0])
    np.testing.assert_array_equal(cols_a["delta_position"], cols_b["delta_position"])
    np.testing.assert_array_equal(cols_a["ee_pose"], cols_b["ee_pose"])
