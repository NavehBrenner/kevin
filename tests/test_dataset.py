"""Tests for the M4 trajectory schema + data-generation driver (LAB-28).

The schema round-trip is the contract M5 depends on; the driver smoke test runs
a couple of short episodes end-to-end and confirms the files load with the right
columns and metadata.
"""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from ai_teleop.data import (
    COLUMN_SHAPES,
    SCHEMA_VERSION,
    EpisodeRecorder,
    GenerationConfig,
    generate_dataset,
    load_episode,
    regenerate_from_metadata,
)
from ai_teleop.data.generate import _UNFINGERPRINTED

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
        tmp_path,
        n_episodes=2,
        config=GenerationConfig(max_steps=120, generated_walls=False),
        baseline=False,
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
        tmp_path,
        n_episodes=2,
        config=GenerationConfig(max_steps=120, generated_walls=False),
        baseline=True,
    )

    meta_path = tmp_path / "metadata.json"
    assert meta_path.exists()
    summary = json.loads(meta_path.read_text(encoding="utf-8"))
    assert summary["master_seed"] == 0
    assert summary["n_episodes"] == 2
    assert summary["schema_version"] == SCHEMA_VERSION
    assert set(summary["expert"]["counts"]) <= {"success", "force_abort", "timeout"}
    # Baseline ran, so an aggregate human-only rate is present (a float in [0, 1]).
    assert 0.0 <= summary["baseline_no_assist"]["success_rate"] <= 1.0
    assert "expert_lift" in summary
    assert len(summary["episodes"]) == 2

    # The corpus config (incl. the LAB-96 deployment-controller + speed-draw
    # knobs) is echoed into metadata.json so the dataset regenerates from it.
    config = summary["config"]
    assert config["max_dpos"] == 0.3 and config["joint_damping"] == 1.5
    assert config["speed_lognormal_median"] == pytest.approx(0.09)
    assert config["speed_lognormal_sigma"] == pytest.approx(0.76)

    # Per-episode trajectory metadata carries the paired baseline outcome...
    _, ep_meta = load_episode(paths[0])
    assert "baseline_terminal_reason" in ep_meta
    assert isinstance(ep_meta["baseline_success"], bool)
    # ...the controller/operator config replay rebuilds from (LAB-96)...
    assert ep_meta["joint_damping"] == 1.5
    assert ep_meta["speed_lognormal_median"] == pytest.approx(0.09)
    assert ep_meta["speed_lognormal_sigma"] == pytest.approx(0.76)
    # ...and the seeds the episode was generated with.
    assert ep_meta["scene_seed"] == [0, 0]  # [master_seed, episode_index]
    assert isinstance(ep_meta["human_seed"], int)
    # Seeds are surfaced in the dataset summary too.
    assert summary["episodes"][0]["human_seed"] == ep_meta["human_seed"]
    assert summary["episodes"][0]["scene_seed"] == [0, 0]


@pytest.mark.skipif(not SCENE_PATH.exists(), reason="scene file not found")
def test_generate_dataset_no_baseline_omits_baseline_stats(tmp_path):
    generate_dataset(
        tmp_path,
        n_episodes=1,
        config=GenerationConfig(max_steps=80, generated_walls=False),
        baseline=False,
    )
    summary = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert "baseline_no_assist" not in summary
    assert "expert_lift" not in summary


@pytest.mark.skipif(not SCENE_PATH.exists(), reason="scene file not found")
def test_regenerate_from_metadata_reproduces_episodes(tmp_path):
    # Only metadata.json is committed; regenerating from it must reproduce the
    # exact episodes (byte-identical columns) the original config implies.
    original = generate_dataset(
        tmp_path / "orig",
        n_episodes=2,
        config=GenerationConfig(seed=1, max_steps=100, generated_walls=False),
        baseline=False,
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
    src_meta = json.loads((tmp_path / "orig" / "metadata.json").read_text(encoding="utf-8"))
    regen_meta = json.loads((tmp_path / "regen" / "metadata.json").read_text(encoding="utf-8"))
    assert regen_meta["fingerprint"] == src_meta["fingerprint"]


def test_fingerprint_of_legacy_config_is_unchanged():
    # Pre-LAB-96 metadata carries no joint_damping / speed_lognormal_* keys, and
    # its committed fingerprint was hashed over the old five-field payload. The
    # legacy config (kd=4.0, speed draw disabled) must keep producing that exact
    # hash, or every committed pre-LAB-96 manifest would spuriously mismatch on
    # regeneration. Pinned to data/dataset_6's committed fingerprint.
    dataset_6 = GenerationConfig(
        seed=6,
        max_steps=6000,
        max_dpos=0.025,
        expert_d_far=0.1,
        generated_walls=True,
        expert_brake_gain=0.0,
        delta_clamp=0.02,
        speed_lognormal_sigma=0.76,
    )
    legacy = replace(dataset_6, joint_damping=4.0, speed_lognormal_median=0.0).fingerprint()
    assert legacy == "b8dafbe9171f768f"
    # And the LAB-96 knobs do enter the hash once they leave the legacy config.
    assert (
        replace(dataset_6, joint_damping=1.5, speed_lognormal_median=0.09).fingerprint() != legacy
    )


def test_fingerprint_of_legacy_delta_clamp_is_unchanged():
    # Pre-LAB-100 metadata carries no delta_clamp key — those corpora were
    # clamped at the then-module-wide ±2 cm bound. The legacy bound must keep
    # producing the exact committed hash (pinned to data/dataset_8's), and the
    # knob must enter the hash once it leaves the legacy value.
    dataset_8 = GenerationConfig(
        seed=8,
        max_steps=6000,
        max_dpos=0.3,
        expert_d_far=0.15,
        generated_walls=True,
        joint_damping=1.5,
        speed_lognormal_median=0.09,
        speed_lognormal_sigma=0.76,
        expert_brake_gain=1.0,
        expert_brake_lead_floor=0.008,
    )
    assert replace(dataset_8, delta_clamp=0.02).fingerprint() == "492c9509df3c11cb"
    assert replace(dataset_8, delta_clamp=0.03).fingerprint() != "492c9509df3c11cb"


def test_fingerprint_covers_every_config_field():
    # The point of GenerationConfig: a knob that changes trajectories must change
    # the fingerprint, or a regenerated dataset silently differs from its metadata.
    # Perturbing any field flips the hash — except the three termination thresholds,
    # which are a known, documented hole (see `_UNFINGERPRINTED`). Adding a field
    # without hashing it fails here.
    base = GenerationConfig()
    baseline_hash = base.fingerprint()
    for field in fields(GenerationConfig):
        current = getattr(base, field.name)
        perturbed = (not current) if isinstance(current, bool) else current + 1
        changed = replace(base, **{field.name: perturbed}).fingerprint() != baseline_hash
        assert changed is (field.name not in _UNFINGERPRINTED), field.name


def test_generation_config_round_trips_through_metadata():
    # `from_metadata` must invert `to_dataset_config` — this is what makes a
    # committed manifest regenerate the corpus it describes.
    config = GenerationConfig(seed=5, max_steps=1234, lateral_tolerance=0.006)
    metadata = {"master_seed": config.seed, "config": config.to_dataset_config()}
    assert GenerationConfig.from_metadata(metadata) == config  # type: ignore[arg-type]


@pytest.mark.skipif(not SCENE_PATH.exists(), reason="scene file not found")
def test_generate_dataset_is_reproducible(tmp_path):
    config = GenerationConfig(seed=3, max_steps=80, generated_walls=False)
    paths_a = generate_dataset(tmp_path / "a", n_episodes=1, config=config, baseline=False)
    paths_b = generate_dataset(tmp_path / "b", n_episodes=1, config=config, baseline=False)
    cols_a, _ = load_episode(paths_a[0])
    cols_b, _ = load_episode(paths_b[0])
    np.testing.assert_array_equal(cols_a["delta_position"], cols_b["delta_position"])
    np.testing.assert_array_equal(cols_a["ee_pose"], cols_b["ee_pose"])
