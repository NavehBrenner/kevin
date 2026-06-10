"""Tests for the procedural wall generator.

Split in two: the sampler tests are pure-Python (numpy only) and always run;
the geometry round-trip needs the `scenegen` extra (CadQuery/shapely/trimesh)
and is skipped when that isn't installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_teleop.sim.scenegen import HoleSpec, sample_wall_spec

# --- Sampler (always runs) -----------------------------------------------

def test_seed_is_reproducible():
    a = sample_wall_spec(seed=42)
    b = sample_wall_spec(seed=42)
    assert [h.pos for h in a.holes] == [h.pos for h in b.holes]
    assert [h.size for h in a.holes] == [h.size for h in b.holes]


def test_distractor_count_modes():
    assert len(sample_wall_spec(seed=1, distractors=3).holes) == 4  # target + 3
    assert len(sample_wall_spec(seed=1, distractors=0).holes) == 1  # target only
    # None -> uniform[0, 10]; total is target + that count.
    assert 1 <= len(sample_wall_spec(seed=1).holes) <= 11


def test_target_is_first_and_flagged():
    spec = sample_wall_spec(seed=2, distractors=2)
    assert spec.holes[0].is_target
    assert not any(h.is_target for h in spec.holes[1:])
    assert spec.target_hole is spec.holes[0]


def test_explicit_fields_are_respected():
    spec = sample_wall_spec(
        seed=3,
        true_hole={"pos": (0.05, -0.03), "size": {"diameter": 0.012}, "chamfer": 0.002},
        distractors=0,
    )
    hole = spec.target_hole
    assert hole.pos == (0.05, -0.03)
    assert hole.size == {"diameter": 0.012}
    assert hole.chamfer == 0.002


def test_overlapping_explicit_holes_fail_loudly():
    with pytest.raises(ValueError, match="overlap"):
        sample_wall_spec(
            seed=4,
            true_hole={"pos": (0.0, 0.0), "size": {"diameter": 0.020}},
            distractors=[{"pos": (0.005, 0.0), "size": {"diameter": 0.020}}],
        )


def test_off_edge_explicit_hole_fails_loudly():
    with pytest.raises(ValueError, match="edge margin"):
        sample_wall_spec(seed=5, true_hole={"pos": (0.199, 0.0)}, distractors=0)


def test_sampled_holes_do_not_overlap():
    spec = sample_wall_spec(seed=7, distractors=8)
    centers = [np.asarray(h.pos) for h in spec.holes]
    radii = [h.bounding_radius() for h in spec.holes]
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            gap = np.linalg.norm(centers[i] - centers[j]) - radii[i] - radii[j]
            assert gap >= 0.0, f"holes {i},{j} overlap"


def test_bounding_radius_includes_chamfer():
    hole = HoleSpec("circle", (0.0, 0.0), {"diameter": 0.010}, chamfer=0.002)
    assert hole.bounding_radius() == pytest.approx(0.005 + 0.002)


# --- Geometry round-trip (needs the scenegen extra) ----------------------

@pytest.fixture(scope="module")
def mujoco_mod():
    return pytest.importorskip("mujoco")


def _build(tmp_path):
    pytest.importorskip("cadquery")
    pytest.importorskip("shapely")
    from ai_teleop.sim.scenegen import WallSpec
    from ai_teleop.sim.scenegen.generate import generate_from_spec

    spec = WallSpec(
        seed=1,
        wall_size=(0.02, 0.40, 0.40),
        holes=[HoleSpec("circle", (0.10, 0.05), {"diameter": 0.014}, 0.002, True)],
    )
    return generate_from_spec(spec, tmp_path)


def test_generated_wall_loads_and_has_sites(tmp_path, mujoco_mod):
    scene = _build(tmp_path)
    model = mujoco_mod.MjModel.from_xml_path(scene.mjcf_path)
    assert model.nsite == 1
    assert mujoco_mod.mj_name2id(model, mujoco_mod.mjtObj.mjOBJ_SITE, "hole_0") >= 0
    assert len(scene.collision_mesh_paths) > 1  # decomposed, not one filled hull


def test_generated_bore_is_open_and_wall_is_solid(tmp_path, mujoco_mod):
    scene = _build(tmp_path)
    model = mujoco_mod.MjModel.from_xml_path(scene.mjcf_path)
    data = mujoco_mod.MjData(model)
    mujoco_mod.mj_forward(model, data)

    def ray_hits(y, z):  # along +x through the wall body at (0.80, 0, 0.45)
        origin = np.array([0.70, y, 0.45 + z])
        geomid = np.array([-1], dtype=np.int32)
        return mujoco_mod.mj_ray(model, data, origin, np.array([1.0, 0, 0]),
                                 None, 1, -1, geomid) >= 0

    assert not ray_hits(0.10, 0.05)   # through the bore -> open
    assert ray_hits(-0.10, -0.10)     # through solid wall -> blocked
