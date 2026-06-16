"""Manual eyeball of the sampler: defaults, explicit fields, distractor modes,
reproducibility, and the loud-failure paths. Generates one full wall to disk.

Run: uv run python scripts/dev/manual_test_sampler.py
"""

from ai_teleop.sim.scenegen import sample_wall_spec
from ai_teleop.sim.scenegen.generate import generate_wall


def show(title, spec):
    print(f"\n=== {title} ===")
    print(
        f"seed={spec.seed} given={spec.seed_was_given} wall={spec.wall_size} "
        f"n_holes={len(spec.holes)}"
    )
    for h in spec.holes:
        tag = "TARGET" if h.is_target else "distr "
        print(
            f"  [{tag}] shape={h.shape} pos=({h.pos[0]:+.3f},{h.pos[1]:+.3f}) "
            f"size={h.size} chamfer={h.chamfer * 1000:.1f}mm r={h.bounding_radius() * 1000:.1f}mm"
        )


# 1) Everything sampled from a fixed seed (distractors=None -> count in [0,10]).
show("all-sampled, seed=7", sample_wall_spec(seed=7))

# 2) Reproducibility: same seed -> identical spec.
a, b = sample_wall_spec(seed=7), sample_wall_spec(seed=7)
same = [h.pos for h in a.holes] == [h.pos for h in b.holes]
print(f"\nreproducible (seed=7 twice identical positions)? {same}")

# 3) Explicit true hole + fixed distractor count.
show(
    "explicit target + 3 distractors",
    sample_wall_spec(
        seed=1,
        true_hole={"pos": (0.0, 0.0), "size": {"diameter": 0.010}, "chamfer": 0.002},
        distractors=3,
    ),
)

# 4) Explicit list of distractor dicts (mixed: some posed, some sampled).
show(
    "explicit distractor list",
    sample_wall_spec(
        seed=2,
        true_hole={"pos": (-0.10, 0.10)},
        distractors=[{"pos": (0.10, -0.10), "size": {"diameter": 0.020}}, {}],
    ),
)  # second distractor fully sampled

# 5) Loud failure: two explicit holes overlapping.
print("\n=== overlap failure (expected) ===")
try:
    sample_wall_spec(
        seed=3,
        true_hole={"pos": (0.0, 0.0), "size": {"diameter": 0.020}},
        distractors=[{"pos": (0.005, 0.0), "size": {"diameter": 0.020}}],
    )
    print("  NO ERROR — BUG")
except ValueError as e:
    print(f"  raised ValueError: {e}")

# 6) Loud failure: explicit hole off the wall edge.
print("\n=== edge-margin failure (expected) ===")
try:
    sample_wall_spec(seed=4, true_hole={"pos": (0.199, 0.0)})
    print("  NO ERROR — BUG")
except ValueError as e:
    print(f"  raised ValueError: {e}")

# 7) Full generate to disk (default out_dir).
scene = generate_wall(seed=123, distractors=4)
print(
    f"\n=== generated to disk ===\n  mjcf={scene.mjcf_path}\n  "
    f"parts={len(scene.collision_mesh_paths)} holes={len(scene.spec.holes)}"
)
