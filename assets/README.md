# assets/

Static project assets — MJCF (MuJoCo) scene files, meshes, textures.

```
mjcf/
├── menagerie_panda/    # Franka Panda + Franka Hand from mujoco_menagerie (vendored)
├── wall_with_holes.xml # Custom: vertical wall, three through-openings (chamfer pending M2)
├── peg.xml             # Custom: 8 mm × 60 mm cylindrical peg
└── full_scene.xml      # Combined: panda + wall + peg + ambient/wrist lights + wrist camera + weld
```

Mesh files live flat inside `mjcf/menagerie_panda/` (originally a sub-folder
in the upstream repo; flattened so MuJoCo's `<include>` resolves them correctly
from `full_scene.xml`).

The vendored panda XML has project-local extras inlined into the `hand` body:
a wrist camera (`wrist_cam`), a wrist spot light (`wrist_light`), an F/T
measurement site (`wrist_site`), a TCP reference site (`tcp_site`), plus
`force`/`torque` sensors on the wrist site.

The full-scene `home` keyframe places the arm so the wrist camera looks
horizontally at the wall; the peg's free-joint pose is pre-computed so the
peg-grasp weld is satisfied at t=0.
