"""End-to-end proof of the extrude-based scenegen pipeline on one chamfered
round hole: CadQuery visual + analytic collision -> MJCF -> mj_loadXML, then
mj_ray validation that the bore is open and the surrounding wall is solid.

Run: uv run python scripts/dev/proof_scenegen_single_hole.py
"""

import time
from pathlib import Path

import mujoco
import numpy as np

from ai_teleop.sim.scenegen import HoleSpec, WallSpec
from ai_teleop.sim.scenegen.emit import WALL_BODY_POS
from ai_teleop.sim.scenegen.generate import generate_from_spec

OUT = Path("outputs/walls/_proof_single")

spec = WallSpec(
    seed=42,
    wall_size=(0.02, 0.40, 0.40),
    holes=[
        HoleSpec(
            shape="circle",
            pos=(0.10, 0.05),
            size={"diameter": 0.014},
            chamfer=0.002,
            is_target=True,
        )
    ],
)

t0 = time.perf_counter()
scene = generate_from_spec(spec, OUT)
gen_time = time.perf_counter() - t0
print(f"generation time: {gen_time * 1000:.0f} ms")
print(f"collision parts: {len(scene.collision_mesh_paths)}")

model = mujoco.MjModel.from_xml_path(scene.mjcf_path)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
print(f"ngeom={model.ngeom}  nmesh={model.nmesh}  nsite={model.nsite}")


def ray_hits(world_y: float, world_z: float) -> bool:
    """Cast a ray along +x at (y, z) through the wall; True if it hits a geom."""
    bx, _, bz = WALL_BODY_POS
    origin = np.array([bx - 0.10, world_y, bz + world_z])  # in front (robot side)
    direction = np.array([1.0, 0.0, 0.0])
    geomid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(model, data, origin, direction, None, 1, -1, geomid)
    return dist >= 0


bore_blocked = ray_hits(0.10, 0.05)  # straight through the hole centre
solid_blocked = ray_hits(-0.10, -0.10)  # through a solid region
print(f"ray through BORE blocked?  {bore_blocked}  (want False)")
print(f"ray through SOLID blocked? {solid_blocked}  (want True)")
print("VERDICT:", "GOOD" if (not bore_blocked and solid_blocked) else "BAD")
