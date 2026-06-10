"""Throwaway probe: confirm how CadQuery's `.faces("<X").workplane()` local
(x, y) coordinates map to world (Y, Z) when drilling offset holes, and check
that tessellation at mm-scale is clean. Build in mm.

Run: uv run python scripts/dev/probe_cadquery_hole_frame.py
"""

import cadquery as cq

# Plate: thickness along world X, 400 mm (Y) x 400 mm (Z) face. Built on "YZ"
# workplane so box(length->Y, width->Z, height->X-normal).
thickness, width, height = 20.0, 400.0, 400.0
plate = cq.Workplane("YZ").box(width, height, thickness)

# Drill one hole offset to +Y, +Z. If the face-workplane axes match world,
# the resulting hole centre should sit at world (Y=+120, Z=+80).
target_y, target_z = 120.0, 80.0
drilled = (
    plate.faces("<X").workplane()
    .pushPoints([(target_y, target_z)])
    .hole(10.0)
)

# Find the centre of the cylindrical hole face to read back its world coords.
cyl_faces = drilled.faces("%CYLINDER").vals()
print(f"num cylindrical faces (expect 1): {len(cyl_faces)}")
for f in cyl_faces:
    c = f.Center()
    print(f"  hole face centre world = ({c.x:.1f}, {c.y:.1f}, {c.z:.1f})  [thickness, Y, Z]")

# Chamfer only the robot-facing (<X) rim and confirm edge selection works.
chamfered = drilled.faces("<X").edges("%CIRCLE").chamfer(2.0)
print("chamfer on <X face: OK")

# Tessellate and report triangle count + bbox to sanity-check mm-scale meshing.
verts, tris = chamfered.val().tessellate(tolerance=0.2)
print(f"tessellation: {len(verts)} verts, {len(tris)} tris")
bb = chamfered.val().BoundingBox()
print(f"bbox (mm): x[{bb.xmin:.1f},{bb.xmax:.1f}] y[{bb.ymin:.1f},{bb.ymax:.1f}] z[{bb.zmin:.1f},{bb.zmax:.1f}]")
