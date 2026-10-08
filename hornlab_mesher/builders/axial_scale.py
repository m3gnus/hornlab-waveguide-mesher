"""Author the certified absolute-axis meridian for both mesh and STEP."""
import math

import numpy as np

from ._occ import require_gmsh, make_planar_fill_from_boundary
from ..axial_scale import FIT_TOL_MM
from ..geometry import BuiltGeometry
from ..normals import open_shell_wall_orientation_references
from ..tags import PhysicalGroup


def build_axial(geometry):
    gmsh = require_gmsh()
    occ = gmsh.model.occ
    model = geometry.axial_model
    z = model.stations(FIT_TOL_MM, fit=True)
    positions, derivatives = model.body(z)
    controls = [positions[0]]
    for j, h in enumerate(np.diff(z)):
        controls.extend((positions[j]+h*derivatives[j]/3,
                         positions[j+1]-h*derivatives[j+1]/3, positions[j+1]))
    tags = [occ.addPoint(float(p[1]), 0, float(p[0])) for p in controls]
    curve = occ.addBSpline(tags, degree=3, knots=z.tolist(), multiplicities=[4]+[3]*(len(z)-2)+[4])
    wall = [(d,t) for d,t in occ.revolve([(1,curve)],0,0,0,0,0,1,2*math.pi) if d == 2]
    occ.synchronize()
    source = make_planar_fill_from_boundary(wall, source_axis="z", use_min=True, closed=True)
    if not source:
        raise ValueError("absolute-axis body could not share the source rim")
    occ.synchronize()
    wall_tags, source_tags = [t for _,t in wall], [t for _,t in source]
    points, normals = open_shell_wall_orientation_references(geometry.inner_points, closed=True)
    return BuiltGeometry(surface_groups={int(PhysicalGroup.RIGID_WALL):wall_tags,
        int(PhysicalGroup.PRIMARY_SOURCE):source_tags}, axial_bounds_mm=(0,model.length), source_axis="z",
        mesh_surface_groups={"inner":wall_tags, "throat_disc":source_tags},
        open_shell_wall_points_mm=points, open_shell_wall_normals=normals, metadata=model.metadata())
