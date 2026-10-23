"""OCC authoring of the shared circular mouth lip."""
from __future__ import annotations

import math
import numpy as np

from ._occ import require_gmsh
from .circle_certificate import certify_circle_facets
from ..composite_meridian import CircleArc
from ..geometry import BuiltGeometry, MESH_ALGORITHM_FRONTAL_DELAUNAY
from ..mouth_roundover import FIT_TOL_MM
from ..tags import PhysicalGroup


def build_roundover(geometry):
    gmsh = require_gmsh()
    occ = gmsh.model.occ
    model = geometry.roundover
    def point(zr):
        return occ.addPoint(float(zr[1]), 0, float(zr[0]))
    def revolve(curve):
        return [(d,t) for d,t in occ.revolve([(1,curve)], 0,0,0, 0,0,1, 2*math.pi) if d == 2]
    def body(outer):
        z = model.fit_stations(outer)
        positions, derivatives = model.body(z, outer)
        controls = [positions[0]]
        for j, h in enumerate(np.diff(z)):
            controls.extend((positions[j]+h*derivatives[j]/3,
                             positions[j+1]-h*derivatives[j+1]/3, positions[j+1]))
        if model.throat is not None and not outer:
            # One shared join point, the cubic's exact end; the fit spends the gap.
            controls[0] = model.throat.cubic[-1]
        # Cubic Hermite spans share exact endpoint positions and tangents.
        # The fourth-derivative Hermite bound covers the complete interval.
        tags = [point(p) for p in controls]
        curve = occ.addBSpline(tags, degree=3, knots=z.tolist(),
                               multiplicities=[4]+[3]*(len(z)-2)+[4])
        if model.throat is None:
            return revolve(curve)
        if outer:
            # The certified Hermite fit of the offset cubic ends on the outer
            # conic's first point; the fit spends that gap too.
            u = model.cubic_fit_stations()
            positions, derivatives = model.cubic(u, True)
            positions[-1] = controls[0]
            spans = [positions[0]]
            for j, h in enumerate(np.diff(u)):
                spans.extend((positions[j]+h*derivatives[j]/3,
                              positions[j+1]-h*derivatives[j+1]/3, positions[j+1]))
            throat = occ.addBSpline([point(p) for p in spans[:-1]]+tags[:1], degree=3, knots=u.tolist(),
                                    multiplicities=[4]+[3]*(len(u)-2)+[4])
        else:
            throat = occ.addBezier([point(p) for p in model.throat.cubic[:-1]]+tags[:1])
        return revolve(throat)+revolve(curve)
    def lip(outer):
        split = math.pi/2-model.beta
        angles = [0, split, model.sweep]
        center = point(model.center)
        endpoints = [point(model.arc(p, outer)) for p in angles]
        result = []
        for i in range(2):
            result.extend(revolve(occ.addCircleArc(endpoints[i], center, endpoints[i+1])))
        return result
    inner = body(False)
    outer = body(True)
    inner_lip = lip(False)
    outer_lip = lip(True)
    rim = revolve(occ.addLine(point(model.arc(model.sweep)), point(model.arc(model.sweep, True))))
    outer_throat = model.rear_start
    rear_point = np.array([-model.wall, outer_throat[1]])
    rear_return = revolve(occ.addLine(point(outer_throat), point(rear_point)))
    rear = [(2, occ.addDisk(0,0,-model.wall, float(rear_point[1]),float(rear_point[1])))]
    source = [(2, occ.addDisk(0,0,0,model.source_radius,model.source_radius))]
    groups = [inner, outer, inner_lip, outer_lip, rim, rear_return+rear, source]
    flat = [item for group in groups for item in group]
    _, mapping = occ.fragment(flat[:-1], flat[-1:])
    occ.synchronize()
    mapped = []
    cursor = 0
    for group in groups:
        mapped.append(sorted({int(t) for parts in mapping[cursor:cursor+len(group)] for d,t in parts if d == 2}))
        cursor += len(group)
    inside, outside, inner_lips, outer_lips, rim_tags, rear_tags, source_tags = mapped
    lips = inner_lips+outer_lips
    rigid = sorted(set(inside+outside+lips+rim_tags+rear_tags))
    reach = model.center[1]+model.radius
    # Triangle chord error spends half its budget in each principal direction.
    # The smaller lip radius is the larger meridian curvature.
    cap = min(math.sqrt(8*(model.radius-model.wall)*0.004), math.sqrt(8*reach*0.004),
              (model.radius-model.wall)*model.sweep/6)
    return BuiltGeometry(surface_groups={int(PhysicalGroup.RIGID_WALL):rigid,
                         int(PhysicalGroup.PRIMARY_SOURCE):source_tags},
        axial_bounds_mm=(-model.wall, float(model.center[0]+model.radius)), source_axis="z",
        mesh_surface_groups={"inner":inside, "outer":outside, "lip":lips+rim_tags,
                             "lip_inner":inner_lips, "lip_outer":outer_lips,
                             "rear":rear_tags, "throat_disc":source_tags},
        mesh_algorithm=MESH_ALGORITHM_FRONTAL_DELAUNAY,
        metadata={**model.metadata(), "mouthRoundoverMeshSizeMm":float(cap)})


def certify_lip_mesh(model, built):
    """Bound every actual curved lip triangle against its analytic torus."""
    groups = [(CircleArc(float(model.center[0]), float(model.center[1]), radius, model.beta, math.pi),
               built.mesh_surface_groups[key])
              for key, radius in (("lip_inner", model.radius), ("lip_outer", model.radius-model.wall))]
    largest, _ = certify_circle_facets(groups, "mouth roundover")
    if largest > 0.01:
        raise ValueError(f"mouth roundover mesh cannot certify 0.01 mm chord error (bound {largest:.6g} mm); reduce mouth resolution")
    built.metadata["mouthRoundoverMeshChordBoundMm"] = largest
