"""OCC authoring of the shared circular mouth lip."""
from __future__ import annotations

import math
import numpy as np

from ._occ import require_gmsh
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
        # Cubic Hermite spans share exact endpoint positions and tangents.
        # The fourth-derivative Hermite bound covers the complete interval.
        tags = [point(p) for p in controls]
        curve = occ.addBSpline(tags, degree=3, knots=z.tolist(),
                               multiplicities=[4]+[3]*(len(z)-2)+[4])
        return revolve(curve)
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
    outer_throat = model.body(0, True)[0]
    rear_point = np.array([-model.wall, outer_throat[1]])
    rear_return = revolve(occ.addLine(point(outer_throat), point(rear_point)))
    rear = [(2, occ.addDisk(0,0,-model.wall, float(rear_point[1]),float(rear_point[1])))]
    source = [(2, occ.addDisk(0,0,0,model.r0,model.r0))]
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
    """Bound every actual curved triangle using the torus Hessian.

    Interpolating a twice differentiable surface over a parameter triangle
    has error at most half its Hessian times the weighted parameter variance.
    Each coordinate variance is at most range squared / 4. The mixed covariance
    is at most the product of ranges / 4, giving the bound used below.
    """
    gmsh = require_gmsh()
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = np.asarray(xyz).reshape(-1,3)
    lookup = {int(tag):i for i,tag in enumerate(tags)}
    largest = 0.0
    for key, radius in (("lip_inner",model.radius),("lip_outer",model.radius-model.wall)):
        for face in built.mesh_surface_groups[key]:
            types, _, nodes = gmsh.model.mesh.getElements(2,face)
            for element_type, element_nodes in zip(types,nodes):
                if int(element_type) != 2:
                    raise ValueError("mouth roundover requires first-order triangle elements")
                indexes = np.fromiter((lookup[int(tag)] for tag in element_nodes),dtype=np.int64)
                vertices = xyz[indexes].reshape(-1,3,3)
                r = np.hypot(vertices[:,:,0],vertices[:,:,1])
                phi = np.arctan2(vertices[:,:,2]-model.center[0],model.center[1]-r)
                # At the terminal lip, tiny OCC roundoff can straddle +/-pi.
                # Use a local continuous parameter, as for azimuth below.
                phi = phi[:,:1]+(phi-phi[:,:1]+math.pi)%(2*math.pi)-math.pi
                az = np.arctan2(vertices[:,:,1],vertices[:,:,0])
                az = az[:,0,None]+(az-az[:,0,None]+math.pi)%(2*math.pi)-math.pi
                dp,da = np.ptp(phi,axis=1),np.ptp(az,axis=1)
                residual = np.max(abs(np.hypot(vertices[:,:,2]-model.center[0],r-model.center[1])-radius),axis=1)
                bound = (radius*dp*dp+2*radius*dp*da+(model.center[1]+radius)*da*da)/8+residual
                largest = max(largest,float(np.max(bound,initial=0)))
    if largest > 0.01:
        raise ValueError(f"mouth roundover mesh cannot certify 0.01 mm chord error (bound {largest:.6g} mm); reduce mouth resolution")
    built.metadata["mouthRoundoverMeshChordBoundMm"] = largest
