"""Exact OCC conic and circular surfaces with shared acoustic interfaces."""
import math

import numpy as np

from ._occ import require_gmsh
from ..geometry import BuiltGeometry,MESH_ALGORITHM_FRONTAL_DELAUNAY
from ..tags import PhysicalGroup


def build_arc(geometry):
    model=geometry.arc_meridian
    if (geometry.vertical_offset_mm != model.offset or not geometry.closed or geometry.symmetry_planes
            or geometry.build_mode.value != "bare" or geometry.wall_thickness_mm != 0
            or geometry.source_shape != 0 or geometry.source_curv != 0 or geometry.source_radius_mm != -1
            or geometry.source_auto_angle_deg != math.degrees(math.atan(model.throat_slope))):
        raise ValueError("terminating arc geometry must preserve its canonical placement, bare coverage and flat source")
    if (geometry.interfaces or geometry.interface_offset_mm != 0 or geometry.topology_mode != "acoustic"
            or geometry.preserve_grid is not False or geometry.surface_fit != "interpolate" or geometry.wg_topology is not True
            or geometry.freeform_axis_samples_mm is not None or geometry.freeform_report is not None):
        raise ValueError("terminating arc does not support direct interfaces, alternate topology, preserved grids or approximate fitting")
    gmsh=require_gmsh()
    occ=gmsh.model.occ
    cache={}
    def point(zr):
        key=tuple(float(v) for v in zr)
        if key not in cache:
            cache[key]=occ.addPoint(key[1],0,key[0])
        return cache[key]
    body=occ.addBSpline([point(row) for row in model.poles],degree=2,weights=model.weights.tolist(),
        knots=[0.,1.],multiplicities=[3,3])
    angles=[model.beta]
    if model.beta < math.pi/2 < model.end:
        angles.append(math.pi/2)
    angles.append(model.end)
    center=point((model.center_z,model.center_r))
    # The conic and circle start at the same exact stored join pole. Avoid
    # roundoff in a trigonometric reconstruction creating a second endpoint.
    endpoints=[point(model.poles[-1])]+[point(model.arc(phi)) for phi in angles[1:]]
    arcs=[occ.addCircleArc(endpoints[i],center,endpoints[i+1]) for i in range(len(angles)-1)]
    groups=[]
    for curves in ([body],arcs):
        surfaces=[]
        for quadrant in range(4):
            copied=occ.copy([(1,c) for c in curves])
            if quadrant:
                occ.rotate(copied,0,0,0,0,0,1,quadrant*math.pi/2)
            surfaces.extend((d,t) for d,t in occ.revolve(copied,0,0,0,0,0,1,math.pi/2) if d == 2)
        groups.append(surfaces)
    occ.remove([(1,c) for c in [body,*arcs]],recursive=True)
    source=[(2,occ.addDisk(0,0,0,model.r0,model.r0))]
    groups.append(source)
    flat=[item for group in groups for item in group]
    _,mapping=occ.fragment(flat[:-1],flat[-1:],removeObject=True,removeTool=True)
    mapped=[];cursor=0
    for group in groups:
        mapped.append(sorted({int(tag) for part in mapping[cursor:cursor+len(group)] for d,tag in part if d == 2}))
        cursor+=len(group)
    body_tags,arc_tags,source_tags=mapped
    if not body_tags or not arc_tags or not source_tags or set(body_tags+arc_tags)&set(source_tags):
        raise ValueError("terminating arc requires distinct nonempty wall and source descendants")
    occ.synchronize()
    # Orientation references come directly from the immutable analytic
    # authority, even if a direct API caller replaces a compatibility grid.
    z=(np.arange(8)+.5)*model.join_z/8
    body_points,derivatives=model.body(z)
    phi=model.beta+(np.arange(24)+.5)*(model.end-model.beta)/24
    meridian=np.vstack((body_points,model.arc(phi)))
    tangent=np.concatenate((np.arctan(derivatives[:,1]),phi))
    az=(np.arange(8)+.5)*2*math.pi/8
    refs=np.stack((meridian[:,1,None]*np.cos(az),meridian[:,1,None]*np.sin(az),
        np.broadcast_to(meridian[:,0,None],(len(meridian),len(az)))),axis=-1).reshape(-1,3)
    normals=np.stack((-np.cos(tangent)[:,None]*np.cos(az),-np.cos(tangent)[:,None]*np.sin(az),
        np.broadcast_to(np.sin(tangent)[:,None],(len(tangent),len(az)))),axis=-1).reshape(-1,3)
    # The Hessian certificate below, rather than the size target, is final.
    join_radius=float(model.poles[-1,1])
    span_cap=model.radius*(model.end-model.beta)/6
    cap=min(.8*math.sqrt(8*.004/(1/model.radius+3/join_radius)),span_cap)
    wall=sorted(set(body_tags+arc_tags))
    return BuiltGeometry(surface_groups={int(PhysicalGroup.RIGID_WALL):wall,int(PhysicalGroup.PRIMARY_SOURCE):source_tags},
        axial_bounds_mm=(0.,model.bounds[1][2]),source_axis="z",
        mesh_surface_groups={"inner":body_tags,"arc":arc_tags,"throat_disc":source_tags},
        mesh_algorithm=MESH_ALGORITHM_FRONTAL_DELAUNAY,open_shell_wall_points_mm=refs,open_shell_wall_normals=normals,
        open_shell_bore_surface_tags=tuple(body_tags),
        metadata={**model.metadata(),"terminatingArcMeshSizeMm":cap,
            "terminatingArcBodySurfaceTags":body_tags,
            "terminatingArcSizing":{"circleRadiusMm":model.radius,"centerRadiusMm":model.center_r,
                "joinRadiusMm":join_radius,"startAngleRad":model.beta,"endAngleRad":model.end,
                "spanCapMm":span_cap,"safetyFactor":.8,"sizingBudgetMm":.004}})


def certify_arc_mesh(model,built):
    """Bound each actual circle facet's distance to its analytic surface.

    The surface Hessian bounds interpolation over each unwrapped parameter
    triangle. This is a facet-to-surface certificate, not a coverage claim
    and not a certificate for the separately tessellated retained conic.
    """
    gmsh=require_gmsh()
    tags,xyz,_=gmsh.model.mesh.getNodes()
    xyz=np.asarray(xyz).reshape(-1,3)
    lookup={int(tag):i for i,tag in enumerate(tags)}
    largest=0.;count=0
    for face in built.mesh_surface_groups["arc"]:
        types,_,nodes=gmsh.model.mesh.getElements(2,face)
        for kind,element_nodes in zip(types,nodes):
            if int(kind) != 2:
                raise ValueError("terminating arc requires first-order triangle elements")
            indexes=np.fromiter((lookup[int(tag)] for tag in element_nodes),dtype=np.int64)
            vertices=xyz[indexes].reshape(-1,3,3)
            r=np.hypot(vertices[:,:,0],vertices[:,:,1])
            phi=np.arctan2(vertices[:,:,2]-model.center_z,model.center_r-r)
            phi=phi[:,:1]+(phi-phi[:,:1]+math.pi)%(2*math.pi)-math.pi
            # Project onto this finite suffix, rather than its extended circle.
            # Endpoint OCC roundoff then contributes to the actual residual.
            phi=np.clip(phi,model.beta,model.end)
            az=np.arctan2(vertices[:,:,1],vertices[:,:,0])
            az=az[:,:1]+(az-az[:,:1]+math.pi)%(2*math.pi)-math.pi
            dp,da=np.ptp(phi,axis=1),np.ptp(az,axis=1)
            if np.any(dp>=math.pi) or np.any(da>=math.pi):
                raise ValueError("terminating arc triangle has an ambiguous angular parameter chart; reduce mouth resolution")
            projected=model.arc(phi)
            guard=1e-10*max(1.,model.radius,abs(model.center_z),model.center_r,model.reach)
            residual=np.max(np.hypot(vertices[:,:,2]-projected[:,:,0],r-projected[:,:,1]),axis=1)+guard
            # Radius increases on [beta,end] within [0,pi]. Every convex
            # parameter triangle therefore has r <= r(max(phi)). Its angular
            # Hessian norm uses this local supremum, including upward roundoff.
            local_reach=np.nextafter(model.center_r-model.radius*np.cos(np.max(phi,axis=1))+guard,np.inf)
            bound=(model.radius*dp*dp+2*model.radius*dp*da+local_reach*da*da)/8+residual
            largest=max(largest,float(np.max(bound,initial=0)))
            count+=len(vertices)
    if not count or largest > .01:
        raise ValueError(f"terminating arc mesh cannot certify 0.01 mm facet-to-surface error (bound {largest:.6g} mm); reduce mouth resolution")
    built.metadata["terminatingArcMeshChordBoundMm"]=largest
    built.metadata["terminatingArcMeshCertificateScope"]="circle-facets-to-analytic-surface-only"
