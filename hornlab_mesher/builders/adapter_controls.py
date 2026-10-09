"""Exact native cubic/conic surfaces and whole-facet chart certificates."""
import math

import numpy as np

from ._occ import require_gmsh
from ..adapter_controls import validate_geometry
from ..geometry import BuiltGeometry
from ..tags import PhysicalGroup


def build_controls(geometry):
    validate_geometry(geometry)
    gmsh=require_gmsh();occ=gmsh.model.occ;m=geometry.controls_meridian
    points={}
    def point(zr):
        key=tuple(map(float,zr))
        if key not in points:points[key]=occ.addPoint(key[1],0,key[0])
        return points[key]
    cubic=(1,occ.addBezier([point(p) for p in m.cubic_poles]))
    conic=(1,occ.addBSpline([point(p) for p in m.body_poles],degree=2,weights=m.body_weights.tolist(),knots=[0.,1.],multiplicities=[3,3]))
    objects=[];branches=[]
    for name,curve in (("cubic",cubic),("body",conic)):
        for q in range(4):
            copied=occ.copy([curve])
            if q:occ.rotate(copied,0,0,0,0,0,1,q*math.pi/2)
            faces=[(d,t) for d,t in occ.revolve(copied,0,0,0,0,0,1,math.pi/2) if d==2]
            if len(faces)!=1:raise ValueError("exact controls revolution requires one face per branch quadrant")
            objects.extend(faces);branches.append(name)
    occ.remove([cubic,conic],recursive=True)
    source=occ.addDisk(0,0,0,m.source_radius,m.source_radius)
    _,ancestry=occ.fragment(objects,[(2,source)],removeObject=True,removeTool=True)
    tags={"cubic":[],"body":[]}
    for branch,part in zip(branches,ancestry[:-1]):tags[branch].extend(t for d,t in part if d==2)
    tags={k:sorted(set(v)) for k,v in tags.items()};source_tags=sorted({t for d,t in ancestry[-1] if d==2})
    if any(len(v)!=4 for v in tags.values()) or len(source_tags)!=1:
        raise ValueError("exact controls construction requires eight shared wall faces and one source disk")
    occ.synchronize()
    refs=[];normals=[];az=np.arange(32)*2*math.pi/32
    for branch in ("cubic","body"):
        p,d,_=m.branch(branch,np.linspace(0,1,65));d=d/np.linalg.norm(d,axis=-1)[:,None]
        refs.append(np.stack((p[:,1,None]*np.cos(az),p[:,1,None]*np.sin(az),np.broadcast_to(p[:,0,None],(len(p),len(az)))),axis=-1).reshape(-1,3))
        normals.append(np.stack((-d[:,0,None]*np.cos(az),-d[:,0,None]*np.sin(az),np.broadcast_to(d[:,1,None],(len(p),len(az)))),axis=-1).reshape(-1,3))
    wall=tags["cubic"]+tags["body"]
    return BuiltGeometry(surface_groups={int(PhysicalGroup.RIGID_WALL):wall,int(PhysicalGroup.PRIMARY_SOURCE):source_tags},
        axial_bounds_mm=(0,m.depth),source_axis="z",mesh_surface_groups={"inner":wall,"throat_disc":source_tags},
        open_shell_wall_points_mm=np.vstack(refs),open_shell_wall_normals=np.vstack(normals),
        metadata={**m.metadata(),"controlsBranchSurfaceTags":tags},controls_meridian=m)


def recover_parameter(m,branch,z):
    """A finite-domain projection; clipping never removes its vertex residual."""
    if branch=="body":return np.clip((z-m.adapter_length)/m.length,0,1)
    lo=np.zeros_like(z);hi=np.ones_like(z)
    for _ in range(56):
        mid=(lo+hi)/2
        left=m.cubic(mid)[0][...,0]<z
        lo=np.where(left,mid,lo);hi=np.where(left,hi,mid)
    return (lo+hi)/2


def certify_arrays(m,points,triangles,surfaces,branch_tags,*,check_normals=False):
    worst=0.;reports={}
    guard=1e-10*max(1.,m.depth,m.reach,abs(m.offset),float(np.max(np.abs(m.cubic_poles))))
    for branch in ("cubic","body"):
        mask=np.isin(surfaces,branch_tags[branch]);t=triangles[mask]
        if not len(t):raise ValueError("exact controls facet certificate is missing a branch")
        p=points[t];r=np.hypot(p[...,0],p[...,1])
        u=recover_parameter(m,branch,p[...,2])
        angle=np.arctan2(p[...,1],p[...,0]);angle=angle[:,0,None]+np.remainder(angle-angle[:,0,None]+math.pi,2*math.pi)-math.pi
        da=np.ptp(angle,axis=1);du=np.ptp(u,axis=1)
        if np.any(da>=math.pi):raise ValueError("exact controls facet angle chart is ambiguous")
        projected=m.branch(branch,u)[0]
        residual=np.max(np.hypot(p[...,2]-projected[...,0],r-projected[...,1]),axis=1)+guard
        Muu,Mr,rmax,_,_=m.interval_bounds(branch,u.min(axis=1),u.max(axis=1))
        bound=(Muu*du**2+2*Mr*du*da+rmax*da**2)/8+residual
        current=float(bound.max())
        if not np.all(np.isfinite(bound)) or current>.01:
            raise ValueError(f"exact controls {branch} facets exceed the 0.01 mm continuous geometric bound ({current:.8g}); refine the mm resolutions")
        reports[branch]={"triangleCount":len(t),"chordBoundMm":current,"vertexResidualBoundMm":float(residual.max())}
        worst=max(worst,current)
        if check_normals:
            center=p.mean(axis=1);u0=recover_parameter(m,branch,center[:,2]);d=m.branch(branch,u0)[1]
            az=np.arctan2(center[:,1],center[:,0]);speed=np.linalg.norm(d,axis=1)
            expected=np.column_stack((-d[:,0]*np.cos(az),-d[:,0]*np.sin(az),d[:,1]))/speed[:,None]
            normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);normal/=np.linalg.norm(normal,axis=1)[:,None]
            cosine=np.einsum("ij,ij->i",normal,expected)
            if np.min(cosine)<.90:raise ValueError("exact controls wall facets do not face their analytic branch bore")
            reports[branch]["minimumNormalCosine"]=float(cosine.min())
    return {"controlsMeshChordBoundMm":worst,"controlsMeshBranchCertificates":reports,
        "controlsMeshCertificateScope":"finite smooth branch facets to analytic surface; one-way; excludes join crease continuity"}


def certify_occ_mesh(m,built):
    gmsh=require_gmsh();node_tags,xyz,_=gmsh.model.mesh.getNodes()
    points=np.asarray(xyz).reshape(-1,3);order=np.argsort(node_tags);sorted_tags=np.asarray(node_tags)[order]
    faces=[];surfaces=[]
    for branch in ("cubic","body"):
        for face in built.metadata["controlsBranchSurfaceTags"][branch]:
            kinds,_,nodes=gmsh.model.mesh.getElements(2,face)
            for kind,raw in zip(kinds,nodes):
                if kind!=2:raise ValueError("exact controls requires first-order triangular facets")
                t=order[np.searchsorted(sorted_tags,np.asarray(raw))].reshape(-1,3)
                faces.append(t);surfaces.extend([face]*len(t))
    built.metadata.update(certify_arrays(m,points,np.vstack(faces),np.asarray(surfaces),built.metadata["controlsBranchSurfaceTags"]))


def configure_density(built,density):
    """Local derivative sizing is heuristic; publication uses the hard bound."""
    import gmsh
    from ..density import _record_and_check_triangle_estimate,_surface_area_mm2,TRIANGLES_PER_AREA_OVER_H2
    m=built.controls_meridian;tags=built.metadata["controlsBranchSurfaceTags"]
    fields=[];sizes=[];regions=[];records=[]
    # Small fixed intervals keep whole-interval bounds useful for unordered
    # polygons. Unresolved derivative hulls are subdivided before allocation.
    for branch in ("cubic","body"):
        intervals=[];pending=[(i/64,(i+1)/64,0) for i in range(64)]
        while pending:
            lo,hi,depth=pending.pop();Muu,Mr,rmax,rmin,zmin=m.interval_bounds(branch,lo,hi)
            if min(rmin,zmin)<=1e-4:
                if depth>=20 or len(intervals)+len(pending)>1024:raise ValueError("controls local sizing exceeds its certificate work budget")
                mid=(lo+hi)/2;pending.extend(((lo,mid,depth+1),(mid,hi,depth+1)));continue
            speed=max(float(zmin),float(m.sizing_speed_bound(branch,lo,hi)))
            coefficient=float(Muu/speed**2+2*Mr/(speed*rmin)+rmax/rmin**2)
            target=min(density.throat_res_mm+(density.mouth_res_mm-density.throat_res_mm)*lo,.7*math.sqrt(8*.003/coefficient))
            p0,p1=m.branch(branch,np.asarray([lo,hi]))[0]
            intervals.append((float(p0[0]),float(p1[0]),target,lo,hi))
        intervals.sort();sizes.extend(row[2] for row in intervals)
        # Piecewise constants apply a local minimum on the whole branch. The
        # value also extends to its boundary, preserving shared curve sizing.
        boxes=[]
        for loz,hiz,h,_,_ in intervals:
            box=gmsh.model.mesh.field.add("Box")
            for key,value in {"VIn":h,"VOut":1e22,"XMin":-100000,"XMax":100000,
                "YMin":-100000,"YMax":100000,"ZMin":loz,"ZMax":hiz,"Thickness":0}.items():
                gmsh.model.mesh.field.setNumber(box,key,value)
            boxes.append(box)
        field=gmsh.model.mesh.field.add("Min");gmsh.model.mesh.field.setNumbers(field,"FieldsList",boxes)
        restrict=gmsh.model.mesh.field.add("Restrict");gmsh.model.mesh.field.setNumber(restrict,"InField",field)
        gmsh.model.mesh.field.setNumbers(restrict,"SurfacesList",tags[branch])
        curves=sorted({abs(t) for f in tags[branch] for d,t in gmsh.model.getBoundary([(2,f)],combined=False,oriented=False) if d==1})
        gmsh.model.mesh.field.setNumbers(restrict,"CurvesList",curves);fields.append(restrict)
        # Positive quadrature estimates area-weighted field cost. Counts remain
        # estimates and are never substituted for the actual budget check.
        total=0.
        for _,_,h,lo,hi in intervals:
            nodes,weights=np.polynomial.legendre.leggauss(8);u=lo+(nodes+1)*(hi-lo)/2
            p,d,_=m.branch(branch,u);area=math.pi*(hi-lo)*float(np.dot(weights,p[:,1]*np.linalg.norm(d,axis=1)))
            total+=area/(h*h)
        area=_surface_area_mm2(tags[branch]);effective=math.sqrt(area/total)
        regions.append((area,effective,branch+" exact wall"));records.append({"branch":branch,"intervalCount":len(intervals),"minimumSizeMm":min(row[2] for row in intervals),"effectiveAreaSizeMm":effective})
    source=built.mesh_surface_groups["throat_disc"]
    field=gmsh.model.mesh.field.add("MathEval");gmsh.model.mesh.field.setString(field,"F",str(density.throat_res_mm))
    restrict=gmsh.model.mesh.field.add("Restrict");gmsh.model.mesh.field.setNumber(restrict,"InField",field)
    gmsh.model.mesh.field.setNumbers(restrict,"SurfacesList",source);fields.append(restrict)
    regions.append((_surface_area_mm2(source),density.throat_res_mm,"source disk"))
    minimum=gmsh.model.mesh.field.add("Min");gmsh.model.mesh.field.setNumbers(minimum,"FieldsList",fields);gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
    gmsh.option.setNumber("Mesh.MeshSizeMin",min(sizes+[density.throat_res_mm])*.25)
    gmsh.option.setNumber("Mesh.MeshSizeMax",max(sizes+[density.throat_res_mm,density.mouth_res_mm]))
    for name in ("Mesh.MeshSizeExtendFromBoundary","Mesh.MeshSizeFromPoints","Mesh.MeshSizeFromCurvature"):gmsh.option.setNumber(name,0)
    built.metadata["controlsLocalSizing"]={"method":"whole-interval-derivative-target; heuristic","intervals":records,"targetBudgetMm":.003,"safetyFactor":.7}
    _record_and_check_triangle_estimate(built,density,regions)
