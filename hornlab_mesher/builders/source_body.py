"""Exact OCC faces for a closed exterior source body."""
import math

from ..geometry import BuiltGeometry
from ..source_body import StandaloneSourceGeometry


def build(model):
    import gmsh
    from ..mesher import MesherError
    if type(model) is not StandaloneSourceGeometry:
        raise TypeError("an explicit StandaloneSourceGeometry is required")
    r,R,d = model.radius_mm,model.outer_radius_mm,model.depth_mm
    volume=gmsh.model.occ.addCylinder(0,0,-d,0,0,d,R)
    disk=gmsh.model.occ.addDisk(0,0,0,r,r)
    _,fragment_map=gmsh.model.occ.fragment([(3,volume)],[(2,disk)])
    gmsh.model.occ.synchronize()
    volumes=gmsh.model.getEntities(3)
    if len(volumes)!=1:
        raise MesherError("standalone source requires exactly one OCC volume")
    faces=[abs(t) for _,t in gmsh.model.getBoundary(volumes,oriented=False)]
    # The second input is the moving disk. Its fragment mapping identifies
    # that exact face even when the surrounding annulus has the same area.
    source=[abs(t) for dim,t in fragment_map[1] if dim==2] if len(fragment_map)==2 else []
    if len(faces)!=4 or len(source)!=1 or source[0] not in faces:
        raise MesherError("standalone source OCC faces could not be classified")
    face=source[0]
    center=gmsh.model.occ.getCenterOfMass(2,face)
    if (gmsh.model.getType(2,face)!="Plane" or any(abs(v)>1e-8 for v in center)
            or abs(gmsh.model.occ.getMass(2,face)-math.pi*r*r)>=1e-7*r*r):
        raise MesherError("standalone source mapped face is not the declared front disk")
    rim=[abs(t) for dim,t in gmsh.model.getBoundary([(2,face)],oriented=False) if dim==1]
    if (len(rim)!=1 or gmsh.model.getType(1,rim[0]) not in {"Circle","Ellipse"}
            or abs(gmsh.model.occ.getMass(1,rim[0])-2*math.pi*r)>=1e-7*r
            or any(abs(v)>1e-8 for v in gmsh.model.occ.getCenterOfMass(1,rim[0]))):
        raise MesherError("standalone source disk does not have its declared circular rim")
    # OCC may name an equal-radii disk boundary Ellipse. Check its geometry,
    # rather than depending on the curve representation's type name.
    lo,hi=gmsh.model.getParametrizationBounds(1,rim[0])
    samples=gmsh.model.getValue(1,rim[0],[float(lo[0])+(float(hi[0])-float(lo[0]))*i/8 for i in range(8)])
    if any(abs(math.hypot(samples[i],samples[i+1])-r)>=1e-7*r or abs(samples[i+2])>=1e-8
            for i in range(0,len(samples),3)):
        raise MesherError("standalone source rim is not the declared front circle")
    shared=[t for t in faces if t!=face and rim[0] in
        {abs(edge) for dim,edge in gmsh.model.getBoundary([(2,t)],oriented=False) if dim==1}]
    annulus_area=math.pi*(R*R-r*r)
    if (len(shared)!=1 or gmsh.model.getType(2,shared[0])!="Plane"
            or abs(gmsh.model.occ.getCenterOfMass(2,shared[0])[2])>=1e-8
            or abs(gmsh.model.occ.getMass(2,shared[0])-annulus_area)>=1e-7*annulus_area):
        raise MesherError("standalone source disk must share its rim with the front annulus")
    rigid=sorted(set(faces)-set(source))
    return BuiltGeometry({1:rigid,2:source},(-d,0.0),source_axis="z",
        mesh_surface_groups={"source_body.disk":source,"source_body.rigid":rigid},mesh_algorithm=6,
        metadata=model.metadata())
