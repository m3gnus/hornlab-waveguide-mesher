"""An explicit closed exterior body with one normal-driven circular disk."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from .config_parser import ConfigError

FORMULA = "SOURCE-DISK"
MODE = "standalone-source"
HARD_TRIANGLE_LIMIT = 200_000


def _number(value, name):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite scalar number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite scalar number")
    return number


@dataclass(frozen=True)
class StandaloneSourceGeometry:
    """Finished millimetres; the moving front disk is at z=0, rear at -depth."""

    radius_mm: float = 18.0
    outer_radius_mm: float = 22.0
    depth_mm: float = 8.0
    vertical_offset_mm: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            object.__setattr__(self, name, _number(value, name))
        if not 0.1 <= self.radius_mm <= 200:
            raise ValueError("radius_mm must be in [0.1, 200]")
        if not self.radius_mm + 0.1 <= self.outer_radius_mm <= 500:
            raise ValueError("outer_radius_mm must exceed radius_mm by at least 0.1 mm and be <=500")
        if not 0.1 <= self.depth_mm <= 500:
            raise ValueError("depth_mm must be in [0.1, 500]")
        if abs(self.vertical_offset_mm) > 10000:
            raise ValueError("vertical_offset_mm magnitude must be <=10000")

    @property
    def bounds(self):
        R, y = self.outer_radius_mm, self.vertical_offset_mm
        return ((-R, y-R, -self.depth_mm), (R, y+R, 0.0))

    @property
    def volume_mm3(self):
        return math.pi*self.outer_radius_mm**2*self.depth_mm

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.recipe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    @property
    def recipe(self):
        return {"contractRevision": 1, "formula": FORMULA, "controls": asdict(self)}

    def metadata(self):
        return {"sourceBody": {**self.recipe, "constructionFingerprint": self.fingerprint,
            "domain": "closed-exterior", "sourceMotion": "normal", "sourceNormal": [0.0,0.0,1.0],
            "boundsMm": self.bounds, "sourceAreaMm2": math.pi*self.radius_mm**2,
            "volumeMm3": self.volume_mm3, "sourceTag": 2, "rigidTag": 1}}

    def datums(self):
        y = self.vertical_offset_mm
        return {"SOURCE_AXIS": {"type":"axis", "origin_mm":[0.0,y,0.0], "direction":[0.0,0.0,1.0]},
            "SOURCE_PLANE": {"type":"plane", "origin_mm":[0.0,y,0.0], "normal":[0.0,0.0,1.0]},
            "SOURCE_OUTLINE": {"type":"circle", "center_mm":[0.0,y,0.0], "normal":[0.0,0.0,1.0], "radius_mm":self.radius_mm},
            "BODY_REAR_PLANE": {"type":"plane", "origin_mm":[0.0,y,-self.depth_mm], "normal":[0.0,0.0,-1.0]},
            "BODY_BOUNDS": {"type":"bounds", "bounds_mm":self.bounds}}


def configuration(config):
    """Resolve source-only intent before the legacy horn parser sees it."""
    from .native_boundary import validate_native_boundary
    marker = validate_native_boundary(config)
    if marker != FORMULA:
        return None
    def keys(section, allowed, name):
        if not isinstance(section, Mapping):
            raise ConfigError(f"{name} must be an object")
        extra = set(section)-set(allowed)
        if extra:
            raise ConfigError(f"Standalone source refused: unsupported {name} keys: {', '.join(sorted(map(str,extra)))}")
    keys(config, ("formula","type","mode","source_body","mesh","output","path","output_path"), "root")
    if not any(isinstance(config.get(k),str) and config[k].strip().upper()==FORMULA for k in ("formula","type")):
        raise ConfigError("Standalone source requires a root SOURCE-DISK formula")
    if "mode" in config and config["mode"] != MODE:
        raise ConfigError("Standalone source mode must be standalone-source")
    body = config.get("source_body")
    keys(body, ("radius_mm","outer_radius_mm","depth_mm"), "source_body")
    if set(body) != {"radius_mm","outer_radius_mm","depth_mm"}:
        raise ConfigError("source_body requires radius_mm, outer_radius_mm and depth_mm")
    mesh = config.get("mesh", {})
    keys(mesh, ("size_mm","vertical_offset_mm","verticalOffset","quadrants","scale_to_metres","scaleToMetres",
        "max_triangles","maxTriangles","allow_large_mesh","allowLargeMesh"), "mesh")
    if "output" in config:
        keys(config["output"], ("path","output_path"), "output")
    size = _number(mesh.get("size_mm",3.0), "mesh.size_mm")
    if not 0.1 <= size <= 50:
        raise ConfigError("mesh.size_mm must be in [0.1, 50]")
    for key in ("max_triangles","maxTriangles"):
        if key in mesh and (type(mesh[key]) is not int or not 0 < mesh[key] <= 10_000_000):
            raise ConfigError("standalone source max_triangles must be a positive integer <=10000000")
    model = StandaloneSourceGeometry(**body, vertical_offset_mm=mesh.get("vertical_offset_mm",mesh.get("verticalOffset",0.0)))
    return {"sourceBody": asdict(model), "type": FORMULA}, FORMULA, MODE


def resolve(config, params, allow_large_mesh=None):
    from .config_builder import ResolvedGeometry
    from .geometry import MeshDensity
    mesh = config.get("mesh", {})
    size = float(mesh.get("size_mm",3.0))
    allow = mesh.get("allow_large_mesh",mesh.get("allowLargeMesh",False)) if allow_large_mesh is None else allow_large_mesh
    density = MeshDensity(throat_res_mm=size,mouth_res_mm=size,rear_res_mm=size,
        max_triangles=mesh.get("max_triangles",mesh.get("maxTriangles",18000)),allow_large_mesh=allow)
    model = StandaloneSourceGeometry(**params["sourceBody"])
    validate_density(model,density)
    return ResolvedGeometry(model,density,FORMULA,MODE,"1234",None,
        mesh.get("scale_to_metres",mesh.get("scaleToMetres",True)),{})


def default_density():
    from .geometry import MeshDensity
    return MeshDensity(throat_res_mm=3.0,mouth_res_mm=3.0,rear_res_mm=3.0)


def validate_density(model, density):
    from .geometry import MeshDensity
    if type(model) is not StandaloneSourceGeometry or type(density) is not MeshDensity:
        raise ValueError("standalone source requires explicit StandaloneSourceGeometry and MeshDensity")
    defaults = MeshDensity()
    permitted = {"throat_res_mm","mouth_res_mm","rear_res_mm","max_triangles","allow_large_mesh"}
    for name,value in asdict(density).items():
        if name not in permitted and (type(value) is not type(getattr(defaults,name)) or value != getattr(defaults,name)):
            raise ValueError(f"standalone source does not support MeshDensity.{name}")
    sizes = [_number(getattr(density,k),k) for k in ("throat_res_mm","mouth_res_mm","rear_res_mm")]
    if len(set(sizes)) != 1 or not 0.1 <= sizes[0] <= 50:
        raise ValueError("standalone source MeshDensity sizes must agree and be in [0.1,50] mm")
    if type(density.allow_large_mesh) is not bool:
        raise ValueError("allow_large_mesh must be a boolean")
    if type(density.max_triangles) is not int or not 0 < density.max_triangles <= 10_000_000:
        raise ValueError("standalone source max_triangles must be a positive integer <=10000000")
    effective = min(sizes[0], model.radius_mm/8)
    area = 2*math.pi*model.outer_radius_mm*(model.outer_radius_mm+model.depth_mm)
    estimate = 3*area/effective**2
    if estimate > HARD_TRIANGLE_LIMIT:
        raise ValueError("standalone source estimated allocation exceeds the hard 200000 triangle envelope; increase mesh.size_mm or reduce the body")
    if not density.allow_large_mesh and estimate > 2*density.max_triangles:
        from .mesher import TriangleBudgetExceeded
        raise TriangleBudgetExceeded("standalone source estimated triangles exceed mesh.max_triangles; increase mesh.size_mm, raise max_triangles or set allow_large_mesh=true")
    return sizes[0], effective, estimate


def configure_density(built, density):
    import gmsh
    model = StandaloneSourceGeometry(**built.metadata["sourceBody"]["controls"])
    requested,effective,estimate = validate_density(model,density)
    gmsh.option.setNumber("Mesh.MeshSizeMin",effective)
    gmsh.option.setNumber("Mesh.MeshSizeMax",effective)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints",0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary",1)
    for _,tag in gmsh.model.getEntities(1):
        # A shared source rim needs enough segments for <1% area loss even
        # when a caller requests a coarse mesh. This is a minimum, not a cap.
        length = gmsh.model.occ.getMass(1,tag)
        gmsh.model.mesh.setTransfiniteCurve(tag,max(33 if gmsh.model.getType(1,tag)=="Circle" else 2,math.ceil(length/effective)+1))
    built.metadata["sourceBody"].update({"requestedMeshSizeMm":requested,"effectiveMeshSizeMm":effective,
        "triangleEstimate":estimate,"triangleEstimateIsUpperBound":False,"hardTriangleLimit":HARD_TRIANGLE_LIMIT})


def certify_mesh(model, path, units):
    """Check the published candidate, including topology and physical roles."""
    import meshio
    from .mesher import MesherError
    mesh = meshio.read(path,file_format="gmsh")
    points = np.asarray(mesh.points,dtype=float)*(1000 if units=="m" else 1)
    triangles = np.vstack([c.data for c in mesh.cells if c.type=="triangle"])
    tags = np.concatenate([v for c,v in zip(mesh.cells,mesh.cell_data["gmsh:physical"]) if c.type=="triangle"])
    def check(ok,message):
        if not ok:
            raise MesherError("Standalone source mesh refused: "+message)
    check(0 < len(triangles) <= HARD_TRIANGLE_LIMIT,"hard triangle limit exceeded")
    check(set(tags)=={1,2},"exactly rigid tag1 and source tag2 are required")
    p = points[triangles]
    cross = np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    area = np.linalg.norm(cross,axis=1)/2
    check(bool(np.isfinite(p).all() and (area>0).all()),"nonfinite or zero-area triangle")
    edges = np.sort(np.concatenate((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1)
    unique,inverse,counts = np.unique(edges,axis=0,return_inverse=True,return_counts=True)
    check(bool((counts==2).all()),"every edge must have incidence2")
    # Connectivity is checked between triangles across shared edges. A pair
    # of closed shells touching at one vertex must not pass as one body.
    edge_triangles = np.tile(np.arange(len(triangles)),3)
    order=np.argsort(inverse)
    owners=edge_triangles[order].reshape(-1,2)
    adjacency=[[] for _ in range(len(triangles))]
    for a,b in owners:
        adjacency[a].append(int(b));adjacency[b].append(int(a))
    visited={0};pending=[0]
    while pending:
        for tri in adjacency[pending.pop()]:
            if tri not in visited:
                visited.add(tri);pending.append(tri)
    check(len(visited)==len(triangles),"body must be one edge-connected shell")
    y,r,R,d = model.vertical_offset_mm,model.radius_mm,model.outer_radius_mm,model.depth_mm
    tolerance = 2e-7
    check(len(np.unique(triangles))==len(points),"unused vertices are not part of the closed body")
    radial = np.hypot(points[:,0],points[:,1]-y)
    check(bool((radial<=R+tolerance).all() and (points[:,2]>=-d-tolerance).all()
        and (points[:,2]<=tolerance).all()),"vertices leave the declared physical envelope")
    source = tags==2
    check(bool((np.abs(p[source,:,2])<tolerance).all() and (cross[source,2]>0).all()),"source plane or +Z normals")
    check(bool((np.hypot(p[source,:,0],p[source,:,1]-y)<=r+tolerance).all()),"source outside declared disk")
    edge_tags = np.tile(tags,3)
    lo = np.full(len(unique),3,dtype=int);hi=np.zeros(len(unique),dtype=int)
    np.minimum.at(lo,inverse,edge_tags);np.maximum.at(hi,inverse,edge_tags)
    rim = unique[(lo==1)&(hi==2)]
    check(len(rim)>=32,"source must share its rim with the rigid annulus")
    rp = points[np.unique(rim)]
    rim_error = float(np.max(abs(np.hypot(rp[:,0],rp[:,1]-y)-r)))
    check(rim_error<tolerance and bool((np.abs(rp[:,2])<tolerance).all()),"source rim geometry")
    analytic_area=math.pi*r*r; emitted_area=float(area[source].sum())
    area_error=abs(emitted_area-analytic_area)/analytic_area
    check(area_error<0.01,"source polygon area differs by >=1%")
    rigid = ~source
    front = rigid & (np.max(abs(p[:,:,2]),axis=1)<tolerance)
    rear = rigid & (np.max(abs(p[:,:,2]+d),axis=1)<tolerance)
    side = rigid & ~front & ~rear
    centers=p.mean(axis=1)
    check(bool((cross[front,2]>0).all() and (cross[rear,2]<0).all()),"front/rear rigid normals")
    check(bool(front.any() and rear.any() and side.any()),"all rigid body patches are required")
    check(bool((np.abs(np.hypot(p[side,:,0],p[side,:,1]-y)-R)<tolerance).all()),"side vertices must lie on cylinder")
    check(bool((cross[side,0]*centers[side,0]+cross[side,1]*(centers[side,1]-y)>0).all()),"side exterior normals")
    check(bool((np.abs(radial[triangles[front]]-r)<=R-r+tolerance).all()),"annulus outside body")
    check(bool((radial[triangles[front]]>=r-tolerance).all()),"rigid annulus overlaps source")
    shifted=p-np.array([0.,y,0.])
    volume=float(np.einsum('ij,ij->i',shifted[:,0],np.cross(shifted[:,1],shifted[:,2])).sum()/6)
    volume_error=abs(volume-model.volume_mm3)/model.volume_mm3
    check(volume>0 and volume_error<0.01,"positive volume within1% of exact cylinder required")
    return {"edgeIncidence":2,"connectedComponents":1,"sourceRimErrorMm":rim_error,
        "emittedSourceAreaMm2":emitted_area,"sourceAreaRelativeError":area_error,
        "signedMeshVolumeMm3":volume,"volumeRelativeError":volume_error,"certificate":"closed-exterior-topology-and-roles"}
