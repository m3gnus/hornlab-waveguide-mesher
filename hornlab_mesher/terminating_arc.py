"""Explicit curvature-matched circular termination of a circular OSSE body."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config_parser import ConfigError
from .throat_adapter import _conic_residual_bound

FORMULA = "OSSE-ARC"
KEY = "terminating_arc"


def _fail(message):
    raise ConfigError("Terminating arc refused: " + message)


def _number(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        valid = False
    if not valid:
        _fail(f"{name} must be a finite scalar number.")
    return float(value)


def configuration(config, resolver):
    """Inspect original intent before delegating the bounded ordinary body."""
    pending, payloads, seen = [config], [], set()
    while pending:
        section = pending.pop()
        if not isinstance(section, Mapping) or id(section) in seen:
            continue
        seen.add(id(section))
        for key, value in section.items():
            if str(key).replace("_", "").lower() == "terminatingarc":
                payloads.append((section, key))
            if isinstance(value, Mapping):
                pending.append(value)
    profile = config.get("profile", {})
    profile = profile if isinstance(profile, Mapping) else {}
    names = [section[key] for section in (config, profile) for key in ("formula", "type") if key in section]
    active = any(isinstance(name, str) and name.upper() == FORMULA for name in names)
    if not active and not payloads:
        return None
    if not active or not payloads or any(section is not config or key != KEY for section, key in payloads):
        _fail(f"{KEY} requires the exact root key and formula {FORMULA}.")
    for name in names:
        if not isinstance(name, str) or name.upper() != FORMULA:
            _fail(f"every supplied formula and type must be {FORMULA}.")
    if set(profile)-{"formula","type","L_mm","r0_mm","a_deg","a0_deg","k","s"}:
        _fail("only canonical scalar unterminated profile controls are qualified.")
    if "axial_scale" in config:
        _fail("independent axial scaling cannot be combined with this construction.")
    value = config[KEY]
    if not isinstance(value, Mapping) or set(value) != {"contract_revision", "join_t", "end_tangent_deg"}:
        _fail("terminating_arc requires exactly contract_revision, join_t and end_tangent_deg.")
    if type(value["contract_revision"]) is not int or value["contract_revision"] != 1:
        _fail("contract_revision must be integer 1.")
    payload = {"contract_revision": 1, "join_t": _number(value["join_t"], "join_t"),
               "end_tangent_deg": _number(value["end_tangent_deg"], "end_tangent_deg")}
    # Reuse the strict scalar body's original-intent boundary, with equal
    # radial and axial factors. This temporary parser request never escapes.
    from .axial_scale import configuration as axial_configuration
    clean = copy.deepcopy(dict(config))
    clean.pop(KEY)
    clean["formula"] = "OSSE-AXIAL"
    clean.pop("type", None)
    for key in ("formula", "type"):
        clean.setdefault("profile", {}).pop(key, None)
    clean["axial_scale"] = config.get("scale", 1)
    params, _, mode = axial_configuration(clean, resolver)
    params.pop("absoluteAxialScale")
    params.pop("axialScaleFingerprint", None)
    params["terminatingArc"] = payload
    params["type"] = FORMULA
    model = ArcMeridian.from_params(params)
    params["terminatingArcFingerprint"] = model.fingerprint
    return params, FORMULA, mode


@dataclass(frozen=True)
class ArcMeridian:
    """Immutable physical meridian; arrays are backed by immutable bytes.

    The retained conic uses physical z in [0,join_z]. The circle uses the
    absolute oriented tangent angle phi in [beta,end]. All lengths are mm.
    """
    length: float
    r0: float
    k: float
    slope: float
    throat_slope: float
    join_t: float
    end: float
    offset: float
    radius: float
    beta: float
    center_z: float
    center_r: float
    _poles: bytes
    _weights: bytes
    _identity: str

    @classmethod
    def from_params(cls, params):
        from .axial_scale import AxialModel
        p = dict(params, absoluteAxialScale=params.get("scale", 1))
        body = AxialModel.from_params(p)
        payload = params["terminatingArc"]
        t = _number(payload["join_t"], "join_t")
        end = math.radians(_number(payload["end_tangent_deg"], "end_tangent_deg"))
        if not 0 < t <= 1 or body.length*t < .1:
            _fail("join_t requires 0 < join_t <= 1 and a retained body of at least 0.1 mm.")
        z = body.length*t
        J, tangent = body.body(z)
        d = float(tangent[1])
        dd = body.second_bound(z, z)
        if not math.isfinite(dd) or dd <= 0:
            _fail("positive body curvature could not be resolved.")
        beta = math.atan(d)
        radius = math.hypot(1, d)**3/dd
        if not math.isfinite(radius) or not .1 <= radius <= 10000:
            _fail("curvature-matched radius must be in [0.1, 10000] mm.")
        if not beta < end <= math.pi or radius*(end-beta) < .1:
            _fail("end_tangent_deg must exceed the join tangent, be at most 180, and retain 0.1 mm of arc.")
        cz, cr = float(J[0])-radius*math.sin(beta), float(J[1])+radius*math.cos(beta)
        end_z = cz+radius*math.sin(end)
        end_r = cr-radius*math.cos(end)
        if end_z < .1 or end_r > 10000:
            _fail("the complete arc must clear the source by 0.1 mm and its radius must not exceed 10000 mm.")
        # Exact rational hyperbola segment, from its hyperbolic midpoint.
        ta, t0, kr = body.slope, body.throat_slope, body.k*body.r0
        K = kr*kr*(1-(t0/ta)**2)
        if K <= 1e-9*kr*kr:
            _fail("the conic is too close to degeneracy to certify.")
        center = -kr*t0/(ta*ta)
        eta0 = math.asinh(ta*(-center)/math.sqrt(K))
        eta1 = math.asinh(ta*(z-center)/math.sqrt(K))
        weight = math.cosh((eta1-eta0)/2)
        fraction = math.tanh((eta1-eta0)/2)**2
        poles = np.array([[0, body.r0], [.5*z*(1-fraction)+center*fraction,
            .5*(body.r0+float(J[1]))*(1-fraction)+body.r0*(1-body.k)*fraction], J])
        weights = np.array([1., weight, 1.])
        size = max(1., body.length, radius, abs(cz), cr, end_r, abs(body.offset)+end_r)
        margin=1e-10*size
        if min(z,radius*(end-beta),end_z) <= .1+margin:
            _fail("the retained span, arc or source-plane clearance could not be certified above 0.1 mm.")
        if size > 100000 or np.min(np.diff(poles[:,0])) <= 1e-4 or np.min(poles[:,1]) <= 1e-4:
            _fail("exact conic controls or complete coordinates exceed the qualified precision range.")
        error = _conic_residual_bound(poles, weights, length=body.length, join_z=0,
            adapter_length=0, r0=body.r0, k=body.k, tan_a=ta, tan_a0=t0)
        if error+1e-10*size > 1e-4:
            _fail("continuous rational-conic position precision could not be certified.")
        vectors = (poles[1]-poles[0], poles[2]-poles[1])
        for vector, angle in zip(vectors, (math.atan(t0), beta)):
            margin = 64*np.finfo(float).eps*size/np.linalg.norm(vector)
            if abs(math.atan2(vector[1],vector[0])-angle)+margin > 5e-10:
                _fail("conic endpoint tangent precision could not be certified.")
        identity = {"contractRevision":1,"construction":"circular-osse-osculating-arc-v1",
            "formula":FORMULA,"authoredBody":{key:float(params[key]) for key in ("L","r0","a","a0","k","scale")},
            "body":{"lengthMm":body.length,"throatRadiusMm":body.r0,"k":body.k,
                    "slope":ta,"throatSlope":t0},
            "arc":{"joinT":t,"endTangentDeg":float(payload["end_tangent_deg"])},
            "verticalOffsetMm":body.offset,"source":"flat-throat-rim-v1"}
        return cls(body.length,body.r0,body.k,ta,t0,t,end,body.offset,radius,beta,cz,cr,
            poles.tobytes(),weights.tobytes(),json.dumps(identity,sort_keys=True,separators=(",",":"),allow_nan=False))

    @property
    def poles(self):
        return np.frombuffer(self._poles,dtype=np.float64).reshape(3,2)

    @property
    def weights(self):
        return np.frombuffer(self._weights,dtype=np.float64)

    @property
    def identity(self):
        return json.loads(self._identity)

    @property
    def fingerprint(self):
        return hashlib.sha256(self._identity.encode()).hexdigest()

    @property
    def join_z(self):
        return self.length*self.join_t

    @property
    def coefficients(self):
        return (self.k*self.r0)**2, self.k*self.r0*self.throat_slope, self.slope**2

    def body(self,z):
        z=np.asarray(z,dtype=np.float64)
        A,B,C=self.coefficients
        root=np.sqrt(A+2*B*z+C*z*z)
        p=np.stack((z,self.r0+(2*B*z+C*z*z)/(root+self.k*self.r0)),axis=-1)
        tangent=np.stack((np.ones_like(z),(B+C*z)/root),axis=-1)
        return p,tangent

    def second_bound(self,lo,hi):
        A,B,C=self.coefficients
        return (A*C-B*B)/(A+2*B*lo+C*lo*lo)**1.5

    def arc(self,phi):
        phi=np.asarray(phi,dtype=np.float64)
        return np.stack((self.center_z+self.radius*np.sin(phi),self.center_r-self.radius*np.cos(phi)),axis=-1)

    @property
    def opening(self):
        return self.arc(self.end)

    @property
    def reach(self):
        return float(self.opening[1])

    @property
    def bounds(self):
        zmax=self.center_z+self.radius if self.beta <= math.pi/2 <= self.end else float(self.opening[0])
        return [[-self.reach,self.offset-self.reach,0.],[self.reach,self.offset+self.reach,zmax]]

    def metadata(self):
        return {"construction_fingerprint":self.fingerprint,"terminatingArc":{
            **self.identity,"fingerprint":self.fingerprint,"joinMm":self.body(self.join_z)[0].tolist(),
            "openingMm":self.opening.tolist(),"nominalBodyMouthZMm":self.length,
            "radiusMm":self.radius,"joinTangentDeg":math.degrees(self.beta),"boundsMm":self.bounds,
            "conicPositionBoundMm":0.0001}}


def resolve(config,params,allow_large_mesh=None):
    from .config_builder import ResolvedGeometry,_mesh_density_from_config
    from .geometry import _ArcPointGridHornGeometry
    model=ArcMeridian.from_params(params)
    body=model.body(np.linspace(0,model.join_z,33))[0]
    arc=model.arc(np.linspace(model.beta,model.end,33))[1:]
    meridian=np.vstack((body,arc))
    az=np.arange(64)*2*math.pi/64
    grid=np.stack((np.cos(az)[:,None]*meridian[:,1],np.sin(az)[:,None]*meridian[:,1],
        np.broadcast_to(meridian[:,0],(64,len(meridian)))),axis=-1)
    grid=np.frombuffer(grid.tobytes(),dtype=np.float64).reshape(grid.shape)
    geometry=_ArcPointGridHornGeometry(inner_points=grid,wall_thickness_mm=0,source_shape=0,
        source_radius_mm=-1,source_curv=0,source_auto_angle_deg=math.degrees(math.atan(model.throat_slope)),
        closed=True,symmetry_planes=(),vertical_offset_mm=model.offset,arc_meridian=model)
    return ResolvedGeometry(geometry,_mesh_density_from_config(config,allow_large_mesh=allow_large_mesh),
        FORMULA,"bare","1234",None,config.get("mesh",{}).get("scale_to_metres",True),model.metadata())


def validate_density(density):
    """Keep strict native resource intent separate from legacy coercions."""
    from dataclasses import asdict
    from .geometry import MeshDensity
    if type(density) is not MeshDensity:
        _fail("direct density must be MeshDensity or None.")
    defaults=MeshDensity()
    allowed={"throat_res_mm","mouth_res_mm","rear_res_mm","max_triangles","allow_large_mesh"}
    for name,value in asdict(density).items():
        if name not in allowed and (type(value) is not type(getattr(defaults,name)) or value != getattr(defaults,name)):
            _fail(f"unsupported direct density control {name}.")
    for name in ("throat_res_mm","mouth_res_mm","rear_res_mm"):
        if not .01 <= _number(getattr(density,name),name) <= 10000:
            _fail(f"{name} must be in [0.01,10000] mm.")
    if type(density.allow_large_mesh) is not bool:
        _fail("allow_large_mesh must be a boolean.")
    if type(density.max_triangles) is not int or not 0 < density.max_triangles <= 10_000_000:
        _fail("max_triangles must be a positive integer no larger than 10000000.")
