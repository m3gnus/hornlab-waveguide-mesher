"""One physical authority for an authored adapter with absolute axial scaling."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

import numpy as np

from .config_parser import ConfigError
from .throat_adapter import CONTROLS, resolve_adapter, _conic_residual_bound

FORMULA = "OSSE-ADAPTER-AXIAL"
KEY = "adapterAxialScale"


def _fail(message):
    raise ConfigError("Adapter axial scale refused: " + message)


def _number(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        _fail(f"{name} must be a finite scalar number.")
    return float(value)


def configuration(config, resolver):
    sections = [config]+[config[k] for k in ("profile", "parameters") if isinstance(config.get(k), Mapping)]
    names = [section[k] for section in sections for k in ("formula", "type") if k in section]
    if not any(isinstance(v, str) and v.strip().upper() == FORMULA for v in names):
        return None
    allowed = {
        "root": {"formula", "type", "axial_scale", "scale", "throat_adapter", "profile", "mesh", "source", "mode", "quadrants", "vertical_offset_mm", "output", "path", "output_path"},
        "profile": {"formula", "type", "L_mm", "r0_mm", "a_deg", "a0_deg", "k", "s"},
        "mesh": {"wall_thickness_mm", "angular_segments", "length_segments", "throat_res_mm", "mouth_res_mm", "rear_res_mm", "scale_to_metres", "topology_mode", "surface_fit", "max_triangles", "allow_large_mesh", "quadrants", "vertical_offset_mm"},
        "source": {"source_shape", "source_radius_mm", "source_curv"},
    }
    for name, section in (("root", config), *((k, config.get(k, {})) for k in ("profile", "mesh", "source"))):
        if not isinstance(section, Mapping):
            _fail(f"{name} must be an object.")
        extras = set(section)-allowed[name]
        if extras:
            _fail(f"unsupported {name} controls: {', '.join(sorted(map(str, extras)))}.")
        for key in set(section)-{"formula", "type", "throat_adapter", "profile", "mesh", "source", "mode", "quadrants", "topology_mode", "surface_fit", "scale_to_metres", "allow_large_mesh", "output", "path", "output_path"}:
            _number(section[key], key)
    if "axial_scale" not in config or "throat_adapter" not in config:
        _fail("the formula requires root axial_scale and an active root throat_adapter.")
    adapter = config["throat_adapter"]
    if not isinstance(adapter, Mapping):
        _fail("throat_adapter must be an active authored object.")
    extras = set(adapter)-{"mode", "contract_revision", *CONTROLS}
    if extras:
        _fail(f"unsupported adapter controls: {', '.join(sorted(map(str,extras)))}.")
    mesh, source = config.get("mesh", {}), config.get("source", {})
    if config.get("mode", "bare") != "bare" or mesh.get("wall_thickness_mm", 0) != 0:
        _fail("only bare zero-wall models are supported.")
    if config.get("profile", {}).get("s", 0) != 0:
        _fail("termination must be zero.")
    if source.get("source_shape", 0) != 0 or source.get("source_curv", 0) != 0 or source.get("source_radius_mm", -1) != -1:
        _fail("only the native flat driver source is supported.")
    if mesh.get("topology_mode", "acoustic") != "acoustic" or mesh.get("surface_fit", "auto") not in ("auto", "interpolate"):
        _fail("acoustic topology and exact interpolating construction are required.")
    for key in ("throat_res_mm", "mouth_res_mm", "rear_res_mm"):
        if key in mesh and not .01 <= mesh[key] <= 10000:
            _fail(f"{key} must be in [0.01,10000] mm.")
    output = config.get("output", {})
    if not isinstance(output, Mapping) or set(output)-{"path", "output_path"}:
        _fail("output supports only path and output_path routing controls.")
    for section in (config, output):
        for key in ("path", "output_path"):
            if key in section and section[key] is not None and not isinstance(section[key], str):
                _fail(f"{key} must be a string or null.")
    clean = copy.deepcopy(dict(config))
    axis = _number(clean.pop("axial_scale"), "axial_scale")
    clean["formula"] = "OSSE-ADAPTER"
    clean.pop("type", None)
    clean.setdefault("profile", {}).pop("formula", None)
    clean["profile"].pop("type", None)
    clean["mode"] = "bare"
    clean.setdefault("mesh", {}).setdefault("wall_thickness_mm", 0)
    clean.setdefault("source", {}).setdefault("source_shape", 0)
    params, _, mode = resolver(clean)
    if not params.get("throat_adapter"):
        _fail("an active authored adapter is required.")
    params[KEY] = axis
    model = PhysicalAdapter.from_params(params)
    params["construction_fingerprint"] = model.fingerprint
    return params, FORMULA, mode


@dataclass(frozen=True, eq=False)
class PhysicalAdapter:
    """Affine image of exact intrinsic cubic/conic controls; placement is last."""
    xy_scale: float
    axial_scale: float
    offset: float
    cubic: np.ndarray
    body_poles: np.ndarray
    body_weights: np.ndarray
    _identity_json: str

    @classmethod
    def from_params(cls, params):
        xy = _number(params.get("scale", 1), "scale")
        axis = _number(params[KEY], "axial_scale")
        offset = _number(params.get("verticalOffset", 0), "vertical_offset_mm")
        if not (.01 <= xy <= 10 and .01 <= axis <= 10 and .1 <= axis/xy <= 10):
            _fail("both scales must be in [0.01,10], with axial_scale/scale in [0.1,10].")
        if abs(offset) > 10000:
            _fail("finished vertical placement must be within 10000 mm.")
        # Retain intrinsic construction validity; physical checks below use the
        # actual components, rather than treating an anisotropic map as uniform.
        intrinsic = dict(params, scale=1., verticalOffset=0.)
        intrinsic.pop(KEY, None)
        try:
            authored = resolve_adapter(intrinsic)
        except ValueError as exc:
            _fail(str(exc))
        if authored is None:
            _fail("an active authored adapter is required.")
        a, a0, k = (float(params[n]) for n in ("a", "a0", "k"))
        depth = axis*(authored.payload["length_mm"]+authored.length*(1-authored.payload["join_t"]))
        radius = xy*authored.payload["driver_exit_diameter_mm"]/2
        if not (5 <= a <= 75 and 0 <= a0 < a and .5 <= k <= 10 and 1 <= depth <= 2000 and 1 <= radius <= 200):
            _fail("physical depth, driver radius, k or body angles are outside the qualified domain.")
        transform = np.array([axis, xy])
        cubic, poles = authored.cubic*transform, authored.body_poles*transform
        poles[0] = cubic[-1]  # One exact shared join value for every terminal.
        clearance = min(np.min(cubic[:,1]), np.min(poles[:,1]), np.min(np.diff(cubic[:,0])), np.min(np.diff(poles[:,0])))
        magnitude = max(1., float(np.max(np.abs(cubic))), float(np.max(np.abs(poles))), abs(offset))
        if clearance <= 1e-4 or magnitude+abs(offset) > 1e5 or max(np.max(cubic[:,1]), np.max(poles[:,1])) > 10000:
            _fail("physical control clearance or full envelope is outside the certified range.")
        expected = [math.atan2(xy*math.sin(math.radians(authored.payload["exit_half_angle_deg"])), axis*math.cos(math.radians(authored.payload["exit_half_angle_deg"])))]
        for t in (authored.payload["join_t"], authored.payload["join_t"], 1.):
            d = authored.base_tangent(t)*transform
            expected.append(math.atan2(d[1], d[0]))
        for d, angle in zip((cubic[1]-cubic[0], cubic[3]-cubic[2], poles[1]-poles[0], poles[2]-poles[1]), expected):
            error = abs(math.atan2(d[1],d[0])-angle)+64*np.finfo(float).eps*magnitude/np.linalg.norm(d)
            if error > 5e-9:
                _fail("physical endpoint tangent precision could not be certified.")
        residual = _conic_residual_bound(authored.body_poles, authored.body_weights,
            length=authored.length, join_z=authored.length*authored.payload["join_t"],
            adapter_length=authored.payload["length_mm"], r0=authored.r0, k=authored.k,
            tan_a=authored.tan_a, tan_a0=authored.tan_a0)
        if residual*xy+1e-10*magnitude > 1e-4:
            _fail("physical continuous conic precision could not be certified.")
        identity = {"construction":"circular-cubic-conic-absolute-axis-v1", "contract_revision":1,
            "formula":FORMULA, "adapter":dict(authored.payload), "body":dict(authored.identity["body"]),
            "scale":xy, "axial_scale":axis, "vertical_offset_mm":offset,
            "source":"flat-driver-rim-v1", "quadrants":"1234"}
        # Bytes backing prevents callers from re-enabling writes on an owned
        # NumPy buffer, as well as detaching the authority from input mappings.
        cubic,poles,weights = (np.frombuffer(array.tobytes(),dtype=np.float64).reshape(array.shape)
            for array in (cubic,poles,np.array(authored.body_weights,copy=True)))
        return cls(xy,axis,offset,cubic,poles,weights,json.dumps(identity,sort_keys=True,separators=(",",":"),allow_nan=False))

    @property
    def identity(self):
        return json.loads(self._identity_json)

    @property
    def payload(self):
        return self.identity["adapter"]

    @property
    def fingerprint(self):
        return hashlib.sha256(self._identity_json.encode()).hexdigest()

    def __eq__(self, other):
        return isinstance(other, PhysicalAdapter) and self._identity_json == other._identity_json

    def __hash__(self):
        return hash(self._identity_json)

    @property
    def source_angle(self):
        d = self.cubic[1]-self.cubic[0]
        return math.degrees(math.atan2(d[1],d[0]))

    def cubic_values(self, u):
        u = np.asarray(u,dtype=float)
        v = 1-u
        c = self.cubic
        point = v[...,None]**3*c[0]+3*v[...,None]**2*u[...,None]*c[1]+3*v[...,None]*u[...,None]**2*c[2]+u[...,None]**3*c[3]
        d = 3*(v[...,None]**2*(c[1]-c[0])+2*v[...,None]*u[...,None]*(c[2]-c[1])+u[...,None]**2*(c[3]-c[2]))
        dd = 6*(v[...,None]*(c[2]-2*c[1]+c[0])+u[...,None]*(c[3]-2*c[2]+c[1]))
        return point,d,dd

    def body_values(self, t):
        t = np.asarray(t,dtype=float)
        body = self.identity["body"]
        L,r0,k = (body[n] for n in ("L","r0","k"))
        ta,t0 = (math.tan(math.radians(body[n])) for n in ("a","a0"))
        z = L*t
        increment = 2*k*r0*z*t0+z*z*ta*ta
        root = np.sqrt((k*r0)**2+increment)
        r = r0+increment/(root+k*r0)
        slope = (k*r0*t0+z*ta*ta)/root
        second = (k*r0)**2*(ta*ta-t0*t0)/root**3
        p = np.stack((self.axial_scale*(self.payload["length_mm"]+z-L*self.payload["join_t"]), self.xy_scale*r),axis=-1)
        d = np.stack((np.full_like(t,self.axial_scale*L),self.xy_scale*L*slope),axis=-1)
        dd = np.stack((np.zeros_like(t),self.xy_scale*L*L*second),axis=-1)
        return p,d,dd

    @property
    def reach(self):
        # Cubic radial extrema are roots of its quadratic derivative. Include
        # all interior extrema, even when the driver is wider than the mouth.
        c = self.cubic[:,1]
        coefficients = [3*(-c[0]+3*c[1]-3*c[2]+c[3]),6*(c[0]-2*c[1]+c[2]),3*(c[1]-c[0])]
        candidates = [0.,1.]
        for root in np.roots(np.trim_zeros(coefficients,"f")):
            if abs(root.imag) < 1e-12 and 0 < root.real < 1:
                candidates.append(float(root.real))
        return max(float(np.max(self.cubic_values(candidates)[0][:,1])),float(self.body_poles[-1,1]))

    @property
    def bounds(self):
        r = self.reach
        return [[-r,self.offset-r,0.],[r,self.offset+r,float(self.body_poles[-1,0])]]

    @property
    def dimensions(self):
        return {"mouth_opening":[2*float(self.body_poles[-1,1])]*2,
            "horn_overall":[2*self.reach,2*self.reach,float(self.body_poles[-1,0])]}

    def metadata(self):
        return {"construction_fingerprint":self.fingerprint,"throat_adapter":self.payload,
            "adapterAbsoluteAxialScale":{"contractRevision":1,"xyScale":self.xy_scale,"axialScale":self.axial_scale,
                "verticalOffsetMm":self.offset,"sourceAngleDeg":self.source_angle,"boundsMm":self.bounds,
                "joinZMm":float(self.cubic[-1,0]),"mouthZMm":float(self.body_poles[-1,0])}}


def resolve(config, params, allow_large_mesh=None):
    from .config_builder import ResolvedGeometry, _mesh_density_from_config
    from .geometry import _AdapterPointGridHornGeometry
    model = PhysicalAdapter.from_params(params)
    t = np.linspace(model.payload["join_t"],1.,33)
    meridian = np.vstack((model.cubic_values(np.linspace(0,1,17))[0][:-1],model.body_values(t)[0]))
    # The exact authority governs terminals; this density-independent grid is
    # only a compatibility sizing/orientation sample, including cardinal axes.
    az = np.arange(64)*2*math.pi/64
    grid = np.stack((np.cos(az)[:,None]*meridian[:,1],np.sin(az)[:,None]*meridian[:,1],np.broadcast_to(meridian[:,0],(64,len(meridian)))),axis=-1)
    geometry = _AdapterPointGridHornGeometry(inner_points=grid,wall_thickness_mm=0,source_shape=0,
        source_radius_mm=-1,source_curv=0,source_auto_angle_deg=model.source_angle,
        closed=True,symmetry_planes=(),vertical_offset_mm=model.offset,adapter_meridian=model,adapter_scale=1.)
    return ResolvedGeometry(geometry,_mesh_density_from_config(config,allow_large_mesh=allow_large_mesh),
        FORMULA,"bare","1234",None,config.get("mesh",{}).get("scale_to_metres",True),model.metadata())
