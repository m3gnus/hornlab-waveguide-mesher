"""Canonical circular mouth lip and its bounded first native configuration."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config_parser import ConfigError

FORMULA = "OSSE-ROUNDOVER"
KEY = "mouth_roundover_radius_mm"
FIT_TOL_MM = 0.0001


def _fail(message):
    raise ConfigError("Mouth roundover refused: " + message)


def _number(value, name):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))
    except (ValueError, OverflowError, TypeError):
        valid = False
    if not valid:
        _fail(f"{name} must be a finite scalar number.")
    return float(value)


def configuration(config, resolver):
    """Return active normalized parameters, or None for the unchanged path."""
    profile = config.get("profile", {})
    mesh = config.get("mesh", {})
    profile = profile if isinstance(profile, Mapping) else {}
    mesh = mesh if isinstance(mesh, Mapping) else {}
    formula = str(config.get("formula", config.get("type", profile.get("formula", profile.get("type", "OSSE"))))).upper()
    radius = _number(mesh.get(KEY, 0), KEY)
    supplied_marker = any(
        isinstance(section.get(key), str) and section[key].upper() == FORMULA
        for section in (config, profile) for key in ("formula", "type"))
    if radius == 0 and not supplied_marker:
        return None
    if radius <= 0:
        _fail("radius must be finite and positive.")
    if formula != FORMULA:
        _fail(f"an active radius requires formula {FORMULA} so older readers refuse it.")
    for name in ("profile", "mesh", "source"):
        if name in config and not isinstance(config[name], Mapping):
            _fail(f"{name} must be an object.")
    # Validate inactive editor state before removing it from the body recipe.
    if "throat_adapter" in config and "throat_adapter" in profile:
        _fail("throat_adapter must be supplied in only one location.")
    from .throat_adapter import normalize_adapter
    for section in (config, profile):
        if "throat_adapter" in section:
            try:
                adapter = normalize_adapter(section["throat_adapter"])
            except ValueError as exc:
                _fail(str(exc))
            if adapter is not None:
                _fail("combined active throat adapter and mouth roundover are not supported.")
    allowed = {
        "root": {"formula", "type", "profile", "mesh", "source", "mode", "scale", "quadrants", "vertical_offset_mm", "throat_adapter"},
        "profile": {"formula", "type", "L_mm", "r0_mm", "a_deg", "a0_deg", "k", "s", "n", "q", "throat_adapter"},
        "mesh": {KEY, "wall_thickness_mm", "angular_segments", "length_segments", "throat_res_mm", "mouth_res_mm", "rear_res_mm", "scale_to_metres", "topology_mode", "surface_fit", "max_triangles", "allow_large_mesh", "quadrants", "vertical_offset_mm"},
        "source": {"source_shape", "source_radius_mm", "source_curv"},
    }
    for name, section in (("root", config), ("profile", profile), ("mesh", mesh), ("source", config.get("source", {}))):
        if not isinstance(section, Mapping):
            _fail(f"{name} must be an object.")
        extras = set(section) - allowed[name]
        if extras:
            _fail(f"unsupported {name} controls: {', '.join(sorted(extras))}.")
        numeric = {
            "root":{"scale","vertical_offset_mm"},
            "profile":{"L_mm","r0_mm","a_deg","a0_deg","k","s","n","q"},
            "mesh":{KEY,"wall_thickness_mm","vertical_offset_mm"},
            "source":{"source_shape","source_radius_mm","source_curv"},
        }[name]
        for key in numeric.intersection(section):
            _number(section[key],key)
    # Inspect every supplied discriminator before legacy precedence or cleanup
    # can hide a competing geometry request.
    for section in (config, profile):
        for key in ("formula", "type"):
            if key in section and (not isinstance(section[key], str) or section[key].upper() != FORMULA):
                _fail(f"every supplied {key} must be {FORMULA}.")
    for section in (config, mesh):
        if "quadrants" in section:
            value = section["quadrants"]
            if not ((type(value) is int and value == 1234) or (type(value) is str and value == "1234")):
                _fail("quadrants must be exactly 1234 for full-circle coverage.")
    if "vertical_offset_mm" in config and "vertical_offset_mm" in mesh:
        if config["vertical_offset_mm"] != mesh["vertical_offset_mm"]:
            _fail("duplicate vertical_offset_mm controls must agree.")
    for key in ("length_segments", "angular_segments"):
        if key in mesh:
            value = _number(mesh[key], key)
            if value <= 0 or not value.is_integer():
                _fail(f"{key} must be a positive integer.")
    for key in ("allow_large_mesh", "scale_to_metres"):
        if key in mesh and type(mesh[key]) is not bool:
            _fail(f"{key} must be a boolean.")
    clean = copy.deepcopy(dict(config))
    clean.pop("throat_adapter", None)
    clean["formula"] = "OSSE"
    clean.pop("type", None)
    clean.setdefault("profile", {}).pop("formula", None)
    clean["profile"].pop("type", None)
    clean["profile"].pop("throat_adapter", None)
    clean.setdefault("mesh", {}).pop(KEY, None)
    clean.setdefault("source", {}).setdefault("source_shape", 0)
    params, _, mode = resolver(clean)
    if mode != "freestanding":
        _fail("the lip requires a freestanding wall.")
    for key in ("L", "r0", "a", "a0", "k", "s", "n", "q", "scale", "verticalOffset", "wallThickness", "sourceShape", "sourceRadius", "sourceCurv"):
        if key in params:
            _number(params[key], key)
    if params.get("s", 0) != 0:
        _fail("termination must be zero in the initial circular construction.")
    if str(params.get("quadrants", "1234")) != "1234":
        _fail("only full-circle models are qualified.")
    if params.get("sourceShape") != 0 or params.get("sourceCurv", 0) != 0 or params.get("sourceRadius", -1) != -1:
        _fail("only the native flat source is qualified.")
    if mesh.get("topology_mode", "acoustic") != "acoustic" or mesh.get("surface_fit", "auto") not in {"auto", "interpolate"}:
        _fail("the analytic construction requires acoustic topology and an interpolating fit.")
    params["mouthRoundoverRadiusMm"] = radius
    model = Roundover.from_params(params)
    params["mouthRoundoverFingerprint"] = model.fingerprint
    return params, FORMULA, mode


@dataclass(frozen=True)
class Roundover:
    length: float
    r0: float
    a: float
    a0: float
    k: float
    wall: float
    radius: float
    offset: float = 0.0

    @classmethod
    def from_params(cls, p):
        scale = _number(p.get("scale", 1), "scale")
        if not 0.01 <= scale <= 10:
            _fail("scale must be in [0.01, 10].")
        model = cls(float(p["L"])*scale, float(p["r0"])*scale,
                    math.radians(float(p["a"])), math.radians(float(p["a0"])),
                    float(p["k"]), float(p["wallThickness"]),
                    float(p["mouthRoundoverRadiusMm"]), float(p.get("verticalOffset", 0)))
        if not (1 <= model.length <= 2000 and 1 <= model.r0 <= 200 and 0.5 <= model.k <= 10
                and 0 <= model.a0 < model.a <= math.radians(75) and model.a >= math.radians(5)):
            _fail("body coefficients are outside the qualified circular domain.")
        if not (0.1 <= model.wall < model.radius <= 200 and model.radius-model.wall >= 0.1):
            _fail("radius must exceed the positive wall thickness by at least 0.1 mm.")
        if abs(model.offset) > 10000:
            _fail("vertical placement exceeds 10000 mm.")
        if model.wall*model.second_bound(0, model.length, False) >= 0.4:
            _fail("the outer offset approaches a body curvature fold.")
        if model.center[0] <= 0.1:
            _fail("the lip folds or intersects the rear closure.")
        return model

    @property
    def coefficients(self):
        return (self.k*self.r0)**2, self.k*self.r0*math.tan(self.a0), math.tan(self.a)**2

    def body(self, z, outer=False):
        z = np.asarray(z, dtype=np.float64)
        A, B, C = self.coefficients
        v = np.sqrt(A+2*B*z+C*z*z)
        r = v+self.r0*(1-self.k)
        d = (B+C*z)/v
        d2 = (C*A-B*B)/v**3
        norm = np.sqrt(1+d*d)
        if outer:
            factor = 1-self.wall*d2/norm**3
            return np.stack((z-self.wall*d/norm, r+self.wall/norm), axis=-1), np.stack((factor, d*factor), axis=-1)
        return np.stack((z, r), axis=-1), np.stack((np.ones_like(z), d), axis=-1)

    def second_bound(self, lo, hi, outer=False):
        A, B, C = self.coefficients
        v = math.sqrt(A+2*B*lo+C*lo*lo)
        H = C*A-B*B
        d2 = H/v**3
        d3 = 3*H*(B+C*hi)/v**5
        return d2+self.wall*(d3+3*d2*d2) if outer else d2

    @property
    def beta(self):
        return math.atan(float(self.body(self.length)[1][1]))

    @property
    def center(self):
        mouth = self.body(self.length)[0]
        return mouth+self.radius*np.array([-math.sin(self.beta), math.cos(self.beta)])

    def arc(self, phi, outer=False):
        angle = self.beta+np.asarray(phi, dtype=np.float64)
        rad = self.radius-self.wall if outer else self.radius
        return self.center+rad*np.stack((np.sin(angle), -np.cos(angle)), axis=-1)

    @property
    def sweep(self):
        return math.pi-self.beta

    @property
    def bounds(self):
        reach = float(self.center[1]+self.radius)
        return [[-reach, self.offset-reach, -self.wall], [reach, self.offset+reach, float(self.center[0]+self.radius)]]

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps({"contract":1, **self.__dict__}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def stations(self, tolerance, outer=False):
        result = [0.0]
        def split(lo, hi):
            if self.second_bound(lo, hi, outer)*(hi-lo)**2/2 <= tolerance:
                result.append(hi)
            else:
                mid = (lo+hi)/2
                split(lo, mid)
                split(mid, hi)
        split(0, self.length)
        if len(result) > 8192:
            _fail("the body fit exceeds the station budget.")
        return np.asarray(result)

    def fourth_bound(self, lo, hi, outer=False):
        """Whole-interval derivative bound for the cubic Hermite remainder."""
        A, B, C = self.coefficients
        def powers(a,b,c,p):
            qmin = a+2*b*lo+c*lo*lo
            qmax = a+2*b*hi+c*hi*hi
            u = 2*(b+c*hi)
            v = 2*c
            def term(falling, power, factors):
                return abs(falling)*qmin**power*factors
            p2, p3, p4 = p*(p-1), p*(p-1)*(p-2), p*(p-1)*(p-2)*(p-3)
            return [max(qmin**p,qmax**p),
                term(p,p-1,u), term(p2,p-2,u*u)+term(p,p-1,v),
                term(p3,p-3,u**3)+term(3*p2,p-2,u*v),
                term(p4,p-4,u**4)+term(6*p3,p-3,u*u*v)+term(3*p2,p-2,v*v)]
        v = powers(A,B,C,0.5)
        if not outer:
            return v[4]
        f = powers(A+B*B,B+B*C,C+C*C,-0.5)
        z4 = self.wall*((B+C*hi)*f[4]+4*C*f[3])
        r4 = v[4]+self.wall*sum(math.comb(4,j)*v[j]*f[4-j] for j in range(5))
        return math.hypot(z4,r4)

    def fit_stations(self, outer=False):
        result = [0.0]
        def split(lo,hi):
            if self.fourth_bound(lo,hi,outer)*(hi-lo)**4/384 <= FIT_TOL_MM:
                result.append(hi)
            else:
                mid = (lo+hi)/2
                split(lo,mid)
                split(mid,hi)
        split(0,self.length)
        if len(result) > 8192:
            _fail("the body fit exceeds the station budget.")
        return np.asarray(result)

    def metadata(self):
        return {"mouthRoundover": {"contractRevision":1, "radiusMm":self.radius,
                "wallThicknessMm":self.wall, "mouthZMm":self.length,
                "mouthRadiusMm":float(self.body(self.length)[0][1]),
                "betaDeg":math.degrees(self.beta), "centerZRmm":self.center.tolist(),
                "sweepDeg":math.degrees(self.sweep), "boundsMm":self.bounds,
                "fingerprint":self.fingerprint, "bodyFitBoundMm":FIT_TOL_MM}}


def resolve(config, params, allow_large_mesh=None):
    from .config_builder import ResolvedGeometry, _mesh_density_from_config, _bool
    from .geometry import _RoundoverPointGridHornGeometry
    if allow_large_mesh is not None and type(allow_large_mesh) is not bool:
        _fail("allow_large_mesh override must be a boolean or None.")
    model = Roundover.from_params(params)
    z = np.linspace(0, model.length, 33)
    phi = np.linspace(0, model.sweep, 65)
    az = np.linspace(0, 2*math.pi, 64, endpoint=False)
    def grid(outer):
        meridian = np.vstack((model.body(z, outer)[0], model.arc(phi, outer)[1:]))
        return np.stack((np.cos(az)[:,None]*meridian[:,1], np.sin(az)[:,None]*meridian[:,1],
                         np.broadcast_to(meridian[:,0], (len(az),len(meridian)))), axis=-1)
    geometry = _RoundoverPointGridHornGeometry(inner_points=grid(False), outer_points=grid(True),
        wall_thickness_mm=model.wall, source_shape=0, source_radius_mm=-1, source_curv=0,
        closed=True, symmetry_planes=(), vertical_offset_mm=model.offset, roundover=model)
    return ResolvedGeometry(geometry, _mesh_density_from_config(config, allow_large_mesh=allow_large_mesh),
        FORMULA, "freestanding", "1234", None, _bool(config.get("mesh", {}), names=("scale_to_metres",), default=True), model.metadata())
