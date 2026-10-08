"""Bounded native circular bodies with an absolute axial scale."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config_parser import ConfigError

FORMULA = "OSSE-AXIAL"
KEY = "axial_scale"
FIT_TOL_MM = 0.0001


def _fail(message):
    raise ConfigError("Absolute axial scale refused: " + message)


def _number(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(float(value))
    except (ValueError, OverflowError, TypeError):
        valid = False
    if not valid:
        _fail(f"{name} must be a finite scalar number.")
    return float(value)


def configuration(config, resolver):
    """Resolve active inputs internally; leave absent ordinary inputs untouched."""
    profile = config.get("profile", {})
    profile = profile if isinstance(profile, Mapping) else {}
    names = [section[key] for section in (config, profile) for key in ("formula", "type") if key in section]
    active = any(isinstance(name, str) and name.upper() == FORMULA for name in names)
    misplaced = any(isinstance(config.get(section), Mapping) and KEY in config[section] for section in ("profile", "parameters", "mesh", "source", "Source"))
    if misplaced:
        _fail(f"{KEY} must be supplied at the root.")
    if KEY not in config and not active:
        return None
    if not active:
        _fail(f"a supplied {KEY} requires formula {FORMULA} so older readers refuse it.")
    if KEY not in config:
        _fail(f"formula {FORMULA} requires {KEY}.")
    for name in names:
        if not isinstance(name, str) or name.upper() != FORMULA:
            _fail(f"every supplied formula and type must be {FORMULA}.")
    for key in ("profile", "mesh", "source"):
        if key in config and not isinstance(config[key], Mapping):
            _fail(f"{key} must be an object.")
    mesh, source = config.get("mesh", {}), config.get("source", {})
    allowed = {
        "root": {"formula", "type", KEY, "profile", "mesh", "source", "mode", "scale", "quadrants", "vertical_offset_mm", "output", "path", "output_path"},
        "profile": {"formula", "type", "L_mm", "r0_mm", "a_deg", "a0_deg", "k", "s", "n", "q"},
        "mesh": {"wall_thickness_mm", "angular_segments", "length_segments", "throat_res_mm", "mouth_res_mm", "rear_res_mm", "scale_to_metres", "topology_mode", "surface_fit", "max_triangles", "allow_large_mesh", "quadrants", "vertical_offset_mm"},
        "source": {"source_shape", "source_radius_mm", "source_curv"},
    }
    for name, section in (("root", config), ("profile", profile), ("mesh", mesh), ("source", source)):
        extras = set(section)-allowed[name]
        if extras:
            _fail(f"unsupported {name} controls: {', '.join(sorted(map(str, extras)))}.")
        for key in set(section)-{"formula", "type", "profile", "mesh", "source", "mode", "quadrants", "topology_mode", "surface_fit", "scale_to_metres", "allow_large_mesh", "output", "path", "output_path"}:
            _number(section[key], key)
    output = config.get("output", {})
    if not isinstance(output, Mapping):
        _fail("output must be an object.")
    if set(output)-{"path", "output_path"}:
        _fail("output supports only path and output_path routing controls.")
    for section in (config, output):
        for key in ("path", "output_path"):
            if key in section and section[key] is not None and not isinstance(section[key], str):
                _fail(f"{key} must be a string or null.")
    for section in (config, mesh):
        if "quadrants" in section and not ((type(section["quadrants"]) is int and section["quadrants"] == 1234) or (type(section["quadrants"]) is str and section["quadrants"] == "1234")):
            _fail("quadrants must be exactly 1234 for full-circle coverage.")
    if "vertical_offset_mm" in config and "vertical_offset_mm" in mesh and config["vertical_offset_mm"] != mesh["vertical_offset_mm"]:
        _fail("duplicate vertical_offset_mm controls must agree.")
    for key in ("angular_segments", "length_segments", "max_triangles"):
        if key in mesh and (mesh[key] <= 0 or not float(mesh[key]).is_integer() or mesh[key] > 10_000_000):
            _fail(f"{key} must be a positive integer no larger than 10000000.")
    for key in ("allow_large_mesh", "scale_to_metres"):
        if key in mesh and type(mesh[key]) is not bool:
            _fail(f"{key} must be a boolean.")
    for key in ("throat_res_mm", "mouth_res_mm", "rear_res_mm"):
        if key in mesh and not 0.01 <= mesh[key] <= 10000:
            _fail(f"{key} must be in [0.01, 10000] mm.")
    if config.get("mode", "bare") != "bare" or mesh.get("wall_thickness_mm", 0) != 0:
        _fail("only bare models with zero wall thickness are qualified.")
    clean = copy.deepcopy(dict(config))
    clean["mode"] = "bare"
    axis = _number(clean.pop(KEY), KEY)
    clean["formula"] = "OSSE"
    clean.pop("type", None)
    clean.setdefault("profile", {}).pop("formula", None)
    clean["profile"].pop("type", None)
    clean.setdefault("mesh", {}).setdefault("wall_thickness_mm", 0)
    clean.setdefault("source", {}).setdefault("source_shape", 0)
    params, _, mode = resolver(clean)
    if mode != "bare" or params.get("wallThickness") != 0:
        _fail("only bare models with zero wall thickness are qualified.")
    if params.get("s", 0) != 0:
        _fail("termination must be zero.")
    if params.get("sourceShape") != 0 or params.get("sourceCurv", 0) != 0 or params.get("sourceRadius", -1) != -1:
        _fail("only the native flat source is qualified.")
    if mesh.get("topology_mode", "acoustic") != "acoustic" or mesh.get("surface_fit", "auto") not in {"auto", "interpolate"}:
        _fail("acoustic topology and an interpolating fit are required.")
    params["absoluteAxialScale"] = axis
    params["type"] = FORMULA
    model = AxialModel.from_params(params)
    params["axialScaleFingerprint"] = model.fingerprint
    return params, FORMULA, mode


@dataclass(frozen=True)
class AxialModel:
    length: float
    r0: float
    slope: float
    throat_slope: float
    k: float
    xy_scale: float
    axial_scale: float
    offset: float = 0.0

    @classmethod
    def from_params(cls, p):
        xy, axis = _number(p.get("scale", 1), "scale"), _number(p["absoluteAxialScale"], KEY)
        if not (0.01 <= xy <= 10 and 0.01 <= axis <= 10 and 0.1 <= axis/xy <= 10):
            _fail("both scales must be in [0.01, 10], with axial_scale/scale in [0.1, 10].")
        a, a0 = _number(p["a"], "a"), _number(p["a0"], "a0")
        if not (5 <= a <= 75 and 0 <= a0 < a):
            _fail("opening angles require 5 <= a <= 75 and 0 <= a0 < a degrees.")
        model = cls(_number(p["L"], "L")*axis, _number(p["r0"], "r0")*xy,
                    math.tan(math.radians(a))*xy/axis, math.tan(math.radians(a0))*xy/axis,
                    _number(p["k"], "k"), xy, axis, _number(p.get("verticalOffset", 0), "vertical_offset_mm"))
        if not (1 <= model.length <= 2000 and 1 <= model.r0 <= 200 and 0.5 <= model.k <= 10 and abs(model.offset) <= 10000):
            _fail("physical length, throat radius, k or finished placement is outside the qualified domain.")
        if float(model.body(model.length)[0][1]) > 10000:
            _fail("physical mouth radius exceeds 10000 mm.")
        return model

    @property
    def coefficients(self):
        return (self.k*self.r0)**2, self.k*self.r0*self.throat_slope, self.slope**2

    def body(self, z):
        z = np.asarray(z, dtype=np.float64)
        A, B, C = self.coefficients
        root = np.sqrt(A+2*B*z+C*z*z)
        r = root+self.r0*(1-self.k)
        derivative = (B+C*z)/root
        return np.stack((z, r), axis=-1), np.stack((np.ones_like(z), derivative), axis=-1)

    def second_bound(self, lo, hi):
        A, B, C = self.coefficients
        return self.curvature_numerator/(A+2*B*lo+C*lo*lo)**1.5

    @property
    def curvature_numerator(self):
        return (self.k*self.r0)**2*(self.slope-self.throat_slope)*(self.slope+self.throat_slope)

    def fourth_bound(self, lo, hi):
        """Absolute fourth derivative bound of sqrt(quadratic) on [lo,hi]."""
        A, B, C = self.coefficients
        q = A+2*B*lo+C*lo*lo
        u, v = 2*(B+C*hi), 2*C
        return 15/16*q**-3.5*u**4+9/4*q**-2.5*u*u*v+3/4*q**-1.5*v*v

    def stations(self, tolerance, *, fit=False):
        result = [0.0]
        def split(lo, hi):
            bound = self.fourth_bound(lo, hi)*(hi-lo)**4/384 if fit else self.second_bound(lo, hi)*(hi-lo)**2/8
            if bound <= tolerance:
                result.append(hi)
                if len(result) > 8192:
                    _fail("continuous sampling exceeds the station budget.")
            else:
                mid = (lo+hi)/2
                split(lo, mid)
                split(mid, hi)
        split(0, self.length)
        return np.asarray(result)

    @property
    def bounds(self):
        radius = float(self.body(self.length)[0][1])
        return [[-radius, self.offset-radius, 0.0], [radius, self.offset+radius, self.length]]

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps({"contract":1, **self.__dict__}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def metadata(self):
        return {"absoluteAxialScale": {"contractRevision":1, "xyScale":self.xy_scale,
            "axialScale":self.axial_scale, "lengthMm":self.length, "throatRadiusMm":self.r0,
            "sourceAngleDeg":math.degrees(math.atan(self.throat_slope)), "boundsMm":self.bounds,
            "fingerprint":self.fingerprint, "bodyFitBoundMm":FIT_TOL_MM}}


def resolve(config, params, allow_large_mesh=None):
    from .config_builder import ResolvedGeometry, _mesh_density_from_config
    from .geometry import _AxialPointGridHornGeometry
    if allow_large_mesh is not None and type(allow_large_mesh) is not bool:
        _fail("allow_large_mesh override must be a boolean or None.")
    model = AxialModel.from_params(params)
    meridian = model.body(np.linspace(0, model.length, 33))[0]
    azimuth = np.linspace(0, 2*math.pi, 64, endpoint=False)
    grid = np.stack((np.cos(azimuth)[:,None]*meridian[:,1], np.sin(azimuth)[:,None]*meridian[:,1],
                     np.broadcast_to(meridian[:,0], (64,33))), axis=-1)
    geometry = _AxialPointGridHornGeometry(inner_points=grid, wall_thickness_mm=0,
        source_shape=0, source_radius_mm=-1, source_curv=0,
        source_auto_angle_deg=math.degrees(math.atan(model.throat_slope)), closed=True,
        symmetry_planes=(), vertical_offset_mm=model.offset, axial_model=model)
    return ResolvedGeometry(geometry, _mesh_density_from_config(config, allow_large_mesh=allow_large_mesh),
        FORMULA, "bare", "1234", None, config.get("mesh", {}).get("scale_to_metres", True), model.metadata())
