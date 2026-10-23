"""Canonical circular mouth lip and its bounded first native configuration."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .axial_scale import AxialModel
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
    from .native_boundary import native_features
    # This construction owns every implemented OSSE-NATIVE composite with a
    # roundover; the rule table has already refused the others.
    features = native_features(config) if formula == "OSSE-NATIVE" else frozenset()
    composite = "roundover" in features
    marker = "OSSE-NATIVE" if composite else FORMULA
    supplied_marker = any(
        isinstance(section.get(key), str) and section[key].upper() == FORMULA
        for section in (config, profile) for key in ("formula", "type"))
    if radius == 0 and not supplied_marker:
        return None
    if radius <= 0:
        _fail("radius must be finite and positive.")
    if formula != marker:
        _fail(f"an active radius requires formula {marker} so older readers refuse it.")
    for name in ("profile", "mesh", "source"):
        if name in config and not isinstance(config[name], Mapping):
            _fail(f"{name} must be an object.")
    # Validate inactive editor state before removing it from the body recipe.
    if "throat_adapter" in config and "throat_adapter" in profile:
        _fail("throat_adapter must be supplied in only one location.")
    from .throat_adapter import normalize_adapter
    adapter = None
    for section in (config, profile):
        if "throat_adapter" in section:
            try:
                adapter = normalize_adapter(section["throat_adapter"])
            except ValueError as exc:
                _fail(str(exc))
            if adapter is not None and "adapter" not in features:
                _fail("combined active throat adapter and mouth roundover are not supported.")
    allowed = {
        "root": {"formula", "type", "profile", "mesh", "source", "mode", "scale", "quadrants", "vertical_offset_mm", "throat_adapter"}|({"axial_scale"} if "axial" in features else set()),
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
            "root":{"scale","vertical_offset_mm","axial_scale"},
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
            if key in section and (not isinstance(section[key], str) or section[key].upper() != marker):
                _fail(f"every supplied {key} must be {marker}.")
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
    clean.pop("axial_scale", None)
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
    if "axial" in features:
        params["absoluteAxialScale"] = _number(config["axial_scale"], "axial_scale")
    if adapter is not None:
        params["throat_adapter"] = adapter
    if composite:
        params["type"] = marker
    model = Roundover.from_params(params)
    params["mouthRoundoverFingerprint"] = model.fingerprint
    return params, marker, mode


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

    formula = FORMULA
    # Class constants, not fields: the legacy record and fingerprint keep their
    # field set. A throat composite overrides them with its own fields.
    start = 0.0
    throat = None

    @classmethod
    def from_params(cls, p):
        scale = _number(p.get("scale", 1), "scale")
        if not 0.01 <= scale <= 10:
            _fail("scale must be in [0.01, 10].")
        if p.get("throat_adapter"):
            model = AdapterRoundover.construct(p)
        elif "absoluteAxialScale" not in p:
            model = cls(float(p["L"])*scale, float(p["r0"])*scale,
                        math.radians(float(p["a"])), math.radians(float(p["a0"])),
                        float(p["k"]), float(p["wallThickness"]),
                        float(p["mouthRoundoverRadiusMm"]), float(p.get("verticalOffset", 0)))
        else:
            # Contract order: the axial feature maps the body (its own model,
            # domain and mouth limit), then the offset wall and lip are built
            # in finished millimetres against it.
            body = AxialModel.from_params(p)
            model = AxialRoundover(body.length, body.r0,
                        math.radians(float(p["a"])), math.radians(float(p["a0"])),
                        body.k, float(p["wallThickness"]),
                        float(p["mouthRoundoverRadiusMm"]), body.offset,
                        body.xy_scale, body.axial_scale, body.slope, body.throat_slope)
        if not (1 <= model.length <= 2000 and 1 <= model.r0 <= 200 and 0.5 <= model.k <= 10
                and 0 <= model.a0 < model.a <= math.radians(75) and model.a >= math.radians(5)):
            _fail("body coefficients are outside the qualified circular domain.")
        if not (0.1 <= model.wall < model.radius <= 200 and model.radius-model.wall >= 0.1):
            _fail("radius must exceed the positive wall thickness by at least 0.1 mm.")
        if abs(model.offset) > 10000:
            _fail("vertical placement exceeds 10000 mm.")
        if model.wall*model.second_bound(model.start, model.length, False) >= 0.4:
            _fail("the outer offset approaches a body curvature fold.")
        if model.center[0] <= 0.1:
            _fail("the lip folds or intersects the rear closure.")
        if model.throat is not None:
            model.certify_throat()
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

    @property
    def source_radius(self):
        return self.r0

    @property
    def rear_start(self):
        """Outer wall at the throat plane, where the flat rear return starts."""
        return self.body(0, True)[0]

    @property
    def fit_tolerance(self):
        return FIT_TOL_MM

    def stations(self, tolerance, outer=False):
        result = [self.start]
        def split(lo, hi):
            if self.second_bound(lo, hi, outer)*(hi-lo)**2/2 <= tolerance:
                result.append(hi)
            else:
                mid = (lo+hi)/2
                split(lo, mid)
                split(mid, hi)
        split(self.start, self.length)
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
        result = [self.start]
        tolerance = self.fit_tolerance
        def split(lo,hi):
            if self.fourth_bound(lo,hi,outer)*(hi-lo)**4/384 <= tolerance:
                result.append(hi)
            else:
                mid = (lo+hi)/2
                split(lo,mid)
                split(mid,hi)
        split(self.start,self.length)
        if len(result) > 8192:
            _fail("the body fit exceeds the station budget.")
        return np.asarray(result)

    def metadata(self):
        return {"mouthRoundover": {**self._composite_metadata(), "contractRevision":1, "radiusMm":self.radius,
                "wallThicknessMm":self.wall, "mouthZMm":self.length,
                "mouthRadiusMm":float(self.body(self.length)[0][1]),
                "betaDeg":math.degrees(self.beta), "centerZRmm":self.center.tolist(),
                "sweepDeg":math.degrees(self.sweep), "boundsMm":self.bounds,
                "fingerprint":self.fingerprint, "bodyFitBoundMm":FIT_TOL_MM}}


    def _composite_metadata(self):
        return {}


@dataclass(frozen=True)
class AxialRoundover(Roundover):
    """Roundover x Axial: the same lip on a body mapped by Z and radius factors.

    Length, throat radius and conic slopes are AxialModel's physical values;
    the authored angles stay for the record. Wall and lip stay in finished mm.
    """
    xy_scale: float = 1.0
    axial_scale: float = 1.0
    slope: float = 0.0
    throat_slope: float = 0.0
    formula = "OSSE-NATIVE"
    # The mapped conic is AxialModel's: same fields, same coefficients.
    coefficients = AxialModel.coefficients

    @property
    def identity(self):
        return {"contractRevision":1, "formula":self.formula, "construction":"circular-osse-axial-lip-v1",
                "controls":dict(self.__dict__)}

    @property
    def fingerprint(self):
        # As for every composite: the hash of the recipe the STEP header carries.
        return hashlib.sha256(json.dumps(self.identity, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()

    def _composite_metadata(self):
        return {"formula":self.formula, "xyScale":self.xy_scale, "axialScale":self.axial_scale}

    def metadata(self):
        return {"construction_fingerprint":self.fingerprint, **super().metadata()}


def _hull(coefficients, lo, hi):
    """Bernstein hull of a power-basis polynomial on [lo,hi], widened for roundoff."""
    from .adapter_axial import restrict_controls
    a = np.atleast_1d(np.asarray(coefficients, dtype=np.float64))
    n = len(a)-1
    b = [sum(math.comb(j, i)/math.comb(n, i)*a[i] for i in range(j+1)) for j in range(n+1)]
    b = restrict_controls(b, lo, hi)
    guard = 1e-12*float(np.sum(np.abs(a)))
    return float(np.min(b))-guard, float(np.max(b))+guard


@dataclass(frozen=True)
class AdapterRoundover(AxialRoundover):
    """Adapter x Roundover (x Axial): the authored throat cubic, then the lip.

    The throat is the OSSE-ADAPTER-AXIAL PhysicalAdapter: its cubic on
    u in [0,1] from the driver rim at z=0 to the adapter join at ``start``.
    The retained conic is AxialModel's mapped body from the adapter join,
    translated by ``shift`` into composite z (see ``coefficients``), so every
    inherited body, offset, fit and lip expression applies unchanged and the
    lip is built in finished mm at the body mouth ``length``. The wall offset
    of the cubic is new: a cubic Hermite fit certified by a whole-interval
    fourth-derivative bound, as for the conic.
    """
    start: float = 0.0
    shift: float = 0.0
    features: tuple = ()
    throat: object = None

    @classmethod
    def construct(cls, p):
        from .composite_meridian import mapped_throat
        # The same body, cubic, checks and shift as Adapter x Arc; the wall is the lip's.
        body, throat, _, shift = mapped_throat(p)
        start = float(throat.cubic[-1, 0])
        authored = "absoluteAxialScale" in p
        return cls(shift+body.length, body.r0, math.radians(float(p["a"])), math.radians(float(p["a0"])),
                   body.k, float(p["wallThickness"]), float(p["mouthRoundoverRadiusMm"]), body.offset,
                   body.xy_scale, body.axial_scale, body.slope, body.throat_slope, start, shift,
                   ("adapter", "axial", "roundover") if authored else ("adapter", "roundover"), throat)

    @property
    def coefficients(self):
        # The mapped conic in composite z = u+shift; C*A-B*B is unchanged.
        # B' = B-C*shift may be negative, but the conic's vertex (u = -B/C <= 0)
        # stays at or before start, so the inherited bounds' monotone q and
        # B'+C*z >= 0 assumptions hold on [start, length].
        A, B, C = AxialModel.coefficients.fget(self)
        s = self.shift
        return A+s*(C*s-2*B), B-C*s, C

    @property
    def identity(self):
        controls = {key: value for key, value in self.__dict__.items() if key not in ("throat", "features")}
        return {"contractRevision":1, "formula":self.formula, "construction":"circular-cubic-conic-lip-v1",
                "features":list(self.features), "adapter":dict(self.throat.payload),
                "body":self.throat.identity["body"], "controls":controls}

    @property
    def source_radius(self):
        return self.throat.driver_radius

    @property
    def rear_start(self):
        return self.cubic(0., True)[0]

    @property
    def join_gap(self):
        """Largest roundoff gap between the cubic and conic ends at the join."""
        inner = np.linalg.norm(self.body(self.start)[0]-self.throat.cubic[-1])
        outer = np.linalg.norm(self.body(self.start, True)[0]-self.cubic(1., True)[0])
        return float(max(inner, outer))

    @property
    def fit_tolerance(self):
        # Builders share one point at the join, so each fit spends its gap.
        return FIT_TOL_MM-self.join_gap-1e-10*max(1., self.length, self.center[1]+self.radius)

    def cubic(self, u, outer=False):
        """Throat cubic points and d/du derivatives; outer is the wall offset.

        The offset o = c+wall*n has o' = c'*(1-wall*kappa), with kappa the
        curvature signed towards the outer normal.
        """
        p, d, _ = self.throat.cubic_values(u)
        if not outer:
            return p, d
        kappa = self.cubic_curvature(u)
        normal = np.stack((-d[..., 1], d[..., 0]), axis=-1)/np.linalg.norm(d, axis=-1)[..., None]
        return p+self.wall*normal, d*(1-self.wall*kappa)[..., None]

    def cubic_curvature(self, u, outer=False):
        """Meridian curvature of the cubic (or its offset), signed towards the outer normal."""
        _, d, dd = self.throat.cubic_values(u)
        kappa = (d[..., 0]*dd[..., 1]-d[..., 1]*dd[..., 0])/np.linalg.norm(d, axis=-1)**3
        return kappa/(1-self.wall*kappa) if outer else kappa

    def _polynomials(self):
        """Power-basis z', r', z'', r'', speed squared g and curvature numerator."""
        P = np.polynomial.polynomial
        c = self.throat.cubic
        power = np.array([c[0], 3*(c[1]-c[0]), 3*(c[0]-2*c[1]+c[2]), -c[0]+3*c[1]-3*c[2]+c[3]])
        dz, dr = P.polyder(power[:, 0]), P.polyder(power[:, 1])
        ddz, ddr = P.polyder(dz), P.polyder(dr)
        return (dz, dr, ddz, ddr, P.polyadd(P.polymul(dz, dz), P.polymul(dr, dr)),
                P.polysub(P.polymul(dz, ddr), P.polymul(dr, ddz)))

    def offset_bounds(self, lo, hi):
        """Upper bounds of the second and fourth derivative norms of the outer
        cubic offset on [lo,hi].

        o = c+wall*N*h with N = (-r',z') quadratic, h = g**-1/2 and g = |c'|**2
        quartic. Faa di Bruno bounds the derivatives of h from the Bernstein
        hulls of g and its derivatives on the interval; the cubic's fourth
        derivative and N's third derivative vanish.
        """
        P = np.polynomial.polynomial
        dz, dr, ddz, ddr, g, _ = self._polynomials()
        gmin, gmax = _hull(g, lo, hi)
        if gmin <= 0:
            _fail("the throat cubic speed could not be bounded away from zero.")
        G = [max(abs(v) for v in _hull(P.polyder(g, j), lo, hi)) for j in range(1, 5)]
        f = [.5*gmin**-1.5, .75*gmin**-2.5, 1.875*gmin**-3.5, 6.5625*gmin**-4.5]
        h0 = gmin**-.5
        h1 = f[0]*G[0]
        h2 = f[1]*G[0]**2+f[0]*G[1]
        h3 = f[2]*G[0]**3+3*f[1]*G[0]*G[1]+f[0]*G[2]
        h4 = f[3]*G[0]**4+6*f[2]*G[0]**2*G[1]+3*f[1]*G[1]**2+4*f[1]*G[0]*G[2]+f[0]*G[3]
        n0 = math.sqrt(gmax)
        n1 = math.sqrt(max(0., _hull(P.polyadd(P.polymul(ddz, ddz), P.polymul(ddr, ddr)), lo, hi)[1]))
        n2 = math.hypot(float(P.polyder(ddz)[0]), float(P.polyder(ddr)[0]))
        second = n1+self.wall*(n2*h0+2*n1*h1+n0*h2)
        fourth = self.wall*(n0*h4+4*n1*h3+6*n2*h2)
        return second, fourth

    def cubic_fit_stations(self):
        result = [0.0]
        tolerance = self.fit_tolerance
        def split(lo, hi):
            if self.offset_bounds(lo, hi)[1]*(hi-lo)**4/384 <= tolerance:
                result.append(hi)
                if len(result) > 8192:
                    _fail("the throat offset fit exceeds the station budget.")
            else:
                mid = (lo+hi)/2
                split(lo, mid)
                split(mid, hi)
        split(0., 1.)
        return np.asarray(result)

    def certify_throat(self):
        """Refuse a throat whose offset wall could fold or touch another face.

        Inner and outer walls are graphs over z (positive axial hodograph,
        offset factor 1-wall*kappa >= 0.6). If every wall tangent angle lies
        in a range narrower than 90 degrees, every outer point lies above the
        inner graph at its own z, so the walls cannot cross. The cubic and
        its offset stay inside z <= start+wall and r <= cubic reach+wall
        (conservative: the offset's z is monotone, so it ends at start), which
        must clear the lip (z >= centre, r >= mouth radius) by 0.1 mm.
        """
        if not abs(self.throat.source_angle) <= 75:
            _fail("the adapter exit angle must be within 75 degrees of the axis.")
        hodograph = 3*np.diff(self.throat.cubic, axis=0)
        angles = np.arctan2(hodograph[:, 1], hodograph[:, 0])
        join = math.atan(float(self.body(self.start)[1][1]))
        if max(float(np.max(angles)), self.beta)-min(float(np.min(angles)), join) >= math.radians(85):
            _fail("the wall tangent turns 85 degrees or more; the outer offset cannot be certified clear of the inner wall.")
        _, _, _, _, g, X = self._polynomials()
        def regular(lo, hi, depth):
            numerator, speed = _hull(X, lo, hi)[1], _hull(g, lo, hi)[0]
            if numerator <= 0 or (speed > 0 and self.wall*numerator/speed**1.5 < 0.4):
                return True
            mid = (lo+hi)/2
            return depth > 0 and regular(lo, mid, depth-1) and regular(mid, hi, depth-1)
        if not regular(0., 1., 12):
            _fail("the outer offset approaches a throat curvature fold.")
        reach = self.throat.cubic_reach+self.wall
        if not (self.start+self.wall <= self.center[0]-.1 or reach <= float(self.body(self.length)[0][1])-.1):
            _fail("the lip must clear the throat adapter by 0.1 mm.")
        if reach > self.center[1]+self.radius:
            _fail("the throat adapter must stay within the lip's outer reach.")
        if self.join_gap > 1e-6:
            _fail("the adapter join could not be certified.")
        self.cubic_fit_stations()

    def _composite_metadata(self):
        return {**super()._composite_metadata(), "adapterJoinMm":self.throat.cubic[-1].tolist(),
                "sourceRadiusMm":self.source_radius, "sourceAngleDeg":self.throat.source_angle,
                "throatAdapter":dict(self.throat.payload)}


def resolve(config, params, allow_large_mesh=None):
    from .config_builder import ResolvedGeometry, _mesh_density_from_config, _bool
    from .geometry import _RoundoverPointGridHornGeometry
    if allow_large_mesh is not None and type(allow_large_mesh) is not bool:
        _fail("allow_large_mesh override must be a boolean or None.")
    model = Roundover.from_params(params)
    z = np.linspace(model.start, model.length, 33)
    phi = np.linspace(0, model.sweep, 65)
    az = np.linspace(0, 2*math.pi, 64, endpoint=False)
    def grid(outer):
        meridian = np.vstack((model.body(z, outer)[0], model.arc(phi, outer)[1:]))
        if model.throat is not None:
            meridian = np.vstack((model.cubic(np.linspace(0, 1, 17)[:-1], outer)[0], meridian))
        return np.stack((np.cos(az)[:,None]*meridian[:,1], np.sin(az)[:,None]*meridian[:,1],
                         np.broadcast_to(meridian[:,0], (len(az),len(meridian)))), axis=-1)
    geometry = _RoundoverPointGridHornGeometry(inner_points=grid(False), outer_points=grid(True),
        wall_thickness_mm=model.wall, source_shape=0, source_radius_mm=-1, source_curv=0,
        closed=True, symmetry_planes=(), vertical_offset_mm=model.offset, roundover=model)
    return ResolvedGeometry(geometry, _mesh_density_from_config(config, allow_large_mesh=allow_large_mesh),
        model.formula, "freestanding", "1234", None, _bool(config.get("mesh", {}), names=("scale_to_metres",), default=True), model.metadata())
