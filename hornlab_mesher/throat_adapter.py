"""Native circular throat adapters and their canonical meridian construction."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np


CONTROLS = (
    "driver_exit_diameter_mm", "exit_half_angle_deg", "length_mm", "join_t",
    "driver_handle_mm", "body_handle_mm",
)


def _conic_residual_bound(poles, weights, *, length, join_z, adapter_length, r0, k, tan_a, tan_a0):
    """Bound the whole rational conic's radial error using Bernstein algebra.

    Homogenize the implicit equation with its positive quadratic denominator.
    Its quartic Bernstein coefficients bound the residual over the complete
    interval, without solving roots or checking sampled stations. Decimal
    arithmetic keeps cancellation in that certificate below its error margin.
    """
    with localcontext() as context:
        context.prec = 80
        D = lambda value: Decimal.from_float(float(value))
        W = [D(value) for value in weights]
        Z = [W[i]*(D(poles[i, 0])-D(adapter_length)+D(join_z)) for i in range(3)]
        X = [W[i]*(D(poles[i, 1])-D(r0)) for i in range(3)]

        def product(left, right):
            return [sum((Decimal(math.comb(2, i)*math.comb(2, j))*left[i]*right[j]
                         for i in range(3) for j in range(3) if i+j == n), Decimal(0))
                    /Decimal(math.comb(4, n)) for n in range(5)]

        xx, xw, zw, zz = product(X, X), product(X, W), product(Z, W), product(Z, Z)
        kr = D(k)*D(r0)
        residual = [xx[i]+2*kr*xw[i]-2*kr*D(tan_a0)*zw[i]-D(tan_a)**2*zz[i] for i in range(5)]
        arithmetic_margin = Decimal("1e-60")*max(Decimal(1), *(abs(value) for values in (xx, xw, zw, zz) for value in values))
        # Factor F(r)-F(r_exact): the second factor is the sum of the
        # positive shifted radii. Convex controls bound the actual side.
        denominator = kr + min(D(row[1])-D(r0)+kr for row in poles)
        if denominator <= 0:
            return math.inf
        return float((max(abs(value) for value in residual)+arithmetic_margin)/(min(W)**2*denominator))


def _number(value: Any, name: str) -> float:
    if isinstance(value, (bool, str)) or not np.isscalar(value):
        raise ValueError(f"Curved adapter refused: {name} must be a finite numeric scalar.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Curved adapter refused: {name} must be a finite numeric scalar.") from exc
    if not math.isfinite(number):
        raise ValueError(f"Curved adapter refused: {name} must be a finite numeric scalar.")
    return number


def normalize_adapter(value: Any) -> dict[str, Any] | None:
    """Validate the discriminant without changing an absent/off construction."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Curved adapter refused: throat_adapter must be an object.")
    unknown = set(value).difference({"mode", "contract_revision", *CONTROLS})
    if unknown:
        raise ValueError("Curved adapter refused: unknown field(s): " + ", ".join(sorted(unknown)))
    mode = value.get("mode")
    if not isinstance(mode, str) or mode not in {"off", "authored"}:
        raise ValueError("Curved adapter refused: mode must be off or authored.")
    revision = value.get("contract_revision", 1)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision != 1:
        raise ValueError("Curved adapter refused: unsupported contract_revision.")
    if mode == "off":
        return None
    missing = set(CONTROLS).difference(value)
    if missing:
        raise ValueError("Curved adapter refused: missing control(s): " + ", ".join(sorted(missing)))
    out = {"mode": "authored", "contract_revision": 1}
    out.update({key: _number(value[key], key) for key in CONTROLS})
    for key in ("driver_exit_diameter_mm", "length_mm", "driver_handle_mm", "body_handle_mm"):
        if not 1e-6 < out[key] <= 1e6:
            raise ValueError(f"Curved adapter refused: {key} must be positive and within the certified numeric range.")
    if not -89.0 < out["exit_half_angle_deg"] < 89.0:
        raise ValueError("Curved adapter refused: exit_half_angle_deg must be between -89 and 89 degrees.")
    if not 0.0 <= out["join_t"] < 1.0 - 1e-9:
        raise ValueError("Curved adapter refused: join_t must satisfy 0 <= join_t < 1 with a resolvable retained body.")
    return out


@dataclass(frozen=True)
class AdapterMeridian:
    """One cubic followed by a translated, retained OSSE conic.

    Composite sampling uses s in [0,1], with the exact join at s=.5. On the
    first half u=2s; on the second t=join_t+(2s-1)*(1-join_t). These are
    parameter coordinates, not axial fractions or arc-length fractions.
    """

    payload: Mapping[str, Any]
    cubic: np.ndarray
    body_poles: np.ndarray
    body_weights: np.ndarray
    length: float
    r0: float
    k: float
    tan_a: float
    tan_a0: float
    fingerprint: str
    identity: Mapping[str, Any]

    def base(self, t: Any) -> tuple[np.ndarray, np.ndarray]:
        z = np.asarray(t, dtype=np.float64) * self.length
        root = np.sqrt((self.k*self.r0)**2 + 2*self.k*self.r0*z*self.tan_a0 + z*z*self.tan_a**2)
        increment = 2*self.k*self.r0*z*self.tan_a0 + z*z*self.tan_a**2
        return z, self.r0 + increment/(root+self.k*self.r0)

    def base_tangent(self, t: float) -> np.ndarray:
        z = t*self.length
        root = math.sqrt((self.k*self.r0)**2 + 2*self.k*self.r0*z*self.tan_a0 + z*z*self.tan_a**2)
        return np.array([self.length, self.length*(self.k*self.r0*self.tan_a0+z*self.tan_a**2)/root])

    def evaluate(self, s: Any) -> tuple[np.ndarray, np.ndarray]:
        s = np.asarray(s, dtype=np.float64)
        if not np.all(np.isfinite(s)) or np.any((s < 0) | (s > 1)):
            raise ValueError("Curved adapter refused: composite parameter must be in [0,1].")
        u = np.minimum(s*2, 1.0)
        v = 1-u
        curve = (v[..., None]**3*self.cubic[0] + 3*v[..., None]**2*u[..., None]*self.cubic[1]
                 + 3*v[..., None]*u[..., None]**2*self.cubic[2] + u[..., None]**3*self.cubic[3])
        t = self.payload["join_t"] + np.maximum(2*s-1, 0)*(1-self.payload["join_t"])
        z, r = self.base(t)
        z_join = self.payload["join_t"]*self.length
        z = self.payload["length_mm"] + z-z_join
        return np.where(s <= .5, curve[..., 0], z), np.where(s <= .5, curve[..., 1], r)

    def preview_chord_bound(self, stations: Any, angles: Any, scale: float) -> float:
        """Upper bound the positional error of a circular triangle lattice.

        Each meridian chord has error <= max||C''||*delta**2/8. The radial
        loss of a triangle inside a circular wedge is at most
        max(radius)*(1-cos(delta_phi/2)). Their sum bounds distance to the
        analytic surface, including intervals without interior master samples.
        """
        stations = np.asarray(stations, dtype=np.float64)
        angles = np.asarray(angles, dtype=np.float64)
        if .5 not in stations or stations[0] != 0 or stations[-1] != 1:
            raise ValueError("Curved adapter refused: preview omitted a mandatory station.")
        second0 = 6*(self.cubic[2]-2*self.cubic[1]+self.cubic[0])
        second1 = 6*(self.cubic[3]-2*self.cubic[2]+self.cubic[1])
        meridian_error = 0.
        for left, right in zip(stations[:-1], stations[1:]):
            if right <= .5:
                u0, u1 = 2*left, 2*right
                d0, d1 = (1-u0)*second0+u0*second1, (1-u1)*second0+u1*second1
                error = max(np.linalg.norm(d0), np.linalg.norm(d1))*(u1-u0)**2/8
            else:
                t = self.payload["join_t"]+(2*left-1)*(1-self.payload["join_t"])
                z = t*self.length
                root = math.sqrt((self.k*self.r0)**2+2*self.k*self.r0*z*self.tan_a0+z*z*self.tan_a**2)
                curvature_bound = (self.k*self.r0)**2*(self.tan_a**2-self.tan_a0**2)/root**3
                dz = 2*(right-left)*(1-self.payload["join_t"])*self.length
                error = curvature_bound*dz*dz/8
            meridian_error = max(meridian_error, float(error))
        gaps = np.diff(np.append(angles, angles[0]+2*math.pi))
        max_gap = float(np.max(gaps))
        radius_bound = max(float(np.max(self.cubic[:, 1])), float(self.body_poles[-1, 1]))
        bound = scale*(meridian_error+radius_bound*2*math.sin(max_gap/4)**2)
        built_size = max(1., radius_bound*scale, float(self.body_poles[-1, 0])*scale,
                         abs(float(self.identity["vertical_offset_mm"]))+radius_bound*scale)
        # Render vertices use binary32; include their coordinate rounding.
        return bound + math.sqrt(3)*np.finfo(np.float32).eps*built_size + 1e-10*built_size


def resolve_adapter(params: Mapping[str, Any]) -> AdapterMeridian | None:
    payload = normalize_adapter(params.get("throat_adapter"))
    if payload is None:
        return None
    if str(params.get("type", "OSSE")).upper() != "OSSE":
        raise ValueError("Curved adapter refused: authored mode currently requires scalar OSSE profile 1.")
    values = {key: _number(params.get(key, default), key) for key, default in
              (("L", 120), ("r0", 12.7), ("k", 1), ("a", 60), ("a0", 15.5), ("scale", 1))}
    for key in ("L", "r0", "k", "scale"):
        if not 1e-6 < values[key] <= 1e6:
            raise ValueError(f"Curved adapter refused: {key} is outside the certified positive numeric range.")
    if max(values["L"], values["r0"]) > 1e4 or values["k"] > 100 or not 1e-3 <= values["scale"] <= 100:
        raise ValueError("Curved adapter refused: body dimensions, k or scale exceed the certified conic range.")
    if not 0 <= values["a0"] < values["a"] < 80 or values["a"] < 1 or values["a"]-values["a0"] < 1e-6:
        raise ValueError("Curved adapter refused: the certified conic domain requires 0 <= a0 < a < 80 degrees and a >= 1 degree.")
    for key in ("s", "h", "rot", "s1", "s2", "throatExtLength", "throatExtAngle", "slotLength",
                "morphTarget", "gcurveType", "wallThickness", "encDepth", "sourceCurv"):
        if _number(params.get(key, 0), key) != 0:
            raise ValueError(f"Curved adapter refused: {key} has no qualified construction in authored mode.")
    if _number(params.get("throatProfile", params.get("throat_profile", 1)), "throat profile") != 1:
        raise ValueError("Curved adapter refused: authored mode requires profile 1.")
    if _number(params.get("sourceShape", 1), "sourceShape") != 0:
        raise ValueError("Curved adapter refused: authored mode currently requires a flat source (source_shape=0).")
    if params.get("lookupProfile") is not None:
        raise ValueError("Curved adapter refused: lookup profiles are not supported.")
    if str(params.get("quadrants", "1234")) != "1234":
        raise ValueError("Curved adapter refused: authored mode currently requires the full circular domain.")
    cross = params.get("profileSystem", {}).get("crossSection", {})
    if _number(cross.get("exponent", 2), "cross-section exponent") != 2 or _number(cross.get("aspectRatio", 1), "aspect ratio") != 1:
        raise ValueError("Curved adapter refused: authored mode requires a circular cross section.")
    if params.get("subdomainSlices") not in (None, "", [], ()) or params.get("interfaceOffset") not in (None, "", 0, 0.0):
        raise ValueError("Curved adapter refused: subdomain interfaces are not supported.")
    if params.get("_athLengthMode") is not None:
        raise ValueError("Curved adapter refused: authored mode requires native body-length semantics.")
    L, r0, k = values["L"], values["r0"], values["k"]
    ta, t0 = math.tan(math.radians(values["a"])), math.tan(math.radians(values["a0"]))
    tJ, A = payload["join_t"], payload["length_mm"]
    zJ = tJ*L
    rootJ = math.sqrt((k*r0)**2+2*k*r0*zJ*t0+zJ*zJ*ta*ta)
    rJ = r0+(2*k*r0*zJ*t0+zJ*zJ*ta*ta)/(rootJ+k*r0)
    slopeJ = (k*r0*t0+zJ*ta*ta)/rootJ
    tangent = np.array([1., slopeJ])/math.hypot(1, slopeJ)
    alpha = math.radians(payload["exit_half_angle_deg"])
    cubic = np.array([[0, payload["driver_exit_diameter_mm"]/2],
                      [payload["driver_handle_mm"]*math.cos(alpha), payload["driver_exit_diameter_mm"]/2+payload["driver_handle_mm"]*math.sin(alpha)],
                      [A-payload["body_handle_mm"]*tangent[0], rJ-payload["body_handle_mm"]*tangent[1]], [A, rJ]])
    # Convex-hull certificates: all radius Bernstein coefficients are positive,
    # and every axial derivative coefficient 3*(z[i+1]-z[i]) is positive.
    # The conservative margin dominates ordinary roundoff in this bounded
    # coefficient range; unresolved near-contact controls are refused.
    magnitude = max(1., float(np.max(np.abs(cubic))), L, k*r0, values["scale"])
    margin = 1e-9*magnitude
    if not np.all(np.isfinite(cubic)) or np.min(cubic[:, 1]) <= margin or np.min(np.diff(cubic[:, 0])) <= margin:
        raise ValueError("Curved adapter refused: handles cross, leave the adapter span or reach the axis; validity could not be certified.")
    if (1-tJ)*L <= margin or min(r0, rJ)*values["scale"] <= margin:
        raise ValueError("Curved adapter refused: validity could not be certified at the requested tolerance.")
    # For z>=0, positive k/r0 and nonnegative a0 imply root>=k*r0,
    # hence radius>=r0. Body z'=L>0. Cubic and body occupy disjoint open
    # axial domains (0,A) and (A,A+L*(1-tJ)); only the intended join touches.
    # Revolution therefore has nonzero Jacobian and no nonlocal contacts.
    K = (k*r0)**2*(1-(t0/ta)**2)
    if not K > 1e-9*(k*r0)**2:
        raise ValueError("Curved adapter refused: the conic is too close to degeneracy to certify.")
    rootM = math.sqrt((k*r0)**2+2*k*r0*L*t0+L*L*ta*ta)
    rM = r0+(2*k*r0*L*t0+L*L*ta*ta)/(rootM+k*r0)
    centerZ = -k*r0*t0/(ta*ta)
    eta0 = math.asinh(ta*(zJ-centerZ)/math.sqrt(K))
    eta1 = math.asinh(ta*(L-centerZ)/math.sqrt(K))
    weight = math.cosh((eta1-eta0)/2)
    # Conic midpoint identity avoids subtracting nearly equal endpoint slopes.
    # Compute the small centre coefficient with tanh rather than 1-1/w**2.
    centre_fraction = math.tanh((eta1-eta0)/2)**2
    zPole = .5*(zJ+L)*(1-centre_fraction)+centerZ*centre_fraction
    rPole = .5*(rJ+rM)*(1-centre_fraction)+r0*(1-k)*centre_fraction
    poles = np.array([[A, rJ], [A+zPole-zJ, rPole], [A+L-zJ, rM]])
    if not np.all(np.isfinite(poles)) or np.min(poles[:, 1]) <= margin or np.min(np.diff(poles[:, 0])) <= margin or not 1 <= weight < 1e6:
        raise ValueError("Curved adapter refused: conic control bounds could not be certified.")
    # Bound built coordinates and cancellation ratios independently of sampling.
    # Within this range binary64's operation-error envelope is far below the
    # .01 mm surface budget. Near-axis and unresolved short spans are excluded.
    placement = _number(params.get("verticalOffset", 0), "verticalOffset")
    if max(float(np.max(np.abs(cubic))), float(np.max(np.abs(poles))))*values["scale"] + abs(placement) > 1e5:
        raise ValueError("Curved adapter refused: scaled coordinates exceed the certified conic range.")
    if min(float(np.min(cubic[:, 1])), float(np.min(poles[:, 1])),
           float(np.min(np.diff(cubic[:, 0]))), float(np.min(np.diff(poles[:, 0]))),
           (1-tJ)*L)*values["scale"] <= 1e-4:
        raise ValueError("Curved adapter refused: scaled feature clearance could not be certified.")
    expected_angles = [alpha, math.atan(slopeJ), math.atan(slopeJ),
                       math.atan((k*r0*t0+L*ta*ta)/rootM)]
    derivative_vectors = [cubic[1]-cubic[0], cubic[3]-cubic[2],
                          poles[1]-poles[0], poles[2]-poles[1]]
    coordinate_size = max(1., float(np.max(np.abs(cubic))), float(np.max(np.abs(poles))))
    for vector, angle in zip(derivative_vectors, expected_angles):
        angle_error = abs(math.atan2(float(vector[1]), float(vector[0]))-angle)
        roundoff_bound = 64*np.finfo(float).eps*coordinate_size/float(np.linalg.norm(vector))
        if angle_error+roundoff_bound > 5e-10:
            raise ValueError("Curved adapter refused: endpoint tangent precision could not be certified.")
    conic_error = _conic_residual_bound(poles, np.array([1., weight, 1.]), length=L,
                                      join_z=zJ, adapter_length=A, r0=r0, k=k, tan_a=ta, tan_a0=t0)
    if conic_error*values["scale"] + 1e-10*coordinate_size*values["scale"] > 1e-4:
        raise ValueError("Curved adapter refused: continuous conic position precision could not be certified.")
    identity = {"adapter": payload, "body": {key: values[key] for key in ("L", "r0", "k", "a", "a0")},
                "scale": values["scale"], "vertical_offset_mm": placement,
                "construction": "circular-cubic-conic-v1", "source": "flat-driver-rim-v1"}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    weights = np.array([1., weight, 1.])
    for array in (cubic, poles, weights):
        array.setflags(write=False)
    return AdapterMeridian(payload, cubic, poles, weights, L, r0, k, ta, t0, fingerprint, identity)
