"""Core geometry for FREEFORM H/V spline profiles.

The design contract is that every cross-section outline hits the horizontal and
vertical profile axes exactly, station joins are C1 or better (the shape blend is
C2), and both meridians share one axial ``z`` span.  The implementation follows
``/Users/magnus/Code/hornlab-workspace/Waveguide Generator/docs/plans/260801-freeform-hv-spline-profiles.md``
(especially sections 2.1, 2.2, 2.3, and 2.5).

SciPy is deliberately imported only while constructing a profile.  Importing
this module therefore remains cheap and does not load SciPy.
"""

from __future__ import annotations

import json
import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .profile_common import _is_true, eval_param
from .profile_morph import _rounded_rect_radius


_INVERSION_SAMPLE_N = 4001
_FREEFORM_CACHE_MAX = 256
_FREEFORM_PARAM_KEYS = (
    "profileH",
    "profileV",
    "crossSections",
    "overshootPolicy",
    "a0",
)


@dataclass(frozen=True)
class _PlaneSpline:
    name: str
    anchors: np.ndarray
    anchor_u: np.ndarray
    spline: Any
    inverse_z: np.ndarray
    inverse_u: np.ndarray
    throat_angle_deg: float
    mouth_angle_deg: float

    def radii_at_z(self, z: np.ndarray) -> np.ndarray:
        flat_z = np.asarray(z, dtype=float).reshape(-1)
        u = np.interp(flat_z, self.inverse_z, self.inverse_u)
        radii = np.asarray(self.spline(u), dtype=float)[:, 1]
        return radii.reshape(np.asarray(z).shape)


@dataclass
class FreeformGeometry:
    """Validated pair of FREEFORM meridians and its shape-station schedule."""

    _profile_h: _PlaneSpline
    _profile_v: _PlaneSpline
    stations: list[dict[str, Any]]

    @property
    def length_mm(self) -> float:
        """Shared axial span from the throat plane to the mouth plane, in mm."""
        return float(self._profile_h.anchors[-1, 0] - self._profile_h.anchors[0, 0])

    def evaluate_radii(self, z_array: Any) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the H and V meridian radii at absolute axial coordinates."""
        z = np.asarray(z_array, dtype=float)
        if not np.all(np.isfinite(z)):
            raise ValueError("FREEFORM z queries must all be finite")
        z0 = float(self._profile_h.anchors[0, 0])
        z1 = float(self._profile_h.anchors[-1, 0])
        tol = 1.0e-12 * max(1.0, abs(z0), abs(z1))
        if np.any(z < z0 - tol) or np.any(z > z1 + tol):
            raise ValueError(
                f"FREEFORM z queries must lie in the shared [{z0:g}, {z1:g}] mm span"
            )
        return self._profile_h.radii_at_z(z), self._profile_v.radii_at_z(z)

    def cross_section_radius(self, phi_array: Any, t_scalar: float) -> np.ndarray:
        """Evaluate a station-blended outline at normalized axial position ``t``."""
        try:
            t = float(t_scalar)
        except (TypeError, ValueError) as exc:
            raise ValueError("FREEFORM cross-section t must be a finite scalar") from exc
        if not math.isfinite(t) or not (0.0 <= t <= 1.0):
            raise ValueError(f"FREEFORM cross-section t must be in [0, 1], got {t_scalar!r}")

        phi = np.asarray(phi_array, dtype=float)
        if not np.all(np.isfinite(phi)):
            raise ValueError("FREEFORM cross-section phi values must all be finite")

        z0 = float(self._profile_h.anchors[0, 0])
        z = z0 + t * self.length_mm
        r_h, r_v = self.evaluate_radii(np.asarray(z))
        a = float(r_h)
        b = float(r_v)

        span_index = len(self.stations) - 2
        for index in range(len(self.stations) - 1):
            if t <= float(self.stations[index + 1]["t"]):
                span_index = index
                break
        first = self.stations[span_index]
        second = self.stations[span_index + 1]
        t0 = float(first["t"])
        t1 = float(second["t"])
        local_u = min(1.0, max(0.0, (t - t0) / (t1 - t0)))

        rho0 = _station_radius(first, phi, a, b)
        if _station_descriptor(first) == _station_descriptor(second):
            return rho0
        rho1 = _station_radius(second, phi, a, b)
        weight = _smootherstep(local_u)
        return np.asarray((1.0 - weight) * rho0 + weight * rho1, dtype=float)

    def report(self) -> dict[str, Any]:
        """Return spline-vs-anchor deviation and endpoint geometry metadata."""
        deviations = {
            "H": _max_normal_deviation(self._profile_h),
            "V": _max_normal_deviation(self._profile_v),
        }
        return {
            "maxNormalDeviationMm": deviations,
            "throatRadiusMm": float(self._profile_h.anchors[0, 1]),
            "tangentAnglesDeg": {
                "H": {
                    "throat": self._profile_h.throat_angle_deg,
                    "mouth": self._profile_h.mouth_angle_deg,
                },
                "V": {
                    "throat": self._profile_v.throat_angle_deg,
                    "mouth": self._profile_v.mouth_angle_deg,
                },
            },
        }


_FREEFORM_GEOMETRY_CACHE: "OrderedDict[str, FreeformGeometry]" = OrderedDict()


def _freeform_key_normalise(value: Any) -> Any:
    """Recursively encode arrays and numeric sequences without precision loss."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return ["__ndarray__", list(array.shape), str(array.dtype), array.tobytes().hex()]
    if isinstance(value, (list, tuple)):
        try:
            array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        except (TypeError, ValueError):
            return ["__seq__", [_freeform_key_normalise(item) for item in value]]
        if array.ndim >= 1:
            return [
                "__ndarray__",
                list(array.shape),
                str(array.dtype),
                array.tobytes().hex(),
            ]
        return ["__seq__", [_freeform_key_normalise(item) for item in value]]
    if isinstance(value, Mapping):
        return {
            str(key): _freeform_key_normalise(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeform_cache_key(params: Mapping[str, Any]) -> str:
    relevant = {
        key: _freeform_key_normalise(params[key])
        for key in _FREEFORM_PARAM_KEYS
        if key in params
    }
    try:
        return json.dumps(relevant, sort_keys=True, default=repr)
    except TypeError:
        return repr(sorted(relevant.items(), key=lambda item: item[0]))


def _cache_store(key: str, geometry: FreeformGeometry) -> None:
    _FREEFORM_GEOMETRY_CACHE[key] = geometry
    _FREEFORM_GEOMETRY_CACHE.move_to_end(key)
    while len(_FREEFORM_GEOMETRY_CACHE) > _FREEFORM_CACHE_MAX:
        _FREEFORM_GEOMETRY_CACHE.popitem(last=False)


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"FREEFORM {field} must be a finite number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"FREEFORM {field} must be finite, got {value!r}")
    return result


def _parse_anchors(profile: Mapping[str, Any], plane: str) -> np.ndarray:
    points = profile.get("points")
    try:
        anchors = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"FREEFORM profile{plane}.points must be [[z_mm, r_mm], ...] numbers"
        ) from exc
    if anchors.ndim != 2 or anchors.shape[1:] != (2,):
        raise ValueError(
            f"FREEFORM profile{plane}.points must have shape (N, 2), got {anchors.shape}"
        )
    if not (2 <= anchors.shape[0] <= 64):
        raise ValueError(
            f"FREEFORM profile{plane}.points requires 2-64 anchors, got {anchors.shape[0]}"
        )
    if not np.all(np.isfinite(anchors)):
        raise ValueError(f"FREEFORM profile{plane}.points must all be finite")
    if np.any(anchors[:, 1] <= 0.0):
        raise ValueError(f"FREEFORM profile{plane} anchor radii must all be > 0 mm")
    if np.any(np.diff(anchors[:, 0]) <= 0.0):
        raise ValueError(
            f"FREEFORM profile{plane} anchor z values must be strictly increasing"
        )
    return anchors


def _parse_endpoint_values(
    profile: Mapping[str, Any],
    anchors: np.ndarray,
    plane: str,
    default_throat_angle: float,
) -> tuple[float, float, float, float]:
    throat_angle = _finite_float(
        profile.get("throatAngleDeg", default_throat_angle),
        f"profile{plane}.throatAngleDeg",
    )
    last_delta = anchors[-1] - anchors[-2]
    default_mouth_angle = math.degrees(math.atan2(last_delta[1], last_delta[0]))
    mouth_angle = _finite_float(
        profile.get("mouthAngleDeg", default_mouth_angle),
        f"profile{plane}.mouthAngleDeg",
    )
    for field, angle in (
        ("throatAngleDeg", throat_angle),
        ("mouthAngleDeg", mouth_angle),
    ):
        if not (-90.0 <= angle <= 90.0):
            raise ValueError(
                f"FREEFORM profile{plane}.{field} must be in [-90, 90] degrees, got {angle:g}"
            )

    throat_scale = _finite_float(
        profile.get("throatTangentScale", 1.0),
        f"profile{plane}.throatTangentScale",
    )
    mouth_scale = _finite_float(
        profile.get("mouthTangentScale", 1.0),
        f"profile{plane}.mouthTangentScale",
    )
    for field, scale in (
        ("throatTangentScale", throat_scale),
        ("mouthTangentScale", mouth_scale),
    ):
        if not (0.0 < scale <= 3.0):
            raise ValueError(
                f"FREEFORM profile{plane}.{field} must be in (0, 3], got {scale:g}"
            )
    return throat_angle, mouth_angle, throat_scale, mouth_scale


def _quadratic_roots(a: float, b: float, c: float) -> list[float]:
    scale = max(1.0, abs(a), abs(b), abs(c))
    eps = 1.0e-14 * scale
    if abs(a) <= eps:
        if abs(b) <= eps:
            return []
        return [-c / b]
    discriminant = b * b - 4.0 * a * c
    if discriminant < -eps * scale:
        return []
    root = math.sqrt(max(0.0, discriminant))
    return [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]


def _derivative_roots(coefficients: np.ndarray, interval: float) -> list[float]:
    roots = _quadratic_roots(
        3.0 * float(coefficients[0]),
        2.0 * float(coefficients[1]),
        float(coefficients[2]),
    )
    tolerance = 1.0e-12 * max(1.0, interval)
    return [min(interval, max(0.0, root)) for root in roots if -tolerance <= root <= interval + tolerance]


def _poly_value(coefficients: np.ndarray, local_u: float) -> float:
    return float(
        ((coefficients[0] * local_u + coefficients[1]) * local_u + coefficients[2])
        * local_u
        + coefficients[3]
    )


def _poly_derivative(coefficients: np.ndarray, local_u: float) -> float:
    return float(
        (3.0 * coefficients[0] * local_u + 2.0 * coefficients[1]) * local_u
        + coefficients[2]
    )


def _validate_plane_spline(plane: _PlaneSpline, overshoot_allowed: bool) -> None:
    coefficients = np.asarray(plane.spline.c, dtype=float)
    knots = np.asarray(plane.spline.x, dtype=float)
    n_segments = knots.size - 1

    for segment in range(n_segments):
        interval = float(knots[segment + 1] - knots[segment])
        z_coeff = coefficients[:, segment, 0]
        r_coeff = coefficients[:, segment, 1]
        z_scale = max(1.0, abs(float(z_coeff[2])), abs(float(z_coeff[1] * interval)))
        derivative_tol = 1.0e-11 * z_scale

        z_candidates = [0.0, interval]
        if abs(float(z_coeff[0])) > 1.0e-15:
            vertex = -float(z_coeff[1]) / (3.0 * float(z_coeff[0]))
            if 0.0 < vertex < interval:
                z_candidates.append(vertex)
        minimum_derivative = min(_poly_derivative(z_coeff, value) for value in z_candidates)
        if minimum_derivative < -derivative_tol:
            raise ValueError(
                f"FREEFORM profile{plane.name} segment {segment} folds backward: "
                "z'(u) must be non-negative"
            )

        for root in _derivative_roots(z_coeff, interval):
            at_curve_start = segment == 0 and root <= 1.0e-10
            at_curve_end = segment == n_segments - 1 and interval - root <= 1.0e-10
            if not (at_curve_start or at_curve_end):
                raise ValueError(
                    f"FREEFORM profile{plane.name} segment {segment} has z'(u)=0 "
                    "away from a curve endpoint"
                )

        dense_local = np.linspace(0.0, interval, 257)
        dense_r = np.asarray([_poly_value(r_coeff, value) for value in dense_local])
        radius_candidates = [0.0, interval, *_derivative_roots(r_coeff, interval)]
        candidate_r = np.asarray(
            [_poly_value(r_coeff, value) for value in radius_candidates], dtype=float
        )
        if not np.all(np.isfinite(dense_r)) or not np.all(np.isfinite(candidate_r)):
            raise ValueError(
                f"FREEFORM profile{plane.name} segment {segment} produces non-finite radius"
            )
        if float(min(np.min(dense_r), np.min(candidate_r))) <= 0.0:
            raise ValueError(
                f"FREEFORM profile{plane.name} segment {segment} produces a non-positive radius"
            )

        if not overshoot_allowed:
            r0 = float(plane.anchors[segment, 1])
            r1 = float(plane.anchors[segment + 1, 1])
            lower, upper = sorted((r0, r1))
            tolerance = 1.0e-10 * max(1.0, abs(lower), abs(upper))
            actual_min = float(min(np.min(dense_r), np.min(candidate_r)))
            actual_max = float(max(np.max(dense_r), np.max(candidate_r)))
            if actual_min < lower - tolerance or actual_max > upper + tolerance:
                raise ValueError(
                    f"FREEFORM profile{plane.name} segment {segment} radius overshoots "
                    f"its anchor range [{lower:g}, {upper:g}] mm; set "
                    "overshootPolicy='allow' to permit this intentionally"
                )


def _build_plane_spline(
    name: str,
    anchors: np.ndarray,
    throat_angle_deg: float,
    mouth_angle_deg: float,
    throat_scale: float,
    mouth_scale: float,
    overshoot_allowed: bool,
) -> _PlaneSpline:
    # Lazy by design: merely importing hornlab_mesher.freeform does not import scipy.
    from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

    chord_lengths = np.linalg.norm(np.diff(anchors, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(chord_lengths)))
    anchor_u = cumulative / cumulative[-1]

    z_pchip = PchipInterpolator(anchor_u, anchors[:, 0])
    r_pchip = PchipInterpolator(anchor_u, anchors[:, 1])
    derivatives = np.column_stack(
        (z_pchip.derivative()(anchor_u), r_pchip.derivative()(anchor_u))
    )

    for index, angle_deg, scale in (
        (0, throat_angle_deg, throat_scale),
        (-1, mouth_angle_deg, mouth_scale),
    ):
        automatic_speed = float(np.linalg.norm(derivatives[index]))
        angle = math.radians(angle_deg)
        derivatives[index] = automatic_speed * scale * np.asarray(
            [math.cos(angle), math.sin(angle)]
        )

    spline = CubicHermiteSpline(anchor_u, anchors, derivatives, axis=0)
    inverse_u = np.unique(np.concatenate((np.linspace(0.0, 1.0, _INVERSION_SAMPLE_N), anchor_u)))
    inverse_points = np.asarray(spline(inverse_u), dtype=float)
    plane = _PlaneSpline(
        name=name,
        anchors=anchors.copy(),
        anchor_u=anchor_u,
        spline=spline,
        inverse_z=inverse_points[:, 0],
        inverse_u=inverse_u,
        throat_angle_deg=throat_angle_deg,
        mouth_angle_deg=mouth_angle_deg,
    )
    _validate_plane_spline(plane, overshoot_allowed)
    return plane


def _normalise_stations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("FREEFORM crossSections must be a list of 2-32 stations")
    if not (2 <= len(value) <= 32):
        raise ValueError(f"FREEFORM crossSections requires 2-32 stations, got {len(value)}")

    stations: list[dict[str, Any]] = []
    for index, raw_station in enumerate(value):
        if not isinstance(raw_station, Mapping):
            raise ValueError(f"FREEFORM crossSections[{index}] must be a station mapping")
        t = _finite_float(raw_station.get("t"), f"crossSections[{index}].t")
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"FREEFORM crossSections[{index}].t must be in [0, 1], got {t:g}")
        shape = str(raw_station.get("shape", "")).strip().lower()
        if shape not in {"circle", "ellipse", "superellipse", "rounded_rectangle"}:
            raise ValueError(
                f"FREEFORM crossSections[{index}].shape must be circle, ellipse, "
                f"superellipse, or rounded_rectangle; got {shape!r}"
            )
        if shape == "circle" and index != 0:
            raise ValueError("FREEFORM shape 'circle' is allowed only at crossSections[0]")

        station: dict[str, Any] = {"t": t, "shape": shape}
        if shape == "superellipse":
            exponent = _finite_float(
                raw_station.get("exponent", 2.0),
                f"crossSections[{index}].exponent",
            )
            if not (2.0 <= exponent <= 16.0):
                raise ValueError(
                    f"FREEFORM crossSections[{index}].exponent must be in [2, 16], "
                    f"got {exponent:g}"
                )
            station["exponent"] = exponent
        elif shape == "rounded_rectangle":
            corner_ratio = _finite_float(
                raw_station.get("cornerRatio", 1.0),
                f"crossSections[{index}].cornerRatio",
            )
            if not (0.02 <= corner_ratio <= 1.0):
                raise ValueError(
                    f"FREEFORM crossSections[{index}].cornerRatio must be in [0.02, 1], "
                    f"got {corner_ratio:g}"
                )
            station["cornerRatio"] = corner_ratio
        stations.append(station)

    if stations[0]["t"] != 0.0:
        raise ValueError("FREEFORM crossSections first station must have t == 0")
    if stations[0]["shape"] not in {"circle", "ellipse"}:
        raise ValueError("FREEFORM crossSections first station shape must be circle or ellipse")
    if stations[-1]["t"] != 1.0:
        raise ValueError("FREEFORM crossSections last station must have t == 1")
    if any(
        float(stations[index + 1]["t"]) <= float(stations[index]["t"])
        for index in range(len(stations) - 1)
    ):
        raise ValueError("FREEFORM crossSections station t values must be strictly increasing")
    return stations


def _station_descriptor(station: Mapping[str, Any]) -> tuple[Any, ...]:
    shape = station["shape"]
    if shape in {"circle", "ellipse"}:
        return ("ellipse",)
    if shape == "superellipse":
        return ("superellipse", float(station["exponent"]))
    return ("rounded_rectangle", float(station["cornerRatio"]))


def _station_radius(
    station: Mapping[str, Any], phi: np.ndarray, a: float, b: float
) -> np.ndarray:
    shape = station["shape"]
    if shape in {"circle", "ellipse", "superellipse"}:
        exponent = float(station.get("exponent", 2.0))
        cos_phi = np.abs(np.cos(phi))
        sin_phi = np.abs(np.sin(phi))
        term = (cos_phi / a) ** exponent + (sin_phi / b) ** exponent
        return np.asarray(term ** (-1.0 / exponent), dtype=float)

    corner_radius = float(station["cornerRatio"]) * min(a, b)
    flat_phi = phi.reshape(-1)
    result = np.fromiter(
        (
            _rounded_rect_radius(
                float(angle),
                half_width=a,
                half_height=b,
                corner_radius=corner_radius,
            )
            for angle in flat_phi
        ),
        dtype=float,
        count=flat_phi.size,
    )
    return result.reshape(phi.shape)


def _smootherstep(value: Any) -> Any:
    u = np.asarray(value, dtype=float)
    result = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)
    return float(result) if result.ndim == 0 else result


def convexity_violations(
    geometry: FreeformGeometry, t_samples: Any, n_phi: int
) -> list[float]:
    """Return sampled ``t`` positions whose polar outline polygon is non-convex."""
    if n_phi < 8:
        raise ValueError(f"FREEFORM convexity check requires n_phi >= 8, got {n_phi}")
    samples = np.asarray(t_samples, dtype=float).reshape(-1)
    if not np.all(np.isfinite(samples)) or np.any(samples < 0.0) or np.any(samples > 1.0):
        raise ValueError("FREEFORM convexity t_samples must be finite values in [0, 1]")
    phi = np.linspace(0.0, 2.0 * math.pi, int(n_phi), endpoint=False)
    violations: list[float] = []
    for t in samples:
        radii = geometry.cross_section_radius(phi, float(t))
        points = np.column_stack((radii * np.cos(phi), radii * np.sin(phi)))
        incoming = points - np.roll(points, 1, axis=0)
        outgoing = np.roll(points, -1, axis=0) - points
        cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
        if np.any(cross < -1.0e-9):
            violations.append(float(t))
    return violations


def _convexity_ingest_samples(stations: list[dict[str, Any]]) -> np.ndarray:
    samples: list[float] = []
    for first, second in zip(stations[:-1], stations[1:]):
        samples.extend(np.linspace(float(first["t"]), float(second["t"]), 9).tolist())
    return np.unique(np.asarray(samples, dtype=float))


def _station_span_name(stations: list[dict[str, Any]], t: float) -> str:
    for first, second in zip(stations[:-1], stations[1:]):
        if float(first["t"]) - 1.0e-12 <= t <= float(second["t"]) + 1.0e-12:
            return f"{float(first['t']):g}..{float(second['t']):g}"
    return "unknown"


def _max_normal_deviation(plane: _PlaneSpline) -> float:
    maximum = 0.0
    for index in range(plane.anchor_u.size - 1):
        u = np.linspace(plane.anchor_u[index], plane.anchor_u[index + 1], 1001)
        points = np.asarray(plane.spline(u), dtype=float)
        start = plane.anchors[index]
        chord = plane.anchors[index + 1] - start
        chord_length = float(np.linalg.norm(chord))
        offsets = points - start
        normal_distance = np.abs(offsets[:, 0] * chord[1] - offsets[:, 1] * chord[0])
        maximum = max(maximum, float(np.max(normal_distance / chord_length)))
    return maximum


def build_freeform_geometry(params: Mapping[str, Any]) -> FreeformGeometry:
    """Parse, validate, construct, and memoize a FREEFORM geometry definition."""
    if not isinstance(params, Mapping):
        raise ValueError("FREEFORM params must be a mapping")
    key = _freeform_cache_key(params)
    cached = _FREEFORM_GEOMETRY_CACHE.get(key)
    if cached is not None:
        _FREEFORM_GEOMETRY_CACHE.move_to_end(key)
        return cached

    profile_h = params.get("profileH")
    profile_v = params.get("profileV")
    if not isinstance(profile_h, Mapping):
        raise ValueError("FREEFORM requires profileH as a mapping with a points list")
    if not isinstance(profile_v, Mapping):
        raise ValueError("FREEFORM requires profileV as a mapping with a points list")

    anchors_h = _parse_anchors(profile_h, "H")
    anchors_v = _parse_anchors(profile_v, "V")
    z_tolerance = 1.0e-9
    if not math.isclose(
        float(anchors_h[0, 0]),
        float(anchors_v[0, 0]),
        rel_tol=0.0,
        abs_tol=z_tolerance,
    ):
        raise ValueError("FREEFORM profileH and profileV must share the same first anchor z")
    if not math.isclose(
        float(anchors_h[-1, 0]),
        float(anchors_v[-1, 0]),
        rel_tol=0.0,
        abs_tol=z_tolerance,
    ):
        raise ValueError("FREEFORM profileH and profileV must share the same last anchor z")
    throat_difference = abs(float(anchors_h[0, 1] - anchors_v[0, 1]))
    if throat_difference > 1.0e-6:
        raise ValueError(
            "FREEFORM profileH and profileV throat radii must be equal within 1e-6 mm "
            f"(difference is {throat_difference:g} mm)"
        )

    default_throat_angle = _finite_float(params.get("a0", 15.5), "a0")
    endpoint_h = _parse_endpoint_values(profile_h, anchors_h, "H", default_throat_angle)
    endpoint_v = _parse_endpoint_values(profile_v, anchors_v, "V", default_throat_angle)
    if abs(endpoint_h[0] - endpoint_v[0]) > 5.0:
        warnings.warn(
            "FREEFORM H/V throat tangent angles differ by more than 5 degrees; "
            "the circular source-cap transition may be inconsistent",
            UserWarning,
            stacklevel=2,
        )

    policy = str(params.get("overshootPolicy", "reject")).strip().lower()
    if policy not in {"reject", "allow"}:
        raise ValueError(
            "FREEFORM overshootPolicy must be 'reject' or 'allow', "
            f"got {params.get('overshootPolicy')!r}"
        )
    stations = _normalise_stations(
        params.get(
            "crossSections",
            [{"t": 0.0, "shape": "circle"}, {"t": 1.0, "shape": "ellipse"}],
        )
    )

    plane_h = _build_plane_spline("H", anchors_h, *endpoint_h, policy == "allow")
    plane_v = _build_plane_spline("V", anchors_v, *endpoint_v, policy == "allow")
    geometry = FreeformGeometry(plane_h, plane_v, stations)

    violations = convexity_violations(
        geometry,
        _convexity_ingest_samples(stations),
        n_phi=64,
    )
    if violations:
        offending_t = violations[0]
        span = _station_span_name(stations, offending_t)
        raise ValueError(
            f"FREEFORM crossSections span {span} produces a non-convex outline "
            f"near t={offending_t:g}; adjust its shape, aspect, or cornerRatio"
        )

    _cache_store(key, geometry)
    return geometry


def _validate_freeform_config(profile_params: Mapping[str, Any]) -> FreeformGeometry:
    """Validate FREEFORM's pipeline-level exclusions and return its geometry.

    The spline/station kernel owns intrinsic geometry validation.  This shared
    entry-point adds the exclusions required by both config ingestion and
    callers that invoke :func:`build_point_grid` directly.
    """
    geometry = build_freeform_geometry(profile_params)

    sample_phi = np.linspace(0.0, math.tau, 33, endpoint=False)
    morph_targets = {
        int(round(eval_param(profile_params.get("morphTarget"), float(phi), 0.0)))
        for phi in sample_phi
    }
    if morph_targets & {1, 2}:
        raise ValueError(
            "FREEFORM does not support active morphTarget shaping; "
            "use crossSections stations instead"
        )

    gcurve_active = any(
        int(round(eval_param(profile_params.get("gcurveType"), float(phi), 0.0)))
        in {1, 2}
        and eval_param(profile_params.get("gcurveWidth"), float(phi), 0.0) > 0.0
        for phi in sample_phi
    )
    if gcurve_active:
        raise ValueError(
            "FREEFORM does not support active guiding curves; "
            "use crossSections stations instead"
        )

    profile_system = profile_params.get("profileSystem")
    cross_section: Mapping[str, Any] = {}
    if isinstance(profile_system, Mapping):
        candidate = profile_system.get("crossSection")
        if isinstance(candidate, Mapping):
            cross_section = candidate
    exponent = float(cross_section.get("exponent", 2.0))
    aspect_ratio = float(cross_section.get("aspectRatio", 1.0))
    if not math.isclose(exponent, 2.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            "FREEFORM requires cross-section exponent=2; "
            "use crossSections stations for outline shaping"
        )
    if not math.isclose(aspect_ratio, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            "FREEFORM requires cross-section aspectRatio=1; "
            "the H/V profiles define the aspect ratio"
        )

    for key in ("rot", "h", "throatExtLength", "throatExtAngle", "slotLength"):
        if abs(eval_param(profile_params.get(key), 0.0, 0.0)) > 1.0e-12:
            raise ValueError(f"FREEFORM does not support active {key}")

    raw_mode = str(profile_params.get("samplingMode") or "").strip().lower()
    mode = raw_mode.replace("_", "-")
    uniform_modes = {"", "uniform", "linear", "canonical", "default"}
    custom_modes = {"zmap", "z-map", "custom", "custom-zmap", "custom-z-map"}
    if _is_true(profile_params.get("athParitySampling")) or mode not in (
        uniform_modes | custom_modes
    ):
        raise ValueError(
            "FREEFORM samplingMode must be uniform or a custom zmap"
        )

    source_shape = int(
        round(eval_param(profile_params.get("sourceShape"), 0.0, 1.0))
    )
    source_radius = eval_param(profile_params.get("sourceRadius"), 0.0, -1.0)
    throat_radius = float(geometry.report()["throatRadiusMm"])
    if source_shape == 1 and source_radius > 0.0 and source_radius < throat_radius:
        raise ValueError(
            "FREEFORM rounded-cap sourceRadius must be at least the throat radius "
            f"({source_radius:g} mm requested, throat radius {throat_radius:g} mm)"
        )

    # TODO(M3): add surface-curvature and generated outer-offset regularity guards.
    # TODO(M3): report OCC-to-analytic H/V deviation in build metadata.
    return geometry


__all__ = ["FreeformGeometry", "build_freeform_geometry", "convexity_violations"]
