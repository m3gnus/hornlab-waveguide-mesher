"""Core geometry for FREEFORM H/V spline profiles.

The design contract is that every cross-section outline hits the horizontal and
vertical profile axes exactly, station joins are C1 or better (the shape blend is
C2), and both meridians share one axial ``z`` span.  The implementation follows
``/Users/magnus/Code/hornlab-workspace/Waveguide Generator/docs/plans/260801-freeform-hv-spline-profiles.md``
(especially sections 2.1, 2.2, 2.3, and 2.5).

Profile anchors accept ``[z, r]``, ``[z, r, angleDeg]``, or
``[z, r, angleDeg, strength]``.  A per-anchor tangent takes precedence over the
corresponding block-level endpoint angle and tangent scale.

Significant reverse-curvature spans are reported by default
(``inflectionPolicy='warn'``).  They may instead be rejected, while ``'allow'``
is retained as a non-blocking intent alias.

SciPy is deliberately imported only while constructing a profile.  Importing
this module therefore remains cheap and does not load SciPy.
"""

from __future__ import annotations

import json
import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
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
    "inflectionPolicy",
    "a0",
)


@dataclass(frozen=True)
class _InflectionSpan:
    z_start_mm: float
    z_end_mm: float
    tangent_drop_deg: float


@dataclass(frozen=True)
class _ActiveStationBlend:
    """The two shape stations and smootherstep weight active at one ``t``."""

    first_index: int
    second_index: int
    weight: float

    def station_weight(self, index: int) -> float:
        if index == self.first_index:
            return 1.0 - self.weight
        if index == self.second_index:
            return self.weight
        return 0.0


def _resolve_active_station_blend(
    stations: list[dict[str, Any]], t: float
) -> _ActiveStationBlend:
    """Resolve the station span used by outlines, validation, and sampling."""

    span_index = len(stations) - 2
    for index in range(len(stations) - 1):
        if t <= float(stations[index + 1]["t"]):
            span_index = index
            break
    t0 = float(stations[span_index]["t"])
    t1 = float(stations[span_index + 1]["t"])
    local_u = min(1.0, max(0.0, (float(t) - t0) / (t1 - t0)))
    return _ActiveStationBlend(
        first_index=span_index,
        second_index=span_index + 1,
        weight=float(_smootherstep(local_u)),
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
    anchor_angles_deg: np.ndarray
    anchor_strengths: np.ndarray

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
    _inflection_spans: dict[str, tuple[_InflectionSpan, ...]] = field(
        default_factory=dict, repr=False
    )
    _curvature_reports: dict[tuple[float, float], dict[str, Any]] = field(
        default_factory=dict, repr=False
    )

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

        blend = _resolve_active_station_blend(self.stations, t)
        first = self.stations[blend.first_index]
        second = self.stations[blend.second_index]

        rho0 = _station_radius(first, phi, a, b)
        if _station_descriptor(first) == _station_descriptor(second):
            return rho0
        rho1 = _station_radius(second, phi, a, b)
        return np.asarray(
            (1.0 - blend.weight) * rho0 + blend.weight * rho1, dtype=float
        )

    def report(self) -> dict[str, Any]:
        """Return spline deviation, tangents, inflections, and endpoint metadata."""
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
            "anchorTangents": {
                "H": _anchor_tangent_report(self._profile_h),
                "V": _anchor_tangent_report(self._profile_v),
            },
            "inflectionSpans": {
                plane: [
                    {
                        "zStartMm": span.z_start_mm,
                        "zEndMm": span.z_end_mm,
                        "tangentDropDeg": span.tangent_drop_deg,
                    }
                    for span in self._inflection_spans.get(plane, ())
                ]
                for plane in ("H", "V")
            },
        }

    def surface_curvature_report(
        self, wall_thickness_mm: float, *, margin: float = 0.4
    ) -> dict[str, Any]:
        """Finite-difference principal-curvature regularity of ``S(t, phi)``.

        The azimuth samples include uniform coverage plus dense samples around
        every rounded-rectangle tangency.  This is important for small corner
        ratios: their largest principal curvature generally does not occur at
        a cardinal or diagonal azimuth.
        """

        thickness = float(wall_thickness_mm)
        limit = float(margin)
        if not math.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("FREEFORM wall thickness must be finite and > 0 mm")
        if not math.isfinite(limit) or not (0.0 < limit < 1.0):
            raise ValueError("FREEFORM shell curvature margin must lie in (0, 1)")
        cache_key = (thickness, limit)
        cached = self._curvature_reports.get(cache_key)
        if cached is not None:
            return dict(cached)

        maximum = 0.0
        offending_t = 0.0
        offending_phi = 0.0
        offending_curvatures = (0.0, 0.0)
        # Include every station exactly. Smootherstep makes the joins C2, so a
        # centred stencil is valid there and catches a station-local maximum.
        t_samples = np.unique(
            np.concatenate(
                (
                    np.linspace(5.0e-4, 1.0 - 5.0e-4, 161),
                    np.asarray([station["t"] for station in self.stations], dtype=float),
                )
            )
        )
        dt = 2.5e-4
        dphi = 2.5e-4

        for t in t_samples:
            tc = min(1.0 - dt, max(dt, float(t)))
            phi = _curvature_phi_samples(self, tc)

            center = _surface_points(self, tc, phi)
            t_minus = _surface_points(self, tc - dt, phi)
            t_plus = _surface_points(self, tc + dt, phi)
            phi_minus = _surface_points(self, tc, phi - dphi)
            phi_plus = _surface_points(self, tc, phi + dphi)
            mixed_pp = _surface_points(self, tc + dt, phi + dphi)
            mixed_pm = _surface_points(self, tc + dt, phi - dphi)
            mixed_mp = _surface_points(self, tc - dt, phi + dphi)
            mixed_mm = _surface_points(self, tc - dt, phi - dphi)

            s_t = (t_plus - t_minus) / (2.0 * dt)
            s_phi = (phi_plus - phi_minus) / (2.0 * dphi)
            s_tt = (t_plus - 2.0 * center + t_minus) / (dt * dt)
            s_phiphi = (phi_plus - 2.0 * center + phi_minus) / (dphi * dphi)
            s_tphi = (mixed_pp - mixed_pm - mixed_mp + mixed_mm) / (
                4.0 * dt * dphi
            )

            cross = np.cross(s_t, s_phi)
            cross_length = np.linalg.norm(cross, axis=1)
            regular = cross_length > 1.0e-12
            if not np.all(regular):
                bad_phi = float(phi[np.flatnonzero(~regular)[0]])
                raise ValueError(
                    "FREEFORM surface curvature grid is singular near "
                    f"t={tc:.4f}, phi={math.degrees(bad_phi) % 360.0:.2f} deg"
                )
            normal = cross / cross_length[:, None]
            first_e = np.sum(s_t * s_t, axis=1)
            first_f = np.sum(s_t * s_phi, axis=1)
            first_g = np.sum(s_phi * s_phi, axis=1)
            second_e = np.sum(normal * s_tt, axis=1)
            second_f = np.sum(normal * s_tphi, axis=1)
            second_g = np.sum(normal * s_phiphi, axis=1)
            determinant = first_e * first_g - first_f * first_f
            if np.any(determinant <= 1.0e-18):
                bad_phi = float(phi[np.flatnonzero(determinant <= 1.0e-18)[0]])
                raise ValueError(
                    "FREEFORM surface first fundamental form is singular near "
                    f"t={tc:.4f}, phi={math.degrees(bad_phi) % 360.0:.2f} deg"
                )
            mean = (
                second_e * first_g
                - 2.0 * second_f * first_f
                + second_g * first_e
            ) / (2.0 * determinant)
            gaussian = (second_e * second_g - second_f * second_f) / determinant
            root = np.sqrt(np.maximum(0.0, mean * mean - gaussian))
            kappa_1 = mean + root
            kappa_2 = mean - root
            scaled = thickness * np.maximum(np.abs(kappa_1), np.abs(kappa_2))
            index = int(np.argmax(scaled))
            value = float(scaled[index])
            if value > maximum:
                maximum = value
                offending_t = tc
                offending_phi = float(phi[index] % math.tau)
                offending_curvatures = (float(kappa_1[index]), float(kappa_2[index]))

        report = {
            "ok": maximum < limit,
            "margin": limit,
            "maxThicknessTimesPrincipalCurvature": maximum,
            "offendingT": offending_t,
            "offendingPhiDeg": math.degrees(offending_phi),
            "principalCurvaturesPerMm": list(offending_curvatures),
        }
        self._curvature_reports[cache_key] = dict(report)
        return report


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


@dataclass(frozen=True)
class _ParsedAnchors:
    points: np.ndarray
    angles_deg: np.ndarray
    strengths: np.ndarray


def _parse_anchors(profile: Mapping[str, Any], plane: str) -> _ParsedAnchors:
    points = profile.get("points")
    if not isinstance(points, (list, tuple, np.ndarray)) or (
        isinstance(points, np.ndarray) and points.ndim == 0
    ):
        raise ValueError(
            f"FREEFORM profile{plane}.points must be a list of 2-4 element rows"
        )
    if not (2 <= len(points) <= 64):
        raise ValueError(
            f"FREEFORM profile{plane}.points requires 2-64 anchors, got {len(points)}"
        )

    anchors = np.empty((len(points), 2), dtype=float)
    angles_deg = np.full(len(points), np.nan, dtype=float)
    strengths = np.full(len(points), np.nan, dtype=float)
    for index, row in enumerate(points):
        row_is_sequence = isinstance(row, (list, tuple, np.ndarray)) and not (
            isinstance(row, np.ndarray) and row.ndim == 0
        )
        row_length = len(row) if row_is_sequence else None
        if row_length not in {2, 3, 4}:
            raise ValueError(
                f"FREEFORM profile{plane}.points[{index}] must have 2, 3, or 4 "
                f"elements, got {row_length if row_length is not None else type(row).__name__}"
            )
        anchors[index, 0] = _finite_float(
            row[0], f"profile{plane}.points[{index}].z"
        )
        anchors[index, 1] = _finite_float(
            row[1], f"profile{plane}.points[{index}].r"
        )
        if len(row) == 4 and row[2] is None:
            raise ValueError(
                f"FREEFORM profile{plane}.points[{index}] strength requires "
                "angleDeg in the same row"
            )
        if len(row) >= 3:
            angle = _finite_float(
                row[2], f"profile{plane}.points[{index}].angleDeg"
            )
            is_endpoint = index in {0, len(points) - 1}
            lower_bracket, upper_bracket = ("[", "]") if is_endpoint else ("(", ")")
            angle_allowed = (
                -90.0 <= angle <= 90.0 if is_endpoint else -90.0 < angle < 90.0
            )
            if not angle_allowed:
                anchor_kind = "endpoint" if is_endpoint else "interior anchor"
                raise ValueError(
                    f"FREEFORM profile{plane}.points[{index}].angleDeg must be in "
                    f"{lower_bracket}-90, 90{upper_bracket} degrees for an "
                    f"{anchor_kind}, got {angle:g}"
                )
            angles_deg[index] = angle
        if len(row) == 4:
            strength = _finite_float(
                row[3], f"profile{plane}.points[{index}].strength"
            )
            if not (0.0 < strength <= 3.0):
                raise ValueError(
                    f"FREEFORM profile{plane}.points[{index}].strength must be in "
                    f"(0, 3], got {strength:g}"
                )
            strengths[index] = strength

    if np.any(anchors[:, 1] <= 0.0):
        raise ValueError(f"FREEFORM profile{plane} anchor radii must all be > 0 mm")
    if np.any(np.diff(anchors[:, 0]) <= 0.0):
        raise ValueError(
            f"FREEFORM profile{plane} anchor z values must be strictly increasing"
        )
    return _ParsedAnchors(anchors, angles_deg, strengths)


def _parse_endpoint_values(
    profile: Mapping[str, Any],
    parsed: _ParsedAnchors,
    plane: str,
    default_throat_angle: float,
) -> tuple[float, float, float, float]:
    anchors = parsed.points
    if np.isfinite(parsed.angles_deg[0]):
        throat_angle = float(parsed.angles_deg[0])
        throat_scale = (
            float(parsed.strengths[0]) if np.isfinite(parsed.strengths[0]) else 1.0
        )
    else:
        throat_angle = _finite_float(
            profile.get("throatAngleDeg", default_throat_angle),
            f"profile{plane}.throatAngleDeg",
        )
        throat_scale = _finite_float(
            profile.get("throatTangentScale", 1.0),
            f"profile{plane}.throatTangentScale",
        )

    last_delta = anchors[-1] - anchors[-2]
    default_mouth_angle = math.degrees(math.atan2(last_delta[1], last_delta[0]))
    if np.isfinite(parsed.angles_deg[-1]):
        mouth_angle = float(parsed.angles_deg[-1])
        mouth_scale = (
            float(parsed.strengths[-1]) if np.isfinite(parsed.strengths[-1]) else 1.0
        )
    else:
        mouth_angle = _finite_float(
            profile.get("mouthAngleDeg", default_mouth_angle),
            f"profile{plane}.mouthAngleDeg",
        )
        mouth_scale = _finite_float(
            profile.get("mouthTangentScale", 1.0),
            f"profile{plane}.mouthTangentScale",
        )

    for field, angle in (
        ("throatAngleDeg", throat_angle),
        ("mouthAngleDeg", mouth_angle),
    ):
        if not (-90.0 <= angle <= 90.0):
            raise ValueError(
                f"FREEFORM profile{plane}.{field} must be in [-90, 90] degrees, got {angle:g}"
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
    parsed: _ParsedAnchors,
    throat_angle_deg: float,
    mouth_angle_deg: float,
    throat_scale: float,
    mouth_scale: float,
    overshoot_allowed: bool,
) -> _PlaneSpline:
    # Lazy by design: merely importing hornlab_mesher.freeform does not import scipy.
    from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

    anchors = parsed.points
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

    for index in np.flatnonzero(np.isfinite(parsed.angles_deg)):
        automatic_speed = float(
            np.linalg.norm(
                [
                    z_pchip.derivative()(anchor_u[index]),
                    r_pchip.derivative()(anchor_u[index]),
                ]
            )
        )
        strength = (
            float(parsed.strengths[index])
            if np.isfinite(parsed.strengths[index])
            else 1.0
        )
        angle = math.radians(float(parsed.angles_deg[index]))
        derivatives[index] = automatic_speed * strength * np.asarray(
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
        anchor_angles_deg=parsed.angles_deg.copy(),
        anchor_strengths=parsed.strengths.copy(),
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
            has_corner_ratio = "cornerRatio" in raw_station
            has_corner_radius = "cornerRadiusMm" in raw_station
            if has_corner_ratio == has_corner_radius:
                raise ValueError(
                    f"FREEFORM crossSections[{index}] rounded_rectangle station must "
                    "specify exactly one of cornerRatio or cornerRadiusMm"
                )
            if has_corner_ratio:
                corner_ratio = _finite_float(
                    raw_station["cornerRatio"],
                    f"crossSections[{index}].cornerRatio",
                )
                if not (0.02 <= corner_ratio <= 1.0):
                    raise ValueError(
                        f"FREEFORM crossSections[{index}].cornerRatio must be in [0.02, 1], "
                        f"got {corner_ratio:g}"
                    )
                station["cornerRatio"] = corner_ratio
            else:
                corner_radius = _finite_float(
                    raw_station["cornerRadiusMm"],
                    f"crossSections[{index}].cornerRadiusMm",
                )
                if corner_radius <= 0.0:
                    raise ValueError(
                        f"FREEFORM crossSections[{index}].cornerRadiusMm must be > 0 mm, "
                        f"got {corner_radius:g}"
                    )
                station["cornerRadiusMm"] = corner_radius
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
    if "cornerRadiusMm" in station:
        return ("rounded_rectangle", "cornerRadiusMm", float(station["cornerRadiusMm"]))
    return ("rounded_rectangle", "cornerRatio", float(station["cornerRatio"]))


def station_corner_radius_mm(
    station: Mapping[str, Any], a: float, b: float
) -> float:
    """Return a rounded-rectangle station's effective corner radius in mm."""
    limit = min(float(a), float(b))
    if "cornerRadiusMm" in station:
        corner = float(station["cornerRadiusMm"])
    else:
        corner = float(station["cornerRatio"]) * limit
    return min(max(corner, 0.0), limit)


def active_rounded_rect_corner_radius_mm(
    stations: list[dict[str, Any]], t: float, a: float, b: float
) -> float:
    """Corner radius for the structural rounded-rectangle family at ``t``.

    Active rounded-rectangle descriptors are blended using their normalized
    station weights.  If the active outline is purely smooth, the nearest
    rounded-rectangle station supplies the descriptor while the local
    semi-axes still supply its scale.
    """

    blend = _resolve_active_station_blend(stations, float(t))
    weighted_corners: list[tuple[float, float]] = []
    for index in (blend.first_index, blend.second_index):
        station = stations[index]
        weight = blend.station_weight(index)
        if station["shape"] == "rounded_rectangle" and weight > 0.0:
            weighted_corners.append(
                (weight, station_corner_radius_mm(station, a, b))
            )
    if weighted_corners:
        total_weight = sum(weight for weight, _corner in weighted_corners)
        return sum(
            weight * corner for weight, corner in weighted_corners
        ) / total_weight

    nearest = min(
        (
            (abs(float(station["t"]) - float(t)), index, station)
            for index, station in enumerate(stations)
            if station["shape"] == "rounded_rectangle"
        ),
        key=lambda item: (item[0], item[1]),
    )[2]
    return station_corner_radius_mm(nearest, a, b)


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

    corner_radius = station_corner_radius_mm(station, a, b)
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


def _surface_points(
    geometry: FreeformGeometry, t: float, phi: np.ndarray
) -> np.ndarray:
    radii = geometry.cross_section_radius(phi, float(t))
    z0 = float(geometry._profile_h.anchors[0, 0])
    z = z0 + float(t) * geometry.length_mm
    return np.column_stack(
        (radii * np.cos(phi), radii * np.sin(phi), np.full(phi.shape, z))
    )


def _curvature_phi_samples(
    geometry: FreeformGeometry, t: float
) -> np.ndarray:
    """All-azimuth curvature probes, enriched at rounded-corner features."""

    z0 = float(geometry._profile_h.anchors[0, 0])
    z = z0 + float(t) * geometry.length_mm
    radii_h, radii_v = geometry.evaluate_radii(np.asarray(z))
    a = float(radii_h)
    b = float(radii_v)
    samples = [np.linspace(0.0, math.tau, 721, endpoint=False)]
    for station in geometry.stations:
        if station["shape"] != "rounded_rectangle":
            continue
        corner = station_corner_radius_mm(station, a, b)
        cx = a - corner
        cy = b - corner
        theta1 = math.atan2(cy, a)
        theta2 = math.atan2(b, cx)
        q1 = np.linspace(theta1, theta2, 41)
        samples.extend(
            (
                q1,
                math.pi - q1,
                math.pi + q1,
                math.tau - q1,
            )
        )
    return np.unique(np.mod(np.concatenate(samples), math.tau))


def _polyline_self_intersects_2d(points: np.ndarray, *, closed: bool) -> bool:
    """Return whether non-neighbouring segments of a 2-D polyline intersect."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 4:
        return False
    segment_count = pts.shape[0] if closed else pts.shape[0] - 1

    def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def intersects(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        scale = max(1.0, *(abs(float(value)) for value in (a - b)), *(abs(float(value)) for value in (c - d)))
        tolerance = 1.0e-10 * scale * scale
        o1 = orientation(a, b, c)
        o2 = orientation(a, b, d)
        o3 = orientation(c, d, a)
        o4 = orientation(c, d, b)
        return o1 * o2 < -tolerance and o3 * o4 < -tolerance

    for first in range(segment_count):
        a = pts[first]
        b = pts[(first + 1) % pts.shape[0]]
        for second in range(first + 1, segment_count):
            if second == first + 1:
                continue
            if closed and first == 0 and second == segment_count - 1:
                continue
            c = pts[second]
            d = pts[(second + 1) % pts.shape[0]]
            if intersects(a, b, c, d):
                return True
    return False


def validate_outer_offset_grid(
    inner_points: Any, outer_points: Any, *, full_circle: bool
) -> None:
    """Reject folds or intersections in FREEFORM's generated offset shell."""

    inner = np.asarray(inner_points, dtype=float)
    outer = np.asarray(outer_points, dtype=float)
    if inner.shape != outer.shape or inner.ndim != 3 or inner.shape[2] != 3:
        raise ValueError("FREEFORM outer offset grid does not match the inner grid")
    if not np.all(np.isfinite(outer)):
        raise ValueError("FREEFORM outer offset grid contains non-finite coordinates")

    phi_count = inner.shape[0] if full_circle else inner.shape[0] - 1
    for i in range(phi_count):
        ni = (i + 1) % inner.shape[0]
        inner_phi = inner[ni, :-1] - inner[i, :-1]
        inner_axial = inner[i, 1:] - inner[i, :-1]
        outer_phi = outer[ni, :-1] - outer[i, :-1]
        outer_axial = outer[i, 1:] - outer[i, :-1]
        inner_normal = np.cross(inner_phi, inner_axial)
        outer_normal = np.cross(outer_phi, outer_axial)
        inner_len = np.linalg.norm(inner_normal, axis=1)
        outer_len = np.linalg.norm(outer_normal, axis=1)
        degenerate = (inner_len <= 1.0e-12) | (outer_len <= 1.0e-12)
        alignment = np.sum(inner_normal * outer_normal, axis=1)
        flipped = degenerate | (alignment <= 0.0)
        if np.any(flipped):
            ring = int(np.flatnonzero(flipped)[0])
            raise ValueError(
                "FREEFORM generated outer offset grid has a normal flip "
                f"near azimuth row {i}, axial interval {ring}"
            )

    # Shared-z FREEFORM cannot hide a non-local axial overlap in Cartesian z;
    # checking every generated meridian in (z, radius) catches offset rollback.
    for i in range(outer.shape[0]):
        meridian = np.column_stack(
            (outer[i, :, 2], np.linalg.norm(outer[i, :, :2], axis=1))
        )
        if _polyline_self_intersects_2d(meridian, closed=False):
            raise ValueError(
                "FREEFORM generated outer offset grid self-intersects "
                f"near azimuth row {i}"
            )

    # Each input ring is convex. Its regular outward offset must retain one
    # consistent turn direction; a sign reversal names the earliest bad ring.
    for j in range(outer.shape[1]):
        xy = outer[:, j, :2]
        if _polyline_self_intersects_2d(xy, closed=full_circle):
            raise ValueError(
                "FREEFORM generated outer offset grid self-intersects "
                f"on axial ring {j}"
            )


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


def _validate_station_corner_radii(geometry: FreeformGeometry) -> None:
    z0 = float(geometry._profile_h.anchors[0, 0])
    for index, station in enumerate(geometry.stations):
        if station["shape"] != "rounded_rectangle" or "cornerRadiusMm" not in station:
            continue
        t = float(station["t"])
        z = z0 + t * geometry.length_mm
        a_value, b_value = geometry.evaluate_radii(np.asarray(z))
        limit = min(float(a_value), float(b_value))
        lower = 0.02 * limit
        radius = float(station["cornerRadiusMm"])
        if not (lower <= radius <= limit):
            raise ValueError(
                f"FREEFORM crossSections[{index}].cornerRadiusMm must be in "
                f"[{lower:g}, {limit:g}] mm at station t={t:g}, got {radius:g} mm"
            )

        active_t_parts: list[np.ndarray] = []
        for first, second in zip(geometry.stations[:-1], geometry.stations[1:]):
            midpoint = 0.5 * (float(first["t"]) + float(second["t"]))
            blend = _resolve_active_station_blend(geometry.stations, midpoint)
            if blend.station_weight(index) > 0.0:
                active_t_parts.append(
                    np.linspace(
                        float(first["t"]), float(second["t"]), 1001, dtype=float
                    )
                )
        if not active_t_parts:
            continue
        active_t = np.unique(np.concatenate(active_t_parts))
        active_z = z0 + active_t * geometry.length_mm
        active_a, active_b = geometry.evaluate_radii(active_z)
        allowed = np.minimum(active_a, active_b)
        max_allowed = float(np.min(allowed))
        if radius <= max_allowed:
            continue
        offending = allowed < radius
        offending_z = active_z[offending]
        raise ValueError(
            f"FREEFORM crossSections[{index}].cornerRadiusMm={radius:g} mm exceeds "
            f"the maximum allowed value {max_allowed:g} mm over its active z range "
            f"[{float(active_z[0]):g}, {float(active_z[-1]):g}] mm; sampled offending "
            f"z range [{float(offending_z[0]):g}, {float(offending_z[-1]):g}] mm"
        )


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


def _anchor_tangent_report(plane: _PlaneSpline) -> list[dict[str, float | None]]:
    return [
        {
            "z": float(anchor[0]),
            "r": float(anchor[1]),
            "angleDeg": float(angle) if math.isfinite(float(angle)) else None,
            "strength": float(strength) if math.isfinite(float(strength)) else None,
        }
        for anchor, angle, strength in zip(
            plane.anchors, plane.anchor_angles_deg, plane.anchor_strengths
        )
    ]


def _significant_inflection_spans(
    plane: _PlaneSpline,
) -> tuple[_InflectionSpan, ...]:
    """Return sampled negative-curvature spans with a visible tangent dip."""

    u = plane.inverse_u
    first = np.asarray(plane.spline.derivative(1)(u), dtype=float)
    second = np.asarray(plane.spline.derivative(2)(u), dtype=float)
    signed_curvature_numerator = (
        first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    )
    negative = signed_curvature_numerator < 0.0
    changes = np.diff(np.concatenate(([False], negative, [False])).astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    tangent_angles_deg = np.degrees(np.arctan2(first[:, 1], first[:, 0]))

    spans: list[_InflectionSpan] = []
    for start, end in zip(starts, ends):
        tangent_drop_deg = float(
            tangent_angles_deg[start] - tangent_angles_deg[end]
        )
        if tangent_drop_deg <= 1.0:
            continue
        endpoints = np.asarray(plane.spline(u[[start, end]]), dtype=float)
        spans.append(
            _InflectionSpan(
                z_start_mm=float(endpoints[0, 0]),
                z_end_mm=float(endpoints[1, 0]),
                tangent_drop_deg=tangent_drop_deg,
            )
        )
    return tuple(spans)


def _reject_inflections(
    plane: _PlaneSpline, spans: tuple[_InflectionSpan, ...]
) -> None:
    if not spans:
        return
    span = spans[0]
    raise ValueError(
        f"FREEFORM profile{plane.name} inflection spans "
        f"z={span.z_start_mm:.3f}..{span.z_end_mm:.3f} mm with a "
        f"{span.tangent_drop_deg:.2f} deg tangent-angle drop; adjust a tangent "
        "handle or add an extra point, or change inflectionPolicy to 'warn' "
        "or 'allow'"
    )


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

    parsed_h = _parse_anchors(profile_h, "H")
    parsed_v = _parse_anchors(profile_v, "V")
    anchors_h = parsed_h.points
    anchors_v = parsed_v.points
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
    endpoint_h = _parse_endpoint_values(profile_h, parsed_h, "H", default_throat_angle)
    endpoint_v = _parse_endpoint_values(profile_v, parsed_v, "V", default_throat_angle)
    if abs(endpoint_h[0] - endpoint_v[0]) > 5.0:
        warnings.warn(
            "FREEFORM H/V throat tangent angles differ by more than 5 degrees; "
            "the circular source-cap transition may be inconsistent",
            UserWarning,
            stacklevel=2,
        )

    overshoot_policy = (
        str(params.get("overshootPolicy", "reject")).strip().lower()
    )
    if overshoot_policy not in {"reject", "allow"}:
        raise ValueError(
            "FREEFORM overshootPolicy must be 'reject' or 'allow', "
            f"got {params.get('overshootPolicy')!r}"
        )
    inflection_policy = str(params.get("inflectionPolicy", "warn")).strip().lower()
    if inflection_policy not in {"warn", "reject", "allow"}:
        raise ValueError(
            "FREEFORM inflectionPolicy must be 'warn', 'reject', or 'allow', "
            f"got {params.get('inflectionPolicy')!r}"
        )
    stations = _normalise_stations(
        params.get(
            "crossSections",
            [{"t": 0.0, "shape": "circle"}, {"t": 1.0, "shape": "ellipse"}],
        )
    )

    plane_h = _build_plane_spline(
        "H", parsed_h, *endpoint_h, overshoot_policy == "allow"
    )
    plane_v = _build_plane_spline(
        "V", parsed_v, *endpoint_v, overshoot_policy == "allow"
    )
    inflection_spans = {
        "H": _significant_inflection_spans(plane_h),
        "V": _significant_inflection_spans(plane_v),
    }
    if inflection_policy == "reject":
        _reject_inflections(plane_h, inflection_spans["H"])
        _reject_inflections(plane_v, inflection_spans["V"])
    geometry = FreeformGeometry(
        plane_h,
        plane_v,
        stations,
        _inflection_spans=inflection_spans,
    )
    _validate_station_corner_radii(geometry)

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
            f"near t={offending_t:g}; adjust its shape, aspect, or corner setting"
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

    wall_thickness = float(profile_params.get("wallThickness") or 0.0)
    enc_depth = float(profile_params.get("encDepth") or 0.0)
    if wall_thickness > 0.0 and enc_depth <= 0.0:
        geometry_scale = float(profile_params.get("scale") or 1.0)
        report = geometry.surface_curvature_report(
            wall_thickness / geometry_scale,
            margin=0.4,
        )
        if not bool(report["ok"]):
            kappa_1, kappa_2 = report["principalCurvaturesPerMm"]
            raise ValueError(
                "FREEFORM wall offset fails the surface-curvature guard near "
                f"t={float(report['offendingT']):.4f}, "
                f"phi={float(report['offendingPhiDeg']):.2f} deg: "
                "|wallThickness*kappa_i|="
                f"{float(report['maxThicknessTimesPrincipalCurvature']):.3f} "
                f">= margin=0.400 (kappa=[{kappa_1:.6g}, {kappa_2:.6g}] 1/mm)"
            )
    return geometry


__all__ = [
    "FreeformGeometry",
    "build_freeform_geometry",
    "convexity_violations",
    "validate_outer_offset_grid",
]
