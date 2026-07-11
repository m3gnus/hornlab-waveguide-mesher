from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .profile_common import _osse_radius, _parse_number_list, eval_param

def _guiding_curve_type(params: Mapping[str, Any], p: float) -> int:
    return int(round(eval_param(params.get("gcurveType"), p, 0.0)))


def _guiding_curve_active(params: Mapping[str, Any], p: float) -> bool:
    return _guiding_curve_type(params, p) in {1, 2} and eval_param(params.get("gcurveWidth"), p, 0.0) > 0.0


def _guiding_curve_target_radius(p: float, params: Mapping[str, Any]) -> float:
    curve_type = _guiding_curve_type(params, p)
    width = eval_param(params.get("gcurveWidth"), p, 0.0)
    if curve_type == 0 or width <= 0.0:
        return 0.0
    aspect = eval_param(params.get("gcurveAspectRatio"), p, 1.0)
    if aspect <= 0.0:
        raise ValueError("gcurveAspectRatio must be positive")

    rotation = math.radians(eval_param(params.get("gcurveRot"), p, 0.0))
    pr = p - rotation
    cos_p = math.cos(pr)
    sin_p = math.sin(pr)

    if curve_type == 1:
        exponent = max(2.0, eval_param(params.get("gcurveSeN"), p, 3.0))
        a = width / 2.0
        b = a * aspect
        term = abs(cos_p / a) ** exponent + abs(sin_p / b) ** exponent
        return term ** (-1.0 / exponent)

    if curve_type != 2:
        raise ValueError(f"unsupported GCurve type {curve_type}")

    sf = _parse_number_list(params.get("gcurveSf", params.get("gcurveSF")))
    if len(sf) >= 6:
        sf_a, sf_b, sf_m1, sf_n1, sf_n2, sf_n3 = sf[:6]
        sf_m2 = sf_m1
    else:
        sf_a = eval_param(params.get("gcurveSfA"), p, 1.0)
        sf_b = eval_param(params.get("gcurveSfB"), p, 1.0)
        sf_m1 = eval_param(params.get("gcurveSfM1"), p, 4.0)
        raw_m2 = params.get("gcurveSfM2")
        sf_m2 = eval_param(raw_m2, p, sf_m1) if raw_m2 is not None else sf_m1
        sf_n1 = eval_param(params.get("gcurveSfN1"), p, 2.0)
        sf_n2 = eval_param(params.get("gcurveSfN2"), p, 2.0)
        sf_n3 = eval_param(params.get("gcurveSfN3"), p, 2.0)
    sf_a = max(abs(sf_a), 1.0e-12)
    sf_b = max(abs(sf_b), 1.0e-12)
    sf_n1 = max(abs(sf_n1), 1.0e-12)
    t1 = abs(math.cos((sf_m1 * pr) / 4.0) / sf_a) ** sf_n2
    t2 = abs(math.sin((sf_m2 * pr) / 4.0) / sf_b) ** sf_n3
    r_norm = (t1 + t2) ** (-1.0 / sf_n1)
    sx = width / 2.0
    sy = sx * aspect
    return math.hypot(r_norm * cos_p * sx, r_norm * sin_p * sy)


def _invert_osse_coverage_angle(
    target_radius: float,
    z_main: float,
    p: float,
    params: Mapping[str, Any],
    *,
    a0_deg: float,
    r0_main: float,
) -> float:
    low = 0.5
    high = 89.0
    for _ in range(24):
        mid = 0.5 * (low + high)
        radius = _osse_radius(z_main, p, params, r0=r0_main, a_deg=mid, a0_deg=a0_deg)
        if radius < target_radius:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _coverage_angle_from_guiding_curve(
    p: float,
    params: Mapping[str, Any],
    *,
    main_length: float,
    a0_deg: float,
    r0_main: float,
) -> float | None:
    if not _guiding_curve_active(params, p):
        return None
    target_radius = _guiding_curve_target_radius(p, params)
    if target_radius <= 0.0:
        return None
    dist = eval_param(params.get("gcurveDist"), p, 1.0)
    target_z = main_length * dist if 0.0 < dist <= 1.0 else dist
    if target_z <= 0.0 or not math.isfinite(target_z):
        target_z = main_length
    target_z = min(main_length, target_z)
    return _invert_osse_coverage_angle(
        target_radius,
        target_z,
        p,
        params,
        a0_deg=a0_deg,
        r0_main=r0_main,
    )



def _morph_target_shape(params: Mapping[str, Any], p: float) -> int:
    return int(round(eval_param(params.get("morphTarget"), p, 0.0)))


def _morph_active(params: Mapping[str, Any], p: float) -> bool:
    return _morph_target_shape(params, p) in {1, 2}


def _rounded_rect_radius(phi: float, half_width: float, half_height: float, corner_radius: float) -> float:
    abs_cos = abs(math.cos(phi))
    abs_sin = abs(math.sin(phi))
    if abs_cos < 1.0e-9:
        return half_height
    if abs_sin < 1.0e-9:
        return half_width

    r = min(max(corner_radius, 0.0), half_width, half_height)
    if r <= 1.0e-9:
        return min(half_width / abs_cos, half_height / abs_sin)

    y_at_x = (half_width * abs_sin) / abs_cos
    if y_at_x <= half_height - r + 1.0e-9:
        return half_width / abs_cos
    x_at_y = (half_height * abs_cos) / abs_sin
    if x_at_y <= half_width - r + 1.0e-9:
        return half_height / abs_sin

    cx = half_width - r
    cy = half_height - r
    b = -2.0 * (abs_cos * cx + abs_sin * cy)
    c = cx * cx + cy * cy - r * r
    disc = max(0.0, b * b - 4.0 * c)
    return (-b + math.sqrt(disc)) / 2.0


def _configured_morph_half_dimension(
    value: Any,
    phi: float,
    *,
    fallback_radius: float,
    implicit_half_dimension: float | None = None,
) -> float:
    # A resolved half-dimension from the grid builder wins over the raw config
    # value: it already folds in implicit-extent derivation and the
    # no-shrinkage dimension floor.
    if implicit_half_dimension is not None and implicit_half_dimension > 0.0:
        return float(implicit_half_dimension)
    dimension = eval_param(value, phi, 0.0)
    if dimension <= 0.0:
        return max(0.0, float(fallback_radius))
    return dimension / 2.0


def _circle_morph_target_radius(
    current_radius: float,
    phi: float,
    params: Mapping[str, Any],
    *,
    implicit_half_width: float | None = None,
    implicit_half_height: float | None = None,
) -> float:
    half_width = _configured_morph_half_dimension(
        params.get("morphWidth"),
        phi,
        fallback_radius=current_radius,
        implicit_half_dimension=implicit_half_width,
    )
    half_height = _configured_morph_half_dimension(
        params.get("morphHeight"),
        phi,
        fallback_radius=current_radius,
        implicit_half_dimension=implicit_half_height,
    )
    return max(half_width, half_height)


def _morph_target_radius_at_angle(
    current_radius: float,
    phi: float,
    params: Mapping[str, Any],
    *,
    implicit_half_width: float | None = None,
    implicit_half_height: float | None = None,
) -> float:
    target = _morph_target_shape(params, phi)
    if target == 0:
        return current_radius
    if target == 2:
        return _circle_morph_target_radius(
            current_radius,
            phi,
            params,
            implicit_half_width=implicit_half_width,
            implicit_half_height=implicit_half_height,
        )
    if target != 1:
        raise ValueError(f"unsupported Morph target {target}")
    half_width = _configured_morph_half_dimension(
        params.get("morphWidth"),
        phi,
        fallback_radius=current_radius,
        implicit_half_dimension=implicit_half_width,
    )
    half_height = _configured_morph_half_dimension(
        params.get("morphHeight"),
        phi,
        fallback_radius=current_radius,
        implicit_half_dimension=implicit_half_height,
    )
    corner = eval_param(params.get("morphCorner"), phi, 0.0)
    return _rounded_rect_radius(phi, half_width, half_height, corner)


def _morph_factor(
    t: float,
    phi: float,
    params: Mapping[str, Any],
    *,
    morph_start: float | None = None,
) -> float:
    if not _morph_active(params, phi):
        return 0.0
    if morph_start is None:
        morph_start = eval_param(params.get("morphFixed"), phi, 0.0)
    if t <= morph_start:
        return 0.0
    rate = eval_param(params.get("morphRate"), phi, 3.0)
    denom = max(1.0e-9, 1.0 - morph_start)
    return min(1.0, max(0.0, (t - morph_start) / denom)) ** rate


def _apply_morphing(
    current_radius: float,
    mouth_radius: float,
    t: float,
    phi: float,
    params: Mapping[str, Any],
    *,
    morph_start: float | None = None,
    implicit_half_width: float | None = None,
    implicit_half_height: float | None = None,
) -> float:
    factor = _morph_factor(t, phi, params, morph_start=morph_start)
    if factor <= 0.0:
        return current_radius
    # OS-SE morphing is a directional target-mouth rule:
    # rm(z, phi) = r(z, phi) + f(z) * (rM(phi) - r(L, phi)).
    # No-shrinkage gating happens at the dimension level when the grid builder
    # resolves the target half-dimensions, not per azimuth: ATH keeps the mouth
    # an exact target curve and enlarges the target dimensions instead.
    target_radius = _morph_target_radius_at_angle(
        mouth_radius,
        phi,
        params,
        implicit_half_width=implicit_half_width,
        implicit_half_height=implicit_half_height,
    )
    return current_radius + (target_radius - mouth_radius) * factor


def _rounded_rect_quadrant_angles(
    points_per_quadrant: int,
    half_width: float,
    half_height: float,
    corner_radius: float,
    corner_segments: int,
) -> np.ndarray:
    """First-quadrant azimuth samples for a rounded-rectangle morph target.

    ATH always samples the corner arc with four profiles per quadrant (both
    wall-tangency endpoints plus two interior points at 30/60 degrees of arc
    parameter). ``Mesh.CornerSegments`` selects this rounded-corner placement
    policy but does not grow the total ``Mesh.AngularSegments`` budget. The
    remaining segments are uniform in azimuth on the two wall spans, split
    proportionally to their angular extents. Verified directly against ATH
    V2025-12 with AngularSegments=80, CornerSegments=4.
    """
    points_per_quadrant = max(1, int(points_per_quadrant))
    corner_radius = min(max(float(corner_radius), 0.0), half_width, half_height)
    del corner_segments  # budget-only in ATH; the arc structure is fixed
    if corner_radius <= 1.0e-9:
        return np.linspace(0.0, math.pi / 2.0, points_per_quadrant + 1, dtype=np.float64)

    theta1 = math.atan2(half_height - corner_radius, half_width)
    theta2 = math.atan2(half_height, half_width - corner_radius)
    arc_segments = 3
    side_segments = max(2, points_per_quadrant - arc_segments)
    span1 = theta1
    span2 = math.pi / 2.0 - theta2
    # A corner equal to a half-dimension removes one straight span entirely.
    # Do not force an interval onto that zero-length span: it would emit two
    # identical azimuths. Keep the fixed angular budget by assigning its
    # interval to the remaining span, or to the arc for a fully round target.
    collapsed_side1 = corner_radius >= half_height
    collapsed_side2 = corner_radius >= half_width
    if collapsed_side1 and collapsed_side2:
        return np.linspace(0.0, math.pi / 2.0, points_per_quadrant + 1, dtype=np.float64)
    if (collapsed_side1 or collapsed_side2) and points_per_quadrant == 1:
        return np.linspace(0.0, math.pi / 2.0, points_per_quadrant + 1, dtype=np.float64)
    if collapsed_side1 or collapsed_side2:
        # Keep the normal three arc intervals when the angular budget permits;
        # low-resolution grids reserve one interval for the surviving wall.
        arc_segments = min(arc_segments, points_per_quadrant - 1)
    if collapsed_side1:
        side1_segments = 0
        side2_segments = points_per_quadrant - arc_segments
    elif collapsed_side2:
        side1_segments = points_per_quadrant - arc_segments
        side2_segments = 0
    else:
        # ATH/C++ rounds an exact .5 upward; Python's bankers-rounding would put
        # the extra span on the opposite side of a symmetric quadrant.
        side1_share = side_segments * span1 / max(span1 + span2, 1.0e-12)
        side1_segments = max(1, int(math.floor(side1_share + 0.5)))
        side2_segments = max(1, side_segments - side1_segments)

    angles: list[float] = []
    if side1_segments:
        for i in range(side1_segments + 1):
            angles.append(theta1 * i / side1_segments)
    else:
        angles.append(0.0)
    cx = half_width - corner_radius
    cy = half_height - corner_radius
    for i in range(1, arc_segments + 1):
        corner_phi = (i / arc_segments) * math.pi / 2.0
        angles.append(math.atan2(cy + corner_radius * math.sin(corner_phi), cx + corner_radius * math.cos(corner_phi)))
    for i in range(1, side2_segments + 1):
        angles.append(theta2 + (math.pi / 2.0 - theta2) * i / side2_segments)
    return np.asarray(angles, dtype=np.float64)
