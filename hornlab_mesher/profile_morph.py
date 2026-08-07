from __future__ import annotations

import math
from typing import Any, Literal, Mapping, NamedTuple

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


_COVERAGE_ANGLE_MIN = 0.5
_COVERAGE_ANGLE_MAX = 89.0


class CoverageInversion(NamedTuple):
    """Result of solving the OSSE coverage angle against a guiding curve.

    ``saturated`` names the bracket end the target fell outside of, or is
    ``None`` when the guiding curve was actually met. The bisection is over a
    fixed ``[0.5, 89]`` degree bracket, and ``_osse_radius`` is monotonically
    increasing in the coverage angle, so a target outside ``[r(0.5), r(89)]``
    is unreachable: the solver then returns the nearest bracket end and the
    mouth silently lands somewhere other than the guiding curve. Callers that
    can report to the user should surface :func:`coverage_angle_saturation`
    instead of letting that pass unnoticed.
    """

    angle_deg: float
    saturated: Literal["min", "max"] | None
    achieved_radius: float
    target_radius: float
    #: Axial station the inversion was solved at. Equals the main length only
    #: when ``gcurveDist`` puts the guiding curve at the mouth, so the achieved
    #: radius is NOT a mouth radius in general.
    station_z: float = 0.0
    at_mouth: bool = True


def _probe_osse_coverage_bracket(
    target_radius: float,
    z_main: float,
    p: float,
    params: Mapping[str, Any],
    *,
    a0_deg: float,
    r0_main: float,
    at_mouth: bool = True,
) -> CoverageInversion | None:
    """Bracket-end verdict for the coverage inversion, without the bisection.

    Returns a saturated :class:`CoverageInversion` when the requested radius
    falls strictly outside ``[r(0.5), r(89)]``, and ``None`` when it does not
    -- either because the target is reachable, or because the radius is
    undefined at a bracket end and no verdict is supportable. ``None`` is
    therefore "nothing to report", not "reachable".

    Two radius evaluations instead of the full inversion's 26. The saturated
    branch of :func:`_invert_osse_coverage_angle` never bisects either, so this
    loses nothing there; what it saves is the *healthy* case, which is the one
    a sweep pays for on every frame. Measured: 5 us per azimuth against 36 us
    for the full inversion, which is what makes a fine-grained azimuth sweep
    affordable in the preview.
    """

    r_low = _osse_radius_or_nan(
        _COVERAGE_ANGLE_MIN, z_main, p, params, a0_deg=a0_deg, r0_main=r0_main
    )
    r_high = _osse_radius_or_nan(
        _COVERAGE_ANGLE_MAX, z_main, p, params, a0_deg=a0_deg, r0_main=r0_main
    )
    if not math.isfinite(r_low) or not math.isfinite(r_high):
        return None
    # Strict comparisons: a target exactly equal to a bracket end is reachable
    # AT that end, not outside it.
    if target_radius < r_low:
        return CoverageInversion(
            _COVERAGE_ANGLE_MIN, "min", r_low, target_radius, z_main, at_mouth
        )
    if target_radius > r_high:
        return CoverageInversion(
            _COVERAGE_ANGLE_MAX, "max", r_high, target_radius, z_main, at_mouth
        )
    return None


def _osse_radius_or_nan(
    a_deg: float,
    z_main: float,
    p: float,
    params: Mapping[str, Any],
    *,
    a0_deg: float,
    r0_main: float,
) -> float:
    # A negative radicand inside _osse_radius (reachable with a negative a0
    # at a small coverage angle) raises rather than returning a non-finite
    # value. The JS engine gets NaN there and abandons the step, so swallow
    # it to the same NaN instead of turning a probe into a crash: the probe
    # visits angles the bisection alone would never have reached.
    try:
        return _osse_radius(
            z_main, p, params, r0=r0_main, a_deg=a_deg, a0_deg=a0_deg
        )
    except (ValueError, OverflowError, ZeroDivisionError):
        return math.nan


def _invert_osse_coverage_angle(
    target_radius: float,
    z_main: float,
    p: float,
    params: Mapping[str, Any],
    *,
    a0_deg: float,
    r0_main: float,
    at_mouth: bool = True,
) -> CoverageInversion:
    def radius_at(a_deg: float) -> float:
        return _osse_radius_or_nan(
            a_deg, z_main, p, params, a0_deg=a0_deg, r0_main=r0_main
        )

    def bisect(low: float, high: float) -> float:
        for _ in range(24):
            mid = 0.5 * (low + high)
            radius = radius_at(mid)
            if not math.isfinite(radius):
                break
            if radius < target_radius:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    low = _COVERAGE_ANGLE_MIN
    high = _COVERAGE_ANGLE_MAX
    # Probe the bracket ends before bisecting: the bisection alone cannot tell
    # "converged onto the target" from "ran into the bracket end", and the
    # latter is wrong geometry rather than an approximation of the requested
    # one. Two extra evaluations per azimuth, hoisted with the inversion.
    #
    # A None here means either "reachable" or "undefined at a bracket end". In
    # the second case the probe supports no verdict at all, so falling through
    # to the plain bisection and reporting no diagnosis is right for both:
    # claiming saturation off the back of a NaN would put one in front of the
    # user.
    saturated = _probe_osse_coverage_bracket(
        target_radius,
        z_main,
        p,
        params,
        a0_deg=a0_deg,
        r0_main=r0_main,
        at_mouth=at_mouth,
    )
    if saturated is not None:
        return saturated

    angle = bisect(low, high)
    return CoverageInversion(
        angle, None, radius_at(angle), target_radius, z_main, at_mouth
    )


def _solve_coverage_from_guiding_curve(
    p: float,
    params: Mapping[str, Any],
    *,
    main_length: float,
    a0_deg: float,
    r0_main: float,
    probe_only: bool = False,
) -> CoverageInversion | None:
    """Solve the coverage angle against the guiding curve at azimuth ``p``.

    ``probe_only`` skips the bisection and answers only "is the guiding curve
    out of reach here": it returns a saturated result or ``None``, never a
    solved angle. Callers that sweep many azimuths purely to look for
    saturation should use it -- it is the difference between 36 us and 5 us per
    azimuth, because the bisection is paid on exactly the healthy azimuths that
    have nothing to report.
    """

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
    solve = (
        _probe_osse_coverage_bracket if probe_only else _invert_osse_coverage_angle
    )
    return solve(
        target_radius,
        target_z,
        p,
        params,
        a0_deg=a0_deg,
        r0_main=r0_main,
        at_mouth=math.isclose(target_z, main_length, rel_tol=1e-12, abs_tol=1e-9),
    )


def _coverage_angle_from_guiding_curve(
    p: float,
    params: Mapping[str, Any],
    *,
    main_length: float,
    a0_deg: float,
    r0_main: float,
) -> float | None:
    solved = _solve_coverage_from_guiding_curve(
        p, params, main_length=main_length, a0_deg=a0_deg, r0_main=r0_main
    )
    return None if solved is None else solved.angle_deg


def coverage_angle_saturation(
    p: float,
    params: Mapping[str, Any],
    *,
    main_length: float,
    a0_deg: float,
    r0_main: float,
    location: str | None = None,
) -> str | None:
    """Human-readable reason the guiding curve could not be met at ``p``.

    ``None`` when there is no guiding curve or the mouth lands on it. The
    message names the actionable direction, because the failure mode users hit
    is a length that has grown past the point where any coverage angle can
    still reach the requested mouth: past that point every other parameter
    stops changing the mouth at all.

    ``location`` overrides the "at phi=N deg" clause for callers that probed
    several azimuths and know the miss is not confined to this one.
    """

    solved = _solve_coverage_from_guiding_curve(
        p, params, main_length=main_length, a0_deg=a0_deg, r0_main=r0_main
    )
    if solved is None or solved.saturated is None:
        return None
    where = location or f"phi={math.degrees(p) % 360.0:.1f} deg"
    if solved.saturated == "min":
        remedy = "shorten the horn (Length), reduce the termination shape s, or widen the guiding curve"
    else:
        remedy = "lengthen the horn (Length) or narrow the guiding curve"
    # The inversion is solved where the guiding curve sits, which is the mouth
    # only when GCurve.Dist is 1. Naming the mid-horn radius a "mouth radius"
    # would send the user chasing a number they cannot measure -- and the v1
    # schema still defaults GCurve.Dist to 0.5, so that is the common case.
    if solved.at_mouth:
        station = "the mouth radius"
    else:
        station = f"the radius at the guiding-curve distance (z={solved.station_z:.1f} mm)"
    return (
        f"guiding curve unreachable at {where}: the coverage angle "
        f"is pinned at {solved.angle_deg:g} deg, so {station} is "
        f"{solved.achieved_radius:.1f} mm instead of the requested "
        f"{solved.target_radius:.1f} mm; {remedy}"
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


class _RoundedRectQuadrantLayout(NamedTuple):
    """Interval budget for one quadrant of a rounded-rectangle morph target."""

    theta1: float
    theta2: float
    corner_radius: float
    arc_segments: int
    side1_segments: int
    side2_segments: int


def _rounded_rect_quadrant_layout(
    points_per_quadrant: int,
    half_width: float,
    half_height: float,
    corner_radius: float,
) -> _RoundedRectQuadrantLayout | None:
    """Resolve the quadrant interval budget, or ``None`` for a uniform quadrant.

    ``None`` means the sampler degenerates to a plain azimuth ``linspace`` (no
    corner radius, or a fully round target): there is no fixed-structure arc,
    so those quadrants refine normally with the angular budget.
    """

    points_per_quadrant = max(1, int(points_per_quadrant))
    corner_radius = min(max(float(corner_radius), 0.0), half_width, half_height)
    if corner_radius <= 1.0e-9:
        return None

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
        return None
    if (collapsed_side1 or collapsed_side2) and points_per_quadrant == 1:
        return None
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
        side1_segments = max(
            1,
            int(round(side_segments * span1 / max(span1 + span2, 1.0e-12))),
        )
        side2_segments = max(1, side_segments - side1_segments)
    return _RoundedRectQuadrantLayout(
        theta1=theta1,
        theta2=theta2,
        corner_radius=corner_radius,
        arc_segments=arc_segments,
        side1_segments=side1_segments,
        side2_segments=side2_segments,
    )


def rounded_rect_corner_arc_span(
    points_per_quadrant: int,
    half_width: float,
    half_height: float,
    corner_radius: float,
) -> tuple[float, float] | None:
    """Azimuth span ``[theta1, theta2]`` covered by the fixed corner arc.

    ``None`` when this target has no fixed-structure arc. The acoustic sampling
    loop uses the span to tell corner intervals (whose chord only shrinks when
    the arc itself is subdivided) apart from ordinary wall intervals.
    """

    layout = _rounded_rect_quadrant_layout(
        points_per_quadrant, half_width, half_height, corner_radius
    )
    if layout is None:
        return None
    return (layout.theta1, layout.theta2)


def _rounded_rect_quadrant_angles(
    points_per_quadrant: int,
    half_width: float,
    half_height: float,
    corner_radius: float,
    corner_segments: int,
    *,
    arc_subdivision: int = 1,
) -> np.ndarray:
    """First-quadrant azimuth samples for a rounded-rectangle morph target.

    ATH always samples the corner arc with four profiles per quadrant (both
    wall-tangency endpoints plus two interior points at 30/60 degrees of arc
    parameter) regardless of ``Mesh.CornerSegments``, which grows the total
    angular point budget. The remaining segments are uniform in azimuth on the
    two wall spans, split proportionally to their angular extents. Verified
    against the ATH m2-clone (CornerSegments 4) and solana (CornerSegments 1)
    reference grids.

    ``arc_subdivision`` splits every arc interval into that many equal
    sub-intervals, leaving the wall budget untouched. It defaults to 1, so the
    public/ATH sampling is unchanged; only the acoustic control-grid fit raises
    it, because ATH's fixed three intervals pin the corner chord at
    ``2*R*sin(15 deg)`` no matter how large the angular budget grows. Splitting
    into ``3k`` equal intervals keeps ATH's four canonical profiles as an exact
    subset (indices ``k``, ``2k``, ``3k`` reproduce the same arc parameters).
    """
    points_per_quadrant = max(1, int(points_per_quadrant))
    corner_radius = min(max(float(corner_radius), 0.0), half_width, half_height)
    del corner_segments  # budget-only in ATH; the arc structure is fixed
    layout = _rounded_rect_quadrant_layout(
        points_per_quadrant, half_width, half_height, corner_radius
    )
    if layout is None:
        return np.linspace(0.0, math.pi / 2.0, points_per_quadrant + 1, dtype=np.float64)

    theta1 = layout.theta1
    theta2 = layout.theta2
    side1_segments = layout.side1_segments
    side2_segments = layout.side2_segments
    arc_segments = layout.arc_segments * max(1, int(arc_subdivision))

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
