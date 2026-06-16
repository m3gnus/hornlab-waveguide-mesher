"""Guards, checks and the station report for the Intrinsic-Curvature Waveguide (ICW).

These are pure geometric / acoustic *prechecks* on a sampled meridian (:class:`ICWSample`
from :mod:`hornlab_mesher.icw.core`). They never mutate the kernel; they only read the
sample fields/gauges and report whether a candidate curve is manufacturable and physically
sensible before it is handed to the mesher / BEM stack.

Dependency-light by design: numpy + scipy only (no gmsh), so the kernel + checks together
stay extractable for a BEM-in-the-loop optimiser.

Vocabulary (see ``core.py`` for the gauge definitions):
    * meridional curvature       ``kappa_meridional``      = kappa
    * circumferential curvature  ``kappa_circumferential`` = cos(theta)/r
    * the two principal curvatures of the surface of revolution; both matter for a *regular*
      (non-self-intersecting) constant-thickness shell offset.

References to the "v3 plan" in comments point at the design notes: BOTH principal curvatures
gate the wall offset; the acoustic aperture is the theta=90 deg crossing (not the polyline
end, which may recede in a rollback); and the cubic B-spline support length is a smoothness
regulariser, NOT a higher-order-mode (HOM) guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Imported lazily-style for typing only; avoids any risk of circular import at module load.
from .core import MONO_EPS, ICWCurve, ICWSample, theta_half_pi_crossings

__all__ = [
    "is_monotone_radius",
    "meridian_self_intersects",
    "ShellOffsetReport",
    "shell_offset_report",
    "hom_cutoff_hz",
    "ApertureReport",
    "aperture_report",
    "feature_scale_ok",
    "StationReport",
    "station_report",
]

# First positive roots of the derivative of the Bessel function J_m (Neumann / hard-wall
# circular duct cutoffs): m=1 (first non-axisymmetric, mode "11") and m=0 (first axisymmetric
# radial, mode "01").
_JPRIME_11 = 1.8411837813406593
_JPRIME_01 = 3.8317059702075125


# ---------------------------------------------------------------------------------------
# 1. monotone radius
# ---------------------------------------------------------------------------------------
def is_monotone_radius(sample: ICWSample, eps: float = MONO_EPS) -> bool:
    """True iff the radius ``r`` is weakly non-decreasing (``dr >= -eps``) along the meridian.

    On the expansion ``dr/ds = sin(theta)``, so a healthy forward-flaring curve has a
    non-decreasing ``r``. The rule is deliberately WEAK -- it matches the solver's feasibility
    check (both use the shared :data:`core.MONO_EPS`) so the two never disagree on the same curve.
    A cylindrical or near-cylindrical section (``theta ~ 0`` => ``dr/ds ~ 0``), at the throat,
    along the body, or at an exit, is therefore accepted rather than falsely rejected; a genuinely
    re-entrant wall (radius turning back by more than the floor) still fails.

    Parameters
    ----------
    sample : sampled meridian.
    eps : tolerance; a step counts as non-decreasing iff ``dr >= -eps``. Defaults to the shared
        :data:`core.MONO_EPS` (the same floor the solver uses). Pass a smaller value for a stricter
        check.
    """
    r = np.asarray(sample.r, dtype=float)
    if r.size < 2:
        return True
    dr = np.diff(r)
    return bool(np.all(dr >= -eps))


# ---------------------------------------------------------------------------------------
# 2. planar polyline self-intersection
# ---------------------------------------------------------------------------------------
def _segments_intersect(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray, eps: float = 1e-12
) -> bool:
    """Robust test for whether segment ``p1->p2`` intersects segment ``p3->p4``.

    Uses signed-area (orientation) tests with a collinear-overlap fallback, so it correctly
    handles touching endpoints and collinear overlaps as intersections.
    """

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    def on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> bool:
        # p is known collinear with a,b; check it lies within the bounding box.
        return (
            min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
        )

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    # Proper crossing: the endpoints of each segment straddle the other's supporting line.
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and (
        (d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)
    ):
        return True

    # Collinear / touching cases.
    if abs(d1) <= eps and on_segment(p3, p4, p1):
        return True
    if abs(d2) <= eps and on_segment(p3, p4, p2):
        return True
    if abs(d3) <= eps and on_segment(p1, p2, p3):
        return True
    if abs(d4) <= eps and on_segment(p1, p2, p4):
        return True
    return False


def meridian_self_intersects(x: np.ndarray, r: np.ndarray) -> bool:
    """General planar self-intersection test of the meridian polyline ``(x[i], r[i])``.

    A rollback (``theta`` past 90 deg) makes the meridian non-single-valued in ``x``, so a
    monotone-in-``x`` shortcut is not safe; we test every non-adjacent segment pair. Adjacent
    segments share an endpoint by construction and are skipped (they would otherwise always
    "touch").

    O(n^2) in the number of segments. Stations default to ~1601, which is fine for a precheck;
    a coarse bounding-box reject keeps the constant small.
    """
    x = np.asarray(x, dtype=float)
    r = np.asarray(r, dtype=float)
    pts = np.column_stack([x, r])
    n = len(pts) - 1  # number of segments
    if n < 2:
        return False

    seg_a = pts[:-1]
    seg_b = pts[1:]
    # Per-segment bounding boxes for a cheap reject.
    xmin = np.minimum(seg_a[:, 0], seg_b[:, 0])
    xmax = np.maximum(seg_a[:, 0], seg_b[:, 0])
    rmin = np.minimum(seg_a[:, 1], seg_b[:, 1])
    rmax = np.maximum(seg_a[:, 1], seg_b[:, 1])

    for i in range(n):
        # j starts at i+2 to skip the adjacent segment i+1 (shares endpoint pts[i+1]).
        for j in range(i + 2, n):
            # Skip the wrap-adjacency of the first and last segment only if they share a point;
            # for an open polyline they do not, so no special case is needed.
            if xmax[i] < xmin[j] or xmax[j] < xmin[i]:
                continue
            if rmax[i] < rmin[j] or rmax[j] < rmin[i]:
                continue
            if _segments_intersect(seg_a[i], seg_b[i], seg_a[j], seg_b[j]):
                return True
    return False


# ---------------------------------------------------------------------------------------
# 3. shell offset (constant-thickness wall) report
# ---------------------------------------------------------------------------------------
@dataclass
class ShellOffsetReport:
    """Result of the constant-thickness wall-offset precheck.

    Attributes
    ----------
    ok : both the curvature precheck passed AND the offset polyline does not self-intersect.
    max_t_kappa_meridional : max ``|t * kappa_meridional|`` over the meridian.
    max_t_kappa_circumferential : max ``|t * kappa_circumferential|`` over the meridian.
    offset_self_intersects : whether the offset meridian polyline self-intersects.
    violations : human-readable reasons the check failed (empty iff ``ok``).
    """

    ok: bool
    max_t_kappa_meridional: float
    max_t_kappa_circumferential: float
    offset_self_intersects: bool
    violations: list[str] = field(default_factory=list)


def shell_offset_report(
    sample: ICWSample, wall_thickness: float, margin: float = 0.4
) -> ShellOffsetReport:
    """Check that a constant-thickness wall offset stays *regular* (no folds, no overlaps).

    A normal offset of a surface by ``t`` keeps offsetting regular only while ``t`` stays
    inside every centre of principal curvature, i.e. ``|t * kappa_i| < 1`` for each principal
    curvature ``kappa_i``. We apply a tighter precheck ``|t * kappa_i| < margin`` (``margin``
    ~0.3-0.5) on BOTH principal curvatures -- meridional ``kappa`` and circumferential
    ``cos(theta)/r`` -- because near ``|t*kappa| -> 1`` the offset is already badly distorted.

    The precheck is necessary but not sufficient (it is pointwise and local), so we THEN
    actually build the inward-normal offset of the meridian polyline and run the global
    self-intersection test :func:`meridian_self_intersects` on it.

    Parameters
    ----------
    sample : sampled meridian.
    wall_thickness : ``t``, the constant wall thickness (mm, same units as ``r``).
    margin : dimensionless curvature-times-thickness budget for the pointwise precheck.

    Notes
    -----
    The meridian unit tangent at angle ``theta`` is ``(cos theta, sin theta)``; a unit normal
    is ``(-sin theta, cos theta)``. For a curve flaring outward (``r`` increasing) that normal
    points *outward* in ``r``, so the *inward* (into-the-wall) normal -- the side a physical
    shell wall occupies -- is ``(sin theta, -cos theta)``. We offset by ``+t`` along that
    inward normal. (v3 plan: the wall lives between the inner acoustic surface and this
    offset.)
    """
    t = float(wall_thickness)
    theta = np.asarray(sample.theta, dtype=float)

    km = np.asarray(sample.kappa_meridional, dtype=float)
    kc = np.asarray(sample.kappa_circumferential, dtype=float)
    # Guard against any non-finite gauge value (e.g. r=0 at a degenerate throat).
    km = km[np.isfinite(km)]
    kc = kc[np.isfinite(kc)]

    max_t_km = float(np.max(np.abs(t * km))) if km.size else 0.0
    max_t_kc = float(np.max(np.abs(t * kc))) if kc.size else 0.0

    violations: list[str] = []
    if max_t_km >= margin:
        violations.append(
            f"|t*kappa_meridional|={max_t_km:.3f} >= margin={margin:.3f} "
            "(offset folds on the meridional principal curvature)"
        )
    if max_t_kc >= margin:
        violations.append(
            f"|t*kappa_circumferential|={max_t_kc:.3f} >= margin={margin:.3f} "
            "(offset folds on the circumferential principal curvature)"
        )

    # Build the actual inward-normal offset polyline and test it globally.
    inward_nx = np.sin(theta)
    inward_nr = -np.cos(theta)
    x_off = np.asarray(sample.x, dtype=float) + t * inward_nx
    r_off = np.asarray(sample.r, dtype=float) + t * inward_nr
    offset_self_intersects = meridian_self_intersects(x_off, r_off)
    if offset_self_intersects:
        violations.append("offset meridian polyline self-intersects")

    ok = not violations
    return ShellOffsetReport(
        ok=ok,
        max_t_kappa_meridional=max_t_km,
        max_t_kappa_circumferential=max_t_kc,
        offset_self_intersects=offset_self_intersects,
        violations=violations,
    )


# ---------------------------------------------------------------------------------------
# 4. circular hard-wall higher-order-mode cutoff
# ---------------------------------------------------------------------------------------
def hom_cutoff_hz(r_mm, mode: str = "11", c: float = 343.0):
    """First circular hard-wall (Neumann) higher-order-mode cutoff frequency.

    For a rigid-walled circular duct of radius ``r`` the cutoff of mode ``(m, n)`` is
    ``f = c * j'_{m,n} / (2 pi r)`` where ``j'_{m,n}`` is the n-th positive root of ``J'_m``.

    * mode ``"11"`` (first non-axisymmetric):  ``f = 1.841 * c / (2 pi r)``
    * mode ``"01"`` (first axisymmetric radial): ``f = 3.832 * c / (2 pi r)``

    Parameters
    ----------
    r_mm : radius in **millimetres** (scalar or array); converted to metres internally.
    mode : ``"11"`` or ``"01"``.
    c : sound speed (m/s).

    Returns
    -------
    Cutoff frequency in Hz (float for scalar ``r``, ``np.ndarray`` for array ``r``).

    Notes
    -----
    This is the *circular* result. For a **non-circular / morphed** cross-section replace
    ``1.841 / r`` (or ``3.832 / r``) with ``sqrt(lambda1)``, where ``lambda1`` is the first
    nonzero Neumann eigenvalue of the Laplacian on the *local* cross-section, giving
    ``f = (c / 2 pi) * sqrt(lambda1)``.

    A cutoff is NOT an excitation: a mode only carries energy if it is both above cutoff AND
    actually driven (asymmetry, off-axis source, wall slope, etc.). Use this gauge as the
    frequency below which a given mode is purely evanescent, not as a prediction that it rings.
    """
    if mode == "11":
        root = _JPRIME_11
    elif mode == "01":
        root = _JPRIME_01
    else:
        raise ValueError(f"unknown mode {mode!r}; expected '11' or '01'")

    r_m = np.asarray(r_mm, dtype=float) / 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        f = root * c / (2.0 * np.pi * r_m)
    if np.ndim(r_mm) == 0:
        return float(f)
    return f


# ---------------------------------------------------------------------------------------
# 5. acoustic aperture (theta = 90 deg) report
# ---------------------------------------------------------------------------------------
@dataclass
class ApertureReport:
    """Where the meridian reaches the mouth plane ``theta = 90 deg`` and what happens after.

    Attributes
    ----------
    n_pi2_crossings : number of times ``theta`` crosses 90 deg along the meridian.
    r_aperture : radius at the acoustic aperture (the crossing of maximal ``x``); ``None`` only
        if it cannot be defined (degenerate input).
    x_aperture : axial coordinate at the acoustic aperture; ``None`` likewise.
    r_end : radius at the polyline end ``r[-1]``.
    x_setback : ``x_aperture - x[-1]``; positive in a rollback (the end recedes behind the
        widest point). ``None`` if ``x_aperture`` is undefined.
    is_rollback : ``theta_end > 90 deg`` with at least one crossing and the end receding.
    """

    n_pi2_crossings: int
    r_aperture: float | None
    x_aperture: float | None
    r_end: float
    x_setback: float | None
    is_rollback: bool


def aperture_report(sample: ICWSample) -> ApertureReport:
    """Locate the acoustic aperture (``theta = 90 deg`` crossing) and detect rollback.

    The acoustic aperture is the widest annulus the wavefront passes through, i.e. the
    ``theta = pi/2`` crossing -- NOT necessarily the polyline end, which in a free-standing
    rollback (``theta`` past 90 deg) recedes back toward the axis. Among all crossings we pick
    the one of maximal ``x`` (the foremost mouth plane); ``r_aperture``/``x_aperture`` are taken
    there by linear interpolation in ``theta``.

    For a flat-baffle / forward curve with no interior crossing (``theta`` stays below 90 deg),
    the aperture degenerates to the polyline end: ``r_aperture = r_end``, ``x_setback = 0``,
    ``is_rollback = False``.

    Parameters
    ----------
    sample : sampled meridian.
    """
    theta = np.asarray(sample.theta, dtype=float)
    x = np.asarray(sample.x, dtype=float)
    r = np.asarray(sample.r, dtype=float)
    r_end = float(r[-1])

    # Use the ONE shared crossing definition (also used by the solver) so n_pi2_crossings here and
    # the solver's rollback count can never disagree on the same curve. It counts every sign change
    # of theta-90deg (ascending/descending/exact node) and collapses plateaus (a flat run on the
    # mouth plane counts once, never per node).
    crossings = theta_half_pi_crossings(theta)
    cross_x = [float(x[i] + frac * (x[i + 1] - x[i])) if i + 1 < x.size else float(x[i])
               for i, frac in crossings]
    cross_r = [float(r[i] + frac * (r[i + 1] - r[i])) if i + 1 < r.size else float(r[i])
               for i, frac in crossings]
    n_crossings = len(crossings)

    if n_crossings == 0:
        # No interior crossing: aperture is the end of the curve.
        return ApertureReport(
            n_pi2_crossings=0,
            r_aperture=r_end,
            x_aperture=float(x[-1]),
            r_end=r_end,
            x_setback=0.0,
            is_rollback=False,
        )

    # Acoustic aperture = crossing of maximal x (foremost mouth plane).
    idx = int(np.argmax(cross_x))
    x_aperture = cross_x[idx]
    r_aperture = cross_r[idx]
    x_setback = x_aperture - float(x[-1])

    theta_end = float(theta[-1])
    is_rollback = (theta_end > np.pi / 2.0) and (n_crossings >= 1) and (x_setback > 0.0)

    return ApertureReport(
        n_pi2_crossings=n_crossings,
        r_aperture=r_aperture,
        x_aperture=x_aperture,
        r_end=r_end,
        x_setback=x_setback,
        is_rollback=is_rollback,
    )


# ---------------------------------------------------------------------------------------
# 6. feature-scale (B-spline support) check
# ---------------------------------------------------------------------------------------
def feature_scale_ok(curve: ICWCurve, min_support_mm: float) -> bool:
    """True iff the curvature B-spline's basis support is at least ``min_support_mm`` long.

    A degree-``d`` B-spline basis function spans ``d + 1`` knot intervals. For the default
    cubic (``d = 3``) that is 4 intervals, so the smallest feature the curvature field can
    represent has full support ``~ (d + 1) * (S / n_interior_intervals)`` in mm, where
    ``n_interior_intervals`` is the number of interior knot spans on ``sigma in [0, 1]``.

    Parameters
    ----------
    curve : the ICW curve (for ``S``, knots, degree).
    min_support_mm : the minimum acceptable feature length in mm.

    Notes
    -----
    This is a **smoothness regulariser**, not a HOM guarantee. The common rule-of-thumb
    ``lambda_min / 4 ~= 4.3 mm`` at 20 kHz (``c = 343 m/s``) bounds how sharp a curvature
    wiggle the wall may carry so the surface stays smooth at audio wavelengths; it does NOT
    by itself prevent higher-order modes (see :func:`hom_cutoff_hz`).
    """
    degree = int(curve.degree)
    knots = np.asarray(curve.knots, dtype=float)
    # Distinct (de-duplicated) knot values; interior spans are the gaps between them on (0,1).
    uniq = np.unique(knots)
    n_interior_intervals = len(uniq) - 1
    if n_interior_intervals < 1:
        return False
    # The smallest feature the curvature field can represent is set by the TIGHTEST (minimum)
    # knot span, not the largest: a degree-d basis function over the narrowest (d+1)-interval
    # cluster has the shortest support, and that is what bounds the sharpest representable wiggle.
    # Density-clustered knots (as the seed-fitter emits) pack spans near a curvature cusp, so a
    # max-span test would report a comfortable feature length while a tight cluster below the floor
    # slips through. Using the minimum span keeps the guard genuinely conservative.
    min_span = float(np.min(np.diff(uniq)))
    support_sigma = (degree + 1) * min_span
    support_mm = float(curve.S) * support_sigma
    return support_mm >= float(min_support_mm)


# ---------------------------------------------------------------------------------------
# 7. station report
# ---------------------------------------------------------------------------------------
@dataclass
class StationReport:
    """Summary maxima/minima of the meridian gauges plus the aperture and validity flags.

    Geometric extrema (``theta_deg``, ``kappa``, ``dkappa_ds``) are over the whole meridian.
    Webster body gauges (``Fx``, ``Qx``) are restricted to ``theta <= theta_body_max`` because
    they diverge as ``theta -> 90 deg`` (the ``1/cos`` and ``1/cos^3`` factors).
    """

    # geometric extrema (whole meridian)
    theta_deg_max: float
    theta_deg_min: float
    kappa_max: float
    kappa_min: float
    abs_kappa_max: float
    dkappa_ds_max: float
    dkappa_ds_min: float
    abs_dkappa_ds_max: float
    # Webster body gauges, restricted to theta <= theta_body_max
    Fx_max: float
    Fx_min: float
    Qx_max: float
    Qx_min: float
    abs_Qx_max: float
    # aperture
    aperture: ApertureReport
    # validity
    theta_body_max_deg: float
    theta_body_valid: bool


def station_report(sample: ICWSample, theta_body_max_deg: float = 72.0) -> StationReport:
    """Build the per-curve station report from a sampled meridian.

    ``dkappa/ds`` is computed by :func:`numpy.gradient` over arc length ``s``. The Webster body
    gauges ``Fx = 2 tan(theta)/r`` and ``Qx = kappa/(r cos^3 theta)`` are evaluated only where
    ``theta <= theta_body_max_deg`` (they diverge near 90 deg and are physically untrustworthy
    there). ``theta_body_valid`` records whether the body actually stays inside that trusted
    band -- i.e. whether the *maximum* meridian angle is at or below ``theta_body_max_deg``.

    Parameters
    ----------
    sample : sampled meridian.
    theta_body_max_deg : the angle (deg) above which ``Fx``/``Qx`` are not trusted.
    """
    theta = np.asarray(sample.theta, dtype=float)
    theta_deg = sample.wall_angle_deg()
    kappa = np.asarray(sample.kappa, dtype=float)
    s = np.asarray(sample.s, dtype=float)

    dkappa_ds = np.gradient(kappa, s)

    theta_deg_max = float(np.max(theta_deg))
    theta_deg_min = float(np.min(theta_deg))

    kappa_max = float(np.max(kappa))
    kappa_min = float(np.min(kappa))
    abs_kappa_max = float(np.max(np.abs(kappa)))

    dk_max = float(np.max(dkappa_ds))
    dk_min = float(np.min(dkappa_ds))
    abs_dk_max = float(np.max(np.abs(dkappa_ds)))

    # Restrict body gauges to the trusted band theta <= theta_body_max.
    body = theta <= np.radians(theta_body_max_deg)
    Fx = np.asarray(sample.flare_rate_Fx, dtype=float)
    Qx = np.asarray(sample.webster_Qx, dtype=float)
    Fx_body = Fx[body]
    Qx_body = Qx[body]
    # Keep only finite values inside the band (defensive; throat r could be 0).
    Fx_body = Fx_body[np.isfinite(Fx_body)]
    Qx_body = Qx_body[np.isfinite(Qx_body)]

    if Fx_body.size:
        Fx_max = float(np.max(Fx_body))
        Fx_min = float(np.min(Fx_body))
    else:
        Fx_max = Fx_min = float("nan")
    if Qx_body.size:
        Qx_max = float(np.max(Qx_body))
        Qx_min = float(np.min(Qx_body))
        abs_Qx_max = float(np.max(np.abs(Qx_body)))
    else:
        Qx_max = Qx_min = abs_Qx_max = float("nan")

    theta_body_valid = theta_deg_max <= theta_body_max_deg

    return StationReport(
        theta_deg_max=theta_deg_max,
        theta_deg_min=theta_deg_min,
        kappa_max=kappa_max,
        kappa_min=kappa_min,
        abs_kappa_max=abs_kappa_max,
        dkappa_ds_max=dk_max,
        dkappa_ds_min=dk_min,
        abs_dkappa_ds_max=abs_dk_max,
        Fx_max=Fx_max,
        Fx_min=Fx_min,
        Qx_max=Qx_max,
        Qx_min=Qx_min,
        abs_Qx_max=abs_Qx_max,
        aperture=aperture_report(sample),
        theta_body_max_deg=theta_body_max_deg,
        theta_body_valid=theta_body_valid,
    )
