"""Seed-fitter: reproduce existing analytic meridians as ICW curvature curves.

Given a meridian polyline ``(x[], r[])`` from an analytic profile (OSSE / R-OSSE), this
module recovers the intrinsic-curvature representation the ICW kernel expects:

    s      = cumulative arc length of (x, r)            S = s[-1]
    theta  = unwrap(atan2(dr, dx))                      tangent angle (rad)
    kappa  = dtheta/ds                                  signed meridional curvature (1/mm)

``kappa`` is then fitted as a clamped cubic B-spline over the normalised arc length
``sigma = s/S in [0, 1]`` by plain least squares on the B-spline coefficients (the design
matrix). Curvature may be negative -- rollback lips, inflections -- so **no** non-negativity
constraint is imposed. The throat boundary conditions ``r0 = r[0]``, ``theta0 = theta(0)``,
``x0 = x[0]`` are read straight off the polyline.

Arc-length parameterisation plus angle unwrapping means non-monotone ``x`` (rolled-back
mouths where the wall folds back) and ``theta`` crossing 90 deg are handled with no special
casing: arc length is always monotone and ``unwrap`` follows ``theta`` smoothly past 90 deg.

Two implementation choices matter for the fit quality:

* **Segment-midpoint curvature.** ``theta`` is evaluated per *segment* (its exact chord
  angle) at the segment midpoint, and ``kappa`` is the centred difference of those midpoint
  angles. This gives the true endpoint curvature (e.g. the OSSE throat value) instead of the
  ~2x-too-small one-sided ``np.gradient`` estimate that a naive node-based scheme produces --
  the single biggest lever for sub-micron OSSE fits.
* **Curvature-density knots.** Interior knots are placed by the cumulative ``|dkappa/dsigma|``
  density (with a uniform floor), so they cluster where the curvature actually bends -- the
  OSSE throat, the R-OSSE lip -- rather than spreading uniformly. This converges far faster
  than uniform knots on profiles whose curvature concentrates near an endpoint.

Only numpy + scipy are used (no gmsh), so the fitter stays usable inside a BEM-in-the-loop
optimiser without the meshing stack.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.interpolate import BSpline

from .core import DEFAULT_DEGREE, ICWCurve

# Floor (as a fraction of the peak |dkappa/dsigma|) added to the knot-density weight so smooth
# regions still receive knots. ~0.1 robustly beats uniform knots on both OSSE and R-OSSE.
_KNOT_DENSITY_FLOOR = 0.1


def _arc_length_frame(
    x: np.ndarray, r: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return ``(s, sigma, theta0, kappa)`` for a meridian polyline.

    ``s`` is cumulative arc length (mm), ``sigma = s/S`` is normalised, ``theta0`` is the
    throat tangent angle (rad) and ``kappa`` is the signed curvature (1/mm) sampled at the
    polyline nodes.

    ``kappa`` is built from the exact per-segment chord angles at segment midpoints, then
    centred-differenced; the endpoint nodes are extrapolated from the first/last interior
    pair. This is accurate at the boundaries even where the curvature is large there (the
    OSSE/R-OSSE throat), and robust to non-monotone ``x`` (rollback) and ``theta`` past 90 deg
    because arc length and the unwrapped angle are both well behaved.
    """
    x = np.asarray(x, dtype=float).ravel()
    r = np.asarray(r, dtype=float).ravel()
    if x.size != r.size:
        raise ValueError("x and r must have the same length")
    if x.size < 4:
        raise ValueError("need at least 4 meridian points to fit curvature")

    dx = np.diff(x)
    dr = np.diff(r)
    seg = np.hypot(dx, dr)
    if np.any(seg <= 0.0):
        # Drop zero-length segments so arc length and angles stay well defined.
        keep = np.concatenate([[True], seg > 0.0])
        return _arc_length_frame(x[keep], r[keep])

    s = np.concatenate([[0.0], np.cumsum(seg)])
    S = float(s[-1])
    if S <= 0.0:
        raise ValueError("meridian has zero arc length")
    sigma = s / S

    # Exact tangent angle of each segment, located at the segment midpoint.
    theta_seg = np.unwrap(np.arctan2(dr, dx))
    s_mid = 0.5 * (s[1:] + s[:-1])

    # Curvature at interior nodes = centred difference of midpoint angles over midpoint s.
    kappa = np.empty_like(s)
    kappa_int = np.diff(theta_seg) / np.diff(s_mid)
    s_int = s[1:-1]
    kappa[1:-1] = kappa_int
    # Linear extrapolation of the curvature to the two endpoint nodes.
    kappa[0] = kappa_int[0] + (kappa_int[1] - kappa_int[0]) * (
        s[0] - s_int[0]
    ) / (s_int[1] - s_int[0])
    kappa[-1] = kappa_int[-1] + (kappa_int[-2] - kappa_int[-1]) * (
        s[-1] - s_int[-1]
    ) / (s_int[-2] - s_int[-1])

    # Throat tangent angle: the first midpoint angle minus its turn back to s=0.
    theta0 = float(theta_seg[0] - kappa[0] * (s_mid[0] - s[0]))
    return s, sigma, theta0, kappa


def _density_knots(
    sigma: np.ndarray,
    kappa: np.ndarray,
    n_coeff: int,
    degree: int = DEFAULT_DEGREE,
    floor: float = _KNOT_DENSITY_FLOOR,
) -> np.ndarray:
    """Clamped knot vector on ``[0, 1]`` with interior knots by curvature-bend density.

    Interior knots are placed at equal increments of the cumulative ``|dkappa/dsigma|``
    (regularised by a uniform ``floor``), so they cluster where the curvature changes fastest
    -- the OSSE throat, the R-OSSE lip -- and converge much faster than uniform knots there.
    With ``floor`` large the placement degrades gracefully to uniform.
    """
    n_interior = n_coeff - degree - 1
    if n_interior <= 0:
        return np.concatenate([np.zeros(degree + 1), np.ones(degree + 1)])

    dk = np.abs(np.gradient(kappa, sigma))
    peak = float(dk.max())
    weight = dk + floor * peak + 1e-12
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(sigma))])
    cdf /= cdf[-1]

    targets = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    interior = np.interp(targets, cdf, sigma)
    interior = np.clip(np.sort(interior), 1e-9, 1.0 - 1e-9)
    return np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])


def fit_from_points(
    x: Sequence[float] | np.ndarray,
    r: Sequence[float] | np.ndarray,
    n_coeff: int = 16,
    degree: int = DEFAULT_DEGREE,
) -> ICWCurve:
    """Fit an :class:`ICWCurve` whose sampled meridian reproduces the polyline ``(x, r)``.

    Parameters
    ----------
    x, r : meridian coordinates (mm), throat first. ``x`` may be non-monotone (rollback).
    n_coeff : number of curvature B-spline coefficients. The default 16 reaches sub-micron on
        typical OSSE bodies; raise it for near-vertical / ``q -> 1`` OSSE mouths and for
        R-OSSE lips or throats with a sharp curvature cusp (24-64 progressively tightens the
        fit -- see the module-level notes and ``seed_from_rosse``).
    degree : B-spline degree (cubic by default).

    Returns
    -------
    ICWCurve with ``r0 = r[0]``, ``theta0 = theta(0)``, ``x0 = x[0]`` and ``S`` the total
    arc length of the input polyline.

    The curvature coefficients are recovered by plain least squares (``np.linalg.lstsq``)
    against the clamped-cubic B-spline design matrix; ``kappa`` is allowed to be negative.
    Interior knots are placed by curvature-bend density (see :func:`_density_knots`).
    """
    n_coeff = int(n_coeff)
    if n_coeff < degree + 1:
        # Match clamped_uniform_knots: fewer than degree+1 coefficients cannot define the spline.
        # Without this, _density_knots silently promotes to a degree+1 basis and returns a curve
        # with MORE coefficients than requested.
        raise ValueError(f"need at least degree+1={degree + 1} coefficients, got {n_coeff}")
    x_arr = np.asarray(x, dtype=float).ravel()
    r_arr = np.asarray(r, dtype=float).ravel()
    s, sigma, theta0, kappa = _arc_length_frame(x_arr, r_arr)

    knots = _density_knots(sigma, kappa, n_coeff, degree)
    # Design matrix B[j, i] = B_i(sigma_j); solve B @ coeffs ~= kappa in least squares.
    design = BSpline.design_matrix(sigma, knots, degree).toarray()
    coeffs, *_ = np.linalg.lstsq(design, kappa, rcond=None)

    return ICWCurve(
        coeffs=coeffs,
        S=float(s[-1]),
        r0=float(r_arr[0]),
        theta0=theta0,
        knots=knots,
        degree=degree,
        x0=float(x_arr[0]),
    )


def fit_error(
    curve: "ICWCurve",
    x: Sequence[float] | np.ndarray,
    r: Sequence[float] | np.ndarray,
    n: int = 4000,
) -> dict:
    """Euclidean deviation of a fitted curve from the input polyline.

    The curve is reconstructed with ``curve.sample(n)`` and compared against the polyline.
    Both are re-parameterised by *normalised arc length* and the reconstructed meridian is
    interpolated onto the polyline's arc-length stations, so the (x, r) deviation is measured
    fairly point-for-point rather than against a possibly different sample distribution.

    Returns ``{"max_mm": ..., "mean_mm": ...}`` in millimetres.
    """
    x = np.asarray(x, dtype=float).ravel()
    r = np.asarray(r, dtype=float).ravel()

    # Polyline arc-length parameter (normalised).
    s_poly = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(r)))])
    sig_poly = s_poly / float(s_poly[-1])

    # Reconstructed meridian, re-parameterised by its own normalised arc length so the
    # comparison does not assume the two share a sigma mapping.
    sample = curve.sample(n)
    xs, rs = sample.x, sample.r
    s_rec = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(rs)))])
    sig_rec = s_rec / float(s_rec[-1])

    x_rec = np.interp(sig_poly, sig_rec, xs)
    r_rec = np.interp(sig_poly, sig_rec, rs)

    dist = np.hypot(x_rec - x, r_rec - r)
    return {"max_mm": float(np.max(dist)), "mean_mm": float(np.mean(dist))}


def seed_from_osse(params: dict, n_coeff: int = 20, n_axial: int = 4000) -> ICWCurve:
    """Fit an ICW curve to an OSSE profile.

    ``params`` is an OSSE parameter dict (``params["type"] == "OSSE"``); the existing OSSE
    sampler (:func:`profile_points`) generates the meridian, which is then handed to
    :func:`fit_from_points`. The defaults reach ~20 nm max deviation on typical bodies; raise
    ``n_coeff`` for near-vertical / ``q -> 1`` mouths whose curvature spikes near the throat.
    """
    # Lazy import keeps the kernel standalone/extractable: importing
    # ``hornlab_mesher.icw`` must not pull in the mesher profile layer.
    from ..profile_formulas import profile_points

    pts = profile_points(params, n_axial)
    return fit_from_points(pts[:, 0], pts[:, 1], n_coeff=n_coeff)


def seed_from_rosse(params: dict, n_coeff: int = 20, n_axial: int = 4000) -> ICWCurve:
    """Fit an ICW curve to an R-OSSE profile.

    ``params`` is an R-OSSE parameter dict (``params["type"] == "R-OSSE"``). Rolled-back
    mouths (non-monotone ``x``, ``theta`` crossing 90 deg) are handled by the arc-length /
    unwrap framing in :func:`fit_from_points`; the slightly larger default ``n_coeff=20``
    covers the sharper lip curvature of R-OSSE. Configs with a curvature *cusp* at the throat
    (e.g. low ``q`` with a steep ``a``) need a larger ``n_coeff`` -- raise it to 32-64.
    """
    # Lazy import keeps the kernel standalone/extractable: importing
    # ``hornlab_mesher.icw`` must not pull in the mesher profile layer.
    from ..profile_formulas import profile_points

    pts = profile_points(params, n_axial)
    return fit_from_points(pts[:, 0], pts[:, 1], n_coeff=n_coeff)
