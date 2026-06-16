"""Intrinsic-Curvature Waveguide (ICW) kernel.

The canonical ICW geometry is an intrinsic (Cesaro/Whewell) plane curve: the meridian
**curvature** ``kappa`` is a clamped cubic B-spline over normalised arc length
``sigma = s/S in [0, 1]``; integrating it gives the tangent angle ``theta(s)``; integrating
again gives the meridian ``(x(s), r(s))``::

    kappa(sigma) = sum_i a_i B_i(sigma)         clamped cubic B-spline
    theta(s)     = theta0 + S * integral_0^sigma kappa dv
    x(s)         = x0 + S * integral_0^sigma cos(theta) dv
    r(s)         = r0 + S * integral_0^sigma sin(theta) dv

This module is deliberately dependency-light (numpy + scipy only, **no gmsh**) so the kernel
stays extractable and usable by a future BEM-in-the-loop optimiser without the meshing stack.

Acoustic gauges (valid only in the graph-like body, ``theta`` well below 90 deg):
    flare rate         F_x = 2 tan(theta) / r           = d ln S / dx,  S = pi r^2
    Webster potential  Q_x = kappa / (r cos^3 theta)    = r'' / r   (the term in psi'' + [k^2 - Q]psi = 0)
Geometric quantities (valid everywhere, including the steep lip / rollback):
    meridional curvature       kappa_meridional      = kappa
    circumferential curvature  kappa_circumferential = cos(theta) / r
    arc-length flare           F_s = 2 sin(theta) / r
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import BSpline

DEFAULT_DEGREE = 3
DEFAULT_SAMPLES = 1601

# Shared monotone-radius tolerance (mm). A radius step counts as non-decreasing iff
# ``dr >= -MONO_EPS``. It is WEAK on purpose: a cylindrical throat/exit (theta = 0 => dr/ds = 0)
# or a momentarily flat near-cylindrical section is legitimate and must pass, while the trapezoid
# quadrature leaves O(1e-8 mm) sign noise on such flats. A genuinely re-entrant wall reverses by
# O(0.1 mm)+, far above this floor, so it is still caught. Both the solver's feasibility check and
# :func:`checks.is_monotone_radius` use this one value so they agree on the same curve.
MONO_EPS = 1e-6


def theta_half_pi_crossings(theta: np.ndarray) -> list[tuple[int, float]]:
    """Locate every ``theta = 90 deg`` crossing of a sampled meridian angle.

    This is the *single shared* crossing definition used by both the solver and the checks so
    they can never disagree on the count: we count **all** sign changes of ``g = theta - pi/2``
    (an ascending below->above crossing, a descending above->below crossing, and an exact node
    hit each count once). A clean flat-baffle curve has zero crossings; a clean rollback has
    exactly one (it rises through 90 deg once and stays above). More than one means the wall
    wobbles back and forth across the mouth plane -- the pathology the rollback guard rejects.

    Plateaus are collapsed: a run of consecutive samples sitting exactly at ``theta = 90 deg``
    counts as a single crossing (entered once), so a curve that grazes the mouth plane along a
    flat stretch is not over-counted.

    Parameters
    ----------
    theta : tangent angle samples (rad), in meridian order.

    Returns
    -------
    list of ``(i, frac)`` for each crossing, where ``i`` is the sample index just at/before the
    crossing and ``frac in [0, 1]`` is the linear-interpolation weight to the crossing between
    ``i`` and ``i + 1`` (``frac = 0`` for an exact hit at node ``i``). Linearly interpolate any
    field ``f`` at the crossing as ``f[i] + frac * (f[i + 1] - f[i])``.
    """
    theta = np.asarray(theta, dtype=float)
    half_pi = np.pi / 2.0
    g = theta - half_pi
    out: list[tuple[int, float]] = []
    n = g.size
    i = 0
    plateau_reached_end = False
    while i < n - 1:
        a, b = g[i], g[i + 1]
        if a == 0.0:
            # Exact hit at node i. Count once, then skip any plateau of consecutive exact zeros
            # so a flat run on the mouth plane is a single crossing, not one per node.
            out.append((i, 0.0))
            j = i + 1
            while j < n and g[j] == 0.0:
                j += 1
            if j == n:
                # The plateau just counted runs to the final node, so the trailing-node check
                # below must not count that same plateau a second time.
                plateau_reached_end = True
            i = j
            continue
        if (a < 0.0 < b) or (b < 0.0 < a):
            out.append((i, float(a / (a - b))))  # frac in (0, 1)
        i += 1
    # Trailing exact hit at the very last node (not reachable by the loop above), unless that
    # node was already consumed as the tail of a plateau counted above.
    if n and g[-1] == 0.0 and not plateau_reached_end and not (out and out[-1][0] == n - 1):
        out.append((n - 1, 0.0))
    return out


def clamped_uniform_knots(n_coeff: int, degree: int = DEFAULT_DEGREE) -> np.ndarray:
    """Clamped, uniform-interior knot vector on ``[0, 1]`` for ``n_coeff`` coefficients.

    A clamped cubic B-spline with ``n_coeff`` coefficients has ``n_coeff - degree - 1``
    interior knots. ``kappa`` then interpolates its first/last coefficient at the endpoints,
    which is what makes the flat-baffle endpoint conditions ``kappa(1)=0`` (=> last coeff 0)
    and ``dkappa/ds(1)=0`` (=> last two coeffs 0) easy to express.
    """
    if n_coeff < degree + 1:
        raise ValueError(f"need at least degree+1={degree + 1} coefficients, got {n_coeff}")
    n_interior = n_coeff - degree - 1
    interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    return np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])


def kappa_spline(
    coeffs: np.ndarray,
    knots: np.ndarray | None = None,
    degree: int = DEFAULT_DEGREE,
) -> BSpline:
    """Build the curvature B-spline ``kappa(sigma)`` from coefficients (units 1/mm)."""
    coeffs = np.asarray(coeffs, dtype=float)
    if knots is None:
        knots = clamped_uniform_knots(len(coeffs), degree)
    return BSpline(np.asarray(knots, dtype=float), coeffs, degree, extrapolate=True)


@dataclass
class ICWSample:
    """A sampled ICW meridian and the gauges derived from it. Lengths in mm, angles in rad."""

    sigma: np.ndarray  # normalised arc length, 0..1
    s: np.ndarray  # arc length (mm) = S * sigma
    x: np.ndarray  # axial coordinate (mm)
    r: np.ndarray  # radius (mm)
    theta: np.ndarray  # tangent angle from +axial (rad)
    kappa: np.ndarray  # meridional curvature (1/mm)

    # --- geometric gauges (valid everywhere) ---
    @property
    def kappa_meridional(self) -> np.ndarray:
        return self.kappa

    @property
    def kappa_circumferential(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.cos(self.theta) / self.r

    @property
    def flare_rate_Fs(self) -> np.ndarray:
        """Arc-length flare 2 sin(theta)/r. Finite at theta=90 deg; a geometric regulariser,
        NOT a true wavefront-area flare."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return 2.0 * np.sin(self.theta) / self.r

    # --- Webster body gauges (valid only where theta is well below 90 deg) ---
    @property
    def flare_rate_Fx(self) -> np.ndarray:
        """Axial flare 2 tan(theta)/r = d ln S/dx. Diverges as theta->90 deg; body use only."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return 2.0 * np.tan(self.theta) / self.r

    @property
    def webster_Qx(self) -> np.ndarray:
        """Webster/Sturm-Liouville potential Q = r''/r = kappa/(r cos^3 theta).

        The 1/cos^3 theta factor amplifies kappa->Q by ~x8 at 60 deg and ~x191 at 80 deg, so
        only trust this where theta <= theta_body_max (~70-75 deg). Body use only.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.kappa / (self.r * np.cos(self.theta) ** 3)

    def wall_angle_deg(self) -> np.ndarray:
        return np.degrees(self.theta)


@dataclass
class ICWCurve:
    """An intrinsic-curvature waveguide meridian.

    Parameters
    ----------
    coeffs : curvature B-spline coefficients a_i (1/mm), over sigma in [0, 1].
    S : total meridian arc length (mm).
    r0 : throat radius (mm).
    theta0 : throat tangent angle (rad), measured from the +axial direction.
    knots : optional explicit knot vector; defaults to clamped-uniform for len(coeffs).
    degree : B-spline degree (default cubic => kappa C2, theta C3, (x,r) C4).
    x0 : throat axial coordinate (mm), default 0.
    """

    coeffs: np.ndarray
    S: float
    r0: float
    theta0: float
    knots: np.ndarray | None = None
    degree: int = DEFAULT_DEGREE
    x0: float = 0.0

    def __post_init__(self) -> None:
        self.coeffs = np.asarray(self.coeffs, dtype=float)
        if self.knots is None:
            self.knots = clamped_uniform_knots(len(self.coeffs), self.degree)
        else:
            self.knots = np.asarray(self.knots, dtype=float)

    # --- spline accessors -------------------------------------------------------------
    def kappa_spline(self) -> BSpline:
        return kappa_spline(self.coeffs, self.knots, self.degree)

    def kappa(self, sigma: np.ndarray) -> np.ndarray:
        return self.kappa_spline()(np.asarray(sigma, dtype=float))

    def integral_kappa(self, sigma: float = 1.0) -> float:
        """int_0^sigma kappa dv via the spline antiderivative (exact for the spline)."""
        anti = self.kappa_spline().antiderivative()
        return float(anti(sigma) - anti(0.0))

    def total_turn(self) -> float:
        """Total tangent-angle change theta(S) - theta0 = S * int_0^1 kappa dsigma (rad)."""
        return self.S * self.integral_kappa(1.0)

    def theta_end(self) -> float:
        """Terminal tangent angle theta(S) (rad)."""
        return self.theta0 + self.total_turn()

    # --- sampling ---------------------------------------------------------------------
    def sample(self, n: int = DEFAULT_SAMPLES) -> ICWSample:
        """Sample the meridian on ``n`` uniform-sigma stations.

        theta, x and r are obtained by cumulative trapezoidal integration over sigma. The
        trapezoid error is O(h^2) in a smooth kappa, so n>=~1500 reproduces analytic seed
        profiles to well under a micron; pass a larger n for tighter fits.
        """
        if n < 2:
            raise ValueError("n must be >= 2")
        sigma = np.linspace(0.0, 1.0, n)
        kappa = self.kappa(sigma)
        theta = self.theta0 + self.S * cumulative_trapezoid(kappa, sigma, initial=0.0)
        x = self.x0 + self.S * cumulative_trapezoid(np.cos(theta), sigma, initial=0.0)
        r = self.r0 + self.S * cumulative_trapezoid(np.sin(theta), sigma, initial=0.0)
        return ICWSample(sigma=sigma, s=self.S * sigma, x=x, r=r, theta=theta, kappa=kappa)

    # --- convenience constructors -----------------------------------------------------
    @classmethod
    def cone(cls, r0: float, half_angle_deg: float, length_axial: float, n_coeff: int = 6) -> "ICWCurve":
        """A straight cone (kappa == 0) of the given axial length and wall half-angle.

        Useful as the trivial test case and as a default seed.
        """
        theta0 = np.radians(half_angle_deg)
        S = length_axial / np.cos(theta0)
        return cls(coeffs=np.zeros(n_coeff), S=S, r0=r0, theta0=theta0)
