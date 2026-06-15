"""Feasible-subspace constraint solver for the Intrinsic-Curvature Waveguide (ICW).

This module turns *design intent* (throat geometry, terminal-angle / size targets) into a
concrete :class:`~hornlab_mesher.icw.core.ICWCurve` whose curvature B-spline satisfies the
boundary conditions exactly, while leaving as many local shape degrees of freedom as possible
untouched. It builds on the kernel in :mod:`hornlab_mesher.icw.core` and uses only numpy +
scipy (no gmsh).

Design (the "feasible subspace", v3 plan idea)
----------------------------------------------
The meridian curvature is a clamped cubic B-spline ``kappa(sigma) = sum_i a_i B_i(sigma)`` over
normalised arc length ``sigma = s/S in [0, 1]``. The constraints split cleanly:

**Linear in the coefficients** ``a`` (for a *fixed* arc length ``S``)::

    kappa(0)            = kappa0_throat               throat curvature
    kappa(1)            = 0                            flat-baffle: zero curvature at the lip
    dkappa/dsigma(1)    = 0                            flat-baffle: zero curvature *slope* (default on)
    int_0^1 kappa dsig  = (theta1 - theta0) / S        terminal tangent-angle condition

These assemble into a single linear system ``C a = d``. We never "repair" a curve globally;
instead we compute

  * a *particular* least-norm baseline ``a0 = pinv(C) d`` (the smoothest curvature meeting the
    BCs), and
  * an orthonormal basis ``{phi_j}`` of the **nullspace** of ``C`` (via SVD).

Any curvature ``a = a0 + sum_j b_j phi_j`` then satisfies every linear BC *by construction*,
for **all** choices of the shape controls ``b_j``. Editing a ``b_j`` is therefore a strictly
local move that can never disturb the boundary conditions -- the whole point of working in the
feasible subspace rather than projecting/repairing after the fact.

**Nonlinear** (``theta`` sits inside ``cos``/``sin``)::

    flat baffle:   x(1) = x_target ,  r(1) = r_mouth
    rollback:      r(1)=r_end (opt) ,  aperture radius/axial-depth/setback targets

The size targets are solved for ``S`` plus a *small* number of the shape controls ``b_j`` with
:func:`scipy.optimize.least_squares` (bounded, warm-startable). ``S`` enters the linear RHS via
the terminal-angle row, so the linear subspace is re-derived as ``S`` moves; the shape controls
shift the curvature profile -- and hence the integrated meridian endpoint -- without breaking the
BCs.

Units throughout: lengths mm, angles **radians** internally (degrees on the public targets),
curvature 1/mm. Matches the kernel's conventions.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

from .core import (
    DEFAULT_DEGREE,
    MONO_EPS,
    ICWCurve,
    clamped_uniform_knots,
    theta_half_pi_crossings,
)

# --------------------------------------------------------------------------------------------
# Tolerances (explicit, mm / rad). Kept conservative; callers can read them for assertions.
# --------------------------------------------------------------------------------------------
TOL_LENGTH_MM = 1e-3  # mm: endpoint x/r residual we treat as "hit"
TOL_ANGLE_RAD = 1e-4  # rad: terminal-angle residual (~0.006 deg)
TOL_KAPPA_BC = 1e-6  # 1/mm: residual on kappa(1) / dkappa(1) linear BCs
_NULLSPACE_RANK_TOL = 1e-9  # singular-value threshold for the C-nullspace
_QUAD_N = 2001  # sample count for the meridian endpoint quadrature inside the solve loop
_MONO_WEIGHT = 50.0  # soft-barrier weight keeping theta >= 0 (monotone r) in flat-baffle mode
# Tikhonov / curvature-smoothness regulariser on the shape controls ``b``. We now solve over the
# *full* C-nullspace (all columns of Phi) so the local throat-bending DOFs a deep-narrow turn
# needs are available -- the first few SVD modes alone are smooth GLOBAL modes that cannot bend
# the throat, which is why more n_coeff used to make hard cases *worse*. With the full nullspace
# the size system is under-determined, so this penalty regularises it: ``sqrt(lambda)*b`` keeps
# the solution close to the least-norm (smoothest) baseline ``a0``, i.e. local and well behaved.
# It is appended to the *optimiser* residual only -- never to the size residual feasibility is
# judged on -- and every ``a = a0 + Phi @ b`` already satisfies the linear BCs exactly, so the
# regulariser cannot disturb kappa(0)/kappa(1)/dkappa(1)/the integral condition. lambda is small
# (tuned so easy cases stay exact and the deep-narrow basin becomes feasible).
_REG_LAMBDA = 1e-4  # 1/mm^2 weight on ||b||^2 (curvature deviation from the smooth baseline)


# --------------------------------------------------------------------------------------------
# Public enums / targets / reports
# --------------------------------------------------------------------------------------------
class TerminationMode:
    """String constants for the two supported termination styles."""

    FLAT_BAFFLE = "flat_baffle"  # theta(1)=90deg, kappa(1)=0 -> meets an infinite flat baffle
    ROLLBACK = "rollback"  # theta(1)>90deg -> free-standing curled lip


@dataclass
class ICWTargets:
    """Design intent for one ICW meridian.

    Parameters
    ----------
    mode :
        One of :class:`TerminationMode` (``"flat_baffle"`` or ``"rollback"``).
    r0 :
        Throat radius (mm).
    theta0_deg :
        Throat wall angle from the +axial direction (deg).
    kappa0 :
        Throat curvature (1/mm). Defaults to 0 (a locally-conical throat).
    theta1_deg :
        Terminal wall angle (deg). For ``flat_baffle`` this is forced to 90. For ``rollback``
        it defaults to a value derived from ``rollback_theta_max_deg`` if not given.
    enforce_dkappa_end :
        Flat-baffle only: also force ``dkappa/dsigma(1)=0`` (default on) so the curvature lands
        flat onto the baffle, not just at zero.

    Size targets (flat baffle)
    --------------------------
    x_target :
        Axial depth of the mouth ``x(1)`` (mm).
    r_mouth :
        Mouth radius ``r(1)`` (mm).

    Size targets (rollback)
    -----------------------
    r_aperture :
        Radius at which the wall first reaches ``theta = 90 deg`` (the acoustic aperture, mm).
    x_aperture :
        Axial location of that 90-deg crossing (mm).
    depth :
        Total axial depth ``x(1)`` of the curled tip (mm); alternative to ``x_aperture``.
    x_setback :
        Axial setback of the tip behind the aperture, ``x_aperture - x(1)`` (mm). ``>= 0`` for
        a rollback whose rim sits behind the aperture; matches :func:`checks.aperture_report`'s
        ``x_setback`` so a solved value round-trips with the same sign and magnitude.
    r_end :
        Optional terminal radius ``r(1)`` (mm).
    rollback_theta_max_deg :
        Terminal angle used for ``rollback`` when ``theta1_deg`` is not supplied (deg).
    """

    mode: str
    r0: float
    theta0_deg: float
    kappa0: float = 0.0
    theta1_deg: float | None = None
    enforce_dkappa_end: bool = True

    # flat-baffle size targets
    x_target: float | None = None
    r_mouth: float | None = None

    # rollback size targets
    r_aperture: float | None = None
    x_aperture: float | None = None
    depth: float | None = None
    x_setback: float | None = None
    r_end: float | None = None
    rollback_theta_max_deg: float = 160.0

    def __post_init__(self) -> None:
        if self.mode not in (TerminationMode.FLAT_BAFFLE, TerminationMode.ROLLBACK):
            raise ValueError(
                f"mode must be {TerminationMode.FLAT_BAFFLE!r} or {TerminationMode.ROLLBACK!r}, "
                f"got {self.mode!r}"
            )
        if self.mode == TerminationMode.FLAT_BAFFLE:
            # Flat baffle is *defined* by a 90-deg terminal angle; coerce regardless of input.
            self.theta1_deg = 90.0
        elif self.theta1_deg is None:
            self.theta1_deg = self.rollback_theta_max_deg

    # --- derived radian helpers ---
    @property
    def theta0(self) -> float:
        return np.radians(self.theta0_deg)

    @property
    def theta1(self) -> float:
        return np.radians(self.theta1_deg)


@dataclass
class FeasibilityReport:
    """Outcome of a solve: feasibility plus residuals and (if infeasible) a relaxation hint."""

    feasible: bool
    violations: list[str] = field(default_factory=list)
    residuals: dict = field(default_factory=dict)
    suggested_relaxation: dict | None = None
    message: str = ""


# --------------------------------------------------------------------------------------------
# Linear constraint machinery (the feasible subspace)
# --------------------------------------------------------------------------------------------
def _basis_matrix(knots: np.ndarray, degree: int, n_coeff: int, x, nu: int = 0) -> np.ndarray:
    """Row of basis-function values (or ``nu``-th derivatives) at the points ``x``.

    Evaluating each basis function ``B_i`` via a unit-coefficient spline keeps the *nonuniform*
    end-knot spacing baked in -- so e.g. ``dkappa/dsigma(1)`` is expressed correctly through the
    basis (we never hardcode the ``27`` factor that the clamped-uniform knots happen to produce).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    M = np.zeros((x.size, n_coeff))
    for i in range(n_coeff):
        c = np.zeros(n_coeff)
        c[i] = 1.0
        M[:, i] = BSpline(knots, c, degree, extrapolate=False)(x, nu=nu)
    return M


def _basis_integrals(knots: np.ndarray, degree: int, n_coeff: int) -> np.ndarray:
    """``int_0^1 B_i dsigma`` for every basis function (exact, via the antiderivative)."""
    out = np.zeros(n_coeff)
    for i in range(n_coeff):
        c = np.zeros(n_coeff)
        c[i] = 1.0
        anti = BSpline(knots, c, degree, extrapolate=False).antiderivative()
        out[i] = float(anti(1.0) - anti(0.0))
    return out


def build_linear_constraints(
    targets: ICWTargets,
    S: float,
    knots: np.ndarray,
    n_coeff: int,
    degree: int = DEFAULT_DEGREE,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the linear constraint system ``C a = d`` on the curvature coefficients.

    Rows, in order:

    1. ``kappa(0) = kappa0``                              (always)
    2. ``kappa(1) = 0``                                   (flat baffle only)
    3. ``dkappa/dsigma(1) = 0``                           (flat baffle, if ``enforce_dkappa_end``)
    4. ``int_0^1 kappa dsigma = (theta1 - theta0) / S``   (always; terminal-angle condition)

    Note that row 4 depends on ``S`` -- the linear subspace is re-derived as the nonlinear solver
    moves ``S``.
    """
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    # (1) throat curvature
    rows.append(_basis_matrix(knots, degree, n_coeff, 0.0, nu=0)[0])
    rhs.append(float(targets.kappa0))

    if targets.mode == TerminationMode.FLAT_BAFFLE:
        # (2) zero curvature at the lip
        rows.append(_basis_matrix(knots, degree, n_coeff, 1.0, nu=0)[0])
        rhs.append(0.0)
        # (3) zero curvature slope at the lip
        if targets.enforce_dkappa_end:
            rows.append(_basis_matrix(knots, degree, n_coeff, 1.0, nu=1)[0])
            rhs.append(0.0)

    # (4) terminal-angle condition: integral of kappa fixes the total turn
    rows.append(_basis_integrals(knots, degree, n_coeff))
    rhs.append((targets.theta1 - targets.theta0) / S)

    return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)


def feasible_subspace(C: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(a0, Phi)``: a least-norm particular solution and a nullspace basis of ``C``.

    ``a0 = pinv(C) d`` is the smoothest (minimum-2-norm) curvature meeting the linear BCs.
    ``Phi`` has the nullspace vectors as columns; ``a = a0 + Phi @ b`` satisfies ``C a = d`` for
    every shape vector ``b``.
    """
    a0 = np.linalg.pinv(C) @ d
    # Nullspace via SVD of C (rows = constraints, cols = coeffs).
    _, s, Vt = np.linalg.svd(C)
    rank = int(np.sum(s > _NULLSPACE_RANK_TOL * max(1.0, s[0]))) if s.size else 0
    Phi = Vt[rank:].T.copy()  # columns span ker(C)
    return a0, Phi


@dataclass
class _LinSys:
    """Per-solve cache of the S/b-INDEPENDENT linear-constraint algebra.

    The constraint matrix ``C`` (curvature-coeffs -> [kappa(0), kappa(1), dkappa/dsigma(1),
    int kappa]) is built purely from the *basis* (``knots``/``degree``/``n_coeff``) and the
    constraint *structure* (which rows the mode uses); it does not depend on the optimiser
    variables ``b`` or the arc length ``S``, nor on the terminal angle ``theta1`` (which enters
    only the integral RHS). Therefore ``C``, ``pinv(C)`` and the nullspace basis ``Phi`` are
    invariant across the whole nonlinear solve (and across the continuation theta1-ramp), and we
    precompute them ONCE per solve instead of rebuilding them on every one of the ~3000+ residual
    evaluations.

    Only the RHS ``d`` moves, and only in two cheap scalars: the throat-curvature row
    (``kappa0``, constant within a solve) and the integral row ``(theta1 - theta0)/S``. ``a0``
    is then ``pinv_C @ d`` -- a single matvec. The fine sampling grid ``sigma`` and the per-basis
    matrices are also S-independent and cached here.

    ``rhs_const`` holds the fixed RHS entries (everything except the integral row);
    ``integral_row`` is the index of the ``int kappa`` row whose RHS is ``(theta1 - theta0)/S``.
    """

    C: np.ndarray
    pinv_C: np.ndarray
    Phi: np.ndarray
    rhs_const: np.ndarray  # RHS with the integral row left at 0.0 (filled per-S)
    integral_row: int
    sigma: np.ndarray  # fine quadrature grid (S-independent)

    @property
    def n_shape(self) -> int:
        return self.Phi.shape[1]

    def a0(self, theta0: float, theta1: float, S: float) -> np.ndarray:
        """Least-norm baseline ``a0 = pinv(C) @ d(S)`` for the current ``theta1``/``S``.

        Reproduces ``feasible_subspace(build_linear_constraints(...))[0]`` exactly: the only
        S/theta1-dependent RHS entry is the integral row ``(theta1 - theta0)/S``.
        """
        d = self.rhs_const.copy()
        d[self.integral_row] = (theta1 - theta0) / S
        return self.pinv_C @ d


def _build_linsys(
    targets: ICWTargets,
    knots: np.ndarray,
    n_coeff: int,
    degree: int,
    n_quad: int = _QUAD_N,
) -> _LinSys:
    """Precompute the S/b-independent constraint algebra (matrix, pinv, nullspace, grid).

    Builds ``C`` and ``d`` once (at a placeholder ``S=1``), derives ``pinv(C)`` and the
    nullspace ``Phi``, and records the constant RHS plus the integral-row index so the residual
    can refill the single S-dependent entry. The result is mathematically identical to calling
    :func:`build_linear_constraints` + :func:`feasible_subspace` inside every residual, just
    hoisted out of the hot loop (P1-2: no change to the math/result).
    """
    C, d = build_linear_constraints(targets, 1.0, knots, n_coeff, degree)
    pinv_C = np.linalg.pinv(C)
    # Nullspace basis via SVD (S-independent, since C is S-independent).
    _, s, Vt = np.linalg.svd(C)
    rank = int(np.sum(s > _NULLSPACE_RANK_TOL * max(1.0, s[0]))) if s.size else 0
    Phi = Vt[rank:].T.copy()
    # The integral row is always the LAST row assembled by build_linear_constraints.
    integral_row = C.shape[0] - 1
    rhs_const = d.copy()
    rhs_const[integral_row] = 0.0  # refilled per-S in _LinSys.a0
    sigma = np.linspace(0.0, 1.0, n_quad)
    return _LinSys(
        C=C, pinv_C=pinv_C, Phi=Phi, rhs_const=rhs_const, integral_row=integral_row, sigma=sigma
    )


# --------------------------------------------------------------------------------------------
# Meridian endpoint quadrature (used inside the nonlinear residual)
# --------------------------------------------------------------------------------------------
def _integrate_meridian(
    coeffs: np.ndarray,
    knots: np.ndarray,
    degree: int,
    S: float,
    r0: float,
    theta0: float,
    n: int = _QUAD_N,
    sigma: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cheap (sigma, theta, x, r) integration matching ICWCurve.sample for the residual loop.

    ``sigma`` (the fine quadrature grid) is S-independent; callers in the hot loop pass the
    cached grid to avoid re-allocating ``linspace(0, 1, n)`` every residual evaluation.
    """
    if sigma is None:
        sigma = np.linspace(0.0, 1.0, n)
    kappa = BSpline(knots, coeffs, degree, extrapolate=True)(sigma)
    theta = theta0 + S * cumulative_trapezoid(kappa, sigma, initial=0.0)
    x = S * cumulative_trapezoid(np.cos(theta), sigma, initial=0.0)
    r = r0 + S * cumulative_trapezoid(np.sin(theta), sigma, initial=0.0)
    return sigma, theta, x, r


def _theta90_crossings(theta: np.ndarray) -> list[tuple[int, float]]:
    """All ``theta = 90 deg`` crossings, via the shared :func:`core.theta_half_pi_crossings`.

    Uses the ONE crossing definition shared with :func:`checks.aperture_report` -- every sign
    change of ``theta - pi/2`` (ascending, descending, or exact node), plateaus collapsed -- so
    the solver's count and the aperture report's ``n_pi2_crossings`` can never disagree on the
    same curve. A clean rollback has exactly one; the rollback guard in :func:`_verify` requires
    that. Returns the ``(index, frac)`` crossing list (see the core helper for the convention).
    """
    return theta_half_pi_crossings(theta)


def _aperture_crossing(
    crossings: list[tuple[int, float]], x: np.ndarray
) -> tuple[int, float] | None:
    """The acoustic-aperture crossing: the one of maximal ``x`` (the foremost mouth plane).

    Matches :func:`checks.aperture_report`, which also picks the foremost crossing, so a solved
    aperture round-trips to the same station the report reads back.
    """
    if not crossings:
        return None
    x_at = [x[i] + frac * (x[i + 1] - x[i]) if i + 1 < x.size else x[i] for i, frac in crossings]
    return crossings[int(np.argmax(x_at))]


# --------------------------------------------------------------------------------------------
# The solve
# --------------------------------------------------------------------------------------------
def _chord(targets: ICWTargets) -> float:
    """Lower bound on arc length: the straight-line distance from throat to the relevant target.

    Arc length must be >= chord (with equality only for a straight wall and no rollback). Used
    for the early necessary-feasibility check and to seed ``S``.
    """
    if targets.mode == TerminationMode.FLAT_BAFFLE:
        dx = targets.x_target if targets.x_target is not None else 0.0
        rm = targets.r_mouth if targets.r_mouth is not None else targets.r0
        return float(np.hypot(dx, rm - targets.r0))
    # rollback: use the best available end/aperture target as a proxy chord
    rx = targets.r_end if targets.r_end is not None else targets.r_aperture
    rx = rx if rx is not None else targets.r0
    dx = targets.depth
    if dx is None and targets.x_aperture is not None:
        dx = targets.x_aperture
    if dx is None:
        dx = 0.0
    return float(np.hypot(dx, rx - targets.r0))


def _seed_state(
    targets: ICWTargets,
    Phi: np.ndarray,
    seed: ICWCurve | None,
    n_shape: int,
) -> tuple[float, np.ndarray]:
    """Initial guess for ``(S, b[:n_shape])``."""
    chord = _chord(targets)
    if seed is not None:
        S_seed = float(seed.S)
    else:
        # Start a little above the chord; a turning wall is always longer than the chord.
        S_seed = max(chord * 1.08, chord + 1.0, 1.0)
    return S_seed, np.zeros(n_shape)


def _early_feasibility(targets: ICWTargets) -> list[str]:
    """Necessary (not sufficient) pre-checks that don't require a solve."""
    v: list[str] = []
    if targets.r0 <= 0:
        v.append(f"throat radius r0={targets.r0} must be > 0")
    if targets.mode == TerminationMode.FLAT_BAFFLE:
        if targets.x_target is None or targets.r_mouth is None:
            v.append("flat_baffle requires both x_target and r_mouth")
        else:
            if targets.r_mouth <= targets.r0:
                v.append(
                    f"r_mouth={targets.r_mouth} must exceed r0={targets.r0} (mouth wider than throat)"
                )
            if targets.x_target <= 0:
                v.append(f"x_target={targets.x_target} must be > 0")
    else:  # rollback
        have_axial = any(t is not None for t in (targets.depth, targets.x_aperture, targets.x_setback))
        have_radial = any(t is not None for t in (targets.r_aperture, targets.r_end))
        if not have_axial:
            v.append("rollback requires one axial target (depth, x_aperture, or x_setback)")
        if not have_radial:
            v.append("rollback requires one radial target (r_aperture or r_end)")
        if targets.theta1_deg is not None and targets.theta1_deg <= 90.0:
            v.append(
                f"rollback needs theta1>90deg, got theta1_deg={targets.theta1_deg}"
            )
    return v


def _residual_and_aux(
    p: np.ndarray,
    targets: ICWTargets,
    knots: np.ndarray,
    n_coeff: int,
    degree: int,
    lin: "_LinSys",
    n_shape: int,
):
    """Compute the nonlinear residual for parameter vector ``p = [S, b0..b_{n_shape-1}]``.

    Returns ``(full_residual, size_residual, coeffs, S, sample_tuple)``. ``full_residual`` is what
    the optimiser minimises (size targets + soft geometric barriers); ``size_residual`` holds
    *only* the hard size-target misses, which is what feasibility is judged against -- the barrier
    terms must not be mistaken for unmet size targets. The linear BCs are re-imposed inside (the
    subspace baseline ``a0`` depends on ``S`` through the terminal-angle RHS), so the returned
    coeffs always satisfy ``C a = d`` to machine precision.

    P1-2: the S/b-independent constraint algebra (``C``, ``pinv(C)``, the nullspace ``Phi`` and
    the fine sampling grid) is precomputed once per solve in ``lin`` (:class:`_LinSys`) and reused
    here; only the cheap S-dependent parts are recomputed -- ``a0 = pinv(C) @ d(S)`` (one matvec)
    and the cos/sin quadrature. The result is identical to rebuilding them every call.
    """
    S = float(p[0])
    b = p[1 : 1 + n_shape]

    # Only the RHS (hence a0) depends on S/theta1; C, pinv(C) and Phi are precomputed in ``lin``.
    a0 = lin.a0(targets.theta0, targets.theta1, S)
    Phi = lin.Phi
    coeffs = a0 + Phi[:, :n_shape] @ b if n_shape else a0

    sigma, theta, x, r = _integrate_meridian(
        coeffs, knots, degree, S, targets.r0, targets.theta0, sigma=lin.sigma
    )

    size: list[float] = []  # hard size-target residuals (judged for feasibility)
    barrier: list[float] = []  # soft geometric barriers (guide the solver only)

    if targets.mode == TerminationMode.FLAT_BAFFLE:
        size.append(x[-1] - targets.x_target)  # x(1) = x_target
        size.append(r[-1] - targets.r_mouth)  # r(1) = r_mouth
        # Soft barrier: penalise theta dipping below 0 (-> r decreasing). The least-norm baseline
        # tends to swing theta negative mid-body; this nudges the extra shape DOFs toward a
        # single-valued graph (strictly increasing r) without disturbing the size targets.
        neg = np.clip(-theta, 0.0, None)
        barrier.append(_MONO_WEIGHT * float(np.sqrt(np.mean(neg**2))))
    else:
        # rollback aperture / depth / setback / end targets
        crossings = _theta90_crossings(theta)
        ap = _aperture_crossing(crossings, x)  # foremost (max-x) crossing, as the report uses
        x_ap = r_ap = None
        if ap is not None:
            j, w = ap  # linear interp weight of x,r at theta=pi/2 (shared crossing convention)
            # Guard the forward node: a trailing theta=pi/2 crossing reports j = n-1 (with w=0),
            # so x[j+1]/r[j+1] would index out of bounds and crash the least_squares residual.
            # Mirror the i+1 < size guard used in aperture_report / _aperture_crossing / _verify.
            x_ap = x[j] + w * (x[j + 1] - x[j]) if j + 1 < x.size else float(x[j])
            r_ap = r[j] + w * (r[j + 1] - r[j]) if j + 1 < r.size else float(r[j])

        if targets.r_aperture is not None:
            size.append((r_ap - targets.r_aperture) if r_ap is not None else (r[-1] - targets.r_aperture))
        if targets.x_aperture is not None:
            size.append((x_ap - targets.x_aperture) if x_ap is not None else (x[-1] - targets.x_aperture))
        if targets.depth is not None:
            size.append(x[-1] - targets.depth)
        if targets.x_setback is not None:
            # x_setback = x_aperture - x_end (>= 0 when the rim tip recedes behind the
            # aperture). Same convention as checks.aperture_report so a solved x_setback=D
            # round-trips to aperture_report(sample).x_setback ~= D (matching sign & value).
            base_x = x_ap if x_ap is not None else x[-1]
            size.append((base_x - x[-1]) - targets.x_setback)
        if targets.r_end is not None:
            size.append(r[-1] - targets.r_end)

        # Soft barriers for the rollback geometric invariants the report enforces:
        #   (i) strictly increasing r -> theta in (0, 180) throughout: penalise theta < 0 / > 180,
        #   (ii) exactly one ascending theta=90 crossing -> penalise any *descent* in theta, which
        #        is what produces the multiple-crossing / non-monotone pathologies.
        out_lo = np.clip(-theta, 0.0, None)
        out_hi = np.clip(theta - np.pi, 0.0, None)
        barrier.append(_MONO_WEIGHT * float(np.sqrt(np.mean(out_lo**2 + out_hi**2))))
        dip = np.clip(-np.diff(theta), 0.0, None)
        barrier.append(_MONO_WEIGHT * float(np.sqrt(np.mean(dip**2))))

    # Tikhonov / curvature-smoothness regulariser on the shape controls. We solve over the FULL
    # nullspace (see ``_REG_LAMBDA``); this penalty keeps that under-determined system well posed
    # and the solution close to the smooth least-norm baseline ``a0`` -- i.e. the throat-bending
    # stays LOCAL rather than ringing. It is part of the optimiser residual ONLY (``full``); it is
    # never added to ``size_arr``, so feasibility stays judged purely on the hard size misses and
    # the linear-BC property (any ``a0 + Phi @ b`` satisfies ``C a = d``) is untouched.
    reg = np.sqrt(_REG_LAMBDA) * b if n_shape else np.zeros(0)

    size_arr = np.asarray(size, dtype=float)
    full = np.concatenate(
        [size_arr, np.asarray(barrier, dtype=float), np.asarray(reg, dtype=float)]
    )
    return full, size_arr, coeffs, S, (sigma, theta, x, r)


# NOTE: the former ``_count_shape_dofs`` helper (which sized the shape-control subspace to a small
# fixed count) was removed with the P0-1 fix. Using only the first few SVD modes is exactly what
# made deep-narrow turns infeasible at higher ``n_coeff`` -- those leading modes are smooth GLOBAL
# shapes that cannot bend the throat. ``solve_icw`` now solves over the FULL C-nullspace (all
# columns of ``Phi``), regularised by the Tikhonov term in ``_residual_and_aux``.


def solve_icw(
    targets: ICWTargets,
    n_coeff: int = 12,
    seed: "ICWCurve | None" = None,
    max_nfev: int = 400,
) -> tuple["ICWCurve", FeasibilityReport]:
    """Solve for an :class:`ICWCurve` meeting ``targets`` in the feasible subspace.

    Strategy
    --------
    1. Early necessary-feasibility checks (chord <= S, target sanity).
    2. Build the linear BC system ``C a = d`` and its feasible subspace ``a0 + Phi b``.
    3. Solve the nonlinear size constraints for ``[S, b]`` with bounded least-squares,
       warm-started from ``seed`` if given (continuation-friendly).
    4. Re-check feasibility; if the residual cannot be driven under tolerance, return an
       infeasible report with violations and a suggested relaxation -- never a silent repair.

    For ``ROLLBACK``, a feasible result additionally satisfies ``theta_end > 90 deg`` with
    exactly one ascending ``theta = 90 deg`` crossing and a strictly increasing radius.
    """
    degree = DEFAULT_DEGREE
    knots = clamped_uniform_knots(n_coeff, degree)

    # ---- (1) early necessary feasibility -------------------------------------------------
    violations = _early_feasibility(targets)
    if violations:
        return _infeasible_curve(targets, knots, n_coeff, degree), FeasibilityReport(
            feasible=False,
            violations=violations,
            residuals={},
            suggested_relaxation=_relaxation_hint(targets, violations),
            message="Target set is malformed or violates necessary conditions before solving.",
        )

    # Solve over the FULL C-nullspace. The constraint rank is S-independent, so the basis Phi
    # built at S=1 spans ker(C) for every S; using *all* its columns gives the solver the local
    # throat-bending DOFs a deep-narrow turn needs (the first few SVD modes are smooth global
    # shapes that cannot -- which is why raising n_coeff used to make hard cases strictly worse).
    # The full system is then under-determined in ``b``; the Tikhonov term in ``_residual_and_aux``
    # (_REG_LAMBDA) keeps it well posed and the solution local, without touching the linear BCs.
    #
    # P1-2: C, pinv(C), Phi and the fine sampling grid are S/b-independent (and invariant across
    # the continuation theta1-ramp), so we precompute them ONCE here into ``lin`` and reuse them in
    # every residual evaluation rather than rebuilding them ~3000+ times per solve.
    lin = _build_linsys(targets, knots, n_coeff, degree)
    Phi = lin.Phi
    n_shape = lin.n_shape
    _a0 = lin.a0(targets.theta0, targets.theta1, 1.0)  # baseline at the S=1 RHS (warm-start proj)

    # ---- chord check ---------------------------------------------------------------------
    chord = _chord(targets)
    S_seed, b_seed = _seed_state(targets, Phi, seed, n_shape)
    if seed is not None and seed.coeffs.shape == (n_coeff,):
        # Project the seed's curvature onto the shape subspace for a warm start.
        b_seed = (Phi[:, :n_shape].T @ (seed.coeffs - _a0)) if n_shape else b_seed

    # ---- bounds: S strictly above the chord; shape controls bounded to keep curvature sane.
    S_lo = max(chord * (1.0 + 1e-6), 1e-6)
    S_hi = max(chord * 50.0, S_seed * 10.0, 10.0)
    kappa_scale = 5.0 / max(targets.r0, 1.0)  # generous curvature magnitude bound (1/mm)
    lo = np.concatenate([[S_lo], np.full(n_shape, -kappa_scale)])
    hi = np.concatenate([[S_hi], np.full(n_shape, kappa_scale)])

    def _run(target_set: ICWTargets, p_init: np.ndarray, nfev: int) -> np.ndarray:
        """One bounded least-squares solve; returns the solution parameter vector."""

        def fun(p: np.ndarray) -> np.ndarray:
            res, *_ = _residual_and_aux(p, target_set, knots, n_coeff, degree, lin, n_shape)
            return res

        p_init = np.clip(p_init, lo + 1e-12, hi - 1e-12)
        sol = least_squares(
            fun, p_init, bounds=(lo, hi), max_nfev=nfev, xtol=1e-14, ftol=1e-14, gtol=1e-14
        )
        return sol.x

    # ---- (3) nonlinear solve -------------------------------------------------------------
    # Direct attempt first, then -- for hard cases that miss tolerance or break a geometric
    # invariant -- a CONTINUATION sweep: ramp the terminal angle from 90deg up to the target,
    # warm-starting each step. This keeps rollback solves in the single-crossing, monotone-r
    # basin where the barriers do not fight the size targets (v3 plan: homotopy on theta1).
    p0 = np.concatenate([[S_seed], b_seed])

    def _better(report_a: FeasibilityReport, report_b: FeasibilityReport) -> bool:
        """True iff ``report_a`` is a strictly better outcome than ``report_b``.

        Feasibility wins outright; among equally (in)feasible results, fewer violations then a
        smaller hard size residual. This lets the fallbacks accept only genuine improvements and
        keeps infeasibility TRUTHFUL -- a result is reported feasible only if ``_verify`` says so.
        """
        if report_a.feasible != report_b.feasible:
            return report_a.feasible
        if len(report_a.violations) != len(report_b.violations):
            return len(report_a.violations) < len(report_b.violations)
        return report_a.residuals.get("size_residual_max", np.inf) < report_b.residuals.get(
            "size_residual_max", np.inf
        )

    p_best = _run(targets, p0, max_nfev)
    curve_best, report_best = _assemble(targets, knots, n_coeff, degree, lin, n_shape, p_best)

    # (3a) Continuation rescue (homotopy on the terminal angle) for hard / rollback targets.
    if not report_best.feasible:
        p_cont = _continuation_solve(
            targets, knots, n_coeff, degree, lin, n_shape, _run, p0, max_nfev
        )
        if p_cont is not None:
            curve_c, report_c = _assemble(targets, knots, n_coeff, degree, lin, n_shape, p_cont)
            if _better(report_c, report_best):
                curve_best, report_best = curve_c, report_c

    # (3b) Multistart rescue: deep-narrow basins sometimes hit the size targets but leave r just
    # non-monotone from the single chord-seeded start. Retry from a few diversified seeds (larger
    # S, small curvature perturbations) BEFORE conceding infeasibility. We accept a retry only if
    # it is strictly better (``_better``), so genuinely impossible targets still report
    # feasible=False with their violations -- never a silent repair into a wrong curve.
    if not report_best.feasible:
        rng = np.random.default_rng(0)
        kappa_scale = 5.0 / max(targets.r0, 1.0)
        seeds: list[np.ndarray] = []
        for sf in (1.2, 1.5, 2.0, 3.0):
            seeds.append(np.concatenate([[max(chord * sf, S_lo * 1.001)], np.zeros(n_shape)]))
        for _ in range(4):
            b_pert = rng.normal(scale=0.05 * kappa_scale, size=n_shape) if n_shape else np.zeros(0)
            seeds.append(np.concatenate([[S_seed], b_pert]))
        for sd in seeds:
            try:
                p_try = _run(targets, sd, max_nfev)
            except Exception:
                continue
            curve_t, report_t = _assemble(targets, knots, n_coeff, degree, lin, n_shape, p_try)
            if _better(report_t, report_best):
                curve_best, report_best = curve_t, report_t
            if report_best.feasible:
                break

    return curve_best, report_best


def _assemble(
    targets: ICWTargets,
    knots: np.ndarray,
    n_coeff: int,
    degree: int,
    lin: "_LinSys",
    n_shape: int,
    p: np.ndarray,
) -> tuple[ICWCurve, FeasibilityReport]:
    """Build the ICWCurve + feasibility report for a solved parameter vector ``p``."""
    _full, size_res, coeffs, S, (sigma, theta, x, r) = _residual_and_aux(
        p, targets, knots, n_coeff, degree, lin, n_shape
    )
    curve = ICWCurve(
        coeffs=coeffs, S=S, r0=targets.r0, theta0=targets.theta0, knots=knots, degree=degree
    )
    return curve, _verify(targets, curve, size_res, sigma, theta, x, r)


def _continuation_solve(
    targets: ICWTargets,
    knots: np.ndarray,
    n_coeff: int,
    degree: int,
    lin: "_LinSys",
    n_shape: int,
    run_fn,
    p0: np.ndarray,
    max_nfev: int,
    n_steps: int = 6,
) -> np.ndarray | None:
    """Homotopy on the terminal angle: solve a sequence theta1 = 90 -> target, warm-started.

    Returns the final parameter vector, or ``None`` if the sweep cannot be initialised. Each
    intermediate solve uses a fraction of ``max_nfev``; the last step gets the remainder so the
    final tolerance is tight. ``lin``/``n_shape`` are accepted for call-site symmetry; the actual
    residual algebra is reached via ``run_fn`` (which closes over the precomputed ``_LinSys``).
    """
    if targets.theta1_deg is None:
        return None
    start = 90.0 if targets.mode == TerminationMode.ROLLBACK else targets.theta0_deg
    # If start already equals the target there is nothing to ramp.
    if abs(targets.theta1_deg - start) < 1e-9:
        return None

    p = p0.copy()
    fracs = np.linspace(start, targets.theta1_deg, n_steps + 1)[1:]
    per_step = max(40, max_nfev // n_steps)
    last = None

    for k, th1 in enumerate(fracs):
        sub = dataclasses.replace(targets, theta1_deg=float(th1))
        nfev = max_nfev if k == len(fracs) - 1 else per_step
        try:
            p = run_fn(sub, p, nfev)
        except Exception:
            return last
        last = p.copy()
    return last


# --------------------------------------------------------------------------------------------
# Shape-mode entry point (outer CMA-ES genes SET curvature modes; kernel solves the size DOFs)
# --------------------------------------------------------------------------------------------
# Rationale for the gene / reserved partition (k_max formula)
# ----------------------------------------------------------
# Every ``a = a0(S) + Phi @ b`` satisfies the linear BCs (kappa(0), and for a flat baffle
# kappa(1)=0, dkappa/dsigma(1)=0, plus the terminal-angle integral row) *by construction*, for
# ANY choice of the nullspace coordinates ``b`` -- that is the whole point of working in the
# feasible subspace. So an outer optimiser is free to dial the leading nullspace coordinates as
# *genes* and the kernel will still land the curve on the baffle exactly, regardless of the gene
# values; the only thing the kernel must still do is meet the NONLINEAR *size* targets.
#
# A flat baffle has two nonlinear size targets -- x(1)=x_target and r(1)=r_mouth. The kernel has
# the arc length ``S`` as one free variable (it enters the terminal-angle RHS, hence the whole
# subspace). Two targets with one free var is under-determined / ill-posed on its own, so we
# reserve at least one nullspace coordinate *in addition* to S for the size solve. That gives a
# 2-DOF (S + 1 reserved mode) solve for the 2 size targets -- well-posed -- and leaves the
# remaining nullspace coordinates free to be genes:
#
#     k_max = D - 1            (D = dim ker(C); reserve 1 nullspace mode beyond S)
#     k_max = max(k_max, 0)    (never negative)
#
# With the default basis (n_coeff=12, cubic, flat baffle) C is 4x12 of rank 4, so D = 8 and
# k_max = 7. The reserved coordinates the kernel optimises are the LAST ``D - k`` columns of
# ``Phi`` (indices ``k .. D``); the genes are the FIRST ``k`` columns (indices ``0 .. k``). The
# leading SVD nullspace modes are the smoothest GLOBAL curvature shapes, so handing those to the
# outer optimiser as genes gives it the broad shape control it wants, while the kernel keeps a
# higher-frequency reserved mode (plus S) to fine-tune the endpoint -- a clean separation of
# "shape" (genes) from "size-fixing" (reserved).


def _flat_baffle_lin(
    targets: ICWTargets, n_coeff: int, degree: int
) -> tuple[np.ndarray, "_LinSys"]:
    """Shared setup for the shape-mode entry points: knots + precomputed ``_LinSys``.

    Enforces the flat-baffle-only contract (rollback is out of scope for the 2a entry point) and
    returns the same ``(knots, _LinSys)`` ``solve_icw`` builds, so the nullspace basis ``Phi`` and
    the constraint algebra are *identical* to the all-DOF solver. Raises ``ValueError`` for a
    non-flat-baffle target (truthful, no silent coercion).
    """
    if targets.mode != TerminationMode.FLAT_BAFFLE:
        raise ValueError(
            "shape-mode entry points support only mode='flat_baffle' (rollback is out of scope "
            f"here); got mode={targets.mode!r}"
        )
    knots = clamped_uniform_knots(n_coeff, degree)
    lin = _build_linsys(targets, knots, n_coeff, degree)
    return knots, lin


def n_shape_modes(targets: ICWTargets, n_coeff: int = 12, degree: int = DEFAULT_DEGREE) -> int:
    """Number of nullspace modes available as **genes** for an outer optimiser.

    The feasible subspace is ``kappa = kappa0(S) + sum_j b_j Phi[:, j]`` where ``Phi`` spans the
    nullspace (dimension ``D``) of the linear-BC constraint matrix ``C``. An outer optimiser
    (e.g. CMA-ES) may SET the first ``k`` of these coordinates as genes; the kernel then solves
    the arc length ``S`` plus the remaining ``D - k`` "reserved" coordinates to hit the nonlinear
    size targets. To keep that size solve well-posed we reserve at least one nullspace coordinate
    beyond ``S`` (two free vars for the two flat-baffle size targets ``x_target``/``r_mouth``):

        k_max = max(D - 1, 0)

    so the returned value is the largest gene-vector length :func:`curve_from_shape_modes` accepts.
    See the module-level rationale above the entry points for the full reasoning.

    Parameters
    ----------
    targets : design intent; must be ``mode='flat_baffle'`` (raises otherwise).
    n_coeff, degree : the curvature B-spline basis (same defaults as :func:`solve_icw`).

    Returns
    -------
    ``k_max`` (int): how many leading ``Phi`` columns are exposed as genes. ``>= 0``.
    """
    _knots, lin = _flat_baffle_lin(targets, n_coeff, degree)
    D = lin.n_shape  # nullspace dimension dim ker(C)
    return max(D - 1, 0)


def curve_from_shape_modes(
    b_gene,
    targets: ICWTargets,
    n_coeff: int = 12,
    degree: int = DEFAULT_DEGREE,
    max_nfev: int = 200,
) -> tuple["ICWCurve", FeasibilityReport]:
    """Solve an ICW curve with the leading nullspace coordinates HELD FIXED to ``b_gene``.

    This is the kernel entry point for an outer optimiser (CMA-ES): the optimiser SETS the
    ``len(b_gene)`` leading nullspace coordinates (the "gene" curvature modes) and the kernel
    solves only the *size-fixing* degrees of freedom -- the arc length ``S`` and the remaining
    ("reserved") nullspace coordinates ``b[k:D]`` -- to hit the nonlinear flat-baffle size targets
    ``x(1)=x_target`` and ``r(1)=r_mouth``. Because every ``a = a0(S) + Phi @ b`` satisfies the
    linear boundary conditions by construction, the returned curve meets the flat-baffle BCs
    (``kappa(0)=kappa0``, ``kappa(1)=0``, ``dkappa/dsigma(1)=0``, ``theta(1)=90 deg``) EXACTLY for
    *any* gene vector -- so the outer optimiser gets local shape control with fixed endpoints.

    Parameters
    ----------
    b_gene : 1D array-like, length ``k <= n_shape_modes(targets, n_coeff, degree)``. Coordinates
        on the first ``k`` columns of the nullspace basis ``Phi`` (the gene modes). ``k = 0`` is
        allowed and reproduces the all-reserved solve.
    targets : design intent; must be ``mode='flat_baffle'`` (raises ``ValueError`` otherwise).
    n_coeff, degree : the curvature B-spline basis (same defaults as :func:`solve_icw`).
    max_nfev : least-squares evaluation budget for the size solve over ``[S, b_reserved]``.

    Returns
    -------
    ``(ICWCurve, FeasibilityReport)``. ``feasible=True`` only if the size targets are hit within
    the SAME tolerance :func:`solve_icw` uses AND every guard passes (linear BCs, terminal angle,
    strictly/weakly increasing radius via :data:`core.MONO_EPS`, no self-intersection). On failure
    a TRUTHFUL infeasible report is returned (violations populated) -- never a silent repair.

    Notes
    -----
    Reuses ``solve_icw``'s precomputed :class:`_LinSys` (C, pinv(C), the SVD nullspace ``Phi`` and
    the quadrature grid) -- nothing is rebuilt inside the residual. The gene coordinates are baked
    into a fixed contribution ``Phi[:, :k] @ b_gene`` added to ``a0`` once per residual; only
    ``[S, b_reserved]`` vary. The same Tikhonov style regulariser (:data:`_REG_LAMBDA`) is applied
    to the reserved coordinates ``b_reserved`` for well-posedness (and NOT to the gene coords,
    which are fixed, nor to the size residual feasibility is judged on).
    """
    knots, lin = _flat_baffle_lin(targets, n_coeff, degree)
    Phi = lin.Phi
    D = lin.n_shape  # nullspace dimension

    b_gene = np.atleast_1d(np.asarray(b_gene, dtype=float))
    if b_gene.ndim != 1:
        raise ValueError(f"b_gene must be 1D, got shape {b_gene.shape}")
    k = int(b_gene.size)
    k_max = max(D - 1, 0)
    if k > k_max:
        raise ValueError(
            f"len(b_gene)={k} exceeds n_shape_modes={k_max} (nullspace dim D={D}; one mode is "
            "reserved with S for the size solve)"
        )

    # ---- (1) early necessary feasibility (same gate as solve_icw) ------------------------
    violations = _early_feasibility(targets)
    if violations:
        return _infeasible_curve(targets, knots, n_coeff, degree), FeasibilityReport(
            feasible=False,
            violations=violations,
            residuals={},
            suggested_relaxation=_relaxation_hint(targets, violations),
            message="Target set is malformed or violates necessary conditions before solving.",
        )

    n_reserved = D - k  # nullspace coords the kernel optimises (>= 1 since k <= D - 1)
    # Fixed gene contribution to the curvature coeffs: a = a0(S) + Phi[:, :k] @ b_gene + reserved.
    gene_coeffs = Phi[:, :k] @ b_gene if k else np.zeros(n_coeff)
    Phi_res = Phi[:, k:]  # reserved nullspace columns (k .. D)

    def _residual_fixed(p: np.ndarray):
        """Residual for ``p = [S, b_reserved]`` with the gene coords held fixed.

        Mirrors :func:`_residual_and_aux` (flat-baffle branch) but the curvature coeffs are
        ``a0(S) + gene_coeffs + Phi_res @ b_reserved`` -- gene modes fixed, reserved modes free.
        Returns ``(full_residual, size_residual, coeffs, S, sample_tuple)`` with the same meaning.
        """
        S = float(p[0])
        b_res = p[1 : 1 + n_reserved]
        a0 = lin.a0(targets.theta0, targets.theta1, S)
        coeffs = a0 + gene_coeffs + (Phi_res @ b_res if n_reserved else 0.0)

        sigma, theta, x, r = _integrate_meridian(
            coeffs, knots, degree, S, targets.r0, targets.theta0, sigma=lin.sigma
        )
        # Hard size targets (judged for feasibility): x(1)=x_target, r(1)=r_mouth.
        size = np.array([x[-1] - targets.x_target, r[-1] - targets.r_mouth], dtype=float)
        # Soft monotone-r barrier (guides only), same form/weight as the flat-baffle solver.
        neg = np.clip(-theta, 0.0, None)
        barrier = np.array([_MONO_WEIGHT * float(np.sqrt(np.mean(neg**2)))], dtype=float)
        # Tikhonov regulariser on the RESERVED coords only (genes are fixed). Same style/weight as
        # the all-DOF solver so easy cases stay exact and the size solve stays well posed; it is in
        # the optimiser residual ONLY -- never in ``size`` -- and a0+Phi@b satisfies the BCs exactly.
        reg = np.sqrt(_REG_LAMBDA) * b_res if n_reserved else np.zeros(0)
        full = np.concatenate([size, barrier, reg])
        return full, size, coeffs, S, (sigma, theta, x, r)

    # ---- bounds: S strictly above the chord; reserved coords bounded like the all-DOF solver.
    chord = _chord(targets)
    S_seed = max(chord * 1.08, chord + 1.0, 1.0)
    S_lo = max(chord * (1.0 + 1e-6), 1e-6)
    S_hi = max(chord * 50.0, S_seed * 10.0, 10.0)
    kappa_scale = 5.0 / max(targets.r0, 1.0)
    lo = np.concatenate([[S_lo], np.full(n_reserved, -kappa_scale)])
    hi = np.concatenate([[S_hi], np.full(n_reserved, kappa_scale)])

    def _run(p_init: np.ndarray, nfev: int) -> np.ndarray:
        p_init = np.clip(p_init, lo + 1e-12, hi - 1e-12)
        sol = least_squares(
            lambda p: _residual_fixed(p)[0],
            p_init,
            bounds=(lo, hi),
            max_nfev=nfev,
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
        )
        return sol.x

    def _assemble_fixed(p: np.ndarray) -> tuple[ICWCurve, FeasibilityReport]:
        _full, size_res, coeffs, S, (sigma, theta, x, r) = _residual_fixed(p)
        curve = ICWCurve(
            coeffs=coeffs, S=S, r0=targets.r0, theta0=targets.theta0, knots=knots, degree=degree
        )
        report = _verify(targets, curve, size_res, sigma, theta, x, r)
        # Explicit no-self-intersection guard for this entry point. ``_verify`` already enforces
        # the monotone-r rule (which makes a flat-baffle meridian single-valued), but a gene that
        # bends curvature hard could in principle fold the polyline; we check it directly via the
        # shared geometric test so the report is truthful and never claims a self-crossing wall is
        # feasible. (Lazy import keeps the icw.solver module free of any checks-time import cost.)
        from .checks import meridian_self_intersects

        self_x = meridian_self_intersects(x, r)
        report.residuals["self_intersects"] = bool(self_x)
        if self_x and "meridian self-intersects" not in " ".join(report.violations):
            report.violations.append("meridian self-intersects")
            report.feasible = False
            if report.message.startswith("Feasible"):
                report.message = (
                    "Infeasible: meridian self-intersects. No silent repair performed."
                )
        return curve, report

    # ---- size solve: direct, then a small multistart rescue (same spirit as solve_icw). The
    # gene coords are FIXED throughout; only [S, b_reserved] move. We accept a retry only if it is
    # strictly better, so an unreachable size target (or a gene that forces a non-monotone wall)
    # still reports feasible=False with its violations -- never a silent repair into a wrong curve.
    p0 = np.concatenate([[S_seed], np.zeros(n_reserved)])
    p_best = _run(p0, max_nfev)
    curve_best, report_best = _assemble_fixed(p_best)

    if not report_best.feasible:
        rng = np.random.default_rng(0)
        seeds: list[np.ndarray] = []
        for sf in (1.2, 1.5, 2.0, 3.0):
            seeds.append(np.concatenate([[max(chord * sf, S_lo * 1.001)], np.zeros(n_reserved)]))
        for _ in range(4):
            b_pert = (
                rng.normal(scale=0.05 * kappa_scale, size=n_reserved)
                if n_reserved
                else np.zeros(0)
            )
            seeds.append(np.concatenate([[S_seed], b_pert]))
        for sd in seeds:
            try:
                p_try = _run(sd, max_nfev)
            except Exception:
                continue
            curve_t, report_t = _assemble_fixed(p_try)
            # Reuse solve_icw's ordering: feasibility wins, then fewer violations, then smaller
            # hard size residual.
            better = (
                report_t.feasible != report_best.feasible and report_t.feasible
            ) or (
                report_t.feasible == report_best.feasible
                and (
                    len(report_t.violations) < len(report_best.violations)
                    or (
                        len(report_t.violations) == len(report_best.violations)
                        and report_t.residuals.get("size_residual_max", np.inf)
                        < report_best.residuals.get("size_residual_max", np.inf)
                    )
                )
            )
            if better:
                curve_best, report_best = curve_t, report_t
            if report_best.feasible:
                break

    return curve_best, report_best


# --------------------------------------------------------------------------------------------
# Verification & reporting
# --------------------------------------------------------------------------------------------
def _verify(
    targets: ICWTargets,
    curve: ICWCurve,
    size_res: np.ndarray,
    sigma: np.ndarray,
    theta: np.ndarray,
    x: np.ndarray,
    r: np.ndarray,
) -> FeasibilityReport:
    """Check linear BCs, size residuals, and mode-specific geometric invariants.

    ``size_res`` is the hard size-target residual only (no soft barrier terms).
    """
    res = size_res
    violations: list[str] = []
    residuals: dict = {}

    # --- linear BC residuals (should be ~0 by construction) ---
    kappa1 = float(curve.kappa(np.array([1.0]))[0])
    kappa_spl = curve.kappa_spline()
    dkappa1 = float(kappa_spl(np.array([1.0]), nu=1)[0])
    kappa0_res = float(curve.kappa(np.array([0.0]))[0] - targets.kappa0)
    residuals["kappa0"] = kappa0_res
    residuals["kappa1"] = kappa1
    residuals["dkappa1"] = dkappa1

    if abs(kappa0_res) > TOL_KAPPA_BC:
        violations.append(f"kappa(0) residual {kappa0_res:.2e} exceeds {TOL_KAPPA_BC:.0e}")

    if targets.mode == TerminationMode.FLAT_BAFFLE:
        if abs(kappa1) > TOL_KAPPA_BC:
            violations.append(f"kappa(1)={kappa1:.2e} not zero (flat baffle)")
        if targets.enforce_dkappa_end and abs(dkappa1) > 1e-4:
            violations.append(f"dkappa/dsigma(1)={dkappa1:.2e} not zero (flat baffle)")

    # --- terminal angle ---
    theta_end = float(theta[-1])
    residuals["theta_end_deg"] = float(np.degrees(theta_end))
    angle_res = theta_end - targets.theta1
    residuals["theta_end_res_rad"] = float(angle_res)
    if abs(angle_res) > TOL_ANGLE_RAD:
        violations.append(
            f"terminal angle {np.degrees(theta_end):.3f}deg misses target "
            f"{targets.theta1_deg:.3f}deg by {np.degrees(angle_res):.3f}deg"
        )

    # --- size residuals ---
    residuals["size_residual_max"] = float(np.max(np.abs(res))) if res.size else 0.0
    residuals["x_end"] = float(x[-1])
    residuals["r_end"] = float(r[-1])
    if res.size and np.max(np.abs(res)) > TOL_LENGTH_MM:
        violations.append(
            f"size targets not met: max residual {np.max(np.abs(res)):.3e} mm "
            f"exceeds {TOL_LENGTH_MM:.0e} mm"
        )

    # --- monotone radius (weakly non-decreasing; a cylindrical/flat section is allowed) ---
    # Same WEAK rule (dr >= -MONO_EPS) as checks.is_monotone_radius, so the two agree: a
    # near-cylindrical throat/exit (theta ~ 0) is accepted, a genuinely re-entrant wall is not.
    dr = np.diff(r)
    if np.any(dr < -MONO_EPS):
        violations.append("radius is not monotone (non-decreasing) along the meridian")
    residuals["r_monotone"] = bool(np.all(dr >= -MONO_EPS))

    # --- mode-specific geometric invariants ---
    if targets.mode == TerminationMode.ROLLBACK:
        # Shared all-crossings count (matches aperture_report.n_pi2_crossings exactly).
        ncross = len(_theta90_crossings(theta))
        residuals["theta90_crossings"] = int(ncross)
        if theta_end <= np.pi / 2.0 + 1e-6:
            violations.append(
                f"rollback terminal angle {np.degrees(theta_end):.2f}deg must exceed 90deg"
            )
        if ncross != 1:
            violations.append(
                f"rollback must have exactly one theta=90deg crossing, found {ncross}"
            )

    feasible = len(violations) == 0
    msg = (
        "Feasible: all boundary, angle, size and geometric constraints met within tolerance."
        if feasible
        else "Infeasible within tolerance/bounds; see violations. No silent repair performed."
    )
    return FeasibilityReport(
        feasible=feasible,
        violations=violations,
        residuals=residuals,
        suggested_relaxation=None if feasible else _relaxation_hint(targets, violations, residuals),
        message=msg,
    )


def _relaxation_hint(
    targets: ICWTargets,
    violations: list[str],
    residuals: dict | None = None,
) -> dict:
    """Heuristic suggestions for making an infeasible target reachable."""
    hint: dict = {}
    chord = _chord(targets)
    text = " ".join(violations).lower()

    if targets.mode == TerminationMode.FLAT_BAFFLE:
        if "r_mouth" in text or "mouth wider" in text:
            hint["r_mouth"] = f"increase r_mouth above r0={targets.r0}"
        if "x_target" in text:
            hint["x_target"] = "set x_target > 0"
        if "size targets" in text:
            # endpoint unreachable: usually depth/width too small for a 90-deg landing.
            hint["x_target"] = (
                f"increase x_target (>= ~{chord * 0.6:.1f} mm) so the wall can turn to 90deg"
            )
            hint["r_mouth"] = "or increase r_mouth to give the lip room to flatten"
    else:
        if "axial target" in text:
            hint["depth"] = "supply one of depth / x_aperture / x_setback"
        if "radial target" in text:
            hint["r_aperture"] = "supply one of r_aperture / r_end"
        if "theta1" in text:
            hint["theta1_deg"] = "set theta1_deg > 90 (e.g. 120-160) for a rollback"
        if "crossing" in text or "exceed 90" in text:
            hint["rollback_theta_max_deg"] = (
                f"raise terminal angle (current {targets.theta1_deg:.0f}deg) so the wall curls "
                "past 90deg exactly once"
            )
        if "size targets" in text:
            hint["depth"] = "relax depth / aperture targets (current set may be geometrically over-constrained)"
    if not hint:
        hint["S"] = f"arc length is bounded below by the chord ~{chord:.1f} mm; relax size targets"
    return hint


def _infeasible_curve(
    targets: ICWTargets, knots: np.ndarray, n_coeff: int, degree: int
) -> ICWCurve:
    """A best-effort placeholder curve returned alongside an infeasible report.

    It satisfies the linear BCs (so it is a valid ICWCurve) at a chord-length guess for S, but
    makes no claim of meeting the size targets -- the report says so.
    """
    S = max(_chord(targets), 1.0)
    C, d = build_linear_constraints(targets, S, knots, n_coeff, degree)
    a0, _ = feasible_subspace(C, d)
    return ICWCurve(
        coeffs=a0, S=S, r0=targets.r0, theta0=targets.theta0, knots=knots, degree=degree
    )
