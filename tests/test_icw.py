"""Tests for the Intrinsic-Curvature Waveguide (ICW) kernel.

Covers the public ICW API (core / seed / solver / checks). Uses only the public surface
re-exported from :mod:`hornlab_mesher.icw`, plus numpy/scipy and -- for the seed fits --
:mod:`hornlab_mesher.profile_formulas` to generate the analytic reference meridians.

Includes the round-trip guard for the reconciled ``x_setback`` sign convention
(``x_setback = x_aperture - x_end >= 0`` for a rollback whose rim sits behind the aperture),
which is now consistent between :func:`solve_icw` and :func:`aperture_report`.
"""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.icw import (
    ICWCurve,
    ICWTargets,
    TerminationMode,
    aperture_report,
    feature_scale_ok,
    fit_error,
    hom_cutoff_hz,
    is_monotone_radius,
    meridian_self_intersects,
    seed_from_osse,
    seed_from_rosse,
    shell_offset_report,
    solve_icw,
    station_report,
)
from hornlab_mesher.icw.seed import _density_knots
from hornlab_mesher.profile_formulas import profile_points


# OSSE / R-OSSE reference parameter dicts (from the task spec).
OSSE_PARAMS = {
    "type": "OSSE", "r0": 12.7, "a0": 18, "a": 35, "k": 1,
    "L": 120, "s": 0.8, "n": 4, "q": 0.9,
}
ROSSE_PARAMS = {
    "type": "R-OSSE", "R": 150, "r0": 12.7, "k": 1, "q": 1, "m": 0.85,
    "r": 0.4, "b": 0.2, "a": 60, "a0": 15.5,
}


def _fit_circle(x, r):
    """Algebraic circle fit; returns (cx, cy, R)."""
    A = np.column_stack([x, r, np.ones_like(x)])
    b = x**2 + r**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    R = np.sqrt(sol[2] + cx**2 + cy**2)
    return cx, cy, R


# =====================================================================================
# CORE
# =====================================================================================
class TestCore:
    def test_cone_theta_constant_x_linear_mouth_radius(self):
        r0, half_angle_deg, L = 12.7, 15.0, 100.0
        curve = ICWCurve.cone(r0=r0, half_angle_deg=half_angle_deg, length_axial=L)
        s = curve.sample(1001)

        # theta is constant along the whole meridian (kappa == 0).
        theta_deg = np.degrees(s.theta)
        assert np.allclose(theta_deg, half_angle_deg, atol=1e-9)

        # x is linear in sigma: deviation from the straight line is ~0.
        x_lin = np.linspace(s.x[0], s.x[-1], s.x.size)
        assert np.max(np.abs(s.x - x_lin)) < 1e-9

        # mouth radius = r0 + L * tan(theta0); axial extent = L exactly.
        assert s.x[-1] == pytest.approx(L, abs=1e-9)
        r_mouth = r0 + L * np.tan(np.radians(half_angle_deg))
        assert s.r[-1] == pytest.approx(r_mouth, abs=1e-9)

    def test_constant_curvature_is_circular_arc(self):
        kappa = 0.01  # 1/mm -> radius of curvature 100 mm
        curve = ICWCurve(
            coeffs=np.full(8, kappa), S=80.0, r0=12.7, theta0=np.radians(10)
        )
        s = curve.sample(2000)

        # Curvature really is constant at the requested value.
        assert np.allclose(s.kappa, kappa, atol=1e-12)

        cx, cy, R = _fit_circle(s.x, s.r)
        assert R == pytest.approx(1.0 / kappa, rel=1e-6)

        # Every sample lies on that circle: radius spread is tiny.
        radii = np.hypot(s.x - cx, s.r - cy)
        assert (radii.max() - radii.min()) < 1e-6

    def test_gauges_match_finite_difference_in_body(self):
        curve = ICWCurve(
            coeffs=np.array([0.0, 0.002, 0.004, 0.006, 0.008, 0.006, 0.004, 0.002]),
            S=120.0, r0=12.7, theta0=np.radians(8),
        )
        s = curve.sample(4000)
        x, r = s.x, s.r

        # Finite-difference reference for r' and r'' w.r.t. x.
        rp = np.gradient(r, x)
        rpp = np.gradient(rp, x)
        Fx_fd = 2.0 * rp / r  # 2 r'/r
        Qx_fd = rpp / r  # r''/r

        body = np.degrees(s.theta) < 60.0
        # Trim the band ends where np.gradient is one-sided / the body mask is ragged.
        idx = np.where(body)[0][5:-5]

        relFx = np.abs(s.flare_rate_Fx[idx] - Fx_fd[idx]) / (np.abs(Fx_fd[idx]) + 1e-9)
        relQx = np.abs(s.webster_Qx[idx] - Qx_fd[idx]) / (np.abs(Qx_fd[idx]) + 1e-9)
        assert relFx.max() < 1e-2
        assert relQx.max() < 1e-2

    def test_theta_end_matches_sample(self):
        curve = ICWCurve(
            coeffs=np.array([0.0, 0.002, 0.004, 0.006, 0.008, 0.006, 0.004, 0.002]),
            S=120.0, r0=12.7, theta0=np.radians(8),
        )
        s = curve.sample(4000)
        assert curve.theta_end() == pytest.approx(float(s.theta[-1]), abs=1e-6)


# =====================================================================================
# SEED
# =====================================================================================
class TestSeed:
    def test_osse_seed_submicron(self):
        curve = seed_from_osse(OSSE_PARAMS)
        pts = profile_points(OSSE_PARAMS, 4000)
        err = fit_error(curve, pts[:, 0], pts[:, 1])
        assert err["max_mm"] < 1e-3

    def test_rosse_rollback_seed_preserves_rollback(self):
        curve = seed_from_rosse(ROSSE_PARAMS)
        pts = profile_points(ROSSE_PARAMS, 4000)
        err = fit_error(curve, pts[:, 0], pts[:, 1])
        assert err["max_mm"] < 0.05

        # The reconstructed meridian must curl past 90 deg (rollback preserved).
        s = curve.sample(4000)
        assert np.degrees(s.theta).max() > 90.0


# =====================================================================================
# SOLVER
# =====================================================================================
class TestSolver:
    def test_flat_baffle_feasible_and_terminated(self):
        targets = ICWTargets(
            mode=TerminationMode.FLAT_BAFFLE,
            r0=12.7, theta0_deg=18, x_target=120, r_mouth=116,
        )
        curve, report = solve_icw(targets)
        assert report.feasible, report.violations

        s = curve.sample(4000)
        # theta(1) = 90 deg, kappa(1) = 0, dkappa/dsigma(1) = 0 (flat-baffle landing).
        assert np.degrees(curve.theta_end()) == pytest.approx(90.0, abs=1e-2)
        assert curve.kappa(1.0) == pytest.approx(0.0, abs=1e-6)
        dkappa1 = float(curve.kappa_spline()(np.array([1.0]), nu=1)[0])
        assert dkappa1 == pytest.approx(0.0, abs=1e-4)

        # Size targets hit; radius strictly increasing.
        assert s.x[-1] == pytest.approx(120.0, abs=1e-2)
        assert s.r[-1] == pytest.approx(116.0, abs=1e-2)
        assert np.all(np.diff(s.r) > 0.0)

    def test_rollback_feasible_aperture_roundtrip(self):
        r_aperture, x_setback = 70.0, 6.0
        targets = ICWTargets(
            mode=TerminationMode.ROLLBACK,
            r0=12.7, theta0_deg=12, theta1_deg=110,
            r_aperture=r_aperture, x_setback=x_setback,
        )
        curve, report = solve_icw(targets)
        assert report.feasible, report.violations

        s = curve.sample(4000)
        # theta_end > 90; exactly one ascending theta = 90 crossing; r monotone.
        assert np.degrees(curve.theta_end()) > 90.0
        assert is_monotone_radius(s)

        ar = aperture_report(s)
        assert ar.n_pi2_crossings == 1
        assert ar.is_rollback
        # Requested targets met by the report (incl. x_setback round-trip from TASK 1:
        # same sign, same value -- x_setback = x_aperture - x_end >= 0).
        assert ar.r_aperture == pytest.approx(r_aperture, abs=1e-2)
        assert ar.x_setback == pytest.approx(x_setback, abs=1e-2)

    def test_infeasible_target_reports_violations(self):
        # r_mouth < r0 is a necessary-condition violation: mouth narrower than throat.
        targets = ICWTargets(
            mode=TerminationMode.FLAT_BAFFLE,
            r0=50.0, theta0_deg=18, x_target=120, r_mouth=30,
        )
        _curve, report = solve_icw(targets)
        assert report.feasible is False
        assert report.violations  # non-empty

    def test_warm_start_from_seed(self):
        # Continuation / warm-start: exercise the seed= path with an OSSE seed curve.
        seed = seed_from_osse(OSSE_PARAMS)
        targets = ICWTargets(
            mode=TerminationMode.FLAT_BAFFLE,
            r0=12.7, theta0_deg=18, x_target=120, r_mouth=116,
        )
        _curve, report = solve_icw(targets, seed=seed)
        assert report.feasible, report.violations


# =====================================================================================
# CHECKS
# =====================================================================================
class TestChecks:
    def test_is_monotone_radius(self):
        expanding = ICWCurve.cone(r0=12.7, half_angle_deg=15, length_axial=100)
        assert is_monotone_radius(expanding.sample(1000)) is True

        # Driven well past theta = 180 deg -> radius turns back (non-monotone).
        past_180 = ICWCurve(
            coeffs=np.full(8, 0.05), S=120.0, r0=12.7, theta0=np.radians(10)
        )
        s_bad = past_180.sample(2000)
        assert np.degrees(s_bad.theta).max() > 180.0
        assert is_monotone_radius(s_bad) is False

    def test_meridian_self_intersects(self):
        forward = ICWCurve.cone(r0=12.7, half_angle_deg=15, length_axial=100).sample(1000)
        assert meridian_self_intersects(forward.x, forward.r) is False

        # An explicit crossing polyline (an "X").
        x = np.array([0.0, 1.0, 1.0, 0.0])
        r = np.array([0.0, 1.0, 0.0, 1.0])
        assert meridian_self_intersects(x, r) is True

    def test_shell_offset_report(self):
        # Gentle cone + thin wall -> regular offset, ok.
        gentle = ICWCurve.cone(r0=20.0, half_angle_deg=10.0, length_axial=150.0).sample(2000)
        rep_ok = shell_offset_report(gentle, wall_thickness=1.0, margin=0.4)
        assert rep_ok.ok is True
        assert not rep_ok.violations

        # Tight curve + thick wall -> both principal curvatures flagged.
        tight = ICWCurve(coeffs=np.full(8, 0.05), S=40.0, r0=5.0, theta0=np.radians(5)).sample(2000)
        rep_bad = shell_offset_report(tight, wall_thickness=10.0, margin=0.4)
        assert rep_bad.ok is False
        assert rep_bad.max_t_kappa_meridional >= 0.4
        assert rep_bad.max_t_kappa_circumferential >= 0.4
        assert len(rep_bad.violations) >= 2

    def test_hom_cutoff_hz(self):
        assert hom_cutoff_hz(10, "11") == pytest.approx(10_050.0, rel=2e-3)
        assert hom_cutoff_hz(10, "01") == pytest.approx(20_900.0, rel=2e-3)

    def test_aperture_report_forward_curve(self):
        forward = ICWCurve.cone(r0=20.0, half_angle_deg=10.0, length_axial=150.0).sample(2000)
        ar = aperture_report(forward)
        assert ar.is_rollback is False
        assert ar.n_pi2_crossings == 0
        assert ar.x_setback == 0.0
        assert ar.r_aperture == ar.r_end

    def test_station_report(self):
        curve = ICWCurve(
            coeffs=np.array([0.0, 0.002, 0.004, 0.006, 0.008, 0.006, 0.004, 0.002]),
            S=120.0, r0=12.7, theta0=np.radians(8),
        )
        s = curve.sample(2000)

        # Sane extrema: max >= min for every reported geometric range.
        rep = station_report(s, theta_body_max_deg=72.0)
        assert rep.theta_deg_max >= rep.theta_deg_min
        assert rep.kappa_max >= rep.kappa_min
        assert rep.abs_kappa_max >= 0.0
        assert rep.theta_deg_max == pytest.approx(np.degrees(s.theta).max(), abs=1e-6)

        # The body exceeds a tight theta_body_max -> theta_body_valid is False.
        rep_tight = station_report(s, theta_body_max_deg=30.0)
        assert rep_tight.theta_deg_max > 30.0
        assert rep_tight.theta_body_valid is False


# =====================================================================================
# ADVERSARIAL / REGRESSION GUARDS
#
# These pin the review findings fixed in the Phase-0 ICW kernel so they cannot silently
# regress: the deep-narrow feasibility basin (P0-1), the single shared theta=90 crossing
# definition (P1-2), the weak monotone-radius rule (P1-3), gauge validity near the cap,
# the continuation rescue, and truthful infeasibility reporting.
# =====================================================================================
class TestAdversarial:
    def test_deep_narrow_flat_baffle_feasible_across_n_coeff(self):
        """P0-1: a deep, narrow flat-baffle turn must be feasible at the DEFAULT n_coeff and
        beyond -- not just at the small counts that happened to work before.

        Before the full-nullspace + Tikhonov fix this exact target solved at n_coeff 7/8/10 but
        reported feasible=False at 12 (the default), 14, 16: more coefficients made it strictly
        worse because the solve used only the first few smooth GLOBAL SVD modes, which cannot bend
        the throat. Now every count solves, hits both size targets, and stays monotone.
        """
        for n_coeff in (8, 12, 16):
            targets = ICWTargets(
                mode=TerminationMode.FLAT_BAFFLE,
                r0=12.7, theta0_deg=18, x_target=120, r_mouth=80,
            )
            curve, report = solve_icw(targets, n_coeff=n_coeff)
            assert report.feasible, (n_coeff, report.violations)

            s = curve.sample(4000)
            assert s.x[-1] == pytest.approx(120.0, abs=1e-3)
            assert s.r[-1] == pytest.approx(80.0, abs=1e-3)
            # Weakly monotone (a deep turn may go near-cylindrical mid-body): same rule the
            # solver and is_monotone_radius share.
            assert is_monotone_radius(s)
            # Linear BCs still hold to ~1e-12 by construction (the regulariser must not move them).
            assert curve.kappa(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-12)  # kappa0=0
            assert curve.kappa(np.array([1.0]))[0] == pytest.approx(0.0, abs=1e-9)
            dk1 = float(curve.kappa_spline()(np.array([1.0]), nu=1)[0])
            assert dk1 == pytest.approx(0.0, abs=1e-6)

    def test_deep_narrow_second_basin_and_easy_still_exact(self):
        """P0-1 (cont.): an even deeper target (r_mouth=70) is feasible across the sweep, and the
        original easy target (r_mouth=116) still solves exactly -- the fix must not trade one for
        the other.
        """
        for n_coeff in (8, 12, 16, 20):
            deep = ICWTargets(
                mode=TerminationMode.FLAT_BAFFLE,
                r0=12.7, theta0_deg=18, x_target=120, r_mouth=70,
            )
            c_deep, r_deep = solve_icw(deep, n_coeff=n_coeff)
            assert r_deep.feasible, (n_coeff, r_deep.violations)
            s_deep = c_deep.sample(4000)
            assert s_deep.x[-1] == pytest.approx(120.0, abs=1e-3)
            assert s_deep.r[-1] == pytest.approx(70.0, abs=1e-3)

            easy = ICWTargets(
                mode=TerminationMode.FLAT_BAFFLE,
                r0=12.7, theta0_deg=18, x_target=120, r_mouth=116,
            )
            c_easy, r_easy = solve_icw(easy, n_coeff=n_coeff)
            assert r_easy.feasible, (n_coeff, r_easy.violations)
            s_easy = c_easy.sample(4000)
            assert s_easy.x[-1] == pytest.approx(120.0, abs=1e-3)
            assert s_easy.r[-1] == pytest.approx(116.0, abs=1e-3)
            assert np.all(np.diff(s_easy.r) > 0.0)  # the easy case is strictly increasing

    def test_crossing_count_reconciled_solver_vs_aperture_report(self):
        """P1-2: the solver's theta=90 crossing count and aperture_report.n_pi2_crossings come
        from ONE shared definition, so they agree on the same rollback curve (exactly 1).
        """
        targets = ICWTargets(
            mode=TerminationMode.ROLLBACK,
            r0=12.7, theta0_deg=12, theta1_deg=110, r_aperture=70, x_setback=6,
        )
        curve, report = solve_icw(targets)
        assert report.feasible, report.violations

        s = curve.sample(4000)
        ar = aperture_report(s)
        assert report.residuals["theta90_crossings"] == ar.n_pi2_crossings
        assert ar.n_pi2_crossings == 1  # a clean rollback crosses the mouth plane exactly once
        assert np.degrees(curve.theta_end()) > 90.0

    def test_is_monotone_radius_accepts_near_cylindrical(self):
        """P1-3: a near-cylindrical / cylindrical section (theta0 ~ 0 => dr/ds ~ 0) is weakly
        monotone and must return True, not be rejected by a strict interior rule.
        """
        # Exactly cylindrical: kappa == 0, theta0 == 0 => dr/ds == 0 everywhere.
        cyl = ICWCurve(coeffs=np.zeros(6), S=100.0, r0=20.0, theta0=0.0)
        assert is_monotone_radius(cyl.sample(2000)) is True

        # Nearly cylindrical throat then a gentle flare.
        almost = ICWCurve.cone(r0=20.0, half_angle_deg=0.05, length_axial=100.0)
        assert is_monotone_radius(almost.sample(2000)) is True

        # Sanity: a genuinely re-entrant wall (radius turns back) still fails.
        reentrant = ICWCurve(coeffs=np.full(8, 0.05), S=120.0, r0=12.7, theta0=np.radians(10))
        s_bad = reentrant.sample(2000)
        assert np.degrees(s_bad.theta).max() > 180.0
        assert is_monotone_radius(s_bad) is False

    def test_gauge_amplification_near_cap_grows_like_inv_cos_cubed(self):
        """P1-4: the Webster body gauges blow up like 1/cos^3 as theta -> 90 deg; verify the
        amplification at theta ~ 70 deg matches 1/cos^3 (they are documented body-only and are
        intentionally NOT masked).
        """
        kappa, r0 = 0.01, 20.0
        lo = ICWCurve(coeffs=np.full(8, kappa), S=80.0, r0=r0, theta0=np.radians(10))
        hi = ICWCurve(coeffs=np.full(8, kappa), S=80.0, r0=r0, theta0=np.radians(70))
        s_lo, s_hi = lo.sample(2000), hi.sample(2000)

        # Qx = kappa/(r cos^3 theta); take the throat station where r == r0 for both.
        qx_lo, qx_hi = s_lo.webster_Qx[0], s_hi.webster_Qx[0]
        expected_ratio = (np.cos(np.radians(10)) / np.cos(np.radians(70))) ** 3
        assert (qx_hi / qx_lo) == pytest.approx(expected_ratio, rel=1e-3)
        assert expected_ratio > 20.0  # ~24x: a real, large amplification near the cap

        # Fx = 2 tan(theta)/r likewise amplifies via tan.
        fx_lo, fx_hi = s_lo.flare_rate_Fx[0], s_hi.flare_rate_Fx[0]
        assert (fx_hi / fx_lo) == pytest.approx(
            np.tan(np.radians(70)) / np.tan(np.radians(10)), rel=1e-3
        )

    def test_station_report_theta_body_valid_false_when_curve_exceeds_max(self):
        """P1-4: station_report.theta_body_valid is False once the curve's max angle exceeds
        theta_body_max (the body leaves the trusted band).
        """
        # A flat-baffle curve reaches 90 deg, well past any sane theta_body_max.
        targets = ICWTargets(
            mode=TerminationMode.FLAT_BAFFLE,
            r0=12.7, theta0_deg=18, x_target=120, r_mouth=116,
        )
        curve, report = solve_icw(targets)
        assert report.feasible, report.violations
        s = curve.sample(4000)

        rep = station_report(s, theta_body_max_deg=72.0)
        assert rep.theta_deg_max > 72.0
        assert rep.theta_body_valid is False

        # With a max above the curve's peak, the body is valid again.
        rep_ok = station_report(s, theta_body_max_deg=95.0)
        assert rep_ok.theta_body_valid is True

    def test_feature_scale_ok_uses_minimum_span(self):
        """P1-1: feature_scale_ok is gated by the TIGHTEST interior knot span. With the
        density-clustered knots the seed-fitter emits, a tight cluster below the floor must FAIL
        even though the largest span is comfortably long (the old max-span guard wrongly passed).
        """
        sigma = np.linspace(0.0, 1.0, 500)
        kappa = 0.02 * np.exp(-((sigma) / 0.04) ** 2)  # sharp curvature cusp at the throat
        knots = _density_knots(sigma, kappa, n_coeff=20)
        spans = np.diff(np.unique(knots))
        assert spans.min() < spans.max()  # genuinely non-uniform (clustered) knots

        S = 100.0
        curve = ICWCurve(
            coeffs=np.zeros(20), S=S, r0=12.7, theta0=np.radians(10), knots=knots
        )
        support_min = S * (curve.degree + 1) * spans.min()
        support_max = S * (curve.degree + 1) * spans.max()

        # A floor between the min-span and max-span support: the conservative (min-span) guard
        # must reject it; a max-span guard would have wrongly accepted it.
        floor = 0.5 * (support_min + support_max)
        assert feature_scale_ok(curve, floor) is False
        # Below the true (min-span) support length it passes.
        assert feature_scale_ok(curve, 0.5 * support_min) is True

    def test_continuation_rescues_a_hard_rollback(self):
        """P1-4: a hard rollback the single cold/direct least-squares start cannot solve is
        rescued by solve_icw's continuation (homotopy on the terminal angle from 90 deg up).

        We first confirm the direct-only attempt fails on this target, then that the full
        solve_icw -- which adds the continuation sweep -- reports feasible. This exercises the
        homotopy path, not merely an easy target.
        """
        import hornlab_mesher.icw.solver as solver_mod
        from hornlab_mesher.icw.core import DEFAULT_DEGREE, clamped_uniform_knots
        from scipy.optimize import least_squares

        targets = ICWTargets(
            mode=TerminationMode.ROLLBACK,
            r0=12.7, theta0_deg=5, theta1_deg=179, r_aperture=150, x_setback=40,
        )
        n_coeff, max_nfev = 12, 400
        degree = DEFAULT_DEGREE
        knots = clamped_uniform_knots(n_coeff, degree)

        # --- direct-only attempt (mirror solve_icw's first _run + _assemble, no fallbacks) ---
        assert not solver_mod._early_feasibility(targets)  # passes the cheap necessary checks
        # P1-2: the S/b-independent constraint algebra is now precomputed once into a _LinSys and
        # threaded through the residual/_assemble (replacing the bare Phi argument).
        lin = solver_mod._build_linsys(targets, knots, n_coeff, degree)
        Phi = lin.Phi
        n_shape = lin.n_shape
        chord = solver_mod._chord(targets)
        S_seed, b_seed = solver_mod._seed_state(targets, Phi, None, n_shape)
        S_lo = max(chord * (1.0 + 1e-6), 1e-6)
        S_hi = max(chord * 50.0, S_seed * 10.0, 10.0)
        ks = 5.0 / max(targets.r0, 1.0)
        lo = np.concatenate([[S_lo], np.full(n_shape, -ks)])
        hi = np.concatenate([[S_hi], np.full(n_shape, ks)])

        def fun(p):
            res, *_ = solver_mod._residual_and_aux(
                p, targets, knots, n_coeff, degree, lin, n_shape
            )
            return res

        p0 = np.clip(np.concatenate([[S_seed], b_seed]), lo + 1e-12, hi - 1e-12)
        sol = least_squares(fun, p0, bounds=(lo, hi), max_nfev=max_nfev,
                            xtol=1e-14, ftol=1e-14, gtol=1e-14)
        _c_direct, r_direct = solver_mod._assemble(
            targets, knots, n_coeff, degree, lin, n_shape, sol.x
        )
        assert not r_direct.feasible  # the cold start genuinely struggles here

        # --- full solve_icw: continuation must rescue it ---
        curve, report = solve_icw(targets, n_coeff=n_coeff, max_nfev=max_nfev)
        assert report.feasible, report.violations
        s = curve.sample(4000)
        assert aperture_report(s).n_pi2_crossings == 1
        assert is_monotone_radius(s)

    def test_infeasible_target_message_points_right_direction(self):
        """P1-4: a genuinely impossible target (r_mouth < r0) reports feasible=False AND the
        suggested_relaxation / message point the right way (widen the mouth above the throat).
        """
        targets = ICWTargets(
            mode=TerminationMode.FLAT_BAFFLE,
            r0=50.0, theta0_deg=18, x_target=120, r_mouth=30,
        )
        _curve, report = solve_icw(targets)
        assert report.feasible is False
        assert report.violations
        assert report.suggested_relaxation is not None
        # The hint must mention r_mouth and the corrective direction (increase / above r0).
        assert "r_mouth" in report.suggested_relaxation
        hint_text = (
            " ".join(report.suggested_relaxation.values()) + " " + report.message
        ).lower()
        assert "r_mouth" in hint_text
        assert ("increase" in hint_text) or ("above" in hint_text) or ("wider" in hint_text)


# =====================================================================================
# Regression: review fixes (2026-06-15)
# =====================================================================================
class TestReviewFixes:
    def test_theta90_plateau_to_end_counts_once(self):
        """A theta=90deg plateau running to the final node is ONE crossing, not two.

        The trailing-node check used to double-count a plateau the main scan had already
        consumed (e.g. an all-90deg meridian, or a wall that grazes the mouth plane along a flat
        run to the end). Over-counting wrongly trips the rollback multi-crossing/wobble guard.
        """
        from hornlab_mesher.icw.core import theta_half_pi_crossings

        hp = np.pi / 2.0
        assert len(theta_half_pi_crossings(np.array([hp, hp, hp]))) == 1
        assert len(theta_half_pi_crossings(np.array([0.0, np.pi / 4, hp, hp]))) == 1
        # A single exact last-node hit (no plateau) still counts once.
        assert len(theta_half_pi_crossings(np.array([0.0, np.pi / 4, hp]))) == 1
        # A genuine wobble (up through 90, back below, up again) still counts every crossing.
        wob = np.array([0.0, 0.4 * hp, 1.2 * hp, 0.8 * hp, 1.2 * hp])
        assert len(theta_half_pi_crossings(wob)) == 3

    def test_direct_mode_rejects_nonphysical_inputs(self):
        """DIRECT mode (icw_coeffs) bypasses solve_icw's feasibility gate, so it must validate
        its own inputs rather than build a curve with a non-finite or non-positive radius."""
        from hornlab_mesher.profile_formulas import build_icw_curve

        good = {"type": "ICW", "r0": 12.7, "icw_coeffs": [0, 0, 0, 0, 0, 0], "icw_S": 100.0}
        build_icw_curve(good)  # baseline: a straight (kappa==0) curve builds fine

        bad_cases = [
            {**good, "r0": -1.0},  # negative throat radius (input check)
            {**good, "icw_S": 0.0},  # zero arc length (input check)
            {**good, "icw_coeffs": [0, float("nan"), 0, 0, 0, 0]},  # non-finite coeff (input check)
            # Valid inputs, but a downward throat angle drives the sampled radius negative
            # (r = r0 + S*int sin(theta) with theta == -90deg) -> the sampled-radius guard fires:
            {"type": "ICW", "r0": 12.7, "a0": -90.0, "icw_coeffs": [0, 0, 0, 0, 0, 0], "icw_S": 100.0},
        ]
        for bad in bad_cases:
            with pytest.raises(ValueError):
                build_icw_curve(bad)

    def test_top_level_depth_does_not_enclose_icw(self):
        """A bare top-level ``depth`` is an ICW rollback target, not enclosure depth, so it must
        not coerce a free-standing rollback ICW into enclosure mode. Non-ICW formulas keep the
        historical top-level bare-``depth`` -> enclosure fallback, and an explicit enclosure
        section still encloses an ICW build."""
        from hornlab_mesher.config_builder import _normalise_mode

        icw_rollback = {"formula": "ICW", "termination": "rollback", "r0": 12.7, "depth": 100.0}
        assert _normalise_mode(icw_rollback, {}, {}, formula="ICW") == "freestanding"
        # Back-compat: a non-ICW config with a bare top-level depth still encloses.
        assert _normalise_mode({"depth": 100.0}, {}, {}, formula="OSSE") == "enclosure"
        # An explicit enclosure section still encloses an ICW build.
        assert _normalise_mode({"formula": "ICW"}, {}, {"depth": 300.0}, formula="ICW") == "enclosure"
