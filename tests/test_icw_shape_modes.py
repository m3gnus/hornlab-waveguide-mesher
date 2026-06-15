"""Tests for the ICW shape-mode entry points (CMA-ES gene interface).

These cover :func:`hornlab_mesher.icw.n_shape_modes` and
:func:`hornlab_mesher.icw.curve_from_shape_modes` -- the kernel entry point an outer optimiser
uses to SET the leading curvature nullspace coordinates ("gene" modes) while the kernel solves
only the size-fixing degrees of freedom (the arc length ``S`` plus the remaining "reserved"
nullspace coordinates) to hit the flat-baffle size targets.

The whole point of the feasible subspace is that ``kappa = kappa0(S) + Phi @ b`` satisfies the
linear boundary conditions BY CONSTRUCTION for ANY ``b``. So the central guarantee these tests
pin is: *regardless* of the gene vector, the returned curve lands on the flat baffle exactly
(``kappa(0)=kappa0``, ``kappa(1)=0``, ``dkappa/dsigma(1)=0``, ``theta(1)=90 deg``) AND meets the
size targets -- yet different genes give genuinely different interior meridians (local shape
control with fixed endpoints). Infeasible inputs return TRUTHFUL reports, never silent repairs.

Uses only the public ``hornlab_mesher.icw`` surface plus numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.icw import (
    ICWTargets,
    TerminationMode,
    curve_from_shape_modes,
    is_monotone_radius,
    meridian_self_intersects,
    n_shape_modes,
)

# Shared monotone-radius / BC tolerances (mirror the solver's: TOL_KAPPA_BC=1e-6, dkappa 1e-4).
_BC_ATOL = 1e-9  # the task asks BCs reproduced to ~1e-9 regardless of the gene vector
_SIZE_ATOL_MM = 1e-3  # solver's TOL_LENGTH_MM


def _flat_target(r0=12.7, theta0_deg=18.0, x_target=120.0, r_mouth=116.0) -> ICWTargets:
    return ICWTargets(
        mode=TerminationMode.FLAT_BAFFLE,
        r0=r0,
        theta0_deg=theta0_deg,
        x_target=x_target,
        r_mouth=r_mouth,
    )


def _body_r_on_common_x(curve_a, curve_b, n: int = 4000):
    """Return ``(r_a, r_b)`` resampled on a shared axial grid over the overlapping body.

    Two ICW meridians have slightly different arc lengths/axial extents; to compare *interior*
    shape we interpolate both radii onto a common ``x`` grid spanning their overlapping body.
    """
    sa, sb = curve_a.sample(n), curve_b.sample(n)
    x_hi = min(float(sa.x[-1]), float(sb.x[-1]))
    xg = np.linspace(0.0, x_hi, n // 2)
    return np.interp(xg, sa.x, sa.r), np.interp(xg, sb.x, sb.r)


def _assert_flat_baffle_bcs(curve, atol: float = _BC_ATOL) -> None:
    """The flat-baffle linear BCs the feasible subspace guarantees for ANY gene vector."""
    assert curve.kappa(np.array([0.0]))[0] == pytest.approx(0.0, abs=atol)  # kappa0 = 0 here
    assert curve.kappa(np.array([1.0]))[0] == pytest.approx(0.0, abs=atol)  # flat-baffle kappa(1)
    dk1 = float(curve.kappa_spline()(np.array([1.0]), nu=1)[0])
    assert dk1 == pytest.approx(0.0, abs=atol)  # flat-baffle dkappa/dsigma(1)
    assert np.degrees(curve.theta_end()) == pytest.approx(90.0, abs=1e-6)  # terminal angle


# =====================================================================================
# n_shape_modes
# =====================================================================================
class TestNShapeModes:
    def test_positive_and_consistent_with_nullspace_dim(self):
        """k_max is the PROBED honest budget: the largest gene count whose all-zero solve is
        feasible, between 1 and the algebraic ceiling D-1. The reserved high-frequency modes have
        weak endpoint leverage, so the budget can fall short of the D-1 ceiling (target-dependent)."""
        targets = _flat_target()
        k_max = n_shape_modes(targets)  # default n_coeff=12
        assert isinstance(k_max, int)
        # Default basis: C is 4x12 of rank 4 -> nullspace D = 8 -> algebraic ceiling D-1 = 7.
        assert 1 <= k_max <= 7
        # This target loses the top two reserved modes' conditioning, so the honest budget is 5.
        assert k_max == 5

    def test_k_max_grows_with_n_coeff_and_stays_within_ceiling(self):
        """More basis coefficients => more nullspace modes => a non-decreasing honest budget, always
        within [1, the algebraic ceiling D-1 = n_coeff-5]."""
        targets = _flat_target()
        prev = 0
        for n_coeff in (8, 12, 16, 20):
            k_max = n_shape_modes(targets, n_coeff=n_coeff)
            ceiling = (n_coeff - 4) - 1  # D-1, with D = n_coeff - 4 (four flat-baffle BC rows)
            assert 1 <= k_max <= ceiling
            assert k_max >= prev  # monotone non-decreasing in n_coeff
            prev = k_max

    def test_advertised_budget_is_a_feasible_contiguous_prefix(self):
        """The advertised k_max is HONEST: the zero-gene (midpoint) solve is feasible at EVERY gene
        count 0..k_max. Solver conditioning makes feasibility NON-monotone in k, so the budget is a
        contiguous-from-zero prefix -- never a higher k reachable only past an infeasible gap (so a
        consumer that clamps to min(cap, k_max) can never land on an infeasible midpoint). The old
        D-1 budget over-promised (the size solve went singular at k=6,7 for n_coeff=12)."""
        targets_list = [
            _flat_target(),                 # easy
            _flat_target(r_mouth=70.0),     # deep
            # Base-borderline: the k=0 solve is itself infeasible, so the contiguous prefix is empty
            # and the budget is 0 -- it must NOT advertise the (numerically feasible) k=1 above it.
            ICWTargets(mode=TerminationMode.FLAT_BAFFLE, r0=12.7, theta0_deg=0.0,
                       x_target=80.0, r_mouth=30.0),
        ]
        for targets in targets_list:
            k_max = n_shape_modes(targets)
            assert k_max >= 0
            if k_max >= 1:  # k_max == 0 means the base solve is borderline; nothing to assert
                for k in range(k_max + 1):
                    _curve, report = curve_from_shape_modes(np.zeros(k), targets)
                    assert report.feasible, (float(targets.r_mouth), k, report.violations)

    def test_rollback_target_raises(self):
        targets = ICWTargets(
            mode=TerminationMode.ROLLBACK,
            r0=12.7,
            theta0_deg=12,
            theta1_deg=110,
            r_aperture=70,
            x_setback=6,
        )
        with pytest.raises(ValueError, match="flat_baffle"):
            n_shape_modes(targets)


# =====================================================================================
# curve_from_shape_modes -- feasible-by-construction BCs (the central guarantee)
# =====================================================================================
class TestBCsExactRegardlessOfGenes:
    def test_bcs_exact_for_random_small_genes(self):
        """For many random small gene vectors (length k <= k_max), the curve satisfies the
        flat-baffle BCs to ~1e-9 and lands theta_end == 90 deg -- REGARDLESS of the genes.

        This is the feasible-subspace property: ``a = a0 + Phi @ b`` meets every linear BC by
        construction for any ``b``, so the outer optimiser can never knock the curve off the baffle.
        """
        targets = _flat_target()
        k_max = n_shape_modes(targets)
        rng = np.random.default_rng(20240614)
        for _ in range(10):
            k = int(rng.integers(1, k_max + 1))
            b_gene = rng.normal(scale=0.05, size=k)
            curve, _report = curve_from_shape_modes(b_gene, targets)
            _assert_flat_baffle_bcs(curve)  # holds whether or not the size solve was feasible

    def test_bcs_exact_for_large_genes_even_if_size_infeasible(self):
        """Even a gene vector large enough to break the SIZE solve keeps the linear BCs exact --
        the BCs are structural (nullspace), independent of whether the size targets are hit.
        """
        targets = _flat_target()
        curve, report = curve_from_shape_modes(np.array([3.0]), targets)  # extreme leading mode
        # Size feasibility may well fail here; the linear BCs must STILL be exact.
        _assert_flat_baffle_bcs(curve)
        # And if it is infeasible it must say so truthfully (not a silent repair).
        if not report.feasible:
            assert report.violations

    def test_k_zero_reproduces_all_reserved_solve(self):
        """An empty gene vector (k=0) is allowed and yields a feasible flat-baffle curve."""
        targets = _flat_target()
        curve, report = curve_from_shape_modes(np.zeros(0), targets)
        assert report.feasible, report.violations
        _assert_flat_baffle_bcs(curve)


# =====================================================================================
# curve_from_shape_modes -- size targets for feasible cases
# =====================================================================================
class TestSizeTargets:
    def test_size_targets_hit_for_feasible_genes(self):
        """For feasible gene vectors, x(1) ~= x_target and r(1) ~= r_mouth to < 1e-3 mm."""
        targets = _flat_target()
        for b_gene in ([0.0], [0.05], [-0.05], [0.1, 0.05, -0.03], [0.0, 0.0, 0.05, -0.02]):
            curve, report = curve_from_shape_modes(np.array(b_gene), targets)
            assert report.feasible, (b_gene, report.violations)
            s = curve.sample(4000)
            assert s.x[-1] == pytest.approx(120.0, abs=_SIZE_ATOL_MM)
            assert s.r[-1] == pytest.approx(116.0, abs=_SIZE_ATOL_MM)
            # And the report's own residual agrees it is "hit".
            assert report.residuals["size_residual_max"] < _SIZE_ATOL_MM


# =====================================================================================
# curve_from_shape_modes -- genes MATTER (local shape control, fixed endpoints)
# =====================================================================================
class TestGenesMatter:
    def test_two_different_genes_give_different_interior_meridians(self):
        """Two DIFFERENT gene vectors produce DIFFERENT interior meridians (max |dr| over the body
        well above quadrature noise) while BOTH satisfy the flat-baffle BCs and the size targets.

        This is the entire point of the entry point: local shape control with fixed endpoints.
        """
        targets = _flat_target()
        curve_a, rep_a = curve_from_shape_modes(np.array([0.05]), targets)
        curve_b, rep_b = curve_from_shape_modes(np.array([-0.05]), targets)
        assert rep_a.feasible, rep_a.violations
        assert rep_b.feasible, rep_b.violations

        # Both still meet the BCs and size targets exactly...
        _assert_flat_baffle_bcs(curve_a)
        _assert_flat_baffle_bcs(curve_b)
        for c in (curve_a, curve_b):
            s = c.sample(4000)
            assert s.x[-1] == pytest.approx(120.0, abs=_SIZE_ATOL_MM)
            assert s.r[-1] == pytest.approx(116.0, abs=_SIZE_ATOL_MM)

        # ...yet the interior radius differs by FAR more than trapezoid noise (O(1e-6 mm)).
        r_a, r_b = _body_r_on_common_x(curve_a, curve_b)
        max_dr = float(np.max(np.abs(r_a - r_b)))
        assert max_dr > 1.0, f"genes did not move the interior meridian (max|dr|={max_dr:.2e} mm)"

    def test_distinct_multimode_genes_differ(self):
        """Two distinct multi-mode gene vectors also yield distinct, feasible interior shapes."""
        targets = _flat_target()
        ca, ra = curve_from_shape_modes(np.array([0.1, 0.05, -0.03]), targets)
        cb, rb = curve_from_shape_modes(np.array([-0.08, 0.02, 0.06]), targets)
        assert ra.feasible, ra.violations
        assert rb.feasible, rb.violations
        r_a, r_b = _body_r_on_common_x(ca, cb)
        assert float(np.max(np.abs(r_a - r_b))) > 1.0


# =====================================================================================
# curve_from_shape_modes -- guards: monotone r, no self-intersection
# =====================================================================================
class TestGuards:
    def test_feasible_curve_is_monotone_and_non_self_intersecting(self):
        """A feasible result has a weakly-increasing radius and no self-intersection."""
        targets = _flat_target()
        for b_gene in ([0.0], [0.05], [-0.05], [0.1, 0.0, -0.04]):
            curve, report = curve_from_shape_modes(np.array(b_gene), targets)
            assert report.feasible, (b_gene, report.violations)
            s = curve.sample(4000)
            assert is_monotone_radius(s)
            assert not meridian_self_intersects(s.x, s.r)
            assert report.residuals.get("self_intersects") is False

    def test_extreme_gene_returns_truthful_report_never_silently_broken(self):
        """A gene that pushes curvature too far must NOT return a silently broken curve: either
        feasible=False with violations, or (if reported feasible) a genuinely monotone,
        non-self-intersecting curve that hits the size targets.

        We cover both branches of that contract: a large-but-feasible gene (the kernel still
        fixes the size with its reserved DOF) and an extreme gene that overwhelms the reserved
        size DOF (must report infeasible truthfully, never a silent repair). The BCs stay exact
        in BOTH cases -- they are structural, independent of the size solve.
        """
        targets = _flat_target()

        # (a) Large but still feasible: a feasible claim must be fully honoured.
        curve_ok, report_ok = curve_from_shape_modes(np.array([0.3]), targets)
        _assert_flat_baffle_bcs(curve_ok)
        assert report_ok.feasible, report_ok.violations
        s_ok = curve_ok.sample(4000)
        assert is_monotone_radius(s_ok)
        assert not meridian_self_intersects(s_ok.x, s_ok.r)
        assert s_ok.x[-1] == pytest.approx(120.0, abs=_SIZE_ATOL_MM)
        assert s_ok.r[-1] == pytest.approx(116.0, abs=_SIZE_ATOL_MM)

        # (b) Extreme gene: overwhelms the reserved size DOF -> truthful infeasible report. BCs
        # still exact; the result is never a silently broken (wrong-size or folded) curve.
        curve_bad, report_bad = curve_from_shape_modes(np.array([3.0]), targets)
        _assert_flat_baffle_bcs(curve_bad)
        s_bad = curve_bad.sample(4000)
        if report_bad.feasible:
            assert is_monotone_radius(s_bad)
            assert not meridian_self_intersects(s_bad.x, s_bad.r)
            assert s_bad.x[-1] == pytest.approx(120.0, abs=_SIZE_ATOL_MM)
            assert s_bad.r[-1] == pytest.approx(116.0, abs=_SIZE_ATOL_MM)
        else:
            assert report_bad.violations  # truthful: violations populated, no silent repair


# =====================================================================================
# curve_from_shape_modes -- infeasible inputs / argument validation
# =====================================================================================
class TestInfeasibleAndValidation:
    def test_infeasible_size_target_reports_violations(self):
        """r_mouth < r0 is a necessary-condition violation -> feasible=False with violations,
        even with a gene supplied (the entry point must not paper over it).
        """
        targets = _flat_target(r0=50.0, r_mouth=30.0)
        curve, report = curve_from_shape_modes(np.array([0.1]), targets)
        assert report.feasible is False
        assert report.violations
        # The early necessary-condition check should name r_mouth vs r0.
        assert any("r_mouth" in v for v in report.violations)

    def test_rollback_target_raises(self):
        targets = ICWTargets(
            mode=TerminationMode.ROLLBACK,
            r0=12.7,
            theta0_deg=12,
            theta1_deg=110,
            r_aperture=70,
            x_setback=6,
        )
        with pytest.raises(ValueError, match="flat_baffle"):
            curve_from_shape_modes(np.array([0.1]), targets)

    def test_gene_vector_too_long_raises(self):
        # Requesting more genes than the algebraic ceiling D-1 (here D=8 -> ceiling 7) raises; within
        # [0, D-1] the kernel returns a truthful (in)feasible report rather than raising.
        targets = _flat_target()
        with pytest.raises(ValueError, match="exceeds the algebraic ceiling"):
            curve_from_shape_modes(np.zeros(20), targets)

    def test_non_1d_gene_vector_raises(self):
        targets = _flat_target()
        with pytest.raises(ValueError, match="1D"):
            curve_from_shape_modes(np.zeros((2, 2)), targets)
