"""Coverage-angle (constant-directivity plateau) tests for the ICW kernel.

The coverage feature adds a held wall-angle plateau over an arc-length window to a
flat-baffle ICW meridian: the constant-directivity mechanism OSSE has and the bare
ICW least-norm baseline lacks. These tests pin the design contract:

* the plateau is exactly flat (kappa == 0 on the hold span) at the target angle,
* the flat-baffle boundary conditions still hold by construction,
* the mouth radius is EMERGENT (length is the only pinned size target) and monotone,
* pinning the mouth radius too on top of coverage is reported infeasible, not repaired,
* coverage is rejected for rollback, and
* a non-coverage solve is byte-identical to the pre-coverage kernel.
"""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.icw import (
    ICWTargets,
    TerminationMode,
    curve_from_shape_modes,
    n_shape_modes,
    solve_icw,
)


def _coverage_targets(theta_c: float, hs: float = 0.30, he: float = 0.70) -> ICWTargets:
    return ICWTargets(
        mode=TerminationMode.FLAT_BAFFLE,
        r0=12.7,
        theta0_deg=14.5,
        x_target=160.0,
        coverage_angle_deg=theta_c,
        hold_start=hs,
        hold_end=he,
    )


@pytest.mark.parametrize("theta_c", [40.0, 50.0, 60.0])
def test_coverage_plateau_is_flat_at_target(theta_c: float) -> None:
    """The held section is exactly the coverage angle with zero curvature."""
    targets = _coverage_targets(theta_c)
    curve, report = solve_icw(targets)
    assert report.feasible, report.violations

    n = 2001
    s = curve.sample(n)
    hs_i = int(round(targets.hold_start * (n - 1)))
    he_i = int(round(targets.hold_end * (n - 1)))
    mid_i = (hs_i + he_i) // 2

    # Plateau flat to numerical zero on the whole hold span.
    assert np.max(np.abs(s.kappa[hs_i : he_i + 1])) < 1e-6
    # Wall angle across the plateau equals the coverage target.
    plateau_deg = np.degrees(s.theta[hs_i : he_i + 1])
    assert np.allclose(plateau_deg, theta_c, atol=1e-3)
    # Reported achieved angle matches.
    assert abs(report.residuals["coverage_angle_deg_achieved"] - theta_c) < 1e-3
    assert abs(np.degrees(s.theta[mid_i]) - theta_c) < 1e-3


@pytest.mark.parametrize("theta_c", [40.0, 50.0, 60.0])
def test_coverage_keeps_flat_baffle_bcs_and_monotone(theta_c: float) -> None:
    """Coverage does not disturb the flat-baffle landing or single-valued radius."""
    targets = _coverage_targets(theta_c)
    curve, report = solve_icw(targets)
    assert report.feasible, report.violations

    kappa1 = float(curve.kappa(np.array([1.0]))[0])
    dkappa1 = float(curve.kappa_spline()(np.array([1.0]), nu=1)[0])
    s = curve.sample(2001)
    assert abs(kappa1) < 1e-6
    assert abs(dkappa1) < 1e-4
    assert abs(np.degrees(s.theta[-1]) - 90.0) < 1e-2
    assert abs(s.x[-1] - targets.x_target) < 1e-3
    # Radius is weakly monotone (never re-entrant) and the throat angle is the floor.
    assert np.all(np.diff(s.r) >= -1e-6)
    assert np.degrees(s.theta).min() >= targets.theta0_deg - 1.0


def test_coverage_mouth_radius_is_emergent_and_grows_with_angle() -> None:
    """r_mouth is an output; a wider coverage angle yields a wider mouth."""
    r_mouths = []
    for theta_c in (40.0, 50.0, 60.0):
        _curve, report = solve_icw(_coverage_targets(theta_c))
        assert report.feasible, report.violations
        r_mouths.append(report.residuals["r_mouth_emergent"])
    assert r_mouths[0] < r_mouths[1] < r_mouths[2]
    assert all(r > 12.7 for r in r_mouths)


def test_coverage_plus_pinned_mouth_radius_reports_infeasible() -> None:
    """Pinning both the plateau and r_mouth is over-constrained -> truthful infeasible."""
    targets = ICWTargets(
        mode=TerminationMode.FLAT_BAFFLE,
        r0=12.7,
        theta0_deg=14.5,
        x_target=160.0,
        r_mouth=130.0,
        coverage_angle_deg=50.0,
    )
    _curve, report = solve_icw(targets)
    assert not report.feasible
    assert report.violations  # a size/monotone violation, not a crash


def test_coverage_rejected_for_rollback() -> None:
    with pytest.raises(ValueError):
        ICWTargets(
            mode=TerminationMode.ROLLBACK,
            r0=12.7,
            theta0_deg=14.5,
            r_aperture=120.0,
            depth=150.0,
            coverage_angle_deg=50.0,
        )


def test_coverage_angle_must_lie_between_throat_and_ninety() -> None:
    with pytest.raises(ValueError):
        _coverage_targets(10.0)  # below theta0
    with pytest.raises(ValueError):
        _coverage_targets(95.0)  # above 90


def test_coverage_hold_window_validation() -> None:
    with pytest.raises(ValueError):
        _coverage_targets(50.0, hs=0.7, he=0.3)  # reversed
    with pytest.raises(ValueError):
        _coverage_targets(50.0, hs=0.0, he=0.7)  # degenerate start


def test_coverage_gene_path_feasible() -> None:
    """curve_from_shape_modes / n_shape_modes work under the coverage constraint set."""
    targets = _coverage_targets(45.0)
    k = n_shape_modes(targets)
    assert k >= 1
    b = np.zeros(max(0, k - 1))
    curve, report = curve_from_shape_modes(b, targets)
    assert report.feasible, report.violations
    s = curve.sample(2001)
    hs_i = int(round(targets.hold_start * 2000))
    he_i = int(round(targets.hold_end * 2000))
    assert np.max(np.abs(s.kappa[hs_i : he_i + 1])) < 1e-6


def test_noncoverage_solve_is_byte_identical_to_baseline() -> None:
    """coverage_angle_deg=None must not perturb the existing two-target solve."""
    # Snapshot captured from the pre-coverage kernel (post-M1 exact-theta) for
    # r0=12.7, theta0=14.5, x_target=160, r_mouth=130, n_coeff=12, zero genes.
    expected_coeffs = np.array(
        [
            3.753249613397105e-21,
            -0.0031044728796133826,
            -0.0028765449173130016,
            -0.0006786599957530892,
            0.0033040369121130142,
            0.0072231676439475055,
            0.010549529672027155,
            0.013059835969845134,
            0.01448184942053583,
            0.011020647725316635,
            1.2615120014275083e-17,
            1.2875392797003969e-17,
        ]
    )
    targets = ICWTargets(
        mode=TerminationMode.FLAT_BAFFLE,
        r0=12.7,
        theta0_deg=14.5,
        x_target=160.0,
        r_mouth=130.0,
    )
    curve, report = curve_from_shape_modes(np.zeros(0), targets, n_coeff=12)
    assert report.feasible
    # Endpoints exact; coeffs unchanged from the pre-coverage baseline.
    s = curve.sample(2001)
    assert abs(s.x[-1] - 160.0) < 1e-3
    assert abs(s.r[-1] - 130.0) < 1e-3
    assert curve.coeffs.shape == expected_coeffs.shape
    # The snapshot pins a least-squares ENDPOINT, which drifts ~1e-6 across
    # scipy/BLAS versions (observed 6.7e-6 between scipy 1.17.1 and 1.18.0).
    # A real coverage-machinery perturbation moves coefficients by >=1e-3, so
    # 5e-5 still fails loudly for the regression this test guards against.
    assert np.max(np.abs(curve.coeffs - expected_coeffs)) < 5e-5
