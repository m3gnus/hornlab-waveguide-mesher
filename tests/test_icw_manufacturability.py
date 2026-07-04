"""Solve-time manufacturability/acoustic bounds for the ICW solver."""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.icw import ICWTargets, TerminationMode, curve_from_shape_modes, solve_icw


_BASE_COEFFS_PRE_M4 = np.array(
    [
        4.5925330686827055e-21,
        -2.0134833830402817e-03,
        -1.4729816510788887e-03,
        7.8516347108998838e-04,
        4.2332937524590924e-03,
        7.7025432015209675e-03,
        1.1148309905466277e-02,
        1.4426214718804297e-02,
        1.7283805335761460e-02,
        1.4355412115769108e-02,
        1.5935014794370208e-17,
        1.5754525620000017e-17,
    ]
)
_BASE_KAPPA_PEAK_PRE_M4 = 0.01631958095110688
_BASE_DKAPPA_DS_PEAK_PRE_M4 = 0.13533348335495657
_DEMAND_KAPPA_PEAK_PRE_M4 = 0.031055560453208984
_DEMAND_DKAPPA_DS_PEAK_PRE_M4 = 0.28066651882126054
_SHAPE0_KAPPA_PEAK_PRE_M4 = 0.016163466651497472


def _base_targets(**kwargs) -> ICWTargets:
    return ICWTargets(
        mode=TerminationMode.FLAT_BAFFLE,
        r0=12.7,
        theta0_deg=18.0,
        x_target=120.0,
        r_mouth=116.0,
        **kwargs,
    )


def _demanding_targets(**kwargs) -> ICWTargets:
    return ICWTargets(
        mode=TerminationMode.FLAT_BAFFLE,
        r0=12.7,
        theta0_deg=18.0,
        x_target=120.0,
        r_mouth=70.0,
        **kwargs,
    )


def _sample_peaks(curve) -> tuple[float, float, float]:
    sample = curve.sample(4000)
    dkappa_ds = curve.kappa_spline()(sample.sigma, nu=1)
    return (
        float(np.max(np.abs(sample.kappa))),
        float(np.max(np.abs(dkappa_ds))),
        float(np.degrees(np.max(sample.theta))),
    )


def _has_violation(report, needle: str) -> bool:
    return any(needle in violation for violation in report.violations)


def test_no_bound_regression_matches_pre_m4_coefficients() -> None:
    """Leaving all M4 bounds unset must not perturb the existing solve."""
    targets = _base_targets()
    assert targets.kappa_abs_max is None
    assert targets.dkappa_ds_abs_max is None
    assert targets.theta_max_deg is None

    curve, report = solve_icw(targets)
    assert report.feasible, report.violations
    assert curve.coeffs.shape == _BASE_COEFFS_PRE_M4.shape
    assert np.max(np.abs(curve.coeffs - _BASE_COEFFS_PRE_M4)) < 1e-12


def test_generous_curvature_and_slope_caps_are_inert() -> None:
    """Caps far above the observed curve add no violation and keep the same-ish curve."""
    targets = _base_targets(
        kappa_abs_max=10.0 * _BASE_KAPPA_PEAK_PRE_M4,
        dkappa_ds_abs_max=10.0 * _BASE_DKAPPA_DS_PEAK_PRE_M4,
    )
    curve, report = solve_icw(targets)
    assert report.feasible, report.violations
    assert not _has_violation(report, "max|kappa|")
    assert not _has_violation(report, "max|dkappa/dsigma|")
    assert report.residuals["kappa_abs_peak"] <= targets.kappa_abs_max
    assert report.residuals["dkappa_ds_peak"] <= targets.dkappa_ds_abs_max
    assert np.max(np.abs(curve.coeffs - _BASE_COEFFS_PRE_M4)) < 1e-6


def test_tight_curvature_cap_reports_consistently() -> None:
    """A tight cap either produces a respecting curve or truthfully names the kappa miss."""
    cap = 0.5 * _DEMAND_KAPPA_PEAK_PRE_M4
    curve, report = solve_icw(
        _demanding_targets(kappa_abs_max=cap), n_coeff=12, max_nfev=40
    )
    measured, _dkappa, _theta = _sample_peaks(curve)
    if report.feasible:
        assert measured <= cap * (1.0 + 1e-6) + 1e-12
    else:
        assert _has_violation(report, "max|kappa|")
        assert report.residuals["kappa_abs_peak"] > cap


def test_tight_dkappa_ds_cap_reports_consistently() -> None:
    """A tight curvature-slope cap follows the same truthful reporting contract."""
    cap = 0.5 * _DEMAND_DKAPPA_DS_PEAK_PRE_M4
    curve, report = solve_icw(
        _demanding_targets(dkappa_ds_abs_max=cap), n_coeff=12, max_nfev=40
    )
    _kappa, measured, _theta = _sample_peaks(curve)
    if report.feasible:
        assert measured <= cap * (1.0 + 1e-6) + 1e-12
    else:
        assert _has_violation(report, "max|dkappa/dsigma|")
        assert report.residuals["dkappa_ds_peak"] > cap


def test_rollback_theta_cap_blocks_or_passes_by_peak_angle() -> None:
    bad = ICWTargets(
        mode=TerminationMode.ROLLBACK,
        r0=12.7,
        theta0_deg=12.0,
        theta1_deg=160.0,
        r_aperture=70.0,
        x_setback=6.0,
        theta_max_deg=140.0,
    )
    _curve_bad, report_bad = solve_icw(bad, n_coeff=12, max_nfev=40)
    assert not report_bad.feasible
    assert _has_violation(report_bad, "theta peak")
    assert report_bad.residuals["theta_peak_deg"] > 140.0 + 1e-3
    assert "theta_max_deg" in report_bad.suggested_relaxation

    good = ICWTargets(
        mode=TerminationMode.ROLLBACK,
        r0=12.7,
        theta0_deg=12.0,
        theta1_deg=160.0,
        r_aperture=70.0,
        x_setback=6.0,
        theta_max_deg=170.0,
    )
    _curve_good, report_good = solve_icw(good, n_coeff=12, max_nfev=80)
    assert report_good.feasible, report_good.violations
    assert report_good.residuals["theta_peak_deg"] <= 170.0 + 1e-3


def test_flat_baffle_theta_cap_is_inert() -> None:
    curve, report = solve_icw(_base_targets(theta_max_deg=91.0))
    assert report.feasible, report.violations
    assert not _has_violation(report, "theta peak")
    assert report.residuals["theta_peak_deg"] == pytest.approx(90.0, abs=1e-9)
    assert _sample_peaks(curve)[2] == pytest.approx(90.0, abs=1e-9)


def test_shape_mode_entry_point_uses_same_curvature_cap() -> None:
    """curve_from_shape_modes threads the same targets into the residual barriers."""
    cap = 0.5 * _SHAPE0_KAPPA_PEAK_PRE_M4
    curve, report = curve_from_shape_modes(
        np.zeros(0), _base_targets(kappa_abs_max=cap), n_coeff=12, max_nfev=80
    )
    measured, _dkappa, _theta = _sample_peaks(curve)
    if report.feasible:
        assert measured <= cap * (1.0 + 1e-6) + 1e-12
    else:
        assert _has_violation(report, "max|kappa|")
        assert report.residuals["kappa_abs_peak"] > cap


def test_manufacturability_bound_validation() -> None:
    with pytest.raises(ValueError, match="kappa_abs_max"):
        _base_targets(kappa_abs_max=0.0)
    with pytest.raises(ValueError, match="dkappa_ds_abs_max"):
        _base_targets(dkappa_ds_abs_max=-1.0)
    with pytest.raises(ValueError, match="theta_max_deg"):
        _base_targets(theta_max_deg=90.0)
    with pytest.raises(ValueError, match="theta_max_deg"):
        _base_targets(theta_max_deg=181.0)
