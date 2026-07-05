"""Mesher adapter/config threading for ICW coverage and manufacturability targets."""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.profile_formulas import build_icw_curve
from hornlab_mesher.profiles import profile_points


BASE_COVERAGE_PARAMS = {
    "type": "ICW",
    "r0": 12.7,
    "a0": 14.5,
    "termination": "flat_baffle",
    "L": 160.0,
    "R": 130.0,
    "coverage_angle": 50.0,
    "hold_start": 0.30,
    "hold_end": 0.70,
}


def _assert_coverage_plateau(curve, angle_deg: float, hold_start: float = 0.30, hold_end: float = 0.70) -> None:
    sample = curve.sample(2001)
    hs_i = int(round(hold_start * 2000))
    he_i = int(round(hold_end * 2000))

    assert np.max(np.abs(curve.kappa(np.linspace(hold_start, hold_end, 501)))) < 1.0e-6
    assert np.allclose(np.degrees(sample.theta[hs_i : he_i + 1]), angle_deg, atol=1.0e-3)
    assert np.degrees(sample.theta[-1]) == pytest.approx(90.0, abs=1.0e-3)


def test_build_icw_curve_threads_coverage_with_emergent_mouth_radius() -> None:
    curve = build_icw_curve(BASE_COVERAGE_PARAMS)
    sample = curve.sample(2001)

    _assert_coverage_plateau(curve, 50.0)
    assert sample.x[-1] == pytest.approx(160.0, abs=1.0e-3)
    assert sample.r[-1] != pytest.approx(BASE_COVERAGE_PARAMS["R"], abs=0.05)


def test_coverage_angle_zero_is_off_sentinel_and_pins_length_and_radius() -> None:
    off_params = {
        "type": "ICW",
        "r0": 12.7,
        "a0": 18.0,
        "termination": "flat_baffle",
        "L": 120.0,
        "R": 110.0,
        "coverage_angle": 0.0,
    }
    curve_off = build_icw_curve(off_params)
    curve_plain = build_icw_curve({k: v for k, v in off_params.items() if k != "coverage_angle"})
    sample = curve_off.sample(2001)

    assert curve_off.coeffs.shape == curve_plain.coeffs.shape
    assert np.max(np.abs(curve_off.coeffs - curve_plain.coeffs)) < 1.0e-12
    assert curve_off.S == pytest.approx(curve_plain.S, abs=1.0e-12)
    assert sample.x[-1] == pytest.approx(120.0, abs=1.0e-3)
    assert sample.r[-1] == pytest.approx(110.0, abs=1.0e-3)


def test_coverage_honors_explicit_finer_basis_request() -> None:
    curve = build_icw_curve({**BASE_COVERAGE_PARAMS, "n_coeff": 20})

    assert curve.coeffs.size == 20
    _assert_coverage_plateau(curve, 50.0)


@pytest.mark.parametrize("n_coeff", [6, 8])
def test_coverage_ignores_sub_floor_n_coeff_and_uses_coverage_default(n_coeff: int) -> None:
    # The WG UI ships n_coeff=6 as the plain-ICW default and materialises it into
    # every payload. Coverage needs n_coeff >= degree+6 = 9, so forwarding a sub-floor value
    # used to make the coverage-on happy path raise ("coverage knots need ... got 6")
    # and 422 in the viewport. A sub-floor n_coeff under coverage must instead defer
    # to the kernel's coverage-aware default (16) and build a real plateau.
    curve = build_icw_curve({**BASE_COVERAGE_PARAMS, "n_coeff": n_coeff})

    assert curve.coeffs.size == 16
    _assert_coverage_plateau(curve, 50.0)


def test_negative_coverage_angle_raises() -> None:
    with pytest.raises(ValueError, match="coverage_angle must be non-negative"):
        build_icw_curve({**BASE_COVERAGE_PARAMS, "coverage_angle": -1.0})


def test_non_coverage_still_honors_small_n_coeff() -> None:
    # The sub-floor guard must NOT touch the non-coverage path: a plain ICW build with
    # n_coeff=6 (no coverage) is honoured exactly as before.
    curve = build_icw_curve(
        {"type": "ICW", "r0": 12.7, "a0": 14.5, "termination": "flat_baffle",
         "L": 160.0, "R": 130.0, "n_coeff": 6}
    )

    assert curve.coeffs.size == 6


def test_pin_mouth_radius_with_impossible_coverage_target_surfaces_infeasible() -> None:
    params = {**BASE_COVERAGE_PARAMS, "R": 5.0, "pin_mouth_radius": True}

    with pytest.raises(ValueError, match="infeasible"):
        build_icw_curve(params)


def test_target_mode_threads_curvature_cap_and_reports_infeasible_or_ok() -> None:
    base = {
        "type": "ICW",
        "r0": 12.7,
        "a0": 18.0,
        "termination": "flat_baffle",
        "L": 120.0,
        "R": 116.0,
    }

    with pytest.raises(ValueError, match=r"max\|kappa\|"):
        build_icw_curve({**base, "kappa_abs_max": 0.01})

    curve = build_icw_curve({**base, "kappa_abs_max": 0.10})
    sample = curve.sample(2001)
    assert sample.x[-1] == pytest.approx(120.0, abs=1.0e-3)
    assert sample.r[-1] == pytest.approx(116.0, abs=1.0e-3)


def test_icw_curve_cache_keys_on_coverage_angle() -> None:
    curve_40 = build_icw_curve({**BASE_COVERAGE_PARAMS, "coverage_angle": 40.0})
    curve_60 = build_icw_curve({**BASE_COVERAGE_PARAMS, "coverage_angle": 60.0})

    assert curve_40 is not curve_60
    assert curve_40.sample(1001).r[-1] != pytest.approx(curve_60.sample(1001).r[-1], abs=1.0)


def test_config_builder_forwards_coverage_and_manufacturability_keys() -> None:
    config = {
        "profile": {
            "formula": "ICW",
            "r0_mm": 12.7,
            "a0_deg": 14.5,
            "termination": "flat_baffle",
            "L_mm": 160.0,
            "R_mm": 130.0,
            "coverage_angle": 50.0,
            "hold_start": 0.30,
            "hold_end": 0.70,
            "kappa_abs_max": 1.0,
            "dkappa_ds_abs_max": 10.0,
            "theta_max_deg": 91.0,
            "pin_mouth_radius": "false",
        },
        "mesh": {},
    }

    params, formula, _mode = build_geometry_params(config)

    assert formula == "ICW"
    assert params["coverage_angle"] == 50.0
    assert params["hold_start"] == 0.30
    assert params["hold_end"] == 0.70
    assert params["kappa_abs_max"] == 1.0
    assert params["dkappa_ds_abs_max"] == 10.0
    assert params["theta_max_deg"] == 91.0
    assert params["pin_mouth_radius"] is False


def test_osse_and_rosse_config_paths_remain_unchanged() -> None:
    configs = [
        {
            "profile": {
                "formula": "OSSE",
                "r0_mm": 12.7,
                "a0_deg": 18.0,
                "a_deg": 35.0,
                "L_mm": 120.0,
                "s": 0.8,
                "n": 4.0,
                "q": 0.9,
            },
            "mesh": {},
        },
        {
            "profile": {
                "formula": "R-OSSE",
                "r0_mm": 12.7,
                "a0_deg": 15.5,
                "a_deg": 60.0,
                "R_mm": 130.0,
                "m": 0.85,
                "r": 0.4,
                "b": 0.2,
                "q": 1.0,
            },
            "mesh": {},
        },
    ]

    for config in configs:
        params, formula, _mode = build_geometry_params(config)
        pts = profile_points(params, 48)

        assert formula in {"OSSE", "R-OSSE"}
        assert params["type"] == formula
        assert "coverage_angle" not in params
        assert np.all(np.isfinite(pts))
        assert pts[-1, 1] > pts[0, 1]
