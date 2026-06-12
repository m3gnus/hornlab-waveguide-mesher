from __future__ import annotations

import math

import numpy as np

from hornlab_mesher.profiles import (
    _apply_morphing,
    _guiding_curve_target_radius,
    _morph_target_radius_at_angle,
    _rounded_rect_radius,
    build_point_grid,
    calculate_osse,
)


def _base_osse_params() -> dict[str, float | int | str]:
    return {
        "type": "OSSE",
        "L": 100.0,
        "r0": 10.0,
        "a": 35.0,
        "a0": 8.0,
        "k": 1.0,
        "n": 4.0,
        "q": 0.995,
        "s": 0.0,
        "angularSegments": 16,
        "lengthSegments": 20,
        "wallThickness": 0.0,
        "quadrants": "1234",
    }


def _inner_grid(params: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    grid = build_point_grid(params)
    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    inner = np.asarray(grid["inner_points"], dtype=np.float64).reshape(n_phi, n_length + 1, 3)
    return inner, np.asarray(grid["slice_map"], dtype=np.float64)


def _radii(points: np.ndarray) -> np.ndarray:
    return np.hypot(points[..., 0], points[..., 1])


def test_circle_morph_target_is_constant_radius_not_area_surrogate():
    params = {
        "morphTarget": 2,
        "morphWidth": 240.0,
        "morphHeight": 60.0,
    }

    assert math.isclose(_morph_target_radius_at_angle(75.0, 0.0, params), 120.0)
    assert math.isclose(_morph_target_radius_at_angle(75.0, math.pi / 3.0, params), 120.0)


def test_circle_morph_zero_dimension_preserves_available_raw_dimension():
    params = {
        "morphTarget": 2,
        "morphWidth": 240.0,
        "morphHeight": 0.0,
    }

    assert math.isclose(_morph_target_radius_at_angle(75.0, math.pi / 4.0, params), 120.0)
    assert math.isclose(
        _morph_target_radius_at_angle(
            75.0,
            math.pi / 4.0,
            {"morphTarget": 2, "morphWidth": 0.0, "morphHeight": 0.0},
        ),
        75.0,
    )


def test_morph_fixed_part_is_unchanged_before_transition():
    raw, _ = _inner_grid(_base_osse_params())
    morphed, t_values = _inner_grid(
        {
            **_base_osse_params(),
            "morphTarget": 2,
            "morphWidth": 300.0,
            "morphHeight": 300.0,
            "morphFixed": 0.5,
            "morphAllowShrinkage": 1,
        }
    )

    fixed_stop = int(np.searchsorted(t_values, 0.5, side="left"))
    assert np.allclose(morphed[:, : fixed_stop + 1], raw[:, : fixed_stop + 1], rtol=0.0, atol=1.0e-9)


def test_osse_shrink_morph_respects_fixed_part():
    raw, _ = _inner_grid(_base_osse_params())
    morphed, t_values = _inner_grid(
        {
            **_base_osse_params(),
            "morphTarget": 2,
            "morphWidth": 20.0,
            "morphHeight": 20.0,
            "morphFixed": 0.5,
            "morphAllowShrinkage": 1,
        }
    )

    fixed_stop = int(np.searchsorted(t_values, 0.5, side="left"))
    assert np.allclose(morphed[:, : fixed_stop + 1], raw[:, : fixed_stop + 1], rtol=0.0, atol=1.0e-9)


def test_rectangle_morph_mouth_reaches_directional_target():
    params = {
        **_base_osse_params(),
        "morphTarget": 1,
        "morphWidth": 260.0,
        "morphHeight": 160.0,
        "morphCorner": 0.0,
        "morphAllowShrinkage": 1,
    }
    morphed, _ = _inner_grid(params)
    angles = np.arctan2(morphed[:, -1, 1], morphed[:, -1, 0])
    mouth_radii = _radii(morphed[:, -1])
    expected = np.asarray(
        [_rounded_rect_radius(float(phi), 130.0, 80.0, 0.0) for phi in angles],
        dtype=np.float64,
    )

    assert np.allclose(mouth_radii, expected, rtol=0.0, atol=1.0e-9)


def test_rectangle_morph_uses_ceiled_implicit_extents_when_dimensions_omitted():
    params = {
        **_base_osse_params(),
        "morphTarget": 1,
        "morphWidth": 0.0,
        "morphHeight": 0.0,
        "morphCorner": 12.0,
        "morphAllowShrinkage": 1,
    }
    raw, _ = _inner_grid({**params, "morphTarget": 0})
    morphed, _ = _inner_grid(params)

    # ATH derives implicit target dimensions by rounding the raw mouth
    # extents up to whole millimetres per half-dimension.
    raw_mouth = raw[:, -1]
    half_width = float(math.ceil(np.max(np.abs(raw_mouth[:, 0])) - 1.0e-9))
    half_height = float(math.ceil(np.max(np.abs(raw_mouth[:, 1])) - 1.0e-9))
    angles = np.arctan2(morphed[:, -1, 1], morphed[:, -1, 0])
    mouth_radii = _radii(morphed[:, -1])
    expected = np.asarray(
        [_rounded_rect_radius(float(phi), half_width, half_height, 12.0) for phi in angles],
        dtype=np.float64,
    )

    assert np.allclose(mouth_radii, expected, rtol=0.0, atol=1.0e-9)


def test_morph_does_not_shrink_without_explicit_permission():
    raw, _ = _inner_grid(_base_osse_params())
    morphed, _ = _inner_grid(
        {
            **_base_osse_params(),
            "morphTarget": 2,
            "morphWidth": 20.0,
            "morphHeight": 20.0,
        }
    )

    assert np.all(_radii(morphed) >= _radii(raw) - 1.0e-9)


def test_zero_morph_dimensions_preserve_raw_mouth_dimensions_for_interior_slices():
    params = {
        "morphTarget": 1,
        "morphWidth": 0.0,
        "morphHeight": 0.0,
        "morphRate": 1.0,
        "morphFixed": 0.0,
        "morphAllowShrinkage": 1,
    }

    assert math.isclose(
        _apply_morphing(
            50.0,
            100.0,
            0.5,
            0.0,
            params,
            implicit_half_width=100.0,
            implicit_half_height=80.0,
        ),
        50.0,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )


def test_guiding_curve_inverts_coverage_so_profile_passes_through_curve():
    params = {
        **_base_osse_params(),
        "L": 120.0,
        "a": 25.0,
        "a0": 8.0,
        "gcurveType": 1,
        "gcurveWidth": 120.0,
        "gcurveAspectRatio": 0.6,
        "gcurveDist": 0.5,
        "gcurveSeN": 4.0,
    }
    target_z = 60.0

    for phi in (0.0, math.pi / 4.0, math.pi / 2.0):
        _z, radius = calculate_osse(target_z, phi, params)
        assert math.isclose(radius, _guiding_curve_target_radius(phi, params), rel_tol=0.0, abs_tol=1.0e-4)
