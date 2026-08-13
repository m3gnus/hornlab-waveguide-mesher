from __future__ import annotations

import math

import numpy as np
import pytest

from hornlab_mesher.profiles import (
    _angle_list,
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


def _base_rosse_params() -> dict[str, float | int | str]:
    return {
        "type": "R-OSSE",
        "R": 150.0,
        "r0": 12.7,
        "a0": 12.0,
        "a": 40.0,
        "k": 1.0,
        "r": 0.3,
        "m": 0.8,
        "b": 0.3,
        "q": 0.995,
        "tmax": 1.0,
        "angularSegments": 8,
        "lengthSegments": 4,
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


def test_superellipse_n2_reaches_exact_ellipse_mouth():
    params = {
        **_base_osse_params(),
        "morphTarget": 3,
        "morphWidth": 400.0,
        "morphHeight": 300.0,
        "morphExponent": 2.0,
        "morphAllowShrinkage": 1,
    }
    inner, _ = _inner_grid(params)
    mouth = inner[:, -1]

    assert np.max(np.abs(mouth[:, 0])) == pytest.approx(200.0, abs=1.0e-12)
    assert np.max(np.abs(mouth[:, 1])) == pytest.approx(150.0, abs=1.0e-12)
    diagonal = mouth[np.argmin(np.abs(np.arctan2(mouth[:, 1], mouth[:, 0]) - math.pi / 4.0))]
    assert (diagonal[0] / 200.0) ** 2 + (diagonal[1] / 150.0) ** 2 == pytest.approx(
        1.0, abs=1.0e-12
    )


def test_superellipse_n2_satisfies_ellipse_equation_at_several_azimuths():
    params = {
        "morphTarget": 3,
        "morphWidth": 400.0,
        "morphHeight": 300.0,
        "morphExponent": 2.0,
    }
    azimuths = np.asarray([0.0, 0.13, 0.37, math.pi / 4.0, 1.1, math.pi / 2.0])
    radii = np.asarray(
        [_morph_target_radius_at_angle(75.0, float(phi), params) for phi in azimuths]
    )
    ellipse_equation = (
        (radii * np.cos(azimuths) / 200.0) ** 2
        + (radii * np.sin(azimuths) / 150.0) ** 2
    )

    np.testing.assert_allclose(
        ellipse_equation,
        np.ones_like(ellipse_equation),
        rtol=0.0,
        atol=2.0 * np.finfo(np.float64).eps,
    )


def test_superellipse_n2_with_equal_dimensions_matches_circle_target():
    params = {
        **_base_osse_params(),
        "morphWidth": 400.0,
        "morphHeight": 400.0,
        "morphAllowShrinkage": 1,
    }
    superellipse, _ = _inner_grid(
        {**params, "morphTarget": 3, "morphExponent": 2.0}
    )
    circle, _ = _inner_grid({**params, "morphTarget": 2})

    assert np.array_equal(superellipse, circle)


def test_superellipse_exponent_increases_diagonal_without_leaving_box():
    phi = math.pi / 4.0
    radii = []
    for exponent in (2.0, 6.0, 16.0):
        radius = _morph_target_radius_at_angle(
            75.0,
            phi,
            {
                "morphTarget": 3,
                "morphWidth": 400.0,
                "morphHeight": 300.0,
                "morphExponent": exponent,
            },
        )
        radii.append(radius)
        assert abs(radius * math.cos(phi)) <= 200.0 + 1.0e-12
        assert abs(radius * math.sin(phi)) <= 150.0 + 1.0e-12

    assert radii[0] < radii[1] < radii[2]


@pytest.mark.parametrize("exponent", [2.0, 4.0, 8.0, 16.0])
def test_superellipse_axes_are_exact_across_exponents(exponent):
    params = {
        "morphTarget": 3,
        "morphWidth": 400.0,
        "morphHeight": 300.0,
        "morphExponent": exponent,
    }

    assert _morph_target_radius_at_angle(75.0, 0.0, params) == 200.0
    assert _morph_target_radius_at_angle(75.0, math.pi / 2.0, params) == 150.0


def test_superellipse_cardinal_snap_is_relative_at_extreme_aspect_ratio():
    phi = 5.0e-10
    half_width = 1.0e12
    half_height = 1.0
    exponent = 4.0
    params = {
        "morphTarget": 3,
        "morphWidth": 2.0 * half_width,
        "morphHeight": 2.0 * half_height,
        "morphExponent": exponent,
    }
    expected = (
        (abs(math.cos(phi)) / half_width) ** exponent
        + (abs(math.sin(phi)) / half_height) ** exponent
    ) ** (-1.0 / exponent)

    assert _morph_target_radius_at_angle(1.0, phi, params) == expected
    assert expected != half_width


def test_superellipse_exponent_is_expression_capable_and_clamped():
    base = {
        "morphTarget": 3,
        "morphWidth": 400.0,
        "morphHeight": 300.0,
    }
    phi = math.pi / 4.0
    n2 = _morph_target_radius_at_angle(75.0, phi, {**base, "morphExponent": 2.0})
    n16 = _morph_target_radius_at_angle(75.0, phi, {**base, "morphExponent": 16.0})

    assert _morph_target_radius_at_angle(
        75.0, phi, {**base, "morphExponent": -10.0}
    ) == n2
    assert _morph_target_radius_at_angle(
        75.0, phi, {**base, "morphExponent": 100.0}
    ) == n16
    assert _morph_target_radius_at_angle(
        75.0,
        phi,
        {**base, "morphExponent": "2 + 14*sin(p)^2"},
    ) == pytest.approx(
        _morph_target_radius_at_angle(
            75.0,
            phi,
            {**base, "morphExponent": 9.0},
        ),
        rel=0.0,
        abs=1.0e-12,
    )


def test_guiding_curve_superformula_accepts_expression_params():
    phi = 0.41
    params = {
        "gcurveType": 2,
        "gcurveWidth": "75 + 5*cos(p)",
        "gcurveAspectRatio": "1 + 0.2*sin(p)",
        "gcurveSfA": "1 + 0.1*sin(p)",
        "gcurveSfB": "1.2 - 0.1*cos(p)",
        "gcurveSfM1": "6 + 2*cos(p)",
        "gcurveSfM2": "5 + sin(p)",
        "gcurveSfN1": "1.5 + 0.2*cos(p)",
        "gcurveSfN2": "2 + 0.1*sin(p)",
        "gcurveSfN3": "2.5 + 0.1*cos(p)",
    }

    width = 75 + 5 * math.cos(phi)
    aspect = 1 + 0.2 * math.sin(phi)
    sf_a = 1 + 0.1 * math.sin(phi)
    sf_b = 1.2 - 0.1 * math.cos(phi)
    sf_m1 = 6 + 2 * math.cos(phi)
    sf_m2 = 5 + math.sin(phi)
    sf_n1 = 1.5 + 0.2 * math.cos(phi)
    sf_n2 = 2 + 0.1 * math.sin(phi)
    sf_n3 = 2.5 + 0.1 * math.cos(phi)
    t1 = abs(math.cos((sf_m1 * phi) / 4.0) / sf_a) ** sf_n2
    t2 = abs(math.sin((sf_m2 * phi) / 4.0) / sf_b) ** sf_n3
    r_norm = (t1 + t2) ** (-1.0 / sf_n1)
    expected = math.hypot(
        r_norm * math.cos(phi) * width / 2.0,
        r_norm * math.sin(phi) * width * aspect / 2.0,
    )

    assert _guiding_curve_target_radius(phi, params) == expected


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


def test_stadium_morph_corner_uses_distinct_azimuth_meridians():
    for width, height in ((400.0, 240.0), (240.0, 400.0)):
        params = {
            **_base_osse_params(),
            "angularSegments": 80,
            "morphTarget": 1,
            "morphWidth": width,
            "morphHeight": height,
            "morphCorner": 120.0,
            "morphAllowShrinkage": 1,
        }

        angles, closed = _angle_list(params)

        assert closed
        assert len(angles) == 80
        assert np.all(np.diff(angles) > 1.0e-12)

        inner, _ = _inner_grid(params)
        assert len(np.unique(inner[:, -1, :2], axis=0)) == len(angles)


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


def test_superellipse_implicit_extents_are_exact_while_rectangle_still_ceils():
    base = {
        **_base_osse_params(),
        "morphWidth": 0.0,
        "morphHeight": 0.0,
        "morphAllowShrinkage": 1,
    }
    raw, _ = _inner_grid({**base, "morphTarget": 0})
    superellipse, _ = _inner_grid(
        {**base, "morphTarget": 3, "morphExponent": 2.0}
    )
    rectangle, _ = _inner_grid({**base, "morphTarget": 1, "morphCorner": 0.0})
    raw_half_width = float(np.max(np.abs(raw[:, -1, 0])))
    raw_half_height = float(np.max(np.abs(raw[:, -1, 1])))

    assert np.max(np.abs(superellipse[:, -1, 0])) == pytest.approx(
        raw_half_width, abs=1.0e-12
    )
    assert np.max(np.abs(superellipse[:, -1, 1])) == pytest.approx(
        raw_half_height, abs=1.0e-12
    )
    assert np.max(np.abs(rectangle[:, -1, 0])) == math.ceil(
        raw_half_width - 1.0e-9
    )
    assert np.max(np.abs(rectangle[:, -1, 1])) == math.ceil(
        raw_half_height - 1.0e-9
    )


def test_rosse_tmax_morph_reaches_typed_target():
    params = {
        **_base_rosse_params(),
        "tmax": 0.7,
        "morphTarget": 1,
        "morphWidth": 400.0,
        "morphHeight": 300.0,
        "morphCorner": 0.0,
        "morphRate": 3.0,
        "morphFixed": 0.0,
        "morphAllowShrinkage": 1,
    }
    inner, _ = _inner_grid(params)
    mouth = inner[:, -1]

    # Before progress normalisation, this landed at 156.07 / 138.92 mm.
    assert np.max(np.abs(mouth[:, 0])) == pytest.approx(200.0, abs=1.0e-12)
    assert np.max(np.abs(mouth[:, 1])) == pytest.approx(150.0, abs=1.0e-12)


def test_rosse_tmax_one_morph_grid_is_unchanged():
    params = {
        **_base_rosse_params(),
        "angularSegments": 4,
        "lengthSegments": 2,
        "morphTarget": 1,
        "morphWidth": 400.0,
        "morphHeight": 300.0,
        "morphCorner": 0.0,
        "morphRate": 3.0,
        "morphFixed": 0.0,
        "morphAllowShrinkage": 1,
    }
    actual, _ = _inner_grid(params)
    expected = np.asarray(
        [
            [
                [12.7, 0.0, 0.0],
                [112.65926870803122, 0.0, 68.52814036023145],
                [200.0, 0.0, 60.263885256450806],
            ],
            [
                [7.776507174585692e-16, 12.7, 0.0],
                [6.515688516145052e-15, 106.40926870803122, 68.52814036023145],
                [9.184850993605149e-15, 150.0, 60.263885256450806],
            ],
            [
                [-12.7, 1.5553014349171384e-15, 0.0],
                [-112.65926870803122, 1.3796781281757201e-14, 68.52814036023145],
                [-200.0, 2.4492935982947064e-14, 60.263885256450806],
            ],
            [
                [-2.3329521523757076e-15, -12.7, 0.0],
                [-1.9547065548435155e-14, -106.40926870803122, 68.52814036023145],
                [-2.7554552980815446e-14, -150.0, 60.263885256450806],
            ],
        ],
        dtype=np.float64,
    )

    assert np.array_equal(actual, expected)


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
