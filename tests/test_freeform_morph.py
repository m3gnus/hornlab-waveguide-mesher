"""Analytic smooth-target morphing for FREEFORM profiles."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.freeform import build_freeform_geometry
from hornlab_mesher.profile_morph import _rounded_rect_radius
from hornlab_mesher.profiles import build_point_grid


def _straight_params() -> dict:
    return {
        "type": "FREEFORM",
        "profileH": {
            "points": [[0.0, 10.0], [100.0, 30.0]],
            "throatAngleDeg": 0.0,
            "mouthAngleDeg": 0.0,
        },
        "profileV": {
            "points": [[0.0, 10.0], [100.0, 20.0]],
            "throatAngleDeg": 0.0,
            "mouthAngleDeg": 0.0,
        },
        "crossSections": [
            {"t": 0.0, "shape": "circle"},
            {"t": 1.0, "shape": "superellipse", "exponent": 4.0},
        ],
        "angularSegments": 8,
        "lengthSegments": 2,
        "samplingMode": "uniform",
        "quadrants": "1",
    }


def _config_with_morph(target: object) -> dict:
    params = _straight_params()
    return {
        "formula": "FREEFORM",
        "mode": "bare",
        "profile": {
            "profileH": params["profileH"],
            "profileV": params["profileV"],
            "crossSections": params["crossSections"],
        },
        "morph": {"morphTarget": target},
        "mesh": {
            "angularSegments": 8,
            "lengthSegments": 2,
            "samplingMode": "uniform",
            "quadrants": "1",
        },
    }


def test_unmorphed_freeform_grid_is_bit_for_bit_unchanged() -> None:
    params = _straight_params()
    params["crossSections"][-1] = {"t": 1.0, "shape": "ellipse"}
    actual = np.asarray(build_point_grid(params)["inner_points"]).reshape(3, 3, 3)
    expected = np.asarray(
        [
            [
                [9.9999999999999982, 0.0, 0.0],
                [19.999999999999996, 0.0, 50.0],
                [30.0, 0.0, 100.0],
            ],
            [
                [7.0710678118654755, 7.0710678118654746, 0.0],
                [12.000000000000002, 12.0, 50.0],
                [16.641005886756876, 16.641005886756872, 100.0],
            ],
            [
                [6.1232339957367653e-16, 9.9999999999999982, 0.0],
                [9.1848509936051499e-16, 15.0, 50.0],
                [1.2246467991473531e-15, 19.999999999999996, 100.0],
            ],
        ]
    )
    assert np.array_equal(actual, expected)
    assert np.array_equal(
        np.asarray(
            build_point_grid({**params, "morphTarget": 0})["inner_points"]
        ).reshape(3, 3, 3),
        expected,
    )


def test_shape_only_superellipse_morph_preserves_drawn_mouth_axes() -> None:
    params = {
        **_straight_params(),
        "morphTarget": 3,
        "morphExponent": 2.0,
    }
    geometry = build_freeform_geometry(params)
    phi = np.asarray([0.0, math.pi / 4.0, math.pi / 2.0])
    base = build_freeform_geometry(
        {key: value for key, value in params.items() if not key.startswith("morph")}
    )
    actual = geometry.cross_section_radius(phi, 1.0)

    assert actual[0] == 30.0
    assert actual[2] == 20.0
    assert actual[1] < base.cross_section_radius(phi, 1.0)[1]
    assert actual[1] == pytest.approx(
        ((math.cos(math.pi / 4.0) / 30.0) ** 2
         + (math.sin(math.pi / 4.0) / 20.0) ** 2) ** -0.5,
        abs=1.0e-12,
    )


def test_typed_superellipse_dimensions_are_exact() -> None:
    geometry = build_freeform_geometry(
        {
            **_straight_params(),
            "morphTarget": 3,
            "morphExponent": 4.0,
            "morphWidth": 80.0,
            "morphHeight": 60.0,
        }
    )
    actual = geometry.cross_section_radius(
        np.asarray([0.0, math.pi / 2.0]), 1.0
    )
    assert np.array_equal(actual, np.asarray([40.0, 30.0]))


def test_scalar_morph_fast_path_matches_expression_fallback_and_keeps_axes_exact() -> None:
    params = {
        **_straight_params(),
        "morphTarget": 3,
        "morphExponent": 4.0,
        "morphWidth": 80.0,
        "morphHeight": 60.0,
        "morphRate": 2.0,
        "morphFixed": 0.15,
        "morphAllowShrinkage": 1,
    }
    fast = build_freeform_geometry(params)
    fallback = build_freeform_geometry({**params, "morphRate": "2.0"})
    phi = np.linspace(0.0, math.pi / 2.0, 257)

    for t in (0.15, 0.53, 1.0):
        np.testing.assert_allclose(
            fast.cross_section_radius(phi, t),
            fallback.cross_section_radius(phi, t),
            rtol=0.0,
            atol=1.0e-12,
        )

    assert np.array_equal(
        fast.cross_section_radius(np.asarray([0.0, math.pi / 2.0]), 1.0),
        np.asarray([40.0, 30.0]),
    )


def test_expression_morph_shape_params_match_equivalent_scalars() -> None:
    scalar_params = {
        **_straight_params(),
        "morphTarget": 3,
        "morphWidth": 80.0,
        "morphHeight": 60.0,
        "morphExponent": 4.0,
        "morphRate": 2.0,
        "morphAllowShrinkage": 1,
    }
    expression_params = {
        **scalar_params,
        "morphWidth": "80 + 0*p",
        "morphExponent": "4 + 0*p",
        "morphRate": "2 + 0*p",
    }
    scalar = build_freeform_geometry(scalar_params)
    fallback = build_freeform_geometry(expression_params)
    phi = np.linspace(0.0, math.pi / 2.0, 257)

    for t in (0.0, 0.37, 1.0):
        np.testing.assert_allclose(
            fallback.cross_section_radius(phi, t),
            scalar.cross_section_radius(phi, t),
            rtol=0.0,
            atol=1.0e-12,
        )


def test_no_shrinkage_floors_typed_dimensions_at_drawn_mouth() -> None:
    geometry = build_freeform_geometry(
        {
            **_straight_params(),
            "morphTarget": 3,
            "morphExponent": 2.0,
            "morphWidth": 2.0,
            "morphHeight": 2.0,
        }
    )
    actual = geometry.cross_section_radius(
        np.asarray([0.0, math.pi / 2.0]), 1.0
    )
    assert np.array_equal(actual, np.asarray([30.0, 20.0]))


def test_circle_target_produces_a_circular_freeform_mouth() -> None:
    grid = build_point_grid({**_straight_params(), "morphTarget": 2})
    points = np.asarray(grid["inner_points"]).reshape(
        grid["grid_n_phi"], grid["grid_n_length"] + 1, 3
    )
    mouth_radii = np.linalg.norm(points[:, -1, :2], axis=1)
    np.testing.assert_allclose(mouth_radii, 30.0, rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize("target", [2, 3])
def test_config_builder_accepts_smooth_freeform_morphs(target: int) -> None:
    params, formula, _mode = build_geometry_params(_config_with_morph(target))
    assert formula == "FREEFORM"
    assert params["morphTarget"] == target


def test_config_builder_accepts_rectangle_freeform_morph() -> None:
    params, formula, _mode = build_geometry_params(_config_with_morph(1))
    assert formula == "FREEFORM"
    assert params["morphTarget"] == 1


def _ellipse_rectangle_morph_params(*, corner: float = 6.0) -> dict:
    params = _straight_params()
    params["crossSections"] = [
        {"t": 0.0, "shape": "ellipse"},
        {"t": 1.0, "shape": "ellipse"},
    ]
    params.update(
        angularSegments=32,
        lengthSegments=8,
        morphTarget=1,
        morphWidth=80.0,
        morphHeight=50.0,
        morphCorner=corner,
    )
    return params


def test_rectangle_morph_mouth_matches_typed_rounded_rectangle() -> None:
    params = _ellipse_rectangle_morph_params()
    grid = build_point_grid(params)
    phi_grid = np.asarray(grid["phi_grid"], dtype=np.float64)
    points = np.asarray(grid["inner_points"], dtype=np.float64).reshape(
        grid["grid_n_phi"], grid["grid_n_length"] + 1, 3
    )
    mouth_phi = phi_grid[:, -1]
    mouth_radii = np.linalg.norm(points[:, -1, :2], axis=1)
    expected = np.asarray(
        [_rounded_rect_radius(phi, 40.0, 25.0, 6.0) for phi in mouth_phi]
    )

    assert np.array_equal(mouth_radii[[0, -1]], np.asarray([40.0, 25.0]))
    np.testing.assert_allclose(mouth_radii, expected, rtol=0.0, atol=1.0e-12)


def test_rectangle_morph_sampler_tracks_emerging_corner_tangencies() -> None:
    grid = build_point_grid(_ellipse_rectangle_morph_params())
    phi_grid = np.asarray(grid["phi_grid"], dtype=np.float64)
    theta_1 = math.atan2(25.0 - 6.0, 40.0)
    theta_2 = math.atan2(25.0, 40.0 - 6.0)

    for tangent in (theta_1, theta_2):
        assert np.any(np.isclose(phi_grid[:, -1], tangent, rtol=0.0, atol=1.0e-14))
        assert not np.any(
            np.isclose(phi_grid[:, 0], tangent, rtol=0.0, atol=1.0e-14)
        )
    assert grid["freeform_corner_arc_spans"][0] == []
    np.testing.assert_allclose(
        grid["freeform_corner_arc_spans"][-1],
        [theta_1, theta_2],
        rtol=0.0,
        atol=1.0e-14,
    )


def test_sharp_rectangle_morph_is_rejected_as_non_convex() -> None:
    """The convexity guard rejects the singular radial blend before meshing."""

    with pytest.raises(ValueError) as error:
        build_freeform_geometry(_ellipse_rectangle_morph_params(corner=0.0))
    message = str(error.value)

    assert "morph to the rectangle target" in message
    assert "morphCorner" in message
    assert "crossSections" not in message


def test_nonconvex_station_with_active_morph_is_attributed_to_cross_sections() -> None:
    params = _straight_params()
    params["crossSections"] = [
        {"t": 0.0, "shape": "circle"},
        {
            "t": 1.0,
            "shape": "rounded_rectangle",
            "cornerRadiusMm": 0.5,
        },
    ]
    params.update(morphTarget=1, morphCorner=15.0, morphFixed=0.9)

    with pytest.raises(ValueError) as error:
        build_freeform_geometry(params)

    assert str(error.value).startswith(
        "FREEFORM crossSections span 0..1 produces a non-convex outline"
    )


def test_later_station_defect_wins_over_earlier_morph_failure() -> None:
    params = _straight_params()
    params["crossSections"] = [
        {"t": 0.0, "shape": "circle"},
        {
            "t": 1.0,
            "shape": "rounded_rectangle",
            "cornerRadiusMm": 0.5,
        },
    ]
    params.update(
        morphTarget=2,
        morphWidth=2.0,
        morphHeight=2.0,
        morphRate=1.0,
        morphAllowShrinkage=1,
    )

    with pytest.raises(ValueError) as error:
        build_freeform_geometry(params)

    message = str(error.value)
    assert message.startswith(
        "FREEFORM crossSections span 0..1 produces a non-convex outline"
    )
    assert "t=0.59375" in message
    assert "minimum feasible corner radius here is ~0.4 mm" in message
    assert "morph to" not in message


def test_rectangle_morph_corner_15_still_builds() -> None:
    assert (
        build_freeform_geometry(
            _ellipse_rectangle_morph_params(corner=15.0)
        ).length_mm
        == 100.0
    )


def test_walled_rectangle_morph_builds_without_outer_offset_fold() -> None:
    grid = build_point_grid(
        {**_ellipse_rectangle_morph_params(), "wallThickness": 3.0}
    )
    assert grid["outer_points"] is not None


def test_expression_target_is_still_rejected() -> None:
    with pytest.raises(ConfigError, match="cannot be proven inactive"):
        build_geometry_params(
            _config_with_morph("1 if 0.01 < p < 0.02 else 0")
        )


@pytest.mark.parametrize("target", [4, 7, -1, 99])
def test_out_of_range_static_morph_targets_are_rejected(target: int) -> None:
    message = r"morphTarget.*valid values 0, 1, 2, or 3"
    with pytest.raises(ConfigError, match=message):
        build_geometry_params(_config_with_morph(target))
    with pytest.raises(ValueError, match=message):
        build_point_grid({**_straight_params(), "morphTarget": target})


def test_expression_target_keeps_non_freeform_runtime_path() -> None:
    params = {
        "type": "OSSE",
        "L": 100.0,
        "r0": 10.0,
        "a": 35.0,
        "a0": 8.0,
        "angularSegments": 8,
        "lengthSegments": 2,
        "quadrants": "1",
        "morphTarget": "4 if p > 10 else 0",
    }

    assert build_point_grid(params)["inner_points"]


def test_active_morph_quadrant_one_matches_full_circle_quadrant() -> None:
    params = _ellipse_rectangle_morph_params(corner=15.0)
    full = build_point_grid({**params, "quadrants": "1234"})
    quarter = build_point_grid({**params, "quadrants": "1"})
    full_points = np.asarray(full["inner_points"], dtype=np.float64).reshape(
        full["grid_n_phi"], full["grid_n_length"] + 1, 3
    )
    quarter_points = np.asarray(quarter["inner_points"], dtype=np.float64).reshape(
        quarter["grid_n_phi"], quarter["grid_n_length"] + 1, 3
    )
    full_phi = np.asarray(full["phi_grid"], dtype=np.float64)
    quarter_phi = np.asarray(quarter["phi_grid"], dtype=np.float64)

    selected = np.all(full_phi <= math.pi / 2.0 + 1.0e-14, axis=1)
    np.testing.assert_allclose(
        quarter_points, full_points[selected], rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        quarter_phi, full_phi[selected], rtol=0.0, atol=1.0e-14
    )


def test_vertical_offset_combined_with_active_morph_preserves_symmetry() -> None:
    grid = build_point_grid(
        {
            **_ellipse_rectangle_morph_params(corner=15.0),
            "quadrants": "1234",
            "verticalOffset": 12.0,
        }
    )
    points = np.asarray(grid["inner_points"], dtype=np.float64).reshape(
        grid["grid_n_phi"], grid["grid_n_length"] + 1, 3
    )
    mouth = points[:, -1, :2]

    assert grid["vertical_offset_mm"] == 12.0
    assert np.max(mouth[:, 0]) == -np.min(mouth[:, 0])
    assert np.max(mouth[:, 1]) == -np.min(mouth[:, 1])


@pytest.mark.parametrize(
    ("morph", "expected"),
    [
        (
            {"morphTarget": 2},
            [
                [
                    [9.999999999999998, 0.0, 0.0],
                    [20.0, 0.0, 50.0],
                    [30.0, 0.0, 100.0],
                ],
                [
                    [7.0710678118654755, 7.071067811865475, 0.0],
                    [13.263678907805145, 13.263678907805144, 50.0],
                    [21.213203435596427, 21.213203435596423, 100.0],
                ],
                [
                    [6.123233995736765e-16, 9.999999999999998, 0.0],
                    [9.950255243072245e-16, 16.25, 50.0],
                    [1.83697019872103e-15, 30.0, 100.0],
                ],
            ],
        ),
        (
            {"morphTarget": 3, "morphExponent": 3.0},
            [
                [
                    [9.999999999999998, 0.0, 0.0],
                    [20.0, 0.0, 50.0],
                    [30.0, 0.0, 100.0],
                ],
                [
                    [7.0710678118654755, 7.071067811865475, 0.0],
                    [12.904858793852192, 12.904858793852188, 50.0],
                    [18.34264252397279, 18.34264252397279, 100.0],
                ],
                [
                    [6.123233995736765e-16, 9.999999999999998, 0.0],
                    [9.18485099360515e-16, 15.0, 50.0],
                    [1.2246467991473533e-15, 20.0, 100.0],
                ],
            ],
        ),
    ],
)
def test_smooth_morph_grids_are_bit_for_bit_unchanged(
    morph: dict, expected: list
) -> None:
    actual = np.asarray(
        build_point_grid({**_straight_params(), **morph})["inner_points"]
    ).reshape(3, 3, 3)
    assert np.array_equal(actual, np.asarray(expected))


def test_rounded_rectangle_station_grid_is_bit_for_bit_unchanged() -> None:
    params = {
        **_straight_params(),
        "profileH": {
            "points": [[0.0, 10.0], [100.0, 20.0]],
            "throatAngleDeg": 0.0,
            "mouthAngleDeg": 0.0,
        },
        "profileV": {
            "points": [[0.0, 10.0], [100.0, 15.0]],
            "throatAngleDeg": 0.0,
            "mouthAngleDeg": 0.0,
        },
        "crossSections": [
            {"t": 0.0, "shape": "ellipse"},
            {"t": 1.0, "shape": "rounded_rectangle", "cornerRadiusMm": 5.0},
        ],
    }
    expected = np.asarray(
        [
            [
                [9.999999999999998, 0.0, 0.0],
                [15.0, 0.0, 50.0],
                [20.0, 0.0, 100.0],
            ],
            [
                [8.94427190999916, 4.47213595499958, 0.0],
                [13.93119694284408, 6.96559847142204, 50.0],
                [20.0, 10.0, 100.0],
            ],
            [
                [7.79403831193579, 6.265218814381276, 0.0],
                [12.915215846401743, 9.012631799667837, 50.0],
                [19.330127018922198, 12.500000000000002, 100.0],
            ],
            [
                [6.265218814381277, 7.794038311935788, 0.0],
                [11.206359450427986, 10.605812457460974, 50.0],
                [17.499999999999993, 14.33012701892219, 100.0],
            ],
            [
                [4.4721359549995805, 8.94427190999916, 0.0],
                [9.16025147168922, 11.450314339611522, 50.0],
                [15.000000000000002, 15.0, 100.0],
            ],
            [
                [6.123233995736765e-16, 9.999999999999998, 0.0],
                [7.654042494670958e-16, 12.5, 50.0],
                [9.18485099360515e-16, 15.0, 100.0],
            ],
        ]
    )
    actual = np.asarray(build_point_grid(params)["inner_points"]).reshape(6, 3, 3)
    assert np.array_equal(actual, expected)


def test_convexity_guard_sees_a_bad_morphed_surface() -> None:
    base = {
        "profileH": {
            "points": [[0.0, 10.0], [100.0, 100.0]],
            "throatAngleDeg": 30.0,
            "mouthAngleDeg": 60.0,
        },
        "profileV": {
            "points": [[0.0, 10.0], [100.0, 80.0]],
            "throatAngleDeg": 30.0,
            "mouthAngleDeg": 60.0,
        },
        "crossSections": [
            {"t": 0.0, "shape": "circle"},
            {"t": 1.0, "shape": "ellipse"},
        ],
    }
    build_freeform_geometry(base)
    with pytest.raises(ValueError, match="non-convex outline.*t=0.125"):
        build_freeform_geometry(
            {
                **base,
                "morphTarget": 2,
                "morphWidth": 2.0,
                "morphHeight": 2.0,
                "morphRate": 1.0,
                "morphAllowShrinkage": 1,
            }
        )


def _curvature_sign_changes(geometry) -> int:
    t_values = np.linspace(0.0, 1.0, 513)
    radii = np.asarray(
        [
            geometry.cross_section_radius(
                np.asarray([math.pi / 4.0]), float(t)
            )[0]
            for t in t_values
        ]
    )
    second = np.gradient(np.gradient(radii, t_values), t_values)
    tolerance = float(np.max(np.abs(second))) * 1.0e-5
    signs = np.sign(second[np.abs(second) > tolerance])
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def test_morph_has_less_45_degree_meridian_ripple_than_four_stations() -> None:
    angle_h = math.degrees(math.atan2(40.0, 100.0))
    angle_v = math.degrees(math.atan2(20.0, 100.0))
    profiles = {
        "profileH": {
            "points": [[0.0, 10.0], [100.0, 50.0]],
            "throatAngleDeg": angle_h,
            "mouthAngleDeg": angle_h,
        },
        "profileV": {
            "points": [[0.0, 10.0], [100.0, 30.0]],
            "throatAngleDeg": angle_v,
            "mouthAngleDeg": angle_v,
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        morphed = build_freeform_geometry(
            {
                **profiles,
                "crossSections": [
                    {"t": 0.0, "shape": "circle"},
                    {"t": 1.0, "shape": "ellipse"},
                ],
                "morphTarget": 3,
                "morphExponent": 4.0,
                "morphRate": 3.0,
            }
        )
        scheduled = build_freeform_geometry(
            {
                **profiles,
                "crossSections": [
                    {"t": 0.0, "shape": "circle"},
                    {"t": 1.0 / 3.0, "shape": "superellipse", "exponent": 8.0 / 3.0},
                    {"t": 2.0 / 3.0, "shape": "superellipse", "exponent": 10.0 / 3.0},
                    {"t": 1.0, "shape": "superellipse", "exponent": 4.0},
                ],
            }
        )

    phi = np.asarray([math.pi / 4.0])
    assert morphed.cross_section_radius(phi, 1.0)[0] == pytest.approx(
        scheduled.cross_section_radius(phi, 1.0)[0], abs=1.0e-12
    )
    morph_changes = _curvature_sign_changes(morphed)
    schedule_changes = _curvature_sign_changes(scheduled)
    assert (morph_changes, schedule_changes) == (1, 6)
    assert morph_changes < schedule_changes


def test_geometry_cache_distinguishes_morph_surfaces() -> None:
    first = build_freeform_geometry(
        {**_straight_params(), "morphTarget": 3, "morphExponent": 2.0}
    )
    second = build_freeform_geometry(
        {**_straight_params(), "morphTarget": 3, "morphExponent": 4.0}
    )
    assert first is not second
    phi = np.asarray([math.pi / 4.0])
    assert first.cross_section_radius(phi, 1.0)[0] != second.cross_section_radius(
        phi, 1.0
    )[0]
