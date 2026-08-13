"""Analytic smooth-target morphing for FREEFORM profiles."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.freeform import build_freeform_geometry
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


def test_rectangle_morph_still_points_to_cross_sections() -> None:
    with pytest.raises(
        ConfigError, match="rectangle morphing.*crossSections.*rounded-rectangle"
    ):
        build_geometry_params(_config_with_morph(1))


def test_expression_target_is_still_rejected() -> None:
    with pytest.raises(ConfigError, match="cannot be proven inactive"):
        build_geometry_params(
            _config_with_morph("1 if 0.01 < p < 0.02 else 0")
        )


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
