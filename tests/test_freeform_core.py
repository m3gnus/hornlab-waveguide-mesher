"""Unit tests for the gmsh-free FREEFORM geometry kernel (stage M1)."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from hornlab_mesher.freeform import (
    _curvature_phi_samples,
    _smootherstep,
    build_freeform_geometry,
    convexity_violations,
)
from hornlab_mesher.profile_morph import _rounded_rect_radius


def _profiles(
    h_points: list[list[float]] | None = None,
    v_points: list[list[float]] | None = None,
) -> dict:
    return {
        "profileH": {
            "points": h_points or [[0.0, 12.7], [100.0, 55.0]],
            "throatAngleDeg": 15.5,
        },
        "profileV": {
            "points": v_points or [[0.0, 12.7], [100.0, 45.0]],
            "throatAngleDeg": 15.5,
        },
        "crossSections": [
            {"t": 0.0, "shape": "circle"},
            {"t": 1.0, "shape": "ellipse"},
        ],
    }


def _legacy_default_scale_radii(profile: dict, z: np.ndarray) -> np.ndarray:
    """Evaluate the pre-simplification spline for its default speed of one."""
    from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

    rows = profile["points"]
    anchors = np.asarray([row[:2] for row in rows], dtype=float)
    chord_lengths = np.linalg.norm(np.diff(anchors, axis=0), axis=1)
    anchor_u = np.concatenate(([0.0], np.cumsum(chord_lengths)))
    anchor_u /= anchor_u[-1]
    z_pchip = PchipInterpolator(anchor_u, anchors[:, 0])
    r_pchip = PchipInterpolator(anchor_u, anchors[:, 1])
    derivatives = np.column_stack(
        (z_pchip.derivative()(anchor_u), r_pchip.derivative()(anchor_u))
    )

    last_delta = anchors[-1] - anchors[-2]
    endpoint_angles = (
        float(rows[0][2])
        if len(rows[0]) == 3
        else float(profile.get("throatAngleDeg", 15.5)),
        float(rows[-1][2])
        if len(rows[-1]) == 3
        else float(
            profile.get(
                "mouthAngleDeg",
                math.degrees(math.atan2(last_delta[1], last_delta[0])),
            )
        ),
    )
    for index, angle_deg in ((0, endpoint_angles[0]), (-1, endpoint_angles[1])):
        speed = float(np.linalg.norm(derivatives[index]))
        angle = math.radians(angle_deg)
        derivatives[index] = speed * np.asarray([math.cos(angle), math.sin(angle)])
    for index, row in enumerate(rows):
        if len(row) != 3:
            continue
        speed = float(
            np.linalg.norm(
                [
                    z_pchip.derivative()(anchor_u[index]),
                    r_pchip.derivative()(anchor_u[index]),
                ]
            )
        )
        angle = math.radians(float(row[2]))
        derivatives[index] = speed * np.asarray([math.cos(angle), math.sin(angle)])

    spline = CubicHermiteSpline(anchor_u, anchors, derivatives, axis=0)
    inverse_u = np.unique(np.concatenate((np.linspace(0.0, 1.0, 4001), anchor_u)))
    inverse_z = np.asarray(spline(inverse_u), dtype=float)[:, 0]
    query_u = np.interp(z, inverse_z, inverse_u)
    return np.asarray(spline(query_u), dtype=float)[:, 1]


def _pure_rounded_radius_mm(
    geometry, phi: np.ndarray, t: float, corner_radius_mm: float
) -> np.ndarray:
    z = t * geometry.length_mm
    a, b = (float(value) for value in geometry.evaluate_radii(np.asarray(z)))
    return np.asarray(
        [
            _rounded_rect_radius(
                float(angle),
                half_width=a,
                half_height=b,
                corner_radius=corner_radius_mm,
            )
            for angle in phi
        ]
    )


def test_module_import_does_not_import_scipy() -> None:
    code = "import sys; import hornlab_mesher.freeform; print(json.dumps('scipy' in sys.modules))"
    completed = subprocess.run(
        [sys.executable, "-c", "import json; " + code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.strip()) is False


def test_circle_degenerate_is_radius_independent_of_phi() -> None:
    params = _profiles(
        [[0.0, 12.7], [40.0, 25.0], [100.0, 50.0]],
        [[0.0, 12.7], [40.0, 25.0], [100.0, 50.0]],
    )
    geometry = build_freeform_geometry(params)
    phi = np.linspace(0.0, 2.0 * math.pi, 181)
    for t in np.linspace(0.0, 1.0, 7):
        radius_h, radius_v = geometry.evaluate_radii(np.asarray(t * geometry.length_mm))
        np.testing.assert_allclose(radius_h, radius_v, rtol=0.0, atol=1.0e-13)
        np.testing.assert_allclose(
            geometry.cross_section_radius(phi, float(t)),
            np.full_like(phi, float(radius_h)),
            rtol=1.0e-13,
            atol=1.0e-13,
        )


@pytest.mark.parametrize(
    "station",
    [
        {"shape": "ellipse"},
        {"shape": "superellipse", "exponent": 4.0},
        {"shape": "rounded_rectangle", "cornerRadiusMm": 10.0},
    ],
)
def test_axis_exactness_for_every_shape_and_blend_position(station: dict) -> None:
    params = _profiles()
    params["crossSections"] = [
        {"t": 0.0, "shape": "ellipse"},
        {"t": 1.0, **station},
    ]
    geometry = build_freeform_geometry(params)
    for t in (0.0, 0.17, 0.5, 0.83, 1.0):
        expected_h, expected_v = geometry.evaluate_radii(
            np.asarray(t * geometry.length_mm)
        )
        actual = geometry.cross_section_radius(np.asarray([0.0, math.pi / 2.0]), t)
        np.testing.assert_allclose(
            actual,
            np.asarray([expected_h, expected_v]),
            rtol=1.0e-12,
            atol=1.0e-13,
        )


@pytest.mark.parametrize("exponent", [4.0, 16.0])
def test_superellipse_off_axis_matches_independent_closed_form(
    exponent: float,
) -> None:
    params = _profiles([[0.0, 12.7], [100.0, 55.0]], [[0.0, 12.7], [100.0, 45.0]])
    params["crossSections"] = [
        {"t": 0.0, "shape": "ellipse"},
        {"t": 1.0, "shape": "superellipse", "exponent": exponent},
    ]
    geometry = build_freeform_geometry(params)
    phi = math.pi / 4.0
    expected = (
        (abs(math.cos(phi)) / 55.0) ** exponent
        + (abs(math.sin(phi)) / 45.0) ** exponent
    ) ** (-1.0 / exponent)

    assert float(
        geometry.cross_section_radius(np.asarray([phi]), 1.0)[0]
    ) == pytest.approx(expected, abs=1.0e-12)


def test_rounded_rectangle_regions_match_independent_closed_forms() -> None:
    a, b, corner = 55.0, 45.0, 10.0
    params = _profiles([[0.0, 12.7], [100.0, a]], [[0.0, 12.7], [100.0, b]])
    params["crossSections"] = [
        {"t": 0.0, "shape": "ellipse"},
        {
            "t": 1.0,
            "shape": "rounded_rectangle",
            "cornerRadiusMm": corner,
        },
    ]
    geometry = build_freeform_geometry(params)
    straight_h = 0.3
    corner_phi = 0.7
    straight_v = 1.0
    center_x, center_y = a - corner, b - corner
    center_projection = center_x * math.cos(corner_phi) + center_y * math.sin(
        corner_phi
    )
    arc_radius = center_projection + math.sqrt(
        center_projection**2 - (center_x**2 + center_y**2 - corner**2)
    )
    phi = np.asarray([0.0, straight_h, corner_phi, straight_v, math.pi / 2.0])
    expected = np.asarray(
        [
            a,
            a / math.cos(straight_h),
            arc_radius,
            b / math.sin(straight_v),
            b,
        ]
    )

    np.testing.assert_allclose(
        geometry.cross_section_radius(phi, 1.0),
        expected,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_rounded_rectangle_held_station_matches_shared_primitive() -> None:
    params = _profiles()
    params["crossSections"] = [
        {"t": 0.0, "shape": "ellipse"},
        {"t": 0.35, "shape": "rounded_rectangle", "cornerRadiusMm": 10.0},
        {"t": 1.0, "shape": "rounded_rectangle", "cornerRadiusMm": 10.0},
    ]
    geometry = build_freeform_geometry(params)
    phi = np.linspace(0.0, 2.0 * math.pi, 257, endpoint=False)
    expected = _pure_rounded_radius_mm(geometry, phi, 0.7, 10.0)
    np.testing.assert_allclose(
        geometry.cross_section_radius(phi, 0.7), expected, rtol=0.0, atol=1.0e-13
    )


def test_rounded_rectangle_mm_station_matches_shared_primitive_at_station() -> None:
    params = _profiles(
        [[0.0, 20.0], [100.0, 55.0]],
        [[0.0, 20.0], [100.0, 45.0]],
    )
    params["crossSections"] = [
        {"t": 0.0, "shape": "ellipse"},
        {"t": 1.0, "shape": "rounded_rectangle", "cornerRadiusMm": 13.5},
    ]
    geometry = build_freeform_geometry(params)
    phi = np.linspace(0.0, math.tau, 257, endpoint=False)

    np.testing.assert_array_equal(
        geometry.cross_section_radius(phi, 1.0),
        _pure_rounded_radius_mm(geometry, phi, 1.0, 13.5),
    )


def test_rounded_rectangle_identical_mm_stations_hold_absolute_radius() -> None:
    params = _profiles()
    params["crossSections"] = [
        {"t": 0.0, "shape": "ellipse"},
        {"t": 0.35, "shape": "rounded_rectangle", "cornerRadiusMm": 10.0},
        {"t": 1.0, "shape": "rounded_rectangle", "cornerRadiusMm": 10.0},
    ]
    geometry = build_freeform_geometry(params)
    phi = np.linspace(0.0, math.tau, 257, endpoint=False)
    expected = _pure_rounded_radius_mm(geometry, phi, 0.7, 10.0)

    np.testing.assert_allclose(
        geometry.cross_section_radius(phi, 0.7), expected, rtol=0.0, atol=1.0e-13
    )


def test_rounded_rectangle_rejects_removed_corner_ratio_with_migration_message() -> (
    None
):
    params = _profiles()
    params["crossSections"][-1] = {
        "t": 1.0,
        "shape": "rounded_rectangle",
        "cornerRatio": 0.3,
    }

    with pytest.raises(
        ValueError,
        match=r"crossSections\[1\]\.cornerRatio was removed; use cornerRadiusMm \(mm\)",
    ):
        build_freeform_geometry(params)


def test_rounded_rectangle_requires_corner_radius_mm() -> None:
    params = _profiles()
    params["crossSections"][-1] = {
        "t": 1.0,
        "shape": "rounded_rectangle",
    }

    with pytest.raises(
        ValueError,
        match=r"crossSections\[1\].*must specify cornerRadiusMm \(mm\)",
    ):
        build_freeform_geometry(params)


def test_rounded_rectangle_rejects_mm_radius_outside_station_range() -> None:
    params = _profiles()
    params["crossSections"][-1] = {
        "t": 1.0,
        "shape": "rounded_rectangle",
        "cornerRadiusMm": 0.5,
    }

    with pytest.raises(
        ValueError,
        match=r"crossSections\[1\].*\[0.9, 45\] mm at station t=1",
    ):
        build_freeform_geometry(params)


def test_rounded_rectangle_rejects_mm_radius_outside_full_active_span() -> None:
    params = _profiles(
        [[0.0, 12.7], [35.0, 20.0], [60.0, 4.0], [100.0, 30.0]],
        [[0.0, 12.7], [35.0, 20.0], [60.0, 4.0], [100.0, 30.0]],
    )
    params["crossSections"] = [
        {"t": 0.0, "shape": "circle"},
        {
            "t": 0.35,
            "shape": "rounded_rectangle",
            "cornerRadiusMm": 10.0,
        },
        {"t": 1.0, "shape": "rounded_rectangle", "cornerRadiusMm": 10.0},
    ]

    with pytest.raises(
        ValueError,
        match=(
            r"crossSections\[1\].*binding t=.*z=60\.[0-9]+ mm "
            r"\(local limit 4\.[0-9]+ mm\).*maximum feasible cornerRadiusMm"
        ),
    ):
        build_freeform_geometry(params)


def test_rounded_rectangle_exact_mm_floor_is_accepted() -> None:
    params = _profiles(
        [[0.0, 12.7], [120.0, 140.0]],
        [[0.0, 12.7], [120.0, 140.0]],
    )
    params["crossSections"] = [
        {"t": 0.0, "shape": "circle"},
        {"t": 1.0, "shape": "rounded_rectangle", "cornerRadiusMm": 2.8},
    ]

    geometry = build_freeform_geometry(params)

    assert geometry.stations[-1]["cornerRadiusMm"] == pytest.approx(2.8)


def test_owner_circle_to_rounded_rectangle_then_hold_scenario() -> None:
    params = _profiles(
        [[0.0, 12.7], [100.0, 50.0]],
        [[0.0, 12.7], [100.0, 50.0]],
    )
    params["crossSections"] = [
        {"t": 0.0, "shape": "circle"},
        {"t": 0.4, "shape": "rounded_rectangle", "cornerRadiusMm": 13.2},
        {"t": 1.0, "shape": "rounded_rectangle", "cornerRadiusMm": 13.2},
    ]
    geometry = build_freeform_geometry(params)
    phi = np.asarray([math.pi / 4.0])

    for t in (0.4, 0.7):
        np.testing.assert_allclose(
            geometry.cross_section_radius(phi, t),
            _pure_rounded_radius_mm(geometry, phi, t, 13.2),
            rtol=0.0,
            atol=1.0e-13,
        )

    a_mid = float(geometry.evaluate_radii(np.asarray(0.2 * geometry.length_mm))[0])
    circle = np.asarray([a_mid])
    rounded = _pure_rounded_radius_mm(geometry, phi, 0.2, 13.2)
    blended = geometry.cross_section_radius(phi, 0.2)
    assert circle[0] < blended[0] < rounded[0]
    assert convexity_violations(geometry, [0.0, 0.2, 0.4, 0.7, 1.0], 64) == []


def _convexity_window_params(corner_radius_mm: float) -> dict:
    params = _profiles(
        [[0.0, 12.7], [60.0, 34.0], [120.0, 70.0]],
        [[0.0, 12.7], [60.0, 30.0], [120.0, 50.0]],
    )
    params["crossSections"] = [
        {"t": 0.0, "shape": "circle"},
        {
            "t": 1.0,
            "shape": "rounded_rectangle",
            "cornerRadiusMm": corner_radius_mm,
        },
    ]
    return params


def test_convexity_guard_reports_minimum_feasible_corner_radius() -> None:
    with pytest.raises(
        ValueError, match=r"minimum feasible corner radius here is ~[0-9.]+ mm"
    ):
        build_freeform_geometry(_convexity_window_params(3.0))

    assert build_freeform_geometry(_convexity_window_params(6.0)).length_mm == 120.0


def test_convexity_ingest_rejects_shallow_rounded_rectangle_blend() -> None:
    params = _profiles(
        [[0.0, 5.541511132371429], [100.0, 106.15347704489386]],
        [[0.0, 5.541511132371429], [100.0, 164.0176191763291]],
    )
    params["crossSections"] = [
        {"t": 0.0, "shape": "ellipse"},
        {
            "t": 1.0,
            "shape": "rounded_rectangle",
            "cornerRadiusMm": 6.023758168248212,
        },
    ]

    with pytest.raises(ValueError, match="non-convex outline"):
        build_freeform_geometry(params)


def test_cached_geometry_and_curvature_reports_are_effectively_immutable() -> None:
    params = _profiles(
        [[0.0, 12.7], [120.0, 80.0]],
        [[0.0, 12.7], [120.0, 60.0]],
    )
    params["crossSections"] = [
        {"t": 0.0, "shape": "circle"},
        {"t": 1.0, "shape": "superellipse", "exponent": 4.0},
    ]
    geometry = build_freeform_geometry(params)
    baseline_radius = float(
        geometry.cross_section_radius(np.asarray([math.pi / 4.0]), 1.0)[0]
    )

    with pytest.raises(TypeError):
        geometry.stations[-1]["shape"] = "ellipse"
    with pytest.raises(ValueError):
        geometry._profile_h.anchors[0, 0] = 999.0

    curvature = geometry.surface_curvature_report(0.1)
    baseline_curvatures = curvature["principalCurvaturesPerMm"].copy()
    curvature["principalCurvaturesPerMm"][0] = 999.0

    rebuilt = build_freeform_geometry(params)
    assert float(
        rebuilt.cross_section_radius(np.asarray([math.pi / 4.0]), 1.0)[0]
    ) == pytest.approx(baseline_radius)
    assert rebuilt.surface_curvature_report(0.1)[
        "principalCurvaturesPerMm"
    ] == pytest.approx(baseline_curvatures)


def test_outward_cylinder_wall_uses_signed_curvature_direction() -> None:
    params = _profiles(
        [[0.0, 12.7], [120.0, 12.7]],
        [[0.0, 12.7], [120.0, 12.7]],
    )
    params["profileH"].update(throatAngleDeg=0.0, mouthAngleDeg=0.0)
    params["profileV"].update(throatAngleDeg=0.0, mouthAngleDeg=0.0)

    report = build_freeform_geometry(params).surface_curvature_report(10.0)

    assert report["ok"] is True
    assert report["maxThicknessTimesPrincipalCurvature"] == pytest.approx(0.0)


def test_curvature_sampling_ignores_inactive_rounded_rectangle_stations() -> None:
    params = _profiles()
    params["crossSections"] = [
        {"t": 0.0, "shape": "ellipse"},
        {"t": 0.5, "shape": "ellipse"},
        {"t": 0.75, "shape": "rounded_rectangle", "cornerRadiusMm": 10.0},
        {"t": 1.0, "shape": "rounded_rectangle", "cornerRadiusMm": 20.0},
    ]
    geometry = build_freeform_geometry(params)

    inactive = _curvature_phi_samples(geometry, 0.25)
    active = _curvature_phi_samples(geometry, 0.6)

    assert inactive.size == 721
    assert active.size > inactive.size


@pytest.mark.parametrize("corner_radius_mm", [15.0, 30.0])
def test_weight_aware_corner_cap_accepts_convex_corners_above_throat_radius(
    corner_radius_mm: float,
) -> None:
    geometry = build_freeform_geometry(_convexity_window_params(corner_radius_mm))

    assert geometry.stations[-1]["cornerRadiusMm"] == corner_radius_mm


def test_owner_s_curvature_builds_by_default_and_reports_inflection_spans() -> None:
    params = _profiles(
        [[0.0, 12.7], [60.0, 80.0], [120.0, 160.0]],
        [[0.0, 12.7], [60.0, 60.0], [120.0, 110.0]],
    )
    params["profileH"]["mouthAngleDeg"] = 70.0
    params["profileV"]["mouthAngleDeg"] = 60.0

    report = build_freeform_geometry(params).report()

    assert report["inflectionSpans"]["H"]
    assert report["inflectionSpans"]["H"][0]["tangentDropDeg"] > 5.0


def test_explicit_s_curvature_rejected_only_under_reject_policy() -> None:
    h_points = [[0.0, 12.7], [50.0, 35.0, 40.0], [100.0, 55.0]]
    params = _profiles(h_points, [[0.0, 12.7], [100.0, 55.0]])
    params["profileH"]["mouthAngleDeg"] = 25.0

    assert build_freeform_geometry(params).report()["inflectionSpans"]["H"]

    params["inflectionPolicy"] = "reject"
    with pytest.raises(
        ValueError,
        match=r"profileH.*z=[0-9.]+\.\.[0-9.]+ mm.*[0-9.]+ deg",
    ):
        build_freeform_geometry(params)


def test_straight_cones_have_no_reported_inflection_spans() -> None:
    angle = 20.0
    slope = math.tan(math.radians(angle))
    points = [[0.0, 12.7], [100.0, 12.7 + 100.0 * slope]]
    params = _profiles(copy.deepcopy(points), copy.deepcopy(points))
    for profile in (params["profileH"], params["profileV"]):
        profile.update(throatAngleDeg=angle, mouthAngleDeg=angle)

    assert build_freeform_geometry(params).report()["inflectionSpans"] == {
        "H": [],
        "V": [],
    }


def test_sub_degree_tangent_wiggle_is_not_reported() -> None:
    angle = 20.0
    slope = math.tan(math.radians(angle))
    points = [
        [0.0, 12.7],
        [50.0, 12.7 + 50.0 * slope, angle - 0.5],
        [100.0, 12.7 + 100.0 * slope],
    ]
    params = _profiles(copy.deepcopy(points), copy.deepcopy(points))
    for profile in (params["profileH"], params["profileV"]):
        profile.update(throatAngleDeg=angle, mouthAngleDeg=angle)

    assert build_freeform_geometry(params).report()["inflectionSpans"] == {
        "H": [],
        "V": [],
    }


def test_smootherstep_first_and_second_derivatives_vanish_at_ends() -> None:
    h = 1.0e-5
    assert _smootherstep(0.0) == pytest.approx(0.0)
    assert _smootherstep(1.0) == pytest.approx(1.0)

    derivative_start = (_smootherstep(h) - _smootherstep(0.0)) / h
    derivative_end = (_smootherstep(1.0) - _smootherstep(1.0 - h)) / h
    second_start = (
        _smootherstep(2.0 * h) - 2.0 * _smootherstep(h) + _smootherstep(0.0)
    ) / h**2
    second_end = (
        _smootherstep(1.0) - 2.0 * _smootherstep(1.0 - h) + _smootherstep(1.0 - 2.0 * h)
    ) / h**2
    assert derivative_start == pytest.approx(0.0, abs=2.0e-9)
    assert derivative_end == pytest.approx(0.0, abs=2.0e-9)
    assert second_start == pytest.approx(0.0, abs=7.0e-4)
    assert second_end == pytest.approx(0.0, abs=7.0e-4)


@pytest.mark.parametrize("mouth_angle", [60.0, 90.0])
def test_two_anchor_tangent_curve_is_monotone_and_interpolates_endpoints(
    mouth_angle: float,
) -> None:
    profile = {
        "points": [[0.0, 12.7], [120.0, 150.0]],
        "throatAngleDeg": 15.5,
        "mouthAngleDeg": mouth_angle,
    }
    params = {
        "profileH": copy.deepcopy(profile),
        "profileV": copy.deepcopy(profile),
        "crossSections": [
            {"t": 0.0, "shape": "circle"},
            {"t": 1.0, "shape": "ellipse"},
        ],
    }
    geometry = build_freeform_geometry(params)
    z = np.linspace(0.0, 120.0, 2001)
    radius_h, radius_v = geometry.evaluate_radii(z)
    assert np.all(np.diff(radius_h) > 0.0)
    np.testing.assert_allclose(radius_h, radius_v, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        radius_h[[0, -1]], np.asarray([12.7, 150.0]), rtol=0.0, atol=1.0e-12
    )
    endpoint_dz_du = float(geometry._profile_h.spline.derivative()(1.0)[0])
    if mouth_angle == 90.0:
        assert endpoint_dz_du == pytest.approx(0.0, abs=1.0e-12)
    else:
        assert endpoint_dz_du > 0.0
    dense_u = np.linspace(0.0, 1.0, 2001)
    dz_du = geometry._profile_h.spline.derivative()(dense_u)[:, 0]
    assert np.all(dz_du[:-1] > 0.0)
    assert dz_du[-1] >= -1.0e-12


def _invalid_cases() -> list[tuple[str, callable]]:
    return [
        (
            "2-64 anchors",
            lambda p: p["profileH"].update(points=[[0.0, 12.7]]),
        ),
        (
            "strictly increasing",
            lambda p: p["profileH"].update(
                points=[[0.0, 12.7], [50.0, 30.0], [40.0, 35.0]]
            ),
        ),
        (
            "throat radii",
            lambda p: p["profileV"].update(points=[[0.0, 12.8], [100.0, 45.0]]),
        ),
        (
            "[-90, 90]",
            lambda p: p["profileH"].update(mouthAngleDeg=91.0),
        ),
        (
            "strictly increasing",
            lambda p: p.update(
                crossSections=[
                    {"t": 0.0, "shape": "circle"},
                    {"t": 0.7, "shape": "ellipse"},
                    {"t": 0.6, "shape": "ellipse"},
                    {"t": 1.0, "shape": "ellipse"},
                ]
            ),
        ),
        (
            "first station",
            lambda p: p.update(
                crossSections=[
                    {"t": 0.1, "shape": "circle"},
                    {"t": 1.0, "shape": "ellipse"},
                ]
            ),
        ),
        (
            "inflectionPolicy",
            lambda p: p.update(inflectionPolicy="ignore"),
        ),
    ]


@pytest.mark.parametrize("message,mutate", _invalid_cases())
def test_validation_rejections(message: str, mutate) -> None:
    params = _profiles()
    mutate(params)
    with pytest.raises(ValueError, match=message):
        build_freeform_geometry(params)


def test_removed_overshoot_policy_is_rejected() -> None:
    profile = {
        "points": [[0.0, 10.0], [50.0, 20.0], [100.0, 30.0]],
        "throatAngleDeg": 80.0,
    }
    params = {
        "profileH": copy.deepcopy(profile),
        "profileV": copy.deepcopy(profile),
        "crossSections": [
            {"t": 0.0, "shape": "circle"},
            {"t": 1.0, "shape": "ellipse"},
        ],
    }
    params["overshootPolicy"] = "allow"
    with pytest.raises(
        ValueError, match=r"overshootPolicy was removed.*solved automatically"
    ):
        build_freeform_geometry(params)


def test_identical_param_dicts_return_same_memoized_object() -> None:
    first = _profiles()
    second = copy.deepcopy(first)
    assert build_freeform_geometry(first) is build_freeform_geometry(second)


def test_per_anchor_angle_matches_dense_sampled_tangent_direction() -> None:
    points = [[0.0, 12.7], [50.0, 30.0, 35.0], [100.0, 50.0]]
    params = _profiles(copy.deepcopy(points), copy.deepcopy(points))
    geometry = build_freeform_geometry(params)

    anchor_u = float(geometry._profile_h.anchor_u[1])
    dense_u = anchor_u + np.linspace(-1.0e-8, 1.0e-8, 201)
    dense_points = np.asarray(geometry._profile_h.spline(dense_u), dtype=float)
    sampled_direction = dense_points[-1] - dense_points[0]
    sampled_angle = math.degrees(
        math.atan2(float(sampled_direction[1]), float(sampled_direction[0]))
    )

    assert sampled_angle == pytest.approx(35.0, abs=1.0e-6)


@pytest.mark.parametrize(
    "params",
    [
        _profiles(
            [[0.0, 12.7], [40.0, 25.0], [100.0, 50.0]],
            [[0.0, 12.7], [45.0, 23.0], [100.0, 42.0]],
        ),
        {
            "profileH": {
                "points": [[0.0, 12.7], [60.0, 80.0], [120.0, 160.0]],
                "throatAngleDeg": 15.5,
                "mouthAngleDeg": 70.0,
            },
            "profileV": {
                "points": [[0.0, 12.7], [60.0, 60.0], [120.0, 110.0]],
                "throatAngleDeg": 15.5,
                "mouthAngleDeg": 60.0,
            },
        },
        _profiles(
            [[0.0, 12.7, 8.0], [50.0, 30.0, 20.0], [100.0, 50.0, 30.0]],
            [[0.0, 12.7, 8.0], [55.0, 27.0, 15.0], [100.0, 45.0, 25.0]],
        ),
    ],
)
def test_default_speed_profiles_match_legacy_geometry(params: dict) -> None:
    geometry = build_freeform_geometry(params)
    z = np.linspace(0.0, geometry.length_mm, 401)
    actual = geometry.evaluate_radii(z)

    for plane_index, plane_name in enumerate(("H", "V")):
        expected = _legacy_default_scale_radii(params[f"profile{plane_name}"], z)
        np.testing.assert_allclose(actual[plane_index], expected, rtol=0.0, atol=1.0e-9)
        assert all(
            tangent["speedFactor"] == 1.0
            for tangent in geometry.report()["anchorTangents"][plane_name]
        )


def test_removed_throat_tangent_scale_is_unknown_and_solver_builds_profile() -> None:
    points = [[0.0, 12.7], [60.0, 80.0], [120.0, 160.0]]
    params = _profiles(copy.deepcopy(points), copy.deepcopy(points))
    for profile in (params["profileH"], params["profileV"]):
        profile["mouthAngleDeg"] = 70.0
    params["profileH"]["throatTangentScale"] = 3.0

    with pytest.raises(
        ValueError, match=r"profileH has unknown key 'throatTangentScale'"
    ):
        build_freeform_geometry(params)

    del params["profileH"]["throatTangentScale"]
    assert build_freeform_geometry(params).length_mm == pytest.approx(120.0)


def _assert_segment_radius_brackets(geometry) -> None:
    for plane in (geometry._profile_h, geometry._profile_v):
        for segment in range(plane.anchor_u.size - 1):
            radii = np.asarray(
                plane.spline(
                    np.linspace(
                        plane.anchor_u[segment], plane.anchor_u[segment + 1], 2001
                    )
                ),
                dtype=float,
            )[:, 1]
            lower, upper = sorted(plane.anchors[segment : segment + 2, 1])
            tolerance = max(0.05, 1.0e-3 * max(abs(lower), abs(upper)))
            assert float(lower - np.min(radii)) <= tolerance + 1.0e-8
            assert float(np.max(radii) - upper) <= tolerance + 1.0e-8


def test_small_negative_interior_angle_builds_with_physical_tolerance() -> None:
    points = [[0.0, 12.7], [60.0, 80.0, -5.0], [120.0, 160.0]]
    params = _profiles(copy.deepcopy(points), copy.deepcopy(points))
    for profile in (params["profileH"], params["profileV"]):
        profile["mouthAngleDeg"] = 70.0

    geometry = build_freeform_geometry(params)

    _assert_segment_radius_brackets(geometry)


def test_steep_negative_interior_angle_backs_off_speed() -> None:
    points = [[0.0, 12.7], [60.0, 80.0, -45.0], [120.0, 160.0]]
    params = _profiles(copy.deepcopy(points), copy.deepcopy(points))
    for profile in (params["profileH"], params["profileV"]):
        profile["mouthAngleDeg"] = 70.0

    geometry = build_freeform_geometry(params)
    report = geometry.report()

    assert report["anchorTangents"]["H"][1]["speedFactor"] < 1.0
    _assert_segment_radius_brackets(geometry)
    dense_u = np.linspace(0.0, 1.0, 4001)
    assert np.all(geometry._profile_h.spline.derivative()(dense_u)[:, 0] > 0.0)


def test_solved_tangent_geometry_is_continuous_across_angle_sweep() -> None:
    z = np.linspace(0.0, 100.0, 1001)
    previous = None
    maximum_step = 0.0
    engaged = False
    for angle in range(-60, 61):
        points = [[0.0, 12.7], [50.0, 30.0, float(angle)], [100.0, 50.0]]
        params = _profiles(copy.deepcopy(points), copy.deepcopy(points))
        geometry = build_freeform_geometry(params)
        radii = geometry.evaluate_radii(z)[0]
        engaged |= geometry.report()["anchorTangents"]["H"][1]["speedFactor"] < 1.0
        if previous is not None:
            maximum_step = max(maximum_step, float(np.max(np.abs(radii - previous))))
        previous = radii

    assert engaged
    assert maximum_step < 1.0


def test_circle_alias_is_accepted_at_any_station_and_normalised() -> None:
    params = _profiles()
    params["crossSections"] = [
        {"t": 0.0, "shape": "circle"},
        {"t": 0.5, "shape": "circle"},
        {"t": 1.0, "shape": "circle"},
    ]

    geometry = build_freeform_geometry(params)

    assert [station["shape"] for station in geometry.stations] == [
        "ellipse",
        "ellipse",
        "ellipse",
    ]


def test_two_element_anchor_rows_preserve_exact_legacy_radii() -> None:
    params = _profiles(
        [[0.0, 12.7], [40.0, 25.0], [100.0, 50.0]],
        [[0.0, 12.7], [40.0, 25.0], [100.0, 50.0]],
    )
    z = np.linspace(0.0, 100.0, 17)
    radius_h, radius_v = build_freeform_geometry(params).evaluate_radii(z)
    expected_bytes = bytes.fromhex(
        "6666666666662940e23d8d6da0e72c40ba1244b91a40304030d23b1bf81a3240"
        "298ccee46707344089b6bcae8a0836400eab0bcfa92138402eeb589d4c583a40"
        "dee3c6e789b43c40a96fb52c9e313f40eafaaf9a98e44040fc7260b2483a4240"
        "0e7cfd986596434003896f9e61f54440595fe8fba853464085693adcafad4740"
        "0100000000004940"
    )

    assert radius_h.tobytes() == expected_bytes
    assert radius_v.tobytes() == expected_bytes


def test_endpoint_row_tangent_overrides_block_angle() -> None:
    explicit_points = [
        [0.0, 12.7],
        [50.0, 30.0],
        [100.0, 50.0, 25.0],
    ]
    params = _profiles(copy.deepcopy(explicit_points), copy.deepcopy(explicit_points))
    for profile in (params["profileH"], params["profileV"]):
        profile.update(mouthAngleDeg=-45.0)
    explicit = build_freeform_geometry(params)

    reference_points = [row[:2] for row in explicit_points]
    reference_params = _profiles(
        copy.deepcopy(reference_points), copy.deepcopy(reference_points)
    )
    for profile in (reference_params["profileH"], reference_params["profileV"]):
        profile.update(mouthAngleDeg=25.0)
    reference = build_freeform_geometry(reference_params)

    explicit_derivative = np.asarray(
        explicit._profile_h.spline.derivative()(1.0), dtype=float
    )
    reference_derivative = np.asarray(
        reference._profile_h.spline.derivative()(1.0), dtype=float
    )
    explicit_angle = math.degrees(
        math.atan2(float(explicit_derivative[1]), float(explicit_derivative[0]))
    )
    assert explicit_angle == pytest.approx(25.0, abs=1.0e-12)
    np.testing.assert_allclose(explicit_derivative, reference_derivative, atol=1.0e-12)
    assert explicit.report()["tangentAnglesDeg"]["H"]["mouth"] == 25.0


def test_interior_vertical_tangent_rejected_during_anchor_parsing() -> None:
    params = _profiles(
        [[0.0, 12.7], [50.0, 30.0, 90.0], [100.0, 50.0]],
        [[0.0, 12.7], [50.0, 30.0], [100.0, 50.0]],
    )

    with pytest.raises(
        ValueError,
        match=r"profileH\.points\[1\]\.angleDeg.*\(-90, 90\).*interior anchor",
    ):
        build_freeform_geometry(params)


def test_four_element_anchor_row_rejects_removed_strength() -> None:
    params = _profiles(
        [[0.0, 12.7], [50.0, 30.0, 20.0, 1.0], [100.0, 50.0]],
        [[0.0, 12.7], [50.0, 30.0], [100.0, 50.0]],
    )

    with pytest.raises(
        ValueError,
        match=r"profileH\.points\[1\].*strength was removed.*solved automatically",
    ):
        build_freeform_geometry(params)


def test_cache_distinguishes_anchor_tangent_rows() -> None:
    automatic = _profiles(
        [[0.0, 12.7], [50.0, 30.0], [100.0, 50.0]],
        [[0.0, 12.7], [50.0, 30.0], [100.0, 50.0]],
    )
    explicit = copy.deepcopy(automatic)
    explicit["profileH"]["points"][1].append(20.0)
    explicit["profileV"]["points"][1].append(20.0)

    automatic_geometry = build_freeform_geometry(automatic)
    explicit_geometry = build_freeform_geometry(explicit)
    assert automatic_geometry is not explicit_geometry


def test_cache_distinguishes_inflection_policies() -> None:
    angle = 20.0
    slope = math.tan(math.radians(angle))
    points = [[0.0, 12.7], [100.0, 12.7 + 100.0 * slope]]
    warned = _profiles(copy.deepcopy(points), copy.deepcopy(points))
    for profile in (warned["profileH"], warned["profileV"]):
        profile.update(throatAngleDeg=angle, mouthAngleDeg=angle)
    rejected = copy.deepcopy(warned)
    warned["inflectionPolicy"] = "warn"
    rejected["inflectionPolicy"] = "reject"

    assert build_freeform_geometry(warned) is not build_freeform_geometry(rejected)


def test_allow_inflection_policy_is_rejected_with_migration_message() -> None:
    params = _profiles()
    params["inflectionPolicy"] = "allow"

    with pytest.raises(
        ValueError,
        match=r"inflectionPolicy.*'allow' was removed; use 'warn' or remove the key",
    ):
        build_freeform_geometry(params)


def test_report_includes_authoritative_per_anchor_tangent_rows() -> None:
    points = [[0.0, 12.7, 10.0], [50.0, 30.0], [100.0, 50.0, 25.0]]
    params = _profiles(copy.deepcopy(points), copy.deepcopy(points))
    report = build_freeform_geometry(params).report()

    assert report["anchorTangents"]["H"] == [
        {"z": 0.0, "r": 12.7, "angleDeg": 10.0, "speedFactor": 1.0},
        {"z": 50.0, "r": 30.0, "angleDeg": None, "speedFactor": 1.0},
        {"z": 100.0, "r": 50.0, "angleDeg": 25.0, "speedFactor": 1.0},
    ]


def test_report_deviation_is_zero_for_line_and_positive_for_curved_spline() -> None:
    z = np.asarray([0.0, 18.0, 47.0, 83.0, 120.0])
    throat_radius = 12.7
    h_angle = 20.0
    v_angle = 15.0
    line_params = {
        "profileH": {
            "points": np.column_stack(
                (z, throat_radius + z * math.tan(math.radians(h_angle)))
            ),
            "throatAngleDeg": h_angle,
            "mouthAngleDeg": h_angle,
        },
        "profileV": {
            "points": np.column_stack(
                (z, throat_radius + z * math.tan(math.radians(v_angle)))
            ),
            "throatAngleDeg": v_angle,
            "mouthAngleDeg": v_angle,
        },
    }
    line_report = build_freeform_geometry(line_params).report()
    assert line_report["maxNormalDeviationMm"]["H"] < 1.0e-11
    assert line_report["maxNormalDeviationMm"]["V"] < 1.0e-11
    assert line_report["throatRadiusMm"] == pytest.approx(throat_radius)
    assert line_report["tangentAnglesDeg"]["H"] == {
        "throat": h_angle,
        "mouth": h_angle,
    }

    curved_params = {
        "profileH": {
            "points": [[0.0, 12.7], [65.0, 29.0], [120.0, 75.0]],
            "throatAngleDeg": 10.0,
            "mouthAngleDeg": 50.0,
        },
        "profileV": {
            "points": [[0.0, 12.7], [55.0, 25.0], [120.0, 58.0]],
            "throatAngleDeg": 10.0,
            "mouthAngleDeg": 40.0,
        },
    }
    curved_report = build_freeform_geometry(curved_params).report()
    assert curved_report["maxNormalDeviationMm"]["H"] > 0.1
    assert curved_report["maxNormalDeviationMm"]["V"] > 0.1


def test_report_curve_samples_are_exact_spline_points() -> None:
    """report() exposes 192 authoritative [z, r] samples per plane for the UI."""
    params = {
        "profileH": {
            "points": [[0.0, 12.7], [60.0, 80.0], [120.0, 160.0]],
            "throatAngleDeg": 15.5,
            "mouthAngleDeg": 70.0,
        },
        "profileV": {
            "points": [[0.0, 12.7], [60.0, 60.0], [120.0, 110.0]],
            "throatAngleDeg": 15.5,
            "mouthAngleDeg": 60.0,
        },
        "crossSections": [
            {"t": 0.0, "shape": "circle"},
            {"t": 1.0, "shape": "ellipse"},
        ],
    }
    geometry = build_freeform_geometry(params)
    samples = geometry.report()["curveSamples"]
    for plane in ("H", "V"):
        rows = samples[plane]
        assert len(rows) == 192
        assert rows[0][0] == pytest.approx(0.0, abs=1e-9)
        assert rows[0][1] == pytest.approx(12.7, abs=1e-9)
        assert rows[-1][0] == pytest.approx(120.0, abs=1e-9)
    assert samples["H"][-1][1] == pytest.approx(160.0, abs=1e-9)
    assert samples["V"][-1][1] == pytest.approx(110.0, abs=1e-9)
