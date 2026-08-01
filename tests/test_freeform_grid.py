"""FREEFORM registration, config, and point-grid integration (stage M2)."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.freeform import (
    active_rounded_rect_corner_radius_mm,
    build_freeform_geometry,
)
from hornlab_mesher.profiles import build_point_grid, profile_points


def _owner_config(*, quadrants: str = "1234") -> dict:
    return {
        "formula": "FREEFORM",
        "mode": "bare",
        "profile": {
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
                {
                    "t": 0.4,
                    "shape": "rounded_rectangle",
                    "cornerRatio": 0.12,
                },
                {
                    "t": 1.0,
                    "shape": "rounded_rectangle",
                    "cornerRatio": 0.12,
                },
            ],
        },
        "mesh": {
            "angularSegments": 32,
            "lengthSegments": 8,
            "samplingMode": "uniform",
            "quadrants": quadrants,
        },
    }


def _built_grid(config: dict) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    params, formula, _mode = build_geometry_params(config)
    assert formula == "FREEFORM"
    grid = build_point_grid(params)
    points = np.asarray(grid["inner_points"], dtype=np.float64).reshape(
        int(grid["grid_n_phi"]), int(grid["grid_n_length"]) + 1, 3
    )
    phi_grid = np.asarray(grid["phi_grid"], dtype=np.float64)
    return params, grid, points, phi_grid


def _mixed_schedule_config(*, quadrants: str = "1234") -> dict:
    config = _owner_config(quadrants=quadrants)
    config["profile"]["crossSections"] = [
        {"t": 0.0, "shape": "circle"},
        {
            "t": 0.5,
            "shape": "rounded_rectangle",
            "cornerRadiusMm": 10.0,
        },
        {"t": 1.0, "shape": "ellipse"},
    ]
    return config


def test_owner_grid_honours_axes_and_merges_feature_stations() -> None:
    params, grid, points, phi_grid = _built_grid(_owner_config())
    geometry = build_freeform_geometry(params)
    t_values = np.asarray(grid["slice_map"], dtype=np.float64)
    z = points[0, :, 2]
    expected_h, expected_v = geometry.evaluate_radii(z)

    h_row = np.flatnonzero(np.all(np.isclose(phi_grid, 0.0, atol=1.0e-14), axis=1))
    v_row = np.flatnonzero(
        np.all(np.isclose(phi_grid, math.pi / 2.0, atol=1.0e-14), axis=1)
    )
    assert h_row.size == 1
    assert v_row.size == 1
    radii = np.linalg.norm(points[:, :, :2], axis=2)
    np.testing.assert_allclose(radii[h_row[0]], expected_h, rtol=0.0, atol=1.0e-9)
    np.testing.assert_allclose(radii[v_row[0]], expected_v, rtol=0.0, atol=1.0e-9)

    for feature_t in (0.0, 0.4, 0.5, 1.0):
        assert np.any(np.isclose(t_values, feature_t, rtol=0.0, atol=1.0e-14))
    assert len(t_values) >= 9
    assert np.all(np.diff(z) > 0.0)
    np.testing.assert_allclose(
        points[:, :, 2], np.broadcast_to(z, points[:, :, 2].shape), atol=0.0
    )
    assert np.all(radii > 0.0)


def test_quadrant_one_is_exact_subset_of_full_freeform_grid() -> None:
    _params, _grid, full_points, full_phi = _built_grid(_owner_config())
    _params, _grid, q1_points, q1_phi = _built_grid(
        _owner_config(quadrants="1")
    )
    np.testing.assert_allclose(
        q1_points, full_points[: q1_points.shape[0]], rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        q1_phi, full_phi[: q1_phi.shape[0]], rtol=0.0, atol=1.0e-14
    )


def test_mixed_schedule_quadrant_one_is_exact_subset_of_full_grid() -> None:
    _params, _grid, full_points, full_phi = _built_grid(_mixed_schedule_config())
    _params, _grid, q1_points, q1_phi = _built_grid(
        _mixed_schedule_config(quadrants="1")
    )
    np.testing.assert_allclose(
        q1_points, full_points[: q1_points.shape[0]], rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        q1_phi, full_phi[: q1_phi.shape[0]], rtol=0.0, atol=1.0e-14
    )


def test_intermediate_rounded_rectangle_pins_tangencies_on_nearby_rings() -> None:
    params, grid, points, phi_grid = _built_grid(_mixed_schedule_config())
    geometry = build_freeform_geometry(params)
    t_values = np.asarray(grid["slice_map"], dtype=np.float64)
    radii_h, radii_v = geometry.evaluate_radii(points[0, :, 2])
    middle = int(np.flatnonzero(np.isclose(t_values, 0.5, atol=1.0e-14))[0])

    for ring_index in (middle - 1, middle, middle + 1):
        a = float(radii_h[ring_index])
        b = float(radii_v[ring_index])
        theta_1 = math.atan2(b - 10.0, a)
        theta_2 = math.atan2(b, a - 10.0)
        assert np.any(
            np.isclose(phi_grid[:, ring_index], theta_1, rtol=0.0, atol=1.0e-14)
        )
        assert np.any(
            np.isclose(phi_grid[:, ring_index], theta_2, rtol=0.0, atol=1.0e-14)
        )


def test_mm_corner_mouth_grid_uses_each_rings_active_corner_tangencies() -> None:
    config = _owner_config()
    config["profile"]["crossSections"][-1] = {
        "t": 1.0,
        "shape": "rounded_rectangle",
        "cornerRadiusMm": 10.0,
    }
    params, grid, points, phi_grid = _built_grid(config)
    geometry = build_freeform_geometry(params)
    radii_h, radii_v = geometry.evaluate_radii(points[0, :, 2])
    t_values = np.asarray(grid["slice_map"], dtype=np.float64)
    expected_spans = []
    for t, a, b in zip(t_values, radii_h, radii_v):
        corner = active_rounded_rect_corner_radius_mm(
            geometry.stations, float(t), float(a), float(b)
        )
        expected_spans.append(
            [
                math.atan2(float(b) - corner, float(a)),
                math.atan2(float(b), float(a) - corner),
            ]
        )
    expected_spans = np.asarray(expected_spans)
    actual_spans = np.asarray(grid["freeform_corner_arc_spans"], dtype=np.float64)

    np.testing.assert_allclose(actual_spans, expected_spans, rtol=0.0, atol=1.0e-14)
    for ring_index, (theta_1, theta_2) in enumerate(expected_spans):
        assert np.any(np.isclose(phi_grid[:, ring_index], theta_1, atol=1.0e-14))
        assert np.any(np.isclose(phi_grid[:, ring_index], theta_2, atol=1.0e-14))
    assert np.ptp(expected_spans[:, 0]) > 1.0e-3


def test_circle_degenerate_freeform_grid_is_circular_on_every_ring() -> None:
    config = _owner_config()
    shared = [[0.0, 12.7], [45.0, 31.0], [120.0, 80.0]]
    config["profile"]["profileH"] = {
        "points": copy.deepcopy(shared),
        "throatAngleDeg": 15.5,
        "mouthAngleDeg": 45.0,
    }
    config["profile"]["profileV"] = copy.deepcopy(
        config["profile"]["profileH"]
    )
    config["profile"].pop("crossSections")

    _params, _grid, points, _phi_grid = _built_grid(config)
    radii = np.linalg.norm(points[:, :, :2], axis=2)
    np.testing.assert_allclose(
        radii,
        np.broadcast_to(np.mean(radii, axis=0), radii.shape),
        rtol=1.0e-9,
        atol=1.0e-12,
    )


def test_freeform_config_threading_and_value_based_feature_gates() -> None:
    config = _owner_config()
    config["profile"]["inflectionPolicy"] = "allow"
    params, formula, _mode = build_geometry_params(config)
    assert formula == "FREEFORM"
    for key in ("profileH", "profileV", "crossSections", "inflectionPolicy"):
        assert params[key] == config["profile"][key]

    accepted_zero = copy.deepcopy(config)
    accepted_zero["morph"] = {"morphTarget": 0}
    assert build_geometry_params(accepted_zero)[1] == "FREEFORM"

    active_morph = copy.deepcopy(config)
    active_morph["morph"] = {"morphTarget": 1}
    with pytest.raises(ConfigError, match="FREEFORM.*morphTarget"):
        build_geometry_params(active_morph)

    active_gcurve = copy.deepcopy(config)
    active_gcurve["gcurve"] = {"gcurveType": 1, "gcurveWidth": 100.0}
    with pytest.raises(ConfigError, match="FREEFORM.*guiding curves"):
        build_geometry_params(active_gcurve)

    foreign_coefficient = copy.deepcopy(config)
    foreign_coefficient["profile"]["L"] = 120.0
    with pytest.raises(ConfigError, match="FREEFORM.*coefficient"):
        build_geometry_params(foreign_coefficient)

    small_source = copy.deepcopy(config)
    small_source["source"] = {"sourceShape": 1, "sourceRadius": 10.0}
    with pytest.raises(ConfigError, match="sourceRadius.*throat radius"):
        build_geometry_params(small_source)


def test_freeform_custom_zmap_is_merged_and_ath_map_is_rejected() -> None:
    config = _owner_config()
    config["mesh"].update(
        samplingMode="zmap",
        zMapPoints=[0.2, 0.1, 0.7, 0.85],
    )
    _params, grid, _points, _phi_grid = _built_grid(config)
    t_values = np.asarray(grid["slice_map"], dtype=np.float64)
    assert np.any(np.isclose(t_values, 0.4, atol=1.0e-14))
    assert np.any(np.isclose(t_values, 0.5, atol=1.0e-14))

    config["mesh"]["samplingMode"] = "ath-default-zmap"
    with pytest.raises(ConfigError, match="uniform or a custom zmap"):
        build_geometry_params(config)


def test_profile_points_returns_freeform_h_meridian_and_lookup_polyline() -> None:
    params, _formula, _mode = build_geometry_params(_owner_config())
    geometry = build_freeform_geometry(params)
    points = profile_points(params, 5)
    expected_h, _expected_v = geometry.evaluate_radii(points[:, 0])
    np.testing.assert_allclose(points[:, 1], expected_h, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        points[[0, 2, 4]],
        np.asarray([[0.0, 12.7], [60.0, 80.0], [120.0, 160.0]]),
        rtol=0.0,
        atol=1.0e-12,
    )

    lookup = [[5.0, 10.0], [25.0, 30.0], [45.0, 70.0]]
    lookup_points = profile_points(
        {"type": "LOOKUP", "lookupProfile": lookup}, n_axial=5
    )
    np.testing.assert_allclose(
        lookup_points,
        np.asarray(
            [[5.0, 10.0], [15.0, 20.0], [25.0, 30.0], [35.0, 50.0], [45.0, 70.0]]
        ),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_low_level_grid_reuses_shared_freeform_feature_validation() -> None:
    params, _formula, _mode = build_geometry_params(_owner_config())
    with pytest.raises(ValueError, match="FREEFORM.*morphTarget"):
        build_point_grid({**params, "morphTarget": 1})
    with pytest.raises(ValueError, match="FREEFORM.*guiding curves"):
        build_point_grid({**params, "gcurveType": 1, "gcurveWidth": 100.0})


def test_merged_axial_map_collapses_float_noise_duplicate_stations() -> None:
    """An anchor within float noise of a base sample must not duplicate a ring.

    Regression: an anchor at t = 1/3 + 1 ulp next to the uniform base station
    at t = 1/3 produced two coincident axial rings, which made the outer
    offset shell locally degenerate and tripped the normal-flip guard.
    """
    from hornlab_mesher.freeform import build_freeform_geometry
    from hornlab_mesher.profile_sampling import _freeform_merged_axial_map

    profile = {
        "profileH": {
            "points": [[0.0, 12.7], [40.0 + 7.0e-15, 60.0], [120.0, 140.0]],
            "throatAngleDeg": 15.5,
            "mouthAngleDeg": 60.0,
        },
        "profileV": {
            "points": [[0.0, 12.7], [40.0 + 7.0e-15, 60.0], [120.0, 140.0]],
            "throatAngleDeg": 15.5,
            "mouthAngleDeg": 60.0,
        },
        "crossSections": [
            {"t": 0.0, "shape": "circle"},
            {"t": 1.0, "shape": "ellipse"},
        ],
    }
    geometry = build_freeform_geometry(profile)
    t_values, _mode = _freeform_merged_axial_map(profile, geometry, 3)
    assert t_values[0] == 0.0
    assert t_values[-1] == 1.0
    assert np.all(np.diff(t_values) > 1.0e-7)
