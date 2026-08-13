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
                    "cornerRadiusMm": 13.2,
                },
                {
                    "t": 1.0,
                    "shape": "rounded_rectangle",
                    "cornerRadiusMm": 13.2,
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


@pytest.mark.parametrize("quadrants", ["12", "14"])
def test_half_domains_are_exact_sorted_subsets_of_full_grid(quadrants: str) -> None:
    _params, _grid, full_points, full_phi = _built_grid(_owner_config())
    _params, _grid, reduced_points, reduced_phi = _built_grid(
        _owner_config(quadrants=quadrants)
    )
    reference_phi = full_phi[:, 0]
    if quadrants == "12":
        selected = reference_phi <= math.pi + 1.0e-12
        expected_points = full_points[selected]
        expected_phi = full_phi[selected]
        assert np.all(reduced_phi >= -1.0e-14)
        assert np.all(reduced_phi <= math.pi + 1.0e-14)
    else:
        selected = (reference_phi <= math.pi / 2.0 + 1.0e-12) | (
            reference_phi >= 3.0 * math.pi / 2.0 - 1.0e-12
        )
        selected_points = full_points[selected]
        selected_phi = np.where(
            full_phi[selected] > math.pi,
            full_phi[selected] - math.tau,
            full_phi[selected],
        )
        order = np.argsort(selected_phi[:, 0])
        expected_points = selected_points[order]
        expected_phi = selected_phi[order]
        assert np.all(reduced_phi >= -math.pi / 2.0 - 1.0e-14)
        assert np.all(reduced_phi <= math.pi / 2.0 + 1.0e-14)
        assert np.all(np.diff(reduced_phi, axis=0) > 0.0)
    np.testing.assert_allclose(
        reduced_points, expected_points, rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(reduced_phi, expected_phi, rtol=0.0, atol=1.0e-14)


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
    for station in config["profile"]["crossSections"][1:]:
        station["cornerRadiusMm"] = 10.0
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


@pytest.mark.parametrize(
    "corner_radius_mm",
    [5.9, 13.2, 20.0, 30.0, 40.0],
)
def test_per_ring_azimuth_invariants_across_corner_radius_sweep(
    corner_radius_mm: float,
) -> None:
    config = _owner_config()
    config["mesh"].update(angularSegments=64, lengthSegments=32)
    for station in config["profile"]["crossSections"][1:]:
        station["cornerRadiusMm"] = corner_radius_mm

    _params, _grid, points, phi_grid = _built_grid(config)

    assert np.all(np.diff(phi_grid, axis=0) > 0.0)
    ring_edges = np.roll(points, -1, axis=0) - points
    assert np.all(np.linalg.norm(ring_edges, axis=2) > 0.0)
    for ring_phi in phi_grid.T:
        assert np.any(np.isclose(ring_phi, 0.0, rtol=0.0, atol=1.0e-14))
        assert np.any(
            np.isclose(ring_phi, math.pi / 2.0, rtol=0.0, atol=1.0e-14)
        )


def test_circular_mouth_keeps_vertical_meridian_and_walled_grid_builds() -> None:
    circular = _owner_config()
    shared_profile = {
        "points": [[0.0, 80.0], [120.0, 80.0]],
        "throatAngleDeg": 0.0,
        "mouthAngleDeg": 0.0,
    }
    circular["profile"]["profileH"] = copy.deepcopy(shared_profile)
    circular["profile"]["profileV"] = copy.deepcopy(shared_profile)
    circular["profile"]["crossSections"] = [
        {"t": 0.0, "shape": "circle"},
        {"t": 1.0, "shape": "rounded_rectangle", "cornerRadiusMm": 80.0},
    ]
    circular["mesh"].update(angularSegments=64, lengthSegments=32)

    _params, _grid, points, phi_grid = _built_grid(circular)
    assert np.all(np.diff(phi_grid, axis=0) > 0.0)
    assert np.any(
        np.isclose(phi_grid[:, -1], math.pi / 2.0, rtol=0.0, atol=1.0e-14)
    )
    assert np.all(
        np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=2) > 0.0
    )

    walled = _owner_config()
    walled["mode"] = "freestanding"
    walled["mesh"].update(
        angularSegments=64,
        lengthSegments=32,
        wall_thickness_mm=3.0,
    )
    for station in walled["profile"]["crossSections"][1:]:
        station["cornerRadiusMm"] = 30.0
    _built_grid(walled)


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
    config["profile"]["inflectionPolicy"] = "warn"
    params, formula, _mode = build_geometry_params(config)
    assert formula == "FREEFORM"
    for key in ("profileH", "profileV", "crossSections", "inflectionPolicy"):
        assert params[key] == config["profile"][key]

    accepted_zero = copy.deepcopy(config)
    accepted_zero["morph"] = {"morphTarget": 0}
    assert build_geometry_params(accepted_zero)[1] == "FREEFORM"

    rectangle_morph = copy.deepcopy(config)
    rectangle_morph["morph"] = {"morphTarget": 1}
    with pytest.raises(ConfigError, match="FREEFORM.*crossSections"):
        build_geometry_params(rectangle_morph)

    expression_morph = copy.deepcopy(config)
    expression_morph["morph"] = {
        "morphTarget": "1 if 0.01 < p < 0.02 else 0"
    }
    with pytest.raises(ConfigError, match="cannot be proven inactive"):
        build_geometry_params(expression_morph)

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

    scaled_small_source = copy.deepcopy(config)
    scaled_small_source["scale"] = 2.0
    scaled_small_source["source"] = {"sourceShape": 1, "sourceRadius": 20.0}
    with pytest.raises(ConfigError, match=r"sourceRadius.*scaled throat.*25\.4"):
        build_geometry_params(scaled_small_source)

    scaled_valid_source = copy.deepcopy(scaled_small_source)
    scaled_valid_source["source"]["sourceRadius"] = 26.0
    assert build_geometry_params(scaled_valid_source)[1] == "FREEFORM"

    misspelled_station = copy.deepcopy(config)
    misspelled_station["profile"]["crossSections"][-1]["exponant"] = 16.0
    with pytest.raises(ConfigError, match=r"crossSections\[2\].*exponant"):
        build_geometry_params(misspelled_station)

    misspelled_profile = copy.deepcopy(config)
    misspelled_profile["profile"]["profileH"]["throatAngleDe"] = 45.0
    with pytest.raises(ConfigError, match=r"profileH.*throatAngleDe"):
        build_geometry_params(misspelled_profile)


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


@pytest.mark.parametrize("formula", ["OSSE", "FREEFORM"])
def test_full_sample_zmap_interpolates_after_length_retarget(formula: str) -> None:
    samples = [0.0, 0.02, 0.08, 0.18, 0.32, 0.5, 0.68, 0.84, 1.0]
    if formula == "FREEFORM":
        config = _owner_config()
    else:
        config = {
            "formula": "OSSE",
            "profile": {
                "L_mm": 120.0,
                "r0_mm": 12.7,
                "a_deg": 45.0,
                "a0_deg": 15.5,
            },
            "mesh": {"angularSegments": 16, "lengthSegments": 8},
        }
    config["mesh"].update(
        samplingMode="zmap",
        zMapPoints=samples,
        lengthSegments=8,
    )
    params, _formula, _mode = build_geometry_params(config)

    from hornlab_mesher.profile_sampling import _axial_sample_map

    assert params["zMapKind"] == "samples"
    for steps in (8, 9, 16, 691):
        actual, mode = _axial_sample_map(steps, params)
        expected = np.interp(
            np.linspace(0.0, 1.0, steps + 1),
            np.linspace(0.0, 1.0, len(samples)),
            samples,
        )
        assert mode == "zmap"
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)
        assert np.all(np.diff(actual) >= 0.0)


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
    with pytest.raises(ValueError, match="FREEFORM.*crossSections"):
        build_point_grid({**params, "morphTarget": 1})
    with pytest.raises(ValueError, match="FREEFORM.*guiding curves"):
        build_point_grid({**params, "gcurveType": 1, "gcurveWidth": 100.0})
    with pytest.raises(ValueError, match="cannot be proven inactive"):
        build_point_grid(
            {
                **params,
                "morphTarget": "1 if 0.01 < p < 0.02 else 0",
            }
        )


@pytest.mark.parametrize(
    "source_field",
    ["sourceShape", "sourceRadius"],
)
def test_non_finite_source_fields_raise_config_error(source_field: str) -> None:
    config = _owner_config()
    config["source"] = {source_field: "1e309"}

    with pytest.raises(ConfigError, match=source_field):
        build_geometry_params(config)


def test_oversized_anchor_integer_reports_the_offending_field() -> None:
    config = _owner_config()
    config["profile"]["profileH"]["points"][0][0] = 10**1000

    with pytest.raises(ConfigError, match=r"profileH\.points\[0\]\.z"):
        build_geometry_params(config)


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


def test_merged_axial_map_snaps_base_sample_to_exact_semantic_station() -> None:
    config = _owner_config()
    config["mesh"]["lengthSegments"] = 32
    config["profile"]["crossSections"].insert(
        1,
        {
            "t": 0.375000005,
            "shape": "rounded_rectangle",
            "cornerRadiusMm": 13.2,
        },
    )
    params, _formula, _mode = build_geometry_params(config)
    geometry = build_freeform_geometry(params)

    from hornlab_mesher.profile_sampling import _freeform_merged_axial_map

    t_values, _sampling_mode = _freeform_merged_axial_map(params, geometry, 32)
    assert 0.375000005 in t_values
    assert 0.375 not in t_values
    assert len(t_values) == 34


def test_merged_axial_map_rejects_distinct_nearby_semantic_features() -> None:
    config = _owner_config()
    config["profile"]["crossSections"] = [
        {"t": 0.0, "shape": "circle"},
        {"t": 0.5, "shape": "ellipse"},
        {"t": 0.50000005, "shape": "superellipse", "exponent": 2.0},
        {"t": 1.0, "shape": "ellipse"},
    ]
    params, _formula, _mode = build_geometry_params(config)
    geometry = build_freeform_geometry(params)

    from hornlab_mesher.profile_sampling import _freeform_merged_axial_map

    with pytest.raises(
        ValueError,
        match=r"crossSections\[1\].*0\.5.*crossSections\[2\].*0\.50000005",
    ):
        _freeform_merged_axial_map(params, geometry, 32)
