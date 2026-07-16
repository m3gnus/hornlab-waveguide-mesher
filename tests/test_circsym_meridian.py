import math

import numpy as np
import pytest

from hornlab_mesher import build_meridian, circsym_rejection_reasons
from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.profiles import build_point_grid, profile_points
from hornlab_mesher.tags import PhysicalGroup


def _round_osse_config(**overrides):
    config = {
        "formula": "OSSE",
        "mode": "freestanding",
        "profile": {
            "formula": "OSSE",
            "L": 120.0,
            "r0": 12.7,
            "a": 45.0,
            "a0": 12.0,
            "k": 1.0,
            "n": 4.0,
            "q": 0.995,
            "s": 0.0,
        },
        "mesh": {
            "angularSegments": 64,
            "lengthSegments": 16,
            "samplingMode": "uniform",
            "quadrants": 1234,
            "wallThickness": 6.0,
            "throatResolution": 3.0,
            "mouthResolution": 12.0,
            "rearResolution": 18.0,
        },
        "cross_section": {
            "exponent": 2.0,
            "aspectRatio": 1.0,
        },
        "morph": {
            "morphTarget": 0,
        },
        "source": {
            "sourceShape": 1,
            "sourceRadius": 80.0,
            "sourceCurv": 1,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


def _inner_grid_profile(params):
    grid = build_point_grid(params)
    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    inner = np.asarray(grid["inner_points"], dtype=np.float64).reshape(
        n_phi, n_length + 1, 3
    )
    radius = np.linalg.norm(inner[:, :, :2], axis=2)
    z = inner[:, :, 2]
    return np.column_stack((np.mean(z, axis=0), np.mean(radius, axis=0)))


def test_build_meridian_matches_round_3d_profile_and_source_cap_geometry():
    config = _round_osse_config()
    meridian = build_meridian(config)
    params, _formula, _mode = build_geometry_params(config)

    sampled_profile = profile_points(params, int(params["lengthSegments"]) + 1, phi=0.0)
    grid_profile = _inner_grid_profile(params)
    throat_radius_m = float(sampled_profile[0, 1]) * 0.001
    mouth_radius_m = float(sampled_profile[-1, 1]) * 0.001

    assert meridian.metadata["throatRadiusM"] == pytest.approx(
        throat_radius_m, abs=1.0e-12
    )
    assert meridian.metadata["mouthRadiusM"] == pytest.approx(
        mouth_radius_m, abs=1.0e-12
    )
    assert grid_profile[0, 1] * 0.001 == pytest.approx(
        meridian.metadata["throatRadiusM"], abs=1.0e-12
    )
    assert grid_profile[-1, 1] * 0.001 == pytest.approx(
        meridian.metadata["mouthRadiusM"], abs=1.0e-12
    )

    expected_cap_radius_m = 0.08
    expected_cap_height_m = (
        80.0 - math.sqrt(80.0 * 80.0 - float(sampled_profile[0, 1]) ** 2)
    ) * 0.001
    assert meridian.metadata["sourceCapRadiusM"] == pytest.approx(
        expected_cap_radius_m, abs=1.0e-12
    )
    assert meridian.metadata["sourceCapHeightM"] == pytest.approx(
        expected_cap_height_m, abs=1.0e-12
    )

    source_count = int(meridian.metadata["sourceSegmentCount"])
    source_nodes = meridian.nodes[: source_count + 1]
    center_z = meridian.metadata["sourceCapCenterZM"]
    cap_radius = meridian.metadata["sourceCapRadiusM"]
    assert source_nodes[0, 0] == pytest.approx(0.0, abs=1.0e-15)
    assert source_nodes[-1, 0] == pytest.approx(throat_radius_m, abs=1.0e-12)
    assert source_nodes[-1, 1] == pytest.approx(
        float(sampled_profile[0, 0]) * 0.001, abs=1.0e-12
    )
    assert (
        np.max(
            np.abs(
                source_nodes[:, 0] ** 2
                + (source_nodes[:, 1] - center_z) ** 2
                - cap_radius**2
            )
        )
        < 1.0e-14
    )

    assert set(meridian.physical_tags.tolist()) == {1, 2}
    assert np.all(meridian.physical_tags[:source_count] == 2)
    assert np.all(meridian.physical_tags[source_count:] == 1)
    assert np.all(meridian.normals[:source_count, 1] > 0.0)
    assert meridian.metadata["closedOnAxis"] is True


def test_build_meridian_builds_infinite_baffle_channel_with_aperture_disc():
    config = _round_osse_config(
        mode="infinite-baffle",
        simType=1,
        mesh={"wallThickness": 0.0},
    )

    meridian = build_meridian(config)
    aperture_tag = int(PhysicalGroup.MOUTH_APERTURE)
    source_count = int(meridian.metadata["sourceSegmentCount"])
    inner_count = int(meridian.metadata["innerSegmentCount"])
    aperture_count = int(meridian.metadata["apertureSegmentCount"])
    aperture_start = source_count + inner_count
    aperture_stop = aperture_start + aperture_count
    inner_nodes = meridian.nodes[source_count : aperture_start + 1]
    aperture_nodes = meridian.nodes[aperture_start : aperture_stop + 1]

    assert meridian.metadata["apertureTag"] == aperture_tag
    assert aperture_count > 0
    assert aperture_tag in set(meridian.physical_tags.tolist())
    assert meridian.baffle_z is None
    assert meridian.metadata["baffleZM"] is None

    assert inner_nodes[-1, 0] == pytest.approx(
        meridian.metadata["mouthRadiusM"], abs=1.0e-12
    )
    assert inner_nodes[-1, 1] == pytest.approx(0.0, abs=1.0e-15)
    assert np.all(inner_nodes[:-1, 1] < -1.0e-9)
    assert aperture_nodes[0, 0] == pytest.approx(
        meridian.metadata["mouthRadiusM"], abs=1.0e-12
    )
    assert aperture_nodes[0, 1] == pytest.approx(0.0, abs=1.0e-15)
    assert aperture_nodes[-1, 0] == pytest.approx(0.0, abs=1.0e-15)
    assert np.allclose(aperture_nodes[:, 1], 0.0, atol=1.0e-15)
    assert np.all(meridian.physical_tags[aperture_start:aperture_stop] == aperture_tag)
    assert np.all(meridian.normals[aperture_start:aperture_stop, 1] < 0.0)


def test_build_meridian_uses_mm_resolution():
    fine = build_meridian(_round_osse_config())
    coarse = build_meridian(
        _round_osse_config(
            mesh={
                "throatResolution": 12.0,
                "mouthResolution": 24.0,
                "rearResolution": 36.0,
            }
        )
    )
    lengths = np.linalg.norm(
        fine.nodes[fine.segments[:, 1]] - fine.nodes[fine.segments[:, 0]],
        axis=1,
    )
    assert fine.segments.shape[0] > 2 * coarse.segments.shape[0]
    assert fine.metadata["throatTargetSegmentM"] == pytest.approx(0.003)
    assert fine.metadata["mouthTargetSegmentM"] == pytest.approx(0.012)
    assert fine.metadata["outerTargetSegmentM"] == pytest.approx(0.018)
    assert "freqMaxHz" not in fine.metadata
    assert "wavelengthM" not in fine.metadata
    assert float(np.max(lengths)) <= 0.018 * 1.01


def test_removed_circsym_frequency_kwarg_has_migration_error():
    with pytest.raises(ConfigError, match="millimetre-only mesh contract"):
        build_meridian(_round_osse_config(), freq_max_hz=20_000.0)
    reasons = circsym_rejection_reasons(_round_osse_config(), freq_max_hz=20_000.0)
    assert reasons and "millimetre-only mesh contract" in reasons[0]


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("throatResolution", 0.0),
        ("mouthResolution", -1.0),
        ("rearResolution", 0.0),
        ("apertureResolutionScale", 0.0),
        ("apertureResolutionScale", 0.5),
    ),
)
def test_build_meridian_rejects_invalid_mm_controls_before_resampling(key, value):
    config = _round_osse_config()
    config["mesh"][key] = value

    with pytest.raises(ConfigError, match="must be finite and"):
        build_meridian(config)
    assert circsym_rejection_reasons(config)


def test_circsym_geometry_sampling_does_not_force_meridian_segments():
    counts = {
        build_meridian(
            _round_osse_config(mesh={"lengthSegments": value})
        ).segments.shape[0]
        for value in (8, 16, 64, 256)
    }
    assert counts == {45}


def test_build_meridian_rejects_non_circular_config():
    config = _round_osse_config(cross_section={"aspectRatio": 1.15})

    with pytest.raises(
        ConfigError, match="CircSym requires a circular waveguide: .*aspectRatio"
    ):
        build_meridian(config)


def test_circsym_rejection_reasons_empty_for_eligible_round_config():
    # The authoritative auto-mode gate: a plain round OSSE horn is CircSym-eligible.
    assert circsym_rejection_reasons(_round_osse_config()) == []


def test_circsym_rejection_reasons_flags_non_circular_cross_section():
    reasons = circsym_rejection_reasons(
        _round_osse_config(cross_section={"aspectRatio": 1.15})
    )
    assert reasons
    assert any("aspectRatio" in reason for reason in reasons)


def test_circsym_rejection_reasons_empty_for_infinite_baffle_channel_meridian():
    reasons = circsym_rejection_reasons(
        _round_osse_config(
            mode="infinite-baffle", simType=1, mesh={"wallThickness": 0.0}
        )
    )
    assert reasons == []


def test_circsym_rejection_reasons_flags_enclosure():
    reasons = circsym_rejection_reasons(
        _round_osse_config(mode="enclosure", enclosure={"depth_mm": 60.0})
    )
    assert reasons
    assert any("enclosure" in reason.lower() for reason in reasons)


@pytest.mark.parametrize(
    "mesh_override",
    [
        {"subdomainSlices": "8", "interfaceOffset": 5.0},
        {"interfaceOffset": 5.0},
    ],
)
def test_circsym_rejects_subdomain_interface_controls(mesh_override):
    config = _round_osse_config(mesh=mesh_override)

    reasons = circsym_rejection_reasons(config)
    assert any("subdomain interfaces" in reason for reason in reasons)
    with pytest.raises(ConfigError, match="subdomain interfaces"):
        build_meridian(config)


def test_circsym_rejects_unsupported_source_shape():
    config = _round_osse_config(source={"sourceShape": 2})

    with pytest.raises(ConfigError, match="source_shape=2 is not supported"):
        build_meridian(config)


def test_circsym_reports_positive_driven_source_measure():
    meridian = build_meridian(_round_osse_config())

    assert meridian.metadata["sourceSegmentCount"] > 0
    assert meridian.metadata["sourceSweptAreaM2"] > 0.0


def test_freestanding_circsym_requires_positive_wall_thickness():
    config = _round_osse_config(mesh={"wallThickness": 0.0})

    with pytest.raises(ConfigError, match="freestanding mode requires.*> 0"):
        build_meridian(config)


def test_rosse_freestanding_lip_closure_matches_ath_semicircle_nodes():
    config = {
        "formula": "R-OSSE",
        "mode": "freestanding",
        "profile": {
            "R": 250.0,
            "a": 41.0,
            "a0": 15.5,
            "b": 0.3,
            "k": 1.0,
            "m": 0.8,
            "q": 3.7,
            "r": 0.3,
            "r0": 12.7,
        },
        "mesh": {
            "lengthSegments": 256,
            "wallThickness": 5.0,
            "samplingMode": "ath-default-zmap",
            "topology": "legacy",
        },
        "source": {"sourceShape": 1, "sourceRadius": -1.0, "sourceCurv": 0},
    }

    meridian = build_meridian(config)
    source_count = int(meridian.metadata["sourceSegmentCount"])
    inner_count = int(meridian.metadata["innerSegmentCount"])
    rim_count = int(meridian.metadata["mouthRimSegmentCount"])
    rim_start = source_count + inner_count
    actual_z_r_mm = (
        meridian.nodes[
            rim_start : rim_start + rim_count + 1,
            [1, 0],
        ]
        * 1000.0
    )
    # ATH V2025-12 nodes 257..262 from the supplied R-OSSE CircSym project.
    expected_z_r_mm = np.asarray(
        [
            [97.692, 250.000],
            [96.216, 249.542],
            [95.292, 248.303],
            [95.272, 246.758],
            [96.164, 245.497],
            [97.627, 245.000],
        ],
        dtype=np.float64,
    )

    assert rim_count == 5
    assert np.allclose(actual_z_r_mm, expected_z_r_mm, rtol=0.0, atol=0.02)
