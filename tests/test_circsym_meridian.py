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
    inner = np.asarray(grid["inner_points"], dtype=np.float64).reshape(n_phi, n_length + 1, 3)
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

    assert meridian.metadata["throatRadiusM"] == pytest.approx(throat_radius_m, abs=1.0e-12)
    assert meridian.metadata["mouthRadiusM"] == pytest.approx(mouth_radius_m, abs=1.0e-12)
    assert grid_profile[0, 1] * 0.001 == pytest.approx(meridian.metadata["throatRadiusM"], abs=1.0e-12)
    assert grid_profile[-1, 1] * 0.001 == pytest.approx(meridian.metadata["mouthRadiusM"], abs=1.0e-12)

    expected_cap_radius_m = 0.08
    expected_cap_height_m = (
        80.0 - math.sqrt(80.0 * 80.0 - float(sampled_profile[0, 1]) ** 2)
    ) * 0.001
    assert meridian.metadata["sourceCapRadiusM"] == pytest.approx(expected_cap_radius_m, abs=1.0e-12)
    assert meridian.metadata["sourceCapHeightM"] == pytest.approx(expected_cap_height_m, abs=1.0e-12)

    source_count = int(meridian.metadata["sourceSegmentCount"])
    source_nodes = meridian.nodes[: source_count + 1]
    center_z = meridian.metadata["sourceCapCenterZM"]
    cap_radius = meridian.metadata["sourceCapRadiusM"]
    assert source_nodes[0, 0] == pytest.approx(0.0, abs=1.0e-15)
    assert source_nodes[-1, 0] == pytest.approx(throat_radius_m, abs=1.0e-12)
    assert source_nodes[-1, 1] == pytest.approx(float(sampled_profile[0, 0]) * 0.001, abs=1.0e-12)
    assert np.max(np.abs(source_nodes[:, 0] ** 2 + (source_nodes[:, 1] - center_z) ** 2 - cap_radius**2)) < 1.0e-14

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

    assert inner_nodes[-1, 0] == pytest.approx(meridian.metadata["mouthRadiusM"], abs=1.0e-12)
    assert inner_nodes[-1, 1] == pytest.approx(0.0, abs=1.0e-15)
    assert np.all(inner_nodes[:-1, 1] < -1.0e-9)
    assert aperture_nodes[0, 0] == pytest.approx(meridian.metadata["mouthRadiusM"], abs=1.0e-12)
    assert aperture_nodes[0, 1] == pytest.approx(0.0, abs=1.0e-15)
    assert aperture_nodes[-1, 0] == pytest.approx(0.0, abs=1.0e-15)
    assert np.allclose(aperture_nodes[:, 1], 0.0, atol=1.0e-15)
    assert np.all(meridian.physical_tags[aperture_start:aperture_stop] == aperture_tag)
    assert np.all(meridian.normals[aperture_start:aperture_stop, 1] < 0.0)


def test_build_meridian_adapts_resolution_to_frequency_and_splits_closures():
    freq_max_hz = 20_000.0
    meridian = build_meridian(_round_osse_config(), freq_max_hz=freq_max_hz)
    lengths = np.linalg.norm(
        meridian.nodes[meridian.segments[:, 1]] - meridian.nodes[meridian.segments[:, 0]],
        axis=1,
    )
    source_end = int(meridian.metadata["sourceSegmentCount"])
    inner_end = source_end + int(meridian.metadata["innerSegmentCount"])
    wavelength_m = 343.0 / freq_max_hz

    assert 120 <= meridian.segments.shape[0] <= 220
    assert meridian.metadata["adaptiveLengthSegments"] == 77
    assert meridian.metadata["freqMaxHz"] == pytest.approx(freq_max_hz)
    assert float(np.max(lengths[:inner_end])) <= wavelength_m / 8.0 * (1.0 + 1.0e-12)
    assert int(meridian.metadata["mouthRimSegmentCount"]) > 1
    assert int(meridian.metadata["rearCapSegmentCount"]) > 1
    assert float(np.max(lengths[inner_end:])) <= wavelength_m / 6.0 * (1.0 + 1.0e-12)


def test_build_meridian_rejects_non_circular_config():
    config = _round_osse_config(cross_section={"aspectRatio": 1.15})

    with pytest.raises(ConfigError, match="CircSym requires a circular waveguide: .*aspectRatio"):
        build_meridian(config)


def test_circsym_rejection_reasons_empty_for_eligible_round_config():
    # The authoritative auto-mode gate: a plain round OSSE horn is CircSym-eligible.
    assert circsym_rejection_reasons(_round_osse_config()) == []


def test_circsym_rejection_reasons_flags_non_circular_cross_section():
    reasons = circsym_rejection_reasons(_round_osse_config(cross_section={"aspectRatio": 1.15}))
    assert reasons
    assert any("aspectRatio" in reason for reason in reasons)


def test_circsym_rejection_reasons_empty_for_infinite_baffle_channel_meridian():
    reasons = circsym_rejection_reasons(
        _round_osse_config(mode="infinite-baffle", simType=1, mesh={"wallThickness": 0.0})
    )
    assert reasons == []


def test_circsym_rejection_reasons_flags_enclosure():
    reasons = circsym_rejection_reasons(
        _round_osse_config(mode="enclosure", enclosure={"depth_mm": 60.0})
    )
    assert reasons
    assert any("enclosure" in reason.lower() for reason in reasons)
