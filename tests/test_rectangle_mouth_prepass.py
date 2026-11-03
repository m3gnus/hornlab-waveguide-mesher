"""Rectangle layout measures only the mouth before evaluating the final grid."""

import json
from pathlib import Path

import numpy as np
import pytest

import hornlab_mesher.profile_sampling as sampling

PARAMS = {
    "type": "OSSE",
    "L": 100,
    "r0": 12.7,
    "a": 45,
    "a0": 10,
    "angularSegments": 64,
    "lengthSegments": 32,
    "wallThickness": 0,
    "morphTarget": 1,
    "morphWidth": 0,
    "morphHeight": 0,
    "morphCorner": 10,
}


@pytest.mark.parametrize("quadrants", ["1234", "12", "14", "1"])
@pytest.mark.parametrize("prefix", [False, True])
def test_implicit_rectangle_prepass_evaluates_only_mouth_stations(
    monkeypatch, quadrants, prefix
):
    params = {**PARAMS, "quadrants": quadrants}
    if prefix:
        params.update(throatExtLength=12, throatExtAngle=3, slotLength=8)
    calls = []
    original = sampling._raw_radial_grid

    def counted(*args, **kwargs):
        calls.append((len(args[1]), len(args[2])))
        return original(*args, **kwargs)

    monkeypatch.setattr(sampling, "_raw_radial_grid", counted)
    result = sampling.build_point_grid_arrays(params)
    assert len(calls) == 2
    assert calls[0][1] == 1
    assert calls[1] == (result["grid_n_phi"], 33)
    assert (
        sum(phi * stations for phi, stations in calls) == calls[0][0] + calls[1][0] * 33
    )


@pytest.mark.parametrize("quadrants", ["1234", "12", "14", "1"])
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"throatExtLength": 12, "throatExtAngle": 3, "slotLength": 8},
        {"morphWidth": 260},
        {"morphHeight": 240},
        {"morphCorner": 0},
        {"wallThickness": 4},
        {"samplingMode": "ath-default-zmap"},
        {"angularSegments": 8, "lengthSegments": 1},
    ],
)
def test_prepass_preserves_complete_arrays_and_metadata(
    monkeypatch, quadrants, overrides
):
    params = {**PARAMS, "quadrants": quadrants, **overrides}
    assert sampling._use_rectangle_mouth_prepass(params, "OSSE", 2, 1)
    actual = sampling.build_point_grid_arrays(params)
    monkeypatch.setattr(sampling, "_use_rectangle_mouth_prepass", lambda *a: False)
    expected = sampling.build_point_grid_arrays(params)
    assert actual.keys() == expected.keys()
    for key in expected:
        if isinstance(expected[key], np.ndarray):
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        else:
            assert actual[key] == expected[key], key


@pytest.mark.parametrize(
    "overrides",
    [
        {"profileSystem": {"crossSection": {"exponent": 4, "aspectRatio": 1.5}}},
        {"a": "35+cos(p)"},
        {"s1": 0.45, "s2": 0.2},
        {"a0": -50},
        {"throatProfile": 3},
        {"gcurveType": 1, "gcurveWidth": 300},
        {"_textImportVersion": "ath-2026-09a-v3"},
        {"type": "LOOKUP", "lookupProfile": [[0, 12.7], [100, 50]]},
    ],
)
def test_other_meridians_retain_full_preliminary_grid(monkeypatch, overrides):
    params = {**PARAMS, **overrides}
    calls = []
    original = sampling._raw_radial_grid

    def counted(*args, **kwargs):
        calls.append(len(args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(sampling, "_raw_radial_grid", counted)
    try:
        sampling.build_point_grid_arrays(params)
    except ValueError:
        # Invalid meridians must still be checked on their original stations.
        if overrides != {"a0": -50}:
            raise
    assert calls and all(count == 33 for count in calls)


@pytest.mark.parametrize("fixture_name", ["OS01-default", "OS02-morph-enclosure"])
def test_existing_fixture_mesh_bytes_match_full_prepass(
    tmp_path, monkeypatch, fixture_name
):
    from hornlab_mesher.config_builder import build_from_config

    fixture = (
        Path(__file__).parent / f"fixtures/tessellation_report/{fixture_name}.json"
    )
    config = json.loads(fixture.read_text())
    actual = build_from_config(config, tmp_path / "prepass.msh")
    monkeypatch.setattr(sampling, "_use_rectangle_mouth_prepass", lambda *a: False)
    expected = build_from_config(config, tmp_path / "full.msh")
    assert actual.mesh_path.read_bytes() == expected.mesh_path.read_bytes()


def test_implicit_fixture_mesh_bytes_match_full_prepass(tmp_path, monkeypatch):
    from hornlab_mesher.config_builder import build_from_config, build_geometry_params

    fixture = Path(__file__).parent / "fixtures/throat_stretch/base-compatibility.json"
    config = json.loads(fixture.read_text())["resolved"]["OSSE"]["config"]
    config["morph"] = {
        "morphTarget": 1,
        "morphCorner": 10,
        "morphWidth": 0,
        "morphHeight": 0,
    }
    params, _, _ = build_geometry_params(config)
    assert sampling._use_rectangle_mouth_prepass(
        params, "OSSE", *sampling._cross_section(params)
    )
    actual = build_from_config(config, tmp_path / "prepass.msh")
    monkeypatch.setattr(sampling, "_use_rectangle_mouth_prepass", lambda *a: False)
    expected = build_from_config(config, tmp_path / "full.msh")
    assert actual.mesh_path.read_bytes() == expected.mesh_path.read_bytes()


def test_interior_domain_refusal_precedes_invalid_morph_expression(monkeypatch):
    params = {**PARAMS, "a0": 120, "morphCorner": "bad"}
    with pytest.raises(ValueError) as actual:
        sampling.build_point_grid_arrays(params)
    monkeypatch.setattr(sampling, "_use_rectangle_mouth_prepass", lambda *a: False)
    with pytest.raises(ValueError, match="OS-SE profile is undefined") as expected:
        sampling.build_point_grid_arrays(params)
    assert str(actual.value) == str(expected.value)
