"""Portable points exported by ATH V2025-12, matched by meridian/station."""
from pathlib import Path
import json

import numpy as np
import pytest

from hornlab_mesher.config_builder import ConfigError, build_geometry_params
from hornlab_mesher.config_parser import parse_text_config
from hornlab_mesher.profile_formulas import (
    calculate_osse, calculate_osse_curve, calculate_rosse, calculate_rosse_curve,
)

ROOT = Path(__file__).parent / "fixtures/throat_stretch/ath-v2025-12"
CASES = json.loads((ROOT / "cases.json").read_text())["cases"]


@pytest.mark.parametrize("name", sorted(CASES))
def test_ath_exported_points_or_explicit_import_refusal(name):
    case = CASES[name]
    if case["refused"]:
        with pytest.raises(ConfigError, match="Slot.Length.*not supported"):
            parse_text_config(case["config"])
        return
    params = build_geometry_params(parse_text_config(case["config"]))[0]
    expected_params = dict(case["params"])
    if expected_params["s1"] == 0 or expected_params["s2"] == 0:
        del expected_params["s1"], expected_params["s2"]
    assert params == expected_params
    data = np.loadtxt(ROOT / (name + ".csv"), delimiter=",", skiprows=1)
    scalar, vector = (
        (calculate_osse, calculate_osse_curve) if params["type"] == "OSSE"
        else (calculate_rosse, calculate_rosse_curve)
    )
    for meridian in np.unique(data[:, 0]):
        rows = data[data[:, 0] == meridian]
        phi = rows[0, 2]
        stations = rows[:, 3]
        expected = np.column_stack((rows[:, 6], np.hypot(rows[:, 4], rows[:, 5])))
        for actual in (
            np.array([scalar(t, phi, params) for t in stations]),
            np.column_stack(vector(stations, phi, params)),
        ):
            np.testing.assert_allclose(actual * params["scale"], expected, rtol=0, atol=2.7e-5)


@pytest.mark.parametrize("name", sorted(CASES))
def test_paired_ath_stretch_uses_degrees_main_coordinate_and_post_scale(name):
    case = CASES[name]
    params = case["params"]
    data = np.loadtxt(ROOT / (name + ".csv"), delimiter=",", skiprows=1)
    scale = params["scale"]
    zero_z = data[:, 9] / scale
    prefix = params["throatExtLength"]
    if params["type"] == "R-OSSE":
        prefix += params["slotLength"]
    active = np.ones(len(data), dtype=bool)
    if params["throatExtLength"]:
        active = data[:, 1] >= 10
    expected = zero_z.copy()
    expected[active] += params["s1"] * np.degrees(np.arctan(params["s2"] * (zero_z[active] - prefix)))
    np.testing.assert_allclose(expected * scale, data[:, 6], rtol=0, atol=2e-4)
    np.testing.assert_array_equal(data[:, 4:6], data[:, 7:9])


@pytest.mark.parametrize("extra", [
    "Rot = 10\nThroat.Ext.Length = 12",
    "Rot = 10\nSlot.Length = 8",
    "GCurve.Type = 1\nGCurve.Width = 250\nThroat.Ext.Length = 12",
    "GCurve.Type = 1\nGCurve.Width = 250\nRot = 10",
])
def test_unmeasured_stretch_compositions_refused(extra):
    with pytest.raises(ConfigError, match="unverified|not supported"):
        parse_text_config("OSSE = {\nL = 160\ns1 = 0.5\ns2 = 0.2\n}\n" + extra)


@pytest.mark.parametrize("coefficient", ["0.5"])
def test_unmeasured_flat_slot_stretch_refused(coefficient):
    with pytest.raises(ConfigError, match="OSSE Slot.Length.*unverified"):
        parse_text_config(f"Length = 160\nCoverage.Angle = 40\ns1 = {coefficient}\ns2 = 0.2\nSlot.Length = 8")


def test_unmeasured_rosse_rot_stretch_refused():
    with pytest.raises(ConfigError, match="R-OSSE Rot.*unverified"):
        parse_text_config("R-OSSE = {\nR = 200\ns1 = 1\ns2 = 0.05\n}\nRot = 10")


def test_unmeasured_rosse_length_stretch_refused():
    with pytest.raises(ConfigError, match="top-level Length.*unverified"):
        parse_text_config("R-OSSE = {\nR = 200\ns1 = 1\ns2 = 0.05\n}\nLength = 180")


def test_rot_and_prefix_with_zero_s1_remains_accepted():
    config = parse_text_config("OSSE = {\nL = 160\ns1 = 0\ns2 = 0.2\n}\nRot = 10\nThroat.Ext.Length = 12")
    assert config["profile"]["rot"] == 10


def test_in_block_and_top_level_rotation_share_the_supported_composition():
    inside = parse_text_config("OSSE = {\nL = 160\nRot = 10\ns1 = 0.5\ns2 = 0.2\n}")
    outside = parse_text_config("OSSE = {\nL = 160\ns1 = 0.5\ns2 = 0.2\n}\nRot = 10")
    assert build_geometry_params(inside) == build_geometry_params(outside)


@pytest.mark.parametrize("variant", ["plain", "rotated", "gcurve", "extension"])
def test_prescaled_consumer_coefficient_rule_against_tritonia(variant):
    name = f"osse-tritonia-s-{variant}-stretch"
    params = CASES[name]["params"]
    scale = params["scale"]
    scaled = dict(params)
    for key in ("L", "r0", "throatExtLength", "slotLength", "gcurveWidth", "s1"):
        scaled[key] *= scale
    scaled["s2"] /= scale
    data = np.loadtxt(ROOT / (name + ".csv"), delimiter=",", skiprows=1)
    actual = np.array([calculate_osse(row[3] * scale, row[2], scaled) for row in data])
    expected = np.column_stack((data[:, 6], np.hypot(data[:, 4], data[:, 5])))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-4)
    wrong = {**scaled, "s2": params["s2"]}
    assert abs(calculate_osse(data[-1, 3] * scale, data[-1, 2], wrong)[0] - actual[-1, 0]) > 1
