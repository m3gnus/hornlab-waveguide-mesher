"""Saved application CFGs keep their geometry contract on the direct CLI path."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import meshio

from hornlab_mesher import cli
from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.config_parser import ConfigError, load_config, parse_text_config
from hornlab_mesher.profile_formulas import calculate_osse, calculate_osse_curve
from hornlab_mesher.profile_sampling import build_point_grid_arrays
from hornlab_mesher.text_import import TEXT_IMPORT_VERSION, TEXT_IMPORT_VERSION_KEY

BASE = """OSSE = {
L = 30
r0 = 4
a = 32
a0 = 6
k = 1
s = 0
n = 4
q = .995
}
ABEC.SimType = 2
Mesh.AngularSegments = 32
Mesh.LengthSegments = 24
Mesh.SamplingMode = uniform
Mesh.WallThickness = 0
Mesh.ThroatResolution = 2
Mesh.MouthResolution = 3
Source.Shape = 2
"""


def stamp(version):
    return f"; Waveguide Generator geometry-interpretation: {version}\n"


def raw_mouth_radius():
    return math.sqrt(4**2 + 2*4*30*math.tan(math.radians(6)) + 30**2*math.tan(math.radians(32))**2)


@pytest.mark.parametrize("header,imported", [
    ("", True), (stamp(TEXT_IMPORT_VERSION), True),
    (stamp("native-v1"), False), ("; Parameter config\n", False),
    ("; mWg CONFIG\n", False),
    ("; Parameter config\n" + stamp(TEXT_IMPORT_VERSION), True),
    (stamp("native-v1") + stamp("native-v1"), False),
])
@pytest.mark.parametrize("control", ["circle", "slot"])
def test_load_saved_cfg_and_json_roundtrip_keep_physical_interpretation(tmp_path, header, imported, control):
    cfg = tmp_path / "saved.cfg"
    extra = ("Morph.TargetShape = 2\nMorph.TargetWidth = 60\nMorph.TargetHeight = 60\n"
             if control == "circle" else "Slot.Length = 5\n")
    cfg.write_text(header + BASE + extra, encoding="utf-8")
    loaded = load_config(cfg)
    assert loaded.get(TEXT_IMPORT_VERSION_KEY) == (TEXT_IMPORT_VERSION if imported else None)
    json_path = tmp_path / "saved.json"
    json_path.write_text(json.dumps(loaded), encoding="utf-8")
    for config in (loaded, load_config(json_path)):
        p = build_geometry_params(config)[0]
        grid = build_point_grid_arrays(p)["inner_grid"]
        if control == "circle":
            raw_mouth = raw_mouth_radius()
            assert np.hypot(grid[:, -1, 0], grid[:, -1, 1]) == pytest.approx(
                raw_mouth if imported else 30, abs=1e-9)
        else:
            assert calculate_osse(5, 0, p)[1] == pytest.approx(
                4 + 5 * math.tan(math.radians(6)) if imported else 4, abs=1e-12)
            assert grid[0, -1, 2] == pytest.approx(30 if imported else 35, abs=1e-12)


def test_native_saved_cfg_uses_native_omitted_defaults_and_explicit_length_mode():
    text = stamp("native-v1") + BASE.replace("\na0 = 6\n", "\n").replace("\ns = 0\n", "\n")
    text = text.replace("Mesh.SamplingMode = uniform\n", "") + "Morph.TargetShape = 2\n"
    config = parse_text_config(text)
    p = build_geometry_params(config)[0]
    assert (p["a0"], p["s"], p["morphCorner"], p["morphFixed"], p["samplingMode"]) == (15.5, 0, 0, 0, "uniform")
    assert "_athLengthMode" not in p
    assert "_morphKeepsSlot" not in p
    total = parse_text_config(text + "Length.Mode = total\nSlot.Length = 5\n")
    assert build_point_grid_arrays(build_geometry_params(total)[0])["inner_grid"][0, -1, 2] == pytest.approx(30)


@pytest.mark.parametrize("length_mode", [None, "total"])
def test_native_application_saved_flat_slot_matches_translated_physical_profile(tmp_path, length_mode):
    # Counterpart of the application's canonical writer/preview translation:
    # an empty OSSE selector, flat aliases and optional authored Length.Mode.
    text = stamp("native-v1") + """; Parameter config
OSSE = {
}
Slot.Length = 5
Coverage.Angle = 32
Length = 30
Term.n = 4
Term.q = .995
Term.s = 0
Throat.Angle = 6
Throat.Diameter = 8
Throat.Profile = 1
OS.k = 1
Mesh.AngularSegments = 32
Mesh.LengthSegments = 24
Mesh.SamplingMode = uniform
Mesh.WallThickness = 0
ABEC.SimType = 2
"""
    if length_mode is not None:
        text += f"Length.Mode = {length_mode}\n"
    path = tmp_path / "application.cfg"
    path.write_text(text, encoding="utf-8")
    loaded = load_config(path)
    counterpart = {"formula": "OSSE", "profile": {
        "L": 30, "r0": 4, "a": 32, "a0": 6, "k": 1,
        "n": 4, "q": .995, "s": 0, "slotLength": 5},
        "mesh": {"angularSegments": 32, "lengthSegments": 24,
                 "samplingMode": "uniform", "wallThickness": 0}, "simType": 2}
    if length_mode is not None:
        counterpart["profile"]["_athLengthMode"] = length_mode
    expected, actual = (build_geometry_params(config)[0] for config in (counterpart, loaded))
    assert actual.get("_athLengthMode") == expected.get("_athLengthMode")
    assert TEXT_IMPORT_VERSION_KEY not in loaded
    for phi in (0, .4, math.pi/2):
        stations = np.linspace(0, 30 if length_mode == "total" else 35, 301)
        assert np.array_equal(calculate_osse_curve(stations, phi, actual),
                              calculate_osse_curve(stations, phi, expected))
    assert np.array_equal(build_point_grid_arrays(actual)["inner_grid"],
                          build_point_grid_arrays(expected)["inner_grid"])


@pytest.mark.parametrize("version", ["native-v1", TEXT_IMPORT_VERSION])
@pytest.mark.parametrize("control", ["circle", "slot"])
def test_direct_cli_builds_saved_cfg_physical_dimensions(tmp_path, capfd, version, control):
    extra = ("Morph.TargetShape = 2\nMorph.TargetWidth = 60\nMorph.TargetHeight = 60\n"
             if control == "circle" else "Slot.Length = 5\n")
    cfg, output = tmp_path / "saved.cfg", tmp_path / "saved.msh"
    cfg.write_text(stamp(version) + BASE.replace("Mesh.WallThickness = 0", "Mesh.WallThickness = 2") + extra, encoding="utf-8")
    assert cli.main([str(cfg), "-o", str(output)]) == 0
    _out, err = capfd.readouterr()
    assert "error:" not in err
    mesh = meshio.read(output)
    assert np.isfinite(mesh.points).all()
    imported = version == TEXT_IMPORT_VERSION
    if control == "circle":
        mouth = raw_mouth_radius() if imported else 30
        assert np.min(np.linalg.norm(mesh.points - np.array([mouth, 0, 30])/1000, axis=1)) < 1e-7
    else:
        assert np.max(mesh.points[:, 2]) == pytest.approx((30 if imported else 35) / 1000, abs=1e-7)


@pytest.mark.parametrize("header,message", [
    (stamp("future-v99"), "unsupported geometry interpretation"),
    ("; Parameter config\n" + stamp("future-v99"), "unsupported geometry interpretation"),
    (stamp("native-v1") + stamp(TEXT_IMPORT_VERSION), "conflicting geometry interpretation"),
])
def test_invalid_stamps_fail_load_and_cli_before_publication(tmp_path, capfd, header, message):
    cfg, output = tmp_path / "saved.cfg", tmp_path / "saved.msh"
    cfg.write_text(header + BASE, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(cfg)
    assert cli.main([str(cfg), "-o", str(output)]) == 2
    _out, err = capfd.readouterr()
    assert message in err
    assert "Traceback" not in err
    assert not output.exists()
