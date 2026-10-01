"""The ATH text importer refuses what it does not read.

An item the importer does not recognise used to be dropped without a word, and
the horn built from the rest of the file was simpler than the one configured.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.config_parser import ConfigError, parse_text_config
from hornlab_mesher.profile_formulas import calculate_osse, calculate_osse_curve

ATH_FIXTURES = Path(__file__).parent / "fixtures" / "ath_import" / "ath-v2026-08c"

FLAT_OSSE = """
Throat.Diameter = 25.4
Throat.Angle = 10
Coverage.Angle = 45
Length = 60
Term.s = 1
ABEC.SimType = 2
"""

ROSSE_BLOCK = """
R-OSSE = {
  R = 150
  a = 45
  r0 = 12.7
  a0 = 10
  k = 1
  m = 0.8
  b = 0.3
  r = 0.4
  q = 3.4
%s}
ABEC.SimType = 2
"""

OSSE_BLOCK = """
OSSE = {
  L = 60
  a = 45
  r0 = 12.7
  a0 = 10
%s}
ABEC.SimType = 2
"""


def _rosse(extra_in_block: str = "") -> str:
    return ROSSE_BLOCK % extra_in_block


def _osse(extra_in_block: str = "") -> str:
    return OSSE_BLOCK % extra_in_block


@pytest.mark.parametrize(
    ("config", "named"),
    [
        (FLAT_OSSE + "Scale.Z = 0.5\n", "Scale.Z"),
        (FLAT_OSSE + "Made.Up = 40\n", "Made.Up"),
        (FLAT_OSSE + "Throat.Ext.Length = 20\nThroat.Ext.Angle = 3\nThroat.Ext.Included = 1\n", "Throat.Ext.Included"),
        (FLAT_OSSE + "Throat.Ext.Ctrl = -41.8,17.7,-31.2,18.8,-18.7,20.4\n", "Throat.Ext.Ctrl"),
        (FLAT_OSSE + "Flange.Length = 10\n", "Flange.Length"),
        (FLAT_OSSE + "Joint.Radius = 0.6\n", "Joint.Radius"),
        (FLAT_OSSE + "Source.Array = 1\n", "Source.Array"),
        (FLAT_OSSE + "Source.MadeUp = 2\n", "Source.MadeUp"),
        (FLAT_OSSE + "HornGeometry = 2\n", "HornGeometry"),
        (FLAT_OSSE + "Profile = coords.txt\n", "Profile"),
        (FLAT_OSSE + "Coverage.Angel = 50\n", "Coverage.Angel"),
        (FLAT_OSSE + "Sorce.Shape = 2\n", "Sorce.Shape"),
        (_rosse() + "Rot = 3\n", "Rot"),
        (_rosse() + "Length = 180\n", "Length"),
        (_rosse() + "Throat.Diameter = 25.4\n", "Throat.Diameter"),
        (_osse() + "Coverage.Angle = 50\n", "Coverage.Angle"),
    ],
)
def test_unknown_top_level_key_is_refused_by_name(config: str, named: str) -> None:
    with pytest.raises(ConfigError, match="unsupported item") as excinfo:
        parse_text_config(config)
    assert named in str(excinfo.value)


@pytest.mark.parametrize(
    ("block", "named"),
    [
        ("Horn.Adapter = {\n  Width = 20\n  Height = 40\n}\n", "Horn.Adapter"),
        ("Horn.Part:1 = {\n  L = 50\n}\n", "Horn.Part:1"),
        ("Mesh.Roundover = {\n  Radius = 25\n  Segments = 5\n}\n", "Mesh.Roundover"),
        ("Mesh.MadeUp = {\n  Depth = 200\n}\n", "Mesh.MadeUp"),
        ("BEE = {\n  Height = 500\n  Port = {\n    z = 40\n  }\n}\n", "BEE"),
        ("Vanes = {\n  Points = 16\n}\n", "Vanes"),
        ("MadeUp = {\n  x = 1\n}\n", "MadeUp"),
        ("FusionRotaryProfile = {\n  Thickness = 4\n}\n", "FusionRotaryProfile"),
        ("my_plan = {\n  point P0 100 0 8\n  point PB 0 270 20\n}\n", "my_plan"),
    ],
)
def test_unknown_block_is_refused_by_name(block: str, named: str) -> None:
    with pytest.raises(ConfigError, match="unsupported item") as excinfo:
        parse_text_config(FLAT_OSSE + block)
    assert named in str(excinfo.value)


@pytest.mark.parametrize("key", ["arcterm", "trunc", "made_up", "L", "n", "Rot", "Scale"])
def test_unknown_key_inside_rosse_block_is_refused(key: str) -> None:
    with pytest.raises(ConfigError, match=rf"R-OSSE\.{key}\b"):
        parse_text_config(_rosse(f"  {key} = 1\n"))


@pytest.mark.parametrize("key", ["R", "m", "b", "tmax", "arcterm", "Scale"])
def test_unknown_key_inside_osse_block_is_refused(key: str) -> None:
    with pytest.raises(ConfigError, match=rf"OSSE\.{key}\b"):
        parse_text_config(_osse(f"  {key} = 1\n"))


def test_two_profile_blocks_are_refused() -> None:
    with pytest.raises(ConfigError, match="second profile"):
        parse_text_config(_rosse() + "OSSE = {\n  L = 60\n}\n")


@pytest.mark.parametrize("family", ["OSSE", "R-OSSE", "ROSSE"])
def test_repeated_profile_blocks_are_refused(family: str) -> None:
    dimension = "L" if family == "OSSE" else "R"
    text = f"{family} = {{\n{dimension} = 60\n}}\n{family} = {{\n{dimension} = 120\n}}\n"
    with pytest.raises(ConfigError, match=f"second profile beside {family}"):
        parse_text_config(text)


@pytest.mark.parametrize("in_block", [None, 0, 5])
@pytest.mark.parametrize("top_first", [False, True])
def test_osse_rotation_precedence(in_block: int | None, top_first: bool) -> None:
    block = _osse("" if in_block is None else f"  Rot = {in_block}\n")
    text = "Rot = 10\n" + block if top_first else block + "Rot = 10\n"
    expected = parse_text_config(_osse(f"  Rot = {10 if in_block is None else in_block}\n"))
    assert parse_text_config(text) == expected


@pytest.mark.parametrize("block", [_osse, _rosse])
def test_in_block_os_se_throat_profile_is_supported(block) -> None:
    assert parse_text_config(block("  Throat.Profile = 1\n")) == parse_text_config(block())
    with pytest.raises(ConfigError, match="Throat.Profile = 3.*only the OS-SE profile"):
        parse_text_config(block("  Throat.Profile = 3\n"))


@pytest.mark.parametrize("rotation", ["0", "0.0", "-0", "0e3"])
@pytest.mark.parametrize("stretch", ["", "  s1 = .5\n  s2 = .2\n"])
def test_rosse_numeric_zero_rotation_is_inactive(rotation: str, stretch: str) -> None:
    text = _rosse(stretch)
    assert parse_text_config(text + f"Rot = {rotation}\n") == parse_text_config(text)


@pytest.mark.parametrize("member", ["Inner", "LFSource.B", "Source.Contours", "Rollback", "OSSE"])
@pytest.mark.parametrize("block", [_osse, _rosse])
def test_disabled_nested_ancestry_ignores_every_descendant(member: str, block) -> None:
    parked = f"  _Parked = {{\n    Middle = {{\n      {member} = {{\n        s1 = -1\n      }}\n    }}\n  }}\n"
    assert parse_text_config(block(parked)) == parse_text_config(block())


def test_disabled_top_level_block_ignores_coefficients_and_descendants() -> None:
    parked = "_Parked = {\ns1 = -1\nInner = {\nSource.Contours = {\nx = 1\n}\n}\n}\n"
    assert parse_text_config(FLAT_OSSE + parked) == parse_text_config(FLAT_OSSE)


def test_active_nested_block_is_refused_with_full_ancestry() -> None:
    text = _osse("  Outer = {\n    Inner = {\n      x = 1\n    }\n  }\n")
    with pytest.raises(ConfigError, match=r"OSSE\.Outer\.Inner"):
        parse_text_config(text)


@pytest.mark.parametrize("enclosure", [
    "Mesh.Enclosure.Plan = myplan\n",
    "Mesh.Enclosure = {\nPlan = myplan\n}\n",
])
def test_stretch_composition_message_precedes_unknown_enclosure(enclosure: str) -> None:
    text = _rosse("  s1 = .5\n  s2 = .2\n") + "Rot = 10\n" + enclosure
    with pytest.raises(ConfigError, match=r"throat stretch with Rot.*R-OSSE Rot"):
        parse_text_config(text)


@pytest.mark.parametrize("key", ["Plan", "Dim", "Width"])
def test_unknown_enclosure_item_is_refused(key: str) -> None:
    block = f"Mesh.Enclosure = {{\n  {key} = 115,125,115,125\n  Depth = 200\n  Spacing = 30,30,30,200\n}}\n"
    with pytest.raises(ConfigError, match=rf"Mesh\.Enclosure\.{key}\b"):
        parse_text_config(FLAT_OSSE + block)
    with pytest.raises(ConfigError, match=rf"Mesh\.Enclosure\.{key}\b"):
        parse_text_config(FLAT_OSSE + f"Mesh.Enclosure.Depth = 200\nMesh.Enclosure.{key} = 1\n")


def test_short_enclosure_spacing_is_refused() -> None:
    block = "Mesh.Enclosure = {\n  Depth = 200\n  Spacing = 30\n}\n"
    with pytest.raises(ConfigError, match="four margins"):
        parse_text_config(FLAT_OSSE + block)


def test_nested_block_inside_the_enclosure_is_refused_by_path() -> None:
    block = "Mesh.Enclosure = {\n  Depth = 200\n  Port = {\n    w = 80\n  }\n  Spacing = 30,30,30,200\n}\n"
    with pytest.raises(ConfigError, match=r"Mesh\.Enclosure\.Port"):
        parse_text_config(FLAT_OSSE + block)


def test_nested_lf_source_keeps_the_multi_source_refusal() -> None:
    block = "Mesh.Enclosure = {\n  Depth = 200\n  LFSource.B = {\n    Radius = 80\n  }\n}\n"
    with pytest.raises(ConfigError, match="multi-source.*LFSource.B"):
        parse_text_config(FLAT_OSSE + block)


def test_nested_block_does_not_end_its_parent() -> None:
    """Members written after a nested block still belong to the outer block."""
    block = (
        "Mesh.Enclosure = {\n"
        "  Depth = 200\n"
        "  _LFSource.B = {\n"
        "    Radius = 80\n"
        "  }\n"
        "  Spacing = 30,40,50,200\n"
        "  EdgeRadius = 40\n"
        "}\n"
    )
    enclosure = parse_text_config(FLAT_OSSE + block)["enclosure"]
    assert enclosure["edge_mm"] == 40
    assert (enclosure["space_l_mm"], enclosure["space_t_mm"], enclosure["space_r_mm"], enclosure["space_b_mm"]) == (
        30,
        40,
        50,
        200,
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (FLAT_OSSE + "}\n", "closes no block"),
        (FLAT_OSSE + "Report = {\n  Title = x\n", "never closed"),
        (FLAT_OSSE + "Mesh.Enclosure {\n  Depth = 200\n}\n", "cannot read"),
        (FLAT_OSSE + "garbage line\n", "cannot read"),
    ],
)
def test_malformed_structure_is_refused(config: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_text_config(config)


IGNORED_ITEMS = """
ABEC.MeshFrequency = 1000
ABEC.NumFrequencies = 40
ABEC.f1 = 400
ABEC.f2 = 20000
ABEC.Abscissa = 1
ABEC.Polars:SPL = {
  MapAngleRange = 0,180,37
  NormAngle = 10
  Distance = 2
}
ABEC.SphericalField = {
  Resolution = 10
}
Output.STL = 1
Output.MSH = 0
Output.ABECProject = 1
Output.SubDir = "demos"
LE = generic25
LE.Voltage = 2.83
Gmsh.Algorithm = 6
OutputRootDir = "somewhere"
Report = {
  Title = "Demo"
  NormAngle = 10
  Width = 1024
}
GridExport:f360 = {
  ExportProfiles = 1
  Delimiter = ";"
  Scale = 0.1
}
FRDExport = {
  NamePrefix = wg
}
"""


def test_output_report_and_solver_items_are_ignored() -> None:
    plain = parse_text_config(FLAT_OSSE)
    assert parse_text_config(FLAT_OSSE + IGNORED_ITEMS) == plain


def test_simulation_settings_are_ignored_and_reported(caplog: pytest.LogCaptureFixture) -> None:
    settings = (
        "Simulation.F1 = 400\nSimulation.F2 = 20000\nSimulation.NumFrequencies = 40\n"
        "Simulation.Polars:H = {\nDistance = 2\nSource.Contours = {\nx = 1\n}\n}\n"
    )
    with caplog.at_level(logging.WARNING, logger="hornlab_mesher.config_parser"):
        assert parse_text_config(FLAT_OSSE + settings) == parse_text_config(FLAT_OSSE)
    for name in ("Simulation.F1", "Simulation.F2", "Simulation.NumFrequencies", "Simulation.Polars:H"):
        assert name in caplog.text
    assert "do not change geometry or the source" in caplog.text


@pytest.mark.parametrize("block", [lambda: FLAT_OSSE, _osse, _rosse])
def test_circular_arc_angle_is_dormant_for_supported_profiles(block, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="hornlab_mesher.config_parser"):
        assert parse_text_config(block() + "CircArc.TermAngle = 1\n") == parse_text_config(block())
    assert "ignoring CircArc.TermAngle" in caplog.text
    assert "only to Throat.Profile = 3" in caplog.text


def test_circular_arc_profile_is_still_refused() -> None:
    with pytest.raises(ConfigError, match="Throat.Profile = 3.*only the OS-SE profile"):
        parse_text_config(FLAT_OSSE + "Throat.Profile = 3\nCircArc.TermAngle = 1\n")
    with pytest.raises(ConfigError, match=r"CircArc\.MadeUp"):
        parse_text_config(FLAT_OSSE + "CircArc.MadeUp = 1\n")


def test_underscore_disabled_items_are_ignored() -> None:
    disabled = (
        "_Scale.Z = 1.2\n"
        "_OSSE = {\n  L = 20\n}\n"
        "_Source.Contours = {\n  dome WG0 25 6.71 2.35 -0.54 4 1.2\n}\n"
        "_Mesh.Roundover = {\n  Radius = 25\n  _Inner = {\n    x = 1\n  }\n}\n"
    )
    assert parse_text_config(FLAT_OSSE + disabled) == parse_text_config(FLAT_OSSE)
    assert parse_text_config(_rosse("  _arcterm = 130\n")) == parse_text_config(_rosse())


def test_supported_config_shapes_still_import() -> None:
    config = parse_text_config(
        _rosse("  tmax = 0.9\n")
        + "Throat.Ext.Length = 12\nThroat.Ext.Angle = 3\nSlot.Length = 8\nScale = 0.5\n"
        + "Throat.Profile = 1\nSource.Shape = 2\nSource.Velocity = 1\n"
        + "Mesh.Quadrants = 14\nMesh.VerticalOffset = 80\nMesh.SubdomainSlices =\n"
        + "Mesh.Enclosure = {\n  Spacing = 30,30,30,200\n  Depth = 200\n  EdgeRadius = 20\n  EdgeType = 1\n"
        + "  FrontResolution = 8,8,16,16\n  BackResolution = 20,20,20,20\n}\n"
    )
    assert config["formula"] == "R-OSSE"
    assert config["profile"]["tmax"] == 0.9
    assert config["profile"]["throatExtLength"] == 12
    assert config["scale"] == 0.5
    assert config["enclosure"]["depth_mm"] == 200
    assert config["source"]["sourceShape"] == 0


def test_morph_and_gcurve_namespaces_keep_their_warning(caplog: pytest.LogCaptureFixture) -> None:
    """ATH ignores unknown names in these two namespaces; the importer says so."""
    with caplog.at_level(logging.WARNING, logger="hornlab_mesher.config_parser"):
        parse_text_config(FLAT_OSSE + "Morph.TargetShape = 1\nMorph.Width = 300\n")
    assert "Morph.Width" in caplog.text


@pytest.mark.parametrize("key", ["ThroatSegments", "ThroatExtSegments"])
def test_throat_segment_keys_name_the_replacement(key: str) -> None:
    with pytest.raises(ConfigError, match=rf"Mesh\.{key} is not supported.*ZMapPoints"):
        parse_text_config(FLAT_OSSE + f"Mesh.{key} = 4\n")


# --- measured against ATH V2026-08c ---------------------------------------


def _ath_profiles(name: str) -> list[np.ndarray]:
    text = (ATH_FIXTURES / f"{name}.csv").read_text(encoding="utf-8")
    return [
        np.array([[float(value) for value in line.split(";")] for line in block.splitlines()])
        for block in text.strip().split("\n\n")
    ]


def test_ath_ignores_top_level_length_beside_an_osse_block() -> None:
    with_length = (ATH_FIXTURES / "osse-block-with-top-level-length.csv").read_text(encoding="utf-8")
    without_length = (ATH_FIXTURES / "osse-block-without-top-level-length.csv").read_text(encoding="utf-8")
    assert with_length == without_length


def test_import_matches_ath_for_top_level_length_beside_an_osse_block() -> None:
    text = (ATH_FIXTURES / "osse-block-with-top-level-length.cfg").read_text(encoding="utf-8")
    assert "\nLength = 180\n" in text
    config = parse_text_config(text)
    assert config["profile"]["L"] == 160
    params = build_geometry_params(config)[0]
    profiles = _ath_profiles("osse-block-with-top-level-length")
    assert len(profiles) == 8
    for index, points in enumerate(profiles):
        phi = 2.0 * np.pi * index / len(profiles)
        z = points[:, 2]
        expected_radius = np.hypot(points[:, 0], points[:, 1])
        assert z[-1] == pytest.approx(160.0)
        np.testing.assert_allclose(np.arctan2(points[1:, 1], points[1:, 0]) % (2 * np.pi), phi, atol=1e-6)
        scalar = np.array([calculate_osse(float(station), phi, params) for station in z])
        vector = np.column_stack(calculate_osse_curve(z, phi, params))
        for actual in (scalar, vector):
            np.testing.assert_allclose(actual[:, 0], z, rtol=0, atol=1e-9)
            np.testing.assert_allclose(actual[:, 1], expected_radius, rtol=0, atol=5e-6)


@pytest.mark.parametrize("member", ["Enclosure.Plan", "Enclosure.Depth", "Enclosure.Spacing", "Enclosure.MadeUp", "Enclosure"])
def test_unread_mesh_block_enclosure_members_are_refused(member: str) -> None:
    with pytest.raises(ConfigError) as excinfo:
        parse_text_config(FLAT_OSSE + f"Mesh = {{\n{member} = 200\n}}\n")
    message = str(excinfo.value)
    assert f"Mesh.{member}" in message
    assert "top-level Mesh.Enclosure.*" in message
    assert "separate Mesh.Enclosure = {...} block" in message


@pytest.mark.parametrize("marker", ["Mesh.Enclosure", "Enclosure"])
def test_scalar_enclosure_marker_is_refused(marker: str) -> None:
    with pytest.raises(ConfigError) as excinfo:
        parse_text_config(FLAT_OSSE + f"{marker} = 1\n")
    assert marker in str(excinfo.value)
    assert "top-level Mesh.Enclosure.*" in str(excinfo.value)
    assert "separate Mesh.Enclosure = {...} block" in str(excinfo.value)


@pytest.mark.parametrize("enclosure", [
    "Mesh = {\nEnclosure.Plan = myplan\n}\n",
    "Mesh = {\nEnclosure.Depth = 200\n}\n",
    "Mesh = {\nEnclosure.MadeUp = 1\n}\n",
    "Mesh = {\nEnclosure = 1\n}\n",
    "Mesh.Enclosure = 1\n",
])
def test_c4_composition_precedes_unread_mesh_enclosure(enclosure: str) -> None:
    text = _rosse("  s1 = .5\n  s2 = .2\n") + "Rot = 10\n" + enclosure
    with pytest.raises(ConfigError, match=r"throat stretch with Rot.*R-OSSE Rot"):
        parse_text_config(text)


@pytest.mark.parametrize("member", ["_MadeUp", "_Roundover", "_ThroatSegments", "_ThroatExtSegments", "_RearShape", "_Enclosure.Depth"])
def test_disabled_mesh_block_members_preserve_geometry(member: str, caplog: pytest.LogCaptureFixture) -> None:
    from hornlab_mesher.config_builder import resolve_geometry

    plain = parse_text_config(FLAT_OSSE)
    with caplog.at_level(logging.WARNING, logger="hornlab_mesher.config_parser"):
        parked = parse_text_config(FLAT_OSSE + f"Mesh = {{\n{member} = 2\n}}\n")
    assert parked == plain
    base_geometry = resolve_geometry(plain).geometry
    parked_geometry = resolve_geometry(parked).geometry
    np.testing.assert_array_equal(base_geometry.inner_points, parked_geometry.inner_points)
    np.testing.assert_array_equal(base_geometry.outer_points, parked_geometry.outer_points)
    assert parked_geometry.enclosure == base_geometry.enclosure
    assert f"ignoring Mesh.{member} = 2: underscore-prefixed ATH item is disabled" in caplog.text


@pytest.mark.parametrize(("extra", "name", "reason"), [
    ("ABEC.f1 = 400\n", "ABEC.f1", "solver-run/project settings"),
    ("ABEC.SimTyp = 2\n", "ABEC.SimTyp", "solver-run/project settings"),
    ("Simulation.F1 = 400\n", "Simulation.F1", "do not change geometry or the source"),
    ("Output.STL = 1\n", "Output.STL", "output paths and file switches"),
    ("LE = generic25\n", "LE", "lumped-element driver model"),
    ("LE.Voltage = 2.83\n", "LE.Voltage", "lumped-element driver model"),
    ("Gmsh.Algorithm = 6\n", "Gmsh.Algorithm", "owns its meshing"),
    ("OutputRootDir = somewhere\n", "OutputRootDir", "output paths and file switches"),
    ("MeshCmd = gmsh\n", "MeshCmd", "external mesh command"),
    ("GnuplotPath = gnuplot\n", "GnuplotPath", "plotting executable"),
    ("_Scale.Z = .5\n", "_Scale.Z", "underscore-prefixed ATH item is disabled"),
    ("CircArc.TermAngle = 1\n", "CircArc.TermAngle", "only to Throat.Profile = 3"),
    ("Morph.Width = 300\n", "Morph.Width", "ATH has no such key"),
    ("GCurve.Distance = 300\n", "GCurve.Distance", "ATH has no such key"),
])
def test_each_ignored_key_has_a_named_reason(extra: str, name: str, reason: str, caplog: pytest.LogCaptureFixture) -> None:
    plain = parse_text_config(FLAT_OSSE)
    with caplog.at_level(logging.WARNING, logger="hornlab_mesher.config_parser"):
        assert parse_text_config(FLAT_OSSE + extra) == plain
    assert len(caplog.records) == 1
    assert f"ignoring {name} =" in caplog.text
    assert reason in caplog.text


@pytest.mark.parametrize(("name", "reason"), [
    ("ABEC.Polars:H", "solver-run/project settings"),
    ("Simulation.Polars:H", "do not change geometry or the source"),
    ("Output.Files", "output paths and file switches"),
    ("Report", "report layout"),
    ("GridExport:CAD", "coordinate/response export"),
    ("FRDExport", "coordinate/response export"),
    ("RespExport", "coordinate/response export"),
    ("LE.Driver", "lumped-element driver model"),
    ("Gmsh.Options", "owns its meshing"),
    ("_Parked", "underscore-prefixed ATH item is disabled"),
])
def test_ignored_block_reports_parent_subtree_once(name: str, reason: str, caplog: pytest.LogCaptureFixture) -> None:
    extra = f"{name} = {{\nMadeUp = 1\nSource.Contours = {{\nChild = {{\nx = 1\n}}\n}}\n}}\n"
    plain = parse_text_config(FLAT_OSSE)
    with caplog.at_level(logging.WARNING, logger="hornlab_mesher.config_parser"):
        assert parse_text_config(FLAT_OSSE + extra) == plain
    assert len(caplog.records) == 1
    assert f"ignoring {name} = {{...}} (whole subtree):" in caplog.text
    assert reason in caplog.text
    assert "Source.Contours" not in caplog.text


@pytest.mark.parametrize("block", ["OSSE", "Mesh", "Source", "Mesh.Enclosure", "Morph", "GCurve"])
def test_disabled_block_member_is_reported_once(block: str, caplog: pytest.LogCaptureFixture) -> None:
    text = _osse("_MadeUp = 1\n") if block == "OSSE" else FLAT_OSSE + f"{block} = {{\n_MadeUp = 1\n}}\n"
    with caplog.at_level(logging.WARNING, logger="hornlab_mesher.config_parser"):
        parse_text_config(text)
    assert len(caplog.records) == 1
    assert f"ignoring {block}._MadeUp = 1: underscore-prefixed ATH item is disabled" in caplog.text


def test_disabled_nested_parent_is_reported_once(caplog: pytest.LogCaptureFixture) -> None:
    extra = "_Parked = {\nMiddle = {\nSource.Contours = {\nx = 1\n}\n}\n}\n"
    with caplog.at_level(logging.WARNING, logger="hornlab_mesher.config_parser"):
        assert parse_text_config(_osse(extra)) == parse_text_config(_osse())
    assert len(caplog.records) == 1
    assert "ignoring OSSE._Parked = {...} (whole subtree)" in caplog.text
    assert "Middle" not in caplog.text


def test_unused_osse_top_level_length_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="hornlab_mesher.config_parser"):
        assert parse_text_config(_osse() + "Length = 180\n") == parse_text_config(_osse())
    assert len(caplog.records) == 1
    assert "ignoring Length = 180: ATH uses the OSSE block's L" in caplog.text


def test_osse_overridden_top_level_rotation_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="hornlab_mesher.config_parser"):
        assert parse_text_config(_osse("Rot = 0\n") + "Rot = 10\n") == parse_text_config(_osse("Rot = 0\n"))
    assert "ignoring Rot = 10: ATH uses the OSSE block's Rot" in caplog.text


def test_geometry_relevant_abec_simtype_is_consumed(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="hornlab_mesher.config_parser"):
        assert parse_text_config(FLAT_OSSE)["simType"] == 2
        assert parse_text_config(FLAT_OSSE.replace("ABEC.SimType = 2", "ABEC.SimType = 1"))["simType"] == 1
    assert not caplog.records


@pytest.mark.parametrize(("text", "named", "reason"), [
    ("osse = {\nL = 60\n}\n", "osse", "did you mean OSSE?"),
    ("r-osse = {\nR = 150\n}\n", "r-osse", "did you mean R-OSSE?"),
    ("OS-SE = {\nL = 60\n}\n", "OS-SE", "did you mean OSSE?"),
    ("Horn.Part:1 = {\nL = 60\n}\n", "Horn.Part:1", "two-profile (Horn.Part) geometry is not implemented"),
    ("HornGeometry = 2\n", "HornGeometry", "two-profile (Horn.Part) geometry is not implemented"),
    ("Profile = points.txt\n", "Profile", "profile read from a point file is not implemented"),
    ("Lenght = 60\n", "Lenght", "did you mean Length?"),
    ("Coverage.Angel = 45\n", "Coverage.Angel", "did you mean Coverage.Angle?"),
    ("Throat.Diameter = 25.4\n", "Throat.Diameter", "set Length or supply an OSSE or R-OSSE block"),
])
def test_profile_only_input_is_diagnosed_before_missing_profile(text: str, named: str, reason: str) -> None:
    with pytest.raises(ConfigError) as excinfo:
        parse_text_config(text)
    message = str(excinfo.value)
    assert named in message
    assert reason in message
    assert message.index(named) < message.index("text config must contain")
    assert "ICW is not available via the ATH text format" in message


def test_profile_only_spelling_diagnostic_names_all_candidates() -> None:
    with pytest.raises(ConfigError) as excinfo:
        parse_text_config("Lenght = 60\nCoverage.Angel = 45\nThroat.Diameter = 25.4\n")
    message = str(excinfo.value)
    for name in ("Lenght", "Coverage.Angel", "Throat.Diameter"):
        assert message.index(name) < message.index("text config must contain")
    assert "did you mean Length?" in message
    assert "did you mean Coverage.Angle?" in message


def test_warning_names_preserve_each_supplied_namespace(caplog: pytest.LogCaptureFixture) -> None:
    extra = "Morph.Width = 300\nMORPH.Width = 400\nGCurve = {\nDistance = 20\n}\nGCURVE = {\nDistance = 30\n}\n"
    with caplog.at_level(logging.WARNING, logger="hornlab_mesher.config_parser"):
        assert parse_text_config(FLAT_OSSE + extra) == parse_text_config(FLAT_OSSE)
    assert len(caplog.records) == 4
    for name, value in (("Morph.Width", 300), ("MORPH.Width", 400), ("GCurve.Distance", 20), ("GCURVE.Distance", 30)):
        assert f"ignoring {name} = {value}: ATH has no such key" in caplog.text
