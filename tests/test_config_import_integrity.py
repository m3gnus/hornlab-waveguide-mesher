"""Text imports preserve supplied intent and report malformed enabled input."""

from __future__ import annotations

import logging

import pytest

from hornlab_mesher.config_parser import ConfigError, parse_text_config


BASE = "Length = 60\nCoverage.Angle = 45\n"


@pytest.mark.parametrize("block", ["OSSE", "R-OSSE", "ROSSE", "Mesh", "Source", "Morph", "MORPH", "GCurve", "GCURVE", "Mesh.Enclosure"])
@pytest.mark.parametrize("line", ["a 45", "Inner {", "Scale.Z .5"])
def test_malformed_enabled_block_line_names_line_and_block(block: str, line: str) -> None:
    dimension = "L = 60\n" if block == "OSSE" else "R = 150\n" if block in {"R-OSSE", "ROSSE"} else ""
    prefix = "" if dimension else BASE
    text = prefix + f"{block} = {{\n" + dimension + line + "\n}\n"
    lineno = text.splitlines().index(line) + 1
    with pytest.raises(ConfigError, match=rf"line {lineno}:.*{block}") as error:
        parse_text_config(text)
    assert line in str(error.value)


def test_malformed_nested_opener_cannot_leak_source_members() -> None:
    text = BASE + "Source = {\nInner {\nShape = 2\n}\n"
    with pytest.raises(ConfigError, match=r"line 4:.*Source.*Inner"):
        parse_text_config(text)


@pytest.mark.parametrize("block", ["Report", "GridExport:CAD", "FRDExport", "RespExport", "Output.Files", "Simulation.Fields", "ABEC.Polars:test", "Gmsh.Run", "LE.Driver", "_Parked"])
def test_wholly_ignored_subtree_accepts_script_and_unrelated_stretch(block: str, caplog) -> None:
    text = BASE + f"{block} = {{\ns1 = PlotLabel\ns2 = -1\npoint P0 100 0 8\nInner = {{\npoint P1 0 100 8\n}}\n}}\n"
    with caplog.at_level(logging.WARNING):
        assert parse_text_config(text) == parse_text_config(BASE)
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert block in messages[0] and "whole subtree" in messages[0]


def test_disabled_nested_subtree_preserves_outer_members(caplog) -> None:
    text = BASE + "Source = {\n_Parked = {\npoint P0 100 0 8\nInner = {\ns1 = PlotLabel\npoint P1 0 100 8\n}\n}\nShape = 2\n}\n"
    with caplog.at_level(logging.WARNING):
        result = parse_text_config(text)
    assert result["source"]["sourceShape"] == 0
    assert len(caplog.records) == 1
    assert "Source._Parked" in caplog.records[0].getMessage()


@pytest.mark.parametrize("block", ["my_plan", "Source.Contours"])
def test_unsupported_script_block_keeps_named_refusal(block: str) -> None:
    text = BASE + f"{block} = {{\npoint P0 100 0 8\n}}\n"
    with pytest.raises(ConfigError) as error:
        parse_text_config(text)
    assert block in str(error.value)
    assert "expected 'key = value'" not in str(error.value)


def test_unsupported_nested_script_keeps_full_path_refusal() -> None:
    text = BASE + "Source = {\nContours = {\npoint P0 100 0 8\n}\n}\n"
    with pytest.raises(ConfigError, match=r"Source\.Contours") as error:
        parse_text_config(text)
    assert "expected 'key = value'" not in str(error.value)


@pytest.mark.parametrize("value", ["1.8", "-1.8", "nan", "NaN", "inf", "-inf", "1e309", "bad", "1.0000000000000001", "9007199254740993.1", "1e-400"])
@pytest.mark.parametrize("form", ["flat", "block"])
def test_interface_slice_tokens_require_finite_integral_values(value: str, form: str) -> None:
    extra = f"Mesh.SubdomainSlices = {value}\n" if form == "flat" else f"Mesh = {{\nSubdomainSlices = {value}\n}}\n"
    with pytest.raises(ConfigError, match=r"Mesh\.SubdomainSlices.*integers"):
        parse_text_config(BASE + extra)


@pytest.mark.parametrize("value,expected", [("0", "1"), ("+2", "3"), ("-1", "0"), ("1.0", "2"), ("2e1", "21"), ("9007199254740993", "9007199254740994"), ("0, 1.0, 2e1", "1,2,21"), ("", ""), (" , , ", "")])
def test_interface_slice_integer_spellings_and_explicit_empty(value: str, expected: str) -> None:
    result = parse_text_config(BASE + f"Mesh.SubdomainSlices = {value}\n")
    assert result["mesh"]["subdomainSlices"] == expected


@pytest.mark.parametrize("member", ["Shape", "Velocity", "Unknown", "Array", "Velocity.1"])
def test_flat_disabled_source_member_matches_block_and_reports(member: str, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        flat = parse_text_config(BASE + f"Source._{member} = 2\n")
        block = parse_text_config(BASE + f"Source = {{\n_{member} = 2\n}}\n")
    assert flat == block == parse_text_config(BASE)
    assert len(caplog.records) == 2
    assert all(f"Source._{member}" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("member,value", [("Shape", "3"), ("Velocity", "2"), ("Unknown", "2"), ("Array", "2"), ("Velocity.1", "2")])
def test_disabled_source_sibling_cannot_hide_enabled_member(member: str, value: str) -> None:
    text = BASE + f"Source._{member} = 2\nSource.{member} = {value}\n"
    with pytest.raises(ConfigError):
        parse_text_config(text)


@pytest.mark.parametrize("family", ["OSSE", "R-OSSE", "ROSSE"])
@pytest.mark.parametrize("coefficient", ["s1", "s2"])
def test_unselected_enabled_profile_still_validates_supplied_stretch(family: str, coefficient: str) -> None:
    dimension = "L" if family == "OSSE" else "R"
    other = "R-OSSE" if family == "OSSE" else "OSSE"
    other_dimension = "R" if other == "R-OSSE" else "L"
    text = f"{other} = {{\n{other_dimension} = 150\n}}\n{family} = {{\n{dimension} = 150\n{coefficient} = PlotLabel\n}}\n"
    with pytest.raises(ConfigError, match="throat stretch"):
        parse_text_config(text)


def test_flat_disabled_source_subtree_matches_nested_block_and_reports(caplog) -> None:
    members = "s1 = PlotLabel\npoint P0 100 0 8\nInner = {\nSource.Contours = {\npoint P1 0 100 8\n}\n}\n"
    with caplog.at_level(logging.WARNING):
        flat = parse_text_config(BASE + "Source._Contours:parked = {\n" + members + "}\n")
        nested = parse_text_config(BASE + "Source = {\n_Contours:parked = {\n" + members + "}\n}\n")
    assert flat == nested == parse_text_config(BASE)
    assert len(caplog.records) == 2
    assert all("Source._Contours:parked" in record.getMessage() and "whole subtree" in record.getMessage() for record in caplog.records)


def test_disabled_source_subtree_cannot_hide_enabled_contours() -> None:
    text = BASE + "Source._Contours = {\npoint P0 100 0 8\n}\nSource.Contours = {\npoint P1 0 100 8\n}\n"
    with pytest.raises(ConfigError, match=r"multi-source.*Source\.Contours"):
        parse_text_config(text)


@pytest.mark.parametrize("block", ["Morph", "GCurve", "Source", "Mesh", "Unknown"])
def test_enabled_nonprofile_section_keeps_supplied_stretch_validation(block: str) -> None:
    text = BASE + f"{block} = {{\ns1 = PlotLabel\n}}\n"
    with pytest.raises(ConfigError, match="throat stretch"):
        parse_text_config(text)
