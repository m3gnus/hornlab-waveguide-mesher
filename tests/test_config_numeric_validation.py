"""Supplied geometry dimensions must not silently become default values."""

from __future__ import annotations

import pytest

from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.config_parser import ConfigError, parse_text_config


def _diameter_config(value: str, form: str) -> str:
    if form == "flat":
        return f"Length = 60\nCoverage.Angle = 45\nThroat.Diameter = {value}\n"
    if form == "osse":
        return f"OSSE = {{\nL = 60\na = 45\nThroat.Diameter = {value}\n}}\n"
    return f"R-OSSE = {{\nR = 150\na = 45\nThroat.Diameter = {value}\n}}\n"


@pytest.mark.parametrize("form", ["flat", "osse", "rosse"])
@pytest.mark.parametrize("value", ["fifty", "", "50.8 + 0*p", "nan", "inf", "1e309", "0", "-25.4", "5e-324"])
@pytest.mark.parametrize("with_radius", [False, True])
def test_supplied_invalid_diameter_fails_before_geometry_defaults(value: str, form: str, with_radius: bool) -> None:
    text = _diameter_config(value, form)
    if with_radius:
        text = text.replace("Throat.Diameter", "r0 = 18\nThroat.Diameter")
    with pytest.raises(ConfigError, match=r"Throat\.Diameter.*positive finite number"):
        parse_text_config(text)


@pytest.mark.parametrize("form", ["flat", "osse", "rosse"])
@pytest.mark.parametrize("value", ["50.8", "5.08e1"])
def test_numeric_diameter_sets_the_requested_throat_radius(value: str, form: str) -> None:
    config = parse_text_config(_diameter_config(value, form))
    params, _, _ = build_geometry_params(config)
    assert params["r0"] == pytest.approx(25.4)


@pytest.mark.parametrize("form", ["flat", "osse", "rosse"])
def test_explicit_radius_keeps_precedence_over_diameter(form: str) -> None:
    text = _diameter_config("50.8", form).replace("Throat.Diameter", "r0 = 18\nThroat.Diameter")
    config = parse_text_config(text)
    params, _, _ = build_geometry_params(config)
    assert params["r0"] == pytest.approx(18)
