import logging

import numpy as np
import pytest

from hornlab_mesher.config_parser import ConfigError, parse_text_config
from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.profile_sampling import build_point_grid_arrays

BASE = """OSSE = {
L = 160
a = 40
}
Mesh.AngularSegments = 100
Mesh.CornerSegments = 4
Mesh.LengthSegments = 20
Mesh.WallThickness = 0
Mesh.Quadrants = 1
"""


def grid(text):
    return build_point_grid_arrays(build_geometry_params(parse_text_config(text))[0])["inner_grid"]


@pytest.mark.parametrize("morph", ["", "Morph.TargetShape = 1\nMorph.FixedPart = 0\n"])
@pytest.mark.parametrize("value", [0, 4, 10, 25])
def test_nac_presence_selects_sampling_but_value_is_ignored(morph, value, caplog):
    with caplog.at_level(logging.WARNING, logger="hornlab_mesher.config_parser"):
        actual = grid(BASE + morph + f"NAC = {value}\n")
    assert "NAC" in caplog.text and "ignoring NAC's value" in caplog.text
    assert actual.shape == (26, 21, 3)
    np.testing.assert_array_equal(actual, grid(BASE + morph + "NAC = 10\n"))
    # Without NAC both plain and rectangular profiles use 100 + 4 segments.
    assert grid(BASE + morph).shape == (27, 21, 3)


@pytest.mark.parametrize("value", ["1.5", "inf", "nan", "sin(p)", ""])
def test_nac_requires_an_integer(value):
    with pytest.raises(ConfigError, match="NAC.*integer"):
        parse_text_config(BASE + f"NAC = {value}\n")


def test_nac_presence_survives_dictionary_round_trip():
    import json
    parsed = parse_text_config(BASE + "NAC = 0\n")
    params = build_geometry_params(json.loads(json.dumps(parsed)))[0]
    assert params["_athNacPresent"] is True
    assert build_point_grid_arrays(params)["inner_grid"].shape[0] == 26


def test_native_angular_budget_is_preserved():
    config = {"formula": "OSSE", "profile": {"L": 160, "a": 40},
              "mesh": {"angularSegments": 100, "cornerSegments": 4,
                       "quadrants": "1", "wallThickness": 0}}
    params = build_geometry_params(config)[0]
    assert build_point_grid_arrays(params)["inner_grid"].shape[0] == 26
