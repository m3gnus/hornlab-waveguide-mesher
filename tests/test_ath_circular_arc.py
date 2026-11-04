"""ATH's OSSE block owns its profile even beside circular-arc selectors."""
import numpy as np
import pytest

from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.config_parser import ConfigError, parse_text_config

BLOCK = """OSSE = {
L = 120
r0 = 18
a = 40
a0 = 8
k = 1
s = 0
%s}
Source.Shape = 2
ABEC.SimType = 2
Mesh.WallThickness = 0
Mesh.Quadrants = 1234
Mesh.AngularSegments = 16
Mesh.LengthSegments = 64
Mesh.ZMapPoints = 0,0,1,1
"""


@pytest.mark.parametrize("inside", [False, True])
@pytest.mark.parametrize("options", ["", "CircArc.Radius = 200\n", "CircArc.TermAngle = 60\n",
                                   "CircArc.Radius = 80\nCircArc.TermAngle = 60\n"])
def test_osse_block_ignores_arc_settings_with_byte_identical_geometry(inside, options, caplog):
    keys = "Throat.Profile = 3\n" + options
    text = BLOCK % keys if inside else BLOCK % "" + keys
    expected = parse_text_config(BLOCK % "")
    actual = parse_text_config(text)
    assert actual == expected
    # ATH exports an inner-only grid for zero walls; the public mesher calls
    # that topology bare rather than a positive-wall freestanding shell.
    actual["mode"] = expected["mode"] = "bare"
    first = resolve_geometry(expected).geometry.inner_points
    second = resolve_geometry(actual).geometry.inner_points
    assert first.tobytes() == second.tobytes()
    np.testing.assert_array_equal(first, second)
    for key in ("Throat.Profile", *(line.split(" =")[0] for line in options.splitlines())):
        assert "ignoring " + ("OSSE." if inside else "") + key in caplog.text
    assert "OSSE block" in caplog.text


@pytest.mark.parametrize("header", ["; MWG config\n", *[
    f"; Waveguide Generator geometry-interpretation: {v}\n"
    for v in ("native-v1", "ath-2026-08c-v1", "ath-2026-09a-v2", "ath-2026-09a-v3")]])
def test_saved_mesher_throat_profile_refusals_are_unchanged(header):
    with pytest.raises(ConfigError, match="Throat.Profile"):
        parse_text_config(header + BLOCK % "" + "Throat.Profile = 3\n")


def test_unqualified_flat_arc_refusal_explains_the_end_to_end_boundary():
    with pytest.raises(ConfigError) as caught:
        parse_text_config("Length = 120\nThroat.Profile = 3\nCircArc.Radius = 200\n")
    assert "active flat circular-arc" in str(caught.value)
    assert "ATH" in caught.value.next_step


def test_unknown_circular_arc_key_is_not_hidden_by_noop_selector():
    with pytest.raises(ConfigError, match=r"CircArc\.MadeUp"):
        parse_text_config(BLOCK % "" + "Throat.Profile = 3\nCircArc.MadeUp = 80\n")


@pytest.mark.parametrize("text", [
    "OSSE = {\n}\nLength = 120\nThroat.Profile = 3\n",
    "OSSE = {\nThroat.Profile = 3\n}\nLength = 120\n",
    "OSSE = {\n_L = 120\n}\nLength = 120\nThroat.Profile = 3\n",
    BLOCK.replace("OSSE", "R-OSSE") % "" + "Throat.Profile = 3\n",
    "; Ath version V2025-12\n" + BLOCK % "" + "Throat.Profile = 3\n",
])
def test_unmeasured_profile_positions_and_versions_stay_refused(text):
    with pytest.raises(ConfigError, match="Throat.Profile|must set Length"):
        parse_text_config(text)
