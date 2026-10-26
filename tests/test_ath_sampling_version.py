"""ATH version echoes select import defaults, without native config controls."""
import logging
from pathlib import Path

import numpy as np
import pytest

from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.config_parser import ConfigError, parse_text_config
from hornlab_mesher.profile_sampling import build_point_grid_arrays
from hornlab_mesher.text_import import ATH_AXIAL_RULE_KEY, ATH_AXIAL_CURRENT, ATH_AXIAL_LEGACY


FIXTURE = Path(__file__).parent / "fixtures/ath_import/ath-v2026-08c/osse-block-without-top-level-length.cfg"


def fixture_text():
    # The committed witness explicitly requests a uniform map. Remove that
    # override to exercise the imported default with the same profile.
    return FIXTURE.read_text().replace("Mesh.ZMapPoints = 0,0,1,1\n", "")


@pytest.mark.parametrize("stamp,rule", [
    ("V2024-12a", ATH_AXIAL_LEGACY), ("V2025-06", ATH_AXIAL_LEGACY),
    ("V2025-06c", ATH_AXIAL_LEGACY), (None, ATH_AXIAL_CURRENT),
    ("V2025-07", ATH_AXIAL_CURRENT), ("V2025-11b", ATH_AXIAL_CURRENT),
    ("V2025-12", ATH_AXIAL_CURRENT), ("V2026-09a", ATH_AXIAL_CURRENT),
])
def test_echo_selects_imported_axial_grid(stamp, rule, caplog):
    header = f"; Ath version {stamp}\n" if stamp else ""
    with caplog.at_level(logging.INFO, logger="hornlab_mesher.config_parser"):
        config = parse_text_config(header + fixture_text())
    assert config[ATH_AXIAL_RULE_KEY] == rule
    assert rule in caplog.text
    params = build_geometry_params(config)[0]
    assert params.get(ATH_AXIAL_RULE_KEY, ATH_AXIAL_CURRENT) == rule
    grid = build_point_grid_arrays(params)["inner_grid"]
    # Fixed regression coordinates from the historical fitted map and the
    # current ATH 0.001-parameter map, respectively (160 mm body, 20 segments).
    expected_z = 2.0740138794849217 if rule == ATH_AXIAL_LEGACY else 2.0933444
    assert grid[0, 1, 2] == pytest.approx(expected_z, abs=1e-6)


@pytest.mark.parametrize("stamp,legacy", [("V2025-06", True), ("V2026-09a", False), (None, False)])
def test_zero_fixed_part_morph_start_follows_echo(stamp, legacy):
    header = f"; Ath version {stamp}\n" if stamp else ""
    config = parse_text_config(header + fixture_text())
    raw = build_point_grid_arrays(build_geometry_params(config)[0])["inner_grid"]
    config = parse_text_config(header + fixture_text() + "Morph.TargetShape = 1\n"
                              "Morph.FixedPart = 0\nMorph.Rate = 1\n"
                              "Morph.TargetWidth = 500\nMorph.TargetHeight = 500\n")
    morphed = build_point_grid_arrays(build_geometry_params(config)[0])["inner_grid"]
    if legacy:
        assert abs(morphed[0, 1, 0] - raw[0, 1, 0]) > 0.1
    else:
        np.testing.assert_array_equal(morphed[0, :2], raw[0, :2])


@pytest.mark.parametrize("echo", ["; Ath version V2025-13", "; Ath version V2025-06xx",
                                  "; Ath version V2025-06 extra"])
def test_malformed_echo_uses_current_default(echo):
    assert parse_text_config(echo + "\n" + fixture_text())[ATH_AXIAL_RULE_KEY] == ATH_AXIAL_CURRENT


def test_only_leading_comment_header_selects_rule():
    assert parse_text_config(fixture_text() + "; Ath version V2025-06\n")[ATH_AXIAL_RULE_KEY] == ATH_AXIAL_CURRENT
    native = parse_text_config("; Waveguide Generator geometry-interpretation: native-v1\n"
                               "; Ath version V2025-06\n" + fixture_text())
    assert ATH_AXIAL_RULE_KEY not in native
    assert ATH_AXIAL_RULE_KEY not in build_geometry_params(native)[0]


def test_conflicting_echoes_refused():
    with pytest.raises(ConfigError, match="conflicting ATH version echoes"):
        parse_text_config("; Ath version V2025-06\n; Ath version V2026-09a\n" + fixture_text())


@pytest.mark.parametrize("stamp", ["V2025-06", "V2026-09a"])
def test_explicit_axial_map_overrides_echo(stamp):
    config = parse_text_config(f"; Ath version {stamp}\n" + FIXTURE.read_text())
    grid = build_point_grid_arrays(build_geometry_params(config)[0])["inner_grid"]
    np.testing.assert_allclose(grid[0, :, 2], np.linspace(0, 160, 21), rtol=0, atol=1e-10)


@pytest.mark.parametrize("stamp,half_width,profiles", [("V2025-06", 132, 84), ("V2026-09a", 132, 80)])
def test_implicit_rectangle_uses_versioned_raw_ring(stamp, half_width, profiles):
    text = FIXTURE.read_text().replace("Mesh.AngularSegments = 8", "Mesh.AngularSegments = 80")
    text += ("Mesh.CornerSegments = 4\nMorph.TargetShape = 1\nMorph.AllowShrinkage = 1\n"
             "Morph.CornerRadius = 10\nGCurve.Type = 2\nGCurve.Width = 200\n"
             "GCurve.SF.m1 = 4\nGCurve.SF.m2 = 4\nGCurve.SF.n1 = 1.5\n"
             "GCurve.SF.n2 = 7\nGCurve.SF.n3 = 2.5\n")
    config = parse_text_config(f"; Ath version {stamp}\n" + text)
    grid = build_point_grid_arrays(build_geometry_params(config)[0])["inner_grid"]
    # The older build adds the corner budget to the final ring; ATH V2026-09a
    # exports 80 profiles and a 132 mm half-width (GridExport of this config).
    assert grid.shape[0] == profiles
    np.testing.assert_allclose(np.max(np.abs(grid[:, -1, :2]), axis=0),
                               [half_width, 113], rtol=0, atol=1e-10)


@pytest.mark.parametrize('stamp', ['V2025-06', 'V2026-09a'])
def test_utf8_bom_does_not_hide_version_echo(stamp):
    text = f'; Ath version {stamp}\n' + fixture_text()
    assert parse_text_config('\ufeff' + text) == parse_text_config(text)


# ATH V2026-09a exports 8 profiles here: CornerSegments does not add to the count.
@pytest.mark.parametrize('stamp,segments', [('V2025-06', 8), ('V2026-09a', 8)])
def test_plain_profile_corner_budget_follows_sampling_rule(stamp, segments):
    cfg = parse_text_config(f'; Ath version {stamp}\n' + FIXTURE.read_text() + 'Mesh.CornerSegments = 4\n')
    assert build_point_grid_arrays(build_geometry_params(cfg)[0])['inner_grid'].shape[0] == segments


def test_saved_v1_echo_cannot_upgrade_or_refuse_historical_sampling():
    from hornlab_mesher.text_import import TEXT_IMPORT_VERSION_KEY
    header = '; Waveguide Generator geometry-interpretation: ath-2026-08c-v1\n'
    baseline = parse_text_config(header + fixture_text())
    cfg = parse_text_config(header + '; Ath version V2025-06\n; Ath version V2026-09a\n'
                            + fixture_text() + 'NAC = ignored-by-v1\n')
    assert cfg[TEXT_IMPORT_VERSION_KEY] == 'ath-2026-08c-v1'
    assert ATH_AXIAL_RULE_KEY not in cfg
    np.testing.assert_array_equal(build_point_grid_arrays(build_geometry_params(cfg)[0])['inner_grid'],
                                  build_point_grid_arrays(build_geometry_params(baseline)[0])['inner_grid'])
