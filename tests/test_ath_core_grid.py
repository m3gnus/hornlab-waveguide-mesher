"""Complete inner grids from independent ATH V2026-09a exports."""
from pathlib import Path

import numpy as np
import pytest

from hornlab_mesher.config_parser import parse_text_config
from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.profile_sampling import build_point_grid_arrays

FIXTURES = Path(__file__).parent / "fixtures/ath_import/ath-v2026-09a"
CORE_CASES = [f"f0_{kind}{suffix}" for kind in ("plain", "morph", "stretch", "both")
              for suffix in ("", "_nac")]


def reference(name):
    return np.array([[[float(v) for v in row.split(";")] for row in block.splitlines()]
                     for block in (FIXTURES / f"{name}.csv").read_text(encoding="utf-8").strip().split("\n\n")])


@pytest.mark.parametrize("name", CORE_CASES + ["sf1", "sf2", "sf3"])
@pytest.mark.parametrize("quadrants", ["1", "1234"])
def test_inner_grid_matches_ath(name, quadrants):
    config = parse_text_config((FIXTURES / f"{name}.cfg").read_text(encoding="utf-8"))
    config["mesh"]["quadrants"] = quadrants
    params = build_geometry_params(config)[0]
    actual = build_point_grid_arrays(params)["inner_grid"]
    expected = reference(name)
    # GridExport always exports the full horn, even for Mesh.Quadrants = 1.
    if quadrants == "1":
        expected = expected[:len(expected)//4 + 1]
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-4)


@pytest.mark.parametrize("name", CORE_CASES + ["sf1", "sf2", "sf3"])
def test_fixture_normalisation(name):
    for extension in ("cfg", "csv"):
        data = (FIXTURES / f"{name}.{extension}").read_bytes()
        assert b"\r" not in data
        assert data.endswith(b"\n") and not data.endswith(b"\n\n")
    if name.startswith("f0_"):
        assert (FIXTURES / f"{name}.cfg").read_text(encoding="utf-8").startswith(
            "; Derived from ATH CORE (C) 2026 Marcel Batík, CC-BY-4.0, at-horns.eu/core.html\n")


def test_morph_starts_at_first_positive_pre_stretch_station():
    raw, morphed = reference("f0_plain"), reference("f0_morph")
    t = raw[0, :, 2] / raw[0, -1, 2]
    factor = np.maximum(0, (t-t[1])/(1-t[1]))**3
    mouth_delta = morphed[0, -1, 0] - raw[0, -1, 0]
    np.testing.assert_allclose(morphed[0, :, 0], raw[0, :, 0] + mouth_delta * factor,
                               rtol=0, atol=2e-4)
    # Only the cardinal meridian is shared: rectangle morphs move the other
    # azimuth rows onto their corner-aware angular sampling.
    np.testing.assert_array_equal(raw[0, :2, :], morphed[0, :2, :])


@pytest.mark.parametrize("kind", ["plain", "morph"])
@pytest.mark.parametrize("suffix", ["", "_nac"])
def test_stretch_changes_z_only_after_morph_progress(kind, suffix):
    plain = reference(f"f0_{kind}{suffix}")
    stretched = reference(f"f0_{'stretch' if kind == 'plain' else 'both'}{suffix}")
    np.testing.assert_array_equal(plain[:, :, :2], stretched[:, :, :2])
    expected_z = plain[:, :, 2] + 1.14 * np.degrees(np.arctan(.025 * plain[:, :, 2]))
    np.testing.assert_allclose(stretched[:, :, 2], expected_z, rtol=0, atol=2e-4)
