"""FREEFORM + rectangular Morph: the per-ring azimuth layout is continuous in t.

A smooth-schedule (circle/ellipse) FREEFORM morphing to a rectangle used to
switch each ring's azimuth layout from the uniform one (corner exactly 0, the
throat ring) to the sharp-rectangle one (any non-zero corner, every later
ring) between the first two rings, or at the morph start. Meridians jumped by
up to 15 degrees in azimuth across one axial interval, which the outer
offset check reads as a normal flip and the acoustic axial-chord fit can never
resolve, at any density.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from hornlab_mesher import build_from_config
import hornlab_mesher.profile_sampling as ps
from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.profile_sampling import build_point_grid, freeform_azimuth_twist_note

# ATH-style Morph block: a 300 x 200 mm rectangle with a 25 mm corner.
PILOT_MORPH = {
    "morphTarget": 1,
    "morphWidth": 300.0,
    "morphHeight": 200.0,
    "morphCorner": 25.0,
    "morphRate": 3.0,
    "morphFixed": 0.0,
    "morphAllowShrinkage": 0.0,
}
# WG's UI: pick Rectangle, leave size at "from the mouth", set a corner.
UI_MORPH = {**PILOT_MORPH, "morphWidth": 0.0, "morphHeight": 0.0}


def _config(mode: str = "freestanding", morph: dict | None = None, *, wall: float = 5.0) -> dict:
    config = {
        "formula": "FREEFORM",
        "mode": mode,
        "profile": {
            "profileH": {
                "points": [[0.0, 12.7], [120.0, 140.0]],
                "throatAngleDeg": 15.5,
                "mouthAngleDeg": 60.0,
            },
            "profileV": {
                "points": [[0.0, 12.7], [120.0, 140.0]],
                "throatAngleDeg": 15.5,
                "mouthAngleDeg": 60.0,
            },
            "crossSections": [
                {"t": 0.0, "shape": "ellipse"},
                {"t": 1.0, "shape": "ellipse"},
            ],
        },
        "mesh": {
            "angularSegments": 40,
            "cornerSegments": 4,
            "lengthSegments": 20,
            "samplingMode": "uniform",
            "quadrants": "1234",
            "throat_res_mm": 6.0,
            "mouth_res_mm": 15.0,
            "rear_res_mm": 40.0,
            "wall_thickness_mm": wall,
            "enc_front_res_mm": 25.0,
            "enc_back_res_mm": 40.0,
        },
    }
    if morph is not None:
        config["morph"] = dict(morph)
    if mode == "enclosure":
        config["enclosure"] = {"depth": 200.0}
    return config


def _triangles(tmp_path, config: dict) -> int:
    return build_from_config(config, tmp_path / "m.msh").n_triangles


def _inner_grid(morph: dict, **overrides) -> tuple[np.ndarray, np.ndarray]:
    config = _config(morph={**morph, **overrides}, wall=0.0)
    params, _formula, _mode = build_geometry_params(config)
    grid = build_point_grid(params)
    inner = np.asarray(grid["inner_points"], dtype=np.float64).reshape(
        int(grid["grid_n_phi"]), int(grid["grid_n_length"]) + 1, 3
    )
    return inner, np.asarray(grid["phi_grid"], dtype=np.float64)


def _max_azimuth_step_deg(phi_grid: np.ndarray) -> float:
    step = np.diff(phi_grid, axis=1)
    return float(np.degrees(np.max(np.abs(step))))


def _max_meridian_turn_deg(inner: np.ndarray) -> float:
    segments = inner[:, 1:, :] - inner[:, :-1, :]
    unit = segments / np.linalg.norm(segments, axis=2, keepdims=True)
    cosine = np.clip(np.sum(unit[:, 1:] * unit[:, :-1], axis=2), -1.0, 1.0)
    return float(np.degrees(np.max(np.arccos(cosine))))


@pytest.mark.parametrize("fixed", [0.0, 0.3])
@pytest.mark.parametrize(
    "morph,corner",
    [
        (PILOT_MORPH, 12.0),
        (PILOT_MORPH, 25.0),
        (UI_MORPH, 1.0),
        (UI_MORPH, 10.0),
        (UI_MORPH, 25.0),
    ],
)
def test_rectangle_morph_azimuths_are_continuous_across_the_morph_start(
    morph, corner, fixed
) -> None:
    inner, phi_grid = _inner_grid(morph, morphCorner=corner, morphFixed=fixed)

    assert _max_azimuth_step_deg(phi_grid) < 3.0
    assert _max_meridian_turn_deg(inner) < 10.0


@pytest.mark.parametrize("mode", ["freestanding", "infinite-baffle", "enclosure"])
@pytest.mark.parametrize("morph", [PILOT_MORPH, UI_MORPH], ids=["ath-style", "ui-corner25"])
def test_rectangle_morph_builds_at_default_mm(tmp_path, mode, morph) -> None:
    triangles = _triangles(tmp_path, _config(mode, morph))
    baseline = _triangles(tmp_path, _config(mode))

    # A morph adds a corner and a wider mouth; it must not multiply the mesh.
    assert 0 < triangles < 2.0 * baseline


def test_rectangle_morph_with_a_fixed_part_builds_in_every_mode(tmp_path) -> None:
    morph = {**PILOT_MORPH, "morphFixed": 0.3}
    for mode in ("freestanding", "infinite-baffle", "enclosure"):
        assert _triangles(tmp_path, _config(mode, morph)) > 0


def test_rectangle_morph_triangle_count_is_not_inflated_by_the_corner(tmp_path) -> None:
    counts = {
        corner: _triangles(tmp_path, _config(morph={**UI_MORPH, "morphCorner": corner}))
        for corner in (10.0, 25.0)
    }

    assert counts[10.0] < 1.6 * counts[25.0]


@pytest.mark.parametrize("mode", ["freestanding", "infinite-baffle", "enclosure"])
def test_sharp_ui_default_names_a_corner_that_builds(tmp_path, mode) -> None:
    """Corner 0 is the singular radial blend and stays refused; its hint builds."""

    with pytest.raises(ConfigError) as error:
        _triangles(tmp_path, _config(mode, {**UI_MORPH, "morphCorner": 0.0}))
    match = re.search(r"morphCorner is ~([0-9.]+) mm", str(error.value))
    assert match is not None, str(error.value)

    hinted = {**UI_MORPH, "morphCorner": float(match.group(1))}
    assert _triangles(tmp_path, _config(mode, hinted)) > 0


def test_twist_note_names_the_jump_and_not_the_resolution() -> None:
    steady = np.tile(np.linspace(0.0, math.pi / 2.0, 9)[:, None], (1, 6))
    assert freeform_azimuth_twist_note(steady) == ""

    twisted = steady.copy()
    twisted[1:-1, 3:] += np.radians(12.0)
    note = freeform_azimuth_twist_note(twisted, np.linspace(0.0, 1.0, 6))
    assert "12 degrees" in note
    assert "axial rings 2 and 3" in note
    assert "raise" not in note


def test_reintroduced_twist_is_named_in_both_failure_messages(
    tmp_path, monkeypatch
) -> None:
    """With the old jump back in the grid, neither failure blames mm or wall."""

    def old_layout(quadrant_angles, morph_factor):
        return quadrant_angles, True

    monkeypatch.setattr(ps, "_blend_toward_uniform_layout", old_layout)

    with pytest.raises(ValueError, match="normal flip") as flip:
        _triangles(tmp_path, _config("freestanding", PILOT_MORPH))
    assert "azimuth rows shift by" in str(flip.value)
    assert "cannot fix that" in str(flip.value)

    with pytest.raises(ConfigError, match="cannot fit the acoustic geometry") as fit:
        _triangles(tmp_path, _config("infinite-baffle", PILOT_MORPH))
    assert "azimuth rows shift by" in str(fit.value)
    assert "raise the throat/mouth resolution" not in str(fit.value)
