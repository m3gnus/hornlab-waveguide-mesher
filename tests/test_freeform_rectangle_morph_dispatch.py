"""A FREEFORM rectangle morph must never be resampled through the R-OSSE formula.

The shared point-grid builder re-derives the azimuth list for a rectangle morph
and resamples the raw meridians when that list differs from the first one. For
FREEFORM the resample went through the shared radial grid, whose per-azimuth
loop ends in the R-OSSE branch: the drawn 100 mm horn came back as the default
40 mm R-OSSE. Which consumer was hit depended on the corner radius and on each
consumer's own angular budget, so coarse preview, fine preview and the solve
geometry disagreed with each other.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from hornlab_mesher import profile_sampling
from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.preview.api import build_preview_geometry
from hornlab_mesher.preview.contract import PreviewOptionsV1

DEPTH_MM = 100.0
HALF_WIDTH_MM = 150.0
HALF_HEIGHT_MM = 100.0


def _config(corner_mm: float, angular_segments: int) -> dict:
    return {
        "formula": "FREEFORM",
        "mode": "bare",
        "profile": {
            "profileH": {
                "points": [[0.0, 12.7], [50.0, 70.0], [DEPTH_MM, HALF_WIDTH_MM]],
                "throatAngleDeg": 15.5,
                "mouthAngleDeg": 70.0,
            },
            "profileV": {
                "points": [[0.0, 12.7], [50.0, 50.0], [DEPTH_MM, HALF_HEIGHT_MM]],
                "throatAngleDeg": 15.5,
                "mouthAngleDeg": 60.0,
            },
        },
        "morph": {
            "morph_target": 1,
            "morph_width_mm": 2 * HALF_WIDTH_MM,
            "morph_height_mm": 2 * HALF_HEIGHT_MM,
            "morph_corner_mm": corner_mm,
        },
        "mesh": {
            "angular_segments": angular_segments,
            "length_segments": 32,
            "throat_res_mm": 4.0,
            "mouth_res_mm": 20.0,
            "rear_res_mm": 25.0,
        },
    }


def _preview_extents(config: dict, lod: str) -> np.ndarray:
    geometry = build_preview_geometry(config, PreviewOptionsV1(lod=lod))
    points = np.vstack(
        [np.asarray(surface.positions, dtype=float).reshape(-1, 3) for surface in geometry.surfaces]
    )
    return np.abs(points).max(axis=0)


# Before the fix: the solve geometry was 40.1 mm deep at (60, 32), (80, 32),
# (80, 64), (90, 32) and (90, 64); the coarse preview at every corner of 80 and
# 90; the fine preview at 90.
CASES = [(60.0, 32), (80.0, 32), (80.0, 64), (90.0, 32), (90.0, 64), (90.0, 96), (30.0, 64)]


@pytest.mark.parametrize(("corner_mm", "angular_segments"), CASES)
def test_solve_geometry_keeps_the_drawn_freeform_horn(corner_mm: float, angular_segments: int) -> None:
    inner = np.asarray(resolve_geometry(_config(corner_mm, angular_segments)).geometry.inner_points)
    assert inner[..., 2].max() == pytest.approx(DEPTH_MM, abs=1e-9)
    assert np.abs(inner[..., 0]).max() == pytest.approx(HALF_WIDTH_MM, abs=1e-6)
    assert np.abs(inner[..., 1]).max() == pytest.approx(HALF_HEIGHT_MM, abs=1e-6)


@pytest.mark.parametrize(("corner_mm", "angular_segments"), CASES)
@pytest.mark.parametrize("lod", ["coarse", "fine"])
def test_preview_levels_keep_the_drawn_freeform_horn(corner_mm: float, angular_segments: int, lod: str) -> None:
    extents = _preview_extents(_config(corner_mm, angular_segments), lod)
    np.testing.assert_allclose(extents, [HALF_WIDTH_MM, HALF_HEIGHT_MM, DEPTH_MM], atol=1e-6)


@pytest.mark.parametrize(("corner_mm", "angular_segments"), CASES)
def test_freeform_never_reaches_the_shared_radial_grid(
    corner_mm: float, angular_segments: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    formulas: list[str] = []
    original = profile_sampling._raw_radial_grid

    def spy(params, angles, t_values, t_unit_values, formula, *args, **kwargs):
        formulas.append(formula)
        return original(params, angles, t_values, t_unit_values, formula, *args, **kwargs)

    monkeypatch.setattr(profile_sampling, "_raw_radial_grid", spy)
    config = _config(corner_mm, angular_segments)
    resolve_geometry(copy.deepcopy(config))
    for lod in ("coarse", "fine"):
        build_preview_geometry(copy.deepcopy(config), PreviewOptionsV1(lod=lod))
    assert formulas == []


def test_shared_radial_grid_refuses_a_formula_it_has_no_branch_for() -> None:
    angles = np.array([0.0, np.pi / 2])
    t_values = np.linspace(0.0, 1.0, 3)
    with pytest.raises(ValueError, match="no meridian formula for 'FREEFORM'"):
        profile_sampling._raw_radial_grid({}, angles, t_values, t_values, "FREEFORM", 2.0, 1.0, 2)
