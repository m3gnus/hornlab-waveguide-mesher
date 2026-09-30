"""The outer wall is offset to the same side at every sampling density.

The offset direction used to be chosen from a sum over nodes spread across the
grid. Along an R-OSSE rollback the wall lies on the axis side of the surface,
so those nodes voted against the rest, and the winner depended on how many of
the strided samples landed on the rollback: one design was 300 mm wide at the
coarse and fine preview levels and 310 mm wide, with the wall in front of the
mouth, at the inspection level.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.preview.api import build_preview_geometry
from hornlab_mesher.preview.contract import PreviewOptionsV1
from hornlab_mesher.profile_sampling import _outer_offset_shell

WALL_MM = 5.0

ROLLED_BACK_ROSSE = {
    "formula": "R-OSSE",
    "mode": "freestanding",
    "profile": {
        "R_mm": 150.0,
        "r0_mm": 12.7,
        "a_deg": 45,
        "a0_deg": 10,
        "k": 1.0,
        "r": 0.4,
        "m": 0.8,
        "b": 0.3,
        "q": 3.4,
    },
    "morph": {"morph_target": 1, "morph_corner_mm": 20, "morph_fixed": 0.5},
    "mesh": {
        "angular_segments": 64,
        "length_segments": 32,
        "wall_thickness_mm": WALL_MM,
        "throat_res_mm": 4.0,
        "mouth_res_mm": 12.0,
        "rear_res_mm": 15.0,
    },
}


def _preview_bounds(config: dict, lod: str) -> tuple[np.ndarray, np.ndarray]:
    geometry = build_preview_geometry(config, PreviewOptionsV1(lod=lod))
    points = np.vstack(
        [np.asarray(surface.positions, dtype=float).reshape(-1, 3) for surface in geometry.surfaces]
    )
    return points.min(axis=0), points.max(axis=0)


def _inner_front_z(config: dict) -> float:
    return float(np.asarray(resolve_geometry(config).geometry.inner_points)[..., 2].max())


@pytest.mark.parametrize("angular_segments", [32, 64, 128])
def test_preview_levels_agree_on_the_overall_size(angular_segments: int) -> None:
    config = copy.deepcopy(ROLLED_BACK_ROSSE)
    config["mesh"]["angular_segments"] = angular_segments
    front = _inner_front_z(config)
    bounds = {lod: _preview_bounds(config, lod) for lod in ("coarse", "fine", "inspection")}
    for lod, (low, high) in bounds.items():
        # The wall is behind the acoustic surface: nothing lies in front of
        # the inner wall's forward-most point, and the rollback's own 150 mm
        # radius stays the widest point.
        assert high[2] == pytest.approx(front, abs=0.05), lod
        np.testing.assert_allclose(high[:2] - low[:2], [300.0, 300.0], atol=0.05, err_msg=lod)


def _ring_radius(points: np.ndarray, column: int) -> np.ndarray:
    return np.hypot(points[:, column, 0], points[:, column, 1])


@pytest.mark.parametrize("angular_segments", [32, 64, 128])
@pytest.mark.parametrize("length_segments", [16, 32, 64])
def test_solve_geometry_wall_is_outside_the_throat_and_behind_the_mouth(
    angular_segments: int, length_segments: int
) -> None:
    config = copy.deepcopy(ROLLED_BACK_ROSSE)
    config["mesh"]["angular_segments"] = angular_segments
    config["mesh"]["length_segments"] = length_segments
    geometry = resolve_geometry(config).geometry
    inner = np.asarray(geometry.inner_points)
    outer = np.asarray(geometry.outer_points)
    assert np.all(_ring_radius(outer, 0) > _ring_radius(inner, 0))
    assert outer[..., 2].max() <= inner[..., 2].max() + 1e-9


def _rolled_back_grid(n_phi: int, n_t: int) -> np.ndarray:
    """A bore that opens and rolls back through 150 degrees, as (phi, t, xyz)."""
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    turn = np.radians(np.linspace(10.0, 150.0, n_t))
    step = 120.0 / (n_t - 1)
    z = np.concatenate([[0.0], np.cumsum(np.cos(0.5 * (turn[1:] + turn[:-1])) * step)])
    r = 12.7 + np.concatenate([[0.0], np.cumsum(np.sin(0.5 * (turn[1:] + turn[:-1])) * step)])
    grid = np.empty((n_phi, n_t, 3))
    grid[:, :, 0] = r[None, :] * np.cos(phi)[:, None]
    grid[:, :, 1] = r[None, :] * np.sin(phi)[:, None]
    grid[:, :, 2] = z[None, :]
    return grid


@pytest.mark.parametrize("n_phi", [16, 24, 40, 64, 96])
@pytest.mark.parametrize("n_t", [12, 33, 65, 130])
def test_offset_side_does_not_depend_on_grid_size(n_phi: int, n_t: int) -> None:
    inner = _rolled_back_grid(n_phi, n_t)
    outer = _outer_offset_shell(inner, WALL_MM, full_circle=True)
    assert np.all(_ring_radius(outer, 0) > _ring_radius(inner, 0))
    # Past the forward-most ring the tangent points back, so the wall is on the
    # axis side there.
    assert np.all(_ring_radius(outer, -1) < _ring_radius(inner, -1))
