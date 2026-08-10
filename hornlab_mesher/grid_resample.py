"""Cubic resampling of periodic point grids onto an existing CAD topology."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class ResampledPointGrid:
    """A new grid sampled at an existing grid's point and section topology."""

    points: np.ndarray
    section_arc_positions: np.ndarray
    old_section_indices: tuple[int, ...]
    outer_points: np.ndarray | None = None


def normalized_arc_positions(points: np.ndarray) -> np.ndarray:
    """Return shared normalized cumulative arc positions along grid rays."""
    points = _validated_grid(points, name="points")
    steps = np.linalg.norm(np.diff(points, axis=1), axis=2)
    cumulative = np.concatenate(
        [np.zeros((points.shape[0], 1)), np.cumsum(steps, axis=1)], axis=1
    )
    lengths = cumulative[:, -1:]
    if np.any(lengths <= 0.0):
        raise ValueError("every grid ray must have positive arc length")
    positions = (cumulative / lengths).mean(axis=0)
    if np.any(np.diff(positions) <= 0.0):
        raise ValueError("averaged arc positions must be strictly increasing")
    return positions


def resample_point_grid(
    points: np.ndarray,
    *,
    point_count: int,
    section_arc_positions: np.ndarray,
) -> np.ndarray:
    """Resample periodically in phi, then cubically in normalized arc length."""
    points = _validated_grid(points, name="points")
    if point_count < 3:
        raise ValueError("point_count must be at least 3 for periodic cubic sampling")
    targets = np.asarray(section_arc_positions, dtype=float)
    if targets.ndim != 1 or len(targets) < 2:
        raise ValueError("section_arc_positions must be a one-dimensional array")
    if not np.all(np.isfinite(targets)) or np.any(np.diff(targets) <= 0.0):
        raise ValueError("section_arc_positions must be finite and strictly increasing")
    if targets[0] < 0.0 or targets[-1] > 1.0:
        raise ValueError("section_arc_positions must lie in [0, 1]")

    n_phi = points.shape[0]
    u_in = np.arange(n_phi + 1, dtype=float) / n_phi
    closed = np.concatenate([points, points[:1]], axis=0)
    phi_spline = CubicSpline(u_in, closed, axis=0, bc_type="periodic")
    phi_points = phi_spline(np.arange(point_count, dtype=float) / point_count)
    source_positions = normalized_arc_positions(phi_points)
    output = np.empty((point_count, len(targets), 3), dtype=float)
    for index in range(point_count):
        output[index] = CubicSpline(source_positions, phi_points[index], axis=0)(targets)
    return output


def resample_grid_onto_existing(
    new_points: np.ndarray,
    old_points: np.ndarray,
    *,
    phi_stride: int,
    ring_stride: int,
    new_outer_points: np.ndarray | None = None,
) -> ResampledPointGrid:
    """Resample a new grid onto the exact topology selected from an old grid."""
    new_points = _validated_grid(new_points, name="new_points")
    old_points = _validated_grid(old_points, name="old_points")
    if phi_stride <= 0 or ring_stride <= 0:
        raise ValueError("strides must be positive")
    point_count = len(range(0, old_points.shape[0], phi_stride))
    indices = list(range(0, old_points.shape[1], ring_stride))
    if indices[-1] != old_points.shape[1] - 1:
        indices.append(old_points.shape[1] - 1)
    targets = normalized_arc_positions(old_points)[indices]
    resampled = resample_point_grid(
        new_points, point_count=point_count, section_arc_positions=targets
    )
    outer = None
    if new_outer_points is not None:
        outer = resample_point_grid(
            new_outer_points,
            point_count=point_count,
            section_arc_positions=targets,
        )
    return ResampledPointGrid(resampled, targets, tuple(indices), outer)


def _validated_grid(points: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must have shape (n_phi, n_length, 3)")
    if array.shape[0] < 3 or array.shape[1] < 2:
        raise ValueError(f"{name} must contain at least 3 rays and 2 sections")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite coordinates")
    return array
