from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher import (
    normalized_arc_positions,
    resample_grid_onto_existing,
    resample_point_grid,
)


def _cylindrical_grid(n_phi: int, z: np.ndarray) -> np.ndarray:
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    radius = 2.0 + np.asarray(z) ** 2
    return np.stack(
        [
            radius[None, :] * np.cos(phi)[:, None],
            radius[None, :] * np.sin(phi)[:, None],
            np.broadcast_to(z, (n_phi, len(z))),
        ],
        axis=2,
    )


def test_normalized_arc_positions_follow_distance_not_source_index():
    points = _cylindrical_grid(8, np.asarray([0.0, 0.1, 0.4, 1.0]))
    positions = normalized_arc_positions(points)

    assert positions[0] == 0.0
    assert positions[-1] == 1.0
    assert positions[1] < 1.0 / 3.0
    assert np.all(np.diff(positions) > 0.0)


def test_resample_point_grid_is_periodic_in_phi_and_hits_arc_endpoints():
    points = _cylindrical_grid(8, np.linspace(0.0, 1.0, 6))
    output = resample_point_grid(
        points,
        point_count=16,
        section_arc_positions=np.asarray([0.0, 0.25, 1.0]),
    )

    assert output.shape == (16, 3, 3)
    assert output[0, 0] == pytest.approx(points[0, 0])
    assert output[0, -1] == pytest.approx(points[0, -1])
    assert output[8, :, 1] == pytest.approx(0.0, abs=1.0e-12)
    assert np.all(output[8, :, 0] < 0.0)


def test_resample_grid_onto_existing_preserves_exact_cad_counts_for_both_walls():
    old = _cylindrical_grid(12, np.linspace(0.0, 1.0, 10) ** 2)
    new = _cylindrical_grid(17, np.linspace(0.0, 1.0, 13) ** 1.5)
    outer = new * np.asarray([1.1, 1.1, 1.0])

    result = resample_grid_onto_existing(
        new,
        old,
        phi_stride=3,
        ring_stride=4,
        new_outer_points=outer,
    )

    assert result.points.shape == (4, 4, 3)
    assert result.outer_points is not None
    assert result.outer_points.shape == result.points.shape
    assert result.old_section_indices == (0, 4, 8, 9)
    assert result.section_arc_positions[0] == 0.0
    assert result.section_arc_positions[-1] == 1.0


@pytest.mark.parametrize("stride", [0, -1])
def test_resample_grid_rejects_nonpositive_strides(stride):
    grid = _cylindrical_grid(8, np.linspace(0.0, 1.0, 4))
    with pytest.raises(ValueError, match="strides must be positive"):
        resample_grid_onto_existing(
            grid, grid, phi_stride=stride, ring_stride=1
        )
