"""Rigid placement must not change the freestanding surface construction."""

import numpy as np
import pytest

from hornlab_mesher.builders.point_grid_freestanding import _grid_is_axisymmetric


def _circular_grid(span):
    phi = np.linspace(0.0, span, 17)
    radius = np.array([12.0, 28.0, 70.0])
    return np.stack(
        (
            np.cos(phi)[:, None] * radius,
            np.sin(phi)[:, None] * radius,
            np.broadcast_to([0.0, 40.0, 100.0], (len(phi), 3)),
        ),
        axis=2,
    )


@pytest.mark.parametrize("span", [np.pi / 2, np.pi, 2 * np.pi])
@pytest.mark.parametrize(
    "translation", [[0.0, 0.0, 0.0], [0.0, 80.0, 0.0], [120.0, -300.0, 40.0]]
)
def test_circular_full_and_reduced_grids_are_axisymmetric_after_translation(
    span, translation
):
    assert _grid_is_axisymmetric(_circular_grid(span) + translation)


@pytest.mark.parametrize(
    "deformation", ["ellipse", "moving_center", "tilted_station", "single_point"]
)
def test_translation_does_not_hide_a_non_axisymmetric_grid(deformation):
    grid = _circular_grid(2 * np.pi)
    if deformation == "ellipse":
        grid[:, -1, 0] *= 1.1
    elif deformation == "moving_center":
        grid[:, -1, 1] += 3.0
    elif deformation == "tilted_station":
        grid[:, -1, 2] += np.linspace(0.0, 1.0, len(grid))
    else:
        grid[3, -1, 0] += 1.0
    assert not _grid_is_axisymmetric(grid + [120.0, 80.0, 40.0])
