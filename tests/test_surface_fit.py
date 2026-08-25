"""Acoustic B-spline surface fitting.

``occ.addBSplineSurface`` treats its points as *poles*, so handing it the
sampled profile grid meshed a surface that sat systematically inside the
sampled one -- an error floor no element size could reach. These cover the
interpolating fit that removes it.
"""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.builders._occ import (
    SURFACE_FIT_APPROXIMATE,
    SURFACE_FIT_INTERPOLATE,
    grid_v_parameters,
    interpolating_surface_poles,
)
from hornlab_mesher.geometry import PointGridHornGeometry


def _revolved_grid(n_phi: int = 24, n_len: int = 12) -> np.ndarray:
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    z = np.linspace(0.0, 40.0, n_len)
    grid = np.empty((n_phi, n_len, 3), dtype=np.float64)
    for i, angle in enumerate(phi):
        radius = 10.0 + 0.6 * z
        grid[i, :, 0] = radius * np.cos(angle)
        grid[i, :, 1] = radius * np.sin(angle)
        grid[i, :, 2] = z
    return grid


# --------------------------------------------------------------------------
# B-spline surface fitting
# --------------------------------------------------------------------------


def test_interpolating_poles_reproduce_the_sampled_grid_exactly():
    """The whole point of the fit: the surface passes through its samples.

    A pole fit does not, and that miss is the error floor the acoustic mesh
    was converging onto.
    """
    from scipy.interpolate import BSpline

    grid = _revolved_grid()
    columns = list(range(grid.shape[0])) + [0]
    patch = np.ascontiguousarray(grid[columns, :, :].transpose(1, 0, 2))

    poles, (knots_u, mults_u), (knots_v, mults_v) = interpolating_surface_poles(
        patch, degree_u=3, degree_v=3, v_params=grid_v_parameters(grid)
    )
    assert poles.shape == patch.shape

    full_u = np.repeat(knots_u, mults_u)
    full_v = np.repeat(knots_v, mults_v)
    # A clamped knot vector carries n + degree + 1 entries.
    assert len(full_u) == poles.shape[1] + 3 + 1
    assert len(full_v) == poles.shape[0] + 3 + 1

    u_params = np.linspace(full_u[0], full_u[-1], patch.shape[1])
    v_params = np.linspace(full_v[0], full_v[-1], patch.shape[0])
    worst = 0.0
    for vi, v in enumerate(v_params):
        row = BSpline(full_v, poles, 3)(v)
        for ui, u in enumerate(u_params):
            worst = max(worst, float(np.linalg.norm(BSpline(full_u, row, 3)(u) - patch[vi, ui])))
    assert worst < 1e-9, worst


def test_seam_poles_coincide_so_a_closed_patch_still_closes():
    grid = _revolved_grid()
    columns = list(range(grid.shape[0])) + [0]
    patch = np.ascontiguousarray(grid[columns, :, :].transpose(1, 0, 2))
    poles, _, _ = interpolating_surface_poles(
        patch, degree_u=3, degree_v=3, v_params=grid_v_parameters(grid)
    )
    assert np.allclose(poles[:, 0], poles[:, -1], atol=0.0)


def test_patches_cut_from_one_grid_share_a_v_knot_vector():
    """Neighbouring patches must trace the *same* seam curve.

    Deriving v from each patch's own columns gives each a slightly different
    parameterisation, which tears the shell open along every seam.
    """
    grid = _revolved_grid()
    shared = grid_v_parameters(grid)
    left = np.ascontiguousarray(grid[list(range(0, 13)), :, :].transpose(1, 0, 2))
    right = np.ascontiguousarray(grid[list(range(12, 24)) + [0], :, :].transpose(1, 0, 2))

    _, _, (knots_left, mults_left) = interpolating_surface_poles(
        left, degree_u=3, degree_v=3, v_params=shared
    )
    _, _, (knots_right, mults_right) = interpolating_surface_poles(
        right, degree_u=3, degree_v=3, v_params=shared
    )
    assert np.allclose(knots_left, knots_right)
    assert mults_left == mults_right


def test_degenerate_rings_fall_back_to_a_uniform_parameterisation():
    """A collapsed apex ring must not produce a singular collocation system."""
    grid = _revolved_grid(n_phi=16, n_len=6)
    grid[:, 0, :] = grid[0, 0, :]  # collapse the first ring to a single point
    patch = np.ascontiguousarray(grid[list(range(16)) + [0], :, :].transpose(1, 0, 2))
    poles, _, _ = interpolating_surface_poles(patch, degree_u=3, degree_v=3)
    assert np.all(np.isfinite(poles))


def test_surface_fit_defaults_to_the_approximating_pole_fit():
    geometry = PointGridHornGeometry(inner_points=_revolved_grid())
    assert geometry.surface_fit == SURFACE_FIT_APPROXIMATE


def test_surface_fit_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="surface_fit"):
        PointGridHornGeometry(inner_points=_revolved_grid(), surface_fit="curvature")


def test_config_reads_both_surface_fit_spellings():
    from hornlab_mesher.config_builder import ConfigError, _mesh_surface_fit

    assert _mesh_surface_fit({}) == SURFACE_FIT_APPROXIMATE
    assert _mesh_surface_fit({"surface_fit": "interpolate"}) == SURFACE_FIT_INTERPOLATE
    assert _mesh_surface_fit({"surfaceFit": "Interpolate"}) == SURFACE_FIT_INTERPOLATE
    with pytest.raises(ConfigError, match="surface_fit"):
        _mesh_surface_fit({"surface_fit": "nearest"})
