"""Analytic curvature contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.preview.api import PreviewOptionsV1, build_preview_geometry
from hornlab_mesher.preview.fidelity import (
    analytic_grid_curvature,
    analytic_grid_normals,
)
from tests.test_preview_api import ROSSE_ENCLOSURE


def test_sphere_patch_curvature_matches_returned_inward_normal():
    radius = 25.0
    t = np.linspace(0.25, 1.25, 81)
    phi = np.arange(192, dtype=np.float64) * np.pi * 2.0 / 192
    theta, azimuth = np.meshgrid(t, phi, indexing="ij")
    grid = np.stack(
        (
            radius * np.sin(theta) * np.cos(azimuth),
            radius * np.sin(theta) * np.sin(azimuth),
            radius * np.cos(theta),
        ),
        axis=2,
    )

    normals = analytic_grid_normals(grid, closed_phi=True, t_coordinates=t)
    mean, principal = analytic_grid_curvature(
        grid, closed_phi=True, t_coordinates=t
    )

    assert np.all(np.sum(normals * grid, axis=2) < 0.0)
    assert mean == pytest.approx(np.full(mean.shape, 1.0 / radius), abs=3.0e-5)
    assert principal == pytest.approx(
        np.full(principal.shape, 1.0 / radius), abs=5.0e-5
    )


def test_cylinder_patch_has_one_zero_principal_curvature_and_matching_sign():
    radius = 20.0
    t = np.linspace(-10.0, 10.0, 81)
    phi = np.arange(192, dtype=np.float64) * np.pi * 2.0 / 192
    axial, azimuth = np.meshgrid(t, phi, indexing="ij")
    # z decreases with t, making dP/dphi x dP/dt point radially inward.
    grid = np.stack(
        (
            radius * np.cos(azimuth),
            radius * np.sin(azimuth),
            -axial,
        ),
        axis=2,
    )

    normals = analytic_grid_normals(grid, closed_phi=True, t_coordinates=t)
    mean, principal = analytic_grid_curvature(
        grid, closed_phi=True, t_coordinates=t
    )

    radial = grid.copy()
    radial[:, :, 2] = 0.0
    assert np.all(np.sum(normals * radial, axis=2) < 0.0)
    assert mean == pytest.approx(np.full(mean.shape, 0.5 / radius), abs=2.0e-6)
    assert principal == pytest.approx(
        np.full(principal.shape, 1.0 / radius), abs=3.0e-6
    )
    other_principal = 2.0 * mean - principal
    assert other_principal == pytest.approx(np.zeros_like(mean), abs=3.0e-6)


def test_plane_curvature_is_exactly_zero():
    t = np.linspace(-2.0, 3.0, 11)
    phi = np.linspace(-4.0, 5.0, 13)
    first, second = np.meshgrid(t, phi, indexing="ij")
    grid = np.stack((first, second, 2.0 * first - 0.5 * second), axis=2)
    phi_grid = np.broadcast_to(phi, grid.shape[:2])

    mean, principal = analytic_grid_curvature(
        grid,
        closed_phi=False,
        t_coordinates=t,
        phi_coordinates=phi_grid,
    )

    assert np.array_equal(mean, np.zeros_like(mean))
    assert np.array_equal(principal, np.zeros_like(principal))


def test_real_preview_surfaces_have_finite_row_aligned_curvature():
    preview = build_preview_geometry(
        ROSSE_ENCLOSURE, PreviewOptionsV1(lod="coarse")
    )

    for surface in preview.surfaces:
        assert surface.curvature_mean is not None
        assert surface.curvature_principal is not None
        assert surface.curvature_mean.shape == (len(surface.positions),)
        assert surface.curvature_principal.shape == (len(surface.positions),)
        assert np.all(np.isfinite(surface.curvature_mean))
        assert np.all(np.isfinite(surface.curvature_principal))
        expected = "planar" if surface.normal_method == "exact-planar" else "analytic"
        assert surface.metadata["curvature"] == expected


def test_curvature_can_be_omitted_explicitly():
    preview = build_preview_geometry(
        ROSSE_ENCLOSURE,
        PreviewOptionsV1(lod="coarse", include_curvature=False),
    )

    for surface in preview.surfaces:
        assert surface.curvature_mean is None
        assert surface.curvature_principal is None
        assert surface.metadata["curvature"] == "absent"
