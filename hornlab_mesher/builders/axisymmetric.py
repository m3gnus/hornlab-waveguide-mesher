from __future__ import annotations

import numpy as np

from ..geometry import _AxiHornGeometry, BuiltGeometry
from ..tags import PhysicalGroup
from ._occ import (
    build_bspline_surface_from_rings,
    make_planar_fill_from_ring,
    superellipse_ring,
)


def _validated_profile(geometry: _AxiHornGeometry) -> np.ndarray:
    points = np.asarray(geometry.profile_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("profile_points must be an (N, 2) array with at least two points")
    if not np.all(np.isfinite(points)):
        raise ValueError("profile_points contains non-finite values")
    if np.any(np.diff(points[:, 0]) <= 0):
        raise ValueError("profile_points z coordinates must be strictly increasing")
    if np.any(points[:, 1] <= 0):
        raise ValueError("profile_points radii must be > 0")
    return points


def _build_axisymmetric(geometry: _AxiHornGeometry) -> BuiltGeometry:
    profile = _validated_profile(geometry)
    rings = [
        superellipse_ring(
            z=float(z),
            radius=float(r),
            exponent=geometry.cross_section.exponent,
            aspect_ratio=geometry.cross_section.aspect_ratio,
            n_phi=geometry.n_phi,
        )
        for z, r in profile
    ]
    grid = np.stack(rings, axis=1)

    wall = build_bspline_surface_from_rings(grid)
    throat = make_planar_fill_from_ring(grid[:, 0, :])

    return BuiltGeometry(
        surface_groups={
            int(PhysicalGroup.RIGID_WALL): [tag for _, tag in wall],
            int(PhysicalGroup.PRIMARY_SOURCE): [tag for _, tag in throat],
        },
        axial_bounds_mm=(float(profile[0, 0]), float(profile[-1, 0])),
        source_axis="z",
        mesh_surface_groups={
            "inner": [tag for _, tag in wall],
            "throat_disc": [tag for _, tag in throat],
        },
    )
