from __future__ import annotations

import numpy as np

from ..geometry import BuiltGeometry, PointGridHornGeometry
from ..tags import PhysicalGroup
from ._occ import require_gmsh
from .point_grid_sources import (
    _add_geo_source_cap_surfaces,
    _add_source_surfaces,
)
from .point_grid_surfaces import (
    _GeoSurfaceBuilder,
    _SharedSurfaceBuilder,
    _add_geo_spline_span_mouth_rim_surfaces,
    _add_geo_spline_span_rear_cap,
    _add_geo_spline_span_wall_surfaces,
    _add_grid_wall_surfaces,
    _add_mouth_rim_surfaces,
    _add_rear_cap,
    _rear_rim_points,
    _snap_open_symmetry_grid,
    _validated_grid,
)

def _restored_outer_throat_points(
    inner_points: np.ndarray,
    outer_points: np.ndarray,
    *,
    wall_thickness_mm: float,
) -> np.ndarray:
    """Undo the legacy point-grid throat clamp before adding the rear return.

    WG's legacy payload path flattened the outer throat ring to
    ``inner_z - wallThickness`` as a shortcut for the rear plate. ATH builds
    the outer shell at the throat and then adds a normal/backward rear return.
    """

    out = np.array(outer_points, dtype=np.float64, copy=True)
    expected = inner_points[:, 0, 2] - float(wall_thickness_mm)
    if np.allclose(out[:, 0, 2], expected, rtol=0.0, atol=1.0e-6):
        out[:, 0, 2] = inner_points[:, 0, 2]
    return out


def _outer_wall_axial_ring_indices(inner_points: np.ndarray) -> list[int]:
    """Select axial rings used by the outer return wall.

    The throat ring is always present. Intermediate rings are retained until
    the horn reaches the mouth-side maximum-z plane, avoiding a degenerate
    outer-wall strip at the mouth rim.
    """
    z_by_ring = np.mean(inner_points[:, :, 2], axis=0)
    max_z = float(np.max(z_by_ring))
    tol = max(1.0e-3, 1.0e-8 * max(1.0, abs(max_z)))
    return [
        j
        for j in range(1, inner_points.shape[1])
        if float(z_by_ring[j]) < max_z - tol
    ]


def _build_freestanding_point_grid(geometry: PointGridHornGeometry) -> BuiltGeometry:
    inner_points = _validated_grid(geometry.inner_points, name="inner_points")
    if geometry.outer_points is None:
        raise ValueError("freestanding point-grid build requires outer_points")
    outer_points = _validated_grid(geometry.outer_points, name="outer_points")
    outer_points = _restored_outer_throat_points(
        inner_points,
        outer_points,
        wall_thickness_mm=float(geometry.wall_thickness_mm),
    )
    if geometry.wg_topology:
        return _build_wg_freestanding_point_grid(
            geometry,
            inner_points,
            outer_points,
        )

    n_phi, n_len, _ = inner_points.shape
    rear_z = float(np.mean(inner_points[:, 0, 2]) - float(geometry.wall_thickness_mm))
    rear_points = _rear_rim_points(outer_points, rear_z=rear_z)
    outer_topology = np.empty((n_phi, n_len + 1, 3), dtype=np.float64)
    outer_topology[:, 0, :] = rear_points
    outer_topology[:, 1:, :] = outer_points

    builder = _SharedSurfaceBuilder()
    builder.add_grid("inner", inner_points)
    builder.add_grid("outer", outer_topology)

    wall = _add_grid_wall_surfaces(
        builder,
        "inner",
        n_phi=n_phi,
        n_len=n_len,
        closed=geometry.closed,
    )
    outer_wall = _add_grid_wall_surfaces(
        builder,
        "outer",
        n_phi=n_phi,
        n_len=outer_topology.shape[1],
        closed=geometry.closed,
        reverse=True,
    )
    mouth_dimtags = _add_mouth_rim_surfaces(
        builder,
        n_phi=n_phi,
        n_len=n_len,
        outer_len=outer_topology.shape[1],
        closed=geometry.closed,
    )
    rear_cap = _add_rear_cap(
        builder,
        rear_points,
        grid_name="outer",
        n_phi=n_phi,
        closed=geometry.closed,
    )
    throat = _add_source_surfaces(builder, inner_points, geometry)

    wall_tags = [tag for _, tag in wall]
    outer_tags = [tag for _, tag in outer_wall]
    mouth_tags = [tag for _, tag in mouth_dimtags]
    rear_tags = [tag for _, tag in rear_cap]
    throat_tags = [tag for _, tag in throat]
    rigid_wall_tags = [
        *wall_tags,
        *outer_tags,
        *mouth_tags,
        *rear_tags,
    ]

    z0 = float(np.mean(inner_points[:, 0, 2]))
    z1 = float(np.mean(inner_points[:, -1, 2]))
    return BuiltGeometry(
        surface_groups={
            int(PhysicalGroup.RIGID_WALL): rigid_wall_tags,
            int(PhysicalGroup.PRIMARY_SOURCE): throat_tags,
        },
        axial_bounds_mm=(z0, z1),
        source_axis="z",
        mesh_surface_groups={
            "inner": wall_tags,
            "throat_disc": throat_tags,
            "outer": outer_tags,
            "mouth": mouth_tags,
            "rear": rear_tags,
            "rear_cap": rear_tags,
        },
        symmetry_snap_axes=() if geometry.closed else tuple(geometry.symmetry_planes),
        symmetry_snap_tol_mm=1.0,
    )


def _build_wg_freestanding_point_grid(
    geometry: PointGridHornGeometry,
    inner_points: np.ndarray,
    outer_points: np.ndarray,
) -> BuiltGeometry:
    inner_points = _snap_open_symmetry_grid(
        inner_points, closed=geometry.closed, symmetry_planes=geometry.symmetry_planes
    )
    outer_points = _snap_open_symmetry_grid(
        outer_points, closed=geometry.closed, symmetry_planes=geometry.symmetry_planes
    )

    n_phi, inner_len, _ = inner_points.shape
    outer_indices = _outer_wall_axial_ring_indices(inner_points)
    rear_z = float(np.mean(inner_points[:, 0, 2]) - float(geometry.wall_thickness_mm))
    rear_points = _rear_rim_points(outer_points, rear_z=rear_z)
    outer_topology = np.empty((n_phi, len(outer_indices) + 2, 3), dtype=np.float64)
    outer_topology[:, 0, :] = rear_points
    outer_topology[:, 1, :] = outer_points[:, 0, :]
    for out_j, src_j in enumerate(outer_indices, start=2):
        outer_topology[:, out_j, :] = outer_points[:, src_j, :]

    builder = _GeoSurfaceBuilder()
    # These per-point sizes are inert: density.py sets MeshSizeFromPoints=0
    # and element sizing comes from the Restrict fields.
    inner_mesh_sizes = np.full(inner_points.shape[:2], 8.0, dtype=np.float64)
    inner_mesh_sizes[:, 0] = 5.0
    builder.add_grid("inner", inner_points, mesh_size=inner_mesh_sizes)
    builder.add_grid("outer", outer_topology, mesh_size=25.0)

    wall = _add_geo_spline_span_wall_surfaces(
        builder,
        "inner",
        n_phi=n_phi,
        n_len=inner_len,
        closed=geometry.closed,
    )
    outer_wall = _add_geo_spline_span_wall_surfaces(
        builder,
        "outer",
        n_phi=n_phi,
        n_len=outer_topology.shape[1],
        closed=geometry.closed,
        reverse=True,
    )
    mouth_dimtags = _add_geo_spline_span_mouth_rim_surfaces(
        builder,
        n_phi=n_phi,
        inner_len=inner_len,
        outer_len=outer_topology.shape[1],
        closed=geometry.closed,
    )
    rear_cap = _add_geo_spline_span_rear_cap(
        builder,
        rear_points,
        n_phi=n_phi,
        closed=geometry.closed,
        mesh_size=25.0,
    )
    throat = _add_geo_source_cap_surfaces(
        builder,
        inner_points,
        geometry,
        mesh_size=5.0,
    )
    require_gmsh().model.geo.synchronize()

    wall_tags = [tag for _, tag in wall]
    outer_tags = [tag for _, tag in outer_wall]
    mouth_tags = [tag for _, tag in mouth_dimtags]
    rear_tags = [tag for _, tag in rear_cap]
    throat_tags = [tag for _, tag in throat]
    rigid_wall_tags = [*wall_tags, *outer_tags, *mouth_tags, *rear_tags]

    z0 = float(np.mean(inner_points[:, 0, 2]))
    z1 = float(np.mean(inner_points[:, -1, 2]))
    return BuiltGeometry(
        surface_groups={
            int(PhysicalGroup.RIGID_WALL): rigid_wall_tags,
            int(PhysicalGroup.PRIMARY_SOURCE): throat_tags,
        },
        axial_bounds_mm=(z0, z1),
        source_axis="z",
        mesh_surface_groups={
            "inner": wall_tags,
            "throat_disc": throat_tags,
            "outer": outer_tags,
            "mouth": mouth_tags,
            "rear": rear_tags,
            "rear_cap": rear_tags,
        },
        symmetry_snap_axes=() if geometry.closed else tuple(geometry.symmetry_planes),
        symmetry_snap_tol_mm=1.0,
        mesh_algorithm=2,
    )
