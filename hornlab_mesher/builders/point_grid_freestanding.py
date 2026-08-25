from __future__ import annotations

from typing import Any

import numpy as np

from ..geometry import BuiltGeometry, PointGridHornGeometry
from ..tags import PhysicalGroup
from ._occ import make_planar_fill_from_boundary, require_gmsh
from .point_grid_sources import (
    _add_occ_source_cap_surfaces,
    _add_geo_source_cap_surfaces,
    _add_source_surfaces,
)
from .point_grid_surfaces import (
    _GeoSurfaceBuilder,
    _SharedSurfaceBuilder,
    _add_geo_spline_span_mouth_rim_surfaces,
    _add_geo_spline_span_rear_cap,
    _add_grid_wall_surfaces,
    _add_occ_bspline_patch_wall_surfaces,
    _add_mouth_rim_surfaces,
    _add_rear_cap,
    _add_spline_span_wall_surfaces,
    _rear_rim_points,
    _bspline_patch_phi_groups,
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
        j for j in range(1, inner_points.shape[1]) if float(z_by_ring[j]) < max_z - tol
    ]


def _ring_curvature_radius_mm(ring_xy: np.ndarray, *, closed: bool) -> np.ndarray:
    """Local curvature radius at each point of a cross-section ring.

    The circumradius of each consecutive triple. Three collinear points give an
    infinite radius, which is what a flat run of a section should contribute:
    no constraint. On an open (symmetry-reduced) ring the two ends have no
    triple, so they are left unconstrained rather than wrapped across the cut.
    """

    points = np.asarray(ring_xy, dtype=np.float64)
    count = points.shape[0]
    radius = np.full(points.shape[:-1], np.inf, dtype=np.float64)
    if count < 3:
        return radius

    previous = np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0)
    first = points - previous
    second = following - points
    third = following - previous
    side_a = np.linalg.norm(first, axis=-1)
    side_b = np.linalg.norm(second, axis=-1)
    side_c = np.linalg.norm(third, axis=-1)
    twice_area = np.abs(
        first[..., 0] * third[..., 1] - first[..., 1] * third[..., 0]
    )
    np.divide(
        side_a * side_b * side_c,
        2.0 * twice_area,
        out=radius,
        where=twice_area > 0.0,
    )
    if not closed:
        radius[0] = np.inf
        radius[-1] = np.inf
    return radius


def _wall_clearance_metadata(
    geometry: PointGridHornGeometry, outer_points: np.ndarray
) -> dict[str, Any]:
    """Publish what the density stage needs to keep the shell off the bore.

    ``configure_density`` only receives a ``BuiltGeometry``, which knows nothing
    about wall thickness -- yet the wall is exactly the clearance budget a
    coarsely faceted outer shell spends. Passing it through metadata keeps the
    builder/density split intact and leaves every other builder untouched: no
    block means no cap, and the sizing is byte-identical to before.

    What each ring contributes is its tightest CURVATURE radius, not its
    smallest distance from the axis. The two are the same only for a circular
    ring centred on the axis. A rounded rectangle's nearest point is the middle
    of a flat side, which curves not at all, while its corner fillet may be a
    twentieth of that radius -- measured on 100 mm half-extents with a 5 mm
    fillet, distance from the axis licenses a chord 4.9x coarser than the
    corner can carry.
    """

    curvature = _ring_curvature_radius_mm(
        outer_points[..., :2], closed=bool(geometry.closed)
    )
    ring_curvature = curvature.min(axis=0)
    finite = ring_curvature[np.isfinite(ring_curvature)]
    return {
        "wallThicknessMm": float(geometry.wall_thickness_mm),
        "minOuterCurvatureRadiusMm": float(finite.min()) if finite.size else 0.0,
        "ringMinCurvatureRadiusMm": [float(value) for value in ring_curvature],
        "ringMaxAxialMm": [float(value) for value in outer_points[..., 2].max(axis=0)],
    }


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
    if geometry.topology_mode == "acoustic" and not geometry.preserve_grid:
        return _build_acoustic_freestanding_point_grid(
            geometry,
            inner_points,
            outer_points,
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
    throat = _add_source_surfaces(builder, inner_points, geometry, wall_dimtags=wall)

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
        metadata={
            "outerWallClearance": _wall_clearance_metadata(geometry, outer_points)
        },
    )


def _build_acoustic_freestanding_point_grid(
    geometry: PointGridHornGeometry,
    inner_points: np.ndarray,
    outer_points: np.ndarray,
) -> BuiltGeometry:
    """Freestanding shell with geometry samples decoupled from BEM topology."""

    inner_points = _snap_open_symmetry_grid(
        inner_points, closed=geometry.closed, symmetry_planes=geometry.symmetry_planes
    )
    outer_points = _snap_open_symmetry_grid(
        outer_points, closed=geometry.closed, symmetry_planes=geometry.symmetry_planes
    )
    n_phi = inner_points.shape[0]
    outer_indices = _outer_wall_axial_ring_indices(inner_points)
    rear_z = float(np.mean(inner_points[:, 0, 2]) - float(geometry.wall_thickness_mm))
    rear_points = _rear_rim_points(outer_points, rear_z=rear_z)
    outer_topology = np.empty((n_phi, len(outer_indices) + 2, 3), dtype=np.float64)
    outer_topology[:, 0, :] = rear_points
    outer_topology[:, 1, :] = outer_points[:, 0, :]
    for out_j, src_j in enumerate(outer_indices, start=2):
        outer_topology[:, out_j, :] = outer_points[:, src_j, :]

    phi_groups = _bspline_patch_phi_groups(
        n_phi,
        closed=geometry.closed,
        n_sectors=1,
    )
    inner_wall = _add_occ_bspline_patch_wall_surfaces(
        inner_points,
        closed=geometry.closed,
        phi_groups=phi_groups,
        surface_fit=geometry.surface_fit,
    )
    # The outer shell keeps the approximating pole fit regardless of
    # ``surface_fit``. ``outer_topology`` splices the rear rim onto the wall as
    # a deliberate sharp corner (z jumps from the rear plane back to the throat
    # plane), and a cubic *interpolating* spline overshoots such a corner — on
    # the stock OSSE it dips the rear boundary to z = -9.09 mm instead of the
    # -6.00 mm rear plane, which stops the rear cap's planar-extreme detection
    # from ever finding its loop. The acoustic (inner) surface is the one whose
    # fidelity the BEM result depends on; the outer shell is rigid backing.
    outer_wall = _add_occ_bspline_patch_wall_surfaces(
        outer_topology,
        closed=geometry.closed,
        phi_groups=phi_groups,
    )
    gmsh = require_gmsh()
    gmsh.model.occ.synchronize()

    def boundary_curve_nearest_points(
        surface: tuple[int, int], target_points: np.ndarray
    ) -> int:
        target = np.asarray(target_points, dtype=np.float64)
        target_box = np.concatenate((np.min(target, axis=0), np.max(target, axis=0)))
        candidates = [
            int(tag)
            for dim, tag in gmsh.model.getBoundary(
                [surface], oriented=False, combined=False
            )
            if int(dim) == 1
        ]
        if not candidates:
            raise RuntimeError("freestanding wall patch has no boundary curves")
        return min(
            candidates,
            key=lambda tag: float(
                np.sum(
                    np.abs(
                        np.asarray(gmsh.model.getBoundingBox(1, tag), dtype=np.float64)
                        - target_box
                    )
                )
            ),
        )

    mouth: list[tuple[int, int]] = []
    for indices, inner_surface, outer_surface in zip(
        phi_groups, inner_wall, outer_wall, strict=True
    ):
        inner_curve = boundary_curve_nearest_points(
            inner_surface, inner_points[np.asarray(indices), -1, :]
        )
        outer_curve = boundary_curve_nearest_points(
            outer_surface, outer_topology[np.asarray(indices), -1, :]
        )
        inner_wire = int(gmsh.model.occ.addWire([inner_curve], checkClosed=False))
        outer_wire = int(gmsh.model.occ.addWire([outer_curve], checkClosed=False))
        mouth.extend(
            (int(dim), int(tag))
            for dim, tag in gmsh.model.occ.addThruSections(
                [inner_wire, outer_wire],
                makeSolid=False,
                makeRuled=True,
            )
            if int(dim) == 2
        )

    rear_cap = make_planar_fill_from_boundary(
        outer_wall,
        source_axis="z",
        use_min=True,
        closed=geometry.closed,
    )
    cap_builder = _SharedSurfaceBuilder()
    cap_builder.add_grid("inner", inner_points)
    throat = _add_occ_source_cap_surfaces(
        cap_builder,
        inner_points,
        geometry,
        boundary_phi_groups=phi_groups,
        wall_dimtags=inner_wall,
    )
    gmsh.model.occ.synchronize()

    wall_tags = [int(tag) for _, tag in inner_wall]
    outer_tags = [int(tag) for _, tag in outer_wall]
    mouth_tags = [int(tag) for _, tag in mouth]
    rear_tags = [int(tag) for _, tag in rear_cap]
    throat_tags = [int(tag) for _, tag in throat]
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
        metadata={
            "meshTopologyMode": "acoustic",
            "outerWallClearance": _wall_clearance_metadata(geometry, outer_points),
        },
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

    wall = _add_spline_span_wall_surfaces(
        builder,
        "inner",
        n_phi=n_phi,
        n_len=inner_len,
        closed=geometry.closed,
    )
    outer_wall = _add_spline_span_wall_surfaces(
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
        metadata={
            "outerWallClearance": _wall_clearance_metadata(geometry, outer_points)
        },
    )
