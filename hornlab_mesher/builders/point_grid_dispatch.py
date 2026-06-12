from __future__ import annotations

"""Dispatcher for point-grid topology build modes."""

import numpy as np

from ..geometry import BuiltGeometry, PointGridBuildMode, PointGridHornGeometry
from ..tags import PhysicalGroup
from ._occ import (
    build_surface_from_points,
    make_planar_fill_from_boundary,
    make_planar_fill_from_ring,
    make_planar_sector_fill_from_ring,
    require_gmsh,
)
from .enclosure import build_enclosure_box
from .point_grid_freestanding import _build_freestanding_point_grid
from .point_grid_interfaces import _add_offset_interface_surfaces, _normalise_interface_specs
from .point_grid_sources import (
    SOURCE_SHAPE_FLAT_DISC,
    SOURCE_SHAPE_ROUNDED_CAP,
    _add_occ_source_cap_surfaces,
    _validate_source_shape,
)
from .point_grid_surfaces import (
    _SharedSurfaceBuilder,
    _add_occ_bspline_patch_wall_surfaces,
    _snap_open_symmetry_grid,
    _validated_grid,
)

def build_point_grid(geometry: PointGridHornGeometry) -> BuiltGeometry:
    inner_points = _validated_grid(geometry.inner_points, name="inner_points")
    source_shape = _validate_source_shape(geometry)
    build_mode = geometry.build_mode

    if build_mode is PointGridBuildMode.FREESTANDING:
        return _build_freestanding_point_grid(geometry)

    if build_mode is PointGridBuildMode.ENCLOSURE:
        inner_points = _snap_open_symmetry_grid(inner_points, closed=geometry.closed)
        wall = _add_occ_bspline_patch_wall_surfaces(
            inner_points,
            closed=geometry.closed,
        )
        cap_builder = _SharedSurfaceBuilder()
        cap_builder.add_grid("inner", inner_points)
        throat = _add_occ_source_cap_surfaces(cap_builder, inner_points, geometry)
        require_gmsh().model.occ.synchronize()
        if not throat:
            throat = make_planar_fill_from_ring(inner_points[:, 0, :])
    else:
        wall = build_surface_from_points(
            inner_points,
            closed=geometry.closed,
            preserve_grid=geometry.preserve_grid,
        )
        require_gmsh().model.occ.synchronize()

        if source_shape == SOURCE_SHAPE_ROUNDED_CAP:
            cap_builder = _SharedSurfaceBuilder()
            cap_builder.add_grid("inner", inner_points)
            throat = _add_occ_source_cap_surfaces(cap_builder, inner_points, geometry)
        elif source_shape == SOURCE_SHAPE_FLAT_DISC and geometry.closed:
            throat = make_planar_fill_from_boundary(
                wall,
                source_axis="z",
                use_min=True,
                closed=True,
            )
            if not throat:
                throat = make_planar_fill_from_ring(inner_points[:, 0, :])
        elif source_shape == SOURCE_SHAPE_FLAT_DISC:
            throat = make_planar_sector_fill_from_ring(
                inner_points[:, 0, :],
                source_axis="z",
            )
            if not throat:
                throat = make_planar_fill_from_boundary(
                    wall,
                    source_axis="z",
                    use_min=True,
                    closed=False,
                )
        else:
            raise AssertionError(f"unhandled source shape {source_shape!r}")

    wall_tags = [tag for _, tag in wall]
    throat_tags = [tag for _, tag in throat]

    mesh_surface_groups: dict[str, list[int]] = {
        "inner": list(wall_tags),
        "throat_disc": list(throat_tags),
    }
    rigid_wall_tags: list[int] = list(wall_tags)
    enclosure_tags: list[int] = []
    interface_tags: list[int] = []
    enclosure_bounds: dict[str, float] | None = None

    if build_mode is PointGridBuildMode.INFINITE_BAFFLE:
        # ABEC infinite-baffle topology: the mouth aperture is closed by a
        # planar subdomain interface (I1-2) lying in the baffle plane.
        if geometry.closed:
            mouth_fill = make_planar_fill_from_ring(inner_points[:, -1, :])
        else:
            mouth_fill = make_planar_sector_fill_from_ring(
                inner_points[:, -1, :],
                source_axis="z",
            )
        if not mouth_fill:
            mouth_fill = make_planar_fill_from_boundary(
                wall,
                source_axis="z",
                use_min=False,
                closed=geometry.closed,
            )
        if not mouth_fill:
            raise RuntimeError("failed to build the infinite-baffle mouth interface surface")
        require_gmsh().model.occ.synchronize()
        interface_tags = [tag for _, tag in mouth_fill]
        mesh_surface_groups["interface"] = list(interface_tags)

    if geometry.enclosure is not None:
        interface_dimtags: list[tuple[int, int]] = []
        for interface in _normalise_interface_specs(geometry, inner_points.shape[1]):
            interface_dimtags.extend(
                _add_offset_interface_surfaces(
                    inner_points,
                    slice_index=int(interface.slice_index),
                    closed=geometry.closed,
                    offset_mm=float(interface.offset_mm),
                )
            )
        interface_tags = [tag for _, tag in interface_dimtags]
        if interface_tags:
            mesh_surface_groups["interface"] = list(interface_tags)
        enc_data = build_enclosure_box(
            inner_dimtags=wall,
            inner_points=inner_points,
            enclosure=geometry.enclosure,
            closed=geometry.closed,
        )
        enclosure_tags = [tag for _, tag in enc_data["dimtags"]]
        # All enclosure surfaces join the "enclosure" group so density.py's
        # z-interpolated front/back side-wall formula applies. The roundover
        # surfaces additionally appear in the front/back edge groups so the
        # panel-bilinear formula clamps them via the Min field.
        mesh_surface_groups["enclosure"] = list(enclosure_tags)
        mesh_surface_groups["enclosure_edges_front"] = list(enc_data["front_edges"])
        mesh_surface_groups["enclosure_edges_back"] = list(enc_data["back_edges"])
        enclosure_bounds = dict(enc_data["bounds"])

    z0 = float(np.mean(inner_points[:, 0, 2]))
    z1 = float(np.mean(inner_points[:, -1, 2]))
    surface_groups = {
        int(PhysicalGroup.RIGID_WALL): rigid_wall_tags,
        int(PhysicalGroup.PRIMARY_SOURCE): throat_tags,
    }
    if enclosure_tags:
        surface_groups[int(PhysicalGroup.ENCLOSURE_WALL)] = enclosure_tags
    if interface_tags:
        surface_groups[int(PhysicalGroup.INTERFACE)] = interface_tags
    return BuiltGeometry(
        surface_groups=surface_groups,
        axial_bounds_mm=(z0, z1),
        source_axis="z",
        mesh_surface_groups=mesh_surface_groups,
        enclosure_bounds=enclosure_bounds,
    )
