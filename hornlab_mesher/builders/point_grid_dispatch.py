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
    _add_source_surfaces,
    _source_cap_height,
    _throat_radius,
    _validate_source_shape,
)
from .point_grid_surfaces import (
    _SharedSurfaceBuilder,
    _add_grid_wall_surfaces,
    _add_occ_bspline_patch_wall_surfaces,
    _add_occ_spline_span_wall_surfaces,
    _bspline_patch_phi_groups,
    _phi_segments,
    _snap_open_symmetry_grid,
    _spline_span_phi_groups,
    _validated_grid,
)


_BAFFLE_PLANE_SNAP_TOL_MM = 1.0e-6
_BAFFLE_PROTRUSION_TOL_MM = 1.0e-7


def _open_sector_count(geometry: PointGridHornGeometry) -> int:
    """Quadrant sectors spanned by an open reduced grid.

    A quarter model is bounded by two cut planes (``symmetry_planes`` has two
    entries) and is a single quadrant; a half model is bounded by one cut plane
    and spans two quadrants that meet on the off-cut axis. Closed grids are a
    single full surface (the caller does not split them).
    """

    if geometry.closed:
        return 1
    return 1 if len(tuple(geometry.symmetry_planes)) >= 2 else 2


def _shift_coupled_baffle_grid(points: np.ndarray) -> np.ndarray:
    out = np.array(points, dtype=np.float64, copy=True)
    mouth_z = float(np.mean(out[:, -1, 2]))
    out[:, :, 2] -= mouth_z
    out[:, -1, 2] = 0.0

    max_z = float(np.max(out[:, :, 2]))
    if max_z > _BAFFLE_PROTRUSION_TOL_MM:
        raise ValueError(
            "infinite-baffle coupled aperture mesh requires the mouth ring to "
            "be the front-most z station so the interior cavity lies in z <= 0; "
            f"the translated point grid protrudes through the baffle plane to z={max_z:.6g} mm"
        )
    return out


def _add_mouth_aperture_surfaces(
    builder: _SharedSurfaceBuilder,
    *,
    n_phi: int,
    n_len: int,
    closed: bool,
    preserve_grid: bool,
) -> list[tuple[int, int]]:
    """Fill the mouth aperture on the wall's own rim curves.

    The Rayleigh aperture must share the exact mouth rim authored for the wall:
    a geometrically coincident re-built profile curve gets its own 1D mesh nodes
    and leaves a crack at the coupling boundary.
    """

    mouth_j = n_len - 1
    rim_curves: list[int] = []
    if preserve_grid:
        for i in _phi_segments(n_phi, closed=closed):
            ni = (i + 1) % n_phi
            rim_curves.append(builder.line(("inner", i, mouth_j), ("inner", ni, mouth_j)))
    else:
        rim_curves = [
            builder.bspline_tags([builder.point("inner", i, mouth_j) for i in indices])
            for indices in _spline_span_phi_groups(n_phi, closed=closed)
        ]
    if not rim_curves:
        return []

    if closed:
        return [builder.surface(rim_curves)]

    center_tag = builder.add_point((0.0, 0.0, 0.0))
    end_to_center = builder.line_tags(builder.point("inner", n_phi - 1, mouth_j), center_tag)
    center_to_start = builder.line_tags(center_tag, builder.point("inner", 0, mouth_j))
    aperture: list[tuple[int, int]] = []
    if rim_curves:
        aperture.append(builder.surface([*rim_curves, end_to_center, center_to_start]))
    return aperture


def _build_coupled_baffle_point_grid(
    geometry: PointGridHornGeometry,
    inner_points: np.ndarray,
    source_shape: int,
) -> BuiltGeometry:
    inner_points = _shift_coupled_baffle_grid(inner_points)
    inner_points = _snap_open_symmetry_grid(
        inner_points,
        closed=geometry.closed,
        symmetry_planes=geometry.symmetry_planes,
    )

    n_phi, n_len, _ = inner_points.shape
    builder = _SharedSurfaceBuilder()
    builder.add_grid("inner", inner_points)
    if geometry.preserve_grid:
        wall = _add_grid_wall_surfaces(
            builder,
            "inner",
            n_phi=n_phi,
            n_len=n_len,
            closed=geometry.closed,
        )
        cap_boundary_groups = None
    else:
        cap_boundary_groups = _spline_span_phi_groups(n_phi, closed=geometry.closed)
        wall = _add_occ_spline_span_wall_surfaces(
            builder,
            "inner",
            n_phi=n_phi,
            n_len=n_len,
            closed=geometry.closed,
        )

    aperture = _add_mouth_aperture_surfaces(
        builder,
        n_phi=n_phi,
        n_len=n_len,
        closed=geometry.closed,
        preserve_grid=geometry.preserve_grid,
    )
    throat_radius = _throat_radius(inner_points, closed=geometry.closed)
    cap_height = (
        _source_cap_height(throat_radius, geometry)
        if source_shape == SOURCE_SHAPE_ROUNDED_CAP
        else 0.0
    )
    if geometry.preserve_grid:
        throat = _add_source_surfaces(
            builder,
            inner_points,
            geometry,
            wall_dimtags=wall,
        )
    elif source_shape == SOURCE_SHAPE_ROUNDED_CAP and cap_height > 1.0e-12:
        throat = _add_occ_source_cap_surfaces(
            builder,
            inner_points,
            geometry,
            boundary_phi_groups=cap_boundary_groups,
            throat_use_min=True,
            source_axis_sign=1.0,
            wall_dimtags=wall,
        )
    elif source_shape in {SOURCE_SHAPE_FLAT_DISC, SOURCE_SHAPE_ROUNDED_CAP}:
        throat = _add_occ_source_cap_surfaces(
            builder,
            inner_points,
            geometry,
            boundary_phi_groups=cap_boundary_groups,
            throat_use_min=True,
            source_axis_sign=1.0,
            wall_dimtags=wall,
        )
    else:
        raise AssertionError(f"unhandled source shape {source_shape!r}")

    require_gmsh().model.occ.synchronize()
    wall_tags = [tag for _, tag in wall]
    throat_tags = [tag for _, tag in throat]
    aperture_tags = [tag for _, tag in aperture]
    throat_z = float(np.mean(inner_points[:, 0, 2]))
    mouth_z = float(np.mean(inner_points[:, -1, 2]))
    return BuiltGeometry(
        surface_groups={
            int(PhysicalGroup.RIGID_WALL): wall_tags,
            int(PhysicalGroup.PRIMARY_SOURCE): throat_tags,
            int(PhysicalGroup.MOUTH_APERTURE): aperture_tags,
        },
        axial_bounds_mm=(throat_z, mouth_z),
        source_axis="z",
        mesh_surface_groups={
            "inner": list(wall_tags),
            "throat_disc": list(throat_tags),
            "mouth_aperture": list(aperture_tags),
        },
        symmetry_snap_axes=() if geometry.closed else tuple(geometry.symmetry_planes),
        symmetry_snap_tol_mm=_BAFFLE_PLANE_SNAP_TOL_MM,
        metadata={
            "apertureTag": int(PhysicalGroup.MOUTH_APERTURE),
            "apertureZMm": 0.0,
            "cavityDepthMm": float(abs(throat_z - mouth_z)),
        },
    )


def build_point_grid(geometry: PointGridHornGeometry) -> BuiltGeometry:
    inner_points = _validated_grid(geometry.inner_points, name="inner_points")
    source_shape = _validate_source_shape(geometry)
    build_mode = geometry.build_mode

    if build_mode is PointGridBuildMode.FREESTANDING:
        return _build_freestanding_point_grid(geometry)

    if build_mode is PointGridBuildMode.INFINITE_BAFFLE:
        return _build_coupled_baffle_point_grid(geometry, inner_points, source_shape)

    if build_mode is PointGridBuildMode.ENCLOSURE:
        inner_points = _snap_open_symmetry_grid(
            inner_points, closed=geometry.closed, symmetry_planes=geometry.symmetry_planes
        )
        cap_builder = _SharedSurfaceBuilder()
        cap_builder.add_grid("inner", inner_points)
        cap_boundary_groups: list[list[int]] | None = None
        if geometry.preserve_grid:
            wall = _add_grid_wall_surfaces(
                cap_builder,
                "inner",
                n_phi=inner_points.shape[0],
                n_len=inner_points.shape[1],
                closed=geometry.closed,
            )
            # Faceted walls need a cap whose boundary reuses the same straight
            # chords through the shared point cache; B-spline cap spans through
            # the same points coincide with the chords only at the grid nodes
            # and leave an off-plane open seam ring at the throat.
            throat = _add_source_surfaces(
                cap_builder, inner_points, geometry, wall_dimtags=wall
            )
        else:
            # A reduced half-model grid must split into one wall patch per
            # quadrant so its rear enclosure can attach a sector to each mouth
            # curve; the throat cap reuses the same partition to stay watertight.
            # Closed grids reuse the partition too: a diverging cap span split
            # re-authors the throat curve and cracks the seam.
            wall_groups = _bspline_patch_phi_groups(
                inner_points.shape[0],
                closed=geometry.closed,
                n_sectors=_open_sector_count(geometry),
            )
            wall = _add_occ_bspline_patch_wall_surfaces(
                inner_points,
                closed=geometry.closed,
                phi_groups=wall_groups,
            )
            cap_boundary_groups = wall_groups
            throat = _add_occ_source_cap_surfaces(
                cap_builder,
                inner_points,
                geometry,
                boundary_phi_groups=cap_boundary_groups,
                wall_dimtags=wall,
            )
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
            # Closed grids fill the cap on the wall's own throat edge. Open
            # grids re-author the rim, which must span the wall patch's full
            # angular row so both rims mesh identical 1D nodes and weld.
            throat = _add_occ_source_cap_surfaces(
                cap_builder,
                inner_points,
                geometry,
                boundary_phi_groups=(
                    None if geometry.closed else [list(range(inner_points.shape[0]))]
                ),
                wall_dimtags=wall,
            )
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
            symmetry_planes=geometry.symmetry_planes,
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
        symmetry_snap_axes=() if geometry.closed else tuple(geometry.symmetry_planes),
    )
