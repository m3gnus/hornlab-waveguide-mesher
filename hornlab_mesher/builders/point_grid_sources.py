from __future__ import annotations

import math

import numpy as np

from ..geometry import PointGridHornGeometry
from ._occ import extreme_boundary_loop_curves, require_gmsh
from .point_grid_surfaces import (
    _GeoSurfaceBuilder,
    _SharedSurfaceBuilder,
    _phi_segments,
    _source_cap_phi_groups,
    _spline_span_phi_groups,
)

SOURCE_SHAPE_FLAT_DISC = 0
SOURCE_SHAPE_ROUNDED_CAP = 1
SUPPORTED_SOURCE_SHAPES = {SOURCE_SHAPE_FLAT_DISC, SOURCE_SHAPE_ROUNDED_CAP}


def _source_shape(geometry: PointGridHornGeometry) -> int:
    return int(geometry.source_shape)


def _validate_source_shape(geometry: PointGridHornGeometry) -> int:
    shape = _source_shape(geometry)
    if shape not in SUPPORTED_SOURCE_SHAPES:
        raise NotImplementedError(
            f"PointGridHornGeometry.source_shape={shape} is not supported "
            "by hornlab-mesher (supported: 0=flat disc, 1=rounded cap)."
        )
    return shape


def _geo_radial_source_curves(
    builder: _GeoSurfaceBuilder,
    inner_points: np.ndarray,
    *,
    pole_tag: int,
    sphere_center_tag: int,
    cap_height: float,
    indices: list[int] | None = None,
) -> dict[int, int]:
    n_phi = inner_points.shape[0]
    radial_indices = sorted(set(indices)) if indices is not None else range(n_phi)
    if cap_height <= 1.0e-12:
        return {
            i: builder.line_tags(builder.point("inner", i, 0), pole_tag)
            for i in radial_indices
        }

    return {
        i: builder.circle_arc(builder.point("inner", i, 0), sphere_center_tag, pole_tag)
        for i in radial_indices
    }


_CAP_CONSTRAINT_RING_FRACTIONS = (1.0 / 3.0, 2.0 / 3.0)
_CAP_CONSTRAINT_AZIMUTHS = 12


def _occ_rounded_cap_on_wall_rim(
    wall_dimtags: list[tuple[int, int]],
    geometry: PointGridHornGeometry,
    *,
    center: np.ndarray,
    throat_radius: float,
    cap_height: float,
    throat_use_min: bool = True,
    source_axis_sign: float = 1.0,
) -> list[tuple[int, int]]:
    """Rounded source cap filled directly on the wall's throat boundary curves.

    Sharing the wall's own OCC edges is what welds the cap-throat seam
    watertight: both surfaces mesh the same curve once, so the seam cannot
    accumulate two mismatched node rings. A rim authored independently (an
    exact sphere carve, or a re-built BSpline) meshes its own nodes and leaves
    a free-edge crack around the throat. Interior on-sphere constraint points
    pin the filling to the analytic source sphere; the rim itself follows the
    wall edge, which chords the exact throat circle at grid resolution.
    """

    if throat_radius <= 1.0e-9 or cap_height <= 1.0e-12:
        return []

    gmsh = require_gmsh()
    gmsh.model.occ.synchronize()
    curves = extreme_boundary_loop_curves(
        wall_dimtags, source_axis="z", use_min=throat_use_min
    )
    if not curves:
        return []

    radius = max(_source_cap_radius(throat_radius, geometry), throat_radius * 1.001)
    sign = float(source_axis_sign) * (-1.0 if int(geometry.source_curv) == -1 else 1.0)
    sphere_center = np.array(center, dtype=np.float64)
    sphere_center[2] += sign * (cap_height - radius)
    pole = np.array(center, dtype=np.float64)
    pole[2] += sign * cap_height

    constraint_tags = [
        int(gmsh.model.occ.addPoint(float(pole[0]), float(pole[1]), float(pole[2])))
    ]
    rim_angle = math.asin(max(-1.0, min(1.0, throat_radius / radius)))
    for fraction in _CAP_CONSTRAINT_RING_FRACTIONS:
        polar = rim_angle * fraction
        ring_radius = radius * math.sin(polar)
        ring_z = float(sphere_center[2]) + sign * radius * math.cos(polar)
        for step in range(_CAP_CONSTRAINT_AZIMUTHS):
            azimuth = math.tau * step / _CAP_CONSTRAINT_AZIMUTHS
            constraint_tags.append(
                int(
                    gmsh.model.occ.addPoint(
                        float(center[0] + ring_radius * math.cos(azimuth)),
                        float(center[1] + ring_radius * math.sin(azimuth)),
                        ring_z,
                    )
                )
            )

    loop = int(gmsh.model.occ.addCurveLoop([int(tag) for tag in curves]))
    surf = int(gmsh.model.occ.addSurfaceFilling(loop, pointTags=constraint_tags))
    return [(2, surf)]


def _add_geo_source_cap_surfaces(
    builder: _GeoSurfaceBuilder,
    inner_points: np.ndarray,
    geometry: PointGridHornGeometry,
    *,
    mesh_size: float,
) -> list[tuple[int, int]]:
    shape = _validate_source_shape(geometry)
    n_phi = inner_points.shape[0]
    ring = inner_points[:, 0, :]
    center = np.mean(ring, axis=0)
    center[2] = float(np.mean(ring[:, 2]))
    throat_radius = _throat_radius(inner_points, closed=geometry.closed)
    cap_height = _source_cap_height(throat_radius, geometry) if shape == SOURCE_SHAPE_ROUNDED_CAP else 0.0

    if geometry.closed:
        pole = np.array(center, dtype=np.float64)
        sign = -1.0 if int(geometry.source_curv) == -1 else 1.0
        pole[2] += sign * cap_height
        pole_tag = builder.add_point(pole, mesh_size=mesh_size)
        sphere_center_tag = -1
        if cap_height > 1.0e-12:
            radius = max(_source_cap_radius(throat_radius, geometry), throat_radius * 1.001)
            sphere_center = np.array(center, dtype=np.float64)
            sphere_center[2] += sign * (cap_height - radius)
            sphere_center_tag = builder.add_point(sphere_center, mesh_size=mesh_size)
        # The cap rim must reuse the wall's own throat-edge splines (spline
        # cache): a rim curve of its own -- even one tracing the identical
        # throat circle -- meshes its own 1D nodes and leaves a free-edge
        # crack around the throat. The cap still partitions on the coarser
        # source-cap sectors (ATH builds 4 cap surfaces), so each sector's
        # rim is stitched from the wall spans it covers.
        wall_spans = _spline_span_phi_groups(n_phi, closed=True)
        sector_spans = [[span] for span in wall_spans]
        cap_groups = _source_cap_phi_groups(n_phi, closed=True)
        if len(wall_spans) == 2 * len(cap_groups) and all(
            wall_spans[2 * k][0] == group[0] and wall_spans[2 * k + 1][-1] == group[-1]
            for k, group in enumerate(cap_groups)
        ):
            sector_spans = [
                [wall_spans[2 * k], wall_spans[2 * k + 1]]
                for k in range(len(cap_groups))
            ]
        endpoints = [
            index for spans in sector_spans for index in (spans[0][0], spans[-1][-1])
        ]
        radial_curves = _geo_radial_source_curves(
            builder,
            inner_points,
            pole_tag=pole_tag,
            sphere_center_tag=sphere_center_tag,
            cap_height=cap_height,
            indices=endpoints,
        )
        cap: list[tuple[int, int]] = []
        for spans in sector_spans:
            start = spans[0][0]
            end = spans[-1][-1]
            rim = [builder.spline([("inner", i, 0) for i in span]) for span in spans]
            boundary = [*rim, radial_curves[end], -radial_curves[start]]
            if cap_height > 1.0e-12:
                cap.append(
                    builder.surface(boundary, sphere_center_tag=sphere_center_tag)
                )
            else:
                cap.append(builder.surface(boundary))
        return cap

    center[0] = 0.0
    center[1] = 0.0

    # A rounded cap with zero height (flat auto source) shares the flat-disc
    # construction; the arc-based path below cannot represent it.
    if shape == SOURCE_SHAPE_FLAT_DISC or cap_height <= 1.0e-12:
        center_tag = builder.add_point(center, mesh_size=mesh_size)
        radial_lines = {
            i: builder.line_tags(center_tag, builder.point("inner", i, 0))
            for i in range(n_phi)
        }
        boundary = [
            radial_lines[0],
            *[
                builder.spline([("inner", i, 0) for i in indices])
                for indices in _spline_span_phi_groups(n_phi, closed=False)
            ],
            -radial_lines[n_phi - 1],
        ]
        return [builder.surface(boundary)]

    if throat_radius <= 1.0e-9:
        return []
    radius = max(_source_cap_radius(throat_radius, geometry), throat_radius * 1.001)
    cap_height = _source_cap_height(throat_radius, geometry)
    sign = -1.0 if int(geometry.source_curv) == -1 else 1.0

    pole = np.array(center, dtype=np.float64)
    pole[2] += sign * cap_height
    sphere_center = np.array(center, dtype=np.float64)
    sphere_center[2] += sign * (cap_height - radius)

    pole_tag = builder.add_point(pole, mesh_size=mesh_size)
    sphere_center_tag = builder.add_point(sphere_center, mesh_size=mesh_size)
    arc_start = builder.circle_arc(
        pole_tag,
        sphere_center_tag,
        builder.point("inner", 0, 0),
    )
    arc_end = builder.circle_arc(
        pole_tag,
        sphere_center_tag,
        builder.point("inner", n_phi - 1, 0),
    )
    boundary = [
        arc_start,
        *[
            builder.spline([("inner", i, 0) for i in indices])
            for indices in _spline_span_phi_groups(n_phi, closed=False)
        ],
        -arc_end,
    ]
    return [builder.surface(boundary, sphere_center_tag=sphere_center_tag)]


def _add_occ_source_cap_surfaces(
    builder: _SharedSurfaceBuilder,
    inner_points: np.ndarray,
    geometry: PointGridHornGeometry,
    *,
    boundary_phi_groups: list[list[int]] | None = None,
    throat_use_min: bool = True,
    source_axis_sign: float = 1.0,
    wall_dimtags: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    shape = _validate_source_shape(geometry)
    n_phi = inner_points.shape[0]
    ring = inner_points[:, 0, :]
    center = np.mean(ring, axis=0)
    center[2] = float(np.mean(ring[:, 2]))
    if not geometry.closed:
        center[0] = 0.0
        center[1] = 0.0
    throat_radius = _throat_radius(inner_points, closed=geometry.closed)
    cap_height = _source_cap_height(throat_radius, geometry) if shape == SOURCE_SHAPE_ROUNDED_CAP else 0.0
    sign = float(source_axis_sign) * (-1.0 if int(geometry.source_curv) == -1 else 1.0)
    if geometry.closed and shape == SOURCE_SHAPE_ROUNDED_CAP and cap_height > 1.0e-12:
        if wall_dimtags is None:
            raise ValueError(
                "a closed rounded source cap requires wall_dimtags: the cap is "
                "filled on the wall's own throat boundary curves so the seam "
                "welds watertight"
            )
        return _occ_rounded_cap_on_wall_rim(
            wall_dimtags,
            geometry,
            center=center,
            throat_radius=throat_radius,
            cap_height=cap_height,
            throat_use_min=throat_use_min,
            source_axis_sign=source_axis_sign,
        )

    pole = np.array(center, dtype=np.float64)
    pole[2] += sign * cap_height
    pole_tag = builder.add_point(pole)

    radial_lines: dict[int, int] = {}
    radial_indices = range(n_phi) if geometry.closed else (0, n_phi - 1)
    for i in radial_indices:
        if cap_height <= 1.0e-12:
            radial_lines[i] = builder.line_tags(builder.point("inner", i, 0), pole_tag)
            continue
        control_tags = [builder.point("inner", i, 0)]
        radial = ring[i] - center
        radial[2] = 0.0
        radial_len = float(np.linalg.norm(radial[:2]))
        unit = radial / max(radial_len, 1.0e-12)
        radius = max(_source_cap_radius(throat_radius, geometry), throat_radius * 1.001)
        sqrt_base = math.sqrt(max(0.0, radius * radius - throat_radius * throat_radius))
        for frac in (2.0 / 3.0, 1.0 / 3.0):
            rj = throat_radius * frac
            hj = math.sqrt(max(0.0, radius * radius - rj * rj)) - sqrt_base
            p = np.array(center, dtype=np.float64)
            p[:2] += unit[:2] * rj
            p[2] += sign * hj
            control_tags.append(builder.add_point(p))
        control_tags.append(pole_tag)
        radial_lines[i] = builder.bspline_tags(control_tags)

    cap: list[tuple[int, int]] = []
    if geometry.closed:
        # Align the cap's rim spans with the wall patch boundaries when the
        # caller provides them; a diverging span split re-authors the throat
        # curve and the two coincident rims mesh mismatched node rings.
        spans = (
            boundary_phi_groups
            if boundary_phi_groups is not None
            else _source_cap_phi_groups(n_phi, closed=True)
        )
        for indices in spans:
            start = indices[0]
            end = indices[-1]
            phi_curve = builder.bspline_tags([builder.point("inner", i, 0) for i in indices])
            if start == end:
                # Single wrapped span: the rim is one closed curve.
                cap.append(builder.surface([phi_curve]))
                continue
            cap.append(builder.surface([phi_curve, radial_lines[end], -radial_lines[start]]))
        return cap

    # Open sector cap: the cap's throat boundary must reuse the same phi spans
    # as the inner wall so the two surfaces produce an identical throat curve
    # and weld into a watertight seam. When the enclosure builds a BSpline-patch
    # wall it passes that wall's exact ``boundary_phi_groups`` here; the cap's
    # own ``_source_cap_phi_groups`` split would diverge from the wall and leave
    # an off-plane open-edge ring at the throat that the native reduced-domain
    # solve rejects. Falling back to the cap's own spans keeps every other path
    # (preserved faceted wall, non-enclosure caps) byte-for-byte unchanged.
    spans = (
        boundary_phi_groups
        if boundary_phi_groups is not None
        else _source_cap_phi_groups(n_phi, closed=False)
    )
    boundary: list[int] = []
    for indices in spans:
        boundary.append(builder.bspline_tags([builder.point("inner", i, 0) for i in indices]))
    if boundary:
        cap.append(builder.surface([*boundary, radial_lines[n_phi - 1], -radial_lines[0]]))
    return cap


def _source_cap_radius(
    throat_radius: float,
    geometry: PointGridHornGeometry,
) -> float:
    if geometry.source_radius_mm and geometry.source_radius_mm > 0.0:
        return float(geometry.source_radius_mm)
    if geometry.source_auto_angle_deg is not None:
        angle = math.radians(float(geometry.source_auto_angle_deg))
        if abs(math.sin(angle)) > 1.0e-9:
            return abs(float(throat_radius) / math.sin(angle))
        # ATH matches the auto cap to the throat opening angle; a zero
        # angle means an infinite wavefront radius, i.e. a flat source.
        return math.inf
    return 3.75 * float(throat_radius)


def _source_cap_height(throat_radius: float, geometry: PointGridHornGeometry) -> float:
    if _source_shape(geometry) != SOURCE_SHAPE_ROUNDED_CAP or throat_radius <= 1.0e-9:
        return 0.0
    radius = _source_cap_radius(throat_radius, geometry)
    if not math.isfinite(radius):
        return 0.0
    radius = max(radius, throat_radius * 1.001)
    return radius - math.sqrt(max(0.0, radius * radius - throat_radius * throat_radius))


def _throat_radius(inner_points: np.ndarray, *, closed: bool) -> float:
    ring = inner_points[:, 0, :]
    center = np.mean(ring, axis=0)
    if not closed:
        center[0] = 0.0
        center[1] = 0.0
    radial = ring[:, :2] - center[:2]
    radii = np.linalg.norm(radial, axis=1)
    return float(np.mean(radii[radii > 1.0e-9])) if np.any(radii > 1.0e-9) else 0.0


def _add_source_surfaces(
    builder: _SharedSurfaceBuilder,
    inner_points: np.ndarray,
    geometry: PointGridHornGeometry,
    *,
    wall_dimtags: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    shape = _validate_source_shape(geometry)
    n_phi = inner_points.shape[0]
    closed = bool(geometry.closed)
    ring = inner_points[:, 0, :]
    center = np.mean(ring, axis=0)
    if not closed:
        center[0] = 0.0
        center[1] = 0.0
    center[2] = float(np.mean(ring[:, 2]))

    radial = ring - center
    radial[:, 2] = 0.0
    radii = np.linalg.norm(radial[:, :2], axis=1)
    throat_radius = float(np.mean(radii[radii > 1.0e-9])) if np.any(radii > 1.0e-9) else 0.0
    if throat_radius <= 1.0e-9:
        return []

    cap_height = 0.0
    if shape == SOURCE_SHAPE_ROUNDED_CAP:
        cap_height = _source_cap_height(throat_radius, geometry)
    sign = -1.0 if int(geometry.source_curv) == -1 else 1.0
    if closed and shape == SOURCE_SHAPE_ROUNDED_CAP and cap_height > 1.0e-12:
        if wall_dimtags is None:
            raise ValueError(
                "a closed rounded source cap requires wall_dimtags: the cap is "
                "filled on the wall's own throat boundary curves so the seam "
                "welds watertight"
            )
        return _occ_rounded_cap_on_wall_rim(
            wall_dimtags,
            geometry,
            center=center,
            throat_radius=throat_radius,
            cap_height=cap_height,
        )

    pole = np.array(center, dtype=np.float64)
    pole[2] += sign * cap_height
    pole_tag = builder.add_point(pole)

    radial_lines: dict[int, int] = {}
    radial_indices = range(n_phi) if geometry.closed else (0, n_phi - 1)
    for i in radial_indices:
        if cap_height <= 1.0e-12:
            radial_lines[i] = builder.line_tags(builder.point("inner", i, 0), pole_tag)
            continue
        control_tags = [builder.point("inner", i, 0)]
        unit = radial[i] / max(radii[i], 1.0e-12)
        radius = max(_source_cap_radius(throat_radius, geometry), throat_radius * 1.001)
        sqrt_base = math.sqrt(max(0.0, radius * radius - throat_radius * throat_radius))
        for frac in (2.0 / 3.0, 1.0 / 3.0):
            rj = throat_radius * frac
            hj = math.sqrt(max(0.0, radius * radius - rj * rj)) - sqrt_base
            p = np.array(center, dtype=np.float64)
            p[:2] += unit[:2] * rj
            p[2] += sign * hj
            control_tags.append(builder.add_point(p))
        control_tags.append(pole_tag)
        radial_lines[i] = builder.bspline_tags(control_tags)

    surfaces: list[tuple[int, int]] = []
    cap_boundary: list[int] = []
    for i in _phi_segments(n_phi, closed=closed):
        ni = (i + 1) % n_phi
        cap_boundary.append(builder.line(("inner", i, 0), ("inner", ni, 0)))
        if closed:
            surfaces.append(
                builder.surface(
                    [
                        cap_boundary[-1],
                        radial_lines[ni],
                        -radial_lines[i],
                    ]
                )
            )
    if not closed and cap_boundary:
        surfaces.append(builder.surface([*cap_boundary, radial_lines[n_phi - 1], -radial_lines[0]]))
    return surfaces
