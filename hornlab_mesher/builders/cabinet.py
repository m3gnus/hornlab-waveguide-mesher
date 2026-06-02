from __future__ import annotations

import math

from ..geometry import BuiltGeometry, CabinetGeometry, DriverConfig, SlotConfig
from ..tags import PhysicalGroup
from ._occ import require_gmsh, validate_source_tag


def _point(x: float, y: float, z: float) -> int:
    gmsh = require_gmsh()
    return int(gmsh.model.occ.addPoint(float(x), float(y), float(z)))


def _line(a: int, b: int) -> int:
    gmsh = require_gmsh()
    return int(gmsh.model.occ.addLine(int(a), int(b)))


def _loop(lines: list[int], *, reverse: bool = False) -> int:
    gmsh = require_gmsh()
    oriented = [-tag for tag in reversed(lines)] if reverse else lines
    try:
        return int(gmsh.model.occ.addCurveLoop(oriented, reorient=True))
    except TypeError:
        return int(gmsh.model.occ.addCurveLoop(oriented))


def _rect_loop_y(y: float, xmin: float, xmax: float, zmin: float, zmax: float, *, reverse: bool = False) -> int:
    p1 = _point(xmin, y, zmin)
    p2 = _point(xmax, y, zmin)
    p3 = _point(xmax, y, zmax)
    p4 = _point(xmin, y, zmax)
    return _loop([_line(p1, p2), _line(p2, p3), _line(p3, p4), _line(p4, p1)], reverse=reverse)


def _rect_loop_x(x: float, ymin: float, ymax: float, zmin: float, zmax: float, *, reverse: bool = False) -> int:
    p1 = _point(x, ymin, zmin)
    p2 = _point(x, ymax, zmin)
    p3 = _point(x, ymax, zmax)
    p4 = _point(x, ymin, zmax)
    return _loop([_line(p1, p2), _line(p2, p3), _line(p3, p4), _line(p4, p1)], reverse=reverse)


def _rect_loop_z(z: float, xmin: float, xmax: float, ymin: float, ymax: float, *, reverse: bool = False) -> int:
    p1 = _point(xmin, ymin, z)
    p2 = _point(xmax, ymin, z)
    p3 = _point(xmax, ymax, z)
    p4 = _point(xmin, ymax, z)
    return _loop([_line(p1, p2), _line(p2, p3), _line(p3, p4), _line(p4, p1)], reverse=reverse)


def _surface(loop: int, holes: list[int] | None = None) -> int:
    gmsh = require_gmsh()
    return int(gmsh.model.occ.addPlaneSurface([int(loop), *(holes or [])]))


def _circle_loop_y(x: float, y: float, z: float, radius: float) -> int:
    gmsh = require_gmsh()
    curve = gmsh.model.occ.addCircle(
        float(x),
        float(y),
        float(z),
        float(radius),
        zAxis=[0.0, 1.0, 0.0],
    )
    return _loop([int(curve)])


def _slot_loops(geometry: CabinetGeometry) -> list[int]:
    if geometry.aperture_width_mm > 0.0 and geometry.aperture_height_mm > 0.0:
        half_w = geometry.aperture_width_mm / 2.0
        half_h = geometry.aperture_height_mm / 2.0
        return [_rect_loop_y(0.0, -half_w, half_w, -half_h, half_h)]

    loops: list[int] = []
    half_w = geometry.width_mm / 2.0
    for slot in geometry.slots:
        if slot.topology == "side":
            continue
        loops.append(_front_slot_loop(slot, half_w))
    return loops


def _front_slot_loop(slot: SlotConfig, half_cabinet_width: float) -> int:
    width = float(slot.opening_w_mm)
    height = float(slot.height_mm)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("slot opening width and height must be > 0")
    side = -1.0 if slot.exit_x_offset_mm < 0.0 else 1.0
    outer = side * max(0.0, half_cabinet_width - abs(slot.exit_x_offset_mm))
    inner = outer - side * width
    xmin = min(inner, outer)
    xmax = max(inner, outer)
    zmin = -height / 2.0
    zmax = height / 2.0
    return _rect_loop_y(0.0, xmin, xmax, zmin, zmax)


def _driver_centers(drivers: list[DriverConfig], width: float, depth: float) -> list[tuple[DriverConfig, float, float, float]]:
    if not drivers:
        return []
    spacing = min(width * 0.25, max(d.diameter_mm for d in drivers))
    start = -0.5 * spacing * (len(drivers) - 1)
    out: list[tuple[DriverConfig, float, float, float]] = []
    for idx, driver in enumerate(drivers):
        validate_source_tag(driver.tag)
        if driver.position not in ("rear_panel", "slot_wall"):
            raise ValueError(f"unsupported driver position: {driver.position}")
        x = start + idx * spacing + float(driver.offset_mm)
        out.append((driver, x, depth, 0.0))
    return out


def build_cabinet(geometry: CabinetGeometry) -> BuiltGeometry:
    if geometry.width_mm <= 0.0 or geometry.depth_mm <= 0.0 or geometry.height_mm <= 0.0:
        raise ValueError("cabinet width, depth and height must be > 0")

    w2 = float(geometry.width_mm) / 2.0
    h2 = float(geometry.height_mm) / 2.0
    d = float(geometry.depth_mm)

    front_holes = _slot_loops(geometry)
    source_groups: dict[int, list[int]] = {}
    aperture_surfaces = [_surface(loop) for loop in front_holes]
    if aperture_surfaces:
        source_groups[int(PhysicalGroup.APERTURE)] = aperture_surfaces

    driver_holes: list[int] = []
    driver_surfaces_by_tag: dict[int, list[int]] = {}
    for driver, x, y, z in _driver_centers(geometry.drivers, geometry.width_mm, d):
        radius = float(driver.diameter_mm) / 2.0
        if radius <= 0.0:
            raise ValueError("driver diameter must be > 0")
        if abs(x) + radius >= w2 or radius >= h2:
            raise ValueError("driver disk does not fit on the rear panel")
        loop = _circle_loop_y(x, y, z, radius)
        driver_holes.append(loop)
        driver_surfaces_by_tag.setdefault(int(driver.tag), []).append(_surface(loop))

    wall_surfaces = [
        _surface(_rect_loop_y(0.0, -w2, w2, -h2, h2), holes=front_holes),
        _surface(_rect_loop_y(d, -w2, w2, -h2, h2, reverse=True), holes=driver_holes),
        _surface(_rect_loop_x(-w2, 0.0, d, -h2, h2, reverse=True)),
        _surface(_rect_loop_x(w2, 0.0, d, -h2, h2)),
        _surface(_rect_loop_z(-h2, -w2, w2, 0.0, d, reverse=True)),
        _surface(_rect_loop_z(h2, -w2, w2, 0.0, d)),
    ]

    groups = {int(PhysicalGroup.RIGID_WALL): wall_surfaces, **source_groups}
    for tag, surfaces in driver_surfaces_by_tag.items():
        groups.setdefault(tag, []).extend(surfaces)

    return BuiltGeometry(
        surface_groups=groups,
        axial_bounds_mm=(0.0, d),
        source_axis="y",
        mesh_surface_groups={
            "rear": wall_surfaces,
            "throat_disc": [
                tag
                for surfaces in driver_surfaces_by_tag.values()
                for tag in surfaces
            ],
            "aperture": aperture_surfaces,
        },
    )
