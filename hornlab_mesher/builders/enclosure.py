"""Rear enclosure builder for point-grid waveguides.

Supports closed-domain waveguides (``quadrants=1234``) with:

* ``HornEnclosure.plan_type`` ``∈ {1, 2, 3}`` — rounded rectangle, ellipse,
  superellipse (exponent from ``plan_n``).
* ``HornEnclosure.edge_type`` ``∈ {1, 2}`` — rounded fillet (3 profile
  thru-section) or chamfer (single ruled surface).

Open-domain rounded-rectangle sectors share the same sector builder used by
the closed-domain four-sector path.

The pure XY-plane geometry math (rounded-rect, ellipse, superellipse
samplers) is vendored from ``Waveguide-Generator/server/solver/waveguide_enclosure.py``
without modification.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..geometry import HornEnclosure
from ._occ import require_gmsh

logger = logging.getLogger(__name__)

# Minimum flat-baffle ring (mm) to keep between the outer enclosure wall and the
# edge roundover. When ``enc_edge`` approaches the smallest enclosure margin the
# ring collapses to a sub-millimetre sliver that OCC silently drops, tearing the
# front-baffle-to-side-wall seam open — an off-symmetry-plane free edge that
# fails the solver's open-edge guard. The roundover is clamped so at least this
# much (or a small fraction of the margin, whichever is larger) survives.
_MIN_BAFFLE_CLEARANCE_MM = 0.1
_BAFFLE_CLEARANCE_FRACTION = 0.05

# Below this roundover radius (mm) a rounded-rectangle sector is built as a true
# sharp box instead. ``enc_edge=0`` collapses the front/back roundover inset
# points onto the outer points, so the roundover/corner "arcs" degenerate to
# zero-length lines OCC rejects ("Could not create line"). The threshold is a
# nanometre — far below OCC's point-merge tolerance, so no real roundover is
# ever misrouted, while every degenerate one takes the sharp path.
_SHARP_SECTOR_EDGE_EPS = 1.0e-6


def _clamp_edge_roundover(
    edge_mm: float,
    margin_edge_limit: float,
    half_w: float,
    half_h: float,
) -> float:
    """Clamp the enclosure edge roundover so the box stays meshable.

    The roundover may not exceed the smallest enclosure margin, the half-width,
    or the half-height. Crucially, when ``edge_mm`` approaches the smallest
    margin (e.g. ``enc_edge == enc_space``) the flat-baffle ring between the
    outer wall and the roundover collapses to a sub-millimetre sliver that OCC
    silently drops — tearing the front-baffle-to-side-wall seam open off the
    symmetry plane. Keep a real flat-baffle clearance below the margin, not just
    an exact-tangency epsilon. Returns ``0.0`` (sharp corner) when no positive
    roundover survives the clamp.
    """
    clamped = max(
        0.0,
        min(float(edge_mm), float(margin_edge_limit), float(half_w) - 0.1, float(half_h) - 0.1),
    )
    if margin_edge_limit > 0.0:
        baffle_clearance = max(
            _MIN_BAFFLE_CLEARANCE_MM,
            margin_edge_limit * _BAFFLE_CLEARANCE_FRACTION,
        )
        clamped = min(clamped, max(0.0, margin_edge_limit - baffle_clearance))
    return clamped


# ---------------------------------------------------------------------------
# Pure XY-plane geometry math (no gmsh) — vendored from WG.
# ---------------------------------------------------------------------------

def sample_rounded_rect(
    *,
    bx0: float,
    bx1: float,
    by0: float,
    by1: float,
    corner_radius: float,
    edge_type: int,
    z: float,
    n_per_edge: int = 3,
    n_per_corner: int = 4,
) -> NDArray[np.float64]:
    """Sample CCW points on a rounded rectangle at axial ``z``."""

    half_w = 0.5 * (bx1 - bx0)
    half_h = 0.5 * (by1 - by0)
    r = max(0.0, min(float(corner_radius), half_w - 0.1, half_h - 0.1))
    has_corners = r > 1e-3

    pts: list[tuple[float, float]] = []

    def add_edge(x0: float, y0: float, x1: float, y1: float) -> None:
        for i in range(n_per_edge + 1):
            t = i / (n_per_edge + 1)
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))

    def add_corner(acx: float, acy: float, start_a: float, end_a: float) -> None:
        for i in range(n_per_corner):
            t = i / n_per_corner
            a = start_a + t * (end_a - start_a)
            if edge_type == 2:
                sx = acx + r * math.cos(start_a)
                sy = acy + r * math.sin(start_a)
                ex = acx + r * math.cos(end_a)
                ey = acy + r * math.sin(end_a)
                pts.append((sx + t * (ex - sx), sy + t * (ey - sy)))
            else:
                pts.append((acx + r * math.cos(a), acy + r * math.sin(a)))

    if has_corners:
        add_corner(bx1 - r, by0 + r, -math.pi / 2.0, 0.0)
        add_edge(bx1, by0 + r, bx1, by1 - r)
        add_corner(bx1 - r, by1 - r, 0.0, math.pi / 2.0)
        add_edge(bx1 - r, by1, bx0 + r, by1)
        add_corner(bx0 + r, by1 - r, math.pi / 2.0, math.pi)
        add_edge(bx0, by1 - r, bx0, by0 + r)
        add_corner(bx0 + r, by0 + r, math.pi, 1.5 * math.pi)
        add_edge(bx0 + r, by0, bx1 - r, by0)
    else:
        corners = [(bx1, by0), (bx1, by1), (bx0, by1), (bx0, by0)]
        for i in range(4):
            x0, y0 = corners[i]
            x1, y1 = corners[(i + 1) % 4]
            add_edge(x0, y0, x1, y1)

    out = np.empty((len(pts), 3), dtype=np.float64)
    for i, (x, y) in enumerate(pts):
        out[i, 0] = x
        out[i, 1] = y
        out[i, 2] = z
    return out


def sample_superellipse(
    *,
    bx0: float,
    bx1: float,
    by0: float,
    by1: float,
    n: float,
    z: float,
    n_points: int = 28,
) -> NDArray[np.float64]:
    """Sample CCW points on a superellipse ``|x/a|^n + |y/b|^n = 1`` at axial ``z``."""

    ecx = 0.5 * (bx1 + bx0)
    ecy = 0.5 * (by1 + by0)
    a = 0.5 * (bx1 - bx0)
    b = 0.5 * (by1 - by0)

    exp = 2.0 / float(n)

    def sgn_pow(val: float, e: float) -> float:
        av = abs(val)
        if av < 1e-15:
            return 0.0
        return math.copysign(av ** e, val)

    pts: list[tuple[float, float]] = []
    for i in range(n_points):
        theta = 2.0 * math.pi * i / n_points
        x = ecx + a * sgn_pow(math.cos(theta), exp)
        y = ecy + b * sgn_pow(math.sin(theta), exp)
        pts.append((x, y))

    out = np.empty((len(pts), 3), dtype=np.float64)
    for i, (x, y) in enumerate(pts):
        out[i, 0] = x
        out[i, 1] = y
        out[i, 2] = z
    return out


def sample_ellipse(
    *,
    bx0: float,
    bx1: float,
    by0: float,
    by1: float,
    z: float,
    n_points: int = 28,
) -> NDArray[np.float64]:
    """Sample CCW points on an axis-aligned ellipse inscribed in the box."""

    return sample_superellipse(
        bx0=bx0, bx1=bx1, by0=by0, by1=by1, n=2.0, z=z, n_points=n_points
    )


def sample_enclosure_plan(
    *,
    bx0: float,
    bx1: float,
    by0: float,
    by1: float,
    corner_radius: float,
    edge_type: int,
    z: float,
    plan_type: int = 1,
    plan_n: float = 2.0,
    n_per_edge: int = 3,
    n_per_corner: int = 4,
) -> NDArray[np.float64]:
    """Dispatch to the correct plan shape sampler for enclosure front/back/edges.

    ``plan_type`` selects: 1 = rounded rectangle, 2 = ellipse, 3 = superellipse.
    The ``corner_radius`` / ``edge_type`` arguments are only meaningful for
    plan 1 — for plan 2/3 they are ignored (matches WG's ``_sample_enclosure_plan``).
    """

    if int(plan_type) == 2:
        return sample_ellipse(bx0=bx0, bx1=bx1, by0=by0, by1=by1, z=z)
    if int(plan_type) == 3:
        return sample_superellipse(
            bx0=bx0, bx1=bx1, by0=by0, by1=by1, n=float(plan_n), z=z
        )
    return sample_rounded_rect(
        bx0=bx0,
        bx1=bx1,
        by0=by0,
        by1=by1,
        corner_radius=corner_radius,
        edge_type=edge_type,
        z=z,
        n_per_edge=n_per_edge,
        n_per_corner=n_per_corner,
    )


# ---------------------------------------------------------------------------
# Gmsh helpers (mesher style — go through require_gmsh()).
# ---------------------------------------------------------------------------

def _make_wire(
    points: NDArray[np.float64], *, closed: bool = True
) -> tuple[int, list[int], tuple[int, int]]:
    """Build a single-BSpline wire through ``points``.

    Returns ``(wire_tag, curve_tags, (first_pt, last_pt))``.
    """

    gmsh = require_gmsh()
    n = int(points.shape[0])
    pt_tags = [
        int(gmsh.model.occ.addPoint(float(points[i, 0]), float(points[i, 1]), float(points[i, 2])))
        for i in range(n)
    ]
    first_pt = pt_tags[0]
    last_pt = pt_tags[-1]
    if closed:
        pt_tags.append(pt_tags[0])
    spline = int(gmsh.model.occ.addBSpline(pt_tags))
    wire = int(gmsh.model.occ.addWire([spline]))
    return wire, [spline], (first_pt, last_pt)


def _add_curve_loop_from_curves(curve_tags: list[int]) -> int:
    gmsh = require_gmsh()
    try:
        return int(gmsh.model.occ.addCurveLoop([int(c) for c in curve_tags], reorient=True))
    except TypeError:
        try:
            return int(gmsh.model.occ.addCurveLoop(_ordered_curve_loop(curve_tags)))
        except RuntimeError:
            return int(gmsh.model.occ.addCurveLoop([int(c) for c in curve_tags]))


def _add_reversed_curve_loop_from_curves(curve_tags: list[int]) -> int:
    gmsh = require_gmsh()
    reversed_tags = [-int(c) for c in reversed(curve_tags)]
    try:
        return int(gmsh.model.occ.addCurveLoop(reversed_tags, reorient=True))
    except TypeError:
        try:
            return int(gmsh.model.occ.addCurveLoop(_ordered_curve_loop(reversed_tags)))
        except RuntimeError:
            return int(gmsh.model.occ.addCurveLoop(reversed_tags))


def _curve_endpoints(curve_tag: int) -> tuple[int, int]:
    gmsh = require_gmsh()
    boundary = gmsh.model.getBoundary([(1, int(curve_tag))], oriented=False, combined=False)
    point_tags = [int(tag) for dim, tag in boundary if int(dim) == 0]
    if len(point_tags) != 2:
        raise RuntimeError(f"could not resolve endpoints for curve {curve_tag}")
    return point_tags[0], point_tags[1]


def _ordered_curve_loop(curve_tags: list[int]) -> list[int]:
    """Order and orient curve tags for Gmsh builds without addCurveLoop(reorient)."""

    gmsh = require_gmsh()
    tags = [int(c) for c in curve_tags]
    if not tags:
        return tags

    gmsh.model.occ.synchronize()

    def chain_from(start: int, rest: list[int]) -> list[int] | None:
        loop_start, current_end = _curve_endpoints(start)
        ordered = [start]
        remaining = list(rest)

        while remaining:
            for index, curve in enumerate(remaining):
                curve_start, curve_end = _curve_endpoints(curve)
                if curve_start == current_end:
                    ordered.append(curve)
                    current_end = curve_end
                    remaining.pop(index)
                    break
                if curve_end == current_end:
                    ordered.append(-curve)
                    current_end = curve_start
                    remaining.pop(index)
                    break
            else:
                return None

        if current_end != loop_start:
            return None
        return ordered

    for index, curve in enumerate(tags):
        rest = tags[:index] + tags[index + 1 :]
        ordered = chain_from(curve, rest)
        if ordered is not None:
            return ordered
        ordered = chain_from(-curve, rest)
        if ordered is not None:
            return ordered

    raise RuntimeError("could not order curves into a closed loop")


def _add_ruled_section(loop_a: int, loop_b: int) -> list[tuple[int, int]]:
    gmsh = require_gmsh()
    return list(
        gmsh.model.occ.addThruSections(
            [int(loop_b), int(loop_a)],
            makeSolid=False,
            makeRuled=True,
        )
    )


def _boundary_curves_at_z_extreme(
    dimtags: list[tuple[int, int]], *, want_min_z: bool
) -> list[int]:
    """Return exterior boundary curves at the min or max z extreme of ``dimtags``.

    Requires a prior ``occ.synchronize()``. Internal seam edges (shared by two
    surfaces) are excluded via ``combined=True``.
    """

    gmsh = require_gmsh()
    boundary = gmsh.model.getBoundary(dimtags, oriented=False, combined=True)
    curve_tags = [int(abs(tag)) for dim, tag in boundary if int(dim) == 1]
    if not curve_tags:
        return []

    z_mid_map: dict[int, float] = {}
    z_extreme = float("inf") if want_min_z else float("-inf")
    for ctag in curve_tags:
        _, _, z0, _, _, z1 = gmsh.model.getBoundingBox(1, ctag)
        z_mid = 0.5 * (float(z0) + float(z1))
        z_mid_map[ctag] = z_mid
        if want_min_z:
            z_extreme = min(z_extreme, z_mid)
        else:
            z_extreme = max(z_extreme, z_mid)
    if not math.isfinite(z_extreme):
        return []

    z_mids = list(z_mid_map.values())
    z_span = max(abs(max(z_mids) - min(z_mids)), 1e-6)
    eps = 0.01 * z_span
    return [ctag for ctag in curve_tags if abs(z_mid_map[ctag] - z_extreme) <= eps]


def _classify_enclosure_surfaces(
    dimtags: list[tuple[int, int]], *, z_front: float, z_back: float
) -> dict[str, list[int]]:
    """Split enclosure surfaces into front/back/side by axial bounding box."""

    gmsh = require_gmsh()
    front: list[int] = []
    back: list[int] = []
    sides: list[int] = []
    eps = max(1e-6, abs(z_front - z_back) * 1e-3)

    for dim, tag in dimtags:
        if int(dim) != 2:
            continue
        _, _, z0, _, _, z1 = gmsh.model.getBoundingBox(int(dim), int(tag))
        if abs(z0 - z_front) <= eps and abs(z1 - z_front) <= eps:
            front.append(int(tag))
        elif abs(z0 - z_back) <= eps and abs(z1 - z_back) <= eps:
            back.append(int(tag))
        else:
            sides.append(int(tag))

    return {"front": front, "back": back, "sides": sides}


def _build_rounded_rectangle_enclosure_sector(
    *,
    mouth_curves: list[int],
    bx0: float,
    bx1: float,
    by0: float,
    by1: float,
    z_front: float,
    z_back: float,
    edge_depth: float,
    front_mesh_size: float,
    back_mesh_size: float,
    sign_x: float = 1.0,
    sign_y: float = 1.0,
) -> dict[str, Any]:
    """Build one rounded-rectangle enclosure sector bounded by symmetry axes."""

    gmsh = require_gmsh()
    endpoints = gmsh.model.getBoundary([(1, int(c)) for c in mouth_curves], oriented=False, combined=False)
    point_tags = [int(tag) for dim, tag in endpoints if int(dim) == 0]
    if len(point_tags) < 2:
        raise RuntimeError("could not resolve open-domain mouth endpoints")

    point_tags = list(dict.fromkeys(point_tags))
    coords = {tag: np.asarray(gmsh.model.getValue(0, tag, []), dtype=np.float64) for tag in point_tags}
    sx = 1.0 if sign_x >= 0.0 else -1.0
    sy = 1.0 if sign_y >= 0.0 else -1.0
    mouth_x = min(point_tags, key=lambda tag: (abs(float(coords[tag][1])), -sx * float(coords[tag][0])))
    mouth_y = min(point_tags, key=lambda tag: (abs(float(coords[tag][0])), -sy * float(coords[tag][1])))

    r = max(0.0, float(edge_depth))
    ox = float(bx1) if sx > 0.0 else float(bx0)
    oy = float(by1) if sy > 0.0 else float(by0)
    fx = ox - sx * r
    fy = oy - sy * r
    zf = float(z_front)
    zfo = zf - r
    zb = float(z_back)
    zbo = zb + r

    def pt(x: float, y: float, z: float, size: float) -> int:
        return int(gmsh.model.occ.addPoint(float(x), float(y), float(z), float(size)))

    def line(a: int, b: int) -> int:
        return int(gmsh.model.occ.addLine(int(a), int(b)))

    def arc(a: int, c: int, b: int) -> int:
        return int(gmsh.model.occ.addCircleArc(int(a), int(c), int(b)))

    def surface(curves: list[int], *, plane: bool = False) -> tuple[int, int]:
        loop = _add_curve_loop_from_curves(curves)
        if plane:
            try:
                return (2, int(gmsh.model.occ.addPlaneSurface([loop])))
            except Exception:
                pass
        return (2, int(gmsh.model.occ.addSurfaceFilling(loop)))

    # Sharp box edge (enc_edge=0): the rounded construction below insets the
    # front/back rims by ``r`` (fx=ox-sx*r, zfo=zf-r, ...). With r==0 those inset
    # points coincide with the outer points, so the roundover and corner "arcs"
    # become zero-length lines OCC rejects with "Could not create line". A sharp
    # box is a legitimate request, so build its four real faces directly — front
    # baffle, two side walls meeting at the shared sharp vertical corner edge,
    # and back cap — and omit the degenerate roundover/corner surfaces. Face
    # windings reproduce the rounded path's outward normals (+z front, +x/+y
    # side walls, -z back); front and back reuse the rounded loops verbatim.
    if r <= _SHARP_SECTOR_EDGE_EPS:
        origin_back = pt(0.0, 0.0, zb, back_mesh_size)
        sx_px_f = pt(ox, 0.0, zf, front_mesh_size)
        sx_px_b = pt(ox, 0.0, zb, back_mesh_size)
        sx_py_f = pt(0.0, oy, zf, front_mesh_size)
        sx_py_b = pt(0.0, oy, zb, back_mesh_size)
        sx_c_f = pt(ox, oy, zf, front_mesh_size)
        sx_c_b = pt(ox, oy, zb, back_mesh_size)

        l_mouth_x = line(mouth_x, sx_px_f)
        l_mouth_y = line(mouth_y, sx_py_f)
        l_front_x = line(sx_px_f, sx_c_f)
        l_front_y = line(sx_py_f, sx_c_f)
        l_back_x = line(sx_px_b, sx_c_b)
        l_back_y = line(sx_py_b, sx_c_b)
        l_back_x_axis = line(sx_px_b, origin_back)
        l_back_y_axis = line(origin_back, sx_py_b)
        v_x_axis = line(sx_px_f, sx_px_b)
        v_y_axis = line(sx_py_f, sx_py_b)
        v_corner = line(sx_c_f, sx_c_b)

        front = surface(
            [int(c) for c in mouth_curves] + [l_mouth_y, l_front_y, -l_front_x, -l_mouth_x],
            plane=True,
        )
        x_side = surface([v_x_axis, l_back_x, -v_corner, -l_front_x], plane=True)
        y_side = surface([l_front_y, v_corner, -l_back_y, -v_y_axis], plane=True)
        back = surface([l_back_x_axis, l_back_y_axis, l_back_y, -l_back_x], plane=True)

        return {
            "dimtags": [front, x_side, y_side, back],
            "front": [front[1]],
            "back": [back[1]],
            "sides": [x_side[1], y_side[1]],
            "front_edges": [],
            "back_edges": [],
            "bounds": {
                "bx0": float(bx0),
                "bx1": float(bx1),
                "by0": float(by0),
                "by1": float(by1),
                "z_front": z_front,
                "z_back": z_back,
                "cx": 0.5 * (float(bx0) + float(bx1)),
                "cy": 0.5 * (float(by0) + float(by1)),
            },
        }

    origin_back = pt(0.0, 0.0, zb, back_mesh_size)
    px_f = pt(fx, 0.0, zf, front_mesh_size)
    px_o_f = pt(ox, 0.0, zfo, front_mesh_size)
    px_o_b = pt(ox, 0.0, zbo, back_mesh_size)
    px_b = pt(fx, 0.0, zb, back_mesh_size)
    py_f = pt(0.0, fy, zf, front_mesh_size)
    py_o_f = pt(0.0, oy, zfo, front_mesh_size)
    py_o_b = pt(0.0, oy, zbo, back_mesh_size)
    py_b = pt(0.0, fy, zb, back_mesh_size)
    c_f = pt(fx, fy, zf, front_mesh_size)
    c_x_f = pt(ox, fy, zfo, front_mesh_size)
    c_y_f = pt(fx, oy, zfo, front_mesh_size)
    c_x_b = pt(ox, fy, zbo, back_mesh_size)
    c_y_b = pt(fx, oy, zbo, back_mesh_size)
    c_b = pt(fx, fy, zb, back_mesh_size)

    cx_axis_f = pt(fx, 0.0, zfo, front_mesh_size)
    cx_axis_b = pt(fx, 0.0, zbo, back_mesh_size)
    cy_axis_f = pt(0.0, fy, zfo, front_mesh_size)
    cy_axis_b = pt(0.0, fy, zbo, back_mesh_size)
    corner_front_center = pt(fx, fy, zfo, front_mesh_size)
    corner_back_center = pt(fx, fy, zbo, back_mesh_size)

    l_mouth_x = line(mouth_x, px_f)
    l_front_x = line(px_f, c_f)
    l_front_y = line(py_f, c_f)
    l_mouth_y = line(mouth_y, py_f)
    front = surface([int(c) for c in mouth_curves] + [l_mouth_y, l_front_y, -l_front_x, -l_mouth_x], plane=True)

    l_x_front_outer = line(px_o_f, c_x_f)
    l_x_outer = line(px_o_f, px_o_b)
    l_x_outer_corner = line(c_x_f, c_x_b)
    l_x_back_outer = line(px_o_b, c_x_b)
    l_x_back_inset = line(px_b, c_b)
    l_back_x_axis = line(px_b, origin_back)
    l_back_y_axis = line(origin_back, py_b)
    l_back_y_inset = line(py_b, c_b)
    l_y_front_outer = line(py_o_f, c_y_f)
    l_y_outer = line(py_o_f, py_o_b)
    l_y_outer_corner = line(c_y_f, c_y_b)
    l_y_back_outer = line(py_o_b, c_y_b)

    a_x_front_axis = arc(px_f, cx_axis_f, px_o_f) if r > 0.0 else line(px_f, px_o_f)
    a_x_front_corner = arc(c_f, corner_front_center, c_x_f) if r > 0.0 else line(c_f, c_x_f)
    a_x_back_axis = arc(px_o_b, cx_axis_b, px_b) if r > 0.0 else line(px_o_b, px_b)
    a_x_back_corner = arc(c_x_b, corner_back_center, c_b) if r > 0.0 else line(c_x_b, c_b)
    a_y_front_axis = arc(py_f, cy_axis_f, py_o_f) if r > 0.0 else line(py_f, py_o_f)
    a_y_front_corner = arc(c_f, corner_front_center, c_y_f) if r > 0.0 else line(c_f, c_y_f)
    a_y_back_axis = arc(py_o_b, cy_axis_b, py_b) if r > 0.0 else line(py_o_b, py_b)
    a_y_back_corner = arc(c_y_b, corner_back_center, c_b) if r > 0.0 else line(c_y_b, c_b)
    a_corner_front = arc(c_x_f, corner_front_center, c_y_f) if r > 0.0 else line(c_x_f, c_y_f)
    a_corner_back = arc(c_x_b, corner_back_center, c_y_b) if r > 0.0 else line(c_x_b, c_y_b)

    x_front_edge = surface([a_x_front_axis, l_x_front_outer, -a_x_front_corner, -l_front_x])
    x_side = surface([l_x_outer, l_x_back_outer, -l_x_outer_corner, -l_x_front_outer], plane=True)
    x_back_edge = surface([a_x_back_axis, l_x_back_inset, -a_x_back_corner, -l_x_back_outer])
    back = surface([l_back_x_axis, l_back_y_axis, l_back_y_inset, -l_x_back_inset], plane=True)
    y_front_edge = surface([a_y_front_axis, l_y_front_outer, -a_y_front_corner, -l_front_y])
    corner_front = surface([a_x_front_corner, a_corner_front, -a_y_front_corner])
    corner_side = surface([l_x_outer_corner, a_corner_back, -l_y_outer_corner, -a_corner_front])
    corner_back = surface([a_x_back_corner, -a_y_back_corner, -a_corner_back])
    y_back_edge = surface([a_y_back_axis, l_back_y_inset, -a_y_back_corner, -l_y_back_outer])
    y_side = surface([l_y_outer, l_y_back_outer, -l_y_outer_corner, -l_y_front_outer], plane=True)

    dimtags = [
        front,
        x_front_edge,
        x_side,
        x_back_edge,
        back,
        y_front_edge,
        corner_front,
        corner_side,
        corner_back,
        y_back_edge,
        y_side,
    ]
    tags = [tag for _, tag in dimtags]
    front_edges = [x_front_edge[1], y_front_edge[1], corner_front[1]]
    back_edges = [x_back_edge[1], y_back_edge[1], corner_back[1]]
    return {
        "dimtags": dimtags,
        "front": [front[1]],
        "back": [back[1]],
        "sides": [tag for tag in tags if tag not in {front[1], back[1], *front_edges, *back_edges}],
        "front_edges": front_edges,
        "back_edges": back_edges,
        "bounds": {
            "bx0": float(bx0),
            "bx1": float(bx1),
            "by0": float(by0),
            "by1": float(by1),
            "z_front": z_front,
            "z_back": z_back,
            "cx": 0.5 * (float(bx0) + float(bx1)),
            "cy": 0.5 * (float(by0) + float(by1)),
        },
    }


def _quadrant_for_curve(curve_tag: int) -> tuple[float, float]:
    gmsh = require_gmsh()
    x0, y0, _z0, x1, y1, _z1 = gmsh.model.getBoundingBox(1, int(curve_tag))
    sx = 1.0 if abs(float(x1)) >= abs(float(x0)) else -1.0
    sy = 1.0 if abs(float(y1)) >= abs(float(y0)) else -1.0
    return sx, sy


def _merge_enclosure_parts(parts: list[dict[str, Any]], bounds: dict[str, float]) -> dict[str, Any]:
    return {
        "dimtags": [dimtag for part in parts for dimtag in part["dimtags"]],
        "front": [tag for part in parts for tag in part["front"]],
        "back": [tag for part in parts for tag in part["back"]],
        "sides": [tag for part in parts for tag in part["sides"]],
        "front_edges": [tag for part in parts for tag in part["front_edges"]],
        "back_edges": [tag for part in parts for tag in part["back_edges"]],
        "bounds": bounds,
    }


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_enclosure_box(
    *,
    inner_dimtags: list[tuple[int, int]],
    inner_points: NDArray[np.float64],
    enclosure: HornEnclosure,
    closed: bool = True,
) -> dict[str, Any]:
    """Build a closed-domain rear enclosure around the horn mouth.

    Returns a dict mirroring WG's ``_build_enclosure_box`` output:

    .. code-block:: text

        {
            "dimtags":      [(2, t), ...],   # every enclosure surface
            "front":        [tags],           # front baffle
            "back":         [tags],           # back cap
            "sides":        [tags],           # side wall (between roundovers)
            "front_edges":  [tags],           # front roundover surfaces
            "back_edges":   [tags],           # back roundover surfaces
            "bounds":       {bx0, bx1, by0, by1, z_front, z_back, cx, cy},
        }

    Raises ``NotImplementedError`` for unsupported plan/edge/open-domain
    combinations. Implemented plan values are ``1`` rounded rectangle,
    ``2`` ellipse, and ``3`` superellipse; implemented edge values are ``1``
    rounded fillet and ``2`` chamfer. Open-domain support currently requires
    rounded-rectangle plan geometry.
    """

    if int(enclosure.plan_type) not in (1, 2, 3):
        raise NotImplementedError(
            f"HornEnclosure.plan_type={enclosure.plan_type} not yet supported "
            "by hornlab-mesher (only plan_type ∈ {1, 2, 3} implemented)."
        )
    if int(enclosure.edge_type) not in (1, 2):
        raise NotImplementedError(
            f"HornEnclosure.edge_type={enclosure.edge_type} not yet supported "
            "by hornlab-mesher (only edge_type ∈ {1, 2} implemented)."
        )
    if not inner_dimtags:
        raise ValueError("build_enclosure_box requires non-empty inner_dimtags")

    gmsh = require_gmsh()

    mouth_pts = inner_points[:, -1, :]
    x_min = float(mouth_pts[:, 0].min())
    x_max = float(mouth_pts[:, 0].max())
    y_min = float(mouth_pts[:, 1].min())
    y_max = float(mouth_pts[:, 1].max())
    z_front = float(mouth_pts[:, 2].max())

    # The front baffle is an annular face in the z_front plane whose hole is
    # bounded by the mouth ring. A rolled-back lip that stays radially inside
    # the ring legitimately protrudes through that hole (R-OSSE, ICW
    # rollback), but a wall segment crossing the baffle plane radially
    # *outside* the ring (deep curl past 180 deg, bulbous lip folding back
    # in) would be bisected by the front face: the box still welds
    # watertight around it, so nothing downstream can catch the
    # self-intersection — fail explicitly instead. The final segment of each
    # meridian attaches to the ring itself and is excluded.
    ring_r = np.hypot(mouth_pts[:, 0], mouth_pts[:, 1])
    wall_r = np.hypot(inner_points[:, :, 0], inner_points[:, :, 1])
    wall_z = inner_points[:, :, 2]
    if wall_z.shape[1] >= 3:
        za = wall_z[:, :-2]
        zb = wall_z[:, 1:-1]
        straddle = (za - z_front) * (zb - z_front) < 0.0
        if bool(np.any(straddle)):
            ra = wall_r[:, :-2]
            rb = wall_r[:, 1:-1]
            denom = np.where(np.abs(zb - za) > 1.0e-12, zb - za, 1.0)
            t = np.clip((z_front - za) / denom, 0.0, 1.0)
            r_cross = ra + (rb - ra) * t
            bad = straddle & (r_cross > ring_r[:, None] + 1.0e-6)
            if bool(np.any(bad)):
                worst = float(np.max(np.where(bad, r_cross, -np.inf)))
                raise NotImplementedError(
                    "build_enclosure_box: the horn wall crosses the front-baffle "
                    f"plane (z={z_front:.3f} mm) at radius {worst:.3f} mm, outside "
                    "the mouth ring — the baffle would bisect the wall (deep "
                    "rollback / curled-in lip). Build it free-standing "
                    "(enc_depth=0) instead."
                )

    z_throat = float(np.min(inner_points[:, 0, 2]))
    horn_length = z_front - z_throat
    min_enc_depth = horn_length + float(enclosure.depth_margin_mm)
    enc_depth = float(enclosure.depth_mm)
    if enc_depth < min_enc_depth:
        logger.warning(
            "[hornlab-mesher] enc_depth (%.2f mm) < horn length (%.2f mm) + margin (%.2f mm); "
            "clamping to %.2f mm.",
            enc_depth, horn_length, enclosure.depth_margin_mm, min_enc_depth,
        )
        enc_depth = min_enc_depth
    z_back = z_front - enc_depth

    x_open = not closed and x_min >= -1.0e-6
    y_open = not closed and y_min >= -1.0e-6
    bx0 = 0.0 if x_open else x_min - float(enclosure.space_l_mm)
    bx1 = x_max + float(enclosure.space_r_mm)
    by0 = 0.0 if y_open else y_min - float(enclosure.space_b_mm)
    by1 = y_max + float(enclosure.space_t_mm)

    bounds = {
        "bx0": bx0,
        "bx1": bx1,
        "by0": by0,
        "by1": by1,
        "z_front": z_front,
        "z_back": z_back,
        "cx": 0.5 * (bx0 + bx1),
        "cy": 0.5 * (by0 + by1),
    }

    # Roundover limits must describe the *physical* (mirror-completed) box:
    # on reduced domains the cut plane at x=0 / y=0 is not a wall, so the
    # mirrored half-extent is bx1 / by1 itself, and the spacings on the cut
    # side are never applied and must not participate in the margin clamp
    # (otherwise a quarter build of a design clamps the roundover harder than
    # the identical full build — or forces a sharp box when the unused
    # cut-side spacing is 0).
    half_w = float(bx1) if x_open else 0.5 * (bx1 - bx0)
    half_h = float(by1) if y_open else 0.5 * (by1 - by0)
    applied_spacings = [float(enclosure.space_r_mm), float(enclosure.space_t_mm)]
    if not x_open:
        applied_spacings.append(float(enclosure.space_l_mm))
    if not y_open:
        applied_spacings.append(float(enclosure.space_b_mm))
    margin_edge_limit = max(0.0, min(applied_spacings))
    clamped_edge = _clamp_edge_roundover(
        float(enclosure.edge_mm), margin_edge_limit, half_w, half_h
    )
    edge_depth = min(clamped_edge, max(0.0, enc_depth * 0.5))
    front_mesh_size = float(enclosure.front_mesh_size_mm or 0.0)
    back_mesh_size = float(enclosure.back_mesh_size_mm or 0.0)

    mouth_curves = _boundary_curves_at_z_extreme(inner_dimtags, want_min_z=False)
    if not mouth_curves:
        raise RuntimeError("could not resolve inner-wall mouth boundary curves")
    if not closed:
        if int(enclosure.plan_type) != 1:
            raise NotImplementedError("Open-domain enclosure currently supports only rounded-rectangle plan_type=1.")
        # A reduced grid covers one or two quadrants: quarter -> Q1; half about
        # the xz plane (quadrants 12) -> Q1+Q2; half about the yz plane
        # (quadrants 14) -> Q1+Q4. Build one rounded-rectangle sector per
        # quadrant that carries mouth curves and merge them. Adjacent half-model
        # sectors meet on the off-cut axis (x=0 for an xz half, y=0 for a yz
        # half); their coincident seam geometry welds into a watertight internal
        # edge, leaving free edges only on the real symmetry cut plane(s). A
        # single quarter sector keeps both of its axis edges on the cut planes.
        quadrant_keys = ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))
        grouped: dict[tuple[float, float], list[int]] = {key: [] for key in quadrant_keys}
        for curve in mouth_curves:
            grouped[_quadrant_for_curve(curve)].append(int(curve))
        present = [key for key in quadrant_keys if grouped[key]]
        if not present:
            raise RuntimeError("could not group open-domain mouth curves into quadrants")
        parts = [
            _build_rounded_rectangle_enclosure_sector(
                mouth_curves=grouped[key],
                bx0=bx0,
                bx1=bx1,
                by0=by0,
                by1=by1,
                z_front=z_front,
                z_back=z_back,
                edge_depth=edge_depth,
                front_mesh_size=front_mesh_size,
                back_mesh_size=back_mesh_size,
                sign_x=key[0],
                sign_y=key[1],
            )
            for key in present
        ]
        return _merge_enclosure_parts(parts, bounds)
    if int(enclosure.plan_type) == 1 and int(enclosure.edge_type) == 1:
        grouped: dict[tuple[float, float], list[int]] = {
            (1.0, 1.0): [],
            (-1.0, 1.0): [],
            (-1.0, -1.0): [],
            (1.0, -1.0): [],
        }
        for curve in mouth_curves:
            grouped[_quadrant_for_curve(curve)].append(int(curve))
        if all(grouped[key] for key in grouped):
            parts = [
                _build_rounded_rectangle_enclosure_sector(
                    mouth_curves=grouped[key],
                    bx0=bx0,
                    bx1=bx1,
                    by0=by0,
                    by1=by1,
                    z_front=z_front,
                    z_back=z_back,
                    edge_depth=edge_depth,
                    front_mesh_size=front_mesh_size,
                    back_mesh_size=back_mesh_size,
                    sign_x=key[0],
                    sign_y=key[1],
                )
                for key in ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))
            ]
            return _merge_enclosure_parts(parts, bounds)
    mouth_loop = _add_curve_loop_from_curves(mouth_curves)

    # Minimum BSpline corner radius: an interpolating BSpline through a true
    # rectangle (corner_radius=0) rounds corners by several mm. A tiny radius
    # keeps has_corners=True so the arc-sampler produces points the BSpline
    # can track accurately.
    min_bspline_r = 0.1

    # --- Front baffle: enclosure plan shape with horn-mouth hole. ---
    ring0_pts = sample_enclosure_plan(
        bx0=bx0 + clamped_edge,
        bx1=bx1 - clamped_edge,
        by0=by0 + clamped_edge,
        by1=by1 - clamped_edge,
        corner_radius=min_bspline_r,
        edge_type=int(enclosure.edge_type),
        z=z_front,
        plan_type=int(enclosure.plan_type),
        plan_n=float(enclosure.plan_n),
    )
    # Defensive: the outer ring must lie exactly in the z_front plane so the
    # two-loop addPlaneSurface call below doesn't silently fall back to
    # addSurfaceFilling (which diverges from ATH by warping the hole-cut face).
    ring0_pts[:, 2] = z_front
    ring0_wire, ring0_curves, _ring0_eps = _make_wire(ring0_pts, closed=True)
    ring0_loop = int(gmsh.model.occ.addCurveLoop([int(c) for c in ring0_curves]))
    front_tag = int(gmsh.model.occ.addPlaneSurface([ring0_loop, mouth_loop]))

    generated: list[tuple[int, int]] = [(2, front_tag)]
    front_edges: list[tuple[int, int]] = []
    back_edges: list[tuple[int, int]] = []
    current_profile = ring0_wire

    def make_ring(z: float, radial_t: float) -> tuple[int, list[int], tuple[int, int]]:
        d = clamped_edge * (1.0 - radial_t)
        r = max(min_bspline_r, clamped_edge * radial_t)
        pts = sample_enclosure_plan(
            bx0=bx0 + d,
            bx1=bx1 - d,
            by0=by0 + d,
            by1=by1 - d,
            corner_radius=r,
            edge_type=int(enclosure.edge_type),
            z=z,
            plan_type=int(enclosure.plan_type),
            plan_n=float(enclosure.plan_n),
        )
        return _make_wire(pts, closed=True)

    # --- Front roundover (inset -> outer at z = z_front - edge_depth). ---
    # Rounded mode (edge_type=1): 3 profiles with ruled thru-sections between
    # consecutive pairs (inset -> mid -> outer).
    # Chamfer mode (edge_type=2): single ruled surface inset -> outer at full
    # axial + radial offset.
    if edge_depth > 0.0:
        if int(enclosure.edge_type) == 1:
            prev_wire = current_profile
            for t in (0.5, 1.0):
                angle = t * (math.pi / 2.0)
                axial_t = 1.0 - math.cos(angle)
                radial_t = math.sin(angle)
                z_ring = z_front - axial_t * edge_depth
                ring_wire, _, _ = make_ring(z_ring, radial_t)
                section = _add_ruled_section(prev_wire, ring_wire)
                generated.extend(section)
                front_edges.extend(section)
                prev_wire = ring_wire
            current_profile = prev_wire
        else:
            ring_wire, _, _ = make_ring(z_front - edge_depth, 1.0)
            section = _add_ruled_section(current_profile, ring_wire)
            generated.extend(section)
            front_edges.extend(section)
            current_profile = ring_wire

    # --- Side walls: straight ruled surface from front-outer to back-outer. ---
    z_outer_back = z_back + edge_depth if edge_depth > 0.0 else z_back
    back_outer_pts = sample_enclosure_plan(
        bx0=bx0,
        bx1=bx1,
        by0=by0,
        by1=by1,
        corner_radius=clamped_edge,
        edge_type=int(enclosure.edge_type),
        z=z_outer_back,
        plan_type=int(enclosure.plan_type),
        plan_n=float(enclosure.plan_n),
    )
    back_outer_wire, back_outer_curves, back_outer_eps = _make_wire(
        back_outer_pts, closed=True
    )
    generated.extend(_add_ruled_section(current_profile, back_outer_wire))
    current_profile = back_outer_wire
    current_curves = back_outer_curves
    # --- Back roundover (mirror image of front: outer -> inset at z_back). ---
    if edge_depth > 0.0:
        if int(enclosure.edge_type) == 1:
            prev_wire = current_profile
            for t in (0.5, 1.0):
                angle = t * (math.pi / 2.0)
                axial_t = math.sin(angle)
                radial_t = math.cos(angle)
                z_ring = z_back + (1.0 - axial_t) * edge_depth
                ring_wire, ring_curves, ring_eps = make_ring(z_ring, radial_t)
                section = _add_ruled_section(prev_wire, ring_wire)
                generated.extend(section)
                back_edges.extend(section)
                prev_wire = ring_wire
                current_curves = ring_curves
            current_profile = prev_wire
        else:
            ring_wire, ring_curves, ring_eps = make_ring(z_back, 0.0)
            section = _add_ruled_section(current_profile, ring_wire)
            generated.extend(section)
            back_edges.extend(section)
            current_profile = ring_wire
            current_curves = ring_curves

    # --- Back cap: planar surface from final inset wire at z_back. ---
    back_cap_loop = _add_reversed_curve_loop_from_curves(current_curves)
    try:
        back_cap = int(gmsh.model.occ.addPlaneSurface([back_cap_loop]))
    except Exception:
        back_cap = int(gmsh.model.occ.addSurfaceFilling(current_profile))
    generated.append((2, back_cap))

    gmsh.model.occ.synchronize()

    dimtags = [(2, int(tag)) for dim, tag in generated if int(dim) == 2]
    front_edge_tags = {int(tag) for _, tag in front_edges}
    back_edge_tags = {int(tag) for _, tag in back_edges}
    all_edge_tags = front_edge_tags | back_edge_tags

    split = _classify_enclosure_surfaces(dimtags, z_front=z_front, z_back=z_back)
    front_edge_surfaces = [tag for tag in split["sides"] if tag in front_edge_tags]
    back_edge_surfaces = [tag for tag in split["sides"] if tag in back_edge_tags]
    side_surfaces = [tag for tag in split["sides"] if tag not in all_edge_tags]

    return {
        "dimtags": dimtags,
        "front": split["front"],
        "back": split["back"],
        "sides": side_surfaces,
        "front_edges": front_edge_surfaces,
        "back_edges": back_edge_surfaces,
        "bounds": bounds,
    }
