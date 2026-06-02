"""Cabinet enclosure builder for point-grid horns.

Supports closed-domain horns (``quadrants=1234``, i.e. the only path the WG
frontend and Optimizer actually exercise in production) with:

* ``HornEnclosure.plan_type`` ``∈ {1, 2, 3}`` — rounded rectangle, ellipse,
  superellipse (exponent from ``plan_n``).
* ``HornEnclosure.edge_type`` ``∈ {1, 2}`` — rounded fillet (3 profile
  thru-section) or chamfer (single ruled surface).

The open-domain (``closed=False``) branch is intentionally not ported — the
WG mesh route forces ``quadrants=1234`` so that code path is dead in
production.

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
    """Dispatch to the correct plan shape sampler for the cabinet front/back/edges.

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
        return int(gmsh.model.occ.addCurveLoop([int(c) for c in curve_tags]))


def _add_reversed_curve_loop_from_curves(curve_tags: list[int]) -> int:
    gmsh = require_gmsh()
    reversed_tags = [-int(c) for c in reversed(curve_tags)]
    try:
        return int(gmsh.model.occ.addCurveLoop(reversed_tags, reorient=True))
    except TypeError:
        return int(gmsh.model.occ.addCurveLoop(reversed_tags))


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
    """Build a closed-domain cabinet enclosure around the horn mouth.

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

    Raises ``NotImplementedError`` for ``plan_type != 1`` or ``edge_type != 1``
    or ``closed=False`` — those configurations exist in WG but are not yet
    ported to hornlab-mesher.
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
    if not closed:
        raise NotImplementedError(
            "Open-domain enclosure (closed=False) is not yet supported."
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

    bx0 = x_min - float(enclosure.space_l_mm)
    bx1 = x_max + float(enclosure.space_r_mm)
    by0 = y_min - float(enclosure.space_b_mm)
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

    half_w = 0.5 * (bx1 - bx0)
    half_h = 0.5 * (by1 - by0)
    clamped_edge = max(0.0, min(float(enclosure.edge_mm), half_w - 0.1, half_h - 0.1))
    edge_depth = min(clamped_edge, max(0.0, enc_depth * 0.5))

    mouth_curves = _boundary_curves_at_z_extreme(inner_dimtags, want_min_z=False)
    if not mouth_curves:
        raise RuntimeError("could not resolve inner-wall mouth boundary curves")
    mouth_loop = _add_curve_loop_from_curves(mouth_curves)

    # Minimum BSpline corner radius: an interpolating BSpline through a true
    # rectangle (corner_radius=0) rounds corners by several mm. A tiny radius
    # keeps has_corners=True so the arc-sampler produces points the BSpline
    # can track accurately.
    min_bspline_r = 0.1

    # --- Front baffle: cabinet plan shape with horn-mouth hole. ---
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
    # addSurfaceFilling (which breaks parity by warping the hole-cut face).
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
    profile_pts = back_outer_eps

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
                profile_pts = ring_eps
            current_profile = prev_wire
        else:
            ring_wire, ring_curves, ring_eps = make_ring(z_back, 0.0)
            section = _add_ruled_section(current_profile, ring_wire)
            generated.extend(section)
            back_edges.extend(section)
            current_profile = ring_wire
            current_curves = ring_curves
            profile_pts = ring_eps

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
