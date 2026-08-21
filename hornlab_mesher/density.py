from __future__ import annotations

import math
from typing import Any

import numpy as np

from .cost import TRIANGLES_PER_AREA_OVER_H2
from .geometry import BuiltGeometry, MeshDensity
from .profile_common import _parse_number_list
from .tags import PhysicalGroup, SOURCE_TAGS


_PREMESH_TRIANGLE_LIMIT_SLACK = 2.0
# Mesh-size growth per millimetre of distance from a roundover seam curve
# (1.0 = size equals distance: each element row roughly doubles). Bounds the
# boundary-to-interior size jump the 2D mesher sees at the seam.
# Fraction of the wall thickness a faceted outer shell may spend on chord
# sagitta before it is treated as reaching the acoustic surface. Provisional
# and empirical, not derived: it has to absorb the inner surface's own
# faceting, which does not always bulge the other way.
_WALL_CLEARANCE_FRACTION = 0.35
# Gmsh treats a size field as a target, not a maximum. A 40 mm target has been
# observed to realise a 48.5 mm edge on the free-standing outer wall, and
# sagitta grows with the square of the chord, so the target is divided by this
# before the geometric bound is applied.
_WALL_CLEARANCE_SIZE_OVERSHOOT = 1.25
_ENCLOSURE_SEAM_SIZE_GRADIENT = 1.0
_ENCLOSURE_SEAM_DISTANCE_SAMPLING_MIN = 64.0
_ENCLOSURE_SEAM_DISTANCE_SAMPLING_MAX = 2000.0


def _parse_quadrant_resolutions(
    value: float | str | None, fallback: float
) -> list[float]:
    """Parse WG-style per-quadrant resolution list q1..q4."""

    fallback = float(fallback)
    if value is None:
        return [fallback, fallback, fallback, fallback]

    text = str(value).strip()
    if not text:
        return [fallback, fallback, fallback, fallback]

    try:
        scalar = float(text)
    except ValueError:
        scalar = float("nan")
    if math.isfinite(scalar) and scalar > 0.0:
        return [scalar, scalar, scalar, scalar]

    parts = _parse_number_list(text, invalid="empty", evaluate=False)
    if not parts:
        return [fallback, fallback, fallback, fallback]

    out: list[float] = []
    for i in range(4):
        if i < len(parts) and math.isfinite(parts[i]) and parts[i] > 0.0:
            out.append(float(parts[i]))
        else:
            out.append(fallback)
    return out


def _panel_bilinear_resolution_formula(
    q_values: list[float],
    *,
    bx0: float,
    bx1: float,
    by0: float,
    by1: float,
) -> str:
    dx = max(abs(bx1 - bx0), 1e-6)
    dy = max(abs(by1 - by0), 1e-6)
    u = f"((x - ({bx0:.12g})) / ({dx:.12g}))"
    v = f"((y - ({by0:.12g})) / ({dy:.12g}))"

    q1 = float(q_values[0])
    q2 = float(q_values[1])
    q3 = float(q_values[2])
    q4 = float(q_values[3])

    return (
        f"({q3:.12g})*(1-({u}))*(1-({v})) + "
        f"({q4:.12g})*({u})*(1-({v})) + "
        f"({q2:.12g})*(1-({u}))*({v}) + "
        f"({q1:.12g})*({u})*({v})"
    )


def _enclosure_resolution_formula(
    front_q: list[float],
    back_q: list[float],
    *,
    bx0: float,
    bx1: float,
    by0: float,
    by1: float,
    z_front: float,
    z_back: float,
) -> str:
    dz = max(abs(z_front - z_back), 1e-6)
    t = f"(({z_front:.12g}) - z) / ({dz:.12g})"
    front_expr = _panel_bilinear_resolution_formula(
        front_q,
        bx0=bx0,
        bx1=bx1,
        by0=by0,
        by1=by1,
    )
    back_expr = _panel_bilinear_resolution_formula(
        back_q,
        bx0=bx0,
        bx1=bx1,
        by0=by0,
        by1=by1,
    )
    return f"(({front_expr})*(1-({t})) + ({back_expr})*({t}))"


def _collect_boundary_curves(surface_tags: list[int]) -> list[int]:
    if not surface_tags:
        return []

    import gmsh

    ordered: list[int] = []
    seen: set[int] = set()
    for surface_tag in surface_tags:
        for dim, curve_tag in gmsh.model.getBoundary(
            [(2, int(surface_tag))],
            oriented=False,
            combined=False,
        ):
            if int(dim) != 1:
                continue
            curve_tag_i = int(curve_tag)
            if curve_tag_i not in seen:
                seen.add(curve_tag_i)
                ordered.append(curve_tag_i)
    return ordered


def _enclosure_edge_size_mm(
    q_values: list[float],
) -> float | None:
    """User-derived target for a retained cosmetic enclosure edge.

    Geometry-level defeaturing removes strips narrower than the adjacent user
    target before this function runs.  A retained strip therefore needs no
    radius-derived ``edge / N`` refinement; the finest touching user target is
    sufficient and keeps seam grading bounded by explicit mm values.
    """

    positive_q = [
        float(value)
        for value in q_values
        if math.isfinite(float(value)) and float(value) > 0.0
    ]
    return min(positive_q) if positive_q else None


def _split_enclosure_panel_surfaces(
    surface_tags: list[int],
    *,
    front_edge_tags: list[int],
    back_edge_tags: list[int],
    z_front: float,
    z_back: float,
) -> tuple[list[int], list[int]]:
    """Return non-roundover enclosure front/back panel surfaces."""

    if not surface_tags:
        return [], []

    import gmsh

    edge_tags = {int(tag) for tag in front_edge_tags}
    edge_tags.update(int(tag) for tag in back_edge_tags)
    z_span = max(abs(float(z_front) - float(z_back)), 1.0)
    # OCC face bounding boxes carry a finite gap (~1e-3 mm even for exact
    # planes). The old 1e-6 tolerance silently declassified every real panel,
    # so front/back panels never received their bilinear user-size field and
    # were meshed at the z-graded enclosure size. Stay far below any real front-to-back
    # separation while tolerating the bbox slop.
    eps = max(0.05, z_span * 1.0e-4)
    front: list[int] = []
    back: list[int] = []
    for tag in surface_tags:
        tag_i = int(tag)
        if tag_i in edge_tags:
            continue
        _x0, _y0, z0, _x1, _y1, z1 = gmsh.model.getBoundingBox(2, tag_i)
        if (
            abs(float(z0) - float(z_front)) <= eps
            and abs(float(z1) - float(z_front)) <= eps
        ):
            front.append(tag_i)
        elif (
            abs(float(z0) - float(z_back)) <= eps
            and abs(float(z1) - float(z_back)) <= eps
        ):
            back.append(tag_i)
    return front, back


def _effective_size_mm(values: list[float]) -> float | None:
    sizes = [
        float(value)
        for value in values
        if math.isfinite(float(value)) and float(value) > 0.0
    ]
    if not sizes:
        return None
    inv_sq = sum(1.0 / (value * value) for value in sizes)
    return math.sqrt(float(len(sizes)) / inv_sq) if inv_sq > 0.0 else None


def _axial_ramp_effective_size_mm(
    start: float, end: float
) -> float | None:
    """Effective size for the linear part of the clamped axial size field.

    For ``h(t) = start + (end - start) * t``, integrating ``1 / h(t)^2``
    over the unit interval gives ``1 / (start * end)`` exactly.
    """

    start = float(start)
    end = float(end)
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or start <= 0.0
        or end <= 0.0
    ):
        return None
    return math.sqrt(start * end)


def _bbox_surface_area_estimate_mm2(
    bbox: tuple[float, float, float, float, float, float],
) -> float:
    x0, y0, z0, x1, y1, z1 = (float(value) for value in bbox)
    dims = [abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)]
    nonzero = [value for value in dims if value > 1.0e-9]
    if len(nonzero) < 2:
        return 0.0
    if len(nonzero) == 2:
        return nonzero[0] * nonzero[1]
    dx, dy, dz = dims
    # Fallback only: OCC getMass is preferred. For curved/non-planar surfaces,
    # projected bbox products give a conservative-enough sizing forecast.
    return dx * dy + dx * dz + dy * dz


def _surface_area_mm2(surface_tags: list[int]) -> float:
    if not surface_tags:
        return 0.0

    import gmsh

    total = 0.0
    seen: set[int] = set()
    for raw_tag in surface_tags:
        tag = int(raw_tag)
        if tag in seen:
            continue
        seen.add(tag)
        area: float | None = None
        occ = getattr(gmsh.model, "occ", None)
        for get_mass in (
            getattr(occ, "getMass", None),
            getattr(gmsh.model, "getMass", None),
        ):
            if get_mass is None:
                continue
            try:
                candidate = float(get_mass(2, tag))
            except Exception:
                continue
            if math.isfinite(candidate) and candidate > 0.0:
                area = candidate
                break
        if area is None:
            try:
                area = _bbox_surface_area_estimate_mm2(
                    gmsh.model.getBoundingBox(2, tag)
                )
            except Exception:
                area = 0.0
        if math.isfinite(area) and area > 0.0:
            total += area
    return total


def _enclosure_triangle_regions(
    mesh_groups: dict[str, list[int]],
    *,
    throat_res: float,
    mouth_res: float,
    rear_res: float,
    interface_res: float,
    front_q: list[float],
    back_q: list[float],
    front_panel_q: list[float],
    back_panel_q: list[float],
    front_edge_size: float | None,
    back_edge_size: float | None,
    front_panels: list[int],
    back_panels: list[int],
) -> list[tuple[float, float, str]]:
    regions: list[tuple[float, float, str]] = []
    consumed: set[int] = set()

    def add(surface_tags: list[int], size: float | None, label: str) -> None:
        if size is None or not math.isfinite(float(size)) or float(size) <= 0.0:
            return
        tags: list[int] = []
        for raw_tag in surface_tags:
            tag = int(raw_tag)
            if tag in consumed:
                continue
            consumed.add(tag)
            tags.append(tag)
        area = _surface_area_mm2(tags)
        if area > 0.0:
            regions.append((area, float(size), label))

    axial_size = _axial_ramp_effective_size_mm(throat_res, mouth_res)
    add(mesh_groups.get("inner", []), axial_size, "waveguide wall")
    add(mesh_groups.get("mouth", []), axial_size, "mouth")
    add(mesh_groups.get("outer", []), axial_size, "outer wall")
    add(mesh_groups.get("throat_disc", []), throat_res, "throat")
    add(mesh_groups.get("rear", []), rear_res, "rear")
    add(mesh_groups.get("interface", []), interface_res, "interface")

    front_edge_fallback = _effective_size_mm(front_panel_q)
    back_edge_fallback = _effective_size_mm(back_panel_q)
    add(
        mesh_groups.get("enclosure_edges_front", []),
        front_edge_size or front_edge_fallback,
        "front enclosure edge",
    )
    add(
        mesh_groups.get("enclosure_edges_back", []),
        back_edge_size or back_edge_fallback,
        "back enclosure edge",
    )
    add(front_panels, _effective_size_mm(front_panel_q), "front enclosure panel")
    add(back_panels, _effective_size_mm(back_panel_q), "back enclosure panel")

    enclosure_tags = mesh_groups.get("enclosure", [])
    add(enclosure_tags, _effective_size_mm(front_q + back_q), "enclosure")
    return regions


def _estimate_triangle_count_float(
    regions: list[tuple[float, float, str]],
) -> float:
    total = 0.0
    for area_mm2, size_mm, _label in regions:
        area_mm2 = float(area_mm2)
        size_mm = float(size_mm)
        if size_mm > 0.0 and area_mm2 > 0.0:
            total += TRIANGLES_PER_AREA_OVER_H2 * area_mm2 / (size_mm * size_mm)
    return total


def _enclosure_domain_multiplier(geometry: BuiltGeometry) -> float:
    """Return the mirror multiplier from the meshed sector to full domain."""

    axes = {str(axis).lower() for axis in geometry.symmetry_snap_axes}
    lateral_cut_count = len(axes.intersection({"x", "y"}))
    return float(2**lateral_cut_count)


def effective_triangle_limit(
    geometry: BuiltGeometry, density: MeshDensity
) -> int | None:
    """Actual-domain ceiling corresponding to ``density.max_triangles``.

    ``max_triangles`` is a full-domain-equivalent budget so a quarter mesh and
    its full counterpart are judged by the same geometric complexity.
    """

    if density.max_triangles is None:
        return None
    full_limit = int(density.max_triangles)
    if full_limit <= 0:
        raise ValueError("max_triangles must be > 0 or None")
    return max(1, int(round(full_limit / _enclosure_domain_multiplier(geometry))))


def _generic_triangle_regions(
    mesh_groups: dict[str, list[int]],
    *,
    throat_res: float,
    mouth_res: float,
    rear_res: float,
    interface_res: float,
    aperture_res: float,
) -> list[tuple[float, float, str]]:
    """Area/size regions for non-enclosure pre-mesh cost prediction."""

    regions: list[tuple[float, float, str]] = []
    consumed: set[int] = set()

    def add(surface_tags: list[int], size: float | None, label: str) -> None:
        if size is None or not math.isfinite(float(size)) or float(size) <= 0.0:
            return
        tags = [int(tag) for tag in surface_tags if int(tag) not in consumed]
        consumed.update(tags)
        area = _surface_area_mm2(tags)
        if area > 0.0:
            regions.append((area, float(size), label))

    # Specific semantic groups win over broad wall groups if a builder ever
    # exposes the same surface through more than one role.
    add(mesh_groups.get("throat_disc", []), throat_res, "throat")
    add(mesh_groups.get("mouth_aperture", []), aperture_res, "mouth aperture")
    add(mesh_groups.get("interface", []), interface_res, "interface")
    add(mesh_groups.get("rear", []), rear_res, "rear")
    add(mesh_groups.get("outer", []), rear_res, "outer wall")
    add(mesh_groups.get("mouth", []), mouth_res, "mouth")
    add(
        mesh_groups.get("inner", []),
        _axial_ramp_effective_size_mm(throat_res, mouth_res),
        "waveguide wall",
    )
    return regions


def _record_and_check_triangle_estimate(
    geometry: BuiltGeometry,
    density: MeshDensity,
    regions: list[tuple[float, float, str]],
) -> None:
    estimate = int(round(_estimate_triangle_count_float(regions)))
    multiplier = _enclosure_domain_multiplier(geometry)
    full_estimate = int(round(float(estimate) * multiplier))
    effective_limit = effective_triangle_limit(geometry, density)
    dominant = max(
        regions,
        key=lambda region: (
            TRIANGLES_PER_AREA_OVER_H2 * region[0] / (region[1] * region[1])
        ),
        default=(0.0, 0.0, "unknown region"),
    )
    geometry.metadata.update(
        {
            "meshTriangleEstimate": estimate,
            "meshTriangleEstimateFullDomain": full_estimate,
            "meshDomainMultiplier": multiplier,
            "meshTriangleDominantRegion": dominant[2],
            "meshTriangleDominantTargetMm": dominant[1],
        }
    )
    if density.max_triangles is not None:
        geometry.metadata.update(
            {
                "meshTriangleLimit": int(density.max_triangles),
                "meshEffectiveTriangleLimit": int(effective_limit or 0),
                "meshAllowLarge": bool(density.allow_large_mesh),
            }
        )
    if (
        effective_limit is not None
        and not density.allow_large_mesh
        and estimate > _PREMESH_TRIANGLE_LIMIT_SLACK * effective_limit
    ):
        raise ValueError(
            "estimated mesh size "
            f"{estimate:,} triangles exceeds the effective limit "
            f"{effective_limit:,} by more than the {_PREMESH_TRIANGLE_LIMIT_SLACK:g}x "
            "pre-mesh safety margin; the largest estimated contribution is "
            f"{dominant[2]} at {dominant[1]:g} mm. Increase that mm resolution, "
            "raise max_triangles, or set allow_large_mesh=true explicitly"
        )


def _legacy_mesh_surface_groups(geometry: BuiltGeometry) -> dict[str, list[int]]:
    wall_surfaces = geometry.surface_groups.get(int(PhysicalGroup.RIGID_WALL), [])
    source_surfaces: list[int] = []
    for tag in SOURCE_TAGS:
        source_surfaces.extend(geometry.surface_groups.get(int(tag), []))

    if geometry.source_axis == "z":
        return {
            "inner": list(wall_surfaces),
            "throat_disc": source_surfaces,
        }
    return {
        "rear": list(wall_surfaces),
        "throat_disc": source_surfaces,
    }


def _wall_clearance_chord_mm(radius_mm: Any, wall_mm: float) -> Any:
    """Largest facet chord whose sagitta stays inside the wall's budget.

    A flat facet spanning a chord ``h`` on a surface of local radius ``R``
    departs from that surface by a sagitta ``d``, exactly ``h = 2 sqrt(2 R d -
    d^2)``. Bounding ``d`` by a fraction of the wall therefore bounds the chord,
    and the bound only bites where ``R`` is small -- the throat end of the
    shell -- leaving the mouth end at the size the user asked for.
    """

    sagitta = _WALL_CLEARANCE_FRACTION * float(wall_mm)
    radius = np.asarray(radius_mm, dtype=float)
    return (
        2.0
        * np.sqrt(np.maximum(2.0 * sagitta * radius - sagitta * sagitta, 0.0))
        / _WALL_CLEARANCE_SIZE_OVERSHOOT
    )


def _wall_clearance_axial_ramp(
    ring_radius_mm: Any,
    ring_axial_mm: Any,
    *,
    wall_mm: float,
    rear_res_fallback: float,
) -> tuple[float, float, float]:
    """Fit the cheapest axial ramp that still respects the chord bound.

    Returns ``(base_mm, slope_per_mm, intercept_mm)``; the size field is
    ``max(base, intercept + slope*z)``, capped at rear resolution.

    The bound itself is radial, but a radial size field cannot be used. Two half
    models of the same horn are congruent, and the mesher's parity guard
    requires them to reach the same triangle count; a field varying with x and y
    is evaluated over patches parameterised differently in each half, and Gmsh
    then breaks ties differently. The axial ramp the inner wall already uses
    does not have that problem, because z is what the two halves share.

    ``base`` is the bound at the tightest radius anywhere on the shell, safe at
    every point. The ramp above it must stay under the bound on every ring, so
    the admissible lines are exactly those under the lower convex hull of
    ``(z, bound)``. Which hull edge to extend is then a cost question, not a
    safety one, and the answer is not the obvious one: a horn whose mouth
    roundover folds back in z ends its hull on a near-vertical edge, and
    extending that pins the whole shell at ``base``. So every edge is costed and
    the cheapest wins.
    """

    radius = np.asarray(ring_radius_mm, dtype=float).reshape(-1)
    axial = np.asarray(ring_axial_mm, dtype=float).reshape(-1)
    bound = _wall_clearance_chord_mm(radius, wall_mm)
    # A ring with a straight run has an infinite curvature radius there and so
    # an infinite bound: it constrains nothing, and must not be allowed to
    # decide the fit either. Rings whose bound is not finite are simply dropped.
    finite = np.isfinite(bound) & np.isfinite(axial)
    if not np.any(finite):
        return float(rear_res_fallback), 0.0, float(rear_res_fallback)
    radius, axial, bound = radius[finite], axial[finite], bound[finite]
    base = float(np.min(bound))
    if len(radius) < 2:
        return base, 0.0, base

    order = np.argsort(axial, kind="stable")
    points = np.stack((axial[order], bound[order]), axis=1)
    hull: list[int] = []
    for index in range(len(points)):
        while len(hull) >= 2:
            first, second, third = points[hull[-2]], points[hull[-1]], points[index]
            # Andrew's monotone chain: keep only counter-clockwise turns, which
            # leaves the lower hull.
            cross = (second[0] - first[0]) * (third[1] - first[1]) - (
                second[1] - first[1]
            ) * (third[0] - first[0])
            if cross > 0.0:
                break
            hull.pop()
        hull.append(index)

    # Triangle-count proxy: each ring carries roughly its circumference times
    # its share of the meridian, and a region of size h holds area/h^2 of them.
    # Local meridian step, not its derivative. Differentiating twice weighted
    # the *change* in spacing and skewed the cost on non-uniform ATH sampling.
    step = np.hypot(np.gradient(bound), np.gradient(axial))
    weight = np.maximum(np.abs(step), 1.0e-9)

    def cost(slope: float, intercept: float) -> float:
        size = np.maximum(intercept + slope * axial, base)
        if np.any(size > bound + 1.0e-9):
            return math.inf
        return float(np.sum(weight / np.square(np.maximum(size, 1.0e-9))))

    best = (0.0, base)
    best_cost = cost(0.0, base)
    for position in range(len(hull) - 1):
        start, end = points[hull[position]], points[hull[position + 1]]
        if end[0] <= start[0]:
            continue
        slope = float((end[1] - start[1]) / (end[0] - start[0]))
        if slope <= 0.0:
            continue
        intercept = float(start[1] - slope * start[0])
        candidate = cost(slope, intercept)
        if candidate < best_cost:
            best, best_cost = (slope, intercept), candidate
    return base, best[0], best[1]


def _wall_clearance_size_formula(
    axial_expression: str,
    *,
    rear_res_mm: float,
    base_mm: float,
    slope_per_mm: float,
    intercept_mm: float,
) -> str:
    """The ramp above as a Gmsh MathEval expression.

    Floored at ``base_mm``, the bound at the tightest radius on the shell, which
    is safe at every point. That floor is what covers the throat end and the
    rear return and plate behind it, where the ramp itself dips lower.
    """

    return (
        f"min({float(rear_res_mm):.12g}, max({float(intercept_mm):.12g} + "
        f"({float(slope_per_mm):.12g})*{axial_expression}, {float(base_mm):.12g}))"
    )


def _axis_coordinate_expression(source_axis: str) -> tuple[str, str]:
    axis = str(source_axis or "z").strip().lower()
    sign = "-" if axis.startswith("-") else ""
    axis = axis[1:] if axis[:1] in {"+", "-"} else axis
    if axis not in {"x", "y", "z"}:
        axis = "z"
        sign = ""
    return axis, f"(-{axis})" if sign == "-" else axis


def configure_density(geometry: BuiltGeometry, density: MeshDensity) -> None:
    """Configure waveguide-compatible Gmsh mesh-size fields.

    Role names intentionally mirror the geometry builder: ``inner``/``mouth``
    interpolate throat-to-mouth, free-standing ``outer`` and ``rear`` use rear
    resolution, ``throat_disc`` uses throat resolution, and enclosure groups
    use front/back quadrant interpolation when bounds exist.
    """

    import gmsh

    mesh_groups = geometry.mesh_surface_groups or _legacy_mesh_surface_groups(geometry)
    curve_groups = {
        name: _collect_boundary_curves(surfaces)
        for name, surfaces in mesh_groups.items()
        if surfaces
    }

    throat_res = float(density.throat_res_mm)
    mouth_res = float(density.mouth_res_mm)
    rear_res = float(density.rear_res_mm)
    interface_res = float(density.interface_res_mm or density.mouth_res_mm)
    named_sizes = {
        "throat_res_mm": throat_res,
        "mouth_res_mm": mouth_res,
        "rear_res_mm": rear_res,
        "interface_res_mm": interface_res,
    }
    invalid = [
        name
        for name, value in named_sizes.items()
        if not math.isfinite(value) or value <= 0.0
    ]
    if invalid:
        raise ValueError(
            "mesh resolution values must be finite and > 0: " + ", ".join(invalid)
        )
    aperture_res_scale = float(getattr(density, "aperture_res_scale", 1.0) or 1.0)
    if not math.isfinite(aperture_res_scale) or aperture_res_scale < 1.0:
        aperture_res_scale = 1.0
    aperture_res = mouth_res * aperture_res_scale

    enclosure_resolution_values: list[float] = []
    front_panels: list[int] = []
    back_panels: list[int] = []
    front_q: list[float] = []
    back_q: list[float] = []
    front_panel_q: list[float] = []
    back_panel_q: list[float] = []
    front_edge_size: float | None = None
    back_edge_size: float | None = None

    if geometry.enclosure_bounds:
        bounds = geometry.enclosure_bounds
        z_front = float(bounds["z_front"])
        z_back = float(bounds["z_back"])

        front_q = _parse_quadrant_resolutions(density.enc_front_res_mm, mouth_res)
        back_q = _parse_quadrant_resolutions(density.enc_back_res_mm, mouth_res)
        front_panel_q = list(front_q)
        back_panel_q = list(back_q)
        front_panels, back_panels = _split_enclosure_panel_surfaces(
            mesh_groups.get("enclosure", []),
            front_edge_tags=mesh_groups.get("enclosure_edges_front", []),
            back_edge_tags=mesh_groups.get("enclosure_edges_back", []),
            z_front=z_front,
            z_back=z_back,
        )
        edge_present = float(bounds.get("edge_depth", 0.0) or 0.0) > 0.0
        front_edge_size = _enclosure_edge_size_mm(front_q) if edge_present else None
        back_edge_size = _enclosure_edge_size_mm(back_q) if edge_present else None
        triangle_regions = _enclosure_triangle_regions(
            mesh_groups,
            throat_res=throat_res,
            mouth_res=mouth_res,
            rear_res=rear_res,
            interface_res=interface_res,
            front_q=front_q,
            back_q=back_q,
            front_panel_q=front_panel_q,
            back_panel_q=back_panel_q,
            front_edge_size=front_edge_size,
            back_edge_size=back_edge_size,
            front_panels=front_panels,
            back_panels=back_panels,
        )

        _record_and_check_triangle_estimate(geometry, density, triangle_regions)
    else:
        _record_and_check_triangle_estimate(
            geometry,
            density,
            _generic_triangle_regions(
                mesh_groups,
                throat_res=throat_res,
                mouth_res=mouth_res,
                rear_res=rear_res,
                interface_res=interface_res,
                aperture_res=aperture_res,
            ),
        )

    _axis, coord = _axis_coordinate_expression(geometry.source_axis)
    a0, a1 = geometry.axial_bounds_mm
    span = max(abs(a1 - a0), 1e-9)
    slope = (mouth_res - throat_res) / span
    intercept = throat_res - slope * float(a0)
    # Clamp the throat-to-mouth interpolation so geometry beyond the nominal
    # axial bounds (e.g. R-OSSE rollback) never extrapolates past either size.
    res_lo = min(throat_res, mouth_res)
    res_hi = max(throat_res, mouth_res)
    axial_formula = f"min(max({intercept:.12g} + ({slope:.12g}) * {coord}, {res_lo:.12g}), {res_hi:.12g})"

    fields: list[int] = []

    def add_field(
        formula: str, surfaces: list[int], curves: list[int] | None = None
    ) -> None:
        curves = curves or []
        if not surfaces and not curves:
            return
        base = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(base, "F", formula)
        restrict = gmsh.model.mesh.field.add("Restrict")
        gmsh.model.mesh.field.setNumber(restrict, "InField", base)
        gmsh.model.mesh.field.setNumber(restrict, "IncludeBoundary", 0)
        if surfaces:
            gmsh.model.mesh.field.setNumbers(
                restrict, "SurfacesList", [int(s) for s in surfaces]
            )
        if curves:
            gmsh.model.mesh.field.setNumbers(
                restrict, "CurvesList", [int(c) for c in curves]
            )
        fields.append(restrict)

    for group_key in ("inner", "mouth"):
        add_field(
            axial_formula,
            mesh_groups.get(group_key, []),
            curve_groups.get(group_key, []),
        )
    aperture_surfaces = mesh_groups.get("mouth_aperture", [])
    if aperture_surfaces:
        add_field(f"{aperture_res:.12g}", aperture_surfaces)
        geometry.metadata.update(
            {
                "apertureMeshResolutionScale": float(aperture_res_scale),
                "apertureMeshRimSizeMm": float(mouth_res),
                "apertureMeshInteriorSizeMm": float(aperture_res),
            }
        )

    free_standing_wall_mode = bool(mesh_groups.get("outer")) and not bool(
        mesh_groups.get("enclosure")
    )
    outer_formula = f"{rear_res:.12g}" if free_standing_wall_mode else axial_formula
    rear_boundary_formula: str | None = None
    clearance_target_mm: float | None = None
    clearance = geometry.metadata.get("outerWallClearance")
    if free_standing_wall_mode and clearance:
        wall_mm = float(clearance.get("wallThicknessMm", 0.0) or 0.0)
        min_radius = float(clearance.get("minOuterCurvatureRadiusMm", 0.0) or 0.0)
        ring_radius = clearance.get("ringMinCurvatureRadiusMm") or []
        ring_axial = clearance.get("ringMaxAxialMm") or []
        if wall_mm > 0.0 and min_radius > 0.0 and len(ring_radius) == len(ring_axial) > 0:
            base, slope, intercept = _wall_clearance_axial_ramp(
                ring_radius,
                ring_axial,
                wall_mm=wall_mm,
                rear_res_fallback=rear_res,
            )
            outer_formula = _wall_clearance_size_formula(
                coord,
                rear_res_mm=rear_res,
                base_mm=base,
                slope_per_mm=slope,
                intercept_mm=intercept,
            )
            tightest = min(rear_res, base)
            # The shell and the flat rear cap meet on a rim that is authored
            # once but partly re-created by the planar fill, so the two sides
            # mesh as separate curves and only weld when they ask for the same
            # size -- which they always have, both being rear resolution. Give
            # the cap's boundary the size the shell wants at the rim, as one
            # constant.
            #
            # A constant, not the shell's formula, for two reasons. The rim sits
            # at the shell's smallest radius, so the constant IS the formula
            # there. And on a half model the cap's boundary also runs along the
            # cut plane, straight through the axis, where a radial bound would
            # collapse toward zero and shatter the disc.
            rear_boundary_formula = f"{tightest:.12g}"
            clearance_target_mm = float(tightest)
            geometry.metadata["outerWallClearance"] = {
                **{
                    key: value
                    for key, value in clearance.items()
                    if key not in {"ringMinCurvatureRadiusMm", "ringMaxAxialMm"}
                },
                "clearanceFraction": _WALL_CLEARANCE_FRACTION,
                "sizeOvershoot": _WALL_CLEARANCE_SIZE_OVERSHOOT,
                "requestedRearResolutionMm": float(rear_res),
                "cappedSizeAtMinRadiusMm": float(tightest),
                "capActive": bool(tightest < rear_res),
            }
    add_field(
        outer_formula,
        mesh_groups.get("outer", []),
        curve_groups.get("outer", []),
    )
    add_field(
        f"{throat_res:.12g}",
        mesh_groups.get("throat_disc", []),
        curve_groups.get("throat_disc", []),
    )
    add_field(
        f"{rear_res:.12g}",
        mesh_groups.get("rear", []),
        [] if rear_boundary_formula else curve_groups.get("rear", []),
    )
    if rear_boundary_formula:
        add_field(rear_boundary_formula, [], curve_groups.get("rear", []))
    add_field(
        f"{interface_res:.12g}",
        mesh_groups.get("interface", []),
        curve_groups.get("interface", []),
    )

    if geometry.enclosure_bounds:
        bounds = geometry.enclosure_bounds
        bx0 = float(bounds["bx0"])
        bx1 = float(bounds["bx1"])
        by0 = float(bounds["by0"])
        by1 = float(bounds["by1"])
        z_front = float(bounds["z_front"])
        z_back = float(bounds["z_back"])

        enclosure_resolution_values.extend(front_q)
        enclosure_resolution_values.extend(back_q)
        enclosure_resolution_values.extend(front_panel_q)
        enclosure_resolution_values.extend(back_panel_q)

        front_panel_formula = _panel_bilinear_resolution_formula(
            front_panel_q,
            bx0=bx0,
            bx1=bx1,
            by0=by0,
            by1=by1,
        )
        back_panel_formula = _panel_bilinear_resolution_formula(
            back_panel_q,
            bx0=bx0,
            bx1=bx1,
            by0=by0,
            by1=by1,
        )
        enclosure_formula = _enclosure_resolution_formula(
            front_q,
            back_q,
            bx0=bx0,
            bx1=bx1,
            by0=by0,
            by1=by1,
            z_front=z_front,
            z_back=z_back,
        )

        add_field(
            enclosure_formula,
            mesh_groups.get("enclosure", []),
            curve_groups.get("enclosure", []),
        )
        front_panel_curves = _collect_boundary_curves(front_panels)
        add_field(front_panel_formula, front_panels, front_panel_curves)
        add_field(
            back_panel_formula, back_panels, _collect_boundary_curves(back_panels)
        )

        # Grade a user-requested fine baffle rim into a coarser mouth wall.
        # The threshold is bounded by the two explicit mm targets.
        wall_surfaces = mesh_groups.get("inner", [])
        wall_curves = curve_groups.get("inner", [])
        mouth_rim_size = min(front_panel_q, default=mouth_res)
        if (
            front_panel_curves
            and wall_surfaces
            and mouth_rim_size > 0.0
            and mouth_res > mouth_rim_size
        ):
            distance = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(
                distance, "CurvesList", [int(c) for c in front_panel_curves]
            )
            gmsh.model.mesh.field.setNumber(
                distance, "Sampling", _ENCLOSURE_SEAM_DISTANCE_SAMPLING_MAX
            )
            threshold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMin", mouth_rim_size)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMax", mouth_res)
            gmsh.model.mesh.field.setNumber(threshold, "DistMin", mouth_rim_size)
            gmsh.model.mesh.field.setNumber(
                threshold,
                "DistMax",
                mouth_rim_size
                + (mouth_res - mouth_rim_size) / _ENCLOSURE_SEAM_SIZE_GRADIENT,
            )
            restrict = gmsh.model.mesh.field.add("Restrict")
            gmsh.model.mesh.field.setNumber(restrict, "InField", threshold)
            gmsh.model.mesh.field.setNumber(restrict, "IncludeBoundary", 0)
            gmsh.model.mesh.field.setNumbers(
                restrict, "SurfacesList", [int(s) for s in wall_surfaces]
            )
            if wall_curves:
                gmsh.model.mesh.field.setNumbers(
                    restrict, "CurvesList", [int(c) for c in wall_curves]
                )
            fields.append(restrict)
        if front_edge_size is not None:
            enclosure_resolution_values.append(front_edge_size)
            front_panel_formula = f"{front_edge_size:.12g}"
        if back_edge_size is not None:
            enclosure_resolution_values.append(back_edge_size)
            back_panel_formula = f"{back_edge_size:.12g}"
        add_field(
            front_panel_formula,
            mesh_groups.get("enclosure_edges_front", []),
            curve_groups.get("enclosure_edges_front", []),
        )
        add_field(
            back_panel_formula,
            mesh_groups.get("enclosure_edges_back", []),
            curve_groups.get("enclosure_edges_back", []),
        )

        # Retained enclosure edges are wide enough for the finest touching
        # user target (narrower cosmetic edges are suppressed in mesher.py).
        # Keep bounded grading for unequal adjacent user targets so a direct
        # fine-to-coarse jump cannot recreate the historical needle-fan tear.
        graded_surfaces = mesh_groups.get("enclosure", [])
        graded_curves = curve_groups.get("enclosure", [])
        seam_size_cap = max(
            (
                float(value)
                for value in (*front_q, *back_q, *front_panel_q, *back_panel_q)
                if math.isfinite(float(value)) and float(value) > 0.0
            ),
            default=0.0,
        )
        for edge_size, ring_curves in (
            (front_edge_size, curve_groups.get("enclosure_edges_front", [])),
            (back_edge_size, curve_groups.get("enclosure_edges_back", [])),
        ):
            if edge_size is None or float(edge_size) <= 0.0 or not ring_curves:
                continue
            if seam_size_cap <= float(edge_size):
                continue
            if not graded_surfaces and not graded_curves:
                continue
            # Distance fields sample each curve discretely; keep the sample
            # spacing near the edge size so near-seam distances (and thus
            # sizes) are accurate at the fine end. The ring perimeter bounds
            # every seam curve's length, including single-wire closed rings.
            ring_perimeter = 2.0 * (abs(bx1 - bx0) + abs(by1 - by0))
            sampling = min(
                _ENCLOSURE_SEAM_DISTANCE_SAMPLING_MAX,
                max(
                    _ENCLOSURE_SEAM_DISTANCE_SAMPLING_MIN,
                    math.ceil(ring_perimeter / max(float(edge_size), 0.25)),
                ),
            )
            distance = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(
                distance, "CurvesList", [int(c) for c in ring_curves]
            )
            gmsh.model.mesh.field.setNumber(distance, "Sampling", sampling)
            threshold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMin", float(edge_size))
            gmsh.model.mesh.field.setNumber(threshold, "SizeMax", seam_size_cap)
            gmsh.model.mesh.field.setNumber(threshold, "DistMin", float(edge_size))
            gmsh.model.mesh.field.setNumber(
                threshold,
                "DistMax",
                float(edge_size)
                + (seam_size_cap - float(edge_size)) / _ENCLOSURE_SEAM_SIZE_GRADIENT,
            )
            restrict = gmsh.model.mesh.field.add("Restrict")
            gmsh.model.mesh.field.setNumber(restrict, "InField", threshold)
            gmsh.model.mesh.field.setNumber(restrict, "IncludeBoundary", 0)
            if graded_surfaces:
                gmsh.model.mesh.field.setNumbers(
                    restrict, "SurfacesList", [int(s) for s in graded_surfaces]
                )
            if graded_curves:
                gmsh.model.mesh.field.setNumbers(
                    restrict, "CurvesList", [int(c) for c in graded_curves]
                )
            fields.append(restrict)
    else:
        fallback_formula = f"{mouth_res:.12g}"
        for group_key in (
            "enclosure_sides",
            "enclosure_edges_front",
            "enclosure_edges_back",
            "enclosure_edges",
        ):
            add_field(
                fallback_formula,
                mesh_groups.get(group_key, []),
                curve_groups.get(group_key, []),
            )

    if fields:
        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

    sizes = [throat_res, mouth_res, rear_res, interface_res, aperture_res]
    # The clearance cap is a size this build genuinely asks for, so it belongs
    # in the floor calculation. Left out, Mesh.MeshSizeMin -- derived from the
    # user's resolutions alone -- can clamp the field back above the cap and
    # quietly undo it.
    if clearance_target_mm is not None:
        sizes.append(float(clearance_target_mm))
    sizes.extend(enclosure_resolution_values)
    sizes = [v for v in sizes if math.isfinite(v) and v > 0.0]
    if not sizes:
        sizes = [10.0]
    # MeshSizeMin only clamps field evaluation; it cannot prevent a short CAD
    # curve from requiring an element. Leave optimizer slack without creating
    # any field below a user target, and report realized short edges instead.
    requested_min = min(sizes)
    requested_max = max(sizes)
    min_size = min(
        float(density.min_size_mm) if density.min_size_mm else requested_min * 0.25,
        requested_min,
    )
    max_size = max(
        float(density.max_size_mm) if density.max_size_mm else requested_max,
        requested_max,
    )
    gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
