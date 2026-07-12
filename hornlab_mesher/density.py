from __future__ import annotations

import math
from typing import Any

from .cost import TRIANGLES_PER_AREA_OVER_H2
from .geometry import BuiltGeometry, MeshDensity
from .profile_common import _parse_number_list
from .tags import PhysicalGroup, SOURCE_TAGS


_ENCLOSURE_DEFAULT_MAX_FREQUENCY_HZ = 20_000.0
_ENCLOSURE_PANEL_EPW = 6.0
_ENCLOSURE_FILLET_ELEMENTS = 3.0
_ENCLOSURE_TRIANGLE_CEILING = 18_000
# Mesh-size growth per millimetre of distance from a roundover seam curve
# (1.0 = size equals distance: each element row roughly doubles). Bounds the
# boundary-to-interior size jump the 2D mesher sees at the seam.
_ENCLOSURE_SEAM_SIZE_GRADIENT = 1.0
_ENCLOSURE_SEAM_DISTANCE_SAMPLING_MIN = 64.0
_ENCLOSURE_SEAM_DISTANCE_SAMPLING_MAX = 2000.0


def _parse_quadrant_resolutions(value: float | str | None, fallback: float) -> list[float]:
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


def _enclosure_frequency_ceiling_mm(density: MeshDensity) -> float:
    frequency = (
        float(density.max_frequency_hz)
        if density.max_frequency_hz and density.max_frequency_hz > 0.0
        else _ENCLOSURE_DEFAULT_MAX_FREQUENCY_HZ
    )
    epw = max(_ENCLOSURE_PANEL_EPW, float(density.elements_per_wavelength), 1.0)
    return (float(density.speed_of_sound_m_s) * 1000.0) / (epw * frequency)


def _clamp_enclosure_panel_sizes(q_values: list[float], density: MeshDensity) -> list[float]:
    ceiling = _enclosure_frequency_ceiling_mm(density)
    return [min(float(value), ceiling) for value in q_values]


def _enclosure_edge_size_mm(
    q_values: list[float],
    density: MeshDensity,
    bounds: dict[str, float],
) -> float | None:
    edge_mm = float(bounds.get("clamped_edge", bounds.get("edge_depth", 0.0)) or 0.0)
    if edge_mm <= 0.0:
        return None
    ceiling = min(
        _enclosure_frequency_ceiling_mm(density),
        edge_mm / _ENCLOSURE_FILLET_ELEMENTS,
    )
    positive_q = [float(value) for value in q_values if math.isfinite(float(value)) and float(value) > 0.0]
    if positive_q:
        ceiling = min(ceiling, min(positive_q))
    return ceiling if math.isfinite(ceiling) and ceiling > 0.0 else None


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
    # so front/back panels never received their frequency-clamped bilinear
    # field and were meshed at the much coarser z-graded enclosure size right
    # up to the fine roundover seams. Stay far below any real front-to-back
    # separation while tolerating the bbox slop.
    eps = max(0.05, z_span * 1.0e-4)
    front: list[int] = []
    back: list[int] = []
    for tag in surface_tags:
        tag_i = int(tag)
        if tag_i in edge_tags:
            continue
        _x0, _y0, z0, _x1, _y1, z1 = gmsh.model.getBoundingBox(2, tag_i)
        if abs(float(z0) - float(z_front)) <= eps and abs(float(z1) - float(z_front)) <= eps:
            front.append(tag_i)
        elif abs(float(z0) - float(z_back)) <= eps and abs(float(z1) - float(z_back)) <= eps:
            back.append(tag_i)
    return front, back


def _effective_size_mm(values: list[float]) -> float | None:
    sizes = [float(value) for value in values if math.isfinite(float(value)) and float(value) > 0.0]
    if not sizes:
        return None
    inv_sq = sum(1.0 / (value * value) for value in sizes)
    return math.sqrt(float(len(sizes)) / inv_sq) if inv_sq > 0.0 else None


def _bbox_surface_area_estimate_mm2(bbox: tuple[float, float, float, float, float, float]) -> float:
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
                area = _bbox_surface_area_estimate_mm2(gmsh.model.getBoundingBox(2, tag))
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
) -> list[tuple[float, float]]:
    regions: list[tuple[float, float]] = []
    consumed: set[int] = set()

    def add(surface_tags: list[int], size: float | None) -> None:
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
            regions.append((area, float(size)))

    axial_size = _effective_size_mm([throat_res, mouth_res])
    add(mesh_groups.get("inner", []), axial_size)
    add(mesh_groups.get("mouth", []), axial_size)
    add(mesh_groups.get("outer", []), axial_size)
    add(mesh_groups.get("throat_disc", []), throat_res)
    add(mesh_groups.get("rear", []), rear_res)
    add(mesh_groups.get("interface", []), interface_res)

    front_edge_fallback = _effective_size_mm(front_panel_q)
    back_edge_fallback = _effective_size_mm(back_panel_q)
    add(mesh_groups.get("enclosure_edges_front", []), front_edge_size or front_edge_fallback)
    add(mesh_groups.get("enclosure_edges_back", []), back_edge_size or back_edge_fallback)
    add(front_panels, _effective_size_mm(front_panel_q))
    add(back_panels, _effective_size_mm(back_panel_q))

    enclosure_tags = mesh_groups.get("enclosure", [])
    add(enclosure_tags, _effective_size_mm(front_q + back_q))
    return regions


def _estimate_triangle_count_float(regions: list[tuple[float, float]]) -> float:
    total = 0.0
    for area_mm2, size_mm in regions:
        area_mm2 = float(area_mm2)
        size_mm = float(size_mm)
        if size_mm > 0.0 and area_mm2 > 0.0:
            total += TRIANGLES_PER_AREA_OVER_H2 * area_mm2 / (size_mm * size_mm)
    return total


def _scaled_values(values: list[float], scale: float) -> list[float]:
    return [float(value) * float(scale) for value in values]


def _enclosure_domain_multiplier(geometry: BuiltGeometry) -> float:
    """Return the mirror multiplier from the meshed enclosure sector to full domain."""

    axes = {str(axis).lower() for axis in geometry.symmetry_snap_axes}
    lateral_cut_count = len(axes.intersection({"x", "y"}))
    return float(2**lateral_cut_count)


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

    # Frequency-aware ceilings: clamp each resolution so the requested band
    # stays resolved at that role's elements-per-wavelength target. The mm
    # knobs still apply wherever they are finer. Roles grade the mesh from
    # the throat (finest) toward the mouth and shadowed rear (coarsest).
    freq_active = density.frequency_ceiling_mm() is not None

    def _sz(value: float, role: str = "") -> float:
        out = float(value)
        ceiling = density.role_ceiling_mm(role) if freq_active else None
        return min(out, ceiling) if ceiling else out

    throat_res = _sz(density.throat_res_mm, "throat")
    mouth_res = _sz(density.mouth_res_mm, "mouth")
    rear_res = _sz(density.rear_res_mm, "rear")
    interface_res = _sz(density.interface_res_mm or density.mouth_res_mm, "interface")
    aperture_res_scale = float(getattr(density, "aperture_res_scale", 1.0) or 1.0)
    if not math.isfinite(aperture_res_scale) or aperture_res_scale < 1.0:
        aperture_res_scale = 1.0
    aperture_res = mouth_res * aperture_res_scale
    if freq_active:
        aperture_ceiling = density.role_ceiling_mm("aperture")
        if aperture_ceiling:
            aperture_res = min(aperture_res, aperture_ceiling)

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
        front_panel_q = _clamp_enclosure_panel_sizes(front_q, density)
        back_panel_q = _clamp_enclosure_panel_sizes(back_q, density)
        front_panels, back_panels = _split_enclosure_panel_surfaces(
            mesh_groups.get("enclosure", []),
            front_edge_tags=mesh_groups.get("enclosure_edges_front", []),
            back_edge_tags=mesh_groups.get("enclosure_edges_back", []),
            z_front=z_front,
            z_back=z_back,
        )
        front_edge_size = _enclosure_edge_size_mm(front_q, density, bounds)
        back_edge_size = _enclosure_edge_size_mm(back_q, density, bounds)
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

        # An explicit acoustic-band request is a correctness constraint, not a
        # cost hint. Never coarsen its throat/source or wall role ceilings to
        # satisfy the default enclosure triangle budget. Without an explicit
        # max frequency the historical cost guard remains useful and may scale
        # all enclosure-build roles to keep accidental meshes bounded.
        if not freq_active:
            pre_cap_estimate_raw = _estimate_triangle_count_float(triangle_regions)
            pre_cap_estimate = int(round(pre_cap_estimate_raw))
            enclosure_domain_multiplier = _enclosure_domain_multiplier(geometry)
            enclosure_domain_fraction = 1.0 / enclosure_domain_multiplier
            effective_triangle_ceiling = max(
                1,
                int(round(float(_ENCLOSURE_TRIANGLE_CEILING) * enclosure_domain_fraction)),
            )
            pre_cap_estimate_full_domain = int(
                round(float(pre_cap_estimate_raw) * enclosure_domain_multiplier)
            )
            pre_cap_estimate_full_domain_raw = pre_cap_estimate_raw * enclosure_domain_multiplier
            if pre_cap_estimate_full_domain_raw > _ENCLOSURE_TRIANGLE_CEILING:
                enclosure_cap_scale = math.sqrt(
                    float(pre_cap_estimate_full_domain_raw) / float(_ENCLOSURE_TRIANGLE_CEILING)
                )
                throat_res *= enclosure_cap_scale
                mouth_res *= enclosure_cap_scale
                rear_res *= enclosure_cap_scale
                interface_res *= enclosure_cap_scale
                front_q = _scaled_values(front_q, enclosure_cap_scale)
                back_q = _scaled_values(back_q, enclosure_cap_scale)
                front_panel_q = _scaled_values(front_panel_q, enclosure_cap_scale)
                back_panel_q = _scaled_values(back_panel_q, enclosure_cap_scale)
                if front_edge_size is not None:
                    front_edge_size *= enclosure_cap_scale
                if back_edge_size is not None:
                    back_edge_size *= enclosure_cap_scale
                post_triangle_regions = [
                    (area, size * enclosure_cap_scale) for area, size in triangle_regions
                ]
                post_cap_estimate_raw = _estimate_triangle_count_float(post_triangle_regions)
                post_cap_estimate = int(round(post_cap_estimate_raw))
                post_cap_estimate_full_domain = int(
                    round(float(post_cap_estimate_raw) * enclosure_domain_multiplier)
                )
                geometry.metadata.update(
                    {
                        "enclosureMeshCapped": True,
                        "enclosureMeshTriangleCeiling": int(_ENCLOSURE_TRIANGLE_CEILING),
                        "enclosureMeshEffectiveTriangleCeiling": int(effective_triangle_ceiling),
                        "enclosureMeshDomainFraction": float(enclosure_domain_fraction),
                        "enclosureMeshDomainMultiplier": float(enclosure_domain_multiplier),
                        "enclosureMeshTriangleEstimatePre": int(pre_cap_estimate),
                        "enclosureMeshTriangleEstimatePreFullDomain": int(
                            pre_cap_estimate_full_domain
                        ),
                        "enclosureMeshTriangleEstimatePost": int(post_cap_estimate),
                        "enclosureMeshTriangleEstimatePostFullDomain": int(
                            post_cap_estimate_full_domain
                        ),
                        "enclosureMeshCapScale": float(enclosure_cap_scale),
                    }
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
    axial_formula = (
        f"min(max({intercept:.12g} + ({slope:.12g}) * {coord}, {res_lo:.12g}), {res_hi:.12g})"
    )

    fields: list[int] = []

    def add_field(formula: str, surfaces: list[int], curves: list[int] | None = None) -> None:
        curves = curves or []
        if not surfaces and not curves:
            return
        base = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(base, "F", formula)
        restrict = gmsh.model.mesh.field.add("Restrict")
        gmsh.model.mesh.field.setNumber(restrict, "InField", base)
        gmsh.model.mesh.field.setNumber(restrict, "IncludeBoundary", 0)
        if surfaces:
            gmsh.model.mesh.field.setNumbers(restrict, "SurfacesList", [int(s) for s in surfaces])
        if curves:
            gmsh.model.mesh.field.setNumbers(restrict, "CurvesList", [int(c) for c in curves])
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

    free_standing_wall_mode = bool(mesh_groups.get("outer")) and not bool(mesh_groups.get("enclosure"))
    outer_formula = f"{rear_res:.12g}" if free_standing_wall_mode else axial_formula
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
        curve_groups.get("rear", []),
    )
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

        add_field(enclosure_formula, mesh_groups.get("enclosure", []), curve_groups.get("enclosure", []))
        front_panel_curves = _collect_boundary_curves(front_panels)
        add_field(front_panel_formula, front_panels, front_panel_curves)
        add_field(back_panel_formula, back_panels, _collect_boundary_curves(back_panels))

        # The frequency-clamped baffle field makes the mouth-hole rim finer
        # than the wall's axial target. With boundary-size extension disabled,
        # grade that fine rim into the wall instead of forcing Gmsh to create a
        # needle fan that degenerate cleanup can tear open. The restriction is
        # wall-only; the panel keeps its ATH-compatible fine hole boundary.
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

        # A thin roundover meshes its strip and seam curves at the edge size
        # (~enc_edge/3 mm) while the adjacent panels and side walls target
        # their own, much coarser sizes right up to that shared boundary
        # (MeshSizeExtendFromBoundary is disabled below). An extreme jump
        # (e.g. enc_edge=1 against 40 mm panels) makes the 2D mesher emit
        # sub-micrometre needle fans along the seam; postprocess then drops
        # the needles as degenerate, tearing the enclosure open along the
        # roundover. Grade the enclosure sizes with a distance threshold from
        # each seam ring so the jump at the boundary stays bounded; the Min
        # background field keeps this inert wherever the neighbourhood is
        # already as fine as the seam.
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
    sizes.extend(enclosure_resolution_values)
    sizes = [v for v in sizes if math.isfinite(v) and v > 0.0]
    if not sizes:
        sizes = [10.0]
    min_size = float(density.min_size_mm) if density.min_size_mm else min(sizes) * 0.5
    max_size = float(density.max_size_mm) if density.max_size_mm else max(sizes) * 1.5
    if freq_active:
        # The global cap must honor the band too: surfaces outside every
        # Restrict field fall back to MeshSizeMax. Use the coarsest role
        # ceiling so the cap does not re-impose the strict target globally.
        ceilings = [
            density.role_ceiling_mm(role)
            for role in ("throat", "mouth", "rear", "interface", "aperture")
        ]
        ceilings = [value for value in ceilings if value]
        if ceilings:
            max_size = min(max_size, max(ceilings))
    gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", max(0, int(density.curvature_segments)))
