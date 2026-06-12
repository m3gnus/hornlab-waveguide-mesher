from __future__ import annotations

import math
from typing import Any

from .geometry import BuiltGeometry, MeshDensity
from .tags import PhysicalGroup, SOURCE_TAGS


def _parse_number_list(text: Any) -> list[float]:
    if text is None or not str(text).strip():
        return []
    try:
        return [float(part.strip()) for part in str(text).split(",") if part.strip()]
    except ValueError:
        return []


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

    parts = _parse_number_list(text)
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

    # Frequency-aware ceiling: clamp every resolution so the requested band
    # stays resolved at the configured elements-per-wavelength target. The mm
    # knobs still apply wherever they are finer.
    freq_ceiling = density.frequency_ceiling_mm()

    def _sz(value: float) -> float:
        out = float(value)
        return min(out, freq_ceiling) if freq_ceiling else out

    throat_res = _sz(density.throat_res_mm)
    mouth_res = _sz(density.mouth_res_mm)
    rear_res = _sz(density.rear_res_mm)
    interface_res = _sz(density.interface_res_mm or density.mouth_res_mm)

    coord = {"x": "x", "y": "y", "z": "z"}[geometry.source_axis]
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

    enclosure_resolution_values: list[float] = []
    if geometry.enclosure_bounds:
        bounds = geometry.enclosure_bounds
        bx0 = float(bounds["bx0"])
        bx1 = float(bounds["bx1"])
        by0 = float(bounds["by0"])
        by1 = float(bounds["by1"])
        z_front = float(bounds["z_front"])
        z_back = float(bounds["z_back"])

        front_q = [_sz(v) for v in _parse_quadrant_resolutions(density.enc_front_res_mm, mouth_res)]
        back_q = [_sz(v) for v in _parse_quadrant_resolutions(density.enc_back_res_mm, mouth_res)]
        enclosure_resolution_values.extend(front_q)
        enclosure_resolution_values.extend(back_q)

        front_panel_formula = _panel_bilinear_resolution_formula(
            front_q,
            bx0=bx0,
            bx1=bx1,
            by0=by0,
            by1=by1,
        )
        back_panel_formula = _panel_bilinear_resolution_formula(
            back_q,
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

    sizes = [throat_res, mouth_res, rear_res, interface_res]
    sizes.extend(enclosure_resolution_values)
    sizes = [v for v in sizes if math.isfinite(v) and v > 0.0]
    if not sizes:
        sizes = [10.0]
    min_size = float(density.min_size_mm) if density.min_size_mm else min(sizes) * 0.5
    max_size = float(density.max_size_mm) if density.max_size_mm else max(sizes) * 1.5
    if freq_ceiling:
        # The global cap must honor the band too: surfaces outside every
        # Restrict field fall back to MeshSizeMax.
        max_size = min(max_size, freq_ceiling)
    gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", max(0, int(density.curvature_segments)))
