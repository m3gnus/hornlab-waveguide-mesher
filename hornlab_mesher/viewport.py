from __future__ import annotations

import logging
import math
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .builders.enclosure import enclosure_box_bounds, sample_enclosure_plan
from .config_builder import _enclosure_from_config, _section, build_geometry_params
from .geometry import HornEnclosure
from .profiles import build_point_grid

logger = logging.getLogger(__name__)


def _reshape_grid(raw: Any, n_phi: int, n_length: int, name: str) -> NDArray[np.float64]:
    arr = np.asarray(raw, dtype=np.float64)
    expected = n_phi * (n_length + 1) * 3
    if arr.size != expected:
        raise ValueError(f"{name} has {arr.size} values; expected {expected}")
    return arr.reshape(n_phi, n_length + 1, 3)


def build_enclosure_viewport_grid(
    inner_points: NDArray[np.float64],
    enclosure: HornEnclosure,
    *,
    closed: bool = True,
    symmetry_planes: tuple[str, ...] = (),
    y_origin_offset_mm: float = 0.0,
) -> dict[str, Any]:
    """Return lightweight enclosure rings for viewport tessellation.

    The output stays in mesher point-grid coordinates: x/y are transverse
    millimetres and z is axial millimetres. It intentionally returns rings
    instead of Gmsh surfaces so callers can use a cheaper viewport tessellator.
    Bounds, roundover clamp, and edge depth come from the same helper the mesh
    build uses (``enclosure_box_bounds``) so the preview cannot drift from the
    mesh; ``y_origin_offset_mm`` is the Mesh.VerticalOffset already applied to
    the placed preview points.
    """

    bounds = enclosure_box_bounds(
        inner_points,
        enclosure,
        closed=closed,
        symmetry_planes=tuple(symmetry_planes),
        y_origin_offset_mm=float(y_origin_offset_mm),
        warn_prefix="viewport ",
    )
    clamped_edge = bounds["clamped_edge"]
    edge_depth = bounds["edge_depth"]
    min_bspline_r = 0.1

    def flatten(points: NDArray[np.float64]) -> list[float]:
        return np.asarray(points, dtype=np.float64).reshape(-1).tolist()

    front_inset = sample_enclosure_plan(
        bx0=bounds["bx0"] + clamped_edge,
        bx1=bounds["bx1"] - clamped_edge,
        by0=bounds["by0"] + clamped_edge,
        by1=bounds["by1"] - clamped_edge,
        corner_radius=min_bspline_r,
        edge_type=int(enclosure.edge_type),
        z=bounds["z_front"],
        plan_type=int(enclosure.plan_type),
        plan_n=float(enclosure.plan_n),
    )

    def make_ring(z: float, radial_t: float) -> NDArray[np.float64]:
        d = clamped_edge * (1.0 - radial_t)
        r = max(min_bspline_r, clamped_edge * radial_t)
        return sample_enclosure_plan(
            bx0=bounds["bx0"] + d,
            bx1=bounds["bx1"] - d,
            by0=bounds["by0"] + d,
            by1=bounds["by1"] - d,
            corner_radius=r,
            edge_type=int(enclosure.edge_type),
            z=z,
            plan_type=int(enclosure.plan_type),
            plan_n=float(enclosure.plan_n),
        )

    profile_rings: list[dict[str, Any]] = [
        {"role": "front_inset", "points": flatten(front_inset)},
    ]
    if edge_depth > 0.0:
        if int(enclosure.edge_type) == 1:
            for t in (0.5, 1.0):
                angle = t * (math.pi / 2.0)
                axial_t = 1.0 - math.cos(angle)
                radial_t = math.sin(angle)
                profile_rings.append(
                    {
                        "role": "front_edge",
                        "points": flatten(
                            make_ring(bounds["z_front"] - axial_t * edge_depth, radial_t)
                        ),
                    }
                )
        else:
            profile_rings.append(
                {
                    "role": "front_edge",
                    "points": flatten(make_ring(bounds["z_front"] - edge_depth, 1.0)),
                }
            )

    z_outer_back = bounds["z_back"] + edge_depth if edge_depth > 0.0 else bounds["z_back"]
    back_outer = sample_enclosure_plan(
        bx0=bounds["bx0"],
        bx1=bounds["bx1"],
        by0=bounds["by0"],
        by1=bounds["by1"],
        corner_radius=clamped_edge,
        edge_type=int(enclosure.edge_type),
        z=z_outer_back,
        plan_type=int(enclosure.plan_type),
        plan_n=float(enclosure.plan_n),
    )
    profile_rings.append({"role": "side_back_outer", "points": flatten(back_outer)})

    if edge_depth > 0.0:
        if int(enclosure.edge_type) == 1:
            for t in (0.5, 1.0):
                angle = t * (math.pi / 2.0)
                axial_t = math.sin(angle)
                radial_t = math.cos(angle)
                profile_rings.append(
                    {
                        "role": "back_edge",
                        "points": flatten(
                            make_ring(bounds["z_back"] + (1.0 - axial_t) * edge_depth, radial_t)
                        ),
                    }
                )
        else:
            profile_rings.append(
                {
                    "role": "back_edge",
                    "points": flatten(make_ring(bounds["z_back"], 0.0)),
                }
            )

    return {
        "mouth_points": inner_points[:, -1, :].reshape(-1).tolist(),
        # Backward-compatible names used by older Waveguide Generator adapters.
        "front_outer_points": flatten(front_inset),
        "back_outer_points": flatten(back_outer),
        "profile_rings": profile_rings,
        "bounds": bounds,
        "plan_type": int(enclosure.plan_type),
        "edge_type": int(enclosure.edge_type),
        "edge_mm": clamped_edge,
        "edge_depth": edge_depth,
    }


def build_viewport_geometry_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build point-grid horn and optional enclosure viewport data from config."""

    params, formula, mode = build_geometry_params(config)
    mesh = _section(config, "mesh")
    enclosure_cfg = _section(config, "enclosure")
    enclosure = _enclosure_from_config(config, mesh, enclosure_cfg)
    grid = build_point_grid(params)

    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    inner_points = _reshape_grid(grid["inner_points"], n_phi, n_length, "inner_points")
    # build_point_grid emits the grid at the origin and reports Mesh.VerticalOffset
    # as metadata; re-apply it here as a rigid +y placement so the preview (grid and
    # enclosure alike) matches the finished mesh. The enclosure is then built from
    # the placed points; enclosure_box_bounds receives the offset so its ATH
    # whole-mm extent rounding happens in the origin frame like the mesh build.
    vertical_offset_mm = float(grid.get("vertical_offset_mm", 0.0) or 0.0)
    if vertical_offset_mm:
        inner_points[:, :, 1] += vertical_offset_mm
        grid = {**grid, "inner_points": inner_points.reshape(-1).tolist()}
        if grid.get("outer_points") is not None:
            outer_points = _reshape_grid(grid["outer_points"], n_phi, n_length, "outer_points")
            outer_points[:, :, 1] += vertical_offset_mm
            grid["outer_points"] = outer_points.reshape(-1).tolist()
    enclosure_grid = None
    if enclosure is not None:
        enclosure_grid = build_enclosure_viewport_grid(
            inner_points,
            enclosure,
            closed=bool(grid.get("full_circle", True)),
            symmetry_planes=tuple(grid.get("symmetry_planes") or ()),
            y_origin_offset_mm=vertical_offset_mm,
        )

    return {
        "params": params,
        "formula": formula,
        "mode": mode,
        "grid": grid,
        "enclosure": enclosure_grid,
    }


__all__ = [
    "build_enclosure_viewport_grid",
    "build_viewport_geometry_from_config",
]
