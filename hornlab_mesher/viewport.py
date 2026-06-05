from __future__ import annotations

import logging
import math
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .builders.enclosure import sample_enclosure_plan
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


def _enclosure_bounds(
    inner_points: NDArray[np.float64],
    enclosure: HornEnclosure,
    *,
    closed: bool,
) -> dict[str, float]:
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
            "[hornlab-mesher] viewport enc_depth (%.2f mm) < horn length (%.2f mm) + margin (%.2f mm); "
            "clamping to %.2f mm.",
            enc_depth,
            horn_length,
            enclosure.depth_margin_mm,
            min_enc_depth,
        )
        enc_depth = min_enc_depth
    z_back = z_front - enc_depth

    bx0 = 0.0 if not closed and x_min >= -1.0e-6 else x_min - float(enclosure.space_l_mm)
    bx1 = x_max + float(enclosure.space_r_mm)
    by0 = 0.0 if not closed and y_min >= -1.0e-6 else y_min - float(enclosure.space_b_mm)
    by1 = y_max + float(enclosure.space_t_mm)

    return {
        "bx0": bx0,
        "bx1": bx1,
        "by0": by0,
        "by1": by1,
        "z_front": z_front,
        "z_back": z_back,
        "cx": 0.5 * (bx0 + bx1),
        "cy": 0.5 * (by0 + by1),
    }


def build_enclosure_viewport_grid(
    inner_points: NDArray[np.float64],
    enclosure: HornEnclosure,
    *,
    closed: bool = True,
) -> dict[str, Any]:
    """Return lightweight enclosure rings for viewport tessellation.

    The output stays in mesher point-grid coordinates: x/y are transverse
    millimetres and z is axial millimetres. It intentionally returns rings
    instead of Gmsh surfaces so callers can use a cheaper viewport tessellator.
    """

    bounds = _enclosure_bounds(inner_points, enclosure, closed=closed)
    half_w = 0.5 * (bounds["bx1"] - bounds["bx0"])
    half_h = 0.5 * (bounds["by1"] - bounds["by0"])
    margin_edge_limit = max(
        0.0,
        min(
            float(enclosure.space_l_mm),
            float(enclosure.space_t_mm),
            float(enclosure.space_r_mm),
            float(enclosure.space_b_mm),
        ),
    )
    clamped_edge = max(
        0.0,
        min(float(enclosure.edge_mm), margin_edge_limit, half_w - 0.1, half_h - 0.1),
    )
    enc_depth = bounds["z_front"] - bounds["z_back"]
    edge_depth = min(clamped_edge, max(0.0, enc_depth * 0.5))
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
    enclosure_grid = None
    if enclosure is not None:
        enclosure_grid = build_enclosure_viewport_grid(
            inner_points,
            enclosure,
            closed=bool(grid.get("full_circle", True)),
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
