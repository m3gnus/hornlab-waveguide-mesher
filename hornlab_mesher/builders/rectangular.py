from __future__ import annotations

import math

import numpy as np

from ..geometry import BuiltGeometry, RectHornGeometry
from ..tags import PhysicalGroup
from ._occ import build_bspline_surface_from_rings, make_planar_fill_from_ring, rounded_rect_ring


def _tan_half_angle(deg: float) -> float:
    value = math.tan(math.radians(float(deg)))
    if not math.isfinite(value) or value <= 1e-9:
        raise ValueError("horn half-angles must be finite and > 0")
    return value


def _axis_length(start: float, mouth: float, primary_deg: float, flare2_deg: float, ratio: float) -> float:
    if mouth <= start:
        raise ValueError("mouth dimensions must be larger than the throat")
    if flare2_deg > 0.0:
        junction = start + max(0.05, min(0.95, ratio)) * (mouth - start)
        return (junction - start) / _tan_half_angle(primary_deg) + (mouth - junction) / _tan_half_angle(flare2_deg)
    return (mouth - start) / _tan_half_angle(primary_deg)


def _interp_axis(start: float, mouth: float, t: float, primary_deg: float, flare2_deg: float, ratio: float) -> float:
    if flare2_deg <= 0.0:
        return start + (mouth - start) * t
    junction_t = max(0.05, min(0.95, ratio))
    junction = start + junction_t * (mouth - start)
    if t <= junction_t:
        local = t / junction_t
        return start + (junction - start) * local
    local = (t - junction_t) / (1.0 - junction_t)
    return junction + (mouth - junction) * local


def _throat_axis(start: float, target: float, t: float, length: float, kind: str, driver_deg: float) -> float:
    if length <= 0.0 or t >= 1.0:
        return target
    if kind == "none":
        return target
    m0 = math.tan(math.radians(driver_deg))
    if kind == "quadratic":
        return start + (target - start) * (t * t) + m0 * length * t * (1.0 - t)
    if kind != "osse":
        raise ValueError("throat_type must be 'osse', 'quadratic', or 'none'")
    r2 = start * start + (target * target - start * start) * (t * t) + 2.0 * start * m0 * length * t * (1.0 - t)
    return math.sqrt(max(r2, start * start))


def build_rectangular(geometry: RectHornGeometry) -> BuiltGeometry:
    r0 = float(geometry.throat_diameter_mm) / 2.0
    mouth_h = float(geometry.mouth_width_mm) / 2.0
    mouth_v = float(geometry.mouth_height_mm) / 2.0
    body_h0 = r0
    body_v0 = r0
    body_len = max(
        _axis_length(body_h0, mouth_h, geometry.primary_h_deg, geometry.flare2_h_deg, geometry.flare2_ratio),
        _axis_length(body_v0, mouth_v, geometry.primary_v_deg, geometry.flare2_v_deg, geometry.flare2_ratio),
    )
    throat_len = max(float(geometry.throat_length_mm), 0.0)
    total_len = throat_len + body_len
    n_len = max(int(geometry.n_length), 3)
    n_phi = max(int(geometry.n_phi), 16)

    rings = []
    for j in range(n_len + 1):
        z = total_len * j / n_len
        if z < throat_len and throat_len > 0.0:
            t = z / throat_len
            h = _throat_axis(r0, body_h0, t, throat_len, geometry.throat_type, geometry.throat_driver_deg)
            v = _throat_axis(r0, body_v0, t, throat_len, geometry.throat_type, geometry.throat_driver_deg)
            exponent = 2.0 + 6.0 * t
        else:
            t = 0.0 if body_len <= 0.0 else (z - throat_len) / body_len
            h = _interp_axis(body_h0, mouth_h, t, geometry.primary_h_deg, geometry.flare2_h_deg, geometry.flare2_ratio)
            v = _interp_axis(body_v0, mouth_v, t, geometry.primary_v_deg, geometry.flare2_v_deg, geometry.flare2_ratio)
            exponent = 8.0 if geometry.body_fillet_mm > 0.0 else 14.0
        rings.append(rounded_rect_ring(z=z, half_width=h, half_height=v, exponent=exponent, n_phi=n_phi))

    grid = np.stack(rings, axis=1)
    wall = build_bspline_surface_from_rings(grid)
    throat = make_planar_fill_from_ring(grid[:, 0, :])
    return BuiltGeometry(
        surface_groups={
            int(PhysicalGroup.RIGID_WALL): [tag for _, tag in wall],
            int(PhysicalGroup.PRIMARY_SOURCE): [tag for _, tag in throat],
        },
        axial_bounds_mm=(0.0, float(total_len)),
        source_axis="z",
        mesh_surface_groups={
            "inner": [tag for _, tag in wall],
            "throat_disc": [tag for _, tag in throat],
        },
    )
