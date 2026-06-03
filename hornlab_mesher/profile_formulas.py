from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .profile_common import _DEFAULTS, _deg, _normalise_formula, _osse_radius, eval_param
from .profile_morph import _coverage_angle_from_guiding_curve

def calculate_osse(
    z: float,
    p: float,
    params: Mapping[str, Any],
    *,
    coverage_angle: float | None = None,
) -> tuple[float, float]:
    L = eval_param(params.get("L"), p, 120.0)
    r0_base = eval_param(params.get("r0"), p, 12.7)
    ext_len = max(0.0, eval_param(params.get("throatExtLength"), p, 0.0))
    slot_len = max(0.0, eval_param(params.get("slotLength"), p, 0.0))
    ext_angle = _deg(params.get("throatExtAngle"), p, 0.0)
    r0_main = r0_base + ext_len * math.tan(ext_angle)
    a_deg = eval_param(params.get("a"), p, 60.0)
    a0_deg = eval_param(params.get("a0"), p, 15.5)

    if z <= ext_len:
        radius = r0_base + z * math.tan(ext_angle)
    elif z <= ext_len + slot_len:
        radius = r0_main
    else:
        main_z = z - ext_len - slot_len
        active_a_deg = coverage_angle
        if active_a_deg is None:
            active_a_deg = _coverage_angle_from_guiding_curve(
                p,
                params,
                main_length=L,
                a0_deg=a0_deg,
                r0_main=r0_main,
            )
        if active_a_deg is None:
            active_a_deg = a_deg
        radius = _osse_radius(main_z, p, params, r0=r0_main, a_deg=active_a_deg, a0_deg=a0_deg)

    x = float(z)
    y = float(radius)
    rot_deg = eval_param(params.get("rot"), p, 0.0)
    if math.isfinite(rot_deg) and rot_deg != 0.0:
        rot = math.radians(rot_deg)
        dx = x
        dy = y - r0_base
        x = dx * math.cos(rot) - dy * math.sin(rot)
        y = r0_base + dx * math.sin(rot) + dy * math.cos(rot)
    return x, y


def osse_total_length(params: Mapping[str, Any], p: float = 0.0) -> float:
    return (
        eval_param(params.get("L"), p, 120.0)
        + max(0.0, eval_param(params.get("throatExtLength"), p, 0.0))
        + max(0.0, eval_param(params.get("slotLength"), p, 0.0))
    )


def _rosse_length(params: Mapping[str, Any], p: float) -> float:
    a = _deg(params.get("a"), p, 60.0)
    a0 = _deg(params.get("a0"), p, 15.5)
    k = eval_param(params.get("k"), p, _DEFAULTS["k"])
    r0 = eval_param(params.get("r0"), p, 12.7)
    R = eval_param(params.get("R"), p, 150.0)
    c1 = (k * r0) ** 2
    c2 = 2 * k * r0 * math.tan(a0)
    c3 = math.tan(a) ** 2
    target = R + r0 * (k - 1)
    if abs(c3) < 1.0e-12:
        if abs(c2) < 1.0e-12:
            return 0.0
        return (target**2 - c1) / c2
    discriminant = c2**2 - 4 * c3 * (c1 - target**2)
    if discriminant < 0.0:
        raise ValueError("R is unreachable from r0 with these R-OSSE parameters")
    return (math.sqrt(discriminant) - c2) / (2 * c3)


def rosse_total_length(params: Mapping[str, Any], p: float = 0.0) -> float:
    r0_base = eval_param(params.get("r0"), p, 12.7)
    ext_len = max(0.0, eval_param(params.get("throatExtLength"), p, 0.0))
    slot_len = max(0.0, eval_param(params.get("slotLength"), p, 0.0))
    ext_angle = _deg(params.get("throatExtAngle"), p, 0.0)
    r0_main = r0_base + ext_len * math.tan(ext_angle)
    main_params = {**params, "r0": r0_main}
    return ext_len + slot_len + _rosse_length(main_params, p)


def _calculate_rosse_main(t: float, p: float, params: Mapping[str, Any]) -> tuple[float, float]:
    R = eval_param(params.get("R"), p, 150.0)
    r0 = eval_param(params.get("r0"), p, 12.7)
    k = eval_param(params.get("k"), p, _DEFAULTS["k"])
    q = eval_param(params.get("q"), p, 1.0)
    m = eval_param(params.get("m"), p, _DEFAULTS["m"])
    r = eval_param(params.get("r"), p, _DEFAULTS["r"])
    b = eval_param(params.get("b"), p, _DEFAULTS["b"])
    a = _deg(params.get("a"), p, 60.0)
    a0 = _deg(params.get("a0"), p, 15.5)
    L = _rosse_length(params, p)
    c1 = (k * r0) ** 2
    c2 = 2 * k * r0 * math.tan(a0)
    c3 = math.tan(a) ** 2

    x = L * (math.sqrt(r**2 + m**2) - math.sqrt(r**2 + (t - m) ** 2))
    x += b * L * (math.sqrt(r**2 + (1 - m) ** 2) - math.sqrt(r**2 + m**2)) * (t**2)
    throat_r = math.sqrt(c1 + c2 * L * t + c3 * (L * t) ** 2) + r0 * (1 - k)
    mouth_r = max(0.0, R + L * (1 - math.sqrt(1 + c3 * (t - 1) ** 2)))
    y = (1 - t**q) * throat_r + (t**q) * mouth_r
    return x, y


def calculate_rosse(t: float, p: float, params: Mapping[str, Any]) -> tuple[float, float]:
    r0_base = eval_param(params.get("r0"), p, 12.7)
    ext_len = max(0.0, eval_param(params.get("throatExtLength"), p, 0.0))
    slot_len = max(0.0, eval_param(params.get("slotLength"), p, 0.0))
    ext_angle = _deg(params.get("throatExtAngle"), p, 0.0)
    r0_main = r0_base + ext_len * math.tan(ext_angle)
    main_params = {**params, "r0": r0_main}
    main_length = _rosse_length(main_params, p)

    if ext_len <= 0.0 and slot_len <= 0.0:
        return _calculate_rosse_main(t, p, main_params)

    full_length = ext_len + slot_len + main_length
    if full_length <= 1.0e-12:
        return 0.0, r0_base

    axial_pos = max(0.0, float(t)) * full_length
    if axial_pos <= ext_len:
        return axial_pos, r0_base + axial_pos * math.tan(ext_angle)
    if axial_pos <= ext_len + slot_len:
        return axial_pos, r0_main

    if main_length <= 1.0e-12:
        return ext_len + slot_len, r0_main
    main_t = (axial_pos - ext_len - slot_len) / main_length
    x, y = _calculate_rosse_main(main_t, p, main_params)
    return x + ext_len + slot_len, y


def profile_points(params: Mapping[str, Any], n_axial: int, phi: float = 0.0) -> np.ndarray:
    formula = _normalise_formula(params.get("type", "OSSE"))
    t_max = float(eval_param(params.get("tmax"), phi, 1.0)) if formula == "R-OSSE" else 1.0
    t_values = np.linspace(0.0, t_max, int(n_axial))
    points = np.empty((len(t_values), 2), dtype=np.float64)
    if formula == "OSSE":
        total = osse_total_length(params, phi)
        for idx, t in enumerate(t_values):
            points[idx] = calculate_osse(float(t) * total, phi, params)
    else:
        for idx, t in enumerate(t_values):
            points[idx] = calculate_rosse(float(t), phi, params)
    return points
