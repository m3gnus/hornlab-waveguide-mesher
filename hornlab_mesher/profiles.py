from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


_DEFAULTS = {
    "k": 1.0,
    "n": 4.0,
    "q": 0.995,
    "m": 0.85,
    "r": 0.4,
    "b": 0.2,
}

_EVAL_GLOBALS = {
    "__builtins__": {},
    "abs": abs,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "pi": math.pi,
    "e": math.e,
}


def eval_param(value: Any, p: float = 0.0, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float(default)
    try:
        return float(text)
    except ValueError:
        pass
    expr = text.replace("^", "**")
    try:
        return float(eval(expr, _EVAL_GLOBALS, {"p": float(p)}))
    except Exception as exc:
        raise ValueError(f"invalid parameter expression {value!r}") from exc


def _deg(value: Any, p: float = 0.0, default: float = 0.0) -> float:
    return math.radians(eval_param(value, p, default))


def _osse_radius(z: float, p: float, params: Mapping[str, Any], *, r0: float, a_deg: float, a0_deg: float) -> float:
    L = eval_param(params.get("L"), p, 120.0)
    k = eval_param(params.get("k"), p, _DEFAULTS["k"])
    n = eval_param(params.get("n"), p, _DEFAULTS["n"])
    q = eval_param(params.get("q"), p, _DEFAULTS["q"])
    s = eval_param(params.get("s"), p, 0.0)
    a = math.radians(a_deg)
    a0 = math.radians(a0_deg)

    base = math.sqrt((k * r0) ** 2 + 2 * k * r0 * z * math.tan(a0) + (z**2) * (math.tan(a) ** 2))
    base += r0 * (1 - k)
    if z <= 0 or n <= 0 or q <= 0 or L <= 0:
        return base
    z_norm = q * z / L
    if z_norm > 1.0:
        term = s * L / q
    else:
        term = (s * L / q) * (1 - (1 - z_norm**n) ** (1 / n))
    return base + term


def calculate_osse(z: float, p: float, params: Mapping[str, Any]) -> tuple[float, float]:
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
        radius = _osse_radius(z - ext_len - slot_len, p, params, r0=r0_main, a_deg=a_deg, a0_deg=a0_deg)

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


def calculate_rosse(t: float, p: float, params: Mapping[str, Any]) -> tuple[float, float]:
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


def profile_points(params: Mapping[str, Any], n_axial: int, phi: float = 0.0) -> np.ndarray:
    formula = _normalise_formula(params.get("type", "OSSE"))
    t_values = np.linspace(0.0, float(eval_param(params.get("tmax"), phi, 1.0)), int(n_axial))
    points = np.empty((len(t_values), 2), dtype=np.float64)
    if formula == "OSSE":
        total = osse_total_length(params, phi)
        for idx, t in enumerate(t_values):
            points[idx] = calculate_osse(float(t) * total, phi, params)
    else:
        for idx, t in enumerate(t_values):
            points[idx] = calculate_rosse(float(t), phi, params)
    return points


def _normalise_formula(value: Any) -> str:
    raw = str(value or "OSSE").strip().upper().replace("_", "-")
    if raw == "ROSSE":
        raw = "R-OSSE"
    if raw not in {"OSSE", "R-OSSE"}:
        raise ValueError(f"formula must be OSSE or R-OSSE/ROSSE, got {value!r}")
    return raw


def _angle_list(quadrants: str, angular_segments: int) -> tuple[np.ndarray, bool]:
    q = "".join(ch for ch in str(quadrants or "1234") if ch in "1234")
    if not q or q == "1234":
        return np.linspace(0.0, math.tau, int(angular_segments), endpoint=False, dtype=np.float64), True
    spans = {
        "1": (0.0, math.pi / 2.0),
        "12": (0.0, math.pi),
        "14": (-math.pi / 2.0, math.pi / 2.0),
    }
    start, stop = spans.get(q, (0.0, math.tau))
    n = max(2, int(round(int(angular_segments) * abs(stop - start) / math.tau)) + 1)
    return np.linspace(start, stop, n, endpoint=True, dtype=np.float64), False


def _cross_section(params: Mapping[str, Any]) -> tuple[float, float]:
    profile_system = params.get("profileSystem")
    if isinstance(profile_system, Mapping):
        cross = profile_system.get("crossSection")
        if isinstance(cross, Mapping):
            return float(cross.get("exponent", 2.0)), float(cross.get("aspectRatio", 1.0))
    return 2.0, 1.0


def _superellipse_scale(phi: float, exponent: float, aspect_ratio: float) -> float:
    exponent = max(float(exponent), 1.0e-6)
    aspect_ratio = max(float(aspect_ratio), 1.0e-6)
    c = abs(math.cos(phi)) / aspect_ratio
    s = abs(math.sin(phi))
    denom = (c**exponent + s**exponent) ** (1 / exponent)
    return 1.0 / max(denom, 1.0e-12)


def build_point_grid(params: Mapping[str, Any]) -> dict[str, Any]:
    formula = _normalise_formula(params.get("type", "OSSE"))
    n_length = int(params.get("lengthSegments", 32))
    if n_length < 1:
        raise ValueError("lengthSegments must be a positive integer")
    angles, full_circle = _angle_list(str(params.get("quadrants", "1234")), int(params.get("angularSegments", 64)))
    exponent, aspect_ratio = _cross_section(params)
    t_values = np.linspace(0.0, float(eval_param(params.get("tmax"), 0.0, 1.0)), n_length + 1)
    inner = np.empty((len(angles), n_length + 1, 3), dtype=np.float64)
    for i, phi in enumerate(angles):
        scale = _superellipse_scale(float(phi), exponent, aspect_ratio)
        if formula == "OSSE":
            total = osse_total_length(params, float(phi))
            curve = [calculate_osse(float(t) * total, float(phi), params) for t in t_values]
        else:
            curve = [calculate_rosse(float(t), float(phi), params) for t in t_values]
        for j, (z, radius) in enumerate(curve):
            radial = float(radius) * scale
            inner[i, j] = (radial * math.cos(float(phi)), radial * math.sin(float(phi)), float(z))

    outer = None
    wall = float(eval_param(params.get("wallThickness"), 0.0, 0.0))
    enc_depth = float(eval_param(params.get("encDepth"), 0.0, 0.0))
    if enc_depth <= 0.0 and wall > 0.0:
        outer = inner.copy()
        radial = np.linalg.norm(outer[:, :, :2], axis=2)
        scale = (radial + wall) / np.maximum(radial, 1.0e-12)
        outer[:, :, 0] *= scale
        outer[:, :, 1] *= scale
        outer[:, 0, 2] = inner[:, 0, 2] - wall

    return {
        "inner_points": inner.reshape(-1).tolist(),
        "outer_points": None if outer is None else outer.reshape(-1).tolist(),
        "grid_n_phi": int(inner.shape[0]),
        "grid_n_length": int(n_length),
        "full_circle": bool(full_circle),
        "angle_list": angles.tolist(),
        "slice_map": t_values.tolist(),
    }
