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

_ATH_PARITY_T_20 = np.asarray(
    [
        0.0,
        0.031652775,
        0.069285650,
        0.111291038,
        0.158158738,
        0.208217141,
        0.261010634,
        0.315152186,
        0.371049458,
        0.427239696,
        0.483180970,
        0.538366332,
        0.593546216,
        0.647147114,
        0.701376236,
        0.753382922,
        0.804185680,
        0.854976845,
        0.904174233,
        0.953060714,
        1.0,
    ],
    dtype=np.float64,
)

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


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "ath-parity"}
    return bool(value)


def _normalise_ath_angular_segments(raw_count: int) -> int:
    count = max(4, int(round(float(raw_count))))
    if count % 4 == 0:
        return count
    return max(8, int(math.ceil(count / 8.0) * 8))


def _angle_list(quadrants: str, angular_segments: int, *, ath_parity: bool = False) -> tuple[np.ndarray, bool]:
    if ath_parity:
        angular_segments = _normalise_ath_angular_segments(angular_segments)
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


def _ath_parity_slice_map(n_length: int) -> np.ndarray:
    steps = max(1, int(n_length))
    ref_steps = len(_ATH_PARITY_T_20) - 1
    if steps == ref_steps:
        return _ATH_PARITY_T_20.copy()
    out = np.empty(steps + 1, dtype=np.float64)
    out[0] = 0.0
    out[steps] = 1.0
    for j in range(1, steps):
        pos = (j / steps) * ref_steps
        lo = int(math.floor(pos))
        hi = min(ref_steps, lo + 1)
        frac = pos - lo
        out[j] = _ATH_PARITY_T_20[lo] + (_ATH_PARITY_T_20[hi] - _ATH_PARITY_T_20[lo]) * frac
    return out


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


def _normalise3(vec: np.ndarray, fallback: tuple[float, float, float] = (0.0, -1.0, 0.0)) -> np.ndarray:
    length = float(np.linalg.norm(vec))
    if length <= 1.0e-12:
        return np.asarray(fallback, dtype=np.float64)
    return vec / length


def _horn_indices(n_phi: int, n_length: int, *, full_circle: bool) -> np.ndarray:
    indices: list[tuple[int, int, int]] = []
    radial_steps = n_phi if full_circle else max(0, n_phi - 1)
    for j in range(n_length):
        for i in range(radial_steps):
            row1 = j * n_phi
            row2 = (j + 1) * n_phi
            i2 = (i + 1) % n_phi if full_circle else i + 1
            indices.append((row1 + i, row1 + i2, row2 + i2))
            indices.append((row1 + i, row2 + i2, row2 + i))
    return np.asarray(indices, dtype=np.int64)


def _fill_missing_normals(normals: np.ndarray, vertices: np.ndarray, n_phi: int, n_length: int) -> None:
    def has_normal(index: int) -> bool:
        return float(np.linalg.norm(normals[index])) > 1.0e-12

    for index in range(vertices.shape[0]):
        if has_normal(index):
            continue
        row = index // n_phi
        col = index % n_phi
        neighbor_indices: list[int] = []
        if col > 0:
            neighbor_indices.append(index - 1)
        if col < n_phi - 1:
            neighbor_indices.append(index + 1)
        if row > 0:
            neighbor_indices.append(index - n_phi)
        if row < n_length:
            neighbor_indices.append(index + n_phi)

        total = np.zeros(3, dtype=np.float64)
        for neighbor in neighbor_indices:
            if has_normal(neighbor):
                total += normals[neighbor]
        if float(np.linalg.norm(total)) <= 1.0e-12:
            x = vertices[index, 0]
            z = vertices[index, 2]
            total = _normalise3(np.asarray([x, 0.0, z], dtype=np.float64))
        normals[index] = total


def _outer_offset_shell(inner: np.ndarray, wall: float, *, full_circle: bool) -> np.ndarray:
    n_phi, n_cols, _ = inner.shape
    n_length = n_cols - 1
    vertices = np.empty((n_phi * n_cols, 3), dtype=np.float64)
    for j in range(n_cols):
        for i in range(n_phi):
            idx = j * n_phi + i
            vertices[idx] = (inner[i, j, 0], inner[i, j, 2], inner[i, j, 1])

    normals = np.zeros_like(vertices)
    for a, b, c in _horn_indices(n_phi, n_length, full_circle=full_circle):
        ab = vertices[b] - vertices[a]
        ac = vertices[c] - vertices[a]
        normal = np.cross(ab, ac)
        normals[a] += normal
        normals[b] += normal
        normals[c] += normal
    _fill_missing_normals(normals, vertices, n_phi, n_length)

    sample_step = max(1, vertices.shape[0] // 64)
    dot_sum = 0.0
    samples = 0
    for idx in range(0, vertices.shape[0], sample_step):
        x = vertices[idx, 0]
        z = vertices[idx, 2]
        radial_len = math.hypot(float(x), float(z))
        normal_len = float(np.linalg.norm(normals[idx]))
        if radial_len <= 1.0e-9 or normal_len <= 1.0e-12:
            continue
        dot_sum += (normals[idx, 0] / normal_len) * (x / radial_len)
        dot_sum += (normals[idx, 2] / normal_len) * (z / radial_len)
        samples += 1
    offset_sign = -1.0 if samples == 0 or dot_sum < 0.0 else 1.0

    outer_vertices = np.empty_like(vertices)
    for idx, vertex in enumerate(vertices):
        row = idx // n_phi
        normal = _normalise3(normals[idx])
        if row == 0:
            radial_len = math.hypot(float(normal[0]), float(normal[2]))
            rx = normal[0] / radial_len if radial_len > 1.0e-12 else 0.0
            rz = normal[2] / radial_len if radial_len > 1.0e-12 else 0.0
            outer_vertices[idx] = (
                vertex[0] + offset_sign * wall * rx,
                vertex[1],
                vertex[2] + offset_sign * wall * rz,
            )
        else:
            outer_vertices[idx] = vertex + offset_sign * wall * normal

    outer = np.empty_like(inner)
    for j in range(n_cols):
        for i in range(n_phi):
            idx = j * n_phi + i
            outer[i, j] = (outer_vertices[idx, 0], outer_vertices[idx, 2], outer_vertices[idx, 1])
    outer[:, 0, 2] = inner[:, 0, 2] - wall
    return outer


def build_point_grid(params: Mapping[str, Any]) -> dict[str, Any]:
    formula = _normalise_formula(params.get("type", "OSSE"))
    n_length = int(params.get("lengthSegments", 32))
    if n_length < 1:
        raise ValueError("lengthSegments must be a positive integer")
    ath_parity = _is_true(params.get("athParitySampling")) or str(params.get("samplingMode", "")).strip() == "ath-parity"
    angles, full_circle = _angle_list(
        str(params.get("quadrants", "1234")),
        int(params.get("angularSegments", 64)),
        ath_parity=ath_parity,
    )
    exponent, aspect_ratio = _cross_section(params)
    t_max = float(eval_param(params.get("tmax"), 0.0, 1.0))
    if ath_parity:
        t_values = _ath_parity_slice_map(n_length) * t_max
    else:
        t_values = np.linspace(0.0, t_max, n_length + 1)
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
        outer = _outer_offset_shell(inner, wall, full_circle=full_circle)

    return {
        "inner_points": inner.reshape(-1).tolist(),
        "outer_points": None if outer is None else outer.reshape(-1).tolist(),
        "grid_n_phi": int(inner.shape[0]),
        "grid_n_length": int(n_length),
        "full_circle": bool(full_circle),
        "angle_list": angles.tolist(),
        "slice_map": t_values.tolist(),
    }
