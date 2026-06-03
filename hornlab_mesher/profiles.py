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

_ATH_T_20 = np.asarray(
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


def _parse_number_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        try:
            parts = list(value)
        except TypeError:
            return []
    out: list[float] = []
    for part in parts:
        if part == "":
            continue
        out.append(float(eval_param(part, 0.0, 0.0)))
    return out


def _guiding_curve_type(params: Mapping[str, Any], p: float) -> int:
    return int(round(eval_param(params.get("gcurveType"), p, 0.0)))


def _guiding_curve_active(params: Mapping[str, Any], p: float) -> bool:
    return _guiding_curve_type(params, p) in {1, 2} and eval_param(params.get("gcurveWidth"), p, 0.0) > 0.0


def _guiding_curve_target_radius(p: float, params: Mapping[str, Any]) -> float:
    curve_type = _guiding_curve_type(params, p)
    width = eval_param(params.get("gcurveWidth"), p, 0.0)
    if curve_type == 0 or width <= 0.0:
        return 0.0
    aspect = eval_param(params.get("gcurveAspectRatio"), p, 1.0)
    if aspect <= 0.0:
        raise ValueError("gcurveAspectRatio must be positive")

    rotation = math.radians(eval_param(params.get("gcurveRot"), p, 0.0))
    pr = p - rotation
    cos_p = math.cos(pr)
    sin_p = math.sin(pr)

    if curve_type == 1:
        exponent = max(2.0, eval_param(params.get("gcurveSeN"), p, 3.0))
        a = width / 2.0
        b = a * aspect
        term = abs(cos_p / a) ** exponent + abs(sin_p / b) ** exponent
        return term ** (-1.0 / exponent)

    if curve_type != 2:
        raise ValueError(f"unsupported GCurve type {curve_type}")

    sf = _parse_number_list(params.get("gcurveSf", params.get("gcurveSF")))
    if len(sf) >= 6:
        sf_a, sf_b, sf_m1, sf_n1, sf_n2, sf_n3 = sf[:6]
        sf_m2 = sf_m1
    else:
        sf_a = eval_param(params.get("gcurveSfA"), p, 1.0)
        sf_b = eval_param(params.get("gcurveSfB"), p, 1.0)
        sf_m1 = eval_param(params.get("gcurveSfM1"), p, 4.0)
        raw_m2 = params.get("gcurveSfM2")
        sf_m2 = eval_param(raw_m2, p, sf_m1) if raw_m2 is not None else sf_m1
        sf_n1 = eval_param(params.get("gcurveSfN1"), p, 2.0)
        sf_n2 = eval_param(params.get("gcurveSfN2"), p, 2.0)
        sf_n3 = eval_param(params.get("gcurveSfN3"), p, 2.0)
    sf_a = max(abs(sf_a), 1.0e-12)
    sf_b = max(abs(sf_b), 1.0e-12)
    sf_n1 = max(abs(sf_n1), 1.0e-12)
    t1 = abs(math.cos((sf_m1 * pr) / 4.0) / sf_a) ** sf_n2
    t2 = abs(math.sin((sf_m2 * pr) / 4.0) / sf_b) ** sf_n3
    r_norm = (t1 + t2) ** (-1.0 / sf_n1)
    sx = width / 2.0
    sy = sx * aspect
    return math.hypot(r_norm * cos_p * sx, r_norm * sin_p * sy)


def _invert_osse_coverage_angle(
    target_radius: float,
    z_main: float,
    p: float,
    params: Mapping[str, Any],
    *,
    a0_deg: float,
    r0_main: float,
) -> float:
    low = 0.5
    high = 89.0
    for _ in range(24):
        mid = 0.5 * (low + high)
        radius = _osse_radius(z_main, p, params, r0=r0_main, a_deg=mid, a0_deg=a0_deg)
        if radius < target_radius:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _coverage_angle_from_guiding_curve(
    p: float,
    params: Mapping[str, Any],
    *,
    main_length: float,
    a0_deg: float,
    r0_main: float,
) -> float | None:
    if not _guiding_curve_active(params, p):
        return None
    target_radius = _guiding_curve_target_radius(p, params)
    if target_radius <= 0.0:
        return None
    dist = eval_param(params.get("gcurveDist"), p, 1.0)
    target_z = main_length * dist if 0.0 < dist <= 1.0 else dist
    if target_z <= 0.0 or not math.isfinite(target_z):
        target_z = main_length
    target_z = min(main_length, target_z)
    return _invert_osse_coverage_angle(
        target_radius,
        target_z,
        p,
        params,
        a0_deg=a0_deg,
        r0_main=r0_main,
    )


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
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalise_ath_angular_segments(raw_count: int) -> int:
    count = max(4, int(round(float(raw_count))))
    if count % 4 == 0:
        return count
    return max(8, int(math.ceil(count / 8.0) * 8))


def _morph_target_shape(params: Mapping[str, Any], p: float) -> int:
    return int(round(eval_param(params.get("morphTarget"), p, 0.0)))


def _morph_active(params: Mapping[str, Any], p: float) -> bool:
    return _morph_target_shape(params, p) in {1, 2}


def _rounded_rect_radius(phi: float, half_width: float, half_height: float, corner_radius: float) -> float:
    abs_cos = abs(math.cos(phi))
    abs_sin = abs(math.sin(phi))
    if abs_cos < 1.0e-9:
        return half_height
    if abs_sin < 1.0e-9:
        return half_width

    r = min(max(corner_radius, 0.0), half_width, half_height)
    if r <= 1.0e-9:
        return min(half_width / abs_cos, half_height / abs_sin)

    y_at_x = (half_width * abs_sin) / abs_cos
    if y_at_x <= half_height - r + 1.0e-9:
        return half_width / abs_cos
    x_at_y = (half_height * abs_cos) / abs_sin
    if x_at_y <= half_width - r + 1.0e-9:
        return half_height / abs_sin

    cx = half_width - r
    cy = half_height - r
    b = -2.0 * (abs_cos * cx + abs_sin * cy)
    c = cx * cx + cy * cy - r * r
    disc = max(0.0, b * b - 4.0 * c)
    return (-b + math.sqrt(disc)) / 2.0


def _configured_morph_half_dimension(
    value: Any,
    phi: float,
    *,
    fallback_radius: float,
) -> float:
    dimension = eval_param(value, phi, 0.0)
    if dimension <= 0.0:
        return max(0.0, float(fallback_radius))
    return dimension / 2.0


def _circle_morph_target_radius(current_radius: float, phi: float, params: Mapping[str, Any]) -> float:
    half_width = _configured_morph_half_dimension(params.get("morphWidth"), phi, fallback_radius=current_radius)
    half_height = _configured_morph_half_dimension(params.get("morphHeight"), phi, fallback_radius=current_radius)
    return max(half_width, half_height)


def _morph_target_radius_at_angle(current_radius: float, phi: float, params: Mapping[str, Any]) -> float:
    target = _morph_target_shape(params, phi)
    if target == 0:
        return current_radius
    if target == 2:
        return _circle_morph_target_radius(current_radius, phi, params)
    if target != 1:
        raise ValueError(f"unsupported Morph target {target}")
    half_width = _configured_morph_half_dimension(params.get("morphWidth"), phi, fallback_radius=current_radius)
    half_height = _configured_morph_half_dimension(params.get("morphHeight"), phi, fallback_radius=current_radius)
    corner = eval_param(params.get("morphCorner"), phi, 0.0)
    return _rounded_rect_radius(phi, half_width, half_height, corner)


def _morph_factor(
    t: float,
    phi: float,
    params: Mapping[str, Any],
    *,
    morph_start: float | None = None,
) -> float:
    if not _morph_active(params, phi):
        return 0.0
    if morph_start is None:
        morph_start = eval_param(params.get("morphFixed"), phi, 0.0)
    if t <= morph_start:
        return 0.0
    rate = eval_param(params.get("morphRate"), phi, 3.0)
    denom = max(1.0e-9, 1.0 - morph_start)
    return min(1.0, max(0.0, (t - morph_start) / denom)) ** rate


def _apply_morphing(
    current_radius: float,
    mouth_radius: float,
    t: float,
    phi: float,
    params: Mapping[str, Any],
    *,
    morph_start: float | None = None,
) -> float:
    factor = _morph_factor(t, phi, params, morph_start=morph_start)
    if factor <= 0.0:
        return current_radius
    # OS-SE morphing is a directional target-mouth rule:
    # rm(z, phi) = r(z, phi) + f(z) * (rM(phi) - r(L, phi)).
    target_radius = _morph_target_radius_at_angle(mouth_radius, phi, params)
    allow_shrinkage = _is_true(params.get("morphAllowShrinkage"))
    safe_target = target_radius if allow_shrinkage else max(mouth_radius, target_radius)
    return current_radius + (safe_target - mouth_radius) * factor


def _rounded_rect_quadrant_angles(
    points_per_quadrant: int,
    half_width: float,
    half_height: float,
    corner_radius: float,
    corner_segments: int,
) -> np.ndarray:
    points_per_quadrant = max(1, int(points_per_quadrant))
    corner_radius = min(max(float(corner_radius), 0.0), half_width, half_height)
    if corner_radius <= 1.0e-9:
        return np.linspace(0.0, math.pi / 2.0, points_per_quadrant + 1, dtype=np.float64)

    theta1 = math.atan2(half_height - corner_radius, half_width)
    theta2 = math.atan2(half_height, half_width - corner_radius)
    corner_segments = max(0, int(corner_segments))
    side_segments = max(1, points_per_quadrant - corner_segments - (1 if corner_segments > 0 else 0))
    side1_segments = max(1, int(round(side_segments * theta1 / max(theta1 + (math.pi / 2.0 - theta2), 1.0e-12))))
    side2_segments = max(1, side_segments - side1_segments)

    angles: list[float] = []
    for i in range(side1_segments + 1):
        angles.append(theta1 * i / side1_segments)
    cx = half_width - corner_radius
    cy = half_height - corner_radius
    for i in range(1, corner_segments + 1):
        u = i / (corner_segments + 1)
        corner_phi = u * math.pi / 2.0
        angles.append(math.atan2(cy + corner_radius * math.sin(corner_phi), cx + corner_radius * math.cos(corner_phi)))
    angles.append(theta2)
    for i in range(1, side2_segments + 1):
        angles.append(theta2 + (math.pi / 2.0 - theta2) * i / side2_segments)
    return np.asarray(angles, dtype=np.float64)


def _mirror_quadrant_angles(q1: np.ndarray) -> np.ndarray:
    q = [float(v) for v in q1]
    full: list[float] = []
    full.extend(q)
    full.extend(math.pi - v for v in reversed(q[:-1]))
    full.extend(math.pi + v for v in q[1:])
    full.extend(math.tau - v for v in reversed(q[1:-1]))
    return np.asarray(full, dtype=np.float64)


def _morph_angle_list(params: Mapping[str, Any], angular_segments: int) -> np.ndarray | None:
    if not _morph_active(params, 0.0) or _morph_target_shape(params, 0.0) != 1:
        return None
    width = eval_param(params.get("morphWidth"), 0.0, 0.0)
    height = eval_param(params.get("morphHeight"), 0.0, 0.0)
    if width <= 0.0 or height <= 0.0:
        return None
    half_width = width / 2.0
    half_height = height / 2.0
    corner = eval_param(params.get("morphCorner"), 0.0, 0.0)
    configured_corner_segments = max(0, int(round(eval_param(params.get("cornerSegments"), 0.0, 0.0))))
    points_per_quadrant = max(1, int(round(angular_segments / 4.0)) + configured_corner_segments)
    internal_corner_segments = max(0, configured_corner_segments + 1 if corner > 0.0 else 0)
    return _mirror_quadrant_angles(
        _rounded_rect_quadrant_angles(
            points_per_quadrant,
            half_width,
            half_height,
            corner,
            internal_corner_segments,
        )
    )


def _angle_list(params: Mapping[str, Any]) -> tuple[np.ndarray, bool]:
    quadrants = str(params.get("quadrants", "1234"))
    angular_segments = _normalise_ath_angular_segments(int(params.get("angularSegments", 64)))
    morphed_full = _morph_angle_list(params, angular_segments)
    q = "".join(ch for ch in str(quadrants or "1234") if ch in "1234")
    if morphed_full is not None:
        if not q or q == "1234":
            return morphed_full, True
        if q == "1":
            return morphed_full[morphed_full <= math.pi / 2.0 + 1.0e-12], False
        if q == "12":
            return morphed_full[morphed_full <= math.pi + 1.0e-12], False
        if q == "14":
            selected = morphed_full[(morphed_full <= math.pi / 2.0 + 1.0e-12) | (morphed_full >= 3.0 * math.pi / 2.0 - 1.0e-12)]
            selected = np.where(selected > math.pi, selected - math.tau, selected)
            return np.sort(selected), False
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


_ATH_T_9 = np.asarray(
    [
        0.0,
        0.038238500,
        0.114045714,
        0.239636857,
        0.417665786,
        0.620386214,
        0.792462929,
        0.908557000,
        0.973433571,
        1.0,
    ],
    dtype=np.float64,
)


def _ath_default_zmap(n_length: int) -> np.ndarray:
    steps = max(1, int(n_length))
    if steps == len(_ATH_T_9) - 1:
        return _ATH_T_9.copy()
    ref_steps = len(_ATH_T_20) - 1
    if steps == ref_steps:
        return _ATH_T_20.copy()
    out = np.empty(steps + 1, dtype=np.float64)
    out[0] = 0.0
    out[steps] = 1.0
    for j in range(1, steps):
        pos = (j / steps) * ref_steps
        lo = int(math.floor(pos))
        hi = min(ref_steps, lo + 1)
        frac = pos - lo
        out[j] = _ATH_T_20[lo] + (_ATH_T_20[hi] - _ATH_T_20[lo]) * frac
    return out


def _normalise_sampling_mode(value: Any, *, ath_parity_sampling: Any = None, z_map_points: Any = None) -> str:
    if _is_true(ath_parity_sampling):
        return "ath-default-zmap"
    raw = str(value or "").strip().lower().replace("_", "-")
    if not raw:
        return "zmap" if z_map_points is not None else "uniform"
    if raw in {"uniform", "linear", "canonical", "default"}:
        return "uniform"
    if raw in {"ath", "ath-parity", "ath-zmap", "ath-default", "ath-default-zmap", "default-zmap"}:
        return "ath-default-zmap"
    if raw in {"zmap", "z-map", "custom", "custom-zmap", "custom-z-map"}:
        return "zmap"
    raise ValueError(f"samplingMode must be uniform, ath-default-zmap, or zmap, got {value!r}")


def _zmap_number_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        parts: list[Any] = [part.strip() for part in value.replace(";", ",").split(",")]
    else:
        try:
            parts = list(value)
        except TypeError:
            return []

    out: list[float] = []
    for part in parts:
        if isinstance(part, (list, tuple)):
            out.extend(_zmap_number_list(part))
            continue
        if part == "":
            continue
        out.append(float(eval_param(part, 0.0, 0.0)))
    return out


def _custom_zmap(n_length: int, z_map_points: Any) -> np.ndarray:
    steps = max(1, int(n_length))
    values = _zmap_number_list(z_map_points)
    if not values:
        raise ValueError("zmap sampling requires zMapPoints/Mesh.ZMapPoints")

    if len(values) == steps + 1 and math.isclose(values[0], 0.0, abs_tol=1.0e-12) and math.isclose(values[-1], 1.0, abs_tol=1.0e-12):
        out = np.asarray(values, dtype=np.float64)
    else:
        if len(values) % 2 != 0:
            raise ValueError("zMapPoints must be x,y control-point pairs or a full n+1 sample map")
        controls = [(float(values[i]), float(values[i + 1])) for i in range(0, len(values), 2)]
        controls = [(0.0, 0.0), *controls, (1.0, 1.0)]
        xs = np.asarray([item[0] for item in controls], dtype=np.float64)
        ys = np.asarray([item[1] for item in controls], dtype=np.float64)
        if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
            raise ValueError("zMapPoints must contain finite values")
        if np.any(xs < -1.0e-12) or np.any(xs > 1.0 + 1.0e-12):
            raise ValueError("zMapPoints x values must be within 0..1")
        if np.any(ys < -1.0e-12) or np.any(ys > 1.0 + 1.0e-12):
            raise ValueError("zMapPoints y values must be within 0..1")
        if np.any(np.diff(xs) <= 1.0e-12):
            raise ValueError("zMapPoints x values must be strictly increasing")
        if np.any(np.diff(ys) < -1.0e-12):
            raise ValueError("zMapPoints y values must be non-decreasing")
        out = np.interp(np.linspace(0.0, 1.0, steps + 1, dtype=np.float64), xs, ys)

    if not np.all(np.isfinite(out)):
        raise ValueError("zMapPoints must produce finite samples")
    if len(out) != steps + 1:
        raise ValueError(f"zMapPoints produced {len(out)} samples; expected {steps + 1}")
    if np.any(np.diff(out) < -1.0e-12):
        raise ValueError("zMapPoints samples must be non-decreasing")
    out[0] = 0.0
    out[-1] = 1.0
    return out


def _axial_sample_map(n_length: int, params: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    z_map_points = params.get("zMapPoints", params.get("zmapPoints", params.get("ZMapPoints")))
    mode = _normalise_sampling_mode(
        params.get("samplingMode", params.get("sampling_mode")),
        ath_parity_sampling=params.get("athParitySampling", params.get("ath_parity_sampling")),
        z_map_points=z_map_points,
    )
    if mode == "uniform":
        return np.linspace(0.0, 1.0, max(1, int(n_length)) + 1, dtype=np.float64), mode
    if mode == "ath-default-zmap":
        return _ath_default_zmap(n_length), mode
    if mode == "zmap":
        return _custom_zmap(n_length, z_map_points), mode
    raise AssertionError(f"unhandled sampling mode {mode!r}")


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
    angles, full_circle = _angle_list(params)
    exponent, aspect_ratio = _cross_section(params)
    t_max = float(eval_param(params.get("tmax"), 0.0, 1.0))
    t_unit_values, sampling_mode = _axial_sample_map(n_length, params)
    t_values = t_unit_values * t_max
    configured_morph_start = eval_param(params.get("morphFixed"), 0.0, 0.0)
    morph_start_idx = int(np.searchsorted(t_values, configured_morph_start, side="left"))
    if morph_start_idx >= len(t_values):
        snapped_morph_start = float(t_values[-1])
    else:
        snapped_morph_start = float(t_values[morph_start_idx])
    inner = np.empty((len(angles), n_length + 1, 3), dtype=np.float64)
    for i, phi in enumerate(angles):
        scale = _superellipse_scale(float(phi), exponent, aspect_ratio)
        shrink_coverage = None
        if formula == "OSSE" and _morph_active(params, float(phi)) and _is_true(params.get("morphAllowShrinkage")):
            total = osse_total_length(params, float(phi))
            mouth_z, mouth_radius = calculate_osse(total, float(phi), params)
            target_radius = _morph_target_radius_at_angle(float(mouth_radius) * scale, float(phi), params)
            if target_radius < float(mouth_radius) * scale:
                ext_len = max(0.0, eval_param(params.get("throatExtLength"), float(phi), 0.0))
                slot_len = max(0.0, eval_param(params.get("slotLength"), float(phi), 0.0))
                ext_angle = _deg(params.get("throatExtAngle"), float(phi), 0.0)
                r0_base = eval_param(params.get("r0"), float(phi), 12.7)
                r0_main = r0_base + ext_len * math.tan(ext_angle)
                a0_deg = eval_param(params.get("a0"), float(phi), 15.5)
                shrink_coverage = _invert_osse_coverage_angle(
                    target_radius / max(scale, 1.0e-12),
                    float(mouth_z) - ext_len - slot_len,
                    float(phi),
                    params,
                    a0_deg=a0_deg,
                    r0_main=r0_main,
                )
        if formula == "OSSE":
            total = osse_total_length(params, float(phi))
            curve = [
                calculate_osse(float(t) * total, float(phi), params, coverage_angle=shrink_coverage)
                for t in t_values
            ]
        else:
            curve = [calculate_rosse(float(t), float(phi), params) for t in t_values]
        mouth_radial = float(curve[-1][1]) * scale
        for j, (z, radius) in enumerate(curve):
            radial = float(radius) * scale
            if shrink_coverage is None:
                radial = _apply_morphing(
                    radial,
                    mouth_radial,
                    float(t_values[j]),
                    float(phi),
                    params,
                    morph_start=snapped_morph_start,
                )
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
        "sampling_mode": sampling_mode,
    }
