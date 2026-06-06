from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .profile_common import _is_true, _normalise_formula, eval_param
from .profile_formulas import calculate_osse, calculate_rosse, osse_total_length
from .profile_morph import (
    _apply_morphing,
    _guiding_curve_type,
    _guiding_curve_active,
    _morph_active,
    _morph_target_shape,
    _rounded_rect_quadrant_angles,
)

def _normalise_ath_angular_segments(raw_count: int) -> int:
    count = max(4, int(round(float(raw_count))))
    if count % 4 == 0:
        return count
    return max(8, int(math.ceil(count / 8.0) * 8))



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
    curve_type = _guiding_curve_type(params, 0.0)
    if curve_type not in {0, 1, 2}:
        raise ValueError(f"unsupported GCurve type {curve_type}")
    if formula == "R-OSSE" and _guiding_curve_active(params, 0.0):
        raise ValueError("guiding curves are only supported with formula OSSE")
    n_length = int(params.get("lengthSegments", 32))
    if n_length < 1:
        raise ValueError("lengthSegments must be a positive integer")
    angles, full_circle = _angle_list(params)
    exponent, aspect_ratio = _cross_section(params)
    t_max = float(eval_param(params.get("tmax"), 0.0, 1.0)) if formula == "R-OSSE" else 1.0
    t_unit_values, sampling_mode = _axial_sample_map(n_length, params)
    t_values = t_unit_values * t_max
    configured_morph_start = eval_param(params.get("morphFixed"), 0.0, 0.0)
    morph_start_idx = int(np.searchsorted(t_values, configured_morph_start, side="left"))
    if morph_start_idx >= len(t_values):
        snapped_morph_start = float(t_values[-1])
    else:
        snapped_morph_start = float(t_values[morph_start_idx])
    raw_radials = np.empty((len(angles), n_length + 1), dtype=np.float64)
    z_values = np.empty((len(angles), n_length + 1), dtype=np.float64)
    for i, phi in enumerate(angles):
        scale = _superellipse_scale(float(phi), exponent, aspect_ratio)
        if formula == "OSSE":
            total = osse_total_length(params, float(phi))
            h_bulge = eval_param(params.get("h"), float(phi), 0.0)
            curve = [
                (
                    z,
                    radius + h_bulge * math.sin(float(t_unit) * math.pi),
                )
                for t_unit, (z, radius) in zip(
                    t_unit_values,
                    (
                        calculate_osse(float(t) * total, float(phi), params)
                        for t in t_values
                    ),
                )
            ]
        else:
            curve = [calculate_rosse(float(t), float(phi), params) for t in t_values]
        for j, (z, radius) in enumerate(curve):
            raw_radials[i, j] = float(radius) * scale
            z_values[i, j] = float(z)

    implicit_half_widths = np.max(np.abs(raw_radials * np.cos(angles)[:, None]), axis=0)
    implicit_half_heights = np.max(np.abs(raw_radials * np.sin(angles)[:, None]), axis=0)

    inner = np.empty((len(angles), n_length + 1, 3), dtype=np.float64)
    for i, phi in enumerate(angles):
        mouth_radial = float(raw_radials[i, -1])
        for j in range(n_length + 1):
            radial = float(raw_radials[i, j])
            radial = _apply_morphing(
                radial,
                mouth_radial,
                float(t_values[j]),
                float(phi),
                params,
                morph_start=snapped_morph_start,
                implicit_half_width=float(implicit_half_widths[j]),
                implicit_half_height=float(implicit_half_heights[j]),
            )
            inner[i, j] = (
                radial * math.cos(float(phi)),
                radial * math.sin(float(phi)),
                float(z_values[i, j]),
            )

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
