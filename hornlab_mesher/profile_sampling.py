from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .profile_common import _is_true, _normalise_formula, eval_param
from .profile_formulas import (
    build_icw_curve,
    calculate_osse,
    calculate_rosse,
    icw_meridian_points,
    osse_length_config,
)
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


def _morph_angle_list(
    params: Mapping[str, Any],
    angular_segments: int,
    *,
    half_width: float | None = None,
    half_height: float | None = None,
) -> np.ndarray | None:
    if not _morph_active(params, 0.0) or _morph_target_shape(params, 0.0) != 1:
        return None
    if half_width is None or half_height is None:
        width = eval_param(params.get("morphWidth"), 0.0, 0.0)
        height = eval_param(params.get("morphHeight"), 0.0, 0.0)
        if width <= 0.0 or height <= 0.0:
            # Implicit target extents are not known yet; the grid builder
            # re-derives the angle list once it has resolved them.
            return None
        half_width = width / 2.0
        half_height = height / 2.0
    if half_width <= 0.0 or half_height <= 0.0:
        return None
    corner = eval_param(params.get("morphCorner"), 0.0, 0.0)
    corner_segments = max(0, int(round(eval_param(params.get("cornerSegments"), 0.0, 0.0))))
    # ATH adds CornerSegments to the angular point budget and rounds the
    # total up to a whole number of points per quadrant (m2-clone: 100 + 4 ->
    # 104; solana: 36 + 1 -> 40).
    points_per_quadrant = max(1, int(math.ceil((angular_segments + corner_segments) / 4.0)))
    return _mirror_quadrant_angles(
        _rounded_rect_quadrant_angles(
            points_per_quadrant,
            half_width,
            half_height,
            corner,
            corner_segments,
        )
    )


def _angle_list(
    params: Mapping[str, Any],
    *,
    morph_half_width: float | None = None,
    morph_half_height: float | None = None,
) -> tuple[np.ndarray, bool]:
    quadrants = str(params.get("quadrants", "1234"))
    angular_segments = _normalise_ath_angular_segments(int(params.get("angularSegments", 64)))
    morphed_full = _morph_angle_list(
        params,
        angular_segments,
        half_width=morph_half_width,
        half_height=morph_half_height,
    )
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


# ATH's default OSSE axial slice distribution is a cubic bezier from (0, 0)
# to (1, 1) with control points (0.5, 0.1) and (0.5, 0.95), evaluated at
# uniform abscissa steps. Fitted against the ATH m2-clone 32-segment grid and
# the solana 9-segment GridExport; both match to ~1e-3 of normalized length.
_ATH_OSSE_ZMAP_BEZIER = ((0.5, 0.1), (0.5, 0.95))


def _bezier_zmap(n_length: int, controls: tuple[tuple[float, float], tuple[float, float]]) -> np.ndarray:
    steps = max(1, int(n_length))
    (x1, y1), (x2, y2) = controls
    s = np.linspace(0.0, 1.0, 100001)
    one_minus = 1.0 - s
    bx = 3.0 * x1 * s * one_minus**2 + 3.0 * x2 * s * s * one_minus + s**3
    by = 3.0 * y1 * s * one_minus**2 + 3.0 * y2 * s * s * one_minus + s**3
    out = np.interp(np.linspace(0.0, 1.0, steps + 1), bx, by)
    out[0] = 0.0
    out[steps] = 1.0
    return out


def _ath_default_zmap(n_length: int, formula: str = "OSSE") -> np.ndarray:
    steps = max(1, int(n_length))
    if steps == len(_ATH_T_9) - 1:
        # Exact ATH 9-segment export (solana reference case).
        return _ATH_T_9.copy()
    if formula != "R-OSSE":
        return _bezier_zmap(steps, _ATH_OSSE_ZMAP_BEZIER)
    # R-OSSE keeps the exact 20-segment ATH reference table (asro cases) and
    # interpolates it for other segment counts.
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
        return _ath_default_zmap(n_length, _normalise_formula(params.get("type", "OSSE"))), mode
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
    radial_steps = n_phi if full_circle else max(0, n_phi - 1)
    if n_length <= 0 or radial_steps <= 0:
        return np.empty((0, 3), dtype=np.int64)
    j = np.repeat(np.arange(n_length, dtype=np.int64), radial_steps)
    i = np.tile(np.arange(radial_steps, dtype=np.int64), n_length)
    row1 = j * n_phi
    row2 = row1 + n_phi
    i2 = (i + 1) % n_phi if full_circle else i + 1
    first = np.stack([row1 + i, row1 + i2, row2 + i2], axis=1)
    second = np.stack([row1 + i, row2 + i2, row2 + i], axis=1)
    indices = np.empty((first.shape[0] * 2, 3), dtype=np.int64)
    indices[0::2] = first
    indices[1::2] = second
    return indices


def _fill_missing_normals(normals: np.ndarray, vertices: np.ndarray, n_phi: int, n_length: int) -> None:
    def has_normal(index: int) -> bool:
        return float(np.linalg.norm(normals[index])) > 1.0e-12

    missing = np.flatnonzero(np.linalg.norm(normals, axis=1) <= 1.0e-12)
    for index in missing:
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
    # Grid order is (phi, column); flatten to column-major vertex rows with
    # the y/z components swapped, matching the triangle index convention.
    vertices = np.ascontiguousarray(
        inner[:, :, (0, 2, 1)].transpose(1, 0, 2).reshape(n_phi * n_cols, 3)
    )

    normals = np.zeros_like(vertices)
    tris = _horn_indices(n_phi, n_length, full_circle=full_circle)
    if tris.shape[0]:
        ab = vertices[tris[:, 1]] - vertices[tris[:, 0]]
        ac = vertices[tris[:, 2]] - vertices[tris[:, 0]]
        face_normals = np.cross(ab, ac)
        np.add.at(normals, tris.ravel(), np.repeat(face_normals, 3, axis=0))
    _fill_missing_normals(normals, vertices, n_phi, n_length)

    sample_idx = np.arange(0, vertices.shape[0], max(1, vertices.shape[0] // 64))
    sample_x = vertices[sample_idx, 0]
    sample_z = vertices[sample_idx, 2]
    radial_len = np.hypot(sample_x, sample_z)
    normal_len = np.linalg.norm(normals[sample_idx], axis=1)
    valid = (radial_len > 1.0e-9) & (normal_len > 1.0e-12)
    dot_sum = float(
        np.sum(
            (normals[sample_idx[valid], 0] / normal_len[valid]) * (sample_x[valid] / radial_len[valid])
            + (normals[sample_idx[valid], 2] / normal_len[valid]) * (sample_z[valid] / radial_len[valid])
        )
    )
    offset_sign = -1.0 if not np.any(valid) or dot_sum < 0.0 else 1.0

    # Unit normals with the _normalise3 fallback for degenerate rows.
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    degenerate = lengths[:, 0] <= 1.0e-12
    unit = np.divide(normals, np.where(lengths > 1.0e-12, lengths, 1.0))
    unit[degenerate] = (0.0, -1.0, 0.0)

    outer_vertices = vertices + offset_sign * wall * unit
    # Throat ring (row 0) is offset radially in the xz plane only.
    throat = unit[:n_phi]
    throat_radial_len = np.hypot(throat[:, 0], throat[:, 2])
    safe_len = np.where(throat_radial_len > 1.0e-12, throat_radial_len, 1.0)
    rx = np.where(throat_radial_len > 1.0e-12, throat[:, 0] / safe_len, 0.0)
    rz = np.where(throat_radial_len > 1.0e-12, throat[:, 2] / safe_len, 0.0)
    outer_vertices[:n_phi, 0] = vertices[:n_phi, 0] + offset_sign * wall * rx
    outer_vertices[:n_phi, 1] = vertices[:n_phi, 1]
    outer_vertices[:n_phi, 2] = vertices[:n_phi, 2] + offset_sign * wall * rz

    outer = np.ascontiguousarray(
        outer_vertices.reshape(n_cols, n_phi, 3).transpose(1, 0, 2)[:, :, (0, 2, 1)]
    )
    outer[:, 0, 2] = inner[:, 0, 2] - wall
    return outer


def _lookup_curve(
    params: Mapping[str, Any], t_unit_values: np.ndarray
) -> list[tuple[float, float]]:
    """Sample a LOOKUP profile's (z, radius) curve at the axial stations.

    The caller owns the PCHIP fit and passes a densely-sampled
    ``lookupProfile`` of [z, r] pairs (so the canonical mesher needs no scipy
    dependency). The base radius is linearly interpolated onto the mesher's
    axial sample positions; with a dense source profile the interpolation
    error is negligible. ``z(t)`` is linear over the profile's z-range.
    """
    raw = params.get("lookupProfile", params.get("lookup_profile"))
    if raw is None:
        raise ValueError("LOOKUP formula requires a lookupProfile of [z, r] pairs")
    profile = np.asarray(raw, dtype=np.float64)
    if profile.ndim != 2 or profile.shape[1] != 2 or profile.shape[0] < 2:
        raise ValueError("lookupProfile must be an array of at least two [z, r] pairs")
    if not np.all(np.isfinite(profile)):
        raise ValueError("lookupProfile must contain only finite values")
    z_src = profile[:, 0]
    r_src = profile[:, 1]
    if np.any(np.diff(z_src) <= 0.0):
        raise ValueError("lookupProfile z values must be strictly increasing")
    z0 = float(z_src[0])
    z1 = float(z_src[-1])
    z_at_t = z0 + np.asarray(t_unit_values, dtype=np.float64) * (z1 - z0)
    r_at_t = np.interp(z_at_t, z_src, r_src)
    return [(float(z), float(r)) for z, r in zip(z_at_t, r_at_t)]


def _raw_radial_grid(
    params: Mapping[str, Any],
    angles: np.ndarray,
    t_values: np.ndarray,
    t_unit_values: np.ndarray,
    formula: str,
    exponent: float,
    aspect_ratio: float,
    n_length: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    raw_radials = np.empty((len(angles), n_length + 1), dtype=np.float64)
    z_values = np.empty((len(angles), n_length + 1), dtype=np.float64)
    max_fixed_len = 0.0
    max_total_len = 0.0
    lookup_curve = _lookup_curve(params, t_unit_values) if formula == "LOOKUP" else None
    # ICW is phi-independent in Phase 1 (no guiding curve / no per-phi
    # expressions), so the curvature curve is solved/fit ONCE here, before the
    # per-phi loop, and its meridian is reused for every azimuth. The
    # superellipse scale is layered on top exactly as for OSSE/R-OSSE.
    icw_curve = build_icw_curve(params) if formula == "ICW" else None
    icw_meridian = (
        icw_meridian_points(icw_curve, t_values) if icw_curve is not None else None
    )
    for i, phi in enumerate(angles):
        scale = _superellipse_scale(float(phi), exponent, aspect_ratio)
        if formula == "LOOKUP":
            # LOOKUP defines a free-form axisymmetric base radius r(z); the
            # cross-section (superellipse scale) and morph are layered on top
            # exactly as for OSSE, so the base curve is phi-independent.
            curve = lookup_curve
        elif formula == "ICW":
            curve = list(zip(icw_meridian[:, 0], icw_meridian[:, 1]))
        elif formula == "OSSE":
            _main_len, total, ext_len, slot_len = osse_length_config(params, float(phi))
            max_fixed_len = max(max_fixed_len, float(ext_len) + float(slot_len))
            max_total_len = max(max_total_len, float(total))
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
    return raw_radials, z_values, max_fixed_len, max_total_len


def build_point_grid(params: Mapping[str, Any]) -> dict[str, Any]:
    formula = _normalise_formula(params.get("type", "OSSE"))
    curve_type = _guiding_curve_type(params, 0.0)
    if curve_type not in {0, 1, 2}:
        raise ValueError(f"unsupported GCurve type {curve_type}")
    if formula in {"R-OSSE", "ICW"} and _guiding_curve_active(params, 0.0):
        raise ValueError("guiding curves are only supported with formula OSSE")
    n_length = int(params.get("lengthSegments", 32))
    if n_length < 1:
        raise ValueError("lengthSegments must be a positive integer")
    angles, full_circle = _angle_list(params)
    exponent, aspect_ratio = _cross_section(params)
    t_max = float(eval_param(params.get("tmax"), 0.0, 1.0)) if formula == "R-OSSE" else 1.0
    if formula == "ICW":
        # ICW samples uniformly in sigma (normalised arc length): it has no
        # ATH/R-OSSE reference axial table, and the kernel already concentrates
        # detail by arc length, so a uniform sigma grid is the natural mapping.
        t_unit_values = np.linspace(0.0, 1.0, n_length + 1, dtype=np.float64)
        sampling_mode = "uniform"
    else:
        t_unit_values, sampling_mode = _axial_sample_map(n_length, params)
    t_values = t_unit_values * t_max
    raw_radials, z_values, max_fixed_len, max_total_len = _raw_radial_grid(
        params, angles, t_values, t_unit_values, formula, exponent, aspect_ratio, n_length
    )

    raw_half_width = float(np.max(np.abs(raw_radials[:, -1] * np.cos(angles))))
    raw_half_height = float(np.max(np.abs(raw_radials[:, -1] * np.sin(angles))))

    morph_target = _morph_target_shape(params, 0.0)
    resolved_half_width: float | None = None
    resolved_half_height: float | None = None
    if morph_target in {1, 2}:
        # ATH derives implicit target extents by rounding the raw mouth
        # extents up to whole millimetres per half-dimension.
        width = eval_param(params.get("morphWidth"), 0.0, 0.0)
        height = eval_param(params.get("morphHeight"), 0.0, 0.0)
        resolved_half_width = width / 2.0 if width > 0.0 else float(math.ceil(raw_half_width - 1.0e-9))
        resolved_half_height = height / 2.0 if height > 0.0 else float(math.ceil(raw_half_height - 1.0e-9))
        if not _is_true(params.get("morphAllowShrinkage")):
            # No-shrinkage gates the target dimensions against the raw mouth
            # extents; the mouth still becomes the exact (enlarged) target.
            resolved_half_width = max(resolved_half_width, raw_half_width)
            resolved_half_height = max(resolved_half_height, raw_half_height)
        if morph_target == 1:
            new_angles, full_circle = _angle_list(
                params,
                morph_half_width=resolved_half_width,
                morph_half_height=resolved_half_height,
            )
            if len(new_angles) != len(angles) or not np.allclose(new_angles, angles):
                angles = new_angles
                raw_radials, z_values, max_fixed_len, max_total_len = _raw_radial_grid(
                    params, angles, t_values, t_unit_values, formula, exponent, aspect_ratio, n_length
                )

    configured_morph_start = eval_param(params.get("morphFixed"), 0.0, 0.0)
    morph_start_idx = int(np.searchsorted(t_values, configured_morph_start, side="left"))
    if morph_start_idx >= len(t_values):
        snapped_morph_start = float(t_values[-1])
    else:
        snapped_morph_start = float(t_values[morph_start_idx])
    if formula == "OSSE" and max_total_len > 1.0e-12 and max_fixed_len > 0.0:
        # ATH keeps the throat-extension/slot region unmorphed by reserving
        # ceil(n * (ext + slot) / L) axial slices and starting the morph at
        # that grid slice.
        reserved_idx = min(n_length, int(math.ceil(n_length * max_fixed_len / max_total_len - 1.0e-9)))
        snapped_morph_start = max(snapped_morph_start, float(t_unit_values[reserved_idx]))

    # _apply_morphing is a per-point no-op unless morphTarget resolves to a
    # morph shape (1/2). When the param is absent or a plain non-morph
    # constant it cannot activate at any azimuth — skip the n_phi * n_length
    # no-op calls. Expression values may vary with phi, so they keep the
    # per-point path.
    morph_param = params.get("morphTarget")
    if morph_param is None:
        morph_possible = False
    elif isinstance(morph_param, (int, float)):
        morph_possible = int(round(float(morph_param))) in {1, 2}
    else:
        morph_possible = True

    inner = np.empty((len(angles), n_length + 1, 3), dtype=np.float64)
    for i, phi in enumerate(angles):
        mouth_radial = float(raw_radials[i, -1])
        for j in range(n_length + 1):
            radial = float(raw_radials[i, j])
            # Morph progress is the global normalized axial position (z / L
            # for OSSE), identical for every azimuth: ATH does not shift the
            # blend by the per-azimuth slot length.
            morph_t = float(t_values[j])
            if morph_possible:
                radial = _apply_morphing(
                    radial,
                    mouth_radial,
                    morph_t,
                    float(phi),
                    params,
                    morph_start=snapped_morph_start,
                    implicit_half_width=resolved_half_width,
                    implicit_half_height=resolved_half_height,
                )
            inner[i, j] = (
                radial * math.cos(float(phi)),
                radial * math.sin(float(phi)),
                float(z_values[i, j]),
            )

    # ATH's global Scale multiplies every linear geometry dimension after the
    # profile (and morph-target ceil) is evaluated; Mesh.VerticalOffset then
    # translates the scaled geometry along +y in raw millimetres.
    geom_scale = float(eval_param(params.get("scale"), 0.0, 1.0))
    if not math.isfinite(geom_scale) or geom_scale <= 0.0:
        raise ValueError(f"Scale must be > 0, got {geom_scale!r}")
    if geom_scale != 1.0:
        inner *= geom_scale
    vertical_offset = float(eval_param(params.get("verticalOffset"), 0.0, 0.0))
    if vertical_offset != 0.0:
        inner[:, :, 1] += vertical_offset

    outer = None
    wall = float(eval_param(params.get("wallThickness"), 0.0, 0.0)) * geom_scale
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
