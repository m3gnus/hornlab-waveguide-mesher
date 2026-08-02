from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .freeform import (
    FreeformGeometry,
    _validate_freeform_config,
    active_rounded_rect_corner_radius_mm,
    validate_outer_offset_grid,
)
from .profile_common import (
    _is_true,
    _normalise_formula,
    _normalise_quadrants,
    _parse_number_list,
    _symmetry_planes_for_quadrants,
    eval_param,
)
from .profile_formulas import (
    build_icw_curve,
    calculate_osse,
    calculate_rosse,
    icw_meridian_points,
    osse_coverage_angle,
    osse_length_config,
)
from .profile_morph import (
    _apply_morphing,
    _guiding_curve_type,
    _guiding_curve_active,
    _morph_active,
    _morph_target_shape,
    _rounded_rect_quadrant_layout,
    _rounded_rect_quadrant_angles,
    rounded_rect_corner_arc_span,
)

# Private params key: acoustic-only corner-arc subdivision (see
# ``_morph_corner_arc_subdivision``). Never set by user configs.
ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY = "_acousticCornerArcSubdivision"
FREEFORM_CONTINUOUS_COLLAPSE_KEY = "_freeformContinuousCollapse"


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


def _morph_corner_arc_subdivision(params: Mapping[str, Any]) -> int:
    """Private acoustic-only override; 1 keeps ATH's fixed three arc intervals.

    Only ``config_builder._build_acoustic_sampling_grid`` sets this, so the
    public grid, the viewport preview and the ATH reference path are unaffected.
    """

    try:
        value = int(params.get(ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY) or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, value)


def _morph_quadrant_budget(
    params: Mapping[str, Any], angular_segments: int
) -> int:
    # ATH adds CornerSegments to the angular point budget and rounds the
    # total up to a whole number of points per quadrant (m2-clone: 100 + 4 ->
    # 104; solana: 36 + 1 -> 40).
    corner_segments = max(0, int(round(eval_param(params.get("cornerSegments"), 0.0, 0.0))))
    return max(1, int(math.ceil((angular_segments + corner_segments) / 4.0)))


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
    points_per_quadrant = _morph_quadrant_budget(params, angular_segments)
    return _mirror_quadrant_angles(
        _rounded_rect_quadrant_angles(
            points_per_quadrant,
            half_width,
            half_height,
            corner,
            corner_segments,
            arc_subdivision=_morph_corner_arc_subdivision(params),
        )
    )


def _is_static_number(value: Any) -> bool:
    """True when a param is a plain number, not an azimuth-dependent expression."""

    return value is None or isinstance(value, (int, float)) and not isinstance(value, bool)


def _morph_corner_arc_span(
    params: Mapping[str, Any],
    half_width: float | None,
    half_height: float | None,
) -> tuple[float, float] | None:
    """First-quadrant azimuth span of the fixed corner arc, or ``None``.

    The morph target radius is evaluated at every azimuth, so an expression-valued
    morph parameter can make the corner differ ring by ring and quadrant by
    quadrant. A single span cannot describe that, and a wrong span would route a
    refinement to the wrong channel -- so only claim one when the rounded-rectangle
    structure is statically fixed.
    """

    if not all(
        _is_static_number(params.get(key))
        for key in ("morphTarget", "morphCorner", "morphWidth", "morphHeight")
    ):
        return None
    if not _morph_active(params, 0.0) or _morph_target_shape(params, 0.0) != 1:
        return None
    if not half_width or not half_height or half_width <= 0.0 or half_height <= 0.0:
        return None
    angular_segments = _normalise_ath_angular_segments(int(params.get("angularSegments", 64)))
    return rounded_rect_corner_arc_span(
        _morph_quadrant_budget(params, angular_segments),
        half_width,
        half_height,
        eval_param(params.get("morphCorner"), 0.0, 0.0),
    )


def _angle_list(
    params: Mapping[str, Any],
    *,
    morph_half_width: float | None = None,
    morph_half_height: float | None = None,
) -> tuple[np.ndarray, bool]:
    angular_segments = _normalise_ath_angular_segments(int(params.get("angularSegments", 64)))
    morphed_full = _morph_angle_list(
        params,
        angular_segments,
        half_width=morph_half_width,
        half_height=morph_half_height,
    )
    q = _normalise_quadrants(params.get("quadrants", "1234"))
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
    if formula != "R-OSSE":
        if steps == len(_ATH_T_9) - 1:
            # Exact ATH 9-segment export (solana reference case, an OSSE grid).
            return _ATH_T_9.copy()
        return _bezier_zmap(steps, _ATH_OSSE_ZMAP_BEZIER)
    # R-OSSE keeps the exact 20-segment ATH reference table (asro cases) and
    # interpolates it for other segment counts.
    ref_steps = len(_ATH_T_20) - 1
    if steps == ref_steps:
        return _ATH_T_20.copy()
    positions = (np.arange(1, steps) / steps) * ref_steps
    out = np.empty(steps + 1, dtype=np.float64)
    out[0] = 0.0
    out[steps] = 1.0
    out[1:steps] = np.interp(positions, np.arange(ref_steps + 1), _ATH_T_20)
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
    return _parse_number_list(value, separators=",;", flatten=True)


def _classify_zmap_kind(n_length: int, z_map_points: Any) -> str:
    """Classify a z-map once, before acoustic refinement changes its length."""

    steps = max(1, int(n_length))
    values = _zmap_number_list(z_map_points)
    if not values:
        raise ValueError("zmap sampling requires zMapPoints/Mesh.ZMapPoints")
    if (
        len(values) == steps + 1
        and math.isclose(values[0], 0.0, abs_tol=1.0e-12)
        and math.isclose(values[-1], 1.0, abs_tol=1.0e-12)
    ):
        return "samples"
    return "controls"


def _custom_zmap(
    n_length: int, z_map_points: Any, z_map_kind: Any = None
) -> np.ndarray:
    steps = max(1, int(n_length))
    values = _zmap_number_list(z_map_points)
    if not values:
        raise ValueError("zmap sampling requires zMapPoints/Mesh.ZMapPoints")

    kind = (
        _classify_zmap_kind(steps, values)
        if z_map_kind is None
        else str(z_map_kind).strip().lower().replace("_", "-")
    )
    if kind in {"sample", "samples", "full", "full-samples"}:
        kind = "samples"
    elif kind in {"control", "controls", "control-pairs", "pairs"}:
        kind = "controls"
    else:
        raise ValueError(
            f"zMapKind must be 'samples' or 'controls', got {z_map_kind!r}"
        )

    if kind == "samples":
        sample_values = np.asarray(values, dtype=np.float64)
        if sample_values.size < 2:
            raise ValueError("zMapPoints full sample map requires at least 2 values")
        if not np.all(np.isfinite(sample_values)):
            raise ValueError("zMapPoints must contain finite values")
        if not math.isclose(float(sample_values[0]), 0.0, abs_tol=1.0e-12) or not math.isclose(
            float(sample_values[-1]), 1.0, abs_tol=1.0e-12
        ):
            raise ValueError("zMapPoints full sample map must start at 0 and end at 1")
        if np.any(np.diff(sample_values) < -1.0e-12):
            raise ValueError("zMapPoints samples must be non-decreasing")
        source_x = np.linspace(0.0, 1.0, sample_values.size, dtype=np.float64)
        out = np.interp(
            np.linspace(0.0, 1.0, steps + 1, dtype=np.float64),
            source_x,
            sample_values,
        )
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
        return _custom_zmap(
            n_length,
            z_map_points,
            params.get("zMapKind", params.get("z_map_kind")),
        ), mode
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
    osse_bulge_profile = (
        tuple(math.sin(float(t_unit) * math.pi) for t_unit in t_unit_values)
        if formula == "OSSE"
        else ()
    )
    for i, phi in enumerate(angles):
        phi_value = float(phi)
        scale = _superellipse_scale(phi_value, exponent, aspect_ratio)
        if formula == "LOOKUP":
            # LOOKUP defines a free-form axisymmetric base radius r(z); the
            # cross-section (superellipse scale) and morph are layered on top
            # exactly as for OSSE, so the base curve is phi-independent.
            curve = lookup_curve
        elif formula == "ICW":
            curve = list(zip(icw_meridian[:, 0], icw_meridian[:, 1]))
        elif formula == "OSSE":
            _main_len, total, ext_len, slot_len = osse_length_config(params, phi_value)
            max_fixed_len = max(max_fixed_len, float(ext_len) + float(slot_len))
            max_total_len = max(max_total_len, float(total))
            h_bulge = eval_param(params.get("h"), phi_value, 0.0)
            # The guiding-curve inversion depends only on phi; hoist it out of
            # the per-z loop (a 24-step bisection per grid point otherwise).
            coverage_angle = osse_coverage_angle(params, phi_value)
            curve = [
                (
                    z,
                    radius + h_bulge * bulge_factor,
                )
                for bulge_factor, (z, radius) in zip(
                    osse_bulge_profile,
                    (
                        calculate_osse(
                            float(t) * total,
                            phi_value,
                            params,
                            coverage_angle=coverage_angle,
                        )
                        for t in t_values
                    ),
                )
            ]
        else:
            curve = [calculate_rosse(float(t), phi_value, params) for t in t_values]
        for j, (z, radius) in enumerate(curve):
            raw_radials[i, j] = float(radius) * scale
            z_values[i, j] = float(z)
    return raw_radials, z_values, max_fixed_len, max_total_len


def _freeform_quadrant_angles(
    q1: np.ndarray, quadrants: str
) -> tuple[np.ndarray, bool]:
    full = _mirror_quadrant_angles(q1)
    if not quadrants or quadrants == "1234":
        return full, True
    if quadrants == "1":
        return full[full <= math.pi / 2.0 + 1.0e-12], False
    if quadrants == "12":
        return full[full <= math.pi + 1.0e-12], False
    if quadrants == "14":
        selected = full[
            (full <= math.pi / 2.0 + 1.0e-12)
            | (full >= 3.0 * math.pi / 2.0 - 1.0e-12)
        ]
        selected = np.where(selected > math.pi, selected - math.tau, selected)
        return np.sort(selected), False
    return full, True


def _freeform_rounded_rect_quadrant_angles(
    *,
    half_width: float,
    half_height: float,
    corner_radius: float,
    side1_segments: int,
    side2_segments: int,
    arc_subdivision: int,
    collapse_transition_intervals: float,
) -> np.ndarray:
    """Rounded-corner angles with stable row identity from ring to ring.

    The generic morph sampler rounds the wall-span allocation independently
    for each outline.  When H/V aspect changes along FREEFORM, that integer
    allocation can jump and make one control-net row teleport to another wall
    span.  Fix the two wall budgets from the mouth while retaining each ring's
    own moving tangencies.
    """

    a = float(half_width)
    b = float(half_height)
    corner = min(max(float(corner_radius), 0.0), a, b)
    theta1 = math.atan2(b - corner, a)
    theta2 = math.atan2(b, a - corner)
    arc_segments = 3 * max(1, int(arc_subdivision))
    side1_segments = int(side1_segments)
    side2_segments = int(side2_segments)
    total_segments = side1_segments + arc_segments + side2_segments
    base_layout = _rounded_rect_quadrant_layout(
        side1_segments + side2_segments + 3,
        a,
        b,
        corner,
    )
    uniform_angles = np.linspace(0.0, math.pi / 2.0, total_segments + 1)
    if corner >= b and corner >= a:
        return uniform_angles

    span1 = theta1
    span2 = math.pi / 2.0 - theta2
    arc_segments = (
        3 if base_layout is None else int(base_layout.arc_segments)
    ) * max(1, int(arc_subdivision))

    angles: list[float] = []
    if side1_segments:
        angles.extend(np.linspace(0.0, theta1, side1_segments + 1).tolist())
    else:
        angles.append(theta1)
    cx = a - corner
    cy = b - corner
    for index in range(1, arc_segments + 1):
        arc_phi = index * math.pi / (2.0 * arc_segments)
        angles.append(
            math.atan2(
                cy + corner * math.sin(arc_phi),
                cx + corner * math.cos(arc_phi),
            )
        )
    if side2_segments:
        angles.extend(
            (
                theta2
                + (math.pi / 2.0 - theta2) * index / side2_segments
                for index in range(1, side2_segments + 1)
            )
        )
    structural_angles = np.asarray(angles, dtype=np.float64)
    # The fixed mouth budgets preserve exact tangencies once both walls are
    # developed, but squeezing those fixed rows onto a vanishing wall would
    # duplicate angles. Blend continuously from the fully reassigned uniform
    # layout over a caller-selected number of nominal angular intervals. Unlike
    # changing integer budgets ring by ring, this keeps every control-net row
    # continuous along z, so
    # acoustic axial refinement can converge and walled offsets cannot fold at
    # a budget transition.
    transition_span = (
        max(1.0, float(collapse_transition_intervals))
        * math.pi
        / (2.0 * total_segments)
    )
    progress = min(1.0, max(0.0, min(span1, span2) / transition_span))
    blend = progress * progress * (3.0 - 2.0 * progress)
    return uniform_angles + blend * (structural_angles - uniform_angles)


def _freeform_merged_axial_map(
    params: Mapping[str, Any], geometry: FreeformGeometry, n_length: int
) -> tuple[np.ndarray, str]:
    base_t, sampling_mode = _axial_sample_map(n_length, params)
    z0 = float(params["profileH"]["points"][0][0])
    semantic_features = [
        (float(station["t"]), f"crossSections[{index}]")
        for index, station in enumerate(geometry.stations)
    ]
    for profile_key in ("profileH", "profileV"):
        anchor_z = np.asarray(
            [row[0] for row in params[profile_key]["points"]], dtype=np.float64
        )
        semantic_features.extend(
            (float(t), f"{profile_key}.points[{index}]")
            for index, t in enumerate((anchor_z - z0) / geometry.length_mm)
        )
    # A base sample can land within float noise of a feature station (e.g. an
    # anchor at t=1/3 vs a uniform station at 35/105): np.unique keeps both and
    # the duplicated ring makes the outer offset shell locally degenerate.
    # Collapse clusters tighter than eps onto their semantic feature. Distinct
    # semantic positions this close describe an unmeshable axial sliver, so
    # reject them rather than silently deleting either one.
    eps = 1.0e-7
    entries = [
        (float(value), False, f"base[{index}]")
        for index, value in enumerate(np.asarray(base_t, dtype=np.float64))
    ]
    entries.extend((value, True, label) for value, label in semantic_features)
    entries.sort(key=lambda item: item[0])
    clusters: list[list[tuple[float, bool, str]]] = []
    for entry in entries:
        if not clusters or entry[0] - clusters[-1][-1][0] > eps:
            clusters.append([entry])
        else:
            clusters[-1].append(entry)

    merged_values: list[float] = []
    for cluster in clusters:
        features = [entry for entry in cluster if entry[1]]
        distinct_feature_values = sorted({entry[0] for entry in features})
        if len(distinct_feature_values) > 1:
            first_value, second_value = distinct_feature_values[:2]
            first_label = next(
                entry[2] for entry in features if entry[0] == first_value
            )
            second_label = next(
                entry[2] for entry in features if entry[0] == second_value
            )
            raise ValueError(
                "FREEFORM semantic axial features are closer than normalized-t "
                f"tolerance {eps:g}: {first_label} at t={first_value:.12g} and "
                f"{second_label} at t={second_value:.12g}"
            )
        merged_values.append(features[0][0] if features else cluster[0][0])

    merged = np.asarray(merged_values, dtype=np.float64)
    merged[0] = 0.0
    merged[-1] = 1.0
    if np.any(np.diff(merged) <= 0.0):
        raise ValueError("FREEFORM merged axial stations must be strictly increasing")
    return merged, sampling_mode


def _freeform_raw_radial_grid(
    params: Mapping[str, Any], n_length: int
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
    np.ndarray,
    bool,
    list[list[float]] | None,
]:
    geometry = _validate_freeform_config(params)
    t_values, sampling_mode = _freeform_merged_axial_map(params, geometry, n_length)
    z0 = float(params["profileH"]["points"][0][0])
    shared_z = z0 + t_values * geometry.length_mm
    radii_h, radii_v = geometry.evaluate_radii(shared_z)

    quadrants = _normalise_quadrants(params.get("quadrants", "1234"))
    has_rounded_rectangle = any(
        station["shape"] == "rounded_rectangle" for station in geometry.stations
    )
    if has_rounded_rectangle:
        angular_segments = _normalise_ath_angular_segments(
            int(params.get("angularSegments", 64))
        )
        points_per_quadrant = _morph_quadrant_budget(params, angular_segments)
        corner_segments = max(
            0,
            int(round(eval_param(params.get("cornerSegments"), 0.0, 0.0))),
        )
        arc_subdivision = _morph_corner_arc_subdivision(params)
        reference_a = float(radii_h[-1])
        reference_b = float(radii_v[-1])
        reference_corner = active_rounded_rect_corner_radius_mm(
            geometry.stations, float(t_values[-1]), reference_a, reference_b
        )
        reference_base = _rounded_rect_quadrant_angles(
            points_per_quadrant,
            reference_a,
            reference_b,
            reference_corner,
            corner_segments,
        )
        reference_span = rounded_rect_corner_arc_span(
            points_per_quadrant, reference_a, reference_b, reference_corner
        )
        if reference_span is None:
            side1_segments = max(0, points_per_quadrant - 3)
            side2_segments = 0
        else:
            side1_segments = int(
                np.flatnonzero(
                    np.isclose(reference_base, reference_span[0], atol=1.0e-12)
                )[-1]
            )
            theta2_index = int(
                np.flatnonzero(
                    np.isclose(reference_base, reference_span[1], atol=1.0e-12)
                )[0]
            )
            side2_segments = int(len(reference_base) - 1 - theta2_index)
        ring_angles = []
        corner_arc_spans: list[list[float]] = []
        full_circle = quadrants in {"", "1234"}
        wall_thickness = float(eval_param(params.get("wallThickness"), 0.0, 0.0))
        collapse_transition_intervals = (
            4.0
            if wall_thickness > 0.0
            or _is_true(params.get(FREEFORM_CONTINUOUS_COLLAPSE_KEY))
            else 1.0
        )
        for ring_index, (t_value, a, b) in enumerate(
            zip(t_values, radii_h, radii_v)
        ):
            corner_radius = active_rounded_rect_corner_radius_mm(
                geometry.stations, float(t_value), float(a), float(b)
            )
            q1 = _freeform_rounded_rect_quadrant_angles(
                half_width=float(a),
                half_height=float(b),
                corner_radius=corner_radius,
                side1_segments=side1_segments,
                side2_segments=side2_segments,
                arc_subdivision=arc_subdivision,
                collapse_transition_intervals=collapse_transition_intervals,
            )
            reduced, full_circle = _freeform_quadrant_angles(q1, quadrants)
            if np.any(np.diff(reduced) <= 0.0):
                raise ValueError(
                    f"FREEFORM ring {ring_index} azimuths must be strictly increasing"
                )
            required_cardinals = {
                "1": (0.0, math.pi / 2.0),
                "12": (0.0, math.pi / 2.0, math.pi),
                "14": (-math.pi / 2.0, 0.0, math.pi / 2.0),
            }.get(
                quadrants,
                (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0),
            )
            for cardinal in required_cardinals:
                if not np.any(
                    np.isclose(reduced, cardinal, rtol=0.0, atol=1.0e-12)
                ):
                    raise ValueError(
                        f"FREEFORM ring {ring_index} azimuths omit required "
                        f"cardinal {cardinal:g} rad"
                    )
            ring_angles.append(reduced)
            span = rounded_rect_corner_arc_span(
                points_per_quadrant,
                float(a),
                float(b),
                corner_radius,
            )
            if span is None:
                corner_arc_spans.append([])
            else:
                corner_arc_spans.append([float(span[0]), float(span[1])])
        row_counts = {len(values) for values in ring_angles}
        if len(row_counts) != 1:
            raise ValueError(
                "FREEFORM per-ring azimuth grids must have a constant row count"
            )
        phi_grid = np.column_stack(ring_angles)
    else:
        angles, full_circle = _angle_list(params)
        phi_grid = np.repeat(angles[:, np.newaxis], len(t_values), axis=1)
        corner_arc_spans = None

    raw_radials = np.empty_like(phi_grid)
    for j, t_value in enumerate(t_values):
        raw_radials[:, j] = geometry.cross_section_radius(
            phi_grid[:, j], float(t_value)
        )
    z_values = np.repeat(shared_z[np.newaxis, :], phi_grid.shape[0], axis=0)

    # Any rounded-rectangle station selects the structural corner-aware family
    # for every ring so intermediate station tangencies cannot alias. Smooth
    # rings use the nearest rounded-rectangle descriptor for harmless pinning.
    return (
        raw_radials,
        z_values,
        phi_grid,
        t_values,
        sampling_mode,
        phi_grid[:, -1].copy(),
        full_circle,
        corner_arc_spans,
    )


def build_point_grid(params: Mapping[str, Any]) -> dict[str, Any]:
    formula = _normalise_formula(params.get("type", "OSSE"))
    quadrants = _normalise_quadrants(params.get("quadrants", "1234"))
    symmetry_planes = _symmetry_planes_for_quadrants(quadrants)
    curve_type = _guiding_curve_type(params, 0.0)
    if curve_type not in {0, 1, 2}:
        raise ValueError(f"unsupported GCurve type {curve_type}")
    if formula in {"R-OSSE", "ICW"} and _guiding_curve_active(params, 0.0):
        raise ValueError("guiding curves are only supported with formula OSSE")
    n_length = int(params.get("lengthSegments", 32))
    if n_length < 1:
        raise ValueError("lengthSegments must be a positive integer")
    exponent, aspect_ratio = _cross_section(params)
    phi_grid: np.ndarray | None = None
    if formula == "FREEFORM":
        (
            raw_radials,
            z_values,
            phi_grid,
            t_values,
            sampling_mode,
            angles,
            full_circle,
            freeform_corner_arc_spans,
        ) = _freeform_raw_radial_grid(params, n_length)
        t_unit_values = t_values
        n_length = len(t_values) - 1
        max_fixed_len = 0.0
        max_total_len = 0.0
    else:
        angles, full_circle = _angle_list(params)
        t_max = float(eval_param(params.get("tmax"), 0.0, 1.0)) if formula == "R-OSSE" else 1.0
        if formula == "ICW":
            # ICW samples uniformly in sigma (normalised arc length): it has no
            # ATH/R-OSSE reference axial table, and the kernel already concentrates
            # detail by arc length, so a uniform sigma grid is the natural mapping.
            # An explicit custom z-map cannot be honoured and must not be silently
            # ignored (generic defaulted modes pass through as uniform).
            requested_mode = str(params.get("samplingMode") or "").strip().lower()
            if requested_mode == "zmap" or params.get("zMapPoints") is not None:
                raise ValueError(
                    "ICW does not support samplingMode='zmap'/zMapPoints; "
                    "it always samples uniformly in normalised arc length"
                )
            t_unit_values = np.linspace(0.0, 1.0, n_length + 1, dtype=np.float64)
            sampling_mode = "uniform"
        else:
            t_unit_values, sampling_mode = _axial_sample_map(n_length, params)
        t_values = t_unit_values * t_max
        raw_radials, z_values, max_fixed_len, max_total_len = _raw_radial_grid(
            params, angles, t_values, t_unit_values, formula, exponent, aspect_ratio, n_length
        )

    if formula == "FREEFORM":
        raw_half_width = float(
            np.max(np.abs(raw_radials[:, -1] * np.cos(phi_grid[:, -1])))
        )
        raw_half_height = float(
            np.max(np.abs(raw_radials[:, -1] * np.sin(phi_grid[:, -1])))
        )
    else:
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

    morph_corner_arc_span = _morph_corner_arc_span(
        params, resolved_half_width, resolved_half_height
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
    if formula == "FREEFORM":
        for i in range(len(angles)):
            for j in range(n_length + 1):
                phi_value = float(phi_grid[i, j])
                radial = float(raw_radials[i, j])
                inner[i, j] = (
                    radial * math.cos(phi_value),
                    radial * math.sin(phi_value),
                    float(z_values[i, j]),
                )
    else:
        for i, phi in enumerate(angles):
            phi_value = float(phi)
            cos_phi = math.cos(phi_value)
            sin_phi = math.sin(phi_value)
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
                        phi_value,
                        params,
                        morph_start=snapped_morph_start,
                        implicit_half_width=resolved_half_width,
                        implicit_half_height=resolved_half_height,
                    )
                inner[i, j] = (
                    radial * cos_phi,
                    radial * sin_phi,
                    float(z_values[i, j]),
                )

    # ATH's global Scale multiplies every linear geometry dimension after the
    # profile (and morph-target ceil) is evaluated.
    geom_scale = float(eval_param(params.get("scale"), 0.0, 1.0))
    if not math.isfinite(geom_scale) or geom_scale <= 0.0:
        raise ValueError(f"Scale must be > 0, got {geom_scale!r}")
    if geom_scale != 1.0:
        inner *= geom_scale
    # Mesh.VerticalOffset is a rigid +y placement translation. It is deliberately
    # NOT baked into the grid here: the reduced-domain snap and enclosure builders
    # assume the symmetry cut planes lie on the coordinate axes (x=0 / y=0), so the
    # intrinsic geometry stays at the origin and the offset is returned as metadata.
    # Each terminal re-applies it as a single rigid translation once all cut-plane
    # logic has run at y=0 -- the mesh in mesher._postprocess_mesh, previews in
    # viewport.build_viewport_geometry_from_config. This mirrors ATH, which builds
    # the reduced model on the axes and then translates it while still declaring the
    # symmetry plane at y=0; a y-cut (quadrants 1/12) therefore reconstructs about
    # y=0 rather than the shifted plane (an ATH quirk we reproduce for parity).
    vertical_offset = float(eval_param(params.get("verticalOffset"), 0.0, 0.0))

    outer = None
    wall = float(eval_param(params.get("wallThickness"), 0.0, 0.0))
    enc_depth = float(eval_param(params.get("encDepth"), 0.0, 0.0))
    if enc_depth <= 0.0 and wall > 0.0:
        outer = _outer_offset_shell(inner, wall, full_circle=full_circle)
        if formula == "FREEFORM":
            validate_outer_offset_grid(inner, outer, full_circle=full_circle)

    return {
        "inner_points": inner.reshape(-1).tolist(),
        "outer_points": None if outer is None else outer.reshape(-1).tolist(),
        "grid_n_phi": int(inner.shape[0]),
        "grid_n_length": int(n_length),
        "full_circle": bool(full_circle),
        "quadrants": quadrants,
        "symmetry_planes": list(symmetry_planes),
        "vertical_offset_mm": vertical_offset,
        "angle_list": angles.tolist(),
        "slice_map": t_values.tolist(),
        "sampling_mode": sampling_mode,
        # Azimuth span of the fixed-structure morph corner arc (first quadrant),
        # so the acoustic fit can tell corner intervals from wall intervals.
        "morph_corner_arc_span": (
            None
            if morph_corner_arc_span is None
            else [float(morph_corner_arc_span[0]), float(morph_corner_arc_span[1])]
        ),
        **(
            {
                "phi_grid": phi_grid.tolist(),
                "freeform_corner_arc_spans": freeform_corner_arc_spans,
            }
            if formula == "FREEFORM" and phi_grid is not None
            else {}
        ),
    }
