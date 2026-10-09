from __future__ import annotations

import functools
import logging
import math
from typing import Any, Mapping, NamedTuple, Sequence

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
from .offset_envelope import regularize_outer_offset
from .text_import import uses_text_import_geometry
from .profile_formulas import (
    build_icw_curve,
    calculate_osse_curve,
    calculate_rosse_curve,
    icw_meridian_points,
    lookup_profile_array,
    osse_coverage_angle,
    osse_length_config,
    rosse_axial_layout,
    _rosse_tmax,
)
from .profile_morph import (
    _continuous_morph_start,
    _guiding_curve_type,
    _guiding_curve_active,
    _morph_factor,
    _morph_active,
    _morph_factors,
    _morph_target_radius_at_angle,
    _morph_target_shape,
    _resolve_morph_half_dimensions,
    _validate_static_morph_target,
    _rounded_rect_quadrant_layout,
    _rounded_rect_quadrant_angles,
    rounded_rect_corner_arc_span,
)

# Private params key: acoustic-only corner-arc subdivision (see
# ``_morph_corner_arc_subdivision``). Never set by user configs, and set by two
# callers only -- one here, one in Waveguide Generator. Both are named in that
# function's docstring, because a caller outside this repository cannot be found
# by grepping it.
logger = logging.getLogger(__name__)

ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY = "_acousticCornerArcSubdivision"
FREEFORM_CONTINUOUS_COLLAPSE_KEY = "_freeformContinuousCollapse"
# Private params keys the acoustic fit sets on its working copy so the solve
# grid describes the surface the requested (preview) grid describes, although
# it samples it differently: the requested grid's resolved morph start (blend
# progress) and the normalised axial stations (morph start, subdomain
# interfaces) that must exist as rings. Never set by user configs.
ACOUSTIC_MORPH_START_KEY = "_acousticMorphStart"
ACOUSTIC_AXIAL_STATIONS_KEY = "_acousticAxialStations"
# Private params key: ``False`` gives ATH's own rule, in which Slot.Length is
# part of the morphed horn. Set only by the ATH text importer. Everything else
# keeps this mesher's contract that a throat extension and a slot are never
# morphed (the extension is excluded by construction either way).
MORPH_KEEPS_SLOT_KEY = "_morphKeepsSlot"


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
    q = np.asarray(q1, dtype=np.float64)
    count = len(q)
    if count < 2:
        return q.copy()
    full = np.empty(4 * count - 4, dtype=np.float64)
    full[:count] = q
    full[count : 2 * count - 1] = math.pi - q[-2::-1]
    full[2 * count - 1 : 3 * count - 2] = math.pi + q[1:]
    full[3 * count - 2 :] = math.tau - q[-2:0:-1]
    return full


def _morph_corner_arc_subdivision(params: Mapping[str, Any]) -> int:
    """Private acoustic-only override; 1 keeps ATH's fixed three arc intervals.

    The default leaves the public grid, the viewport preview and the ATH
    reference path untouched. Two callers raise it, and this docstring is the
    only place the second one is visible from here:

    * ``config_builder._build_acoustic_sampling_grid``, when a corner-arc
      violation needs the arc refined rather than the angular budget grown.
    * **Waveguide Generator's inner-surface STEP planner**, on the params it
      hands ``build_point_grid`` to build the *reference* it measures the
      written loft against -- never the grid the file is written from. It needs
      the reference to hold samples between the arc's fixed breakpoints, because
      that is where the ring spline it is checking overshoots and raising
      ``angularSegments`` alone can never put a sample there.

    What that consumer depends on is the refinement property
    ``_rounded_rect_quadrant_angles`` documents: subdividing splits each arc
    interval and leaves the wall budget alone, so a subdivided grid is the
    unsubdivided one plus extra arc azimuths, with every shared azimuth giving
    the same point. A change that broke that -- or that stopped honouring the
    key -- would not fail here; it would quietly make that planner's measurement
    blind again. It has a counterpart test on its own side, and this note is the
    matching half of that contract.
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
    # Finite superellipses (target 3) are smooth and need no corner-tangency azimuths.
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
    # Finite superellipses (target 3) have no corner arc to refine.
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
    # Derived acoustic fit stations never enter the preview or user parameters.
    if "_acoustic_mouth_angles" in params:
        return _restrict_to_quadrants(
            np.asarray(params["_acoustic_mouth_angles"], dtype=np.float64),
            _normalise_quadrants(params.get("quadrants", "1234")),
        )
    angular_segments = _normalise_ath_angular_segments(int(params.get("angularSegments", 64)))
    morphed_full = _morph_angle_list(
        params,
        angular_segments,
        half_width=morph_half_width,
        half_height=morph_half_height,
    )
    q = _normalise_quadrants(params.get("quadrants", "1234"))
    if morphed_full is not None:
        return _restrict_to_quadrants(morphed_full, q)
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
        # ATH lists may include either endpoint. Interior-only native lists
        # retain their implicit endpoints, but explicit ones must not be
        # inserted twice (which would fail the strict x ordering below).
        for x, y in controls:
            if math.isclose(x, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
                if not math.isclose(y, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
                    raise ValueError("zMapPoints endpoint at x=0 must be (0,0)")
            if math.isclose(x, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                if not math.isclose(y, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                    raise ValueError("zMapPoints endpoint at x=1 must be (1,1)")
        if not math.isclose(controls[0][0], 0.0, rel_tol=0.0, abs_tol=1.0e-12):
            controls.insert(0, (0.0, 0.0))
        if not math.isclose(controls[-1][0], 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            controls.append((1.0, 1.0))
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


GCURVE_CROSS_SECTION_CONFLICT = (
    "a guiding curve cannot be combined with a non-circular cross_section "
    "(exponent != 2 or aspect_ratio != 1): the cross-section scale is applied "
    "after the coverage angle is solved against the guiding curve, so the wall "
    "would miss the curve it was solved for; shape the outline with the "
    "guiding curve's own aspect ratio instead"
)


def _cross_section_is_circular(exponent: float, aspect_ratio: float) -> bool:
    return math.isclose(float(exponent), 2.0, rel_tol=0.0, abs_tol=1.0e-12) and math.isclose(
        float(aspect_ratio), 1.0, rel_tol=0.0, abs_tol=1.0e-12
    )


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


def _grid_phi_derivative(
    points: np.ndarray, *, full_circle: bool, phi_coordinates: np.ndarray | None
) -> np.ndarray:
    """Differentiate a ``(t, phi, xyz)`` grid along its azimuth rows."""

    n_t, n_phi = points.shape[:2]
    phi = None
    shared_phi = False
    if phi_coordinates is not None:
        candidate = np.asarray(phi_coordinates, dtype=np.float64)
        if candidate.shape == (n_t, n_phi):
            # The ordinary and morph grids broadcast one azimuth row along t.
            # Validate and unwrap that row once instead of materialising the
            # same coordinate work for every axial station. FREEFORM's moving
            # tangencies have a genuine 2-D grid and retain the general path.
            # The open-domain branch below delegates each row to np.gradient
            # and therefore still requires the full 2-D coordinate shape.
            shared_phi = full_circle and candidate.strides[0] == 0
            finite_candidate = candidate[0] if shared_phi else candidate
            if np.all(np.isfinite(finite_candidate)):
                phi = finite_candidate

    if full_circle:
        if phi is None:
            step = math.tau / n_phi
            return (np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1)) / (
                2.0 * step
            )
        unwrapped = np.unwrap(phi, axis=-1)
        previous = np.roll(unwrapped, 1, axis=-1)
        previous[..., 0] -= math.tau
        following = np.roll(unwrapped, -1, axis=-1)
        following[..., -1] += math.tau
        h_previous = unwrapped - previous
        h_next = following - unwrapped
        if np.any(h_previous <= 0.0) or np.any(h_next <= 0.0):
            step = math.tau / n_phi
            return (np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1)) / (
                2.0 * step
            )
        weight_previous = (-h_next / (h_previous * (h_previous + h_next)))[..., None]
        weight_center = ((h_next - h_previous) / (h_previous * h_next))[..., None]
        weight_next = (h_previous / (h_next * (h_previous + h_next)))[..., None]
        return (
            weight_previous * np.roll(points, 1, axis=1)
            + weight_center * points
            + weight_next * np.roll(points, -1, axis=1)
        )

    if phi is None or np.any(np.diff(phi, axis=1) <= 0.0):
        phi = np.broadcast_to(
            np.linspace(0.0, 1.0, n_phi, dtype=np.float64), (n_t, n_phi)
        )
    derivative = np.empty_like(points)
    edge_order = 2 if n_phi >= 3 else 1
    for row in range(n_t):
        derivative[row] = np.gradient(
            points[row], phi[row], axis=0, edge_order=edge_order
        )
    return derivative


def _grid_surface_normals(
    points: np.ndarray,
    *,
    full_circle: bool,
    t_coordinates: np.ndarray | None = None,
    phi_coordinates: np.ndarray | None = None,
) -> np.ndarray:
    """Unnormalised ``dP/dphi x dP/dt`` at every node of a ``(t, phi, xyz)`` grid.

    Averaging the incident face normals instead is area-weighted, so it leans
    towards whichever neighbouring row is further away. The throat- and
    mouth-clustered axial maps make consecutive intervals differ several-fold
    right where an R-OSSE rollback is still turning, and the mouth row only has
    neighbours on one side at all. The resulting few-hundredths-of-a-degree tilt
    is harmless on the flare but not at the rim, where a rollback compresses the
    offset's arc length by ``(R_curvature - wall) / R_curvature``: the last
    interval is then short enough that a micron of normal error walks the offset
    backwards and reverses ``dP/dt`` for the whole ring. Differentiating the
    parameterisation keeps the rim normal honest.
    """

    n_t, n_phi = points.shape[:2]
    if n_t < 2 or n_phi < 2:
        return np.zeros_like(points)

    t = None
    if t_coordinates is not None:
        candidate = np.asarray(t_coordinates, dtype=np.float64)
        if (
            candidate.shape == (n_t,)
            and np.all(np.isfinite(candidate))
            and np.all(np.diff(candidate) > 0.0)
        ):
            t = candidate
    if t is None:
        t = np.arange(n_t, dtype=np.float64)

    d_t = np.gradient(points, t, axis=0, edge_order=2 if n_t >= 3 else 1)
    d_phi = _grid_phi_derivative(
        points, full_circle=full_circle, phi_coordinates=phi_coordinates
    )
    normals = np.empty_like(points)
    normals[..., 0] = d_phi[..., 1] * d_t[..., 2] - d_phi[..., 2] * d_t[..., 1]
    normals[..., 1] = d_phi[..., 2] * d_t[..., 0] - d_phi[..., 0] * d_t[..., 2]
    normals[..., 2] = d_phi[..., 0] * d_t[..., 1] - d_phi[..., 1] * d_t[..., 0]
    return normals


def _outer_offset_shell(
    inner: np.ndarray,
    wall: float,
    *,
    full_circle: bool,
    t_coordinates: np.ndarray | None = None,
    phi_coordinates: np.ndarray | None = None,
    repair_osse: bool = False,
) -> np.ndarray:
    n_phi, n_cols, _ = inner.shape
    n_length = n_cols - 1
    # Grid order is (phi, column); flatten to column-major vertex rows with
    # the y/z components swapped, matching the triangle index convention.
    vertices = np.ascontiguousarray(
        inner[:, :, (0, 2, 1)].transpose(1, 0, 2).reshape(n_phi * n_cols, 3)
    )

    # The swap above leaves ``vertices`` in exactly the (t, phi, xyz) order the
    # derivatives want; the reflection it introduces flips the cross product's
    # handedness, which ``offset_sign`` below resolves either way.
    normals = _grid_surface_normals(
        vertices.reshape(n_cols, n_phi, 3),
        full_circle=full_circle,
        t_coordinates=t_coordinates,
        phi_coordinates=phi_coordinates,
    ).reshape(n_phi * n_cols, 3)
    # Most valid grids have no missing derivative normal. Reuse this norm for
    # sign selection and unit normalization instead of scanning the full grid
    # once in _fill_missing_normals and then reducing it all over again here.
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    degenerate = lengths[:, 0] <= 1.0e-12
    if np.any(degenerate):
        _fill_missing_normals(normals, vertices, n_phi, n_length)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        degenerate = lengths[:, 0] <= 1.0e-12

    def radial_dot_sum(idx: np.ndarray) -> tuple[float, int]:
        radial_len = np.hypot(vertices[idx, 0], vertices[idx, 2])
        normal_len = lengths[idx, 0]
        valid = (radial_len > 1.0e-9) & (normal_len > 1.0e-12)
        picked = idx[valid]
        total = float(
            np.sum(
                (normals[picked, 0] * vertices[picked, 0] + normals[picked, 2] * vertices[picked, 2])
                / (normal_len[valid] * radial_len[valid])
            )
        )
        return total, int(np.count_nonzero(valid))

    # The derivative normals are oriented consistently over the whole grid, so
    # one sign puts the wall on the right side everywhere, and the throat ring
    # says which: there the wall lies outside the bore, away from the axis. A
    # sum over nodes spread across the grid does not. Along an R-OSSE rollback
    # the wall lies on the axis side of the surface, those nodes vote the other
    # way, and which side won depended on how many of the strided samples fell
    # on the rollback -- the same design was offset outward at one preview
    # level and inward at another.
    dot_sum, n_valid = radial_dot_sum(np.arange(n_phi))
    if n_valid == 0 or abs(dot_sum) <= 1.0e-6 * n_valid:
        # A throat wall perpendicular to the axis has no radial normal to read.
        sample_idx = np.arange(0, vertices.shape[0], max(1, vertices.shape[0] // 64))
        dot_sum, n_valid = radial_dot_sum(sample_idx)
    offset_sign = -1.0 if n_valid == 0 or dot_sum < 0.0 else 1.0

    # Unit normals with the _normalise3 fallback for degenerate rows.
    if np.all(lengths > 1.0e-12):
        unit = np.divide(normals, lengths)
    else:
        unit = np.divide(normals, np.where(lengths > 1.0e-12, lengths, 1.0))
        unit[degenerate] = (0.0, -1.0, 0.0)

    # One rule, every row: offset along the true surface normal.
    #
    # Row 0 used to be special-cased to a purely radial in-plane offset with its
    # z clamped to ``inner_z - wall``, leaving the meshed throat point off the
    # offset surface and creasing the outer wall one ring in. Every other row
    # was already an exact normal offset. The crease was known, but recorded as
    # a shading problem and worked around with its own render role rather than
    # fixed.
    #
    # Nothing required it: ATH places the outer throat point on the exact normal
    # offset too. This normal parameterization can still contain internal loops
    # when the wall exceeds a concavity's curvature radius. The caller validates
    # it and, for OSSE, resamples the exterior envelope when necessary.
    outer_vertices = vertices + offset_sign * wall * unit

    outer = np.ascontiguousarray(
        outer_vertices.reshape(n_cols, n_phi, 3).transpose(1, 0, 2)[:, :, (0, 2, 1)]
    )
    if repair_osse:
        try:
            validate_outer_offset_grid(
                inner, outer, full_circle=full_circle, label="OSSE"
            )
        except ValueError:
            try:
                candidate = regularize_outer_offset(
                    inner, outer, wall, full_circle=full_circle
                )
                validate_outer_offset_grid(
                    inner, candidate, full_circle=full_circle, label="OSSE"
                )
            except ValueError:
                pass  # The caller retains the original fold diagnosis.
            else:
                outer = candidate
                logger.info(
                    "[hornlab-mesher] resolved OSSE outer offset loops "
                    "using the constant-distance exterior envelope"
                )
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
    if params.get("lookupProfile", params.get("lookup_profile")) is None:
        raise ValueError("LOOKUP formula requires a lookupProfile of [z, r] pairs")
    for key in ("throatExtLength", "throatExtAngle", "slotLength"):
        if eval_param(params.get(key), 0.0, 0.0) != 0.0:
            raise ValueError(
                f"LOOKUP formula does not support {key}: the lookupProfile defines "
                "the whole meridian; build the extension into the lookupProfile"
            )
    profile = lookup_profile_array(params)
    z_src = profile[:, 0]
    r_src = profile[:, 1]
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
) -> tuple[np.ndarray, np.ndarray, "_ThroatPrefix"]:
    # The per-azimuth loop below ends in an R-OSSE branch, so any formula it
    # does not name would be evaluated as R-OSSE without a word.
    if formula not in {"OSSE", "R-OSSE", "ICW", "LOOKUP"}:
        raise ValueError(f"no meridian formula for {formula!r} in the shared radial grid")
    raw_radials = np.empty((len(angles), n_length + 1), dtype=np.float64)
    z_values = np.empty((len(angles), n_length + 1), dtype=np.float64)
    prefix_fraction = 0.0
    slot_end_fraction = 0.0
    lookup_curve = _lookup_curve(params, t_unit_values) if formula == "LOOKUP" else None
    lookup_meridian = (
        np.asarray(lookup_curve, dtype=np.float64) if lookup_curve is not None else None
    )
    # ICW is phi-independent in Phase 1 (no guiding curve / no per-phi
    # expressions), so the curvature curve is solved/fit ONCE here, before the
    # per-phi loop, and its meridian is reused for every azimuth. The
    # superellipse scale is layered on top exactly as for OSSE/R-OSSE.
    icw_curve = build_icw_curve(params) if formula == "ICW" else None
    icw_meridian = (
        icw_meridian_points(icw_curve, t_values) if icw_curve is not None else None
    )
    osse_bulge_profile = (
        np.sin(np.asarray(t_unit_values, dtype=np.float64) * math.pi)
        if formula == "OSSE"
        else None
    )
    # Both formula families evaluate a whole meridian per azimuth rather than a
    # point per grid node: their parameter sets, length solves and coverage
    # inversions are all constant along t, and paying them per node is what
    # made a preview build spend ~90% of its time in the profile formulas.
    for i, phi in enumerate(angles):
        phi_value = float(phi)
        scale = _superellipse_scale(phi_value, exponent, aspect_ratio)
        if formula == "LOOKUP":
            # LOOKUP defines a free-form axisymmetric base radius r(z); the
            # cross-section (superellipse scale) and morph are layered on top
            # exactly as for OSSE, so the base curve is phi-independent.
            curve_z = lookup_meridian[:, 0]
            curve_radius = lookup_meridian[:, 1]
        elif formula == "ICW":
            curve_z = icw_meridian[:, 0]
            curve_radius = icw_meridian[:, 1]
        elif formula == "OSSE":
            _main_len, total, ext_len, slot_len = osse_length_config(params, phi_value)
            if total > 1.0e-12:
                prefix_fraction = max(prefix_fraction, float(ext_len) / float(total))
                slot_end_fraction = max(
                    slot_end_fraction, (float(ext_len) + float(slot_len)) / float(total)
                )
            h_bulge = eval_param(params.get("h"), phi_value, 0.0)
            # The guiding-curve inversion depends only on phi; hoist it out of
            # the per-z loop (a 24-step bisection per grid point otherwise).
            coverage_angle = osse_coverage_angle(params, phi_value)
            curve_z, curve_radius = calculate_osse_curve(
                t_values * total,
                phi_value,
                params,
                coverage_angle=coverage_angle,
            )
            curve_radius = curve_radius + h_bulge * osse_bulge_profile
        else:
            layout = rosse_axial_layout(params, phi_value)
            if layout.full_length > 1.0e-12 and (layout.ext_len > 0.0 or layout.slot_len > 0.0):
                prefix_fraction = max(
                    prefix_fraction, layout.ext_len / layout.full_length
                )
                slot_end_fraction = max(
                    slot_end_fraction,
                    (layout.ext_len + layout.slot_len) / layout.full_length,
                )
            curve_z, curve_radius = calculate_rosse_curve(
                t_values, phi_value, params
            )
        raw_radials[i] = curve_radius * scale
        z_values[i] = curve_z
    return raw_radials, z_values, _ThroatPrefix(prefix_fraction, slot_end_fraction)


class _ThroatPrefix(NamedTuple):
    """Normalised-``t`` extent of the straight throat prefix (largest over phi).

    ``prefix_fraction`` ends the throat extension; ``slot_end_fraction`` ends
    the slot that follows it. A future throat adapter segment belongs in the
    prefix as well: whatever lies before the main flare is never morphed.
    """

    prefix_fraction: float
    slot_end_fraction: float


class _MorphSchedule(NamedTuple):
    progress: np.ndarray
    start: float
    start_station: float | None


def _morph_schedule(
    params: Mapping[str, Any],
    formula: str,
    *,
    t_unit_values: np.ndarray,
    t_values: np.ndarray,
    t_max: float,
    throat_prefix: _ThroatPrefix,
) -> _MorphSchedule:
    """Blend progress per axial station and the progress where the blend starts.

    The blend runs over the horn *after* the throat extension:
    ``u = (t - e) / (1 - e)`` with ``e`` the extension's share of the
    normalised axial parameter, so extension stations have ``u <= 0`` and are
    never morphed, whatever axial map samples them (ATH ath.exe GridExport,
    V2025-06: an extension stays round and ``Morph.FixedPart`` is measured
    from its end). ``Morph.FixedPart`` snaps to the first station at or past
    it. The slot is reserved by position too -- the blend starts no earlier
    than the first station at or past the slot's end -- unless
    ``MORPH_KEEPS_SLOT_KEY`` asks for ATH's rule, which morphs the slot.

    Both rules used to count rings instead (``ceil(n * (ext + slot) / L)``),
    which is a position only on a uniform map: the acoustic fit's throat-
    clustered map then morphed the last rings of a straight extension.

    Without an extension this reproduces the historical arithmetic exactly.
    """

    t_unit = np.asarray(t_unit_values, dtype=np.float64)
    t_vals = np.asarray(t_values, dtype=np.float64)
    e = float(throat_prefix.prefix_fraction)
    truncated_rosse = formula == "R-OSSE" and 0.0 < t_max < 1.0
    if e > 0.0:
        unit_progress = (t_unit - e) / (1.0 - e)
        progress = unit_progress
        # Morph.FixedPart is a fraction of the untruncated path, as without
        # an extension (a value at or above tmax disables the morph).
        measure = unit_progress * t_max if formula == "R-OSSE" else unit_progress
    else:
        unit_progress = t_unit
        # ATH blends against an assumed unit path and undershoots the target
        # when R-OSSE tmax truncates that path. HornLab deliberately
        # normalises both progress and its snapped start by the actual path
        # so the mouth reaches a factor of one.
        progress = t_unit if truncated_rosse else t_vals
        measure = t_vals

    last = len(t_unit) - 1
    configured = eval_param(params.get("morphFixed"), 0.0, 0.0)
    index = min(last, int(np.searchsorted(measure, configured, side="left")))
    if e > 0.0:
        # Never inside the extension, even for a negative FixedPart.
        index = max(index, min(last, int(np.searchsorted(unit_progress, -1.0e-12, side="left"))))
    slot_end = float(throat_prefix.slot_end_fraction)
    keeps_slot = _is_true(params.get(MORPH_KEEPS_SLOT_KEY, True))
    if keeps_slot and slot_end > e + 1.0e-15:
        slot_progress = (slot_end - e) / (1.0 - e)
        index = max(
            index,
            min(last, int(np.searchsorted(unit_progress, slot_progress - 1.0e-9, side="left"))),
        )

    if e > 0.0:
        start = float(progress[index])
    elif truncated_rosse:
        start = float(t_vals[index]) / t_max
    else:
        start = float(t_vals[index])
    start_station = float(t_unit[index])

    override = params.get(ACOUSTIC_MORPH_START_KEY)
    if override is not None:
        start = float(override)
        start_station = None
    return _MorphSchedule(np.asarray(progress, dtype=np.float64), start, start_station)


def _pin_axial_stations(
    t_unit_values: np.ndarray, stations: Sequence[float] | None, *, eps: float = 1.0e-9
) -> np.ndarray:
    """Insert required normalised stations into an axial map.

    A station within ``eps`` of an existing one is already present; callers
    locate pinned stations by nearest value, not by equality.
    """

    if not stations:
        return t_unit_values
    values = np.asarray(t_unit_values, dtype=np.float64)
    extra = []
    for raw in stations:
        station = float(raw)
        if not (math.isfinite(station) and 0.0 < station < 1.0):
            continue
        if np.min(np.abs(values - station)) <= eps:
            continue
        if extra and min(abs(station - other) for other in extra) <= eps:
            continue
        extra.append(station)
    if not extra:
        return t_unit_values
    return np.sort(np.concatenate((values, np.asarray(extra, dtype=np.float64))))


def _restrict_to_quadrants(full: np.ndarray, quadrants: str) -> tuple[np.ndarray, bool]:
    """Select a symmetry domain from a mirrored full-circle azimuth list.

    ``quadrants`` is already normalised to ``1``/``12``/``14``/``1234``. The
    ``14`` half is returned on ``[-pi/2, pi/2]`` so it stays increasing.
    """

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


def _freeform_quadrant_angles(
    q1: np.ndarray, quadrants: str
) -> tuple[np.ndarray, bool]:
    return _restrict_to_quadrants(_mirror_quadrant_angles(q1), quadrants)


@functools.lru_cache(maxsize=128)
def _freeform_rounded_rect_static_basis(
    side1_segments: int,
    side2_segments: int,
    arc_subdivision: int,
) -> tuple[np.ndarray, tuple[tuple[float, float], ...]]:
    """Ring-invariant uniform angles and scalar-math corner-arc trig."""

    arc_segments = 3 * max(1, int(arc_subdivision))
    total_segments = int(side1_segments) + arc_segments + int(side2_segments)
    uniform_angles = np.linspace(0.0, math.pi / 2.0, total_segments + 1)
    # Keep math.sin/cos and the expression order used by the per-ring scalar
    # loop. NumPy trig moves a few low bits on some platforms, while these
    # values feed a byte-stable preview contract.
    arc_trig = tuple(
        (
            math.sin(index * math.pi / (2.0 * arc_segments)),
            math.cos(index * math.pi / (2.0 * arc_segments)),
        )
        for index in range(1, arc_segments + 1)
    )
    uniform_angles.flags.writeable = False
    return uniform_angles, arc_trig


# Morph factor at which a smooth-schedule ring has fully taken the rectangle
# corner layout. Below it the layout blends from uniform (see
# _blend_toward_uniform_layout).
_FREEFORM_MORPH_LAYOUT_SATURATION = 1.0


def _blend_toward_uniform_layout(
    quadrant_angles: np.ndarray, morph_factor: float
) -> tuple[np.ndarray, bool]:
    """Blend a rectangle-corner ring layout back toward the uniform one.

    The layout of a ring that is still nearly circular must not be the
    rectangle layout: that packs the corner arc's rows onto the diagonal, so
    the first morphing ring's meridians jump in azimuth (up to ~15 degrees)
    relative to the uniform throat ring, twisting the control net where the
    morph starts. Blending by a smooth function of the morph factor makes each
    row's azimuth continuous in ``t`` and reaches the exact corner layout, with
    its tangencies, once the morph is developed. Returns the angles and whether
    they are the exact corner layout (so corner-arc spans still describe them).
    """

    progress = min(
        1.0, max(0.0, float(morph_factor) / _FREEFORM_MORPH_LAYOUT_SATURATION)
    )
    if progress >= 1.0:
        return quadrant_angles, True
    weight = progress * progress * (3.0 - 2.0 * progress)
    uniform = np.linspace(0.0, math.pi / 2.0, len(quadrant_angles))
    return uniform + weight * (quadrant_angles - uniform), False


# An azimuth shift this large between adjacent axial rings twists the control
# net: the offset shell folds and the axial chord cannot converge at any mm.
_FREEFORM_AZIMUTH_TWIST_DEG = 5.0


def freeform_azimuth_twist_note(
    phi_grid: Any, t_values: Any | None = None
) -> str:
    """Name an azimuth-layout jump between adjacent rings, or return ``""``.

    ``phi_grid`` is the FREEFORM ``(azimuth, ring)`` grid. A failing offset
    check or axial-chord fit that coincides with such a jump has a sampling
    cause, so its message must not send the user to the resolution or the wall.
    """

    phi = np.asarray(phi_grid, dtype=np.float64)
    if phi.ndim != 2 or phi.shape[1] < 2:
        return ""
    steps = np.degrees(np.abs(np.diff(phi, axis=1))).max(axis=0)
    interval = int(np.argmax(steps))
    if not steps[interval] >= _FREEFORM_AZIMUTH_TWIST_DEG:
        return ""
    where = f"axial rings {interval} and {interval + 1}"
    if t_values is not None and len(t_values) > interval + 1:
        where += f" (t={float(t_values[interval]):.3f} to {float(t_values[interval + 1]):.3f})"
    return (
        f"the FREEFORM azimuth rows shift by {steps[interval]:.0f} degrees between "
        f"{where}, which may twist the control net there. If the shift persists at a "
        "finer mm resolution, a finer mesh or a thinner wall will not fix it; soften the "
        "abrupt change that starts there (a Morph.Rate near zero, or cross-sections "
        "that change shape between two nearby stations)"
    )


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
    side1_segments = int(side1_segments)
    side2_segments = int(side2_segments)
    arc_subdivision = max(1, int(arc_subdivision))
    arc_segments = 3 * arc_subdivision
    total_segments = side1_segments + arc_segments + side2_segments
    uniform_angles, arc_trig = _freeform_rounded_rect_static_basis(
        side1_segments,
        side2_segments,
        arc_subdivision,
    )
    if corner >= b and corner >= a:
        # The cached basis is read-only and shared across builds; retain the
        # old function's fresh-array ownership on its direct-return branch.
        return uniform_angles.copy()

    span1 = theta1
    span2 = math.pi / 2.0 - theta2
    # With either fixed wall budget present, _rounded_rect_quadrant_layout can
    # only select ATH's canonical three arc intervals. The sole exception is a
    # zero-wall-budget, one-side collapse; keep its old low-budget resolution
    # (and consequent shape validation) without re-solving every normal ring.
    if (
        side1_segments + side2_segments == 0
        and corner > 1.0e-9
        and (corner >= a or corner >= b)
    ):
        base_layout = _rounded_rect_quadrant_layout(3, a, b, corner)
        arc_segments = (
            3 if base_layout is None else int(base_layout.arc_segments)
        ) * arc_subdivision

    structural_angles = np.empty(
        side1_segments + arc_segments + side2_segments + 1,
        dtype=np.float64,
    )
    cursor = 0
    if side1_segments:
        structural_angles[: side1_segments + 1] = np.linspace(
            0.0, theta1, side1_segments + 1
        )
        cursor = side1_segments + 1
    else:
        structural_angles[0] = theta1
        cursor = 1
    cx = a - corner
    cy = b - corner
    # The rare two-interval compatibility branch above cannot use the cached
    # three-interval basis. It is an invalid zero-wall-budget input in current
    # callers, but evaluating it the old way preserves its exception behavior.
    if len(arc_trig) != arc_segments:
        active_arc_trig = tuple(
            (
                math.sin(index * math.pi / (2.0 * arc_segments)),
                math.cos(index * math.pi / (2.0 * arc_segments)),
            )
            for index in range(1, arc_segments + 1)
        )
    else:
        active_arc_trig = arc_trig
    for arc_sin, arc_cos in active_arc_trig:
        structural_angles[cursor] = math.atan2(
            cy + corner * arc_sin,
            cx + corner * arc_cos,
        )
        cursor += 1
    if side2_segments:
        for index in range(1, side2_segments + 1):
            structural_angles[cursor] = (
                theta2
                + (math.pi / 2.0 - theta2) * index / side2_segments
            )
            cursor += 1
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
    merged = _pin_axial_stations(merged, params.get(ACOUSTIC_AXIAL_STATIONS_KEY))
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
    rectangle_morph = geometry._morph_target == 1
    if has_rounded_rectangle or rectangle_morph:
        angular_segments = _normalise_ath_angular_segments(
            int(params.get("angularSegments", 64))
        )
        points_per_quadrant = _morph_quadrant_budget(params, angular_segments)
        corner_segments = max(
            0,
            int(round(eval_param(params.get("cornerSegments"), 0.0, 0.0))),
        )
        arc_subdivision = _morph_corner_arc_subdivision(params)
        morph_start = _continuous_morph_start(params, 0.0)
        morph_corner = eval_param(params.get("morphCorner"), 0.0, 0.0)

        def effective_outline_parameters(
            t_value: float, station_a: float, station_b: float
        ) -> tuple[float, float, float]:
            if not rectangle_morph:
                return (
                    station_a,
                    station_b,
                    active_rounded_rect_corner_radius_mm(
                        geometry.stations, t_value, station_a, station_b
                    ),
                )

            # These effective values only choose where the analytic outline is
            # sampled; they do not define the surface. cross_section_radius is
            # authoritative and already includes the exact FREEFORM morph.
            effective_axes = geometry.cross_section_radius(
                np.asarray([0.0, math.pi / 2.0]), t_value
            )
            factor = _morph_factor(
                t_value, 0.0, params, morph_start=morph_start
            )
            if has_rounded_rectangle:
                base_corner = active_rounded_rect_corner_radius_mm(
                    geometry.stations, t_value, station_a, station_b
                )
                effective_corner = (
                    (1.0 - factor) * base_corner + factor * morph_corner
                )
            else:
                # A wholly smooth schedule has no structural corner descriptor.
                # Avoid the existing nearest-station fallback's empty sequence.
                effective_corner = factor * morph_corner
            return (
                float(effective_axes[0]),
                float(effective_axes[1]),
                float(effective_corner),
            )

        reference_a, reference_b, reference_corner = effective_outline_parameters(
            float(t_values[-1]), float(radii_h[-1]), float(radii_v[-1])
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
            if rectangle_morph:
                # A sharp or fully rounded mouth has no finite corner arc from
                # which to recover wall budgets. Use a non-degenerate sampling
                # proxy so intermediate emerging corners retain rows on both
                # walls; the proxy never participates in surface evaluation.
                reference_layout = _rounded_rect_quadrant_layout(
                    points_per_quadrant,
                    reference_a,
                    reference_b,
                    0.5 * min(reference_a, reference_b),
                )
                if reference_layout is None:
                    side1_segments = max(0, points_per_quadrant - 3)
                    side2_segments = 0
                else:
                    side1_segments = int(reference_layout.side1_segments)
                    side2_segments = int(reference_layout.side2_segments)
            else:
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
        # A wholly smooth schedule morphing to a rectangle has no structural
        # corner before the morph: its rings start on the uniform layout.
        smooth_rectangle_morph = rectangle_morph and not has_rounded_rectangle
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
            effective_a, effective_b, corner_radius = effective_outline_parameters(
                float(t_value), float(a), float(b)
            )
            if rectangle_morph and corner_radius <= 1.0e-9:
                q1 = np.linspace(
                    0.0,
                    math.pi / 2.0,
                    side1_segments
                    + 3 * max(1, int(arc_subdivision))
                    + side2_segments
                    + 1,
                    dtype=np.float64,
                )
            else:
                q1 = _freeform_rounded_rect_quadrant_angles(
                    half_width=effective_a,
                    half_height=effective_b,
                    corner_radius=corner_radius,
                    side1_segments=side1_segments,
                    side2_segments=side2_segments,
                    arc_subdivision=arc_subdivision,
                    collapse_transition_intervals=collapse_transition_intervals,
                )
            arc_is_exact = True
            if smooth_rectangle_morph and corner_radius > 1.0e-9:
                q1, arc_is_exact = _blend_toward_uniform_layout(
                    q1,
                    _morph_factor(
                        float(t_value), 0.0, params, morph_start=morph_start
                    ),
                )
            reduced, full_circle = _freeform_quadrant_angles(q1, quadrants)
            if np.any(np.diff(reduced) <= 0.0):
                raise ValueError(
                    f"FREEFORM ring {ring_index} azimuths must be strictly increasing"
                )
            ring_angles.append(reduced)
            span = rounded_rect_corner_arc_span(
                points_per_quadrant,
                effective_a,
                effective_b,
                corner_radius,
            )
            if span is None or not arc_is_exact:
                corner_arc_spans.append([])
            else:
                corner_arc_spans.append([float(span[0]), float(span[1])])
        row_counts = {len(values) for values in ring_angles}
        if len(row_counts) != 1:
            raise ValueError(
                "FREEFORM per-ring azimuth grids must have a constant row count"
            )
        phi_grid = np.column_stack(ring_angles)
        required_cardinals = np.asarray(
            {
                "1": (0.0, math.pi / 2.0),
                "12": (0.0, math.pi / 2.0, math.pi),
                "14": (-math.pi / 2.0, 0.0, math.pi / 2.0),
            }.get(
                quadrants,
                (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0),
            ),
            dtype=np.float64,
        )
        # One broadcast comparison keeps the full reduced-domain contract --
        # including protection against a future mirroring regression -- while
        # replacing thousands of tiny np.isclose calls per fine preview. The
        # (ring, cardinal) matrix is row-major so argwhere retains the old first
        # missing ring, then first required-cardinal error ordering.
        has_cardinal = np.any(
            np.isclose(
                phi_grid[:, :, None],
                required_cardinals[None, None, :],
                rtol=0.0,
                atol=1.0e-12,
            ),
            axis=0,
        )
        missing = np.argwhere(~has_cardinal)
        if len(missing):
            ring_index, cardinal_index = (int(value) for value in missing[0])
            cardinal = float(required_cardinals[cardinal_index])
            raise ValueError(
                f"FREEFORM ring {ring_index} azimuths omit required "
                f"cardinal {cardinal:g} rad"
            )
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

    # Any rounded-rectangle station or static rectangle morph selects the
    # structural corner-aware family for every ring so tangencies cannot alias.
    # Smooth station rings use the nearest descriptor for harmless pinning;
    # smooth pre-morph rings use the uniform member of the same fixed-row family.
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


def _reject_non_finite_grid(
    inner: np.ndarray, angles: np.ndarray, t_values: np.ndarray, formula: str
) -> None:
    """Refuse a grid with NaN/inf vertices instead of handing it downstream.

    A non-finite vertex has no meaning to OCC or gmsh; left in place it fails
    much later with an unrelated message, or renders as a broken preview.
    """

    finite = np.isfinite(inner).all(axis=2)
    if finite.all():
        return
    phi_index, t_index = (int(value) for value in np.argwhere(~finite)[0])
    phi_values = np.asarray(angles, dtype=np.float64)
    phi = float(phi_values[phi_index] if phi_values.ndim == 1 else phi_values[phi_index, t_index])
    raise ValueError(
        f"{formula} geometry is not finite at phi={math.degrees(phi) % 360.0:.1f} deg, "
        f"axial station {t_index} of {len(t_values) - 1} (t={float(t_values[t_index]):.4g}); "
        "check the profile parameters and expressions for values that leave "
        "the formula's domain"
    )


def build_point_grid(
    params: Mapping[str, Any], *, defer_osse_offset_repair: bool = False
) -> dict[str, Any]:
    """Point grid with the vertex data as the published flat lists."""

    grid = build_point_grid_arrays(
        params, defer_osse_offset_repair=defer_osse_offset_repair
    )
    inner = grid.pop("inner_grid")
    outer = grid.pop("outer_grid")
    return {
        "inner_points": inner.reshape(-1).tolist(),
        "outer_points": None if outer is None else outer.reshape(-1).tolist(),
        **grid,
    }


def build_point_grid_arrays(
    params: Mapping[str, Any], *, defer_osse_offset_repair: bool = False
) -> dict[str, Any]:
    """The same grid, with the vertices left as ``(phi, t, xyz)`` arrays.

    ``inner_points``/``outer_points`` are a flat-list wire contract, and every
    in-process consumer immediately reshapes them back into the arrays they
    were built from.  On a fine preview each of those round trips costs about
    13 ms to spell the list and 16 ms to parse it back, three times over, so
    the preview path takes ``inner_grid``/``outer_grid`` and never spells them.
    """

    if "sourceBody" in params:
        raise ValueError("standalone source bodies require their dedicated geometry API, not a horn point grid")
    if "absoluteAxialScale" in params:
        raise ValueError("absolute axial scale requires resolve_geometry or the complete preview API")
    if "mouthRoundoverRadiusMm" in params:
        raise ValueError("mouth roundover requires resolve_geometry or the complete preview API; a body-only grid would omit the lip")
    from .throat_adapter import resolve_adapter

    adapter = resolve_adapter(params)
    _validate_static_morph_target(params)
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
    if (
        formula == "OSSE"
        and _guiding_curve_active(params, 0.0)
        and not _cross_section_is_circular(exponent, aspect_ratio)
    ):
        raise ValueError(GCURVE_CROSS_SECTION_CONFLICT)
    phi_grid: np.ndarray | None = None
    if adapter is not None:
        angles, full_circle = _angle_list(params)
        t_unit_values, sampling_mode = _axial_sample_map(n_length, params)
        t_unit_values = _pin_axial_stations(
            t_unit_values, [0.5, *(params.get(ACOUSTIC_AXIAL_STATIONS_KEY) or ())]
        )
        t_unit_values = np.sort(np.append(t_unit_values[np.abs(t_unit_values-.5) > 1e-9], .5))
        t_values = t_unit_values
        n_length = len(t_values)-1
        t_max = 1.0
        z, r = adapter.evaluate(t_values)
        raw_radials = np.broadcast_to(r, (len(angles), len(t_values))).copy()
        z_values = np.broadcast_to(z, raw_radials.shape).copy()
        throat_prefix = _ThroatPrefix(.5, .5)
    elif formula == "FREEFORM":
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
        t_max = 1.0
        throat_prefix = _ThroatPrefix(0.0, 0.0)
    else:
        angles, full_circle = _angle_list(params)
        t_max = _rosse_tmax(params) if formula == "R-OSSE" else 1.0
        if formula == "ICW":
            # ICW samples uniformly in sigma (normalised arc length): it has no
            # ATH/R-OSSE reference axial table, and the kernel already concentrates
            # detail by arc length, so a uniform sigma grid is the natural mapping.
            # An explicit custom z-map cannot be honoured and must not be silently
            # ignored, whichever spelling asked for it. The ATH default map is
            # accepted and sampled uniformly: the acoustic fit requests it for
            # every non-FREEFORM formula.
            z_map_points = params.get(
                "zMapPoints", params.get("zmapPoints", params.get("ZMapPoints"))
            )
            requested_mode = _normalise_sampling_mode(
                params.get("samplingMode", params.get("sampling_mode")),
                z_map_points=z_map_points,
            )
            if requested_mode == "zmap" or z_map_points is not None:
                raise ValueError(
                    "ICW does not support samplingMode='zmap'/zMapPoints; "
                    "it always samples uniformly in normalised arc length"
                )
            t_unit_values = np.linspace(0.0, 1.0, n_length + 1, dtype=np.float64)
            sampling_mode = "uniform"
        else:
            t_unit_values, sampling_mode = _axial_sample_map(n_length, params)
        t_unit_values = _pin_axial_stations(
            t_unit_values, params.get(ACOUSTIC_AXIAL_STATIONS_KEY)
        )
        n_length = len(t_unit_values) - 1
        t_values = t_unit_values * t_max
        raw_radials, z_values, throat_prefix = _raw_radial_grid(
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
    if morph_target in {1, 2, 3}:
        if morph_target == 2 and uses_text_import_geometry(params):
            raw_half_width = raw_half_height = float(np.max(raw_radials[:, -1]))
        # A shape-only superellipse morph (3) preserves the exact raw mouth
        # extents; ATH rounds the implicit rectangle/circle extents up to whole
        # millimetres per half-dimension.
        resolved_half_width, resolved_half_height = _resolve_morph_half_dimensions(
            params,
            0.0,
            raw_half_width,
            raw_half_height,
            round_implicit_up=morph_target != 3,
        )
        # FREEFORM samples its own azimuths per ring, morph included, in
        # ``_freeform_raw_radial_grid``. Re-deriving the rectangle-morph angle
        # list here and resampling through ``_raw_radial_grid`` evaluated the
        # R-OSSE formula on a FREEFORM design: a 100 mm horn came back as the
        # 40 mm default R-OSSE whenever the two angle lists differed.
        if morph_target == 1 and formula != "FREEFORM":
            new_angles, full_circle = _angle_list(
                params,
                morph_half_width=resolved_half_width,
                morph_half_height=resolved_half_height,
            )
            if len(new_angles) != len(angles) or not np.allclose(new_angles, angles):
                angles = new_angles
                raw_radials, z_values, throat_prefix = _raw_radial_grid(
                    params, angles, t_values, t_unit_values, formula, exponent, aspect_ratio, n_length
                )

    morph_corner_arc_span = _morph_corner_arc_span(
        params, resolved_half_width, resolved_half_height
    )

    morph_schedule = _morph_schedule(
        params,
        formula,
        t_unit_values=t_unit_values,
        t_values=t_values,
        t_max=t_max,
        throat_prefix=throat_prefix,
    )

    # The morph is a per-point no-op unless morphTarget resolves to a
    # morph shape (1/2/3). When the param is absent or a plain non-morph
    # constant it cannot activate at any azimuth — skip the n_phi * n_length
    # no-op calls. Expression values may vary with phi, so they keep the
    # per-point path.
    morph_param = params.get("morphTarget")
    if morph_param is None:
        morph_possible = False
    elif isinstance(morph_param, (int, float)):
        morph_possible = int(round(float(morph_param))) in {1, 2, 3}
    else:
        morph_possible = True

    inner = np.empty((len(angles), n_length + 1, 3), dtype=np.float64)
    if formula == "FREEFORM":
        inner[:, :, 0] = raw_radials * np.cos(phi_grid)
        inner[:, :, 1] = raw_radials * np.sin(phi_grid)
        inner[:, :, 2] = z_values
    else:
        radials = raw_radials
        if morph_possible:
            # The morph is a directional target-mouth rule whose target radius
            # and blend rate depend on the azimuth alone; only the blend factor
            # varies along t. Resolving the target once per meridian replaces
            # n_phi * n_length target solves with n_phi of them. Morph progress
            # is the global normalized axial position (z / L for OSSE),
            # identical for every azimuth: ATH does not shift the blend by the
            # per-azimuth slot length.
            morph_progress = morph_schedule.progress
            morph_progress_start = morph_schedule.start
            radials = raw_radials.copy()
            for i, phi in enumerate(angles):
                phi_value = float(phi)
                factors = _morph_factors(
                    morph_progress,
                    phi_value,
                    params,
                    morph_start=morph_progress_start,
                )
                blended = factors > 0.0
                if not blended.any():
                    continue
                mouth_radial = float(raw_radials[i, -1])
                target_radius = _morph_target_radius_at_angle(
                    mouth_radial,
                    phi_value,
                    params,
                    implicit_half_width=resolved_half_width,
                    implicit_half_height=resolved_half_height,
                )
                radials[i, blended] = (
                    raw_radials[i, blended]
                    + (target_radius - mouth_radial) * factors[blended]
                )
        inner[:, :, 0] = radials * np.cos(angles)[:, None]
        inner[:, :, 1] = radials * np.sin(angles)[:, None]
        inner[:, :, 2] = z_values

    _reject_non_finite_grid(inner, angles if phi_grid is None else phi_grid, t_values, formula)

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
    outer_fold: str | None = None
    wall = float(eval_param(params.get("wallThickness"), 0.0, 0.0))
    enc_depth = float(eval_param(params.get("encDepth"), 0.0, 0.0))
    if enc_depth <= 0.0 and wall > 0.0:
        outer = _outer_offset_shell(
            inner,
            wall,
            full_circle=full_circle,
            repair_osse=formula == "OSSE" and not defer_osse_offset_repair,
            t_coordinates=np.asarray(t_values, dtype=np.float64),
            # ``phi_grid`` is (phi, t); the derivative helper wants (t, phi).
            # Outside FREEFORM only the morph angle list is non-uniform, but it
            # is non-uniform exactly where the corner arc turns fastest.
            phi_coordinates=(
                phi_grid.T
                if phi_grid is not None
                else np.broadcast_to(
                    np.asarray(angles, dtype=np.float64), inner.shape[1::-1]
                )
            ),
        )
        # FREEFORM rejects a folded shell outright: its profile is user-drawn,
        # so a fold means the input is wrong and refusing is the right answer.
        #
        # OSSE's normal correspondence can fold inside concave guiding-curve
        # grooves at any axial station. The constant-distance exterior still
        # exists there: discard the internal loops by resampling its envelope.
        # Other formulas retain the existing warning contract.
        if formula == "FREEFORM":
            try:
                validate_outer_offset_grid(inner, outer, full_circle=full_circle)
            except ValueError as exc:
                note = (
                    freeform_azimuth_twist_note(phi_grid, t_values)
                    if phi_grid is not None
                    else ""
                )
                if note:
                    raise ValueError(f"{exc}; {note}") from exc
                raise
        else:
            try:
                validate_outer_offset_grid(
                    inner, outer, full_circle=full_circle, label=str(formula)
                )
            except ValueError as exc:
                # Reported as well as logged. A log line reaches nobody who is
                # looking at a polar plot, and a folded outer shell is still a
                # solver boundary the design does not describe -- "the acoustic
                # surface is unaffected" is true of the analytic profile, not of
                # what the solver ends up integrating over.
                outer_fold = str(exc)
                if not (formula == "OSSE" and defer_osse_offset_repair):
                    logger.warning(
                        "[hornlab-mesher] outer wall self-intersects: %s "
                        "Reduce the wall thickness or open the local curvature; the "
                        "acoustic (inner) surface is unaffected.",
                        exc,
                    )

    return {
        **({"construction_fingerprint": adapter.fingerprint,
            "semantic_stations": {"driver": 0.0, "adapter_join": .5, "mouth": 1.0}}
           if adapter is not None else {}),
        "inner_grid": inner,
        "outer_grid": outer,
        "outer_offset_fold": outer_fold,
        "grid_n_phi": int(inner.shape[0]),
        "grid_n_length": int(n_length),
        "full_circle": bool(full_circle),
        "quadrants": quadrants,
        "symmetry_planes": list(symmetry_planes),
        "vertical_offset_mm": vertical_offset,
        "angle_list": angles.tolist(),
        "slice_map": t_values.tolist(),
        "sampling_mode": sampling_mode,
        # Where the morph blend starts, in blend progress and as the axial
        # station (normalised ``t``) that carries it; ``None`` for FREEFORM,
        # whose morph is evaluated analytically. The acoustic fit reuses these
        # so the solve grid morphs exactly the surface the requested grid shows.
        "morph_start": None if formula == "FREEFORM" else morph_schedule.start,
        "morph_start_station": (
            None if formula == "FREEFORM" else morph_schedule.start_station
        ),
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
