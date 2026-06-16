from __future__ import annotations

import json
import math
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .profile_common import _DEFAULTS, _deg, _normalise_formula, _osse_radius, eval_param
from .profile_morph import _coverage_angle_from_guiding_curve

if TYPE_CHECKING:  # avoid importing the ICW kernel (and its scipy deps) at module import time
    from .icw import ICWCurve


def _circular_arc_radius(
    z_main: float,
    p: float,
    params: Mapping[str, Any],
    *,
    r0_main: float,
    mouth_radius: float,
    length: float,
) -> float:
    p1 = (0.0, r0_main)
    p2 = (length, mouth_radius)
    center: tuple[float, float] | None = None
    arc_radius = eval_param(
        params.get("circArcRadius", params.get("circ_arc_radius")),
        p,
        0.0,
    )

    if math.isfinite(arc_radius) and arc_radius > 0.0:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        chord = math.hypot(dx, dy)
        if chord > 0.0 and arc_radius >= chord / 2.0:
            mid_x = (p1[0] + p2[0]) / 2.0
            mid_y = (p1[1] + p2[1]) / 2.0
            offset = math.sqrt(max(0.0, arc_radius * arc_radius - (chord / 2.0) ** 2))
            nx = -dy / chord
            ny = dx / chord
            c1 = (mid_x + nx * offset, mid_y + ny * offset)
            c2 = (mid_x - nx * offset, mid_y - ny * offset)
            center = c1 if mouth_radius > r0_main else c2

    if center is None:
        term_angle = eval_param(
            params.get("circArcTermAngle", params.get("circ_arc_term_angle")),
            p,
            1.0,
        )
        tangent_angle = math.radians(term_angle)
        tx = math.cos(tangent_angle)
        ty = math.sin(tangent_angle)
        nx = -ty
        ny = tx
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dot = dx * nx + dy * ny
        if abs(dot) > 1.0e-6:
            arc_radius = -((dx * dx + dy * dy) / (2.0 * dot))
            center = (p2[0] + nx * arc_radius, p2[1] + ny * arc_radius)

    if center is None or not math.isfinite(arc_radius) or arc_radius == 0.0:
        return mouth_radius

    dx_center = z_main - center[0]
    under = arc_radius * arc_radius - dx_center * dx_center
    if under < 0.0:
        return mouth_radius

    sign = 1.0 if mouth_radius - center[1] >= 0.0 else -1.0
    return center[1] + sign * math.sqrt(under)


def calculate_osse(
    z: float,
    p: float,
    params: Mapping[str, Any],
    *,
    coverage_angle: float | None = None,
) -> tuple[float, float]:
    L, _, ext_len, slot_len = osse_length_config(params, p)
    r0_base = eval_param(params.get("r0"), p, 12.7)
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
        main_params = {**params, "L": L}
        active_a_deg = coverage_angle
        if active_a_deg is None:
            active_a_deg = _coverage_angle_from_guiding_curve(
                p,
                main_params,
                main_length=L,
                a0_deg=a0_deg,
                r0_main=r0_main,
            )
        if active_a_deg is None:
            active_a_deg = a_deg
        throat_profile = int(
            eval_param(params.get("throatProfile", params.get("throat_profile")), p, 1.0)
            or 1
        )
        if throat_profile == 3:
            mouth_radius = r0_main + L * math.tan(math.radians(active_a_deg))
            radius = _circular_arc_radius(
                main_z,
                p,
                main_params,
                r0_main=r0_main,
                mouth_radius=mouth_radius,
                length=L,
            )
        else:
            radius = _osse_radius(main_z, p, main_params, r0=r0_main, a_deg=active_a_deg, a0_deg=a0_deg)

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


def osse_length_config(params: Mapping[str, Any], p: float = 0.0) -> tuple[float, float, float, float]:
    raw_L = max(0.0, eval_param(params.get("L"), p, 120.0))
    ext_len = max(0.0, eval_param(params.get("throatExtLength"), p, 0.0))
    slot_len = max(0.0, eval_param(params.get("slotLength"), p, 0.0))
    length_mode = params.get("_athLengthMode", params.get("athLengthMode", params.get("lengthMode")))
    if length_mode == "total":
        return max(0.0, raw_L - ext_len - slot_len), raw_L, ext_len, slot_len
    return raw_L, raw_L + ext_len + slot_len, ext_len, slot_len


def osse_total_length(params: Mapping[str, Any], p: float = 0.0) -> float:
    return osse_length_config(params, p)[1]


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


# ============================================================================
# Intrinsic-Curvature Waveguide (ICW) adapter
# ----------------------------------------------------------------------------
# Bridges a mesher parameter dict to the gmsh-free ICW kernel in ``.icw``. The
# kernel is imported lazily inside ``build_icw_curve`` so importing this module
# does not pull in scipy, and so ``icw.seed`` (which imports ``profile_points``
# from here) cannot trigger a circular import at module load time. The kernel
# remains gmsh-free; only this adapter knows about both worlds.
# ============================================================================

# Dense sampling count for ``icw_meridian_points`` (kernel docstring: n>=~1500
# reproduces analytic seeds to sub-micron; 4001 leaves comfortable headroom).
_ICW_SAMPLE_N = 4001

# Module-level memo so a profile's ICWCurve is solved/fit ONCE per parameter
# set rather than per axial point or per azimuth (solving per point would be far
# too slow). Keyed on a hash of the ICW-relevant params; bounded in size.
# LRU memo (OrderedDict): most-recently-used entries are moved to the end on hit,
# and only the *oldest* entry is evicted on overflow (see ``_icw_cache_store``) --
# so a >256 distinct-geometry sweep keeps recent curves warm instead of wiping the
# whole cache and re-solving everything (the old clear-all behaviour thrashed).
_ICW_CURVE_CACHE: "OrderedDict[str, ICWCurve]" = OrderedDict()
_ICW_CACHE_MAX = 256

# Param keys that affect the ICW curve. Used both for the cache key and as the
# allow-list of top-level ICW keys the config validator accepts.
_ICW_PARAM_KEYS = (
    "type",
    "r0",
    "a0",
    "a0_deg",
    "theta0_deg",
    "kappa0",
    "n_coeff",
    "termination",
    "L",
    "L_mm",
    "R",
    "R_mm",
    "theta1",
    "theta1_deg",
    "r_aperture",
    "x_aperture",
    "depth",
    "x_setback",
    "icw_seed",
    "icw_coeffs",
    "icw_S",
)


def _icw_key_normalise(value: Any) -> Any:
    """Recursively coerce a param value into a *lossless*, hashable-as-JSON form.

    The previous ``json.dumps(..., default=str)`` stringified numpy arrays via
    ``str()``, which summarises large arrays ("[a b ... y z]") and rounds to ~8
    significant figures -- so two genuinely different ``icw_coeffs`` arrays could
    collapse to the SAME key and the memo would return a stale/wrong curve (a
    Phase-2 CMA optimiser passing numpy ``icw_coeffs`` is the live risk path).

    Arrays / lists / tuples are encoded with their exact bytes (shape + dtype +
    ``tobytes().hex()``) so every distinct float is reflected in the key; nested
    mappings (e.g. ``icw_seed``) recurse; plain scalars pass through unchanged so
    ``r0``/``a0``/``L``/``R``/``theta1`` etc. still key as before.
    """
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return ["__ndarray__", list(arr.shape), str(arr.dtype), arr.tobytes().hex()]
    if isinstance(value, (list, tuple)):
        # Numeric sequences (incl. ones holding numpy scalars) are hashed losslessly
        # through the same byte path as arrays so e.g. a list ``icw_coeffs`` differing
        # by 1e-12 keys distinctly. Mixed/non-numeric sequences recurse element-wise.
        try:
            arr = np.asarray(value, dtype=np.float64)
        except (ValueError, TypeError):
            return ["__seq__", [_icw_key_normalise(v) for v in value]]
        if arr.dtype == np.float64 and arr.ndim >= 1:
            return ["__ndarray__", list(arr.shape), str(arr.dtype), arr.tobytes().hex()]
        return ["__seq__", [_icw_key_normalise(v) for v in value]]
    if isinstance(value, Mapping):
        return {str(k): _icw_key_normalise(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, np.generic):  # numpy scalar -> exact python scalar
        return value.item()
    return value


def _icw_cache_key(params: Mapping[str, Any]) -> str:
    """Stable, *lossless* hash over the ICW-relevant params for the curve memo.

    Array-like params (numpy arrays / numeric lists, possibly nested in
    ``icw_seed``) are normalised with full precision by :func:`_icw_key_normalise`
    before serialising, so differing coefficient arrays can never collide.
    """
    relevant = {k: _icw_key_normalise(params.get(k)) for k in _ICW_PARAM_KEYS if k in params}
    try:
        blob = json.dumps(relevant, sort_keys=True, default=repr)
    except TypeError:
        blob = repr(sorted(relevant.items(), key=lambda kv: kv[0]))
    return blob


def _icw_float(params: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        if name in params and params[name] is not None:
            return float(eval_param(params[name], 0.0, 0.0))
    return None


def build_icw_curve(params: Mapping[str, Any], phi: float = 0.0) -> "ICWCurve":
    """Build (or fetch from the memo) the ICWCurve for a mesher param dict.

    Three input modes, checked in this order:

    1. **SEED** -- ``params["icw_seed"]`` is a nested OSSE/R-OSSE param dict
       (carrying its own ``type``); fit it with ``seed_from_osse`` /
       ``seed_from_rosse``. Enables migration and the meridian-parity test.
    2. **DIRECT** -- ``params["icw_coeffs"]`` (list) plus optional ``icw_S``,
       ``r0``, ``a0``/``theta0_deg`` -> construct ``ICWCurve`` directly.
    3. **TARGETS** (default) -- assemble ``ICWTargets`` from the params and call
       ``solve_icw``; raise ``ValueError`` (never a silent bad curve) if the
       returned ``FeasibilityReport`` is infeasible.

    The result is cached (module-level memo keyed on the ICW-relevant params) so
    the curve is solved/fit only once per profile. ``phi`` is accepted for API
    symmetry with the other formulas but is unused in Phase 1 (ICW is
    phi-independent: no guiding curve / no per-phi expressions).
    """
    key = _icw_cache_key(params)
    cached = _ICW_CURVE_CACHE.get(key)
    if cached is not None:
        _ICW_CURVE_CACHE.move_to_end(key)  # mark most-recently-used (LRU)
        return cached

    # Local import keeps the ICW/scipy dependency out of module import and
    # avoids the icw.seed -> profile_formulas import cycle.
    from .icw import (
        ICWCurve,
        ICWTargets,
        seed_from_osse,
        seed_from_rosse,
        solve_icw,
    )

    # (1) SEED mode -----------------------------------------------------------
    seed = params.get("icw_seed")
    if isinstance(seed, Mapping):
        seed_formula = _normalise_formula(seed.get("type", "OSSE"))
        # Honor a caller-supplied n_coeff for the seed fit. It was silently dropped before, so a
        # request for a finer basis (e.g. to refine a sharp/cusped OSSE seed) still returned the
        # default 20-coefficient fit.
        n_coeff_val = _icw_float(params, "n_coeff")
        seed_kwargs = {} if n_coeff_val is None else {"n_coeff": int(n_coeff_val)}
        if seed_formula == "OSSE":
            curve = seed_from_osse(dict(seed), **seed_kwargs)
        elif seed_formula == "R-OSSE":
            curve = seed_from_rosse(dict(seed), **seed_kwargs)
        else:
            raise ValueError(
                f"icw_seed type must be OSSE or R-OSSE, got {seed.get('type')!r}"
            )
        _icw_cache_store(key, curve)
        return curve

    r0 = _icw_float(params, "r0")
    if r0 is None:
        r0 = 12.7
    # Throat half-angle: a0 / a0_deg / theta0_deg are synonyms here (deg).
    theta0_deg = _icw_float(params, "theta0_deg", "a0", "a0_deg")
    if theta0_deg is None:
        theta0_deg = 0.0

    # (2) DIRECT mode ---------------------------------------------------------
    if params.get("icw_coeffs") is not None:
        coeffs = [float(eval_param(c, 0.0, 0.0)) for c in params["icw_coeffs"]]
        S = _icw_float(params, "icw_S")
        if S is None:
            raise ValueError("ICW direct mode (icw_coeffs) requires icw_S (arc length, mm)")
        coeffs_arr = np.asarray(coeffs, dtype=np.float64)
        # Direct mode bypasses solve_icw's feasibility gate, so validate the raw inputs and the
        # sampled meridian here. Otherwise a degenerate input (r0<=0, S<=0, non-finite coeffs, or
        # coeffs that drive the radius negative) builds a nonphysical curve that the caller would
        # score as valid instead of routing it to the infeasible/penalty path.
        if not np.all(np.isfinite(coeffs_arr)):
            raise ValueError("ICW direct mode: icw_coeffs must all be finite")
        if not (S > 0.0):
            raise ValueError(f"ICW direct mode: icw_S must be > 0 mm, got {S}")
        if not (r0 > 0.0):
            raise ValueError(f"ICW direct mode: r0 must be > 0 mm, got {r0}")
        curve = ICWCurve(
            coeffs=coeffs_arr,
            S=float(S),
            r0=float(r0),
            theta0=math.radians(float(theta0_deg)),
        )
        # Sample finely enough to catch a narrow negative-radius excursion even for a high-coeff
        # curve: the default 1601 stations can step over a dip between many close knot spans, so
        # scale the check resolution with the coefficient count (no effect for typical n_coeff).
        n_check = max(1601, 16 * coeffs_arr.size)
        samp = curve.sample(n_check)
        if not (
            np.all(np.isfinite(samp.x))
            and np.all(np.isfinite(samp.r))
            and np.all(samp.r > 0.0)
        ):
            raise ValueError(
                "ICW direct mode: sampled meridian is non-finite or has non-positive radius "
                "(degenerate icw_coeffs / icw_S)"
            )
        _icw_cache_store(key, curve)
        return curve

    # (3) TARGETS mode (default) ---------------------------------------------
    termination = str(params.get("termination", "flat_baffle")).strip().lower()
    kappa0 = _icw_float(params, "kappa0")
    n_coeff_val = _icw_float(params, "n_coeff")
    n_coeff = int(n_coeff_val) if n_coeff_val is not None else 12

    target_kwargs: dict[str, Any] = {
        "mode": termination,
        "r0": float(r0),
        "theta0_deg": float(theta0_deg),
    }
    if kappa0 is not None:
        target_kwargs["kappa0"] = float(kappa0)

    if termination == "flat_baffle":
        x_target = _icw_float(params, "L", "L_mm")
        r_mouth = _icw_float(params, "R", "R_mm")
        target_kwargs["x_target"] = x_target
        target_kwargs["r_mouth"] = r_mouth
    elif termination == "rollback":
        theta1 = _icw_float(params, "theta1", "theta1_deg")
        if theta1 is not None:
            target_kwargs["theta1_deg"] = float(theta1)
        r_aperture = _icw_float(params, "R", "r_aperture")
        if r_aperture is not None:
            target_kwargs["r_aperture"] = float(r_aperture)
        x_aperture = _icw_float(params, "x_aperture")
        depth = _icw_float(params, "depth")
        if x_aperture is not None:
            target_kwargs["x_aperture"] = float(x_aperture)
        if depth is not None:
            target_kwargs["depth"] = float(depth)
        x_setback = _icw_float(params, "x_setback")
        if x_setback is not None:
            target_kwargs["x_setback"] = float(x_setback)
    else:
        raise ValueError(
            f"ICW termination must be 'flat_baffle' or 'rollback', got {termination!r}"
        )

    targets = ICWTargets(**target_kwargs)
    curve, report = solve_icw(targets, n_coeff=n_coeff)
    if not report.feasible:
        raise ValueError(
            "ICW target set is infeasible: "
            + "; ".join(report.violations)
            + (f" (hint: {report.suggested_relaxation})" if report.suggested_relaxation else "")
        )
    _icw_cache_store(key, curve)
    return curve


def _icw_cache_store(key: str, curve: "ICWCurve") -> None:
    """Insert ``curve`` under ``key`` with bounded LRU eviction.

    On overflow only the *oldest* (least-recently-used) entry is dropped --
    ``popitem(last=False)`` -- rather than clearing the whole cache, so recent
    curves survive a long distinct-geometry sweep. Re-storing an existing key
    refreshes its recency.
    """
    if key in _ICW_CURVE_CACHE:
        _ICW_CURVE_CACHE.move_to_end(key)
    _ICW_CURVE_CACHE[key] = curve
    while len(_ICW_CURVE_CACHE) > _ICW_CACHE_MAX:
        _ICW_CURVE_CACHE.popitem(last=False)


def icw_meridian_points(curve: "ICWCurve", t_values: np.ndarray) -> np.ndarray:
    """Sample an ICWCurve meridian at the requested ``t_values`` (sigma in [0,1]).

    The curve is finely sampled once (``curve.sample(_ICW_SAMPLE_N)``); x and r
    are then linearly interpolated at the requested ``t_values`` by normalised
    arc length ``sigma``. Returns an ``(N, 2)`` array of ``(x, r)`` columns.
    """
    sample = curve.sample(_ICW_SAMPLE_N)
    t = np.asarray(t_values, dtype=np.float64)
    x = np.interp(t, sample.sigma, sample.x)
    r = np.interp(t, sample.sigma, sample.r)
    return np.column_stack([x, r])


def profile_points(params: Mapping[str, Any], n_axial: int, phi: float = 0.0) -> np.ndarray:
    formula = _normalise_formula(params.get("type", "OSSE"))
    t_max = float(eval_param(params.get("tmax"), phi, 1.0)) if formula == "R-OSSE" else 1.0
    t_values = np.linspace(0.0, t_max, int(n_axial))
    if formula == "ICW":
        curve = build_icw_curve(params, phi)
        return icw_meridian_points(curve, t_values)
    points = np.empty((len(t_values), 2), dtype=np.float64)
    if formula == "OSSE":
        total = osse_total_length(params, phi)
        for idx, t in enumerate(t_values):
            points[idx] = calculate_osse(float(t) * total, phi, params)
    else:
        for idx, t in enumerate(t_values):
            points[idx] = calculate_rosse(float(t), phi, params)
    return points
