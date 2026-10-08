from __future__ import annotations

import json
import math
from collections import OrderedDict
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple

import numpy as np
from numpy.typing import NDArray

from .freeform import build_freeform_geometry
from .config_parser import ConfigError
from .throat_stretch import canonical_stretch_params, stretch_coefficients, validate_stretch_composition
from .profile_common import (
    _DEFAULTS,
    _lossless_key_value,
    _deg,
    _normalise_formula,
    _osse_radius,
    _osse_radius_curve,
    eval_param,
)
from .profile_morph import (
    CoverageInversion,
    _coverage_angle_from_guiding_curve,
    _solve_coverage_from_guiding_curve,
    coverage_angle_saturation,
)

if (
    TYPE_CHECKING
):  # avoid importing the ICW kernel (and its scipy deps) at module import time
    from .icw import ICWCurve


def _circular_arc_center(
    p: float,
    params: Mapping[str, Any],
    *,
    r0_main: float,
    mouth_radius: float,
    length: float,
) -> tuple[tuple[float, float] | None, float]:
    """Solve the terminating arc's centre and radius, which do not vary with z.

    Split out of :func:`_circular_arc_radius` so a whole meridian shares one
    solve instead of repeating it per axial station.
    """

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

    return center, arc_radius


def _circular_arc_radius(
    z_main: float,
    p: float,
    params: Mapping[str, Any],
    *,
    r0_main: float,
    mouth_radius: float,
    length: float,
) -> float:
    center, arc_radius = _circular_arc_center(
        p, params, r0_main=r0_main, mouth_radius=mouth_radius, length=length
    )
    if center is None or not math.isfinite(arc_radius) or arc_radius == 0.0:
        return mouth_radius

    dx_center = z_main - center[0]
    under = arc_radius * arc_radius - dx_center * dx_center
    if under < 0.0:
        return mouth_radius

    sign = 1.0 if mouth_radius - center[1] >= 0.0 else -1.0
    return center[1] + sign * math.sqrt(under)


def _circular_arc_radius_curve(
    z_main: NDArray[np.float64],
    p: float,
    params: Mapping[str, Any],
    *,
    r0_main: float,
    mouth_radius: float,
    length: float,
) -> NDArray[np.float64]:
    """Array form of :func:`_circular_arc_radius` for one azimuth."""

    center, arc_radius = _circular_arc_center(
        p, params, r0_main=r0_main, mouth_radius=mouth_radius, length=length
    )
    radius = np.full_like(z_main, mouth_radius)
    if center is None or not math.isfinite(arc_radius) or arc_radius == 0.0:
        return radius

    dx_center = z_main - center[0]
    under = arc_radius * arc_radius - dx_center * dx_center
    on_arc = under >= 0.0
    sign = 1.0 if mouth_radius - center[1] >= 0.0 else -1.0
    radius[on_arc] = center[1] + sign * np.sqrt(under[on_arc])
    return radius


class _CoverageProblem(NamedTuple):
    """Everything the four coverage entry points pass to the inversion."""

    main_params: dict[str, Any]
    main_length: float
    a0_deg: float
    r0_main: float
    radius_offset: Any


def _coverage_problem(params: Mapping[str, Any], p: float) -> _CoverageProblem:
    L, total, ext_len, slot_len = osse_length_config(params, p)
    h_bulge = eval_param(params.get("h"), p, 0.0)
    radius_offset = None
    if h_bulge != 0.0 and total > 0.0:
        # The sampler adds h*sin(pi*t) over the whole composite length
        # (extension and slot included), so the bulge at a main-section station
        # z_main sits at t = (ext + slot + z_main) / total.
        def radius_offset(z_main: float) -> float:
            return h_bulge * math.sin(math.pi * (ext_len + slot_len + z_main) / total)

    return _CoverageProblem(
        main_params={**params, "L": L},
        main_length=L,
        a0_deg=eval_param(params.get("a0"), p, 15.5),
        r0_main=eval_param(params.get("r0"), p, 12.7),
        radius_offset=radius_offset,
    )


def osse_coverage_angle(params: Mapping[str, Any], p: float) -> float | None:
    """Resolve the guiding-curve coverage angle for azimuth ``p`` once.

    The inversion (a bisection over the OSSE radius) depends only on the
    azimuth, not on z, so grid builders hoist it per azimuth and pass the
    result to :func:`calculate_osse` via ``coverage_angle`` instead of paying
    the bisection for every axial sample (~8x the plain grid cost otherwise).
    Returns ``None`` when no guiding curve is active.
    """
    problem = _coverage_problem(params, p)
    return _coverage_angle_from_guiding_curve(
        p,
        problem.main_params,
        main_length=problem.main_length,
        a0_deg=problem.a0_deg,
        r0_main=problem.r0_main,
        radius_offset=problem.radius_offset,
    )


def osse_coverage_inversion(
    params: Mapping[str, Any], p: float
) -> CoverageInversion | None:
    """Full coverage-inversion result at ``p``, or ``None`` without a guiding curve.

    Companion to :func:`osse_coverage_angle` for callers that need to know
    whether the guiding curve was actually met, and by how much it was missed.
    """

    problem = _coverage_problem(params, p)
    return _solve_coverage_from_guiding_curve(
        p,
        problem.main_params,
        main_length=problem.main_length,
        a0_deg=problem.a0_deg,
        r0_main=problem.r0_main,
        radius_offset=problem.radius_offset,
    )


def osse_coverage_saturation_probe(
    params: Mapping[str, Any], p: float
) -> CoverageInversion | None:
    """Saturation-only screen at ``p``: a saturated result, or ``None``.

    The cheap half of :func:`osse_coverage_inversion` -- it answers "is the
    guiding curve out of reach here" without bisecting for the angle that
    reaches it. ``None`` means "nothing to report" (reachable, no guiding
    curve, or a radius undefined at a bracket end), never "solved".

    The saturated result it returns is identical to the full inversion's, so a
    caller screening many azimuths can rank them on it and only pay the full
    inversion for the one it actually reports.
    """

    problem = _coverage_problem(params, p)
    return _solve_coverage_from_guiding_curve(
        p,
        problem.main_params,
        main_length=problem.main_length,
        a0_deg=problem.a0_deg,
        r0_main=problem.r0_main,
        probe_only=True,
        radius_offset=problem.radius_offset,
    )


def osse_coverage_saturation(
    params: Mapping[str, Any], p: float, *, location: str | None = None
) -> str | None:
    """Reason the guiding curve is unreachable at ``p``, or ``None`` if it is met.

    The coverage bisection clamps to its bracket rather than failing, so
    without this check an unreachable guiding curve produces a mouth that is
    silently off-target while every other parameter appears to stop responding.
    """

    problem = _coverage_problem(params, p)
    return coverage_angle_saturation(
        p,
        problem.main_params,
        main_length=problem.main_length,
        a0_deg=problem.a0_deg,
        r0_main=problem.r0_main,
        location=location,
        radius_offset=problem.radius_offset,
    )


def _validate_osse_termination(params: Mapping[str, Any], p: float) -> None:
    """Refuse termination parameters that silently switch the term off.

    ``_osse_radius`` skips the superellipse termination when ``n`` or ``q`` is
    not positive, so a sign slip in either built a different horn (82 mm
    instead of 117 mm at the mouth in the review case) with no diagnostic.
    """

    if eval_param(params.get("s"), p, 0.0) == 0.0:
        return  # no termination term to switch off
    for name, default in (("n", _DEFAULTS["n"]), ("q", _DEFAULTS["q"])):
        value = eval_param(params.get(name), p, default)
        if not value > 0.0:
            raise ValueError(
                f"OSSE termination parameter {name} must be > 0, got {value:g} at "
                f"phi={math.degrees(p) % 360.0:.1f} deg"
            )


def _stretch_x(x: float, s1: float, s2: float) -> float:
    if s1 == 0.0 or s2 == 0.0:
        return x
    result = x + s1 * math.degrees(math.atan(s2 * x))
    if not math.isfinite(result):
        raise ConfigError("throat stretch produced a non-finite axial coordinate")
    return result


def _stretch_x_curve(x: NDArray[np.float64], s1: float, s2: float) -> NDArray[np.float64]:
    if s1 == 0.0 or s2 == 0.0:
        return x
    # Finite coefficients can still overflow their product; atan(inf) is the
    # correct limiting angle, just as on the scalar path.
    with np.errstate(over="ignore"):
        result = x + s1 * np.degrees(np.arctan(s2 * x))
    _verify_stretched_axial_map(result)
    return result


def _verify_stretched_axial_map(stretched: Any) -> None:
    """Use the scalar path's analytical monotonicity and finite-output rule.

    The positive-coefficient map is analytically increasing everywhere.
    Adjacent representable inputs may round to equal outputs. That does not
    change acceptance on either path or reject R-OSSE's existing foldback.
    """
    if not np.all(np.isfinite(stretched)):
        raise ConfigError("throat stretch produced a non-finite axial coordinate")


def _verify_stretch_junction(prefix: tuple[float, float], main: tuple[float, float]) -> None:
    if not all(math.isfinite(v) for v in (*prefix, *main)) or any(
        abs(a - b) > 64 * np.finfo(float).eps * max(1.0, abs(a), abs(b))
        for a, b in zip(prefix, main)
    ):
        raise ConfigError("throat stretch composite meridian is not continuous at the prefix/main junction")


def _stretch_prefix_endpoint(r0: float, ext_len: float, slot_len: float, ext_angle: float) -> tuple[float, float]:
    prefix = ext_len + slot_len
    radius = r0
    if prefix <= ext_len:
        # Check the arithmetic the evaluator actually uses, including loss of
        # precision in a long tapered extension, rather than assuming r0.
        radius = _throat_extension_start_radius(r0, ext_len, ext_angle) + ext_len * math.tan(ext_angle)
    return prefix, radius


def _validate_osse_stretch_composition(
    params: Mapping[str, Any], p: float, s1: float, s2: float,
    L: float, ext_len: float, slot_len: float, coverage_angle: float | None,
) -> None:
    if s1 == 0.0 or s2 == 0.0 or ext_len + slot_len == 0.0:
        return
    r0 = eval_param(params.get("r0"), p, 12.7)
    a = coverage_angle
    if a is None:
        a = osse_coverage_angle(params, p)
    if a is None:
        a = eval_param(params.get("a"), p, 60.0)
    main_params = {**params, "L": L}
    throat_profile = int(eval_param(params.get("throatProfile", params.get("throat_profile")), p, 1.0) or 1)
    if throat_profile == 3:
        radius = _circular_arc_radius(
            0.0, p, main_params, r0_main=r0,
            mouth_radius=r0 + L * math.tan(math.radians(a)), length=L,
        )
    else:
        radius = _osse_radius(0.0, p, main_params, r0=r0, a_deg=a,
                              a0_deg=eval_param(params.get("a0"), p, 15.5))
    prefix = ext_len + slot_len
    endpoint = _stretch_prefix_endpoint(r0, ext_len, slot_len, _deg(params.get("throatExtAngle"), p, 0.0))
    _verify_stretch_junction(endpoint, (prefix + _stretch_x(0.0, s1, s2), radius))


def calculate_osse(
    z: float,
    p: float,
    params: Mapping[str, Any],
    *,
    coverage_angle: float | None = None,
) -> tuple[float, float]:
    if "sourceBody" in params:
        raise ValueError("standalone source bodies require their dedicated geometry API, not horn profiles")
    if "terminatingArc" in params:
        raise ValueError("terminating arcs require the complete canonical meridian API")
    if "absoluteAxialScale" in params:
        raise ValueError("absolute axial scale requires its canonical physical meridian")
    if "mouthRoundoverRadiusMm" in params:
        raise ValueError("mouth roundover requires the canonical composite meridian, not the body-only formula")
    if "throat_adapter" in params:
        from .throat_adapter import resolve_adapter

        adapter = resolve_adapter(params)
        if adapter is not None:
            if coverage_angle is not None:
                raise ValueError("Curved adapter refused: coverage overrides are not supported.")
            depth = adapter.payload["length_mm"] + adapter.length*(1-adapter.payload["join_t"])
            x, radius = adapter.evaluate(float(z)/depth)
            return float(x), float(radius)
    validate_stretch_composition(params, "OSSE")
    L, _, ext_len, slot_len = osse_length_config(params, p)
    s1, s2 = stretch_coefficients(params)
    _validate_osse_stretch_composition(params, p, s1, s2, L, ext_len, slot_len, coverage_angle)
    _validate_osse_termination(params, p)
    r0_base = eval_param(params.get("r0"), p, 12.7)
    ext_angle = _deg(params.get("throatExtAngle"), p, 0.0)
    # ATH anchors Throat.Diameter (r0) at the MAIN horn throat and tapers the throat
    # extension BACK from r0 to the driver end (r0 - ext*tan(angle)); it does not
    # enlarge the main throat. For a straight extension (angle 0) this is a plain r0
    # tube, identical to before.
    r0_main = r0_base
    r0_throat = _throat_extension_start_radius(r0_base, ext_len, ext_angle)
    a_deg = eval_param(params.get("a"), p, 60.0)
    a0_deg = eval_param(params.get("a0"), p, 15.5)

    if z <= ext_len:
        radius = r0_throat + z * math.tan(ext_angle)
    elif z <= ext_len + slot_len:
        radius = r0_main
    else:
        main_z = z - ext_len - slot_len
        main_params = {**params, "L": L}
        active_a_deg = coverage_angle
        if active_a_deg is None:
            active_a_deg = osse_coverage_angle(params, p)
        if active_a_deg is None:
            active_a_deg = a_deg
        throat_profile = int(
            eval_param(
                params.get("throatProfile", params.get("throat_profile")), p, 1.0
            )
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
            radius = _osse_radius(
                main_z, p, main_params, r0=r0_main, a_deg=active_a_deg, a0_deg=a0_deg
            )

    x = float(z)
    y = float(radius)
    rot_deg = eval_param(params.get("rot"), p, 0.0)
    if math.isfinite(rot_deg) and rot_deg != 0.0:
        rot = math.radians(rot_deg)
        dx = x
        dy = y - r0_base
        x = dx * math.cos(rot) - dy * math.sin(rot)
        y = r0_base + dx * math.sin(rot) + dy * math.cos(rot)
    # ATH V2025-12 stretches after Rot. Rot + prefix is refused above.
    # Keep the disabled path arithmetic unchanged.
    if z > ext_len + slot_len and s1 != 0.0 and s2 != 0.0:
        x = ext_len + slot_len + _stretch_x(x - ext_len - slot_len, s1, s2)
    return x, y


def calculate_osse_curve(
    z_values: Any,
    p: float,
    params: Mapping[str, Any],
    *,
    coverage_angle: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``calculate_osse`` over a whole axial station array at one azimuth.

    Returns ``(z, radius)`` arrays.  The scalar entry point stays the public
    single-point API and the differential oracle for this one; the grid
    builders call this so the parameter set, the coverage inversion and the
    terminating-arc solve are each paid once per meridian.
    """
    if "sourceBody" in params:
        raise ValueError("standalone source bodies require their dedicated geometry API, not horn profiles")
    if "terminatingArc" in params:
        raise ValueError("terminating arcs require the complete canonical meridian API")

    if "absoluteAxialScale" in params:
        raise ValueError("absolute axial scale requires its canonical physical meridian")
    if "mouthRoundoverRadiusMm" in params:
        raise ValueError("mouth roundover requires the canonical composite meridian, not the body-only formula")
    if "throat_adapter" in params:
        from .throat_adapter import resolve_adapter

        adapter = resolve_adapter(params)
        if adapter is not None:
            if coverage_angle is not None:
                raise ValueError("Curved adapter refused: coverage overrides are not supported.")
            depth = adapter.payload["length_mm"] + adapter.length*(1-adapter.payload["join_t"])
            return adapter.evaluate(np.asarray(z_values, dtype=np.float64)/depth)
    validate_stretch_composition(params, "OSSE")
    z = np.asarray(z_values, dtype=np.float64)
    L, _total, ext_len, slot_len = osse_length_config(params, p)
    s1, s2 = stretch_coefficients(params)
    _validate_osse_stretch_composition(params, p, s1, s2, L, ext_len, slot_len, coverage_angle)
    _validate_osse_termination(params, p)
    r0_base = eval_param(params.get("r0"), p, 12.7)
    ext_angle = _deg(params.get("throatExtAngle"), p, 0.0)
    r0_main = r0_base
    r0_throat = _throat_extension_start_radius(r0_base, ext_len, ext_angle)
    a_deg = eval_param(params.get("a"), p, 60.0)
    a0_deg = eval_param(params.get("a0"), p, 15.5)

    radius = np.empty_like(z)
    in_ext = z <= ext_len
    in_slot = ~in_ext & (z <= ext_len + slot_len)
    in_main = ~(in_ext | in_slot)
    radius[in_ext] = r0_throat + z[in_ext] * math.tan(ext_angle)
    radius[in_slot] = r0_main
    if in_main.any():
        main_z = z[in_main] - ext_len - slot_len
        main_params = {**params, "L": L}
        active_a_deg = coverage_angle
        if active_a_deg is None:
            active_a_deg = osse_coverage_angle(params, p)
        if active_a_deg is None:
            active_a_deg = a_deg
        throat_profile = int(
            eval_param(
                params.get("throatProfile", params.get("throat_profile")), p, 1.0
            )
            or 1
        )
        if throat_profile == 3:
            mouth_radius = r0_main + L * math.tan(math.radians(active_a_deg))
            radius[in_main] = _circular_arc_radius_curve(
                main_z,
                p,
                main_params,
                r0_main=r0_main,
                mouth_radius=mouth_radius,
                length=L,
            )
        else:
            radius[in_main] = _osse_radius_curve(
                main_z, p, main_params, r0=r0_main, a_deg=active_a_deg, a0_deg=a0_deg
            )

    x = z
    y = radius
    rot_deg = eval_param(params.get("rot"), p, 0.0)
    if math.isfinite(rot_deg) and rot_deg != 0.0:
        rot = math.radians(rot_deg)
        dx = x
        dy = y - r0_base
        x = dx * math.cos(rot) - dy * math.sin(rot)
        y = r0_base + dx * math.sin(rot) + dy * math.cos(rot)
    if s1 != 0.0 and s2 != 0.0:
        x = x.copy()
        x[in_main] = ext_len + slot_len + _stretch_x_curve(
            x[in_main] - ext_len - slot_len, s1, s2
        )
        _verify_stretched_axial_map(x)
    return x, y


def osse_length_config(
    params: Mapping[str, Any], p: float = 0.0
) -> tuple[float, float, float, float]:
    if "sourceBody" in params:
        raise ValueError("standalone source bodies require their dedicated geometry API, not horn profiles")
    if "terminatingArc" in params:
        raise ValueError("terminating arcs require their complete analytic envelope")
    raw_L = eval_param(params.get("L"), p, 120.0)
    if not raw_L > 0.0:
        # A zero or negative length used to clamp to 0 and build a plain r0
        # tube (plus whatever extension/slot there was) with no warning.
        raise ValueError(
            f"OSSE Length must be > 0, got {raw_L:g} at phi={math.degrees(p) % 360.0:.1f} deg"
        )
    ext_len = max(0.0, eval_param(params.get("throatExtLength"), p, 0.0))
    slot_len = max(0.0, eval_param(params.get("slotLength"), p, 0.0))
    length_mode = params.get(
        "_athLengthMode", params.get("athLengthMode", params.get("lengthMode"))
    )
    if length_mode == "total":
        if slot_len >= raw_L:
            # The slot is carved out of Length in this mode, so a slot at least
            # as long as Length leaves no OS-SE section at all: the horn used
            # to become a straight r0 tube without a word.
            raise ValueError(
                f"Slot.Length {slot_len:g} mm must be shorter than Length {raw_L:g} mm "
                "in total length mode, where the slot is carved out of Length "
                f"(phi={math.degrees(p) % 360.0:.1f} deg); shorten the slot or "
                "lengthen the horn"
            )
        # ATH adds Throat.Ext.Length on TOP of Length (the main horn and its mouth
        # radius are unchanged) but carves Slot.Length OUT of Length. So the main
        # section loses only the slot, and the total axial grows by the extension.
        # The invariant total == ext_len + slot_len + main_L still holds, so the
        # z-sweep in profile_sampling lands exactly at the mouth.
        return max(0.0, raw_L - slot_len), raw_L + ext_len, ext_len, slot_len
    return raw_L, raw_L + ext_len + slot_len, ext_len, slot_len


def osse_total_length(params: Mapping[str, Any], p: float = 0.0) -> float:
    if "sourceBody" in params:
        raise ValueError("standalone source bodies require their dedicated geometry API, not horn profiles")
    if "absoluteAxialScale" in params:
        raise ValueError("absolute axial scale requires its canonical physical length")
    if "mouthRoundoverRadiusMm" in params:
        raise ValueError("mouth roundover requires its full analytic envelope, not the body-only length")
    if "throat_adapter" in params:
        from .throat_adapter import resolve_adapter

        adapter = resolve_adapter(params)
        if adapter is not None:
            return adapter.payload["length_mm"] + adapter.length*(1-adapter.payload["join_t"])
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


def _throat_extension_start_radius(r0_base: float, ext_len: float, ext_angle: float) -> float:
    """Driver-end radius of a throat extension that tapers back from ``r0_base``.

    ATH anchors Throat.Diameter at the MAIN throat for both OSSE and R-OSSE
    and tapers the extension back to the driver end (verified against ath.exe
    GridExport for both formulas); the main curve and the mouth are unchanged
    by the extension.
    """
    r0_throat = r0_base - ext_len * math.tan(ext_angle)
    if r0_throat < 0.0:
        raise ValueError(
            f"Throat.Ext.Length {ext_len:g} at Throat.Ext.Angle tapers below zero radius "
            f"(implied driver-end radius {r0_throat:.3f} mm); shorten the extension or reduce the angle"
        )
    return r0_throat


def rosse_total_length(params: Mapping[str, Any], p: float = 0.0) -> float:
    if "sourceBody" in params:
        raise ValueError("standalone source bodies require their dedicated geometry API, not horn profiles")
    ext_len = max(0.0, eval_param(params.get("throatExtLength"), p, 0.0))
    slot_len = max(0.0, eval_param(params.get("slotLength"), p, 0.0))
    # Like ATH, the extension adds to the total length; the main R-OSSE curve
    # keeps r0 (Throat.Diameter) as its throat radius, unchanged by ext.
    return ext_len + slot_len + _rosse_length(params, p)


class _RosseMainCoefficients(NamedTuple):
    """Everything in the main R-OSSE curve that does not depend on ``t``.

    Splitting these out is what lets a whole meridian be evaluated with one
    parameter resolution instead of one per axial station: the grid builder
    walks ``t`` for a fixed azimuth, and every one of these terms -- including
    the ``_rosse_length`` solve and both tangents -- is constant along it.
    """

    R: float
    r0: float
    k: float
    q: float
    m: float
    r: float
    b: float
    L: float
    c1: float
    c2: float
    c3: float
    s1: float
    s2: float


def _rosse_main_coefficients(
    p: float, params: Mapping[str, Any]
) -> _RosseMainCoefficients:
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
    s1, s2 = stretch_coefficients(params)
    return _RosseMainCoefficients(
        R=R,
        r0=r0,
        k=k,
        q=q,
        m=m,
        r=r,
        b=b,
        L=L,
        c1=(k * r0) ** 2,
        c2=2 * k * r0 * math.tan(a0),
        c3=math.tan(a) ** 2,
        s1=s1,
        s2=s2,
    )


def _calculate_rosse_main(
    t: float, p: float, params: Mapping[str, Any]
) -> tuple[float, float]:
    c = _rosse_main_coefficients(p, params)
    R, r0, k, q, m, r, b, L, c1, c2, c3, s1, s2 = c

    x = L * (math.sqrt(r**2 + m**2) - math.sqrt(r**2 + (t - m) ** 2))
    x += b * L * (math.sqrt(r**2 + (1 - m) ** 2) - math.sqrt(r**2 + m**2)) * (t**2)
    throat_r = math.sqrt(c1 + c2 * L * t + c3 * (L * t) ** 2) + r0 * (1 - k)
    mouth_r = max(0.0, R + L * (1 - math.sqrt(1 + c3 * (t - 1) ** 2)))
    y = (1 - t**q) * throat_r + (t**q) * mouth_r
    return _stretch_x(x, s1, s2), y


def _verify_rosse_stretch_junction(params: Mapping[str, Any], p: float, r0: float,
                                  ext_len: float, slot_len: float, ext_angle: float) -> None:
    try:
        main_x, main_y = _calculate_rosse_main(0.0, p, params)
    except (ArithmeticError, ValueError) as exc:
        raise ConfigError(
            "throat stretch cannot establish a continuous prefix/main junction "
            f"for these R-OSSE parameters: {exc}"
        ) from exc
    prefix = ext_len + slot_len
    _verify_stretch_junction(_stretch_prefix_endpoint(r0, ext_len, slot_len, ext_angle),
                            (prefix + main_x, main_y))
    _verify_rosse_stretch_prefix_intersection(
        _rosse_main_coefficients(p, params), ext_len, slot_len, ext_angle, _rosse_tmax(params)
    )


@lru_cache(maxsize=512)
def _verify_rosse_stretch_prefix_intersection(
    coefficients: _RosseMainCoefficients, ext_len: float, slot_len: float,
    ext_angle: float, tmax: float,
) -> None:
    """Guard foldback against the unchanged straight prefix.

    A main-only map is injective within the main section but can move its
    foldback through a tapered prefix. Probe the complete main meridian even
    for a scalar call or a sparse requested array. The cache contains only
    validation results, keyed by every resolved coefficient and prefix input.
    """
    stations = np.linspace(0.0, tmax, 1025)
    main_x, radius = _rosse_main_curve(stations, coefficients)
    prefix = ext_len + slot_len
    points = list(zip(main_x + prefix, radius))
    if slot_len > 0:
        points.insert(0, (ext_len, coefficients.r0))
    if ext_len > 0:
        points.insert(0, (0.0, _throat_extension_start_radius(coefficients.r0, ext_len, ext_angle)))
    _verify_meridian_self_contact(np.asarray(points, dtype=np.float64))
    _verify_prefix_intersection(main_x + prefix, radius, coefficients.r0, ext_len, slot_len, ext_angle)
    _verify_rosse_source_contact(coefficients, stations, main_x + prefix, radius,
                                 ext_len, slot_len, ext_angle)


def _verify_rosse_source_contact(coefficients: _RosseMainCoefficients,
                                stations: NDArray[np.float64], x: NDArray[np.float64],
                                radius: NDArray[np.float64], ext_len: float,
                                slot_len: float, ext_angle: float) -> None:
    """Refine contacts with the exact prefix lines and the driver plane.

    Chords can miss a continuous-curve contact at a prefix endpoint between
    probe stations. Locate roots and local distance minima on the curve itself,
    and explicitly include its terminating endpoint. The driver disc includes
    its rim, so returning through its interior is also invalid.
    """
    from scipy.optimize import brentq, minimize_scalar

    tolerance = 1e-7
    prefix = ext_len + slot_len

    def point(t):
        z, r = _rosse_main_curve(np.array([t]), coefficients)
        return float(z[0] + prefix), float(r[0])

    def roots(values, evaluate):
        candidates = [float(stations[-1])]
        for i in np.flatnonzero(values[:-1] * values[1:] < 0):
            candidates.append(brentq(evaluate, stations[i], stations[i + 1], xtol=1e-14))
        for i in np.flatnonzero(np.abs(values) <= tolerance):
            candidates.append(float(stations[i]))
        distances = np.abs(values)
        minima = np.flatnonzero((distances[1:-1] < distances[:-2]) &
                                (distances[1:-1] <= distances[2:])) + 1
        for i in minima:
            result = minimize_scalar(lambda t: abs(evaluate(t)),
                                     bounds=(stations[i - 1], stations[i + 1]),
                                     method="bounded", options={"xatol": 1e-14})
            if result.fun <= tolerance:
                candidates.append(float(result.x))
        return (t for t in candidates if t > 1e-12)

    driver_radius = _throat_extension_start_radius(coefficients.r0, ext_len, ext_angle)
    for t in roots(x, lambda t: point(t)[0]):
        z, r = point(t)
        if abs(z) <= tolerance and -tolerance <= r <= driver_radius + tolerance:
            raise ConfigError("throat stretch main meridian has self-contact with the source/driver ring or disc")
    for start, end, slope in ((0.0, ext_len, math.tan(ext_angle)),
                              (ext_len, prefix, 0.0)):
        if end <= start:
            continue
        values = radius - coefficients.r0 - (x - end) * slope
        def distance(t):
            z, r = point(t)
            return r - coefficients.r0 - (z - end) * slope
        for t in roots(values, distance):
            z, _ = point(t)
            if start - tolerance <= z <= end + tolerance and abs(distance(t)) <= tolerance:
                raise ConfigError("throat stretch main meridian has self-contact or self-intersection with the prefix, including its endpoints")


def _verify_meridian_self_contact(points: NDArray[np.float64]) -> None:
    """Include endpoints and collinear contacts, exempting only adjacent joins.

    The exact driver point and prefix segments are part of this polyline. The
    tolerance is the OCC geometric tolerance in millimetres, so endpoint
    roundoff cannot let two rings reach a consumer as coincident topology.
    """
    from .config_parser import ConfigError

    if not np.all(np.isfinite(points)):
        raise ConfigError("throat stretch composite meridian has non-finite points")
    tolerance = 1e-7
    starts, ends = points[:-1], points[1:]
    delta = ends - starts
    lower, upper = np.minimum(starts, ends), np.maximum(starts, ends)

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def point_distance(point, a, direction):
        length_sq = np.sum(direction * direction, axis=-1)
        fraction = np.divide(np.sum((point - a) * direction, axis=-1), length_sq,
                             out=np.zeros_like(length_sq), where=length_sq > 0)
        projection = a + np.clip(fraction, 0, 1)[..., None] * direction
        return np.linalg.norm(point - projection, axis=-1)

    for i in range(len(starts) - 2):
        indices = np.arange(i + 2, len(starts))
        overlap = np.all((lower[i] <= upper[indices] + tolerance) &
                         (lower[indices] <= upper[i] + tolerance), axis=1)
        indices = indices[overlap]
        if not len(indices):
            continue
        a, b = starts[indices], ends[indices]
        directions = delta[indices]
        separation = a - starts[i]
        denominator = cross(delta[i], directions)
        nonparallel = denominator != 0
        u = np.divide(cross(separation, directions), denominator,
                      out=np.full_like(denominator, -1), where=nonparallel)
        v = np.divide(cross(separation, delta[i]), denominator,
                      out=np.full_like(denominator, -1), where=nonparallel)
        crossing = nonparallel & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        contact = np.minimum.reduce([
            point_distance(starts[i], a, directions),
            point_distance(ends[i], a, directions),
            point_distance(a, starts[i], np.broadcast_to(delta[i], directions.shape)),
            point_distance(b, starts[i], np.broadcast_to(delta[i], directions.shape)),
        ]) <= tolerance
        if np.any(crossing | contact):
            raise ConfigError("throat stretch composite meridian has self-contact or self-intersection (including source/driver ring)")


def _verify_prefix_intersection(x: NDArray[np.float64], radius: NDArray[np.float64],
                                r0: float, ext_len: float, slot_len: float, ext_angle: float) -> None:
    # Intersect each sampled main segment with the exact extension/slot line.
    # The shared main-throat endpoint itself is a permitted contact.
    for start, end, slope, end_radius in (
        (0.0, ext_len, math.tan(ext_angle), r0),
        (ext_len, ext_len + slot_len, 0.0, r0),
    ):
        if end <= start:
            continue
        distance = radius - end_radius - (x - end) * slope
        d0, d1 = distance[:-1], distance[1:]
        crosses = ((d0 <= 0) & (d1 >= 0)) | ((d0 >= 0) & (d1 <= 0))
        nonparallel = d0 != d1
        candidates = crosses & nonparallel
        if not np.any(candidates):
            continue
        fraction = d0[candidates] / (d0[candidates] - d1[candidates])
        crossing_x = x[:-1][candidates] + fraction * (x[1:][candidates] - x[:-1][candidates])
        if np.any((crossing_x > start) & (crossing_x < end)):
            raise ConfigError("throat stretch main meridian intersects the extension/slot prefix")


def _rosse_main_curve(
    t: NDArray[np.float64], c: _RosseMainCoefficients
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Array form of :func:`_calculate_rosse_main` for one azimuth.

    Same expressions in the same association order, so the only arithmetic
    that can differ from the scalar path is the squaring: NumPy squares by
    multiplication (which is correctly rounded) while CPython's ``x ** 2``
    goes through the platform ``pow``, which on macOS is occasionally one ulp
    wide of it. ``tests/test_profile_vectorization.py`` pins that bound.
    """

    R, r0, k, q, m, r, b, L, c1, c2, c3, s1, s2 = c

    x = L * (math.sqrt(r**2 + m**2) - np.sqrt(r**2 + (t - m) ** 2))
    x += b * L * (math.sqrt(r**2 + (1 - m) ** 2) - math.sqrt(r**2 + m**2)) * (t**2)
    throat_r = np.sqrt(c1 + c2 * L * t + c3 * (L * t) ** 2) + r0 * (1 - k)
    mouth_r = np.maximum(0.0, R + L * (1 - np.sqrt(1 + c3 * (t - 1) ** 2)))
    t_q = t**q
    y = (1 - t_q) * throat_r + t_q * mouth_r
    return _stretch_x_curve(x, s1, s2), y


def _rosse_tmax(params: Mapping[str, Any]) -> float:
    """The R-OSSE truncation limit, read the way the grid builder reads it."""

    t_max = float(eval_param(params.get("tmax"), 0.0, 1.0))
    if not (math.isfinite(t_max) and t_max > 0.0):
        raise ValueError(f"R-OSSE tmax must be > 0, got {t_max!r}")
    return t_max


class RosseAxialLayout(NamedTuple):
    """How the composite R-OSSE parameter is shared along one meridian.

    ``t`` runs over ``[0, tmax]``. The first ``ext + slot`` millimetres of
    ``full_length`` are the straight prefix; the rest is the main curve, whose
    own parameter ``main_t`` then runs over ``[0, tmax]`` -- so ``tmax``
    truncates the main curve exactly as it does without a prefix, and the
    prefix changes neither the main curve nor the mouth (ATH GridExport:
    identical main-section radii and mouth with and without a 30 mm extension
    at tmax = 0.8).
    """

    ext_len: float
    slot_len: float
    main_length: float
    t_max: float
    full_length: float


def rosse_axial_layout(params: Mapping[str, Any], p: float = 0.0) -> RosseAxialLayout:
    if "sourceBody" in params:
        raise ValueError("standalone source bodies require their dedicated geometry API, not horn profiles")
    ext_len = max(0.0, eval_param(params.get("throatExtLength"), p, 0.0))
    slot_len = max(0.0, eval_param(params.get("slotLength"), p, 0.0))
    main_length = _rosse_length(params, p)
    t_max = _rosse_tmax(params)
    return RosseAxialLayout(
        ext_len=ext_len,
        slot_len=slot_len,
        main_length=main_length,
        t_max=t_max,
        # ``tmax * L`` rather than ``L``: sharing the parameter over the
        # untruncated length and then cutting the *composite* at tmax cut the
        # main curve short (main_t = 0.74 instead of 0.8 in the review case)
        # and moved the mouth by 7 mm. With tmax = 1 this is unchanged.
        full_length=ext_len + slot_len + (main_length if t_max == 1.0 else t_max * main_length),
    )


def calculate_rosse(
    t: float, p: float, params: Mapping[str, Any]
) -> tuple[float, float]:
    if "sourceBody" in params:
        raise ValueError("standalone source bodies require their dedicated geometry API, not horn profiles")
    validate_stretch_composition(params, "R-OSSE")
    s1, s2 = stretch_coefficients(params)  # validate even prefix stations
    r0_base = eval_param(params.get("r0"), p, 12.7)
    ext_angle = _deg(params.get("throatExtAngle"), p, 0.0)
    layout = rosse_axial_layout(params, p)
    ext_len, slot_len, main_length = layout.ext_len, layout.slot_len, layout.main_length
    if s1 != 0.0 and s2 != 0.0:
        _verify_rosse_stretch_junction(params, p, r0_base, ext_len, slot_len, ext_angle)
    # ATH convention (same as OSSE since the c198956 re-anchoring): r0 is the
    # MAIN throat radius; the extension tapers back from r0 to the driver end
    # and the main curve/mouth are unchanged by it. The old code enlarged the
    # main throat instead (r0 + ext*tan), changing L and the mouth.
    r0_throat = _throat_extension_start_radius(r0_base, ext_len, ext_angle)

    if ext_len <= 0.0 and slot_len <= 0.0:
        return _calculate_rosse_main(t, p, params)

    full_length = layout.full_length
    if full_length <= 1.0e-12:
        return 0.0, r0_base

    axial_pos = max(0.0, float(t)) * full_length
    if layout.t_max != 1.0:
        axial_pos = max(0.0, float(t)) / layout.t_max * full_length
    if axial_pos <= ext_len:
        return axial_pos, r0_throat + axial_pos * math.tan(ext_angle)
    if axial_pos <= ext_len + slot_len:
        return axial_pos, r0_base

    if main_length <= 1.0e-12:
        return ext_len + slot_len, r0_base
    main_t = (axial_pos - ext_len - slot_len) / main_length
    x, y = _calculate_rosse_main(main_t, p, params)
    return x + ext_len + slot_len, y


def calculate_rosse_curve(
    t_values: Any, p: float, params: Mapping[str, Any]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``calculate_rosse`` over a whole axial station array at one azimuth.

    Returns ``(z, radius)`` arrays.  The scalar entry point stays the public
    single-point API and the differential oracle for this one; the grid
    builders call this because resolving the parameter set once per meridian
    rather than once per grid point is the whole cost of an R-OSSE preview.
    """
    if "sourceBody" in params:
        raise ValueError("standalone source bodies require their dedicated geometry API, not horn profiles")

    validate_stretch_composition(params, "R-OSSE")
    s1, s2 = stretch_coefficients(params)
    t = np.asarray(t_values, dtype=np.float64)
    r0_base = eval_param(params.get("r0"), p, 12.7)
    ext_angle = _deg(params.get("throatExtAngle"), p, 0.0)
    layout = rosse_axial_layout(params, p)
    ext_len, slot_len, main_length = layout.ext_len, layout.slot_len, layout.main_length
    if s1 != 0.0 and s2 != 0.0:
        _verify_rosse_stretch_junction(params, p, r0_base, ext_len, slot_len, ext_angle)
    r0_throat = _throat_extension_start_radius(r0_base, ext_len, ext_angle)

    if ext_len <= 0.0 and slot_len <= 0.0:
        return _rosse_main_curve(t, _rosse_main_coefficients(p, params))

    full_length = layout.full_length
    if full_length <= 1.0e-12:
        return np.zeros_like(t), np.full_like(t, r0_base)

    axial_pos = np.maximum(0.0, t) * full_length
    if layout.t_max != 1.0:
        axial_pos = np.maximum(0.0, t) / layout.t_max * full_length
    in_ext = axial_pos <= ext_len
    in_slot = ~in_ext & (axial_pos <= ext_len + slot_len)
    in_main = ~(in_ext | in_slot)

    x = np.empty_like(t)
    y = np.empty_like(t)
    x[in_ext] = axial_pos[in_ext]
    y[in_ext] = r0_throat + axial_pos[in_ext] * math.tan(ext_angle)
    x[in_slot] = axial_pos[in_slot]
    y[in_slot] = r0_base
    if main_length <= 1.0e-12:
        x[in_main] = ext_len + slot_len
        y[in_main] = r0_base
    elif in_main.any():
        # Only the main-section stations enter the curve: a negative ``main_t``
        # from an extension station would take ``t ** q`` to NaN for the
        # fractional exponents R-OSSE actually uses.
        main_t = (axial_pos[in_main] - ext_len - slot_len) / main_length
        main_x, main_y = _rosse_main_curve(main_t, _rosse_main_coefficients(p, params))
        x[in_main] = main_x + ext_len + slot_len
        if s1 != 0.0 and s2 != 0.0:
            _verify_stretched_axial_map(x[in_main])
            order = np.argsort(main_t)
            _verify_prefix_intersection(x[in_main][order], main_y[order], r0_base,
                                        ext_len, slot_len, ext_angle)
        y[in_main] = main_y
    return x, y


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
    "coverage_angle",
    "coverage_angle_deg",
    "hold_start",
    "hold_end",
    "kappa_abs_max",
    "dkappa_ds_abs_max",
    "theta_max_deg",
    "pin_mouth_radius",
    "icw_seed",
    "icw_coeffs",
    "icw_S",
)


def _icw_key_normalise(value: Any) -> Any:
    return _lossless_key_value(_canonical_stretch_key(value), overflow_is_error=True)


def _canonical_stretch_key(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _canonical_stretch_key(v) for k, v in canonical_stretch_params(value).items()}
    return value


def _icw_cache_key(params: Mapping[str, Any]) -> str:
    """Stable, *lossless* hash over the ICW-relevant params for the curve memo.

    Array-like params (numpy arrays / numeric lists, possibly nested in
    ``icw_seed``) are normalised with full precision by :func:`_icw_key_normalise`
    before serialising, so differing coefficient arrays can never collide.
    """
    relevant = {
        k: _icw_key_normalise(params.get(k)) for k in _ICW_PARAM_KEYS if k in params
    }
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


def _icw_bool(params: Mapping[str, Any], name: str) -> bool:
    if name not in params or params[name] is None:
        return False
    value = params[name]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    try:
        return bool(eval_param(value, 0.0, 0.0))
    except (TypeError, ValueError):
        return bool(value)


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
        DEFAULT_DEGREE,
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
            raise ValueError(
                "ICW direct mode (icw_coeffs) requires icw_S (arc length, mm)"
            )
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
    coverage = _icw_float(params, "coverage_angle", "coverage_angle_deg")
    if coverage is not None and coverage < 0.0:
        raise ValueError("coverage_angle must be non-negative; use 0 to disable coverage")
    coverage_on = coverage is not None and coverage > 0.0

    target_kwargs: dict[str, Any] = {
        "mode": termination,
        "r0": float(r0),
        "theta0_deg": float(theta0_deg),
    }
    if kappa0 is not None:
        target_kwargs["kappa0"] = float(kappa0)
    # NB: must not shadow ``key`` (the cache key stored at the end of this
    # function) — a shadowed loop variable silently defeated the memo once.
    for cap_name in ("kappa_abs_max", "dkappa_ds_abs_max", "theta_max_deg"):
        value = _icw_float(params, cap_name)
        if value is not None:
            target_kwargs[cap_name] = float(value)

    if termination == "flat_baffle":
        x_target = _icw_float(params, "L", "L_mm")
        r_mouth = _icw_float(params, "R", "R_mm")
        target_kwargs["x_target"] = x_target
        if coverage_on:
            target_kwargs["coverage_angle_deg"] = float(coverage)
            hold_start = _icw_float(params, "hold_start")
            hold_end = _icw_float(params, "hold_end")
            if hold_start is not None:
                target_kwargs["hold_start"] = float(hold_start)
            if hold_end is not None:
                target_kwargs["hold_end"] = float(hold_end)
            if _icw_bool(params, "pin_mouth_radius") and r_mouth is not None:
                target_kwargs["r_mouth"] = r_mouth
        else:
            target_kwargs["r_mouth"] = r_mouth
    elif termination == "rollback":
        if coverage_on:
            target_kwargs["coverage_angle_deg"] = float(coverage)
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
    solve_kwargs: dict[str, Any] = {}
    if n_coeff_val is not None:
        n_coeff_int = int(n_coeff_val)
        # Coverage mode needs a basis large enough to carry the plateau span plus the
        # endpoint/angle rows: ``coverage_knots`` requires ``n_coeff >= degree + 6``.
        # A caller's smaller non-coverage default (the WG UI ships ``n_coeff=6`` for
        # plain ICW, materialised into every payload) would otherwise make EVERY
        # coverage build infeasible. So under coverage we drop a sub-floor value and
        # let ``solve_icw`` apply its coverage-aware default instead of forwarding a
        # basis that literally cannot represent the requested plateau. Non-coverage
        # solves, and explicit coverage bases at/above the floor, are honoured verbatim.
        if not (coverage_on and n_coeff_int < DEFAULT_DEGREE + 6):
            solve_kwargs["n_coeff"] = n_coeff_int
    curve, report = solve_icw(targets, **solve_kwargs)
    if not report.feasible:
        raise ValueError(
            "ICW target set is infeasible: "
            + "; ".join(report.violations)
            + (
                f" (hint: {report.suggested_relaxation})"
                if report.suggested_relaxation
                else ""
            )
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


def lookup_profile_array(params: Mapping[str, Any]) -> NDArray[np.float64]:
    """The validated ``lookupProfile`` of a LOOKUP formula, as an ``(n, 2)`` array."""

    raw = params.get("lookupProfile", params.get("lookup_profile"))
    if raw is None:
        raise ValueError("LOOKUP formula requires a lookupProfile of [z, r] pairs")
    lookup = np.asarray(raw, dtype=np.float64)
    if lookup.ndim != 2 or lookup.shape[1] != 2 or lookup.shape[0] < 2:
        raise ValueError("lookupProfile must be an array of at least two [z, r] pairs")
    if not np.all(np.isfinite(lookup)):
        raise ValueError("lookupProfile must contain only finite values")
    if np.any(np.diff(lookup[:, 0]) <= 0.0):
        raise ValueError("lookupProfile z values must be strictly increasing")
    return lookup


def profile_points(
    params: Mapping[str, Any], n_axial: int, phi: float = 0.0
) -> np.ndarray:
    formula = _normalise_formula(params.get("type", "OSSE"))
    t_max = (
        float(eval_param(params.get("tmax"), phi, 1.0)) if formula == "R-OSSE" else 1.0
    )
    t_values = np.linspace(0.0, t_max, int(n_axial))
    if formula == "ICW":
        curve = build_icw_curve(params, phi)
        return icw_meridian_points(curve, t_values)
    if formula == "FREEFORM":
        geometry = build_freeform_geometry(params)
        profile_h = params["profileH"]["points"]
        z = np.linspace(
            float(profile_h[0][0]),
            float(profile_h[-1][0]),
            int(n_axial),
            dtype=np.float64,
        )
        radius_h, _radius_v = geometry.evaluate_radii(z)
        return np.column_stack((z, radius_h))
    if formula == "LOOKUP":
        lookup = lookup_profile_array(params)
        z = np.linspace(
            float(lookup[0, 0]),
            float(lookup[-1, 0]),
            int(n_axial),
            dtype=np.float64,
        )
        return np.column_stack((z, np.interp(z, lookup[:, 0], lookup[:, 1])))
    points = np.empty((len(t_values), 2), dtype=np.float64)
    if formula == "OSSE":
        total = osse_total_length(params, phi)
        for idx, t in enumerate(t_values):
            points[idx] = calculate_osse(float(t) * total, phi, params)
    else:
        for idx, t in enumerate(t_values):
            points[idx] = calculate_rosse(float(t), phi, params)
    return points
