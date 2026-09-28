"""Horn preview surfaces and the facts the render grid is chosen from.

The outer shell, the semantic axial stations and corner rows the adaptive
grid must keep, the corner-refinement master, and the guiding-curve
saturation screen.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..profile_formulas import osse_coverage_saturation, osse_coverage_saturation_probe
from ..profile_sampling import (
    ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY,
    FREEFORM_CONTINUOUS_COLLAPSE_KEY,
    build_point_grid_arrays,
)
from .contract import PreviewSurfaceV1
from .fidelity import analytic_grid_curvature, analytic_grid_normals
from .primitives import _grid_surface_from_selection


# Dihedral angle across the emitted throat band above which it and the offset
# shell are two faces rather than one curved surface; see _outer_shell_surfaces.
_THROAT_JOG_CREASE_DEG = 15.0


def _band_dihedral_deg(
    rings: list[NDArray[np.float64]], *, closed_phi: bool
) -> float:
    """Median dihedral angle between the two bands three consecutive rings span.

    Measured on the rings that are actually emitted, not on the canonical
    reference: a tangent break the shipped mesh does not resolve cannot shade
    wrong, and one it does resolve must not be smoothed over. This is the same
    criterion the renderer uses to decide a feature edge.
    """

    def band_normals(
        near: NDArray[np.float64], far: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        along = (
            np.roll(near, -1, axis=0) - near
            if closed_phi
            else np.diff(near, axis=0, append=near[-1:])
        )
        normals = np.cross(along, far - near)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / np.where(lengths > 1.0e-12, lengths, 1.0)

    lower = band_normals(rings[0], rings[1])
    upper = band_normals(rings[1], rings[2])
    return float(
        np.median(
            np.degrees(
                np.arccos(np.clip(np.abs(np.sum(lower * upper, axis=1)), -1.0, 1.0))
            )
        )
    )


def _outer_shell_surfaces(
    master: NDArray[np.float64],
    t_indices: NDArray[np.int64],
    phi_indices: NDArray[np.int64],
    *,
    closed_phi: bool,
    t_coordinates: NDArray[np.float64] | None,
    phi_coordinates: NDArray[np.float64] | None,
    include_curvature: bool,
) -> list[PreviewSurfaceV1]:
    """Build the outer wall, splitting the shading at the throat jog.

    ``_outer_offset_shell`` does not put the shell's first station on the
    offset surface: it squares the throat off into a flat rear face, radius
    ``r0 + wall`` at ``z0 - wall``. The band that ring forms with the next
    station therefore meets the offset surface at a real crease, and a station
    on a crease needs two normals -- one per face. Shipping one, taken from a
    central difference straddling the crease, put that ring up to 83 degrees
    off both faces it borders and drew a shading ring around the throat.

    Overriding the ring's normal cannot fix it; whichever face it is made to
    agree with, the other still disagrees. So the jog band becomes its own role
    and carries its own copy of the crease ring, which is where this module
    puts every other hard boundary (``mouth_rim``, ``wall.rear_cap``): the
    normals inside each role stay smooth and the crease falls between them.
    """

    def normals_for(rows: slice) -> NDArray[np.float64]:
        return analytic_grid_normals(
            master[rows],
            closed_phi=closed_phi,
            t_coordinates=None if t_coordinates is None else t_coordinates[rows],
            phi_coordinates=(
                None if phi_coordinates is None else phi_coordinates[rows]
            ),
        )

    def curvature_for(rows: slice) -> tuple[
        NDArray[np.float64] | None, NDArray[np.float64] | None
    ]:
        if not include_curvature:
            return None, None
        return analytic_grid_curvature(
            master[rows],
            closed_phi=closed_phi,
            t_coordinates=None if t_coordinates is None else t_coordinates[rows],
            phi_coordinates=(
                None if phi_coordinates is None else phi_coordinates[rows]
            ),
        )

    def unsplit() -> list[PreviewSurfaceV1]:
        mean, principal = curvature_for(slice(None))
        return [
            _grid_surface_from_selection(
                "horn.outer",
                master,
                normals_for(slice(None)),
                t_indices,
                phi_indices,
                closed_phi=closed_phi,
                curvature_mean=mean,
                curvature_principal=principal,
            )
        ]

    crease = 0.0
    if len(master) >= 4 and len(t_indices) >= 3:
        rings = [master[np.ix_([int(t_indices[k])], phi_indices)][0] for k in range(3)]
        crease = _band_dihedral_deg(rings, closed_phi=closed_phi)
    if crease < _THROAT_JOG_CREASE_DEG:
        return unsplit()

    band_rows = np.asarray(t_indices[:2], dtype=np.int64)
    band_master = master[band_rows]
    band_normals = analytic_grid_normals(
        band_master,
        closed_phi=closed_phi,
        t_coordinates=None if t_coordinates is None else t_coordinates[band_rows],
        phi_coordinates=(
            None if phi_coordinates is None else phi_coordinates[band_rows]
        ),
    )
    band_mean = band_principal = None
    if include_curvature:
        # Two stations carry no second difference in t, so the band's meridian
        # curvature is reported as the straight ruling it is emitted as.
        band_mean = np.zeros(band_master.shape[:2], dtype=np.float64)
        band_principal = np.zeros(band_master.shape[:2], dtype=np.float64)
    shell_mean, shell_principal = curvature_for(slice(1, None))
    shell_normals = np.array(normals_for(slice(None)), dtype=np.float64)
    shell_normals[1:] = normals_for(slice(1, None))
    padded_mean = padded_principal = None
    if shell_mean is not None and shell_principal is not None:
        padded_mean = np.zeros(master.shape[:2], dtype=np.float64)
        padded_principal = np.zeros(master.shape[:2], dtype=np.float64)
        padded_mean[1:] = shell_mean
        padded_principal[1:] = shell_principal
    try:
        return [
            _grid_surface_from_selection(
                "horn.outer",
                master,
                shell_normals,
                np.asarray(t_indices[1:], dtype=np.int64),
                phi_indices,
                closed_phi=closed_phi,
                curvature_mean=padded_mean,
                curvature_principal=padded_principal,
            ),
            _grid_surface_from_selection(
                "wall.throat_band",
                band_master,
                band_normals,
                np.asarray((0, 1), dtype=np.int64),
                phi_indices,
                closed_phi=closed_phi,
                curvature_mean=band_mean,
                curvature_principal=band_principal,
            ),
        ]
    except ValueError:
        # A sharp morph corner has no defined offset direction, so the shell
        # can carry a few facets tipped just past perpendicular there. The
        # whole shell forgives them as the negligible share of its area they
        # are; two rows cut out of it do not have the area to. Splitting is a
        # shading improvement, never a reason to lose the surface, so a band
        # that cannot be wound consistently on its own gives the shading back
        # to the shell it came from.
        return unsplit()


def _nearest_indices(values: NDArray[np.float64], targets: list[float]) -> list[int]:
    return sorted(
        {
            int(np.argmin(np.abs(values - float(target))))
            for target in targets
            if math.isfinite(float(target))
        }
    )


def _radial_extrema(radius: NDArray[np.float64]) -> NDArray[np.int64]:
    """Stations where the mean radius turns, or starts or stops being constant.

    A constant-radius run (a straight throat extension, a cylindrical slot) has
    zero slope throughout, and ``slope[i-1] * slope[i] <= 0`` flagged every
    station inside it: a 30 mm extension forced 56 of 193 master rows into the
    render grid (100 axial rows instead of 58) for no fidelity gain. A run is
    collapsed to its two ends, which are the real shape transitions; strict
    sign changes are kept. Slopes within float noise of zero count as zero.
    """

    values = np.asarray(radius, dtype=np.float64)
    slope = np.diff(values)
    if len(slope) < 2:
        return np.empty(0, dtype=np.int64)
    scale = float(np.max(np.abs(values))) if len(values) else 0.0
    tolerance = 64.0 * np.finfo(np.float64).eps * max(scale, 1.0)
    sign = np.where(np.abs(slope) <= tolerance, 0.0, np.sign(slope))
    before, after = sign[:-1], sign[1:]
    turns = before * after < 0.0
    run_edge = (before == 0.0) != (after == 0.0)
    return np.flatnonzero(turns | run_edge) + 1


def _semantic_t_stations(
    output: Mapping[str, Any], t_values: NDArray[np.float64]
) -> tuple[list[int], list[str], list[str]]:
    targets = [0.0, 1.0]
    inserted = ["throat", "mouth"]
    unavailable = ["OSSE extension/slot boundaries when expression-valued"]
    params = output["params"]

    morph_start = params.get("morphFixed")
    if isinstance(morph_start, (int, float)) and 0.0 < float(morph_start) < 1.0:
        targets.append(float(morph_start))
        inserted.append("morph start")

    # The resolved parameters, not the raw config: the resolver also accepts
    # the profile under ``parameters`` and the formula in several spellings.
    if str(output["formula"]).upper() == "FREEFORM":
        profile = params
        for station in profile.get("crossSections") or ():
            if isinstance(station, Mapping) and isinstance(station.get("t"), (int, float)):
                targets.append(float(station["t"]))
        for key in ("profileH", "profileV"):
            descriptor = profile.get(key)
            if not isinstance(descriptor, Mapping):
                continue
            rows = descriptor.get("points")
            if not isinstance(rows, list) or len(rows) < 2:
                continue
            z0 = float(rows[0][0])
            length = float(rows[-1][0]) - z0
            if length > 0.0:
                targets.extend((float(row[0]) - z0) / length for row in rows)
        inserted.extend(["FREEFORM cross-section stations", "FREEFORM H/V anchors"])
        unavailable.remove("OSSE extension/slot boundaries when expression-valued")

    # Rollback extrema are available additively from the canonical candidate
    # grid even though the profile evaluator does not publish named stations.
    master_grid = output["grid"]
    points = master_grid["inner_grid"]
    radius = np.mean(np.linalg.norm(points[:, :, :2], axis=2), axis=0)
    extrema = _radial_extrema(radius)
    if len(extrema):
        targets.extend(float(t_values[index]) for index in extrema)
        inserted.append("canonical rollback/radial extrema")

    return _nearest_indices(t_values, targets), inserted, unavailable


def _corner_phi_indices(normals: NDArray[np.float64]) -> list[int]:
    """Return the union of curved-arc rows; planar wall rows remain sparse."""

    mouth = normals[-1]
    dots = np.sum(mouth * np.roll(mouth, -1, axis=0), axis=1)
    changing = np.flatnonzero(1.0 - np.clip(dots, -1.0, 1.0) > 1.0e-10)
    size = normals.shape[1]
    return sorted({int(index) for value in changing for index in (value, (value + 1) % size)})


def _silhouette_segments(
    phi: NDArray[np.float64], *, closed_phi: bool
) -> int:
    if closed_phi:
        return int(phi.shape[1])
    equivalents: list[int] = []
    for row in np.asarray(phi, dtype=np.float64):
        span = float(np.unwrap(row)[-1] - np.unwrap(row)[0])
        if span > 0.0:
            equivalents.append(int(round((len(row) - 1) * math.tau / span)))
    return min(equivalents, default=max(0, phi.shape[1] - 1))


def _configuration_has_corners(params: Mapping[str, Any], formula: str) -> bool:
    """Whether the resolved geometry carries true (morph-target 1) corners.

    Reads the parameters ``build_geometry_params`` resolved, so every accepted
    spelling of the formula, the cross sections and the morph target is seen
    exactly as the builder sees it.
    """

    target = params.get("morphTarget", 0)
    if formula == "FREEFORM":
        station_has_corners = any(
            isinstance(station, Mapping)
            and str(station.get("shape", "")).strip().lower() == "rounded_rectangle"
            for station in (params.get("crossSections") or ())
        )
        if station_has_corners:
            return True
        try:
            static_target = float(target)
        except (TypeError, ValueError):
            return False
        return math.isfinite(static_target) and int(round(static_target)) == 1
    # A finite-exponent superellipse (target 3) is smooth and has no true corners.
    return (
        isinstance(target, (int, float))
        and not isinstance(target, bool)
        and math.isfinite(float(target))
        and int(round(float(target))) == 1
    )


def _replace_grid_with_corner_refinement(
    output: dict[str, Any], corner_intervals: int
) -> None:
    params = dict(output["params"])
    params[ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY] = max(
        1, int(math.ceil(corner_intervals / 3.0))
    )
    if str(params.get("type", "")).strip().upper() == "FREEFORM":
        params[FREEFORM_CONTINUOUS_COLLAPSE_KEY] = True
    grid = build_point_grid_arrays(params)
    vertical_offset = float(grid.get("vertical_offset_mm", 0.0) or 0.0)
    if vertical_offset:
        for key in ("inner_grid", "outer_grid"):
            if grid.get(key) is None:
                continue
            grid[key][:, :, 1] += vertical_offset
    output["params"] = params
    output["grid"] = grid


# Azimuths screened for an unreachable guiding curve. The guiding curve, the
# coverage angle and the termination may all be per-azimuth expressions, so a
# single phi=0 probe would miss a mouth that only goes off-target off-axis.
#
# The step was 15 degrees while each azimuth cost a full coverage inversion,
# and that was demonstrably too coarse: gcurveWidth="1000 - 900*sin(12*p)^2"
# is reachable at every multiple of 15 and saturated at 7.5, so the preview
# said nothing at all. Screening with the bracket probe instead of the full
# inversion (osse_coverage_saturation_probe: 2 radius evaluations, not 26)
# buys the resolution back. Measured, healthy OSSE geometry with a reachable
# type-1 guiding curve (36 us per azimuth by inversion, 5 us by probe):
#
#     15 deg / full inversion  (24 azimuths)   0.85 ms   <- was
#      1 deg / full inversion  (360 azimuths) 12.88 ms
#      1 deg / bracket probe   (360 azimuths)  1.79 ms   <- is
#
# The whole preview build for that config is 96 ms coarse / 554 ms fine, so
# 15x the angular resolution costs +0.94 ms, about 1% of a coarse frame. A
# naive tightening without the probe would have cost 12x that.
#
# STILL BEST-EFFORT. One degree resolves anything up to about a 180th-order
# azimuthal term, which is far past any guiding curve a person writes by hand,
# but a sufficiently spiky expression can still hide between probes and no
# fixed step can rule that out. The absence of a warning is therefore not a
# guarantee that the guiding curve is met; the step is published in the
# preview metadata as ``guiding_curve_probe.step_deg`` so a caller can say how
# much the silence is worth.
_GUIDING_CURVE_PROBE_STEP_DEG = 1.0
_GUIDING_CURVE_PROBE_AZIMUTHS = tuple(
    math.radians(index * _GUIDING_CURVE_PROBE_STEP_DEG)
    for index in range(int(round(360.0 / _GUIDING_CURVE_PROBE_STEP_DEG)))
)


def _guiding_curve_warnings(
    params: Mapping[str, Any], formula: Any
) -> list[str]:
    """Warn when the OSSE coverage solver cannot reach the guiding curve.

    The solver clamps to its bracket instead of failing, which reads to the
    user as "the parameters stopped doing anything" — the mouth is no longer on
    the guiding curve and no further edit to the coverage angle can put it
    back. Reported once with the worst-offending azimuth rather than once per
    probe, so a fully unreachable curve does not emit 360 near-identical lines.

    Screened with the bracket probe rather than the full inversion. The probe
    returns the same saturated result the inversion would, so ranking the
    azimuths on it is exact; only the reported azimuth pays for a full solve,
    and even that one returns from the probe branch without bisecting.
    """

    if str(formula).strip().upper() != "OSSE":
        return []
    worst_phi: float | None = None
    worst_error = -1.0
    saturated = 0
    for phi in _GUIDING_CURVE_PROBE_AZIMUTHS:
        try:
            solved = osse_coverage_saturation_probe(params, phi)
        except (ValueError, ZeroDivisionError, OverflowError):
            # A malformed guiding curve is the config validator's error to
            # raise; a preview warning must not mask it with its own failure.
            return []
        if solved is None or solved.saturated is None:
            continue
        saturated += 1
        error = abs(solved.achieved_radius - solved.target_radius)
        if not math.isfinite(error):
            error = math.inf
        # A rotationally symmetric guiding curve misses every azimuth by the
        # same amount up to rounding, so only a materially worse azimuth may
        # displace the incumbent. Otherwise the reported phi is whichever
        # probe happened to accumulate more floating-point error.
        if error > worst_error * (1.0 + 1.0e-9) + 1.0e-9:
            worst_error = error
            worst_phi = phi
    if worst_phi is None:
        return []
    location = (
        "every probed azimuth"
        if saturated == len(_GUIDING_CURVE_PROBE_AZIMUTHS)
        else None
    )
    reason = osse_coverage_saturation(params, worst_phi, location=location)
    return [reason] if reason is not None else []
