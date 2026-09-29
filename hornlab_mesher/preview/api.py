"""Public ``hornlab.preview/1`` geometry API.

Surface orientation is part of the render contract. ``horn.inner`` and
``source_cap`` normals point into the acoustic air domain. ``horn.outer``,
``mouth_rim``, ``wall.rear_cap``, and every ``enclosure.*`` role point toward
the solid exterior (the rim/front roles are front-facing). Every triangle is
counter-clockwise from its shipped normal side: ``cross(b-a, c-a)`` has a
strictly positive dot product with the triangle's average vertex normal.

These rules are checked directly for every emitted triangle with usable area.
Near-degenerate sampling slivers abstain from winding decisions. Signed volume
is deliberately not used because most preview roles are open shells.

``analytic-parametric`` means finite differences of samples evaluated on the
true analytic/canonical surface in its real axial and azimuthal parameter
coordinates.  It never means a derivative of, or normal averaged from, the
emitted triangle mesh.  This definition intentionally includes finite
differences: the method identifies the surface being differentiated, not a
symbolic differentiation implementation.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..config_builder import build_geometry_params
from ..profile_sampling import _outer_offset_shell
from ..profiles import eval_param
from ..viewport import build_viewport_geometry_from_config
from .contract import (
    PreviewGeometryV1,
    PreviewOptionsV1,
    PreviewSurfaceV1,
    _API_VERSION,
    _METADATA_VERSION,
    _validate_finite_metadata,
)
from .enclosure import (
    _adaptive_plan_intervals,
    _enclosure_surfaces,
    _faceted_edge,
    _plan_fidelity,
    _plan_ring,
)
from .fidelity import (
    adaptive_grid_indices,
    analytic_grid_curvature,
    analytic_grid_normals,
)
from .horn import (
    _GUIDING_CURVE_PROBE_AZIMUTHS,
    _GUIDING_CURVE_PROBE_STEP_DEG,
    _configuration_has_corners,
    _corner_phi_indices,
    _guiding_curve_warnings,
    _outer_shell_surfaces,
    _replace_grid_with_corner_refinement,
    _semantic_t_stations,
    _silhouette_segments,
)
from .primitives import (
    _MAX_ARC_INTERVALS,
    _even_indices,
    _flat_cap,
    _grid_surface_from_selection,
    _intervals_for_arc,
    _mouth_exit_direction,
    _smooth_mouth_rim,
    _surface_grid,
)
from .source_cap import _source_cap


_MAX_ANGULAR_SAMPLES = 4096
_MAX_CANONICAL_VERTICES = 1_000_000
# A caller-supplied normal step below this, with no vertex bound of its own, gets
# ``_TIGHT_NORMAL_DEFAULT_VERTEX_CAP``. The LOD presets (8, 3 and 2 degrees)
# never reach it, and neither does a caller that names ``max_vertices``.
_TIGHT_NORMAL_STEP_DEG = 1.0
_TIGHT_NORMAL_DEFAULT_VERTEX_CAP = 200_000

_LOD_PRESETS = {
    "coarse": {
        "chord": 0.15,
        "normal": 8.0,
        "silhouette": 64,
        "axial": 12,
        "roundover": 6,
        "cap": 8,
        "master_axial": 48,
    },
    "fine": {
        "chord": 0.05,
        "normal": 3.0,
        "silhouette": 128,
        "axial": 48,
        "roundover": 12,
        "cap": 16,
        "master_axial": 96,
    },
    "inspection": {
        "chord": 0.025,
        "normal": 2.0,
        "silhouette": 256,
        "axial": 96,
        "roundover": 12,
        "cap": 24,
        "master_axial": 192,
    },
}


def _lod_config(config: Mapping[str, Any], angular: int, axial: int) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    mesh = result.get("mesh")
    if not isinstance(mesh, Mapping):
        mesh = {}
    else:
        mesh = dict(mesh)
    mesh["angular_segments"] = int(angular)
    mesh["length_segments"] = int(axial)
    # Canonical names win over any imported/camel-case aliases in config_builder.
    mesh.pop("angularSegments", None)
    mesh.pop("lengthSegments", None)
    result["mesh"] = mesh
    return result


def _adaptive_lod_config(
    config: Mapping[str, Any],
    angular: int,
    axial: int,
    *,
    power: float,
    formula: str,
) -> dict[str, Any]:
    """Seed the candidate lattice with a nested throat-biased axial map.

    ``formula`` is the resolved formula from ``build_geometry_params``, never
    re-read from the raw config: the resolver accepts it at the top level or
    under ``profile`` (``formula`` or ``type``) and normalises its spelling.
    """

    result = _lod_config(config, angular, axial)
    mesh = dict(result["mesh"])
    sampling = str(
        mesh.get("sampling_mode", mesh.get("samplingMode", "uniform"))
    ).strip().lower()
    if formula != "ICW" and sampling in {"", "uniform", "linear", "canonical", "default"} and not any(
        key in mesh
        for key in ("z_map_points", "zMapPoints", "zmapPoints", "ZMapPoints")
    ):
        mesh["sampling_mode"] = "zmap"
        mesh["z_map_kind"] = "samples"
        parameter = np.linspace(0.0, 1.0, int(axial) + 1, dtype=np.float64)
        # The same analytic map at dyadically related counts makes the default
        # coarse stations exact members of fine/inspection candidate lattices.
        mesh["z_map_points"] = (
            0.5 - 0.5 * np.cos(math.pi * parameter)
        ).tolist()
    result["mesh"] = mesh
    return result


def _fidelity_record(
    achieved: Mapping[str, Any] | None,
    *,
    chord_target: float,
    normal_target: float,
    silhouette_target: int,
    cap_limited: bool = False,
) -> dict[str, Any]:
    measurement_complete = bool((achieved or {}).get("measurement_complete", True))
    raw_chord = (achieved or {}).get("max_chord_error_mm", np.finfo(np.float64).eps)
    chord = None if not measurement_complete or raw_chord is None else float(raw_chord)
    normal = float((achieved or {}).get("max_normal_step_deg", 0.0))
    unmeasured = int((achieved or {}).get("unmeasured_intervals", 0))
    limited = bool(
        (achieved or {}).get("vertex_cap_limited", False)
        or cap_limited
        or not measurement_complete
    )
    return {
        # Stage-1 aliases remain for consumers already reading them.
        "max_chord_error_mm": chord,
        "max_normal_step_deg": normal,
        "reference_density_multiplier": int(
            (achieved or {}).get("reference_density_multiplier", 4)
        ),
        "max_chord_error_mm_requested": chord_target,
        "max_normal_step_deg_requested": normal_target,
        "min_silhouette_segments_requested": silhouette_target,
        "max_chord_error_mm_achieved": chord,
        "max_normal_step_deg_achieved": normal,
        "vertex_cap_limited": limited,
        "measurement_complete": measurement_complete,
        "unmeasured_intervals": unmeasured,
        "silhouette_segments_achieved": None,
    }


@dataclass(frozen=True)
class _MasterLevel:
    """One canonical master sampling and the render grid chosen from it.

    ``inner``/``normals`` are ``(t, phi, xyz)``; ``t``/``phi`` are their real
    parameter coordinates. ``deferred_wall`` is the wall thickness the caller
    must rebuild itself (the corner-refinement and folded-offset paths), carried
    into any denser level sampled after this one.
    """

    output: dict[str, Any]
    grid: dict[str, Any]
    closed_phi: bool
    inner: NDArray[np.float64]
    t: NDArray[np.float64]
    phi: NDArray[np.float64]
    normals: NDArray[np.float64]
    semantic_inserted: list[str]
    semantic_unavailable: list[str]
    corner_rows: list[int]
    t_indices: NDArray[np.int64]
    phi_indices: NDArray[np.int64]
    achieved: dict[str, Any]
    sampling_ms: float
    deferred_wall: float


def _sample_master_level(
    config: Mapping[str, Any],
    *,
    formula: str,
    angular: int,
    axial: int,
    axial_power: float,
    axial_seed: int,
    silhouette_target: int,
    chord_target: float,
    normal_target: float,
    vertex_cap: int | None,
    has_corners: bool,
    corner_intervals: int,
    wall_mm: float,
    deferred_wall: float,
) -> _MasterLevel:
    """Sample the canonical master at ``axial`` rows and choose the render grid."""

    sampling_config = _adaptive_lod_config(
        config, angular, axial, power=axial_power, formula=formula
    )
    if deferred_wall > 0.0:
        mesh = dict(sampling_config["mesh"])
        mesh["wall_thickness_mm"] = 0.0
        for alias in ("wallThickness", "wall_thickness", "WallThickness"):
            mesh.pop(alias, None)
        sampling_config["mesh"] = mesh
    # The preview never spells the grid's flat vertex lists: it reads the
    # arrays they would be built from.
    sampling_start = time.perf_counter()
    output = build_viewport_geometry_from_config(
        sampling_config, point_lists=False, defer_osse_offset_repair=True
    )
    sampling_ms = (time.perf_counter() - sampling_start) * 1000.0
    if formula == "OSSE" and output["grid"].get("outer_offset_fold"):
        # The canonical master is used to select the acoustic render grid;
        # computing a dense envelope there would be discarded immediately.
        # Healthy normal offsets keep their existing master-grid path.
        deferred_wall = wall_mm
        output["grid"]["outer_grid"] = None
    if has_corners:
        _replace_grid_with_corner_refinement(output, corner_intervals)
    if deferred_wall > 0.0:
        output["params"]["wallThickness"] = deferred_wall

    grid_data = output["grid"]
    n_phi = int(grid_data["grid_n_phi"])
    closed_phi = bool(grid_data.get("full_circle", True))
    inner_master = _surface_grid(grid_data["inner_grid"])
    master_t = np.asarray(grid_data.get("slice_map"), dtype=np.float64)
    master_phi = (
        np.asarray(grid_data["phi_grid"], dtype=np.float64).T
        if grid_data.get("phi_grid") is not None
        else np.broadcast_to(
            np.asarray(grid_data["angle_list"], dtype=np.float64),
            inner_master.shape[:2],
        )
    )
    inner_normals = analytic_grid_normals(
        inner_master,
        closed_phi=closed_phi,
        t_coordinates=master_t,
        phi_coordinates=master_phi,
    )
    semantic_t, semantic_inserted, semantic_unavailable = _semantic_t_stations(
        output, master_t
    )
    initial_t = sorted(
        set(_even_indices(len(master_t), axial_seed + 1, closed=False)).union(
            semantic_t
        )
    )
    initial_phi = _even_indices(n_phi, silhouette_target, closed=closed_phi)
    corner_rows: list[int] = []
    if has_corners:
        corner_rows = _corner_phi_indices(inner_normals)
        initial_phi = sorted(set(initial_phi).union(corner_rows))
    t_indices, phi_indices, achieved = adaptive_grid_indices(
        inner_master,
        inner_normals,
        initial_t,
        initial_phi,
        max_chord_error_mm=chord_target,
        max_normal_step_deg=normal_target,
        max_vertices=vertex_cap,
        closed_phi=closed_phi,
        t_coordinates=master_t,
        phi_coordinates=master_phi,
    )
    return _MasterLevel(
        output=output,
        grid=grid_data,
        closed_phi=closed_phi,
        inner=inner_master,
        t=master_t,
        phi=master_phi,
        normals=inner_normals,
        semantic_inserted=semantic_inserted,
        semantic_unavailable=semantic_unavailable,
        corner_rows=corner_rows,
        t_indices=t_indices,
        phi_indices=phi_indices,
        achieved=achieved,
        sampling_ms=sampling_ms,
        deferred_wall=deferred_wall,
    )


def build_preview_geometry(
    config: Mapping[str, Any], options: PreviewOptionsV1 = PreviewOptionsV1()
) -> PreviewGeometryV1:
    """Build complete error-bounded render geometry from a mesher config."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not isinstance(options, PreviewOptionsV1):
        raise TypeError("options must be PreviewOptionsV1")
    lod = str(options.lod).strip().lower()
    if lod not in _LOD_PRESETS:
        raise ValueError("lod must be 'coarse', 'fine', or 'inspection'")
    preset = _LOD_PRESETS[lod]
    chord_target = float(
        preset["chord"]
        if options.max_chord_error_mm is None
        else options.max_chord_error_mm
    )
    normal_target = float(
        preset["normal"]
        if options.max_normal_step_deg is None
        else options.max_normal_step_deg
    )
    try:
        silhouette_target = int(
            preset["silhouette"]
            if options.min_silhouette_segments is None
            else options.min_silhouette_segments
        )
        vertex_cap = None if options.max_vertices is None else int(options.max_vertices)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("silhouette and vertex limits must be finite integers") from exc
    if not math.isfinite(chord_target) or chord_target <= 0.0:
        raise ValueError("max_chord_error_mm must be finite and > 0")
    if not math.isfinite(normal_target) or not 0.0 < normal_target <= 180.0:
        raise ValueError("max_normal_step_deg must be finite and in (0, 180]")
    if silhouette_target < 3:
        raise ValueError("min_silhouette_segments must be >= 3")
    if vertex_cap is not None and vertex_cap < 8:
        raise ValueError(
            "max_vertices must be >= 8 (the minimum robust 2x4 horn topology)"
        )

    warnings: list[str] = []
    preflight_limited = False
    if (
        vertex_cap is None
        and options.max_normal_step_deg is not None
        and math.isfinite(normal_target)
        and normal_target < _TIGHT_NORMAL_STEP_DEG
    ):
        # A sub-degree normal step on a horn asks for hundreds of thousands of
        # vertices (0.35 degrees: about 950k, over 100 s) and, unbounded, can
        # hold a worker for minutes. The bound is reported like any explicit
        # one, in ``requested_fidelity`` and per-surface ``vertex_cap_limited``.
        vertex_cap = _TIGHT_NORMAL_DEFAULT_VERTEX_CAP
        warnings.append(
            f"max_normal_step_deg={normal_target:g} is tighter than "
            f"{_TIGHT_NORMAL_STEP_DEG:g} degrees and no max_vertices was given: "
            f"bounded at {vertex_cap} vertices per surface; pass max_vertices to change it"
        )
    start = time.perf_counter()

    # Resolve the formula, mode and parameters once, through the same resolver
    # the solved build uses. Re-reading ``config["formula"]`` here missed the
    # documented ``profile.formula``/``type`` aliases and alternative
    # spellings, which injected a z-map into ICW and skipped FREEFORM's corner
    # sampling.
    parsed_params, _parsed_formula, _parsed_mode = build_geometry_params(config)
    formula_name = str(_parsed_formula)
    has_corners = _configuration_has_corners(parsed_params, formula_name)
    corner_intervals = _intervals_for_arc(
        200.0,
        90.0,
        chord_target,
        normal_target,
        int(preset["roundover"]),
    )
    if corner_intervals >= _MAX_ARC_INTERVALS:
        preflight_limited = True
        warnings.append(
            f"acoustic corner reference clamped to {_MAX_ARC_INTERVALS} intervals"
        )
    # The canonical corner sampler refines its three stable arc rows in whole
    # multiples. Ordinary/circular grids instead use a 2x candidate lattice.
    if has_corners:
        corner_intervals = 3 * int(math.ceil(corner_intervals / 3.0))
        # Corner grids already add a dense, stable union of analytic arc rows;
        # multiplying the flat-side seed would defeat that adaptive allocation.
        angular_master = max(3, silhouette_target)
    else:
        angular_master = max(
            4 * silhouette_target, 4 * int(math.ceil(360.0 / normal_target))
        )
    if angular_master > _MAX_ANGULAR_SAMPLES:
        angular_master = _MAX_ANGULAR_SAMPLES
        preflight_limited = True
        warnings.append(
            f"canonical azimuth reference clamped to {_MAX_ANGULAR_SAMPLES} samples"
        )
    default_chord = float(preset["chord"])
    chord_ratio = default_chord / chord_target
    axial_scale = math.sqrt(chord_ratio) if math.isfinite(chord_ratio) else math.inf
    scaled_axial = (
        512
        if not math.isfinite(axial_scale)
        or axial_scale >= 512 / int(preset["master_axial"])
        else int(math.ceil(int(preset["master_axial"]) * max(1.0, axial_scale)))
    )
    axial_master = min(512, max(int(preset["master_axial"]), scaled_axial))
    axial_master = min(512, max(axial_master, 4 * int(preset["axial"])))
    # R-OSSE/FREEFORM double the candidate lattice.
    axial_ceiling = (
        min(512, axial_master * 2)
        if formula_name in {"R-OSSE", "FREEFORM"}
        else axial_master
    )
    # The doubling earns its cost on some of those configurations and not
    # others, so the thin master is attempted only where it was measured to
    # settle, and the escalation below is the guard for the rest:
    #
    #   * A smooth R-OSSE fine preview selects the same 97x256 grid from a
    #     193-row master as from a 385-row one -- 2.4e-4 mm apart on a 0.05 mm
    #     budget, and byte-identical on the enclosed variant -- for half the
    #     canonical sampling. That is the case worth taking.
    #   * The coarse preset sits on its own ``master_axial`` floor and its
    #     refinement consumes very nearly every row the master offers, so the
    #     thin pass always starves. Building it and then rebuilding cost a
    #     measured 36 -> 59 ms, and coarse is the lane a drag runs in.
    #   * Corner configurations starve the same way at fine (a rectangle-morph
    #     R-OSSE went 70 -> 105 ms paying for both masters), because their
    #     angular master is the silhouette target rather than a multiple of it
    #     and the axial rows carry correspondingly more of the error budget.
    #   * FREEFORM's rear cap moved 0.86 mm on the thin master, seventeen times
    #     the chord budget, so it keeps the density it was given.
    attempt_thin_master = (
        axial_ceiling > axial_master
        and formula_name == "R-OSSE"
        and not has_corners
        and 4 * int(preset["axial"]) > int(preset["master_axial"])
    )
    if not attempt_thin_master:
        axial_master = axial_ceiling
    angular_product_cap = max(3, _MAX_CANONICAL_VERTICES // (axial_ceiling + 1))
    if angular_master > angular_product_cap:
        angular_master = angular_product_cap
        preflight_limited = True
        warnings.append(
            f"canonical reference clamped to {_MAX_CANONICAL_VERTICES} points"
        )

    # Sampling time only, summed over every master an escalation builds, so it
    # stays the same quantity it was before escalation existed.
    canonical_ms = 0.0
    axial_power = {"coarse": 1.75, "fine": 2.0, "inspection": 2.5}[lod]
    warnings.extend(_guiding_curve_warnings(parsed_params, _parsed_formula))
    # The wall thickness as configured. ``deferred_wall`` below is only set on
    # the corner-refinement path, where the outer shell is rebuilt here instead
    # of by the sampler, so it is not a reliable thickness on its own.
    wall_mm = float(eval_param(parsed_params.get("wallThickness"), 0.0, 0.0))
    deferred_wall = 0.0
    if has_corners and str(_parsed_mode) == "freestanding":
        deferred_wall = float(eval_param(parsed_params.get("wallThickness"), 0.0, 0.0))

    def sample(axial: int, deferred_wall: float) -> _MasterLevel:
        return _sample_master_level(
            config,
            formula=formula_name,
            angular=angular_master,
            axial=axial,
            axial_power=axial_power,
            axial_seed=int(preset["axial"]),
            silhouette_target=silhouette_target,
            chord_target=chord_target,
            normal_target=normal_target,
            vertex_cap=vertex_cap,
            has_corners=has_corners,
            corner_intervals=corner_intervals,
            wall_mm=wall_mm,
            deferred_wall=deferred_wall,
        )

    level = sample(axial_master, deferred_wall)
    canonical_ms += level.sampling_ms
    # Refinement asked for detail the master could not supply, so the doubled
    # master this family used to build unconditionally is worth its cost here.
    # One step only: the ceiling is the density that shipped before, so an
    # escalated build is the old build and a settled one is strictly cheaper.
    escalated = False
    if axial_ceiling > axial_master and level.achieved.get("candidate_starved"):
        axial_master = axial_ceiling
        level = sample(axial_master, level.deferred_wall)
        canonical_ms += level.sampling_ms
        escalated = True
    deferred_wall = level.deferred_wall
    output = level.output
    grid_data = level.grid
    closed_phi = level.closed_phi
    inner_master = level.inner
    master_t = level.t
    master_phi = level.phi
    inner_normals = level.normals
    semantic_inserted = level.semantic_inserted
    semantic_unavailable = level.semantic_unavailable
    corner_rows = level.corner_rows
    t_indices = level.t_indices
    phi_indices = level.phi_indices
    horn_achieved = level.achieved
    inner_canonical = grid_data["inner_grid"]

    if (
        output.get("enclosure") is not None
        and options.include_enclosure
        and not closed_phi
    ):
        # The enclosure surfaces are built as closed rings around the mouth. A
        # reduced (quadrant or half) domain has an open mouth arc, and joining
        # its two ends across the removed quadrants wound the baffle both ways
        # and refused the whole preview. The horn is still drawn.
        warnings.append(
            "enclosure not drawn: the preview draws the enclosure only for the "
            "full model (quadrants 1234); the reduced-domain horn is shown alone"
        )
        output["enclosure"] = None

    # Curvature depends only on the master that survived escalation, so it is
    # evaluated once here rather than inside a level that may be discarded.
    inner_curvature_mean = inner_curvature_principal = None
    if options.include_curvature and options.include_inner:
        inner_curvature_mean, inner_curvature_principal = analytic_grid_curvature(
            inner_master,
            closed_phi=closed_phi,
            t_coordinates=master_t,
            phi_coordinates=master_phi,
        )

    assembly_start = time.perf_counter()
    surfaces: list[PreviewSurfaceV1] = []
    fidelity: dict[str, dict[str, Any]] = {}
    if options.include_inner:
        surfaces.append(
            _grid_surface_from_selection(
                "horn.inner",
                inner_master,
                inner_normals,
                t_indices,
                phi_indices,
                closed_phi=closed_phi,
                normal_sign=-1.0,
                curvature_mean=inner_curvature_mean,
                curvature_principal=inner_curvature_principal,
            )
        )
        fidelity["horn.inner"] = _fidelity_record(
            horn_achieved,
            chord_target=chord_target,
            normal_target=normal_target,
            silhouette_target=silhouette_target,
        )

    selected_inner = inner_canonical[np.ix_(phi_indices, t_indices)]
    outer_canonical = None
    selected_outer = None
    if grid_data.get("outer_grid") is not None:
        outer_canonical = grid_data["outer_grid"]
        selected_outer = outer_canonical[np.ix_(phi_indices, t_indices)]
        if options.include_outer:
            outer_master = _surface_grid(outer_canonical)
            for outer_surface in _outer_shell_surfaces(
                outer_master,
                t_indices,
                phi_indices,
                closed_phi=closed_phi,
                t_coordinates=master_t,
                phi_coordinates=master_phi,
                include_curvature=options.include_curvature,
            ):
                surfaces.append(outer_surface)
                fidelity[outer_surface.role] = _fidelity_record(
                    horn_achieved,
                    chord_target=chord_target,
                    normal_target=normal_target,
                    silhouette_target=silhouette_target,
                )
            surfaces.append(
                _smooth_mouth_rim(
                    selected_inner[:, -1, :],
                    selected_outer[:, -1, :],
                    closed_phi=closed_phi,
                    exit_direction=_mouth_exit_direction(selected_inner),
                    include_curvature=options.include_curvature,
                )
            )
            fidelity["mouth_rim"] = _fidelity_record(
                horn_achieved,
                chord_target=chord_target,
                normal_target=normal_target,
                silhouette_target=silhouette_target,
            )
    elif deferred_wall > 0.0:
        offset_inner = selected_inner.copy()
        vertical_offset = float(grid_data.get("vertical_offset_mm", 0.0) or 0.0)
        offset_inner[:, :, 1] -= vertical_offset
        selected_outer = _outer_offset_shell(
            offset_inner,
            deferred_wall,
            full_circle=closed_phi,
            repair_osse=str(_parsed_formula) == "OSSE",
            t_coordinates=master_t[t_indices],
            phi_coordinates=master_phi[np.ix_(t_indices, phi_indices)],
        )
        selected_outer[:, :, 1] += vertical_offset
        outer_canonical = selected_outer
        if options.include_outer:
            selected_outer_master = _surface_grid(selected_outer)
            # The deferred-wall shell is already the selected grid, so its rows
            # are its own stations.
            for outer_surface in _outer_shell_surfaces(
                selected_outer_master,
                np.arange(selected_outer_master.shape[0], dtype=np.int64),
                np.arange(selected_outer_master.shape[1], dtype=np.int64),
                closed_phi=closed_phi,
                t_coordinates=master_t[t_indices],
                phi_coordinates=master_phi[np.ix_(t_indices, phi_indices)],
                include_curvature=options.include_curvature,
            ):
                surfaces.append(outer_surface)
                fidelity[outer_surface.role] = _fidelity_record(
                    horn_achieved,
                    chord_target=chord_target,
                    normal_target=normal_target,
                    silhouette_target=silhouette_target,
                )
            surfaces.append(
                _smooth_mouth_rim(
                    selected_inner[:, -1, :],
                    selected_outer[:, -1, :],
                    closed_phi=closed_phi,
                    exit_direction=_mouth_exit_direction(selected_inner),
                    include_curvature=options.include_curvature,
                )
            )
            fidelity["mouth_rim"] = _fidelity_record(
                horn_achieved,
                chord_target=chord_target,
                normal_target=normal_target,
                silhouette_target=silhouette_target,
            )

    if options.include_source_cap:
        cap_intervals = int(preset["cap"])
        cap_limited = False
        if vertex_cap is not None:
            allowed_radial = max(1, (vertex_cap - 1) // len(phi_indices))
            if cap_intervals > allowed_radial:
                cap_intervals = allowed_radial
                cap_limited = True
        while True:
            cap_surface, cap_fidelity, source_details = _source_cap(
                selected_inner,
                output["params"],
                output["formula"],
                cap_intervals,
                closed_phi=closed_phi,
                include_curvature=options.include_curvature,
            )
            passes = cap_fidelity is None or (
                cap_fidelity["max_chord_error_mm"] <= chord_target
                and cap_fidelity["max_normal_step_deg"] <= normal_target
            )
            if passes:
                break
            proposed = cap_intervals + 1
            if vertex_cap is not None and 1 + proposed * len(phi_indices) > vertex_cap:
                cap_limited = True
                break
            cap_intervals = proposed
        surfaces.append(cap_surface)
        fidelity["source_cap"] = _fidelity_record(
            cap_fidelity,
            chord_target=chord_target,
            normal_target=normal_target,
            silhouette_target=silhouette_target,
            cap_limited=cap_limited,
        )
    else:
        source_details = {}
        cap_intervals = 0

    if output.get("enclosure") is not None:
        # The preview draws every implemented plan, but only the rounded
        # rectangle has a watertight enclosure builder -- ``build_enclosure_box``
        # raises NotImplementedError for the ellipse and superellipse plans in
        # the closed domain, and its open-domain route accepts plan_type=1 only.
        # Say so here rather than let the shape look finished until build time.
        # The toggle above is a display filter, not a config change, so this
        # warning does not depend on it.
        preview_plan_type = int(output["enclosure"].get("plan_type", 1))
        if preview_plan_type in (2, 3):
            warnings.append(
                f"enclosure plan_type={preview_plan_type} is previewed but not "
                "buildable: only the rounded-rectangle plan (plan_type=1) has a "
                "watertight closed-enclosure builder, and the open (reduced) "
                "domain supports plan_type=1 only"
            )

    if output.get("enclosure") is not None and options.include_enclosure:
        enclosure_payload = dict(output["enclosure"])
        enclosure_config = config.get("enclosure")
        if isinstance(enclosure_config, Mapping):
            plan_n = enclosure_config.get(
                "plan_n", enclosure_config.get("planN", enclosure_config.get("encPlanN"))
            )
            if plan_n is not None:
                enclosure_payload["plan_n"] = float(plan_n)
        roundover_intervals = _intervals_for_arc(
            float(enclosure_payload.get("edge_depth", 0.0)),
            90.0,
            chord_target,
            normal_target,
            int(preset["roundover"]),
        )
        plan_corner_intervals, plan_preflight_limited = _adaptive_plan_intervals(
            enclosure_payload,
            chord_target,
            normal_target,
            1,
        )
        if plan_preflight_limited:
            preflight_limited = True
            warnings.append(
                f"enclosure plan reference clamped to {_MAX_ARC_INTERVALS} intervals per quarter"
            )
        enclosure_cap_limited = plan_preflight_limited
        if vertex_cap is not None:
            def enclosure_vertex_estimates(
                radial_intervals: int, plan_intervals: int
            ) -> dict[str, int]:
                plan_vertices = (
                    4 * plan_intervals + 8
                    if int(enclosure_payload["plan_type"]) == 1
                    else 4 * plan_intervals
                )
                estimates = {
                    "enclosure.front": 2 * len(phi_indices),
                    "enclosure.side": (
                        4 * plan_vertices
                        if int(enclosure_payload["edge_type"]) == 2
                        else 2 * plan_vertices
                    ),
                }
                if float(enclosure_payload.get("edge_depth", 0.0)) > 0.0:
                    estimates["enclosure.roundover"] = len(phi_indices) + (
                        2 * radial_intervals + 1
                    ) * plan_vertices
                if options.include_rear_cap:
                    estimates["enclosure.rear"] = plan_vertices + 1
                return estimates

            minimum_estimates = enclosure_vertex_estimates(1, 1)
            topology_minimum = max(minimum_estimates.values())
            if vertex_cap < topology_minimum:
                raise ValueError(
                    f"max_vertices={vertex_cap} is below the enclosure topological "
                    f"minimum {topology_minimum}"
                )
            while max(
                enclosure_vertex_estimates(
                    roundover_intervals, plan_corner_intervals
                ).values()
            ) > vertex_cap and (
                roundover_intervals > 1 or plan_corner_intervals > 1
            ):
                enclosure_cap_limited = True
                if roundover_intervals >= plan_corner_intervals and roundover_intervals > 1:
                    roundover_intervals -= 1
                elif plan_corner_intervals > 1:
                    plan_corner_intervals -= 1
        enclosure_surfaces, enclosure_fidelity = _enclosure_surfaces(
            enclosure_payload,
            selected_inner[:, -1, :],
            roundover_intervals,
            plan_corner_intervals,
            chord_target=chord_target,
            normal_target=normal_target,
            include_rear=options.include_rear_cap,
            include_curvature=options.include_curvature,
        )
        surfaces.extend(enclosure_surfaces)
        plan_measurement = _plan_fidelity(
            enclosure_payload, plan_corner_intervals
        )
        plan_chord = plan_measurement["max_chord_error_mm"]
        plan_normal = plan_measurement["max_normal_step_deg"]
        plan_vertices = len(
            _plan_ring(
                enclosure_payload,
                0.0,
                1.0,
                corner_intervals=plan_corner_intervals,
            )
        )
        # The roundover arc model only describes a fillet. A chamfer sweeps one
        # ruled interval between two polygons and discretises nothing, so
        # charging it a fillet's chord error and normal step reported an error
        # the emitted band does not carry.
        if _faceted_edge(enclosure_payload):
            round_chord = 0.0
            round_normal = 0.0
        else:
            round_chord = float(enclosure_payload.get("edge_depth", 0.0)) * (
                1.0 - math.cos(math.pi / (4.0 * roundover_intervals))
            )
            round_normal = 90.0 / roundover_intervals
        for surface in enclosure_surfaces:
            measured = enclosure_fidelity.get(surface.role, {})
            if surface.role == "enclosure.roundover":
                measured = dict(measured)
                measured["max_chord_error_mm"] = max(
                    float(measured.get("max_chord_error_mm", 0.0)),
                    plan_chord,
                    round_chord,
                )
                measured["max_normal_step_deg"] = max(
                    float(measured.get("max_normal_step_deg", 0.0)),
                    plan_normal,
                    round_normal,
                )
            elif surface.role == "enclosure.side":
                measured = dict(plan_measurement)
            fidelity[surface.role] = _fidelity_record(
                measured,
                chord_target=chord_target,
                normal_target=normal_target,
                silhouette_target=silhouette_target,
                cap_limited=enclosure_cap_limited,
            )
    elif selected_outer is not None and options.include_rear_cap:
        # The rear plate sits on the plane the MESH puts it on, which is not a
        # property of the outer ring at all:
        #     rear_z = mean(inner throat z) - wall
        # See ``point_grid_freestanding.py`` (rear_z) and ``_rear_rim_points``,
        # which keeps x/y and moves only z.
        #
        # Under the old throat clamp the outer throat ring happened to sit on
        # that same plane, so capping straight off it matched the mesh BY
        # ACCIDENT. Now that row 0 lies on the offset surface it does not, and
        # deriving the plane from the ring put the previewed rear face ~4.4 mm
        # forward of the real one and silently dropped the whole rear return.
        rear_z = float(np.mean(selected_inner[:, 0, 2]) - wall_mm)
        rear_ring = np.array(selected_outer[:, 0, :], dtype=np.float64, copy=True)
        rear_ring[:, 2] = rear_z

        # The band between the outer throat ring and the rear rim IS the rear
        # return. The mesh builds it by prepending this ring to the outer shell;
        # the preview has already emitted its shell, so ship the band on its own.
        if options.include_outer:
            # (t, phi, xyz), rear rim first, exactly as the mesh orders it.
            return_master = np.stack(
                (rear_ring, selected_outer[:, 0, :]), axis=0
            )
            # A straight axial extrusion of the throat ring, so its meridian
            # curvature is zero; the hoop term is carried by the ring itself.
            return_curvature = (
                np.zeros(return_master.shape[:2], dtype=np.float64)
                if options.include_curvature
                else None
            )
            surfaces.append(
                _grid_surface_from_selection(
                    "wall.rear_return",
                    return_master,
                    analytic_grid_normals(return_master, closed_phi=closed_phi),
                    np.asarray((0, 1), dtype=np.int64),
                    np.arange(return_master.shape[1], dtype=np.int64),
                    closed_phi=closed_phi,
                    curvature_mean=return_curvature,
                    curvature_principal=return_curvature,
                )
            )

        surfaces.append(
            _flat_cap(
                "wall.rear_cap",
                rear_ring,
                (0.0, 0.0, -1.0),
                closed_phi=closed_phi,
                include_curvature=options.include_curvature,
            )
        )
        fidelity["wall.rear_cap"] = _fidelity_record(
            horn_achieved,
            chord_target=chord_target,
            normal_target=normal_target,
            silhouette_target=silhouette_target,
        )

    # Every rendered role receives target-vs-achieved data, including exact
    # planar faces whose interior error is zero and whose boundary inherits the
    # adjacent adaptive ring.
    for surface in surfaces:
        fidelity.setdefault(
            surface.role,
            _fidelity_record(
                horn_achieved,
                chord_target=chord_target,
                normal_target=normal_target,
                silhouette_target=silhouette_target,
            ),
        )

    folded_outer = sum(
        int(surface.metadata.get("foldedTriangles", 0))
        for surface in surfaces
        if surface.role == "horn.outer"
    )
    if folded_outer:
        warnings.append(
            f"outer wall folds over itself in {folded_outer} triangles: the wall "
            "thickness exceeds the local radius of curvature (typically at a "
            "rolled-back mouth, a throat extension or slot, or a morph corner). "
            "The preview shows the folded wall; the acoustic (inner) surface is "
            "unaffected. Reduce the wall thickness or open the local curvature."
        )

    selected_phi_coordinates = master_phi[np.ix_(t_indices, phi_indices)]
    horn_silhouette = _silhouette_segments(
        selected_phi_coordinates, closed_phi=closed_phi
    )
    enclosure_plan_roles = {
        "enclosure.roundover",
        "enclosure.side",
        "enclosure.rear",
    }
    for surface in surfaces:
        achieved_silhouette = (
            plan_vertices
            if surface.role in enclosure_plan_roles
            and output.get("enclosure") is not None
            and options.include_enclosure
            else horn_silhouette
        )
        fidelity[surface.role]["silhouette_segments_achieved"] = int(
            achieved_silhouette
        )
        if achieved_silhouette < silhouette_target:
            fidelity[surface.role]["vertex_cap_limited"] = True
        if preflight_limited:
            fidelity[surface.role]["preflight_limited"] = True
        if vertex_cap is not None and len(surface.positions) > vertex_cap:
            raise ValueError(
                f"max_vertices={vertex_cap} cannot represent {surface.role}; "
                f"minimum emitted topology has {len(surface.positions)} vertices"
            )

    assembly_ms = (time.perf_counter() - assembly_start) * 1000.0
    total_ms = (time.perf_counter() - start) * 1000.0
    metadata: dict[str, Any] = {
        "api_version": _API_VERSION,
        "metadata_version": _METADATA_VERSION,
        "units": "mm",
        "coordinate_frame": "mesher-xyz",
        "formula": output["formula"],
        "mode": output["mode"],
        "lod": lod,
        "actual_segment_counts": {
            "horn_phi": len(phi_indices),
            "horn_axial": len(t_indices) - 1,
            "enclosure_roundover_quarter": (
                roundover_intervals
                if output.get("enclosure") is not None
                and float(output["enclosure"].get("edge_depth", 0.0)) > 0.0
                and options.include_enclosure
                and int(output["enclosure"].get("edge_type", 1)) == 1
                else 1
                if output.get("enclosure") is not None
                and float(output["enclosure"].get("edge_depth", 0.0)) > 0.0
                and options.include_enclosure
                else 0
            ),
            "source_cap_radial": (
                cap_intervals
                if options.include_source_cap
                and float(source_details.get("source_cap_height_mm", 0.0)) > 0.0
                else (1 if options.include_source_cap else 0)
            ),
        },
        "timings_ms": {
            "canonical_sampling": canonical_ms,
            "surface_assembly_and_fidelity": assembly_ms,
            "total": total_ms,
        },
        # The true-surface reference the emitted grid was chosen from, and
        # whether a thinner one was tried and found too coarse first. Canonical
        # sampling is linear in this product, so it is the number that explains
        # what a build cost.
        "canonical_reference": {
            "axial_rows": int(inner_master.shape[0]),
            "azimuth_rows": int(inner_master.shape[1]),
            "escalated": escalated,
        },
        "fidelity": fidelity,
        "warnings": warnings,
        # How finely the guiding-curve saturation screen actually looked. A
        # fixed step can always be out-resolved by a spiky enough expression,
        # so publish it rather than let the absence of a warning read as a
        # proof that the guiding curve is met everywhere.
        "guiding_curve_probe": {
            "step_deg": _GUIDING_CURVE_PROBE_STEP_DEG,
            "azimuths": len(_GUIDING_CURVE_PROBE_AZIMUTHS),
            "best_effort": True,
        },
        "requested_fidelity": {
            "max_chord_error_mm": chord_target,
            "max_normal_step_deg": normal_target,
            "min_silhouette_segments": silhouette_target,
            "max_vertices": vertex_cap,
        },
        "angular_sampling": {
            "strategy": "stable-union-corner-grid" if has_corners else "adaptive-periodic",
            "corner_arc_rows": len(set(phi_indices).intersection(corner_rows)),
            "flat_side_rows": len(phi_indices)
            - len(set(phi_indices).intersection(corner_rows)),
            # The faceted chamfer band rules per plan column and never
            # zippers; only the curved fillet path still stitches unequal
            # rings.
            "uses_zipper_indices": bool(
                output.get("enclosure") is not None
                and not _faceted_edge(output["enclosure"])
            ),
        },
        "vertex_accounting": {
            surface.role: {
                "vertices": len(surface.positions),
                "max_vertices": vertex_cap,
                "vertex_cap_limited": fidelity[surface.role]["vertex_cap_limited"],
            }
            for surface in surfaces
        },
        "semantic_stations": {
            "inserted_first": semantic_inserted
            + ["corner tangencies/cardinals", "enclosure transitions"],
            "unavailable_additively": semantic_unavailable,
        },
        "nesting": {
            "horn_phi": "coarse is a subset of fine for circular grids; corner grids use the stable union row identities",
            "horn_axial": "12 -> 48 -> 96 base stations are nested; semantic stations are shared by value",
            "exceptions": [
                "custom tolerances/caps may choose different refinement paths",
                "canonical ATH z-map modes are not guaranteed to be nested",
            ],
        },
        "normal_convention": (
            "role-oriented analytic/exact normals; every triangle is counter-clockwise "
            "when viewed from its normal side"
        ),
        "normal_method_notes": {
            "analytic-parametric": (
                "finite differences of true analytic/canonical surface samples in "
                "their real axial and azimuthal parameter coordinates; never mesh-derived"
            ),
            "exact-planar": "the exact constant normal of an emitted planar face",
        },
        "surface_metadata": {
            surface.role: dict(surface.metadata) for surface in surfaces
        },
        **source_details,
    }
    _validate_finite_metadata(metadata)
    return PreviewGeometryV1(surfaces=surfaces, metadata=metadata)


__all__ = [
    "PreviewGeometryV1",
    "PreviewOptionsV1",
    "PreviewSurfaceV1",
    "build_preview_geometry",
]
