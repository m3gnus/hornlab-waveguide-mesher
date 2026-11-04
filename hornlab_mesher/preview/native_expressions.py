"""Preview the same native span grids consumed by OCC and CAD export."""
from __future__ import annotations

import math

import numpy as np

from ..native_expressions import ExpressionSurface
from .contract import PreviewGeometryV1
from .fidelity import analytic_grid_normals
from .primitives import _combine_surfaces, _flat_cap, _smooth_grid_surface


def _surfaces(surface, reference, options, targets):
    from .api import _fidelity_record
    chord, normal, silhouette = targets
    surfaces = []
    fidelity = {}
    for (role, grid), (_, master) in zip(surface.branches, reference.branches):
        if ((role == "horn.inner" and not options.include_inner)
            or (role in {"horn.outer", "wall.rear_return"} and not options.include_outer)
            or (role == "mouth_rim" and not (options.include_inner and options.include_outer))):
            continue
        points = grid.transpose(1, 0, 2).copy()
        points[:, :, 1] += surface.offset
        master = master.transpose(1, 0, 2).copy()
        master[:, :, 1] += surface.offset
        # The shared grid owner follows dP/dphi x dP/dt: the meridian's
        # right normal, including where a lip turns back towards the throat.
        hint = analytic_grid_normals(points, closed_phi=True)
        if role == "horn.inner":
            hint = -hint
        if role == "mouth_rim":
            hint[:] = [0, 0, -1]
        preview, measured = _smooth_grid_surface(role, points, master, closed_phi=True,
            orientation_hint=hint, include_curvature=options.include_curvature and points.shape[0] > 2)
        surfaces.append(preview)
        record = _fidelity_record(measured, chord_target=chord, normal_target=normal, silhouette_target=silhouette)
        record["method"] = "native-expression-sampled-grid"
        record["silhouette_segments_achieved"] = len(grid)
        previous = fidelity.get(role)
        if previous is not None:
            for key in ("max_chord_error_mm", "max_chord_error_mm_achieved", "max_normal_step_deg", "max_normal_step_deg_achieved"):
                record[key] = max(record[key], previous[key])
        fidelity[role] = record
    inner = surface.grid("horn.inner")
    if options.include_source_cap:
        ring = inner[:, 0].copy()
        ring[:, 1] += surface.offset
        surfaces.append(_flat_cap("source_cap", ring, (0, 0, 1), closed_phi=True,
                                  include_curvature=options.include_curvature))
    rear = next((grid[:, 0] for role, grid in surface.branches if role == "wall.rear_return"), None)
    if rear is not None and options.include_rear_cap:
        ring = rear.copy()
        ring[:, 1] += surface.offset
        surfaces.append(_flat_cap("wall.rear_cap", ring, (0, 0, -1), closed_phi=True,
                                  include_curvature=options.include_curvature))
    surfaces = [_combine_surfaces(role, [part for part in surfaces if part.role == role])
                for role in dict.fromkeys(part.role for part in surfaces)]
    return surfaces, fidelity


def build(params, options):
    from .api import _LOD_PRESETS
    if options.lod not in _LOD_PRESETS:
        raise ValueError("lod must be 'coarse', 'fine', or 'inspection'")
    preset = _LOD_PRESETS[options.lod]
    chord = preset["chord"] if options.max_chord_error_mm is None else options.max_chord_error_mm
    normal = preset["normal"] if options.max_normal_step_deg is None else options.max_normal_step_deg
    silhouette = preset["silhouette"] if options.min_silhouette_segments is None else options.min_silhouette_segments
    limit = 1_000_000 if options.max_vertices is None else options.max_vertices
    if type(chord) not in (float, int) or not math.isfinite(chord) or chord <= 0:
        raise ValueError("max_chord_error_mm must be finite and positive")
    if type(normal) not in (float, int) or not math.isfinite(normal) or not 0 < normal <= 180:
        raise ValueError("max_normal_step_deg must be in (0,180]")
    if type(silhouette) is not int or not 0 <= silhouette <= 1024:
        raise ValueError("Native expression min_silhouette_segments must be from 0 to 1024. Reduce the silhouette target.")
    if type(limit) is not int or limit < 8:
        raise ValueError("max_vertices must be an integer of at least 8")
    nphi = 4*math.ceil(max(preset["silhouette"], silhouette)/4)
    intervals = max(32, preset["axial"])
    config = params["nativeExpressionConfig"]
    # Preview density belongs to LOD, just as in the ordinary preview owner.
    # Mesh angular/length controls only affect the solve/export surface.
    surface = ExpressionSurface.from_config(config, sampling=(nphi, intervals))
    dimensions = ExpressionSurface.from_config(config, sampling=(64, 32)).dimensions()
    for attempt in range(3):
        reference = ExpressionSurface.from_config(config, sampling=(2*nphi, 2*intervals))
        surfaces, fidelity = _surfaces(surface, reference, options, (chord, normal, silhouette))
        if sum(len(part.positions) for part in surfaces) > limit:
            raise ValueError("Native OSSE preview exceeds the vertex budget. Increase the preview vertex budget.")
        satisfied = all(record["max_chord_error_mm"] <= chord and record["max_normal_step_deg"] <= normal
                        for record in fidelity.values())
        if satisfied or attempt == 2 or 4*nphi*(4*intervals+1) > 150_000:
            break
        surface = reference
        nphi *= 2
        intervals *= 2
    rear = surface.grid("horn.outer")
    # The denser reference checks sampled error; no continuous angular bound
    # is claimed for arbitrary expressions (the ordinary OSSE contract too).
    warnings = [] if satisfied else ["Native OSSE preview did not reach the requested sampled fidelity. Increase the preview sample budget or simplify the body expressions."]
    points = np.concatenate([grid.reshape(-1, 3) for _, grid in surface.branches])
    points[:, 1] += surface.offset
    return PreviewGeometryV1(surfaces, {
        "api_version": "hornlab.preview/1", "metadata_version": "hornlab.preview/1.4",
        "formula": params["type"], "mode": "freestanding" if rear is not None else "bare",
        "units": "mm", "coordinate_frame": "mesher-xyz", "lod": options.lod,
        "boundsMm": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "dimensions_mm": dimensions, "dimensions_status": "current",
        "dimensions_sampling": {"method": "native-expression-sampled-grid", "lod_independent": True},
        "fidelity": fidelity, "warnings": warnings, "nativeExpressionBody": True,
        "source_shape": 0, "source_cap_height_mm": 0,
        "surface_metadata": {part.role: dict(part.metadata) for part in surfaces},
    })
