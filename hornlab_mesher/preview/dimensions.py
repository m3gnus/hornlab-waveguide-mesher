"""Cheap full-object measurements of the preview's analytic reference surface."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

from ..config_builder import (build_geometry_params, _validate_mode_contract,
                              _enclosure_from_config, _section)
from ..profile_sampling import build_point_grid_arrays
from ..viewport import build_enclosure_viewport_grid
from .fidelity import analytic_grid_normals


DIMENSIONS_SAMPLING = {
    "method": "preview-reference-normal-offset",
    "lod_independent": True,
    "tolerance_mm": 1.0,
    "best_effort": True,
    "wall": "unrepaired normal offset; no solve/CAD exterior-envelope repair",
    "reduced_domain_reference": {"angular_segments_min": 128, "angular_segments_max": 512,
                                 "length_segments": 48},
}


def _full_reference(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bounded fallback for standalone measurements and reduced previews.

    Evaluate the full object, rather than reflecting a possibly asymmetric
    expression from the visible quadrant. Never fit acoustic mesh density or
    build/validate/repair a second wall. ICW's profile kernel is already cached
    by the preview that calls us.
    """
    params, _, mode = build_geometry_params(config)
    _validate_mode_contract(params, mode)
    sampling = copy.deepcopy(params)
    sampling.update(quadrants="1234", verticalOffset=0.0, wallThickness=0.0,
                    angularSegments=min(512, max(128, int(params.get("angularSegments", 128)))),
                    lengthSegments=48)
    grid = build_point_grid_arrays(sampling)
    inner = grid["inner_grid"].transpose(1, 0, 2)
    phi = grid.get("phi_grid")
    phi = (np.asarray(phi).T if phi is not None else
           np.broadcast_to(np.asarray(grid["angle_list"]), inner.shape[:2]))
    normals = analytic_grid_normals(inner, closed_phi=True,
        t_coordinates=np.asarray(grid["slice_map"]), phi_coordinates=phi)
    # Use the shared enclosure bounds only on this uncommon fallback. The
    # normal full-domain hot path reuses bounds the preview already computed.
    enclosure = _enclosure_from_config(config, _section(config, "mesh"),
                                      _section(config, "enclosure"))
    bounds = None
    if enclosure is not None:
        bounds = build_enclosure_viewport_grid(grid["inner_grid"], enclosure)["bounds"]
    return {"inner": inner, "normals": normals, "params": params,
            "mode": mode, "bounds": bounds, "closed_phi": True}


def canonical_dimensions(config: Mapping[str, Any], *,
                         reference: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Measure master samples, before render selection or visibility filtering.

    Extrema use the existing analytic normals and an unrepaired normal-offset
    shell. The approximation (including finite sampling) is disclosed in
    dimensions_sampling; corpus tests bound its difference from solve/CAD.
    This function never mutates the reference or resolves solve/CAD geometry.
    """
    mesh = config.get("mesh")
    if mesh is not None and not isinstance(mesh, Mapping):
        raise TypeError("dimension measurement requires mesh to be a mapping")
    if (reference is None or not reference["closed_phi"] or
        int(reference["params"].get("angularSegments", 0)) > reference["inner"].shape[1]):
        reference = _full_reference(config)
    params = reference["params"]
    _validate_mode_contract(params, str(reference["mode"]))
    inner = np.asarray(reference["inner"])
    lower = inner.min(axis=(0, 1))
    upper = inner.max(axis=(0, 1))
    wall = float(params.get("wallThickness") or 0.0)
    # Enclosed horns have no separate shell, regardless of a wall setting.
    if wall > 0.0 and float(params.get("encDepth") or 0.0) <= 0.0:
        outer = inner + wall * reference["normals"]
        lower = np.minimum(lower, outer.min(axis=(0, 1)))
        upper = np.maximum(upper, outer.max(axis=(0, 1)))
        rear_z = float(inner[0, :, 2].mean() - wall)
        lower[2] = min(lower[2], rear_z)
        upper[2] = max(upper[2], rear_z)
    dimensions = {
        "mouth_opening": np.ptp(inner[-1, :, :2], axis=0).tolist(),
        "horn_overall": (upper - lower).tolist(),
    }
    bounds = reference["bounds"]
    if bounds is not None:
        dimensions["enclosure_overall"] = [
            bounds["bx1"] - bounds["bx0"],
            bounds["by1"] - bounds["by0"],
            bounds["z_front"] - bounds["z_back"],
        ]
    return dimensions
