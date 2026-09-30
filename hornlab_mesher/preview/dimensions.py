"""Full-object canonical control-geometry extents, independent of render LOD."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from typing import Any, Mapping

import numpy as np

from ..config_builder import resolve_geometry
from ..viewport import build_enclosure_viewport_grid


def canonical_dimensions(config: Mapping[str, Any]) -> dict[str, Any]:
    """Measure the full canonical object, never the visible/reduced triangles.

    Like CAD/control geometry, the modelled wall is the canonical normal offset
    (including the OSSE envelope repair), not an added width/depth allowance.
    The design's resolved control geometry keeps sampling and offset normals
    independent of preview LOD and matches the solve/CAD input.
    """
    full_config = copy.deepcopy(dict(config))
    mesh = dict(full_config.get("mesh") or {})
    mesh.update(quadrants="1234", vertical_offset_mm=0.0)
    full_config["mesh"] = mesh
    try:
        key = json.dumps(full_config, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        # In-process callers may carry non-JSON lookup-profile objects.
        return _measure_dimensions(full_config)
    # Coarse/fine requests for one revision share this expensive canonical
    # resolution. Every design/config field participates in the cache key.
    return copy.deepcopy(_cached_dimensions(key))


@lru_cache(maxsize=16)
def _cached_dimensions(key: str) -> dict[str, Any]:
    return _measure_dimensions(json.loads(key))


def _measure_dimensions(full_config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = resolve_geometry(full_config)
    geometry = resolved.geometry
    inner = geometry.inner_points
    points = [inner.reshape(-1, 3)]
    outer = geometry.outer_points
    if outer is not None:
        points.append(outer.reshape(-1, 3))
        rear = outer[:, 0, :].copy()
        rear[:, 2] = np.mean(inner[:, 0, 2]) - geometry.wall_thickness_mm
        points.append(rear)
    dimensions = {
        "mouth_opening": np.ptp(inner[:, -1, :2], axis=0).tolist(),
        "horn_overall": np.ptp(np.concatenate(points), axis=0).tolist(),
    }
    enclosure = geometry.enclosure
    if enclosure is not None:
        bounds = build_enclosure_viewport_grid(inner, enclosure)["bounds"]
        dimensions["enclosure_overall"] = [
            bounds["bx1"] - bounds["bx0"],
            bounds["by1"] - bounds["by0"],
            bounds["z_front"] - bounds["z_back"],
        ]
    return dimensions
