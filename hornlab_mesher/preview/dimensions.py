"""Settled-frame measurements of the solve/CAD control geometry."""
from __future__ import annotations

import copy
import json
import logging
import threading
from collections import OrderedDict
from typing import Any, Mapping

import numpy as np

from ..config_builder import resolve_geometry
from ..builders.enclosure import enclosure_box_bounds
from .contract import _validate_finite_metadata

DIMENSIONS_SAMPLING = {"method": "resolved-canonical-geometry", "lod_independent": True}
# Only small readouts are retained, never the resolved point grids. The full
# supplied config is the identity (including sampling, scale and morph inputs).
_DIMENSIONS_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_CACHE_LIMIT = 128


class _MeasurementWarnings(logging.Filter):
    def __init__(self):
        super().__init__()
        self.thread = threading.get_ident()

    def filter(self, record):
        # The preview already reported this warning. Do not silence unrelated
        # warnings or another thread's preview/solve work.
        return record.thread != self.thread or "outer wall" not in record.getMessage()


def canonical_dimensions(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the full object using exactly the solve/CAD geometry path."""
    full = copy.deepcopy(dict(config))
    mesh = full.get("mesh", {})
    if not isinstance(mesh, Mapping):
        raise TypeError("dimension measurement requires mesh to be a mapping")
    full["mesh"] = dict(mesh, quadrants="1234", vertical_offset_mm=0.0)
    full["mesh"].pop("verticalOffset", None)
    logger = logging.getLogger("hornlab_mesher.profile_sampling")
    warning_filter = _MeasurementWarnings()
    logger.addFilter(warning_filter)
    try:
        geometry = resolve_geometry(full).geometry
    finally:
        logger.removeFilter(warning_filter)
    inner = geometry.inner_points
    points = [inner.reshape(-1, 3)]
    if geometry.outer_points is not None:
        points.append(geometry.outer_points.reshape(-1, 3))
        rear = geometry.outer_points[:, 0].copy()
        rear[:, 2] = inner[:, 0, 2].mean() - geometry.wall_thickness_mm
        points.append(rear)
    result = {
        "mouth_opening": np.ptp(inner[:, -1, :2], axis=0).tolist(),
        "horn_overall": np.ptp(np.concatenate(points), axis=0).tolist(),
    }
    if geometry.enclosure is not None:
        bounds = enclosure_box_bounds(inner, geometry.enclosure, closed=True)
        result["enclosure_overall"] = [bounds["bx1"] - bounds["bx0"],
            bounds["by1"] - bounds["by0"], bounds["z_front"] - bounds["z_back"]]
    return result


def dimension_metadata(config: Mapping[str, Any], lod: str) -> dict[str, Any]:
    """Coarse frames only look up; settled frames resolve each design once.

    Content serialization keeps mutable caller dictionaries safe. Options and
    visibility are deliberately excluded; no preview buffers enter this path.
    Errors are cached too, so repeated requests for an invalid design are cheap.
    """
    try:
        key = json.dumps(dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
        cached = _DIMENSIONS_CACHE.get(key)
        if cached is not None:
            return copy.deepcopy(cached)
        if lod == "coarse":
            return {"dimensions_mm": None, "dimensions_status": "pending"}
        dimensions = canonical_dimensions(config)
        if dimensions is None:
            raise ValueError("canonical measurement returned no dimensions")
        _validate_finite_metadata(dimensions, "dimensions_mm")
        result = {"dimensions_mm": dimensions, "dimensions_status": "current"}
    except Exception as exc:
        result = {"dimensions_mm": None, "dimensions_status": "unavailable",
                  "dimensions_error": str(exc) or type(exc).__name__}
        # Unsupported config identities cannot be cached, but never break a draft.
        if "key" not in locals():
            return result
    _DIMENSIONS_CACHE[key] = copy.deepcopy(result)
    if len(_DIMENSIONS_CACHE) > _CACHE_LIMIT:
        _DIMENSIONS_CACHE.popitem(last=False)
    return result
