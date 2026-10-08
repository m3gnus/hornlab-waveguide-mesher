"""Settled-frame sizes of the canonical resolved design geometry."""
from __future__ import annotations

import copy
import json
import logging
import threading
from collections import OrderedDict
from concurrent.futures import Future
from typing import Any, Mapping

import numpy as np

from ..config_builder import build_geometry_params, resolve_geometry
from ..builders.enclosure import enclosure_box_bounds
from ..profiles import eval_param
from .contract import _validate_finite_metadata

DIMENSIONS_SAMPLING = {"method": "resolved-design-geometry", "lod_independent": True}
# Only small readouts are retained, never the resolved point grids. The full
# supplied config is the identity (including sampling, scale and morph inputs).
_DIMENSIONS_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_CACHE_LIMIT = 128
_CACHE_LOCK = threading.RLock()
_IN_FLIGHT: dict[str, Future] = {}


class _MeasurementWarnings(logging.Filter):
    def __init__(self):
        super().__init__()
        self.thread = threading.get_ident()

    def filter(self, record):
        # The preview already reported this warning. Do not silence unrelated
        # warnings or another thread's preview/solve work.
        return record.thread != self.thread or not any(
            text in record.getMessage() for text in ("outer wall", "enc_depth", "enc_edge")
        )


def canonical_dimensions(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bound the full design's resolved controls, before CAD surface fitting."""
    from ..native_boundary import validate_native_boundary
    from ..axial_scale import configuration
    native = validate_native_boundary(config)
    # Check each native feature's original physical domain before zeroing placement.
    native_params = build_geometry_params(config) if native is not None else None
    axial = native_params if native == "OSSE-AXIAL" else configuration(config, build_geometry_params)
    if axial is not None:
        from ..axial_scale import AxialModel
        model = AxialModel.from_params(axial[0])
        diameter = 2*float(model.body(model.length)[0][1])
        return {"mouth_opening":[diameter]*2,"horn_overall":[diameter,diameter,model.length]}
    full = copy.deepcopy(dict(config))
    mesh = full.get("mesh", {})
    if not isinstance(mesh, Mapping):
        raise TypeError("dimension measurement requires mesh to be a mapping")
    full["mesh"] = dict(mesh, quadrants="1234", vertical_offset_mm=0.0)
    full["mesh"].pop("verticalOffset", None)
    if native is not None:
        full.pop("vertical_offset_mm", None)
        full.pop("verticalOffset", None)
    loggers = [logging.getLogger(name) for name in (
        "hornlab_mesher.profile_sampling", "hornlab_mesher.builders.enclosure")]
    warning_filter = _MeasurementWarnings()
    for logger in loggers:
        logger.addFilter(warning_filter)
    try:
        geometry = resolve_geometry(full).geometry
        return _geometry_dimensions(geometry)
    finally:
        for logger in loggers:
            logger.removeFilter(warning_filter)


def _geometry_dimensions(geometry) -> dict[str, Any]:
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


def _requested_dimensions(config: Mapping[str, Any]) -> dict[str, Any] | None:
    # Parse the same aliases and expressions as the resolver; these are the
    # requested targets, before implicit sizing and the no-shrink floor.
    if not config.get("morph") and not any(key in config for key in ("morphTarget", "morph_target")):
        return None
    params, _, _ = build_geometry_params(config)
    if not eval_param(params.get("morphTarget"), 0.0, 0.0):
        return None
    scale = eval_param(params.get("scale"), 0.0, 1.0)
    return {"mouth_opening": [
        scale * eval_param(params.get("morphWidth"), 0.0, 0.0),
        scale * eval_param(params.get("morphHeight"), np.pi / 2, 0.0),
    ]}


def _unavailable(exc: Exception) -> dict[str, Any]:
    return {"dimensions_mm": None, "dimensions_status": "unavailable",
            "dimensions_error": str(exc) or type(exc).__name__}


def dimension_metadata(config: Mapping[str, Any], lod: str) -> dict[str, Any]:
    """Coarse frames only look up; settled frames resolve each design once.

    Resolve an owned snapshot decoded from its immutable content key. Options and
    visibility are deliberately excluded; no preview buffers enter this path.
    Errors are cached too, so repeated requests for an invalid design are cheap.
    """
    try:
        key = json.dumps(dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with _CACHE_LOCK:
            cached = _DIMENSIONS_CACHE.get(key)
            if cached is not None:
                return copy.deepcopy(cached)
            if lod == "coarse":
                return {"dimensions_mm": None, "dimensions_status": "pending"}
            future = _IN_FLIGHT.get(key)
            owner = future is None
            if owner:
                future = _IN_FLIGHT[key] = Future()
        if not owner:
            return copy.deepcopy(future.result())
    except Exception as exc:
        return _unavailable(exc)
    try:
        try:
            snapshot = json.loads(key)
            dimensions = canonical_dimensions(snapshot)
            if dimensions is None:
                raise ValueError("canonical measurement returned no dimensions")
            result = {"dimensions_mm": dimensions, "dimensions_status": "current"}
            requested = _requested_dimensions(snapshot)
            if requested is not None:
                result["dimensions_requested_mm"] = requested
            _validate_finite_metadata(result, "dimensions")
            stored = copy.deepcopy(result)
        except Exception as exc:
            result = _unavailable(exc)
            # Even failure-copy errors belong to measurement. A result can be
            # returned without caching if the cache cannot safely own it.
            try:
                stored = copy.deepcopy(result)
            except Exception:
                stored = None
        with _CACHE_LOCK:
            try:
                if stored is not None:
                    _DIMENSIONS_CACHE[key] = stored
                    if len(_DIMENSIONS_CACHE) > _CACHE_LIMIT:
                        _DIMENSIONS_CACHE.popitem(last=False)
            except Exception as exc:
                # Publication can fail after insertion (during eviction). Do
                # not leave a successful value behind for this failed outcome.
                result = _unavailable(exc)
                stored = None
                try:
                    _DIMENSIONS_CACHE.pop(key, None)
                except Exception:
                    pass
            future.set_result(stored if stored is not None else result)
        return result
    except BaseException as exc:
        future.set_exception(exc)
        raise
    finally:
        with _CACHE_LOCK:
            _IN_FLIGHT.pop(key, None)
