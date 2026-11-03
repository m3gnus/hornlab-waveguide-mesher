"""Settled-frame sizes of the canonical resolved design geometry."""
from __future__ import annotations

import copy
import json
import logging
import threading
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..config_builder import ResolvedGeometry, build_geometry_params, resolve_geometry, resolve_enclosure_bounds
from ..builders.enclosure import EnclosureBaffleError, measure_enclosure_box_bounds
from ..profiles import eval_param
from ..profile_common import _symmetry_planes_for_quadrants
from .contract import _validate_finite_metadata

DIMENSIONS_SAMPLING = {"method": "resolved-design-geometry", "lod_independent": True}
# Only small readouts are retained, never the resolved point grids. The full
# supplied config is the identity (including sampling, scale and morph inputs).
_DIMENSIONS_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_CACHE_LIMIT = 128
_CACHE_LOCK = threading.RLock()
_IN_FLIGHT: dict[str, Future] = {}


@dataclass(frozen=True)
class _ResolvedDimensionInput:
    """Per-frame physical geometry; never retained in the readout cache."""

    config: Mapping[str, Any]
    resolved: ResolvedGeometry
    enclosure_bounds: Mapping[str, float]

    def matches(self, full: Mapping[str, Any]) -> bool:
        if (self.resolved.quadrants != "1234"
                or self.resolved.geometry.vertical_offset_mm != 0.0):
            return False
        try:
            return _config_key(full) == _config_key(_measurement_frame(
                copy.deepcopy(dict(self.config))
            ))
        except Exception:
            # Transient construction data need not be JSON serializable. A
            # failed reuse check must leave independent measurement available.
            return False


def _config_key(config: Mapping[str, Any]) -> str:
    return json.dumps(dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _measurement_frame(full: dict[str, Any]) -> dict[str, Any]:
    mesh = full.get("mesh", {})
    if not isinstance(mesh, Mapping):
        raise TypeError("dimension measurement requires mesh to be a mapping")
    # Measurement removes rigid placement at both supported locations.
    full.pop("vertical_offset_mm", None)
    full.pop("verticalOffset", None)
    full["mesh"] = dict(mesh, quadrants="1234", vertical_offset_mm=0.0)
    full["mesh"].pop("verticalOffset", None)
    return full


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


def canonical_dimensions(
    config: Mapping[str, Any], resolved_input: _ResolvedDimensionInput | None = None,
) -> dict[str, Any]:
    """Bound the full design's resolved controls, before CAD surface fitting."""
    from ..native_boundary import validate_native_boundary
    from ..axial_scale import configuration
    full = copy.deepcopy(dict(config))
    native = validate_native_boundary(full)
    # Check each native feature's original physical domain before zeroing placement.
    native_params = build_geometry_params(full) if native is not None else None
    if native == "OSSE-ADAPTER-CONTROLS":
        from ..adapter_controls import ControlsMeridian
        model=ControlsMeridian.from_params(native_params[0])
        return {"source_diameter":[2*model.source_radius]*2,"join_diameter":[2*model.r0]*2,
                "mouth_opening":[2*float(model.body(1)[0][1])]*2,"horn_overall":[2*model.reach,2*model.reach,model.depth]}
    if native_params is not None and "terminatingArc" in native_params[0]:
        from ..composite_meridian import CompositeMeridian
        model=CompositeMeridian.from_params(native_params[0])
        box=np.asarray(model.bounds)
        return {"mouth_opening":[2*model.reach]*2,"horn_overall":(box[1]-box[0]).tolist()}
    if native == "SOURCE-DISK":
        from ..source_body import StandaloneSourceGeometry
        model=StandaloneSourceGeometry(**native_params[0]["sourceBody"])
        return {"source_diameter":[2*model.radius_mm]*2,"body_overall":[2*model.outer_radius_mm,2*model.outer_radius_mm,model.depth_mm]}
    if native == "OSSE-ADAPTER-AXIAL":
        from ..adapter_axial import PhysicalAdapter
        return PhysicalAdapter.from_params(native_params[0]).dimensions
    # Other native markers never own an axial-only body (Roundover x Axial
    # is measured from its resolved roundover below).
    axial = native_params if native == "OSSE-AXIAL" else None if native is not None else configuration(full, build_geometry_params)
    if axial is not None:
        from ..axial_scale import AxialModel
        model = AxialModel.from_params(axial[0])
        diameter = 2*float(model.body(model.length)[0][1])
        return {"mouth_opening":[diameter]*2,"horn_overall":[diameter,diameter,model.length]}
    full = _measurement_frame(full)
    if resolved_input is not None and resolved_input.matches(full):
        # Only domain and zero-placement aliases were normalized. Retained ATH
        # construction, sampling, and all physical inputs must still match.
        return _geometry_dimensions(
            resolved_input.resolved.geometry,
            enclosure_bounds=resolved_input.enclosure_bounds,
        )
    loggers = [logging.getLogger(name) for name in (
        "hornlab_mesher.profile_sampling", "hornlab_mesher.builders.enclosure")]
    warning_filter = _MeasurementWarnings()
    for logger in loggers:
        logger.addFilter(warning_filter)
    try:
        params, _, mode = build_geometry_params(config)
        enclosure_bounds = resolve_enclosure_bounds(config) if mode == "enclosure" else None
        if enclosure_bounds is not None:
            # Measure the mirror-completed physical box from the accepted build's
            # bounds, rather than validating another full-domain mouth sampling.
            cuts = _symmetry_planes_for_quadrants(params.get("quadrants"))
            if "x" in cuts:
                enclosure_bounds["bx0"] = -enclosure_bounds["bx1"]
            if "y" in cuts:
                enclosure_bounds["by0"] = -enclosure_bounds["by1"]
        geometry = resolve_geometry(full).geometry
        return _geometry_dimensions(geometry, enclosure_bounds=enclosure_bounds)
    finally:
        for logger in loggers:
            logger.removeFilter(warning_filter)


def _geometry_dimensions(
    geometry, *, enclosure_bounds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if getattr(geometry, "roundover", None) is not None:
        model = geometry.roundover
        box = np.asarray(model.bounds)
        return {"mouth_opening":[2*float(model.body(model.length)[0][1])]*2,
                "horn_overall":(box[1]-box[0]).tolist()}
    inner = geometry.inner_points
    points = [inner.reshape(-1, 3)]
    if getattr(geometry, "ath_rear_points", None) is not None:
        points.append(geometry.ath_rear_points.reshape(-1, 3))
        points.append(np.asarray(geometry.ath_rear_apex).reshape(1, 3))
    elif geometry.outer_points is not None:
        points.append(geometry.outer_points.reshape(-1, 3))
        rear = geometry.outer_points[:, 0].copy()
        rear[:, 2] = inner[:, 0, 2].mean() - geometry.wall_thickness_mm
        points.append(rear)
    result = {
        "mouth_opening": np.ptp(inner[:, -1, :2], axis=0).tolist(),
        "horn_overall": np.ptp(np.concatenate(points), axis=0).tolist(),
    }
    if geometry.enclosure is not None:
        bounds = enclosure_bounds if enclosure_bounds is not None else measure_enclosure_box_bounds(
            inner, geometry.enclosure, closed=True)
        result["enclosure_overall"] = [bounds["bx1"] - bounds["bx0"],
            bounds["by1"] - bounds["by0"], bounds["z_front"] - bounds["z_back"]]
        if bounds.get("flush_mount"):
            result["mouth_opening"] = result["enclosure_overall"][:2]
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


def dimension_metadata(
    config: Mapping[str, Any], lod: str,
    resolved_input: _ResolvedDimensionInput | None = None,
) -> dict[str, Any]:
    """Coarse frames only look up; settled frames resolve each design once.

    Resolve an owned snapshot decoded from its immutable content key. Options and
    visibility are deliberately excluded; only equivalent physical resolution
    can be reused, never display-LOD buffers.
    Measurement errors are cached too. Build refusals propagate to the caller.
    """
    try:
        key = _config_key(config)
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
    except EnclosureBaffleError:
        raise
    except Exception as exc:
        return _unavailable(exc)
    try:
        try:
            snapshot = json.loads(key)
            dimensions = (canonical_dimensions(snapshot) if resolved_input is None
                          else canonical_dimensions(snapshot, resolved_input))
            if dimensions is None:
                raise ValueError("canonical measurement returned no dimensions")
            result = {"dimensions_mm": dimensions, "dimensions_status": "current"}
            requested = _requested_dimensions(snapshot)
            if requested is not None:
                result["dimensions_requested_mm"] = requested
            _validate_finite_metadata(result, "dimensions")
            stored = copy.deepcopy(result)
        except EnclosureBaffleError:
            raise
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
