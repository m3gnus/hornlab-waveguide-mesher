"""Explicit curvature-matched circular termination of a circular OSSE body."""
from __future__ import annotations

import copy
import math
from typing import Mapping

import numpy as np

from .composite_meridian import CompositeMeridian, _fail, _number

FORMULA = "OSSE-ARC"
KEY = "terminating_arc"


def configuration(config, resolver):
    """Inspect original intent before delegating the bounded ordinary body."""
    from .native_boundary import native_features, validate_native_boundary
    marker = validate_native_boundary(config)
    features = native_features(config)
    if marker != FORMULA and "arc" not in features:
        return None
    profile = config.get("profile", {})
    profile = profile if isinstance(profile, Mapping) else {}
    # The shared boundary strips markers; this construction never accepted padding.
    names = [section[key] for section in (config, profile) for key in ("formula", "type") if key in section]
    if KEY not in config or any(not isinstance(name, str) or name.upper() != marker for name in names):
        _fail(f"{KEY} requires the exact root key and formula {marker or FORMULA}.")
    if set(profile)-{"formula","type","L_mm","r0_mm","a_deg","a0_deg","k","s"}:
        _fail("only canonical scalar unterminated profile controls are qualified.")
    value = config[KEY]
    if not isinstance(value, Mapping) or set(value) != {"contract_revision", "join_t", "end_tangent_deg"}:
        _fail("terminating_arc requires exactly contract_revision, join_t and end_tangent_deg.")
    if type(value["contract_revision"]) is not int or value["contract_revision"] != 1:
        _fail("contract_revision must be integer 1.")
    payload = {"contract_revision": 1, "join_t": _number(value["join_t"], "join_t"),
               "end_tangent_deg": _number(value["end_tangent_deg"], "end_tangent_deg")}
    # Reuse the strict scalar body (or adapter) boundary; map the body before
    # the circle. This temporary parser request never escapes.
    clean = copy.deepcopy(dict(config))
    clean.pop(KEY)
    clean.pop("type", None)
    for key in ("formula", "type"):
        clean.setdefault("profile", {}).pop(key, None)
    clean["axial_scale"] = config["axial_scale"] if "axial" in features else config.get("scale", 1)
    if "adapter" in features:
        from .adapter_axial import KEY as ADAPTER_KEY, configuration as adapter_configuration
        clean["formula"] = "OSSE-ADAPTER-AXIAL"
        params, _, mode = adapter_configuration(clean, resolver)
        params.pop(ADAPTER_KEY)
        params.pop("construction_fingerprint")
        if "axial" in features:
            params["absoluteAxialScale"] = _number(config["axial_scale"], "axial_scale")
    else:
        from .axial_scale import configuration as axial_configuration
        clean["formula"] = "OSSE-AXIAL"
        params, _, mode = axial_configuration(clean, resolver)
        if "axial" not in features:
            params.pop("absoluteAxialScale")
        params.pop("axialScaleFingerprint", None)
    params["terminatingArc"] = payload
    params["type"] = marker
    model = CompositeMeridian.from_params(params)
    params["terminatingArcFingerprint"] = model.fingerprint
    return params, marker, mode


def resolve(config,params,allow_large_mesh=None):
    from .config_builder import ResolvedGeometry,_mesh_density_from_config
    from .geometry import _ArcPointGridHornGeometry
    model=CompositeMeridian.from_params(params)
    body=model.body(np.linspace(model.conic_start_z,model.join_z,33))[0]
    arc=model.arc(np.linspace(model.beta,model.end,33))[1:]
    meridian=np.vstack((body,arc))
    if model.throat is not None:
        meridian=np.vstack((model.throat.cubic_values(np.linspace(0,1,17))[0][:-1],meridian))
    az=np.arange(64)*2*math.pi/64
    grid=np.stack((np.cos(az)[:,None]*meridian[:,1],np.sin(az)[:,None]*meridian[:,1],
        np.broadcast_to(meridian[:,0],(64,len(meridian)))),axis=-1)
    grid=np.frombuffer(grid.tobytes(),dtype=np.float64).reshape(grid.shape)
    geometry=_ArcPointGridHornGeometry(inner_points=grid,wall_thickness_mm=0,source_shape=0,
        source_radius_mm=-1,source_curv=0,source_auto_angle_deg=model.source_angle_deg,
        closed=True,symmetry_planes=(),vertical_offset_mm=model.offset,arc_meridian=model)
    return ResolvedGeometry(geometry,_mesh_density_from_config(config,allow_large_mesh=allow_large_mesh),
        model.identity["formula"],"bare","1234",None,config.get("mesh",{}).get("scale_to_metres",True),model.metadata())


def validate_density(density):
    """Keep strict native resource intent separate from legacy coercions."""
    from dataclasses import asdict
    from .geometry import MeshDensity
    if type(density) is not MeshDensity:
        _fail("direct density must be MeshDensity or None.")
    defaults=MeshDensity()
    allowed={"throat_res_mm","mouth_res_mm","rear_res_mm","max_triangles","allow_large_mesh"}
    for name,value in asdict(density).items():
        if name not in allowed and (type(value) is not type(getattr(defaults,name)) or value != getattr(defaults,name)):
            _fail(f"unsupported direct density control {name}.")
    for name in ("throat_res_mm","mouth_res_mm","rear_res_mm"):
        if not .01 <= _number(getattr(density,name),name) <= 10000:
            _fail(f"{name} must be in [0.01,10000] mm.")
    if type(density.allow_large_mesh) is not bool:
        _fail("allow_large_mesh must be a boolean.")
    if type(density.max_triangles) is not int or not 0 < density.max_triangles <= 10_000_000:
        _fail("max_triangles must be a positive integer no larger than 10000000.")
