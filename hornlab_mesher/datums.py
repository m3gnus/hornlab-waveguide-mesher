"""Derive stable CAD-link datums from realized point-grid geometry.

The sampled grid owns the horn interfaces and :class:`BuiltGeometry` owns the
realized enclosure.  Keeping both inputs explicit prevents requested config
values from leaking into the CAD contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .builders.enclosure import sample_enclosure_plan
from .geometry import BuiltGeometry, PointGridBuildMode, PointGridHornGeometry, StandaloneSourceGeometry


DEFAULT_PLANE_TOLERANCE_MM = 1.0e-6


def _axis_direction(source_axis: str) -> list[float]:
    axis = str(source_axis)
    sign = -1.0 if axis.startswith("-") else 1.0
    letter = axis[-1:]
    if letter not in "xyz":
        raise ValueError(f"unsupported source axis {source_axis!r}")
    direction = [0.0, 0.0, 0.0]
    direction["xyz".index(letter)] = sign
    return direction


def _placed(points: NDArray[np.float64], offset_mm: float) -> NDArray[np.float64]:
    result = np.array(points, dtype=np.float64, copy=True)
    result[:, 1] += float(offset_mm)
    return result


def _polyline(points: NDArray[np.float64]) -> dict[str, Any]:
    return {
        "type": "polyline",
        "closed": True,
        "points_mm": np.asarray(points, dtype=np.float64).tolist(),
    }


def _fit_plane(
    points: NDArray[np.float64], *, tolerance_mm: float
) -> tuple[bool, dict[str, Any]]:
    samples = np.asarray(points, dtype=np.float64)
    origin = np.mean(samples, axis=0)
    _u, _s, vh = np.linalg.svd(samples - origin, full_matrices=False)
    normal = np.asarray(vh[-1], dtype=np.float64)
    # Plane orientation is deterministic and follows the horn's usual +z axis.
    dominant = int(np.argmax(np.abs(normal)))
    if normal[dominant] < 0.0:
        normal *= -1.0
    errors = np.abs((samples - origin) @ normal)
    max_error = float(np.max(errors)) if len(errors) else 0.0
    return max_error <= float(tolerance_mm), {
        "type": "plane",
        "origin_mm": origin.tolist(),
        "normal": normal.tolist(),
        "max_error_mm": max_error,
        "tolerance_mm": float(tolerance_mm),
    }


def _axis_plane(axis: str, value: float) -> dict[str, Any]:
    normal = [0.0, 0.0, 0.0]
    normal["xyz".index(axis)] = 1.0
    origin = [0.0, 0.0, 0.0]
    origin["xyz".index(axis)] = float(value)
    return {
        "type": "plane",
        "origin_mm": origin,
        "normal": normal,
        "exact": True,
    }


def derive_datums(
    geometry: PointGridHornGeometry | StandaloneSourceGeometry,
    built: BuiltGeometry | None,
    *,
    plane_tolerance_mm: float = DEFAULT_PLANE_TOLERANCE_MM,
) -> dict[str, Any]:
    """Return the v1 datum catalogue for a realized exportable build.

    Only freestanding and enclosure solids have defined v1 semantics.  Point
    rings are placed by ``vertical_offset_mm`` here because the STORED geometry
    keeps its unshifted grid; the bundle's point-grid payload is shipped
    already placed so every artifact in a bundle shares one link-local frame.
    """

    if type(geometry) is StandaloneSourceGeometry:
        return geometry.datums()
    controls = getattr(geometry, "controls_meridian", None)
    if controls is not None:
        az=np.linspace(0,2*np.pi,64,endpoint=False)
        radius=float(controls.body(1)[0][1])
        mouth=np.column_stack((radius*np.cos(az),radius*np.sin(az)+controls.offset,np.full(64,controls.depth)))
        plane=lambda z:{**_axis_plane("z",z),"origin_mm":[0.,controls.offset,z],"nominal":False}
        return {"rim_planar":True,"construction_fingerprint":controls.fingerprint,
            "WG_AXIS":{"type":"axis","origin_mm":[0.,controls.offset,0.],"direction":[0.,0.,1.]},
            "WG_THROAT_PLANE":plane(0.),"WG_ADAPTER_JOIN_PLANE":plane(controls.adapter_length),
            "WG_MOUTH_PLANE":plane(controls.depth),"WG_MOUTH_OUTLINE_INNER":_polyline(mouth),
            "WG_GEOM_MIDPLANE_Y":_axis_plane("y",controls.offset),
            "WG_SOLVER_CUT_PLANE_Y":_axis_plane("y",0.),"WG_SOLVER_CUT_PLANE_X":_axis_plane("x",0.)}
    arc=getattr(geometry,"arc_meridian",None)
    if arc is not None:
        az=np.arange(64)*2*np.pi/64
        z,r=arc.opening
        mouth=np.column_stack((r*np.cos(az),r*np.sin(az)+arc.offset,np.full(64,z)))
        def plane(z):
            return {**_axis_plane("z",z),"origin_mm":[0.,arc.offset,float(z)]}
        return {"rim_planar":True,"WG_AXIS":{"type":"axis","origin_mm":[0.,arc.offset,0.],"direction":[0.,0.,1.]},
            "WG_THROAT_PLANE":{**plane(0),"nominal":False},"WG_ARC_JOIN_PLANE":plane(arc.join_z),
            "WG_MOUTH_PLANE":plane(z),"WG_MOUTH_OUTLINE_INNER":_polyline(mouth),
            "WG_BODY_MOUTH_REFERENCE_PLANE":{**plane(arc.length),"nominal":True,"reference":True},
            "WG_GEOM_MIDPLANE_Y":_axis_plane("y",arc.offset),"WG_SOLVER_CUT_PLANE_Y":_axis_plane("y",0.),
            "WG_SOLVER_CUT_PLANE_X":_axis_plane("x",0.)}
    from .adapter_axial import PhysicalAdapter
    adapter = getattr(geometry, "adapter_meridian", None)
    if isinstance(adapter, PhysicalAdapter):
        az = np.arange(64)*2*np.pi/64
        offset = adapter.offset
        mouth_z = float(adapter.body_poles[-1,0])
        radius = float(adapter.body_poles[-1,1])
        mouth = np.column_stack((radius*np.cos(az),radius*np.sin(az)+offset,np.full(64,mouth_z)))
        def plane(z):
            return {**_axis_plane("z",z),"origin_mm":[0.,offset,z],"nominal":False}
        return {"rim_planar":True,"WG_AXIS":{"type":"axis","origin_mm":[0.,offset,0.],"direction":[0.,0.,1.]},
            "WG_THROAT_PLANE":plane(0.),"WG_ADAPTER_JOIN_PLANE":plane(float(adapter.cubic[-1,0])),
            "WG_MOUTH_PLANE":plane(mouth_z),"WG_MOUTH_OUTLINE_INNER":_polyline(mouth),
            "WG_GEOM_MIDPLANE_Y":_axis_plane("y",offset),"WG_SOLVER_CUT_PLANE_Y":_axis_plane("y",0.),
            "WG_SOLVER_CUT_PLANE_X":_axis_plane("x",0.)}
    axial = getattr(geometry, "axial_model", None)
    if axial is not None:
        az = np.linspace(0,2*np.pi,64,endpoint=False)
        radius = float(axial.body(axial.length)[0][1])
        mouth = np.column_stack((radius*np.cos(az),radius*np.sin(az)+axial.offset,np.full(64,axial.length)))
        throat_plane = {**_axis_plane("z",0),"origin_mm":[0.0,axial.offset,0.0],"nominal":False}
        mouth_plane = {**_axis_plane("z",axial.length),"origin_mm":[0.0,axial.offset,axial.length]}
        return {"rim_planar":True,"WG_AXIS":{"type":"axis","origin_mm":[0.0,axial.offset,0.0],"direction":[0.0,0.0,1.0]},
            "WG_THROAT_PLANE":throat_plane,"WG_MOUTH_PLANE":mouth_plane,
            "WG_MOUTH_OUTLINE_INNER":_polyline(mouth),"WG_GEOM_MIDPLANE_Y":_axis_plane("y",axial.offset),
            "WG_SOLVER_CUT_PLANE_Y":_axis_plane("y",0.0),"WG_SOLVER_CUT_PLANE_X":_axis_plane("x",0.0)}
    mode = geometry.build_mode
    if mode not in {PointGridBuildMode.FREESTANDING, PointGridBuildMode.ENCLOSURE}:
        raise ValueError(
            "wglink datums support only FREESTANDING and ENCLOSURE builds; "
            f"got {mode.value.upper()}"
        )
    points = np.asarray(geometry.inner_points, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3 or points.shape[0] < 3:
        raise ValueError("inner_points must have shape (n_phi, n_length, 3)")

    offset = float(geometry.vertical_offset_mm)
    throat = _placed(points[:, 0, :], offset)
    mouth = _placed(points[:, -1, :], offset)
    roundover = getattr(geometry, "roundover", None)
    if roundover is not None:
        azimuth = np.arctan2(points[:,0,1],points[:,0,0])
        radius = float(roundover.body(roundover.length)[0][1])
        mouth = np.column_stack((radius*np.cos(azimuth),radius*np.sin(azimuth)+offset,
                                 np.full(len(azimuth),roundover.length)))
    throat_planar, throat_plane = _fit_plane(
        throat, tolerance_mm=plane_tolerance_mm
    )
    mouth_planar, mouth_plane = _fit_plane(mouth, tolerance_mm=plane_tolerance_mm)

    datums: dict[str, Any] = {
        "rim_planar": mouth_planar,
        "WG_AXIS": {
            "type": "axis",
            "origin_mm": [0.0, offset, 0.0],
            "direction": _axis_direction(built.source_axis),
        },
        "WG_THROAT_PLANE": {
            **throat_plane,
            "exact": throat_planar,
            "nominal": not throat_planar,
        },
        "WG_MOUTH_OUTLINE_INNER": _polyline(mouth),
        "WG_GEOM_MIDPLANE_Y": _axis_plane("y", offset),
        "WG_SOLVER_CUT_PLANE_Y": _axis_plane("y", 0.0),
        "WG_SOLVER_CUT_PLANE_X": _axis_plane("x", 0.0),
    }
    if mouth_planar:
        datums["WG_MOUTH_PLANE"] = {**mouth_plane, "exact": True}

    if mode is PointGridBuildMode.FREESTANDING:
        if geometry.outer_points is None:
            raise ValueError("freestanding datums require outer_points")
        outer = np.asarray(geometry.outer_points, dtype=np.float64)
        datums["WG_MOUTH_OUTLINE_OUTER"] = _polyline(
            _placed(outer[:, -1, :], offset) if roundover is None else
            np.column_stack((float(roundover.body(roundover.length,True)[0][1])*np.cos(azimuth),
                             float(roundover.body(roundover.length,True)[0][1])*np.sin(azimuth)+offset,
                             np.full(len(azimuth),float(roundover.body(roundover.length,True)[0][0]))))
        )
        return datums

    if geometry.enclosure is None:
        raise ValueError("enclosure datums require HornEnclosure metadata")
    if int(geometry.enclosure.plan_type) != 1:
        raise ValueError(
            f"enclosure plan_type={geometry.enclosure.plan_type} is not supported "
            "for wglink export; only plan_type=1 is buildable"
        )
    bounds = built.enclosure_bounds
    if bounds is None:
        raise ValueError("enclosure datums require realized BuiltGeometry.enclosure_bounds")
    required = {
        "bx0",
        "bx1",
        "by0",
        "by1",
        "z_front",
        "z_back",
        "enc_depth",
        "clamped_edge",
    }
    missing = sorted(required.difference(bounds))
    if missing:
        raise ValueError("enclosure_bounds is missing " + ", ".join(missing))

    z_front = float(bounds["z_front"])
    edge = float(bounds["clamped_edge"])
    sample_args = {
        "edge_type": int(geometry.enclosure.edge_type),
        "z": z_front,
        "plan_type": 1,
        "plan_n": float(geometry.enclosure.plan_n),
    }
    face = sample_enclosure_plan(
        bx0=float(bounds["bx0"]) + edge,
        bx1=float(bounds["bx1"]) - edge,
        by0=float(bounds["by0"]) + edge,
        by1=float(bounds["by1"]) - edge,
        corner_radius=0.1,
        **sample_args,
    )
    envelope = sample_enclosure_plan(
        bx0=float(bounds["bx0"]),
        bx1=float(bounds["bx1"]),
        by0=float(bounds["by0"]),
        by1=float(bounds["by1"]),
        corner_radius=max(0.1, edge),
        **sample_args,
    )
    face[:, 1] += offset
    envelope[:, 1] += offset
    datums.update(
        {
            # The enclosure's outer material boundary at the mouth is the
            # baffle face perimeter, not the inner acoustic bore rim.
            "WG_MOUTH_OUTLINE_OUTER": _polyline(face),
            "WG_BAFFLE_PLANE": _axis_plane("z", z_front),
            "WG_BAFFLE_OUTLINE_FACE": _polyline(face),
            "WG_BAFFLE_OUTLINE_ENVELOPE": _polyline(envelope),
            "WG_ENC_BACK_PLANE": _axis_plane("z", float(bounds["z_back"])),
        }
    )
    return datums


__all__ = ["DEFAULT_PLANE_TOLERANCE_MM", "derive_datums"]
