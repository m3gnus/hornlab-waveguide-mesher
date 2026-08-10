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
from .geometry import BuiltGeometry, PointGridBuildMode, PointGridHornGeometry


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
    geometry: PointGridHornGeometry,
    built: BuiltGeometry,
    *,
    plane_tolerance_mm: float = DEFAULT_PLANE_TOLERANCE_MM,
) -> dict[str, Any]:
    """Return the v1 datum catalogue for a realized exportable build.

    Only freestanding and enclosure solids have defined v1 semantics.  Point
    rings are placed by ``vertical_offset_mm`` here because the STORED geometry
    keeps its unshifted grid; the bundle's point-grid payload is shipped
    already placed so every artifact in a bundle shares one link-local frame.
    """

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
            _placed(outer[:, -1, :], offset)
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
