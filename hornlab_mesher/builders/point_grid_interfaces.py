from __future__ import annotations

import numpy as np

from ..geometry import HornInterface, PointGridHornGeometry
from ._occ import require_gmsh

def _interface_phi_groups(n_phi: int, *, closed: bool) -> list[list[int]]:
    if not closed:
        return [list(range(n_phi))]
    if n_phi < 4:
        return [[*range(n_phi), 0]]
    step = max(1, n_phi // 4)
    groups: list[list[int]] = []
    for group in range(4):
        start = group * step
        stop = (group + 1) * step
        if group == 3:
            groups.append(list(range(start, n_phi)) + [0])
        else:
            groups.append(list(range(start, stop + 1)))
    return groups


def _split_interface_group(indices: list[int]) -> list[list[int]]:
    if len(indices) < 3:
        return [indices]
    mid = len(indices) // 2
    return [indices[: mid + 1], indices[mid:]]


def _normalise_interface_specs(geometry: PointGridHornGeometry, n_length: int) -> tuple[HornInterface, ...]:
    if geometry.interfaces:
        return tuple(
            HornInterface(slice_index=int(spec.slice_index), offset_mm=float(spec.offset_mm))
            for spec in geometry.interfaces
            if float(spec.offset_mm) > 0.0 and 0 <= int(spec.slice_index) < n_length
        )
    if geometry.interface_offset_mm <= 0.0:
        return ()
    return (HornInterface(slice_index=n_length - 1, offset_mm=float(geometry.interface_offset_mm)),)


def _add_offset_interface_surfaces(
    inner_points: np.ndarray,
    *,
    slice_index: int,
    closed: bool,
    offset_mm: float,
) -> list[tuple[int, int]]:
    if offset_mm <= 0.0:
        return []

    gmsh = require_gmsh()
    base = np.asarray(inner_points[:, int(slice_index), :], dtype=np.float64)
    offset = np.array(base, dtype=np.float64, copy=True)
    offset[:, 2] += float(offset_mm)
    center = np.asarray(
        (
            float(np.mean(base[:, 0])) if closed else 0.0,
            float(np.mean(base[:, 1])) if closed else 0.0,
            float(np.mean(offset[:, 2])),
        ),
        dtype=np.float64,
    )

    base_tags = [
        int(gmsh.model.occ.addPoint(float(p[0]), float(p[1]), float(p[2])))
        for p in base
    ]
    offset_tags = [
        int(gmsh.model.occ.addPoint(float(p[0]), float(p[1]), float(p[2])))
        for p in offset
    ]
    center_tag = int(gmsh.model.occ.addPoint(float(center[0]), float(center[1]), float(center[2])))
    radial_lines = {
        i: int(gmsh.model.occ.addLine(center_tag, offset_tags[i]))
        for i in range(len(offset_tags))
    }

    def spline(tags: list[int]) -> int:
        return int(gmsh.model.occ.addBSpline([int(tag) for tag in tags]))

    def line(a: int, b: int) -> int:
        return int(gmsh.model.occ.addLine(int(a), int(b)))

    def surface(curves: list[int], *, plane: bool = False) -> tuple[int, int]:
        try:
            loop = int(gmsh.model.occ.addCurveLoop([int(c) for c in curves], reorient=True))
        except TypeError:
            loop = int(gmsh.model.occ.addCurveLoop([int(c) for c in curves]))
        if plane:
            try:
                return (2, int(gmsh.model.occ.addPlaneSurface([loop])))
            except Exception:
                pass
        return (2, int(gmsh.model.occ.addSurfaceFilling(loop)))

    surfaces: list[tuple[int, int]] = []
    for group in _interface_phi_groups(len(base_tags), closed=closed):
        offset_curves: list[int] = []
        for span in _split_interface_group(group):
            base_curve = spline([base_tags[i] for i in span])
            offset_curve = spline([offset_tags[i] for i in span])
            left = line(base_tags[span[0]], offset_tags[span[0]])
            right = line(base_tags[span[-1]], offset_tags[span[-1]])
            surfaces.append(surface([base_curve, right, -offset_curve, -left]))
            offset_curves.append(offset_curve)
        surfaces.append(surface([radial_lines[group[0]], *offset_curves, -radial_lines[group[-1]]], plane=True))
    return surfaces
