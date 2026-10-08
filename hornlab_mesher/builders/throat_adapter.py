"""Exact circular cubic/conic surfaces for native zero-wall adapters."""

from __future__ import annotations

import math

import numpy as np

from ._occ import require_gmsh


def build_adapter_surfaces(geometry):
    """Return welded wall faces and a flat cap on the actual driver edge."""
    gmsh = require_gmsh()
    occ = gmsh.model.occ
    meridian = geometry.adapter_meridian
    scale = geometry.adapter_scale
    cubic = np.asarray(meridian.cubic)*scale
    body = np.asarray(meridian.body_poles)*scale
    point_tags = {}

    def point(zr):
        key = tuple(float(value) for value in zr)
        if key not in point_tags:
            point_tags[key] = occ.addPoint(key[1], 0, key[0])
        return point_tags[key]

    curves = [
        (1, occ.addBezier([point(row) for row in cubic])),
        (1, occ.addBSpline([point(row) for row in body], degree=2,
                           weights=meridian.body_weights.tolist(), knots=[0., 1.], multiplicities=[3, 3])),
    ]
    wall = []
    for quadrant in range(4):
        rotated = occ.copy(curves)
        if quadrant:
            occ.rotate(rotated, 0, 0, 0, 0, 0, 1, quadrant*math.pi/2)
        revolved = occ.revolve(rotated, 0, 0, 0, 0, 0, 1, math.pi/2)
        wall.extend((dim, tag) for dim, tag in revolved if dim == 2)
    occ.remove(curves, recursive=True)
    # Boolean fragments weld all quarter seams and the cubic/conic join.
    # The source disk participates in the same operation, sharing its rim.
    cap = occ.addDisk(0, 0, 0, cubic[0, 1], cubic[0, 1])
    _, mapping = occ.fragment(wall, [(2, cap)], removeObject=True, removeTool=True)
    wall_faces = sorted({(dim, tag) for part in mapping[:-1] for dim, tag in part if dim == 2})
    cap_faces = sorted({(dim, tag) for dim, tag in mapping[-1] if dim == 2})
    if not wall_faces or not cap_faces:
        raise ValueError("Curved adapter refused: exact surface construction produced an empty wall or source.")
    occ.synchronize()
    return wall_faces, cap_faces
