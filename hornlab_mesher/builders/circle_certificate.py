"""One facet-to-surface certificate for revolved circular meridian arcs."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._occ import require_gmsh


@dataclass(frozen=True)
class CircleArc:
    """Meridian circle (z,r) = center+radius*(sin phi,-cos phi), phi in [start,end].

    Radius grows with phi on [start,end] within [0,pi]; both the terminating
    arc and the mouth lip satisfy this, and the local reach below relies on it.
    """
    center_z: float
    center_r: float
    radius: float
    start: float
    end: float

    def points(self, phi):
        phi = np.asarray(phi, dtype=np.float64)
        return np.stack((self.center_z+self.radius*np.sin(phi), self.center_r-self.radius*np.cos(phi)), axis=-1)


def certify_circle_facets(groups, label):
    """Bound each actual triangle's distance to its revolved analytic circle.

    ``groups`` pairs a CircleArc with the OCC faces meshed on it. The surface
    Hessian bounds interpolation over each unwrapped parameter triangle: each
    coordinate variance is at most range squared / 4 and the mixed covariance
    at most the product of ranges / 4. Vertices are projected onto the finite
    arc, so endpoint OCC roundoff contributes to the actual residual. Returns
    the largest bound in mm and the number of triangles certified.
    """
    gmsh = require_gmsh()
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = np.asarray(xyz).reshape(-1, 3)
    lookup = {int(tag): i for i, tag in enumerate(tags)}
    largest = 0.
    count = 0
    for circle, faces in groups:
        # Unwrap around the arc's middle: the arc spans less than 2*pi, so a
        # terminal at +/-pi (a lip ending at pi) keeps one continuous chart.
        middle = (circle.start+circle.end)/2
        reach = float(circle.points(circle.end)[1])
        guard = 1e-10*max(1., circle.radius, abs(circle.center_z), circle.center_r, reach)
        for face in faces:
            types, _, nodes = gmsh.model.mesh.getElements(2, face)
            for kind, element_nodes in zip(types, nodes):
                if int(kind) != 2:
                    raise ValueError(f"{label} requires first-order triangle elements")
                indexes = np.fromiter((lookup[int(tag)] for tag in element_nodes), dtype=np.int64)
                vertices = xyz[indexes].reshape(-1, 3, 3)
                r = np.hypot(vertices[:, :, 0], vertices[:, :, 1])
                phi = np.arctan2(vertices[:, :, 2]-circle.center_z, circle.center_r-r)
                phi = middle+(phi-middle+math.pi) % (2*math.pi)-math.pi
                phi = np.clip(phi, circle.start, circle.end)
                az = np.arctan2(vertices[:, :, 1], vertices[:, :, 0])
                az = az[:, :1]+(az-az[:, :1]+math.pi) % (2*math.pi)-math.pi
                dp, da = np.ptp(phi, axis=1), np.ptp(az, axis=1)
                if np.any(dp >= math.pi) or np.any(da >= math.pi):
                    raise ValueError(f"{label} triangle has an ambiguous angular parameter chart; reduce mouth resolution")
                projected = circle.points(phi)
                residual = np.max(np.hypot(vertices[:, :, 2]-projected[:, :, 0], r-projected[:, :, 1]), axis=1)+guard
                # Every convex parameter triangle has r <= r(max(phi)). Its
                # angular Hessian norm uses this local supremum, rounded up.
                local_reach = np.nextafter(circle.center_r-circle.radius*np.cos(np.max(phi, axis=1))+guard, np.inf)
                bound = (circle.radius*dp*dp+2*circle.radius*dp*da+local_reach*da*da)/8+residual
                largest = max(largest, float(np.max(bound, initial=0)))
                count += len(vertices)
    return largest, count
