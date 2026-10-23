"""One facet-to-surface certificate for revolved circular meridian arcs."""
from __future__ import annotations

import math
import numpy as np

from ._occ import require_gmsh
from ..composite_meridian import CircleArc


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
        # A triangle's chart starts at its first vertex; the whole triangle is
        # then moved by whole turns so that vertex lies in the window around
        # the arc's middle. A terminal at +/-pi (a lip ending at pi) keeps one
        # continuous chart, and other triangles are left exactly as they are.
        middle = (circle.start+circle.end)/2
        reach = float(circle.points(circle.end)[1])
        # Add a size-scaled roundoff guard to both residual and local reach.
        # Unlike the v0.2.6 lip certificate, this can raise a rim triangle's
        # bound slightly even though local reach tightens bounds elsewhere.
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
                phi = phi[:, :1]+(phi-phi[:, :1]+math.pi) % (2*math.pi)-math.pi
                phi = phi+2*math.pi*np.floor((middle+math.pi-phi[:, :1])/(2*math.pi))
                phi = np.clip(phi, circle.start, circle.end)
                az = np.arctan2(vertices[:, :, 1], vertices[:, :, 0])
                az = az[:, :1]+(az-az[:, :1]+math.pi) % (2*math.pi)-math.pi
                dp, da = np.ptp(phi, axis=1), np.ptp(az, axis=1)
                if np.any(dp >= math.pi) or np.any(da >= math.pi):
                    raise ValueError(f"{label} triangle has an ambiguous angular parameter chart; reduce mouth resolution")
                projected = circle.points(phi)
                residual = np.max(np.hypot(vertices[:, :, 2]-projected[:, :, 0], r-projected[:, :, 1]), axis=1)+guard
                # Every convex parameter triangle has r <= r(max(phi)). Its
                # angular Hessian norm uses this local supremum plus the
                # guard, rounded up; the residual above includes the guard too.
                local_reach = np.nextafter(circle.center_r-circle.radius*np.cos(np.max(phi, axis=1))+guard, np.inf)
                bound = (circle.radius*dp*dp+2*circle.radius*dp*da+local_reach*da*da)/8+residual
                largest = max(largest, float(np.max(bound, initial=0)))
                count += len(vertices)
    return largest, count
