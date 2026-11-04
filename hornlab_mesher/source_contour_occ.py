"""Shared exact OCC revolution of a SourceContour meridian."""
import math


def revolve_contour(occ, contour, center_mm=(0, 0, 0), *, half_x=False):
    """Return point and segment face maps; half_x emits only x >= center.x."""
    cx, cy, cz = center_mm
    phi = -math.pi / 2 if half_x else 0

    def point(r, z):
        return occ.addPoint(cx + r * math.cos(phi), cy + r * math.sin(phi), cz + z)

    points = {p.id: point(p.r_mm, p.z_mm) for p in contour.points}
    faces = {}
    for i, segment in enumerate(contour.segments):
        a, b = points[segment.start], points[segment.end]
        if segment.kind == "line":
            edge = occ.addLine(a, b)
        else:
            # Three points retain the authored branch, including semicircles.
            midpoint = point(*contour.evaluate(i, 0.5))
            edge = occ.addCircleArc(a, midpoint, b, center=False)
        swept = occ.revolve([(1, edge)], cx, cy, cz, 0, 0, 1,
                            math.pi if half_x else 2 * math.pi)
        patches = [tag for dim, tag in swept if dim == 2]
        if len(patches) != 1:
            raise ValueError("a contour segment must revolve into exactly one face")
        faces[segment.id] = patches[0]
    return points, faces


def fit_contour_rim(occ, contour, rim_curves, center_mm=(0, 0, 0)):
    """Fit only the outer patch on the design wall's existing WG0 edges.

    Interior segments retain exact revolutions. Each outer sector shares its
    actual wall edge and radial seams; source points constrain the interior.
    The caller must certify and report the departure from the exact meridian.
    """
    import gmsh
    import numpy as np

    _, exact = revolve_contour(occ, contour, center_mm)
    index = len(contour.segments) - 1
    outer = contour.segments[index]
    occ.remove([(2, exact[outer.id])], recursive=False)
    cx, cy, cz = center_mm
    inner = contour.points[index]
    radial_edges = {}
    poles = {}

    def point(r, z, phi):
        return occ.addPoint(cx + r * math.cos(phi), cy + r * math.sin(phi), cz + z)

    def radial(vertex):
        if vertex in radial_edges:
            return radial_edges[vertex]
        x, y, _ = gmsh.model.getValue(0, vertex, [])
        phi = math.atan2(y - cy, x - cx)
        if inner.r_mm == 0:
            start = poles.setdefault('axis', point(0, inner.z_mm, 0))
        else:
            start = point(inner.r_mm, inner.z_mm, phi)
        points = [start] + [point(*contour.evaluate(index, j / 16), phi) for j in range(1, 16)] + [vertex]
        radial_edges[vertex] = (occ.addSpline(points), start, phi)
        return radial_edges[vertex]

    sectors = []
    for curve in rim_curves:
        lo, hi = gmsh.model.getParametrizationBounds(1, abs(curve))
        ends = gmsh.model.getBoundary([(1, abs(curve))], oriented=False)
        if len(ends) != 2:
            raise ValueError('fitted WG0 rim must have distinct sector endpoints')
        vertices = [tag for _, tag in ends]
        first = gmsh.model.getValue(1, abs(curve), [float(lo[0])])
        if np.linalg.norm(np.asarray(gmsh.model.getValue(0, vertices[0], [])) - first) > 1e-7:
            vertices.reverse()
        a, b = vertices
        ea, pa, phi_a = radial(a)
        eb, pb, phi_b = radial(b)
        loop = [abs(curve), -eb]
        if inner.r_mm:
            delta = (phi_b - phi_a + math.pi) % (2 * math.pi) - math.pi
            midpoint = point(inner.r_mm, inner.z_mm, phi_a + delta / 2)
            arc = occ.addCircleArc(pb, midpoint, pa, center=False)
            loop.append(arc)
        loop.append(ea)
        # Sample each actual WG0 sector's azimuth, while all interior
        # constraints remain on the authored canonical surface.
        mid = gmsh.model.getValue(1, abs(curve), [(float(lo[0])+float(hi[0]))/2])
        phi_mid = math.atan2(mid[1]-cy, mid[0]-cx)
        delta = (phi_b - phi_a + math.pi) % (2 * math.pi) - math.pi
        constraints = [point(*contour.evaluate(index, j/16), phi)
                       for phi in (phi_a+delta/4, phi_mid, phi_a+3*delta/4)
                       for j in range(1,16)]
        sectors.append(occ.addSurfaceFilling(occ.addCurveLoop(loop), pointTags=constraints,
                                             degree=3, numPointsOnCurves=32, numIter=5,
                                             tol2d=1e-9, tol3d=1e-7, maxDegree=12, maxSegments=32))
    return {segment: [face] for segment, face in exact.items() if segment != outer.id} | {outer.id: sectors}
