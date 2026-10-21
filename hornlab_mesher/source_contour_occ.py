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
