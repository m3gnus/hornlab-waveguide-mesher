"""Validate the emitted free-standing bore/shell, without changing density."""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree

from .facet_fit import repair_fitted_bore_facets
from .preview.intersections import _triangles_intersect


class BoundaryIndex:
    """Radius-bucketed broad phase; large caps do not inflate every query."""

    def __init__(self, facets):
        self.facets = facets
        self.low, self.high = facets.min(1), facets.max(1)
        centers = facets.mean(1)
        radii = np.linalg.norm(facets - centers[:, None], axis=2).max(1)
        bins = np.floor(np.log2(np.maximum(radii, 1e-8))).astype(int)
        self.buckets = []
        for key in np.unique(bins):
            ids = np.flatnonzero(bins == key)
            self.buckets.append((ids, cKDTree(centers[ids]), radii[ids].max()))

    def hits(self, facets):
        pairs = []
        for start in range(0, len(facets), 512):
            chunk = facets[start:start + 512]
            centers = chunk.mean(1)
            radii = np.linalg.norm(chunk - centers[:, None], axis=2).max(1)
            low, high = chunk.min(1), chunk.max(1)
            for ids, tree, radius in self.buckets:
                candidates = tree.query_ball_point(centers, radii + radius + 1e-8)
                lengths = np.fromiter(map(len, candidates), dtype=int)
                if not lengths.sum():
                    continue
                ii = np.repeat(np.arange(len(chunk)), lengths)
                jj = ids[np.concatenate(candidates).astype(int)]
                overlap = np.all((self.high[jj] >= low[ii] - 1e-8)
                                 & (self.low[jj] <= high[ii] + 1e-8), axis=1)
                ii, jj = ii[overlap], jj[overlap]
                if len(ii):
                    hit = _triangles_intersect(chunk[ii], self.facets[jj])
                    pairs.extend(zip((ii[hit] + start).tolist(), jj[hit].tolist()))
        return sorted(pairs)


def _shared_contact_only(points, first, second):
    """Allow the common topological vertex/edge, never a shrunken overlap.

    For one shared vertex, intersect each face with the other's plane. Their
    intervals must extend in opposite directions from that vertex. Coplanar
    faces instead use their vertex cones, including boundary rays: overlapping
    cones imply contact beyond the common vertex, however narrow it is.
    Ambiguous/degenerate results conservatively reject the proposal.
    """
    common = set(first).intersection(second)
    if len(common) not in (1, 2):
        return False
    a, b = points[first], points[second]
    na = np.cross(a[1] - a[0], a[2] - a[0])
    nb = np.cross(b[1] - b[0], b[2] - b[0])
    if min(np.linalg.norm(na), np.linalg.norm(nb)) == 0:
        return False
    na, nb = na / np.linalg.norm(na), nb / np.linalg.norm(nb)
    direction = np.cross(na, nb)
    if np.linalg.norm(direction) <= 1e-12:
        axis = int(np.argmax(np.abs(na)))
        if len(common) == 2:
            u, v = sorted(common)
            edge = np.delete(points[v] - points[u], axis)
            pa = np.delete(points[next(k for k in first if k not in common)] - points[u], axis)
            pb = np.delete(points[next(k for k in second if k not in common)] - points[u], axis)
            sa = edge[0] * pa[1] - edge[1] * pa[0]
            sb = edge[0] * pb[1] - edge[1] * pb[0]
            return sa * sb < 0
        origin = points[next(iter(common))]
        va = np.delete(points[[k for k in first if k not in common]] - origin, axis, axis=1)
        vb = np.delete(points[[k for k in second if k not in common]] - origin, axis, axis=1)
        for rays, cone in ((va, vb), (vb, va)):
            if np.linalg.det(cone.T) == 0:
                return False
            coordinates = np.linalg.solve(cone.T, rays.T)
            if np.any(np.all(coordinates >= 0, axis=0)):
                return False
        return True
    if len(common) == 2:
        # Distinct planes containing the same edge meet only on that edge.
        return True
    direction /= np.linalg.norm(direction)
    origin = points[next(iter(common))]
    def extent(face, normal):
        ends = points[[k for k in face if k not in common]] - origin
        distances = ends @ normal
        if distances[0] * distances[1] > 0:
            return None  # This face meets the other plane only at the vertex.
        denominator = distances[0] - distances[1]
        if denominator == 0:
            return 0.0  # Ambiguous: do not authorize a swap.
        endpoint = ends[0] + distances[0] / denominator * (ends[1] - ends[0])
        return float(endpoint @ direction)
    ta, tb = extent(first, nb), extent(second, na)
    return ta is None or tb is None or ta * tb < 0


def validate_shell_facets(points, triangles, surfaces, physical, groups,
                          wall_mm, symmetry_axes=()):
    """Inspect final mm arrays; repair in place or refuse before publication.

    CAD patch identities survive all postprocessing filters. Clean arrays are
    never written to. Repairs keep vertices, patch boundaries, physical groups
    and triangle count, and must survive the same sliver predicate as output.
    """
    import gmsh
    from .mesher import MesherError

    start = time.monotonic()
    inner_ids = np.flatnonzero(np.isin(surfaces, groups.get('inner', ())))
    outer_ids = np.flatnonzero(np.isin(surfaces, groups.get('outer', ())))
    if not len(inner_ids) or not len(outer_ids):
        raise MesherError('free-standing boundary has no bore or outer shell facets')
    shell = BoundaryIndex(points[triangles[outer_ids]])
    pairs = shell.hits(points[triangles[inner_ids]])
    stats = {'pairs_before': len(pairs), 'pairs_after': len(pairs),
             'changed_facets': 0, 'seconds': time.monotonic() - start}
    if not pairs:
        return stats

    # Only failing meshes pay for OCC separation queries. A small true CAD
    # gap cannot be fixed by changing a diagonal and must not be disguised.
    patch_pairs = sorted({(int(surfaces[inner_ids[i]]),
                           int(surfaces[outer_ids[j]])) for i, j in pairs})
    clearance = float('inf')
    for bore, outer in patch_pairs:
        try:
            distance = gmsh.model.occ.getDistance(2, bore, 2, outer)[0]
        except Exception as exc:
            raise MesherError('free-standing bore/shell facets intersect; CAD '
                              'clearance measurement failed. Change wall '
                              'thickness or rollback and rebuild.') from exc
        if not np.isfinite(distance) or distance < 0:
            raise MesherError('free-standing bore/shell facets intersect; CAD '
                              'clearance could not be measured. Change wall '
                              'thickness or rollback and rebuild.')
        clearance = min(clearance, distance)
    stats['cad_clearance_mm'] = clearance
    if clearance < max(0.005, 0.1 * wall_mm):
        raise MesherError(
            f'free-standing CAD bore and outer shell nearly touch or cross '
            f'(clearance {clearance:.6g} mm, wall thickness {wall_mm:.6g} mm; '
            f'{len(pairs)} emitted facet intersections). Reduce rollback or '
            'change wall thickness; mesh resolution cannot restore CAD clearance.')

    original = triangles.copy()
    # Guard against every other emitted face, including rim, source and rear
    # caps. Accepted proposals cannot introduce a crossing with a new face.
    def guard(adjacent, proposed):
        if len(set(physical[adjacent])) != 1:
            return False
        xyz = points[proposed]
        for axis in symmetry_axes:
            if np.any(np.all(np.abs(xyz[:, :, 'xyz'.index(axis)]) <= 1e-9, axis=1)):
                return False
        obstacles = BoundaryIndex(points[triangles])
        for i, j in obstacles.hits(xyz):
            # The opposite wall already has the stricter decreasing-count /
            # no-new-shell-facet guard in the swap routine. Existing crossings
            # may need several successive swaps to disappear.
            if j in adjacent or surfaces[j] in active_shell:
                continue
            if not set(proposed[i]).intersection(triangles[j]):
                return False
            if not _shared_contact_only(points, proposed[i], triangles[j]):
                return False
        return True

    # Same OCC-parameter convexity, winding, sliver and no-new-crossing guards
    # as fitted grids, now on the emitted boundary and on either wall.
    remaining = len(pairs)
    try:
        for _ in range(remaining):
            for roles in (groups, {'inner': groups['outer'], 'outer': groups['inner']}):
                active_shell = set(roles['outer'])
                repair_fitted_bore_facets(points, triangles, surfaces, roles,
                                         index_factory=BoundaryIndex, proposed_guard=guard)
            after = BoundaryIndex(points[triangles[outer_ids]]).hits(points[triangles[inner_ids]])
            if not after or len(after) >= remaining:
                break
            remaining = len(after)
    except Exception as exc:
        triangles[:] = original
        raise MesherError(
            f'free-standing bore/shell boundary has {len(pairs)} facet '
            'intersections and its CAD patches do not support safe local repair. '
            'Change the requested mesh resolution or wall thickness.') from exc
    if after:
        triangles[:] = original
        raise MesherError(
            f'free-standing bore/shell tessellation has {len(after)} remaining '
            f'facet intersections after safe local diagonal swaps '
            f'({len(pairs)} before; CAD clearance {clearance:.6g} mm, '
            f'wall thickness {wall_mm:.6g} mm). Change the requested mesh '
            'resolution or wall thickness; no vertices or elements were added.')
    stats.update(pairs_after=0,
                 changed_facets=int(np.any(triangles != original, axis=1).sum()),
                 seconds=time.monotonic() - start)
    return stats
