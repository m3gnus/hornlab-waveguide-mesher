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
    faces instead use unshrunk 2D projections: positive-area overlap is
    prohibited, however narrow it is; boundary-only touching is allowed.
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
        return _coplanar_contact_only(a, b, na)
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


def _coplanar_contact_only(a, b, normal):
    """Zero-area contact is allowed; positive-area overlap is prohibited.

    Use unshrunk 2D projections. The tolerance covers floating-point roundoff,
    not welding or a geometric margin that could erase a narrow overlap.
    """
    scale = max(np.abs(a).max(), np.abs(b).max(), 1.0)
    tolerance = 32 * np.finfo(float).eps * scale
    if np.max(np.abs((b - a[0]) @ normal)) > tolerance:
        return False
    axis = int(np.argmax(np.abs(normal)))
    a, b = np.delete(a - a[0], axis, axis=1), np.delete(b - a[0], axis, axis=1)
    for face in (a, b):
        for edge in np.roll(face, -1, axis=0) - face:
            length = np.linalg.norm(edge)
            if length == 0:
                return False
            direction = np.array([-edge[1], edge[0]]) / length
            pa, pb = a @ direction, b @ direction
            if min(pa.max(), pb.max()) - max(pa.min(), pb.min()) <= tolerance:
                return True
    return False


def _contact_only(a, b):
    # Exact equal coordinates represent the welded topological vertices. No
    # proximity tolerance is used to manufacture joins between disjoint faces.
    matches = np.all(a[:, None] == b[None, :], axis=2)
    if matches.any():
        second = np.arange(3, 6)
        for i, j in zip(*np.nonzero(matches)):
            second[j] = i
        return _shared_contact_only(np.vstack((a, b)), np.arange(3), second)
    na, nb = np.cross(a[1]-a[0], a[2]-a[0]), np.cross(b[1]-b[0], b[2]-b[0])
    la, lb = np.linalg.norm(na), np.linalg.norm(nb)
    if la == 0 or lb == 0:
        return False
    na, nb = na/la, nb/lb
    return (np.linalg.norm(np.cross(na, nb)) <= 1e-12
            and _coplanar_contact_only(a, b, na))


class CrossingIndex(BoundaryIndex):
    """Conservative candidates followed by the final contact predicate.

    Validation and repair use the same prohibited-pair count, so harmless
    contacts never trigger a CAD query or a connectivity change.
    """

    def hits(self, facets):
        return [(i, j) for i, j in super().hits(facets)
                if not _contact_only(facets[i], self.facets[j])]


def _proposal_guard(points, triangles, surfaces, physical, symmetry_axes,
                    active_shell, adjacent, proposed):
    if len(set(physical[adjacent])) != 1:
        return False
    xyz = points[proposed]
    for axis in symmetry_axes:
        if np.any(np.all(np.abs(xyz[:, :, 'xyz'.index(axis)]) <= 1e-9, axis=1)):
            return False
    obstacles = CrossingIndex(points[triangles])
    for _, j in obstacles.hits(xyz):
        # The opposite wall has the decreasing-count/no-new-face guard in
        # facet_fit. Every other emitted face must be safe, including faces
        # sharing a vertex or edge with the proposal.
        if j not in adjacent and surfaces[j] not in active_shell:
            return False
    return True


def _refusal_context(points, triangles, surfaces, inner_ids, boundary_ids,
                     pairs, groups, wall_mm, mesh_density):
    faces = triangles[inner_ids[sorted({i for i, _ in pairs})]]
    low, high = points[faces, 2].min(), points[faces, 2].max()
    bore_low, bore_high = points[triangles[inner_ids], 2].min(), points[triangles[inner_ids], 2].max()
    patches = sorted({int(surfaces[boundary_ids[j]]) for _, j in pairs})
    if low >= bore_low + .65 * (bore_high-bore_low):
        region, knob = 'mouth', 'mouth_res_mm'
    elif high <= bore_low + .35 * (bore_high-bore_low):
        region, knob = 'throat', 'throat_res_mm'
    else:
        region, knob = 'bore', 'mouth_res_mm and throat_res_mm'
    if any(tag in groups.get('rear', ()) or tag in groups.get('rear_cap', ()) for tag in patches):
        region, knob = 'rear return/disc', 'rear_res_mm'
    context = (f'{region} region z={low:.6g}..{high:.6g} mm, boundary patches {patches}; '
               f'wall_thickness_mm={wall_mm:.6g} mm')
    if mesh_density is None:
        return context + '; requested mm sizes unavailable to this array-only call', (
            f'Try a thicker wall_thickness_mm (suggested {wall_mm*1.5:.6g} mm) '
            f'or finer (smaller mm) {knob} in the {region} region; '
            'suggestions require rebuilding and validation.')
    names = ('throat_res_mm', 'mouth_res_mm', 'rear_res_mm')
    values = {name: getattr(mesh_density, name) for name in names}
    def show(value):
        if np.isscalar(value):
            return f'{value:.6g}'
        return '[' + ', '.join(f'{v:.6g}' for v in value) + ']'
    context += '; requested ' + ', '.join(f'{name}={show(value)} mm' for name, value in values.items())
    suggestions = []
    for name in knob.split(' and '):
        value = values[name]
        suggestion = np.minimum(np.asarray(value)*.5, 8).tolist()
        suggestions.append(f'{name}={show(suggestion)} mm')
    advice = (f'Try a thicker wall_thickness_mm (suggested {wall_mm*1.5:.6g} mm) '
              f'or finer (smaller mm) resolution in the {region} region: '
              + ', '.join(suggestions) + ' (suggested, not a guaranteed remedy).')
    return context, advice


def _crossing_witnesses(a, b):
    """Sample the actual intersection, not the faces' unrelated centroids."""
    normals = [np.cross(f[1]-f[0], f[2]-f[0]) for f in (a, b)]
    lengths = [np.linalg.norm(n) for n in normals]
    if min(lengths) == 0:
        return []
    na, nb = [n/length for n, length in zip(normals, lengths)]
    tolerance = 64*np.finfo(float).eps*max(np.abs(a).max(), np.abs(b).max(), 1)
    if np.linalg.norm(np.cross(na, nb)) <= 1e-12:
        if np.max(np.abs((a-b[0]) @ nb)) > tolerance:
            return []
        # Clip the coplanar triangle by all three half-planes of the other.
        polygon = list(a)
        for start, end in zip(b, np.roll(b, -1, axis=0)):
            inward = np.cross(nb, end-start)
            inward /= np.linalg.norm(inward)
            clipped = []
            for x, y in zip(polygon, polygon[1:]+polygon[:1]):
                dx, dy = (x-start) @ inward, (y-start) @ inward
                if dx >= -tolerance:
                    clipped.append(x)
                if (dx < -tolerance) != (dy < -tolerance):
                    clipped.append(x + dx/(dx-dy)*(y-x))
            polygon = clipped
            if not polygon:
                return []
        return [np.mean(polygon, axis=0)]
    hits = []
    for face, other, normal in ((a, b, nb), (b, a, na)):
        def inside(x):
            uv = np.linalg.lstsq((other[1:]-other[0]).T, x-other[0], rcond=None)[0]
            return min(uv[0], uv[1], 1-uv.sum()) >= -1e-12
        for x, y in zip(face, np.roll(face, -1, axis=0)):
            dx, dy = (x-other[0]) @ normal, (y-other[0]) @ normal
            if abs(dx) <= tolerance and inside(x):
                hits.append(x)
            if dx*dy < 0:
                hit = x + dx/(dx-dy)*(y-x)
                if inside(hit):
                    hits.append(hit)
    if not hits:
        return []
    direction = np.cross(na, nb)
    ordered = sorted(hits, key=lambda x: x @ direction)
    lo, hi = ordered[0], ordered[-1]
    return [lo + fraction*(hi-lo) for fraction in (.5, .25, .75)]


def _local_cad_clearance(bore, boundary, witnesses, threshold):
    """Confirm near-contact at a facet crossing on trimmed OCC patches.

    Global extrema can occur at a remote legitimate join, with shared or
    duplicated CAD edge tags. Point-to-shape queries stay at the witness and
    respect trimming (unlike orthogonal surface projection). Both patches
    must be within the contact threshold of the witness. Exclude the trim
    neighbourhood of coincident trims so a joined edge cannot supply the remedy.
    This only selects advice: every prohibited pair still repairs or refuses.
    """
    import gmsh

    ends = [gmsh.model.getBoundary([(2, tag)], oriented=False)
            for tag in (bore, boundary)]
    clearance = float('inf')
    join_vertices = None
    for witness in witnesses:
        distances, closest = [], []
        for tag in (bore, boundary):
            entities, values, coords = gmsh.model.occ.getClosestEntities(
                *witness, [(2, tag)])
            if (entities != [(2, tag)] or len(values) != 1 or len(coords) != 3
                    or not np.all(np.isfinite([*values, *coords])) or values[0] < 0):
                raise ValueError('invalid local CAD distance')
            xyz = np.asarray(coords)
            # Check the returned location too: a remote extremum cannot
            # become local evidence even if an API reports zero distance.
            distances.append(max(float(values[0]), np.linalg.norm(xyz-witness)))
            closest.append(xyz)
        gap = max(sum(distances), np.linalg.norm(closest[0]-closest[1]))
        if gap >= threshold:
            continue
        # Check geometric coincidence as well as proximity: separate CAD
        # tags may describe a join, but two merely close trim edges can be
        # genuine missing clearance and must retain CAD-contact advice.
        on_trim, trim_points = [], []
        for xyz, edges in zip(closest, ends):
            if not edges:
                on_trim.append(False)
                trim_points.append(None)
                continue
            _, values, coords = gmsh.model.occ.getClosestEntities(*xyz, edges)
            if (len(values) != 1 or len(coords) != 3
                    or not np.all(np.isfinite([*values, *coords])) or values[0] < 0):
                raise ValueError('invalid CAD trim distance')
            on_trim.append(values[0] <= threshold)
            trim_points.append(coords)
        joined = False
        if all(on_trim):
            joined = True
            for xyz, opposite_edges in zip(trim_points, reversed(ends)):
                _, values, _ = gmsh.model.occ.getClosestEntities(*xyz, opposite_edges)
                if len(values) != 1 or not np.isfinite(values[0]) or values[0] < 0:
                    raise ValueError('invalid CAD join distance')
                joined &= values[0] <= 1e-7
            if not joined:
                # Trims can diverge from a common endpoint. Their nearest
                # points need not coincide even inside that join's neighbourhood.
                # Compare coordinates, since OCC can give the endpoint distinct
                # tags on the two patches (as at a quadrant mouth corner).
                if join_vertices is None:
                    vertices = []
                    for tag in (bore, boundary):
                        vertices.append([gmsh.model.getValue(0, vertex, [])
                            for dim, vertex in gmsh.model.getBoundary(
                                [(2, tag)], oriented=False, recursive=True) if dim == 0])
                    join_vertices = [np.asarray(a) for a in vertices[0]
                        if any(np.linalg.norm(np.asarray(a)-b) <= 1e-7 for b in vertices[1])]
                joined = any(np.linalg.norm(witness-vertex) <= threshold
                             for vertex in join_vertices)
        if not joined:
            return gap
    return clearance


def validate_shell_facets(points, triangles, surfaces, physical, groups,
                          wall_mm, symmetry_axes=(), *, mesh_density=None):
    """Inspect all final mm boundary components; repair or refuse publication.

    Clean arrays remain untouched. Repairs keep vertices, patch boundaries,
    physical groups and triangle counts, and survive the output sliver filter.
    """
    import gmsh
    from .mesher import MesherError

    start = time.monotonic()
    inner_ids = np.flatnonzero(np.isin(surfaces, groups.get('inner', ())))
    # Include every emitted non-bore component, also source and rear caps.
    # Group aliases (rear/rear_cap) cannot duplicate facets in this index.
    boundary_ids = np.flatnonzero(~np.isin(surfaces, groups.get('inner', ())))
    if not len(inner_ids) or not len(boundary_ids):
        raise MesherError('free-standing boundary has no bore or shell facets')
    def scan():
        return CrossingIndex(points[triangles[boundary_ids]]).hits(points[triangles[inner_ids]])
    pairs = scan()
    stats = {'pairs_before': len(pairs), 'pairs_after': len(pairs),
             'changed_facets': 0, 'seconds': time.monotonic() - start}
    if not pairs:
        return stats
    context, advice = _refusal_context(points, triangles, surfaces, inner_ids,
                                      boundary_ids, pairs, groups, wall_mm, mesh_density)
    patch_pairs = sorted({(int(surfaces[inner_ids[i]]),
                           int(surfaces[boundary_ids[j]])) for i, j in pairs})
    clearance = float('inf')
    local_clearance = float('inf')
    threshold = max(0.005, 0.1 * wall_mm)
    for bore, boundary in patch_pairs:
        try:
            distance = gmsh.model.occ.getDistance(2, bore, 2, boundary)[0]
            if not np.isfinite(distance) or distance < 0:
                raise ValueError('invalid CAD distance')
            # A positive global lower bound rules out CAD contact. A small
            # global value proves nothing about the prohibited facet location.
            if distance < threshold:
                witnesses = []
                for i, j in pairs:
                    if (surfaces[inner_ids[i]], surfaces[boundary_ids[j]]) == (bore, boundary):
                        witnesses.extend(_crossing_witnesses(
                            points[triangles[inner_ids[i]]], points[triangles[boundary_ids[j]]]))
                local_clearance = min(local_clearance, _local_cad_clearance(
                    bore, boundary, witnesses, threshold))
            clearance = min(clearance, distance)
        except Exception as exc:
            raise MesherError(f'free-standing bore/shell facets intersect ({context}); '
                              f'CAD clearance measurement failed. {advice}') from exc
    if np.isfinite(clearance):
        stats['cad_clearance_mm'] = clearance
    if local_clearance < threshold:
        raise MesherError(
            f'free-standing CAD bore and shell nearly touch or cross '
            f'(local clearance {local_clearance:.6g} mm at crossing; {len(pairs)} prohibited emitted facet pairs; '
            f'{context}). Try reducing wall_thickness_mm '
            f'(suggested {wall_mm*.5:.6g} mm), then rebuild and check CAD clearance; '
            'this is not a guaranteed remedy. Mesh resolution cannot restore CAD clearance.')

    original = triangles.copy()
    def guard(adjacent, proposed):
        return _proposal_guard(points, triangles, surfaces, physical, symmetry_axes,
                               active_shell, adjacent, proposed)
    remaining = len(pairs)
    after = pairs
    try:
        for _ in range(remaining):
            for roles in (groups, {'inner': groups['outer'], 'outer': groups['inner']}):
                active_shell = set(roles['outer'])
                repair_fitted_bore_facets(points, triangles, surfaces, roles,
                                         index_factory=CrossingIndex, proposed_guard=guard)
            after = scan()
            if not after or len(after) >= remaining:
                break
            remaining = len(after)
    except Exception as exc:
        triangles[:] = original
        raise MesherError(f'free-standing bore/shell boundary has {len(pairs)} prohibited '
                          f'facet pairs and its CAD patches do not support safe local repair '
                          f'({context}). {advice}') from exc
    if after:
        # Locate the residual failure before restoring the original connectivity.
        context, advice = _refusal_context(points, triangles, surfaces, inner_ids,
                                          boundary_ids, after, groups, wall_mm, mesh_density)
        triangles[:] = original
        gap = f'{clearance:.6g} mm' if np.isfinite(clearance) else 'unavailable'
        if clearance < threshold:
            gap += ' globally; no local interior CAD contact confirmed'
        raise MesherError(
            f'free-standing bore/shell tessellation has {len(after)} remaining '
            f'prohibited facet pairs after safe local diagonal swaps '
            f'({len(pairs)} before; CAD clearance {gap}; {context}). {advice} '
            'No vertices or elements were added.')
    stats.update(pairs_after=0,
                 changed_facets=int(np.any(triangles != original, axis=1).sum()),
                 seconds=time.monotonic() - start)
    return stats
