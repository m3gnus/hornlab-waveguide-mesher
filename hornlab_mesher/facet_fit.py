"""Local diagonal swaps for fitted freestanding bore facets.

Keep the vertices on the built surfaces and the requested element count. This
addresses independent bore/shell tessellations, not folded CAD wall offsets.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .mesh_repair import DEGENERATE_MIN_QUALITY
from .preview.intersections import _triangles_intersect


class _FacetIndex:
    def __init__(self, facets):
        self.facets = facets
        self.low, self.high = facets.min(axis=1), facets.max(axis=1)
        centers = facets.mean(axis=1)
        self.tree = cKDTree(centers)
        self.radius = np.linalg.norm(facets - centers[:, None], axis=2).max()

    def hits(self, facets):
        pairs = []
        for i, triangle in enumerate(facets):
            center = triangle.mean(axis=0)
            radius = np.linalg.norm(triangle - center, axis=1).max()
            candidates = np.asarray(self.tree.query_ball_point(
                center, radius + self.radius + 1e-8), dtype=int)
            candidates = candidates[np.all(
                (self.high[candidates] >= triangle.min(axis=0) - 1e-8)
                & (self.low[candidates] <= triangle.max(axis=0) + 1e-8), axis=1)]
            if len(candidates):
                mask = _triangles_intersect(
                    np.broadcast_to(triangle, self.facets[candidates].shape),
                    self.facets[candidates])
                pairs.extend((i, int(j)) for j in candidates[mask])
        return pairs


def repair_fitted_bore_facets(points, triangles, surface_tags, groups, *,
                              index_factory=_FacetIndex, proposed_guard=None):
    """Swap internal diagonals only when their local crossing count decreases.

    A convex quadrilateral in the actual OCC patch parameter plane, consistent
    face normals, and the existing sliver threshold protect the patch topology.
    No boundary edge, vertex, physical group or element count changes. Work is
    bounded by the initial number of crossing pairs: every swap removes at least
    one, and cannot introduce a crossing elsewhere against the shell.
    """
    import gmsh

    inner_ids = np.flatnonzero(np.isin(surface_tags, groups['inner']))
    outer_ids = np.flatnonzero(np.isin(surface_tags, groups['outer']))
    if not len(inner_ids) or not len(outer_ids):
        return
    inner = triangles[inner_ids].copy()
    shell = index_factory(points[triangles[outer_ids]])
    pairs = shell.hits(points[inner])
    # SAT includes touching contacts: a swap can occur with zero strict
    # crossings. This is harmless under the same patch/vertex/boundary guards;
    # connectivity stays unchanged when there are no detected contacts.
    if not pairs:
        return
    edges = {}
    for i, face in enumerate(inner):
        for u, v in zip(face, np.roll(face, -1)):
            edges.setdefault(tuple(sorted((int(u), int(v)))), set()).add(i)
    uv_cache = {}
    for _ in range(len(pairs)):
        bad = {i for i, _ in pairs}
        changed = False
        for edge in sorted({tuple(sorted((int(u), int(v)))) for i in bad
                            for u, v in zip(inner[i], np.roll(inner[i], -1))}):
            adjacent = sorted(edges[edge])
            if len(adjacent) != 2:
                continue
            a, b = adjacent
            tag = int(surface_tags[inner_ids[a]])
            if tag != surface_tags[inner_ids[b]]:
                continue
            u, v = edge
            first, second = inner[adjacent]
            w = next(int(k) for k in first if k not in edge)
            x = next(int(k) for k in second if k not in edge)
            if w == x or tuple(sorted((w, x))) in edges:
                continue
            for k in (u, v, w, x):
                if (tag, k) not in uv_cache:
                    uv_cache[tag, k] = np.asarray(gmsh.model.getParametrization(
                        2, tag, points[k].tolist()))
            uv = np.array([uv_cache[tag, k] for k in (u, v, w, x)])
            def side(start, end, p):
                d, q = end-start, p-start
                return d[0]*q[1] - d[1]*q[0]
            # Both diagonals must lie inside the quadrilateral.
            if (side(uv[0], uv[1], uv[2])*side(uv[0], uv[1], uv[3]) >= -1e-16
                    or side(uv[2], uv[3], uv[0])*side(uv[2], uv[3], uv[1]) >= -1e-16):
                continue
            # Inherit the direction of first's boundary edges exactly.
            j = int(np.flatnonzero(first == w)[0])
            u1, v1 = int(first[(j+1) % 3]), int(first[(j+2) % 3])
            proposed = np.array([[w, u1, x], [w, x, v1]])
            old_xyz, new_xyz = points[inner[adjacent]], points[proposed]
            old_normal = np.cross(old_xyz[:, 1]-old_xyz[:, 0], old_xyz[:, 2]-old_xyz[:, 0])
            new_normal = np.cross(new_xyz[:, 1]-new_xyz[:, 0], new_xyz[:, 2]-new_xyz[:, 0])
            longest_sq = np.square(np.linalg.norm(
                np.roll(new_xyz, -1, axis=1)-new_xyz, axis=2)).max(axis=1)
            # Same area convention and threshold as postprocessing, so a swapped
            # facet is never one that _remove_degenerate_triangles then drops.
            if (np.any(new_normal @ old_normal.sum(axis=0) <= 0)
                    or np.any(0.5*np.linalg.norm(new_normal, axis=1)
                              <= DEGENERATE_MIN_QUALITY*longest_sq)):
                continue
            old_pairs = [(i, j) for i, j in pairs if i in adjacent]
            new_pairs = shell.hits(new_xyz)
            # Fewer pairs is not enough: a swap may only keep crossings with
            # shell facets the replaced pair already crossed, never add one.
            if (len(new_pairs) >= len(old_pairs)
                    or {j for _, j in new_pairs} - {j for _, j in old_pairs}):
                continue
            if proposed_guard is not None and not proposed_guard(
                    inner_ids[adjacent], proposed):
                continue
            # Avoid passing the new diagonal through a disjoint bore facet.
            bore = index_factory(points[inner])
            conflicts = bore.hits(new_xyz)
            if any(not set(proposed[i]).intersection(inner[j]) for i, j in conflicts):
                continue
            for i in adjacent:
                for s, t in zip(inner[i], np.roll(inner[i], -1)):
                    key = tuple(sorted((int(s), int(t))))
                    edges[key].discard(i)
                    if not edges[key]:
                        del edges[key]
            inner[adjacent] = proposed
            if proposed_guard is not None:
                triangles[inner_ids[adjacent]] = proposed
            for i in adjacent:
                for s, t in zip(inner[i], np.roll(inner[i], -1)):
                    edges.setdefault(tuple(sorted((int(s), int(t)))), set()).add(i)
            pairs = [(i, j) for i, j in pairs if i not in adjacent]
            pairs.extend((adjacent[i], j) for i, j in new_pairs)
            changed = True
            break
        if not changed or not pairs:
            break
    triangles[inner_ids] = inner


def repair_fitted_wall_mesh(groups):
    """Update Gmsh's connectivity so postprocessing sees the repaired facets."""
    import gmsh

    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    points = np.asarray(coordinates).reshape(-1, 3)
    lookup = {int(tag): i for i, tag in enumerate(node_tags)}
    chunks, tags, element_tags = [], [], []
    for surface in [*groups['inner'], *groups['outer']]:
        types, elements, nodes = gmsh.model.mesh.getElements(2, surface)
        for kind, ids, ns in zip(types, elements, nodes, strict=True):
            if kind != 2:
                raise ValueError('fitted wall repair requires linear triangles')
            chunks.append(np.array([lookup[int(k)] for k in ns]).reshape(-1, 3))
            tags.extend([surface]*len(ids))
            element_tags.extend(ids)
    triangles = np.concatenate(chunks)
    original = triangles.copy()
    tags, element_tags = np.asarray(tags), np.asarray(element_tags)
    repair_fitted_bore_facets(points, triangles, tags, groups)
    changed = np.any(triangles != original, axis=1)
    for surface in np.unique(tags[changed]):
        mask = changed & (tags == surface)
        gmsh.model.mesh.removeElements(2, int(surface), element_tags[mask].tolist())
        gmsh.model.mesh.addElementsByType(int(surface), 2,
            element_tags[mask].tolist(), node_tags[triangles[mask]].ravel().tolist())
