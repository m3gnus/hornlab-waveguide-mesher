"""Tessellate the authored ATH mesh patches without resampling their rings."""
from __future__ import annotations

from contextlib import contextmanager
import math
import time
import uuid

import numpy as np

from ..builders.point_grid_freestanding import _build_ath_import_surfaces
from ..config_builder import resolve_geometry
from ..mesher import _GMSH_LOCK, _PINNED_MESH_OPTIONS
from ..native_env import preserve_native_windows_path
from .contract import (PreviewGeometryV1, PreviewSurfaceV1, _API_VERSION,
                       _METADATA_VERSION, _orient_indices_to_normals, _classify_triangle_orientation)
from .dimensions import DIMENSIONS_SAMPLING, dimension_metadata
from .fidelity import _point_triangle_squared

# Mid-edges, centroid and interior quarter stations measure the actual emitted
# triangles against the same parametric patches the acoustic mesh uses.
_MAX_REFINEMENT_ITERATIONS = 64

_PROBES = np.array([[.5, .5, 0], [0, .5, .5], [.5, 0, .5],
                    [1/3, 1/3, 1/3], [.5, .25, .25],
                    [.25, .5, .25], [.25, .25, .5]])


@contextmanager
def _model():
    """Own one model and restore a caller's model/options, including on failure."""
    import gmsh
    options = {**_PINNED_MESH_OPTIONS, 'General.Terminal': 0,
               'General.NumThreads': 1, 'Mesh.MeshSizeMin': 0,
               'Mesh.MeshSizeMax': 20, 'Mesh.MeshSizeFromPoints': 0,
               'Mesh.MeshSizeFromCurvature': 0, 'Mesh.MeshSizeExtendFromBoundary': 0,
               'Mesh.Algorithm': 6}
    with _GMSH_LOCK:
        opened = not gmsh.isInitialized()
        saved, previous, owned = {}, None, None
        try:
            if opened:
                with preserve_native_windows_path():
                    gmsh.initialize(interruptible=False)
            else:
                previous = gmsh.model.getCurrent()
                saved = {name: gmsh.option.getNumber(name) for name in options}
            for name, value in options.items():
                gmsh.option.setNumber(name, value)
            owned = 'HornLabPreview-' + uuid.uuid4().hex
            gmsh.model.add(owned)
            yield gmsh
        finally:
            if opened:
                gmsh.finalize()
            else:
                if owned is not None:
                    gmsh.model.setCurrent(owned)
                    gmsh.model.remove()
                if previous:
                    gmsh.model.setCurrent(previous)
                for name, value in saved.items():
                    gmsh.option.setNumber(name, value)


def _roles(built, geometry):
    groups = built.mesh_surface_groups
    outer = groups['outer']
    stride = 0 if geometry.ath_rear_points is None else geometry.ath_rear_points.shape[1] - 1
    return {**{tag: 'horn.inner' for tag in groups['inner']},
            **{tag: ('wall.rear_return' if stride and i % stride == 0 else 'horn.outer')
               for i, tag in enumerate(outer)},
            **{tag: 'wall.rear_cap' for tag in groups['rear_cap']},
            **{tag: 'mouth_rim' for tag in groups['mouth']},
            **{tag: 'source_cap' for tag in groups['throat_disc']}}


def _initial(gmsh, roles):
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    points = {int(tag): p for tag, p in zip(tags, np.asarray(xyz).reshape(-1, 3))}
    patches = []
    for tag, role in roles.items():
        ids, _, uv = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)
        lookup = {int(node): i for i, node in enumerate(ids)}
        types, _, nodes = gmsh.model.mesh.getElements(2, tag)
        triangles = [np.array([lookup[int(node)] for node in block]).reshape(-1, 3)
                     for kind, block in zip(types, nodes) if kind == 2]
        if triangles:
            patches.append({'tag': tag, 'role': role, 'ids': list(map(int, ids)),
                            'uv': np.asarray(uv).reshape(-1, 2),
                            'tri': np.concatenate(triangles),
                            'crease_curves': [abs(curve) for _, curve in gmsh.model.getBoundary([(2, tag)])]})
    curves = {}
    for _, tag in gmsh.model.getEntities(1):
        ids, _, param = gmsh.model.mesh.getNodes(1, tag, includeBoundary=True)
        lookup = dict(zip(map(int, ids), param))
        types, _, nodes = gmsh.model.mesh.getElements(1, tag)
        for kind, block in zip(types, nodes):
            if kind == 1:
                for a, b in np.asarray(block).reshape(-1, 2):
                    a, b = sorted((int(a), int(b)))
                    curves[a, b] = (tag, float(lookup[a]), float(lookup[b]))
    return points, patches, curves


def _normals(gmsh, tag, uv):
    normals = np.asarray(gmsh.model.getNormal(tag, np.asarray(uv).reshape(-1).tolist())).reshape(-1, 3)
    lengths = np.linalg.norm(normals, axis=1)
    if not np.all(np.isfinite(normals)) or np.any(lengths < 1e-12):
        raise ValueError('ATH preview patch has undefined parametric normals')
    return normals / lengths[:, None]


def _measure(gmsh, patch, points):
    tri, uv = patch['tri'], patch['uv']
    xyz = np.array([points[node] for node in patch['ids']])
    low, high = (np.asarray(v) for v in gmsh.model.getParametrizationBounds(2, patch['tag']))
    vertex_normals = patch.get('normal_cache', np.empty((0, 3)))
    if len(vertex_normals) < len(uv):
        query_uv = np.clip(uv[len(vertex_normals):], low + (high-low)*1e-6, high - (high-low)*1e-6)
        vertex_normals = np.concatenate((vertex_normals, _normals(gmsh, patch['tag'], query_uv)))
        patch['normal_cache'] = vertex_normals
    # Stable local triangle identities, independent of the growing node count.
    keys = tri.astype("<i8", copy=False).view("S24").ravel().tolist()
    cache = patch.setdefault('error_cache', {})
    errors = np.array([cache.get(key, np.nan) for key in keys])
    fresh = np.isnan(errors)
    if fresh.any():
        query = np.einsum('pk,tkd->tpd', _PROBES, uv[tri[fresh]])
        reference = np.asarray(gmsh.model.getValue(2, patch['tag'], query.reshape(-1).tolist())).reshape(-1, len(_PROBES), 3)
        a, b, c = (np.broadcast_to(xyz[tri[fresh, i]][:, None], reference.shape) for i in range(3))
        errors[fresh] = np.sqrt(np.maximum(0, _point_triangle_squared(reference, a, b, c))).max(axis=1)
        cache.update((keys[i], float(errors[i])) for i in np.flatnonzero(fresh))
    normals = vertex_normals[tri]
    steps = np.zeros(len(tri))
    edge_steps = []
    for i in range(3):
        cosine = np.sum(normals[:, i] * normals[:, (i + 1) % 3], axis=1)
        edge_steps.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        steps = np.maximum(steps, edge_steps[-1])
    patch["edge_steps"] = np.stack(edge_steps, axis=1)
    cross = np.cross(xyz[tri[:, 1]] - xyz[tri[:, 0]], xyz[tri[:, 2]] - xyz[tri[:, 0]])
    dot = np.sum(cross * normals.sum(axis=1), axis=1)
    sign = 1 if np.median(dot) >= 0 else -1
    _, bad_orientation = _classify_triangle_orientation(xyz, tri, vertex_normals * sign)
    patch["orientation_bad"] = bad_orientation
    return xyz, vertex_normals, errors, steps


def _curvature(gmsh, patch, normals):
    """Differentiate the actual patch, using the preview curvature convention.

    Endpoint limits are evaluated just inside the parameter domain; singular
    metric determinants return zero, as in analytic_grid_curvature.
    """
    low, high = (np.asarray(v) for v in gmsh.model.getParametrizationBounds(2, patch['tag']))
    h = (high - low) * 1e-4
    uv = np.clip(patch['uv'], low + 4*h, high - 4*h)
    if gmsh.model.getType(2, patch['tag']) == 'Surface' and len(gmsh.model.getBoundary([(2, patch['tag'])])) == 3:
        # Built-in triangular fillings use 0 <= v <= u <= 1.
        uv[:, 1] = np.minimum(uv[:, 1], uv[:, 0] - 3*h[0])
        uv[:, 0] = np.maximum(uv[:, 0], uv[:, 1] + 3*h[0])
    def value(du, dv):
        query = uv + np.array([du*h[0], dv*h[1]])
        return np.asarray(gmsh.model.getValue(2, patch['tag'], query.reshape(-1).tolist())).reshape(-1, 3)
    p = value(0, 0)
    pu, mu, pv, mv = value(1, 0), value(-1, 0), value(0, 1), value(0, -1)
    u, v = (pu-mu)/(2*h[0]), (pv-mv)/(2*h[1])
    uu, vv = (pu-2*p+mu)/h[0]**2, (pv-2*p+mv)/h[1]**2
    uv2 = (value(1,1)-value(1,-1)-value(-1,1)+value(-1,-1))/(4*h[0]*h[1])
    e, f, g = np.sum(u*u, axis=1), np.sum(u*v, axis=1), np.sum(v*v, axis=1)
    l, m, n = (np.sum(vector*normals, axis=1) for vector in (uu, uv2, vv))
    determinant = e*g-f*f
    valid = determinant > 64*np.finfo(float).eps*np.maximum(e*g, f*f)
    mean, gaussian = np.zeros(len(uv)), np.zeros(len(uv))
    mean[valid] = (g[valid]*l[valid]-2*f[valid]*m[valid]+e[valid]*n[valid])/(2*determinant[valid])
    gaussian[valid] = (l[valid]*n[valid]-m[valid]**2)/determinant[valid]
    root = np.sqrt(np.maximum(0, mean**2-gaussian))
    first, second = mean+root, mean-root
    principal = np.where(np.abs(first) >= np.abs(second), first, second)
    return np.nan_to_num(mean, nan=0, posinf=0, neginf=0), np.nan_to_num(principal, nan=0, posinf=0, neginf=0)


def _refine(gmsh, points, patches, curves, marked, *, faceted=False):
    """Conforming red/green refinement; query new nodes in batches."""
    midpoints = {}
    next_node = max(points) + 1
    radix = next_node
    marked_keys = np.array([a*radix+b for a, b in marked], dtype=np.int64)
    for patch in patches:
        ids, uv, tri = patch['ids'], patch['uv'], patch['tri']
        local_edges = np.stack((tri, np.roll(tri, -1, axis=1)), axis=-1)
        global_edges = np.sort(np.asarray(ids)[local_edges], axis=-1).reshape(-1, 2)
        packed = global_edges[:, 0]*radix + global_edges[:, 1]
        selected = np.flatnonzero(np.isin(packed, marked_keys))
        if not len(selected):
            continue
        patch.pop('measured', None)
        _, first, inverse = np.unique(packed[selected], return_index=True, return_inverse=True)
        local = local_edges.reshape(-1, 2)[selected[first]]
        edges = global_edges[selected[first]]
        values = uv[local].mean(axis=1)
        added_nodes, pending_surface, pending_curves = [], [], {}
        for edge, value in zip(edges, values):
            edge = tuple(edge)
            node = midpoints.get(edge)
            if node is None:
                node = next_node
                next_node += 1
                midpoints[edge] = node
                if edge in curves:
                    curve, t0, t1 = curves.pop(edge)
                    tm = .5 * (t0 + t1)
                    pending_curves.setdefault(curve, []).append((node, tm))
                    curves[tuple(sorted((edge[0], node)))] = (curve, t0, tm)
                    curves[tuple(sorted((edge[1], node)))] = (curve, t1, tm)
                else:
                    pending_surface.append((node, value))
            added_nodes.append(node)
        if pending_surface:
            new_xyz = np.asarray(gmsh.model.getValue(2, patch['tag'],
                np.asarray([v for _, v in pending_surface]).reshape(-1).tolist())).reshape(-1, 3)
            points.update((node, xyz) for (node, _), xyz in zip(pending_surface, new_xyz))
        for curve, entries in pending_curves.items():
            new_xyz = np.asarray(gmsh.model.getValue(1, curve, [t for _, t in entries])).reshape(-1, 3)
            points.update((node, xyz) for (node, _), xyz in zip(entries, new_xyz))
        if faceted:
            values = np.asarray(gmsh.model.getParametrization(2, patch['tag'],
                np.asarray([points[node] for node in added_nodes]).reshape(-1).tolist())).reshape(-1, 2)
        # Each edge adds exactly one new vertex to this patch, even at seams.
        mids = np.full(tri.size, -1, dtype=int)
        mids[selected] = len(ids) + inverse
        patch['ids'].extend(added_nodes)
        patch['uv'] = np.concatenate((uv, values))
        mids = mids.reshape(-1, 3)
        count = (mids >= 0).sum(axis=1)
        pieces = [tri[count == 0]]
        t, m = tri[count == 3], mids[count == 3]
        if len(t):
            a, b, c = t.T
            x, y, z = m.T
            pieces.extend(np.stack(v, axis=1) for v in ((a,x,z),(x,b,y),(z,y,c),(x,y,z)))
        for cuts in (1, 2):
            t, m = tri[count == cuts], mids[count == cuts]
            if not len(t):
                continue
            k = np.argmax(m >= 0 if cuts == 1 else m < 0, axis=1)
            if cuts == 2:
                k = (k + 1) % 3
            order = (k[:, None] + np.arange(3)) % 3
            aa, bb, cc = np.take_along_axis(t, order, axis=1).T
            x, y, _ = np.take_along_axis(m, order, axis=1).T
            if cuts == 1:
                pieces.extend(np.stack(v, axis=1) for v in ((aa,x,cc),(x,bb,cc)))
            else:
                positions = np.asarray([points[node] for node in patch['ids']])
                use_alternate = (np.linalg.norm(positions[aa]-positions[y], axis=1)
                                 < np.linalg.norm(positions[x]-positions[cc], axis=1))
                pieces.append(np.stack((bb,y,x), axis=1))
                for mask, variants in ((~use_alternate, ((aa,x,cc),(x,y,cc))),
                                       (use_alternate, ((aa,x,y),(aa,y,cc)))):
                    pieces.extend(np.stack(v, axis=1)[mask] for v in variants)
        patch['tri'] = np.concatenate(pieces)


def build(config, options):
    from .api import _LOD_PRESETS, _fidelity_record
    lod = str(options.lod).strip().lower()
    if lod not in _LOD_PRESETS:
        raise ValueError("lod must be 'coarse', 'fine', or 'inspection'")
    preset = _LOD_PRESETS[lod]
    chord = float(preset['chord'] if options.max_chord_error_mm is None else options.max_chord_error_mm)
    normal = float(preset['normal'] if options.max_normal_step_deg is None else options.max_normal_step_deg)
    try:
        silhouette = int(preset['silhouette'] if options.min_silhouette_segments is None else options.min_silhouette_segments)
        cap = 200000 if options.max_vertices is None else int(options.max_vertices)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('silhouette and vertex limits must be finite integers') from exc
    if not math.isfinite(chord) or chord <= 0:
        raise ValueError('max_chord_error_mm must be finite and > 0')
    if not math.isfinite(normal) or not 0 < normal <= 180:
        raise ValueError('max_normal_step_deg must be finite and in (0, 180]')
    if silhouette < 3 or cap < 8:
        raise ValueError('silhouette must be >= 3 and max_vertices must be >= 8')
    start = time.perf_counter()
    resolved = resolve_geometry(config)
    geometry = resolved.geometry
    surfaces, fidelity = [], {}
    with _model() as gmsh:
        built = _build_ath_import_surfaces(geometry)
        roles = _roles(built, geometry)
        extent = np.ptp(geometry.inner_points.reshape(-1, 3), axis=0).max()
        gmsh.option.setNumber('Mesh.MeshSizeMax', max(.1, float(extent) / 24))
        # Seed each ring loft as a strip, avoiding unstructured skinny faces.
        # Patch seams remain authored crease features, with separate vertex
        # normals on each side rather than a smoothness target across the seam.
        ring_nodes = {'coarse': 17, 'fine': 33, 'inspection': 65}[lod]
        for _, curve in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(curve, 3 if gmsh.model.getType(1, curve) == 'Line' else ring_nodes)
        for tag in roles:
            if len(gmsh.model.getBoundary([(2, tag)])) == 4:
                gmsh.model.mesh.setTransfiniteSurface(tag)
        mouth_curves = built.metadata['athMouthCurveTags']
        angular_span = math.tau if geometry.closed else (len(str(resolved.quadrants)) * math.pi / 2)
        mouth_intervals = max(2, math.ceil(silhouette * angular_span / math.tau / len(mouth_curves)))
        for curve in mouth_curves:
            gmsh.model.mesh.setTransfiniteCurve(curve, max(mouth_intervals + 1, ring_nodes))
        # A shared 1D tessellation makes caps and every loft close exactly.
        gmsh.model.mesh.generate(2)
        points, patches, curves = _initial(gmsh, roles)
        if len(points) > cap:
            raise ValueError(f'max_vertices={cap} cannot represent the authored ATH patch topology')
        limited = False
        limit_reason = None
        edge_floor = max(1e-8, float(extent)*1e-10)
        for iteration in range(_MAX_REFINEMENT_ITERATIONS):
            marked = set()
            for patch in patches:
                if 'measured' not in patch:
                    patch['measured'] = _measure(gmsh, patch, points)
            for patch in patches:
                xyz, normals, errors, steps = patch['measured']
                ids = np.asarray(patch['ids'])[patch['tri']]
                edges = np.stack((ids, np.roll(ids, -1, axis=1)), axis=-1)
                split = np.broadcast_to((errors > chord)[:, None] | patch['orientation_bad'][:, None], (len(ids), 3)).copy()
                # Fine strips use complete cuts to preserve triangle quality;
                # coarse strips can use selective edge cuts at their wider target.
                split |= patch['edge_steps'] > normal if lod == 'coarse' else (steps > normal)[:, None]
                marked.update(map(tuple, np.sort(edges[split], axis=1)))
            if not marked:
                break
            eligible = {edge for edge in marked if np.linalg.norm(points[edge[0]]-points[edge[1]]) > edge_floor}
            if not eligible:
                limited, limit_reason = True, 'edge-size'
                break
            marked = eligible
            if iteration == _MAX_REFINEMENT_ITERATIONS-1 or len(points) + len(marked) > cap:
                limited = True
                limit_reason = 'iterations' if iteration == _MAX_REFINEMENT_ITERATIONS-1 else 'vertices'
                break
            previous_count = len(points)
            _refine(gmsh, points, patches, curves, marked,
                    faceted=geometry.topology_mode == 'legacy' and (geometry.preserve_grid or not geometry.wg_topology))
            if len(points) == previous_count:
                limited, limit_reason = True, 'no-progress'
                break
        # A no-progress callback can have invalidated measurement caches.
        for patch in patches:
            if 'measured' not in patch:
                patch['measured'] = _measure(gmsh, patch, points)
        by_role = {}
        for patch in patches:
            role = patch['role']
            visible = (options.include_inner if role == 'horn.inner' else
                       options.include_source_cap if role == 'source_cap' else
                       options.include_rear_cap and options.include_outer if role == 'wall.rear_cap' else
                       options.include_outer)
            if not visible:
                continue
            xyz, normals, errors, steps = patch['measured']
            # The fan curve loop faces +z; the material rear faces -z.
            if role == 'wall.rear_cap':
                normals = -normals
            # OCC may choose either orientation when filling a faceted loop.
            if role in ('horn.inner', 'horn.outer', 'wall.rear_return'):
                dot = np.sum(normals[:, :2] * xyz[:, :2], axis=1)
                if (np.median(dot) > 0) == (role == 'horn.inner'):
                    normals = -normals
            if role == 'source_cap' and np.median(normals[:, 2]) < 0:
                normals = -normals
            oriented = _orient_indices_to_normals(role, xyz, patch['tri'].astype(np.uint32).reshape(-1), normals,
                wind_folds_individually=True)
            record = {'max_chord_error_mm': float(errors.max()), 'max_normal_step_deg': float(steps.max()),
                      'measurement_complete': True, 'vertex_cap_limited': limit_reason == 'vertices',
                      'iteration_bound_limited': limit_reason == 'iterations',
                      'refinement_stalled': limit_reason in ('edge-size', 'no-progress'),
                      'minimum_edge_length_mm': edge_floor,
                      'probe_count_per_triangle': len(_PROBES),
                      'refinement_limit': limit_reason,
                      'foldedTriangles': oriented.folded_triangles,
                      'foldedTriangleIndices': list(oriented.folded_triangle_indices)}
            curvature = _curvature(gmsh, patch, normals) if options.include_curvature else (None, None)
            by_role.setdefault(role, []).append((xyz, normals, oriented.indices, record, curvature, patch['tag'], patch['crease_curves']))
        for role, pieces in by_role.items():
            offsets = np.cumsum([0] + [len(p[0]) for p in pieces[:-1]])
            xyz = np.concatenate([p[0] for p in pieces])
            xyz[:, 1] += geometry.vertical_offset_mm
            normals = np.concatenate([p[1] for p in pieces])
            indices = np.concatenate([p[2] + offset for p, offset in zip(pieces, offsets)])
            fidelity[role] = {**pieces[0][3],
                'max_chord_error_mm': max(p[3]['max_chord_error_mm'] for p in pieces),
                'max_normal_step_deg': max(p[3]['max_normal_step_deg'] for p in pieces)}
            triangle_offsets = np.cumsum([0] + [len(p[2])//3 for p in pieces[:-1]])
            folds = [int(i + offset) for p, offset in zip(pieces, triangle_offsets)
                     for i in p[3]['foldedTriangleIndices']]
            final_orientation = _orient_indices_to_normals(role, xyz, indices, normals, wind_folds_individually=True)
            indices = final_orientation.indices
            folds = sorted(set(folds) | set(final_orientation.folded_triangle_indices))
            fidelity[role] = {**fidelity[role], **_fidelity_record(fidelity[role],
                chord_target=chord, normal_target=normal, silhouette_target=silhouette)}
            fidelity[role]['silhouette_segments_achieved'] = mouth_intervals * len(mouth_curves)
            if len(xyz) > cap:
                raise ValueError(f'max_vertices={cap} cannot represent {role}')
            surfaces.append(PreviewSurfaceV1(role, xyz, indices, normals, 'smooth',
                'analytic-parametric', geometry.closed,
                curvature_mean=np.concatenate([p[4][0] for p in pieces]) if options.include_curvature else None,
                curvature_principal=np.concatenate([p[4][1] for p in pieces]) if options.include_curvature else None,
                metadata={'samplingAuthority': 'authored-mesh-patches', 'fidelity': fidelity[role],
                          'foldedTriangles': len(folds), 'foldedTriangleIndices': folds,
                          'creaseFeatures': [{'surface': p[5], 'boundaryCurves': p[6]} for p in pieces],
                          'authoredPatchSpans': [{'surface': p[5], 'start': int(offset), 'count': len(p[0])}
                                                for p, offset in zip(pieces, offsets)]}))
    warnings = [f'ATH preview refinement stopped at {limit_reason}; inspect achieved fidelity'] if limited else []
    folded = sum(surface.metadata['foldedTriangles'] for surface in surfaces)
    if folded:
        warnings.append(f'Authored ATH patches carry {folded} folded triangles; winding follows the local parametric normals')
    return PreviewGeometryV1(surfaces, {
        'api_version': _API_VERSION, 'metadata_version': _METADATA_VERSION,
        'units': 'mm', 'coordinate_frame': 'mesher-xyz', 'lod': lod,
        'formula': resolved.formula, 'mode': resolved.mode,
        'sampling_authority': 'authored-mesh-patches',
        'requested_fidelity': {'max_chord_error_mm': chord, 'max_normal_step_deg': normal,
                               'min_silhouette_segments': silhouette, 'max_vertices': cap},
        'fidelity': fidelity, 'warnings': warnings,
        'surface_metadata': {surface.role: surface.metadata for surface in surfaces},
        'actual_segment_counts': {'horn_phi': mouth_intervals * len(mouth_curves),
                                  'horn_axial': geometry.inner_points.shape[1] - 1},
        'dimensions_sampling': dict(DIMENSIONS_SAMPLING), **dimension_metadata(config, lod),
        'timings_ms': {'total': (time.perf_counter() - start) * 1000},
    })
