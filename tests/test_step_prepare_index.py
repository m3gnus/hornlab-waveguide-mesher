"""Exact ordered-scan oracle for the OCC mirror broad phase."""
from pathlib import Path
from unittest.mock import patch

import gmsh
import numpy as np
import pytest

import hornlab_mesher.step_prepare as sp
from test_step_prepare import _gmsh_session, _box_surfaces


def full_scan(point, *, candidates, bboxes, tolerance):
    """Frozen pre-index algorithm, including bbox-only nearest diagnostics."""
    matched_distance = -1.0
    nearest = -1.0
    matched = []
    px, py, pz = (float(value) for value in point)
    for surface in candidates:
        bbox = bboxes.get(surface)
        if bbox is None:
            continue
        gap = float(np.linalg.norm([
            max(bbox[0] - px, px - bbox[3], 0.0),
            max(bbox[1] - py, py - bbox[4], 0.0),
            max(bbox[2] - pz, pz - bbox[5], 0.0),
        ]))
        if gap > tolerance:
            if nearest < 0.0 or gap < nearest:
                nearest = gap
            continue
        try:
            closest, parametric = gmsh.model.getClosestPoint(2, surface, [px, py, pz])
        except Exception:
            continue
        distance = float(np.linalg.norm(np.asarray(closest) - np.asarray([px, py, pz])))
        if nearest < 0.0 or distance < nearest:
            nearest = distance
        if distance > tolerance or not sp._parametric_point_on_face(surface, parametric):
            continue
        matched.append(surface)
        if matched_distance < 0.0 or distance < matched_distance:
            matched_distance = distance
    return matched_distance, nearest, matched


def assert_mirror_equivalent(samples, roles, tolerance):
    bboxes = {tag: tuple(gmsh.model.getBoundingBox(2, tag)) for tag in samples}
    candidates = sorted(samples)
    index = sp._OccMirrorIndex(candidates, bboxes, tolerance)
    for plane, axis in sp.SYMMETRY_AXIS_FOR_PLANE.items():
        points = np.concatenate(list(samples.values())).copy()
        points[:, axis] *= -1
        for point, local in zip(points, index.candidates_for_points(points)):
            assert index.closest(point, local) == full_scan(
                point, candidates=candidates, bboxes=bboxes, tolerance=tolerance
            )
        kwargs = dict(samples=samples, roles=roles, bboxes=bboxes,
                      tolerance=tolerance, span_tolerance=tolerance)
        accelerated = sp.evaluate_occ_plane_symmetry(plane, **kwargs)
        with patch.object(sp._OccMirrorIndex, 'candidates_for_points',
                          lambda self, points: [self.candidates] * len(points)):
            reference = sp.evaluate_occ_plane_symmetry(plane, **kwargs)
        assert accelerated == reference


@pytest.mark.parametrize('seed', range(4))
def test_randomized_triangle_shells_match_every_point_and_verdict(seed):
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(seed)
    # A closed, mirrored random convex shell, then a deliberately asymmetric
    # shell with the same topology. Roles include mismatches and coincident hits.
    octant = rng.uniform(0.2, 1.0, (4, 3))
    points = np.concatenate([octant * [x, y, z]
                             for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    if seed % 2:
        points[0] += [0.03, 0.02, -0.01]
    triangles = ConvexHull(points).simplices
    with _gmsh_session():
        vertices = [gmsh.model.occ.addPoint(*p) for p in points]
        edges = {}
        for triangle in triangles:
            lines = []
            for a, b in zip(triangle, np.roll(triangle, -1)):
                key = (min(a, b), max(a, b))
                if key not in edges:
                    edges[key] = gmsh.model.occ.addLine(vertices[key[0]], vertices[key[1]])
                lines.append(edges[key] if a < b else -edges[key])
            gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop(lines)])
        gmsh.model.occ.synchronize()
        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
        samples = {tag: sp.sample_occ_surface_points(tag, grid=3) for tag in surfaces}
        roles = {tag: ('source' if seed > 1 and tag % 3 == 0 else 'wall') for tag in surfaces}
        assert_mirror_equivalent(samples, roles, 0.002)


@pytest.mark.parametrize('case', ['box', 'hole', 'cylinder', 'periodic-step'])
def test_cad_fixtures_match_every_point_and_verdict(case):
    with _gmsh_session():
        if case == 'box':
            _box_surfaces(-1, -1, -1, 2, 2, 2)
        elif case == 'hole':
            outer = gmsh.model.occ.addDisk(0.1, 0, 0, 2, 2)
            inner = gmsh.model.occ.addDisk(0.4, 0, 0, 0.4, 0.4)
            gmsh.model.occ.cut([(2, outer)], [(2, inner)])
        elif case == 'cylinder':
            gmsh.model.occ.addCylinder(0, 0, -1, 0, 0, 2, 1)
        else:
            gmsh.model.occ.importShapes(str(
                Path(__file__).parent / 'fixtures/step_prepare/periodic-cylinder.step'))
        gmsh.model.occ.synchronize()
        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
        samples = {tag: sp.sample_occ_surface_points(tag, grid=5) for tag in surfaces}
        assert_mirror_equivalent(samples, dict.fromkeys(surfaces, 'wall'), 0.002)


@pytest.mark.parametrize('distance', [0.0, 0.1, np.nextafter(0.1, 0), np.nextafter(0.1, 1), 20.0])
def test_index_keeps_ties_input_order_missing_boxes_and_far_gap_diagnostics(distance):
    with _gmsh_session():
        faces = [gmsh.model.occ.addRectangle(-1, -1, 0, 2, 2) for _ in range(3)]
        gmsh.model.occ.synchronize()
        candidates = [faces[2], 99999, faces[0], faces[1]]
        bboxes = {tag: tuple(gmsh.model.getBoundingBox(2, tag)) for tag in faces}
        point = np.array([0, 0, distance])
        index = sp._OccMirrorIndex(candidates, bboxes, 0.1)
        local = index.candidates_for_points([point])[0]
        assert index.closest(point, local) == full_scan(
            point, candidates=candidates, bboxes=bboxes, tolerance=0.1)
        if distance <= 0.1:
            assert index.closest(point, local)[2] == [faces[2], faces[0], faces[1]]


def _cut_snapshot(case, indexed, tmp_path):
    with _gmsh_session():
        if case == 'box':
            _box_surfaces(-1, -1, 0, 2, 2, 3)
        elif case == 'sphere-step':
            gmsh.model.occ.addSphere(0, 0, 0, 1)
        elif case == 'cylinder-step':
            gmsh.model.occ.addCylinder(0, 0, -1, 0, 0, 2, 1)
        else:
            gmsh.model.occ.importShapes(str(
                Path(__file__).parent / 'fixtures/step_prepare/periodic-cylinder.step'))
        gmsh.model.occ.synchronize()
        if case in ('sphere-step', 'cylinder-step'):
            path = tmp_path / f'{case}.step'
            gmsh.write(str(path))
            gmsh.clear()
            gmsh.model.add('roundtrip')
            gmsh.model.occ.importShapes(str(path))
            gmsh.model.occ.synchronize()
        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
        roles = {tag: 'wall' for tag in surfaces}
        # Keep each original face in its own group to expose tag/ancestry drift.
        groups = [sp.OccSurfaceGroup(f'face-{tag}', sp.OccSurfaceSelector([tag]),
                                   sp.OccSurfaceRole(roles[tag])) for tag in surfaces]
        with patch.object(sp._OccMirrorIndex, 'candidates_for_points',
                          sp._OccMirrorIndex.candidates_for_points if indexed else
                          lambda self, points: [self.candidates] * len(points)):
            result = sp.auto_cut_occ_geometry(groups, planes=('x0', 'y0'), grid=5)
        geometry = [(g.name, g.selector.surface_tags,
                     [gmsh.model.occ.getMass(2, tag) for tag in g.selector.surface_tags])
                    for g in result.groups]
        topology = [(dim, tag, gmsh.model.getBoundary([(dim, tag)], oriented=True))
                    for dim, tag in gmsh.model.getEntities(2)]
        gmsh.option.setNumber('Mesh.MeshSizeMin', 0.5)
        gmsh.option.setNumber('Mesh.MeshSizeMax', 0.7)
        gmsh.option.setNumber('General.NumThreads', 1)
        gmsh.model.mesh.generate(2)
        mesh_counts = [(g.name, sum(len(gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)[0])
                                   for tag in g.selector.surface_tags),
                        sum(len(gmsh.model.mesh.getElementsByType(2, tag)[0])
                            for tag in g.selector.surface_tags)) for g in result.groups]
        return (result, geometry, topology, len(gmsh.model.mesh.getNodes()[0]),
                len(gmsh.model.mesh.getElementsByType(2)[0]), mesh_counts)


@pytest.mark.parametrize('case', ['box', 'sphere-step', 'cylinder-step', 'periodic-step'])
def test_cut_tags_topology_areas_and_mesh_counts_are_identical(case, tmp_path):
    assert _cut_snapshot(case, True, tmp_path) == _cut_snapshot(case, False, tmp_path)


def test_index_falls_back_when_finite_tree_distance_bounds_overflow():
    candidates = [7]
    bboxes = {7: (1e308, 0., 0., 1e308, 0., 0.)}
    point = np.array([-1e308, 0., 0.])
    index = sp._OccMirrorIndex(candidates, bboxes, 0.001)
    with np.errstate(over='ignore'):
        local = index.candidates_for_points([point])[0]
        assert index.closest(point, local) == full_scan(
            point, candidates=candidates, bboxes=bboxes, tolerance=0.001)


@pytest.mark.parametrize('scale', [1e-200, 1e-100, 1., 1e100, 1e150])
def test_index_preserves_scalar_norm_at_extreme_scales_and_tolerance_edges(monkeypatch, scale):
    rng = np.random.default_rng(42)
    low = rng.uniform(-2, 2, (64, 3)) * scale
    high = low + rng.uniform(0, 0.5, (64, 3)) * scale
    bboxes = {tag: tuple([*lo, *hi]) for tag, (lo, hi) in enumerate(zip(low, high))}
    candidates = list(reversed(bboxes))
    tolerance = 0.01 * scale
    points = np.concatenate([
        rng.uniform(-3, 3, (32, 3)) * scale,
        high + [tolerance, 0, 0],
        high + [np.nextafter(tolerance, 0), 0, 0],
        high + [np.nextafter(tolerance, np.inf), 0, 0],
    ])

    def closest(_dim, tag, point):
        # Deterministic stand-in isolates broad-phase rounding from OCC's own
        # geometric precision limits, including scalar-norm underflow.
        box = bboxes[tag]
        return np.clip(point, box[:3], box[3:]), [0., 0.]

    monkeypatch.setattr(gmsh.model, 'getClosestPoint', closest)
    monkeypatch.setattr(sp, '_parametric_point_on_face', lambda *_: True)
    index = sp._OccMirrorIndex(candidates, bboxes, tolerance)
    for point, local in zip(points, index.candidates_for_points(points)):
        assert index.closest(point, local) == full_scan(
            point, candidates=candidates, bboxes=bboxes, tolerance=tolerance)


def test_recovering_nearest_gap_does_not_retry_a_failed_occ_projection(monkeypatch):
    candidates = [1, 2]
    bboxes = {1: (10., 0., 0., 11., 1., 1.), 2: (-1., -1., 0., 1., 1., 0.)}
    point = np.zeros(3)
    calls = []

    def flaky_projection(_dim, surface, query):
        calls.append(surface)
        if len(calls) == 1:
            raise RuntimeError('transient projection failure')
        return query, [0., 0.]

    monkeypatch.setattr(gmsh.model, 'getClosestPoint', flaky_projection)
    monkeypatch.setattr(sp, '_parametric_point_on_face', lambda *_: True)
    expected = full_scan(point, candidates=candidates, bboxes=bboxes, tolerance=0.1)
    assert calls == [2]
    calls.clear()
    index = sp._OccMirrorIndex(candidates, bboxes, 0.1)
    local = index.candidates_for_points([point])[0]
    assert index.closest(point, local) == expected == (-1., 10., [])
    assert calls == [2]


@pytest.mark.parametrize('far_first', [True, False])
def test_recovering_nearest_preserves_original_nan_reduction_order(monkeypatch, far_first):
    candidates = [1, 2, 3] if far_first else [2, 3, 1]
    bboxes = {1: (10., 0., 0., 11., 1., 1.),
              2: (-1., -1., 0., 1., 1., 0.), 3: (-1., -1., 0., 1., 1., 0.)}
    point = np.zeros(3)
    calls = []

    def projection(_dim, surface, query):
        calls.append(surface)
        return ([float('nan'), 0., 0.] if surface == 2 else query), [0., 0.]

    monkeypatch.setattr(gmsh.model, 'getClosestPoint', projection)
    monkeypatch.setattr(sp, '_parametric_point_on_face', lambda *_: True)
    expected = full_scan(point, candidates=candidates, bboxes=bboxes, tolerance=0.1)
    assert calls == [2, 3]
    calls.clear()
    index = sp._OccMirrorIndex(candidates, bboxes, 0.1)
    actual = index.closest(point, index.candidates_for_points([point])[0])
    assert calls == [2, 3]
    assert np.isnan(actual[0]) and np.isnan(expected[0])
    assert actual[2] == expected[2] == [2, 3]
    if far_first:
        assert actual[1] == expected[1] == 0.
    else:
        assert np.isnan(actual[1]) and np.isnan(expected[1])
