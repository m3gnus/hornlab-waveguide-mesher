"""Sample the realized corner wall, independently of the final mesh density."""
from __future__ import annotations

import json
import math
import time

import gmsh
import numpy as np
import pytest

from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.corner_wall import build_corner_wall, occ_radius, mitred_outer_grid
from test_freeform_corner_wall import case_config, session  # noqa: F401


def accuracy_config(name):
    if name == 'sharp-morph':
        from test_freeform_corner_qual import morph_config
        config = morph_config('bare')
    elif name == 'five-anchor':
        config = case_config('a')
        plane = {'points': [{'z': z, 'r': r, 'join': 'corner'}
                            for z, r in [(0, 12.7), (30, 28), (70, 60), (110, 110), (150, 170)]]}
        config['profile']['profileH'] = plane
        config['profile']['profileV'] = plane
    elif name == 'endpoint':
        config = case_config('a')
        for plane in ('profileH', 'profileV'):
            config['profile'][plane]['points'].pop(1)
    elif name == 'superellipse':
        config = case_config('a')
        config['profile']['crossSections'] = [
            {'t': 0, 'shape': 'circle'}, {'t': 1, 'shape': 'superellipse', 'exponent': 4}]
    elif name == 'morph':
        config = case_config('a')
        config['morph'] = {'morphTarget': 1, 'morphWidth': 140, 'morphHeight': 110,
                           'morphCorner': 10, 'morphFixed': .2, 'morphRate': 3}
    else:
        case, mode = {'circle': ('a', 'bare'), 'rounded-rectangle': ('c', 'bare'),
                      'quadratic': ('e', 'bare'), 'enclosure': ('c', 'enclosure'),
                      'baffle': ('dmouth', 'infinite-baffle'), 'throat': ('dthroat', 'bare'),
                      'outer-wall': ('b', 'freestanding')}[name]
        config = case_config(case, mode)
    # Deliberately coarse final elements: geometry accuracy has its own budget.
    config['mesh'].update(throat_res_mm=15, mouth_res_mm=25)
    return config


ACCURACY_CASES = ['sharp-morph', 'five-anchor', 'endpoint', 'circle', 'superellipse',
                  'rounded-rectangle', 'quadratic', 'morph', 'enclosure', 'baffle', 'throat', 'outer-wall']


def sample_wall_accuracy(geometry, built, *, points=None, recipe=None, outer_reference=None):
    points = built.freeform_wall_points_mm if points is None else points
    recipe = geometry.freeform_wall if recipe is None else recipe
    tags = built.mesh_surface_groups['inner']
    uv_max = 0.
    for tag in tags:
        lo, hi = map(np.asarray, gmsh.model.getParametrizationBounds(2, tag))
        u, v = np.meshgrid(np.linspace(lo[0], hi[0], 129), np.linspace(lo[1], hi[1], 65))
        xyz = np.asarray(gmsh.model.getValue(2, tag, np.c_[u.ravel(), v.ravel()].ravel())).reshape(65, 129, 3)
        for row in xyz:
            phi = np.arctan2(row[:, 1], row[:, 0])
            expected = recipe.radius(phi, float(row[0, 2]), points)
            uv_max = max(uv_max, float(np.max(abs(np.linalg.norm(row[:, :2], axis=1) - expected))))
    if getattr(built, 'freeform_wall_segments', None) is not None:
        from hornlab_mesher.corner_wall import _occ_probe
        probe = _occ_probe(points, recipe, [(2, tag) for tag in tags],
                           built.freeform_wall_segments, built.freeform_wall_groups)
    else:
        probe = lambda phi, z, expected: occ_radius(tags, phi, z, expected)
    sections = {}
    for label, phi in [('H', 0.), ('V', math.pi / 2), ('D', math.pi / 4),
                       ('D135', 3 * math.pi / 4), ('D225', 5 * math.pi / 4),
                       ('D315', 7 * math.pi / 4)]:
        errors = []
        for z in np.unique(np.r_[np.linspace(points[0, 0, 2], points[0, -1, 2], 121), recipe.cuts(points)]):
            expected = float(recipe.radius(phi, float(z), points))
            errors.append(abs(probe(phi, float(z), expected) - expected))
        sections[label] = max(errors)
    errors = {'UV': uv_max, **sections}
    if 'outer' in built.mesh_surface_groups:
        # This fixture has circular, piecewise straight meridians: its mitred
        # offset is exactly linear between the shared offset corner rings.
        outer = (mitred_outer_grid(points, geometry.wall_thickness_mm, full_circle=geometry.closed,
                                  corner_z=recipe.cuts(points))
                 if outer_reference is None else outer_reference)
        outer_error = 0.
        for tag in built.mesh_surface_groups['outer']:
            lo, hi = map(np.asarray, gmsh.model.getParametrizationBounds(2, tag))
            u, v = np.meshgrid(np.linspace(lo[0], hi[0], 129), np.linspace(lo[1], hi[1], 65))
            xyz = np.asarray(gmsh.model.getValue(2, tag, np.c_[u.ravel(), v.ravel()].ravel())).reshape(-1, 3)
            expected = np.interp(xyz[:, 2], outer[0, :, 2], np.linalg.norm(outer[0, :, :2], axis=1))
            outer_error = max(outer_error, float(np.max(abs(np.linalg.norm(xyz[:, :2], axis=1) - expected))))
        errors['outerUV'] = outer_error
    return errors


@pytest.mark.parametrize('name', ACCURACY_CASES)
def test_corner_wall_sampled_accuracy(session, tmp_path, name):
    geometry = resolve_geometry(accuracy_config(name)).geometry
    built = build_corner_wall(geometry)
    errors = sample_wall_accuracy(geometry, built)
    (tmp_path / f'{name}.json').write_text(json.dumps(errors, indent=2))
    assert max(errors.values()) <= .01, errors


def many_corner_config(mode, n=64):
    plane = {'points': [{'z': float(z), 'r': float(12.7 + (150-12.7)*z/600), 'join': 'corner'}
                        for z in np.linspace(0, 600, n)]}
    config = {'formula': 'FREEFORM', 'mode': mode,
              'profile': {'profileH': plane, 'profileV': plane,
                          'crossSections': [{'t': 0, 'shape': 'circle'}, {'t': 1, 'shape': 'circle'}]},
              'mesh': {'allow_large_mesh': True}}
    if mode == 'enclosure':
        config['enclosure'] = {'depth_mm': 650, 'edge_mm': 0}
    return config


def _timed_many_corner_build(config, tmp_path, mode, label):
    from hornlab_mesher import build_from_config
    started = time.perf_counter()
    result = build_from_config(config, tmp_path / f'{label}-{mode}.msh')
    elapsed = time.perf_counter() - started
    assert len(result.metadata['freeformReport']['cornerWall']['corners']) == 62
    (tmp_path / f'{label}-{mode}-timing.json').write_text(json.dumps({'elapsed_s': elapsed}))
    assert elapsed < 120, elapsed
    return result


@pytest.mark.parametrize('mode', ['bare', 'enclosure'])
def test_64_corner_public_build_time(tmp_path, mode):
    _timed_many_corner_build(many_corner_config(mode), tmp_path, mode, 'straight')


@pytest.mark.parametrize('mode', ['bare', 'enclosure'])
def test_64_bend_public_build_time(tmp_path, mode):
    config = many_corner_config(mode)
    # H and V share this circular meridian. Give every interior anchor a
    # distinct slope, keeping the expansion monotone and the endpoints exact.
    for index, point in enumerate(config['profile']['profileH']['points'][1:-1]):
        point['r'] += .2 * (-1) ** index
    result = _timed_many_corner_build(config, tmp_path, mode, 'bends')
    records = result.metadata['freeformReport']['cornerWall']['corners']
    assert all(abs(record['azimuths'][0]['occJumpDeg']) > 1 for record in records)


@pytest.mark.parametrize('case', ['b', 'c', 'e'])
@pytest.mark.parametrize('quadrants', ['1234', '1', '12', '14'])
def test_indexed_occ_checks_keep_values_and_cache_bounds(session, monkeypatch, quadrants, case):
    import hornlab_mesher.corner_wall as cw
    config = case_config(case)
    config['mesh']['quadrants'] = quadrants
    geometry = resolve_geometry(config).geometry
    built = build_corner_wall(geometry)
    points, recipe = built.freeform_wall_points_mm, geometry.freeform_wall
    wall = [(2, tag) for tag in built.mesh_surface_groups['inner']]
    reference = cw.measure_occ(points, recipe, wall)
    bounds = gmsh.model.getBoundingBox
    queried = []

    def counted(dim, tag):
        queried.append((dim, tag))
        return bounds(dim, tag)

    radius = cw.occ_radius
    candidates = []
    def checked(tags, *args, **kwargs):
        candidates.append(len(tags))
        assert kwargs['boxes']
        return radius(tags, *args, **kwargs)

    monkeypatch.setattr(gmsh.model, 'getBoundingBox', counted)
    monkeypatch.setattr(cw, 'occ_radius', checked)
    indexed = cw.measure_occ(points, recipe, wall, segments=built.freeform_wall_segments,
                             groups=built.freeform_wall_groups)
    assert indexed == reference
    assert sorted(queried) == sorted(wall)
    # A one-sided slope query touches one span and at most two sectors.
    assert max(candidates) <= 2


@pytest.mark.parametrize('count', [2, 3, 4, 17, 65])
def test_angular_fit_preserves_interpolation(count):
    from hornlab_mesher.corner_wall import _angular_poles
    from hornlab_mesher.builders._occ import sample_patch_boundary
    phi = np.linspace(0, math.pi / 2, count)
    points = np.array([np.c_[(10+z) * np.cos(phi), (10+z) * np.sin(phi), np.full(count, z)]
                       for z in (0., 2., 4.)]).transpose(1, 0, 2)
    groups = [list(range(count))]
    poles = _angular_poles(points, groups, None)
    actual = sample_patch_boundary(poles, groups[0], np.linspace(0, 1, count), surface_fit='approximate')
    # OCC knot serialization rounds normalized knots to twelve decimals.
    np.testing.assert_allclose(actual, points[:, -1], rtol=0, atol=2e-11)
    dense = sample_patch_boundary(poles, groups[0], np.linspace(0, 1, 1025), surface_fit='approximate')
    assert np.max(np.linalg.norm(dense[:, :2], axis=1)) <= 14. + 1e-10


def test_noncircular_outer_wall_and_rear_share_angular_basis(session):
    from test_freeform_corner_wall import mesh, raw_geometry
    geometry = raw_geometry(case_config('e', 'freestanding'))
    built = build_corner_wall(geometry)
    _, free = mesh(built)
    assert len(free) == 0


@pytest.mark.parametrize('coupled_z', [False, True])
def test_radial_intersection_verifies_isocurve_and_keeps_general_fallback(session, monkeypatch, coupled_z):
    import hornlab_mesher.corner_wall as cw
    from hornlab_mesher.builders._occ import add_bspline_patch
    points = np.array([[[10., 0., 0.], [10., 0., 20.]],
                       [[0., 10., 2. if coupled_z else 0.],
                        [0., 10., 22. if coupled_z else 20.]]])
    tag = add_bspline_patch(points, [0, 1], degree_v=1)
    gmsh.model.occ.synchronize()
    solver = cw.least_squares
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return solver(*args, **kwargs)
    monkeypatch.setattr(cw, 'least_squares', counted)
    phi = .6
    expected = 10. / (math.cos(phi) + math.sin(phi))
    assert cw.occ_radius([tag], phi, 9., expected) == pytest.approx(expected, abs=1e-8)
    assert bool(calls) is coupled_z


def test_corner_patch_discards_construction_points(session):
    from hornlab_mesher.corner_wall import _angular_poles
    from hornlab_mesher.builders._occ import add_bspline_patch
    phi = np.linspace(0, math.pi / 2, 33)
    points = np.array([np.c_[(10+z) * np.cos(phi), (10+z) * np.sin(phi), np.full(len(phi), z)]
                       for z in (0., 2., 4., 6.)]).transpose(1, 0, 2)
    groups = [list(range(len(phi)))]
    poles = _angular_poles(points, groups, None)
    tag = add_bspline_patch(poles, groups[0], degree_v=3)
    gmsh.model.occ.synchronize()
    assert len(gmsh.model.getEntities(0)) == 4
    xyz = np.asarray(gmsh.model.getValue(2, tag, [.5, 0., .5, 1.])).reshape(-1, 3)
    np.testing.assert_allclose(np.linalg.norm(xyz[:, :2], axis=1), [10., 16.], rtol=0, atol=1e-10)
    np.testing.assert_allclose(xyz[:, 2], [0., 6.], rtol=0, atol=1e-12)
