"""Emitted free-standing crossings are never published at user density."""
from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.mesher import MesherError, build_mesh_with_info
from hornlab_mesher.shell_facets import BoundaryIndex
from hornlab_mesher.preview.intersections import _triangles_intersect


def test_broad_phase_keeps_crossings_contacts_and_coplanar_overlap():
    a = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                  [[50, 0, 0], [51, 0, 0], [50, 1, 0]]], float)
    b = np.array([[[.25, .25, -1], [.25, .25, 1], [1, 1, 0]],
                  [[0, 0, 0], [-1, 0, 1], [0, -1, 1]],
                  [[.1, .1, 0], [.2, .1, 0], [.1, .2, 0]],
                  [[-100, -100, 2], [100, -100, 2], [0, 100, 2]]], float)
    brute = [(i, j) for i, x in enumerate(a) for j, y in enumerate(b)
             if _triangles_intersect(x[None], y[None])[0]]
    assert brute == [(0, 0), (0, 1), (0, 2)]
    assert BoundaryIndex(b).hits(a) == brute


def _config(formula):
    profile = ({'r0_mm': 12.7, 'a0_deg': 18., 'termination': 'flat_baffle',
                'L_mm': 120., 'R_mm': 110.} if formula == 'ICW' else
               {'R_mm': 150., 'r0_mm': 12.7, 'a0_deg': 15.5, 'a_deg': 55.,
                'k': 1., 'q': 1., 'm': .85, 'r': .35, 'b': .4, 'tmax': 1.,
                's1': .7, 's2': .3})
    return {'formula': formula, 'mode': 'freestanding', 'profile': profile,
            'mesh': {'quadrants': '14' if formula == 'ICW' else '1234',
                     'allow_large_mesh': True,
                     'wall_thickness_mm': .6 if formula == 'ICW' else 3},
            'source': {'source_shape': 0}, 'scale': .35 if formula == 'ICW' else .48,
            'morph': {'morphTarget': 1, 'morphWidth': 360, 'morphHeight': 210,
                      'morphCorner': 10 if formula == 'ICW' else 0,
                      'morphAllowShrinkage': 1, 'morphFixed': .4 if formula == 'ICW' else .35}}


@pytest.mark.parametrize('formula,reason', [
    ('ICW', r'tessellation.*99 before.*CAD clearance 0\.59.*resolution or wall thickness'),
    ('R-OSSE', r'CAD bore and outer shell nearly touch or cross.*271 emitted.*Reduce rollback'),
])
def test_intersecting_real_requests_refuse_without_replacing_output(tmp_path, formula, reason):
    resolved = resolve_geometry(_config(formula))
    assert (resolved.density.throat_res_mm, resolved.density.mouth_res_mm,
            resolved.density.rear_res_mm) == (4, 26, 15)
    path = tmp_path / 'solve.msh'
    path.write_bytes(b'previous valid result')
    with pytest.raises(MesherError, match=reason):
        build_mesh_with_info(resolved.geometry, resolved.density, path)
    assert path.read_bytes() == b'previous valid result'
    assert list(tmp_path.iterdir()) == [path]


def test_real_unmorphed_icw_repairs_emitted_boundary_at_user_density(tmp_path, monkeypatch):
    import meshio
    import hornlab_mesher.shell_facets as sf
    from collections import Counter
    from hornlab_mesher.mesh_repair import _remove_degenerate_triangles, DEGENERATE_MIN_QUALITY

    config = _config('ICW')
    config.pop('morph')
    config['scale'] = 1
    config['source']['source_shape'] = 1
    config['mesh'].update(quadrants='1234', wall_thickness_mm=1.5)
    resolved = resolve_geometry(config)
    assert (resolved.density.throat_res_mm, resolved.density.mouth_res_mm,
            resolved.density.rear_res_mm) == (4, 26, 15)
    captured = {}
    check = sf.validate_shell_facets
    def audit(points, triangles, surfaces, physical, groups, wall, axes):
        before = triangles.copy()
        xyz, tags, phys = points.copy(), surfaces.copy(), physical.copy()
        def boundary(t):
            edges = Counter(tuple(sorted((int(a), int(b)))) for f in t
                            for a, b in zip(f, np.roll(f, -1)))
            return {k: v for k, v in edges.items() if v != 2}
        result = check(points, triangles, surfaces, physical, groups, wall, axes)
        np.testing.assert_array_equal(points, xyz)
        np.testing.assert_array_equal(surfaces, tags)
        np.testing.assert_array_equal(physical, phys)
        for tag in np.unique(tags):
            a, b = before[tags == tag], triangles[tags == tag]
            assert boundary(a) == boundary(b)
            assert set(a.ravel()) == set(b.ravel())
        changed = np.any(before != triangles, axis=1)
        assert _remove_degenerate_triangles(
            points, triangles[changed], tags[changed],
            min_quality=DEGENERATE_MIN_QUALITY)[2] == 0
        captured.update(tags=tags, groups=groups, before=before, result=result)
        return result
    monkeypatch.setattr(sf, 'validate_shell_facets', audit)
    path, info = build_mesh_with_info(resolved.geometry, resolved.density,
                                      tmp_path / 'repaired.msh', scale_to_metres=False)
    assert captured['result']['pairs_before'] == 10
    assert captured['result']['pairs_after'] == 0
    assert captured['result']['changed_facets'] > 0
    assert info.n_triangles == len(captured['before'])
    # Check the actual written facets independently, not the repair's report.
    mesh = meshio.read(path)
    facets = mesh.points[mesh.cells_dict['triangle']]
    tags, groups = captured['tags'], captured['groups']
    bore = facets[np.isin(tags, groups['inner'])]
    shell = facets[np.isin(tags, groups['outer'])]
    from hornlab_mesher.facet_fit import _FacetIndex
    assert not _FacetIndex(shell).hits(bore)


def test_clean_boundary_keeps_solve_bytes_and_metadata(tmp_path, monkeypatch):
    import hornlab_mesher.shell_facets as sf
    config = _config('ICW')
    config['mesh'].update(quadrants='1234', wall_thickness_mm=3)
    config['scale'] = .48
    config['morph'].update(morphCorner=0, morphFixed=.35)
    resolved = resolve_geometry(config)
    path, info = build_mesh_with_info(resolved.geometry, resolved.density,
                                      tmp_path / 'checked.msh')
    assert 'shellFacetValidation' not in info.metadata
    monkeypatch.setattr(sf, 'validate_shell_facets', lambda *args: None)
    baseline, old_info = build_mesh_with_info(resolved.geometry, resolved.density,
                                              tmp_path / 'unchecked.msh')
    assert path.read_bytes() == baseline.read_bytes()
    assert info.metadata == old_info.metadata


def test_shared_vertex_never_hides_a_narrow_new_overlap():
    from hornlab_mesher.shell_facets import _shared_contact_only
    from hornlab_mesher.mesh_repair import _remove_degenerate_triangles, DEGENERATE_MIN_QUALITY
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                       [2e-9, 2, 0], [-2, 2e-9, 0],
                       [.5, .5, -1], [.5, .5, 1],
                       [-.5, -.5, -1], [-.5, -.5, 1],
                       [0, -1, 0], [.5, .5, 0]], float)
    triangles = np.array([[0, 1, 2], [0, 3, 4]])
    # Both faces are healthy, and nonshared vertices are far beyond the weld
    # tolerance. Their overlap is much narrower than a 1e-7 shrink margin.
    assert _remove_degenerate_triangles(points, triangles, np.ones(2),
                                        min_quality=DEGENERATE_MIN_QUALITY)[2] == 0
    assert not _shared_contact_only(points, triangles[0], triangles[1])
    assert not _shared_contact_only(points, [0, 1, 2], [0, 5, 6])
    assert _shared_contact_only(points, [0, 1, 2], [0, 7, 8])
    assert _shared_contact_only(points, [0, 1, 2], [0, 1, 9])
    assert not _shared_contact_only(points, [0, 1, 2], [0, 1, 10])
