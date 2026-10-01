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


# Crossing counts depend on the platform's tessellation (CI saw 99/117 and
# 271/261 for the same requests), so match the message family, the remedy and
# a positive count, and keep the clearance only to its stable precision.
@pytest.mark.parametrize('formula,reason', [
    ('ICW', r'tessellation has [1-9]\d* remaining.*\([1-9]\d* before; CAD clearance 0\.59\d*'
            r'.*finer \(smaller mm\) resolution'),
    ('R-OSSE', r'CAD bore and shell nearly touch or cross.*[1-9]\d* prohibited emitted.*Try reducing wall_thickness_mm'),
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
    def audit(points, triangles, surfaces, physical, groups, wall, axes, **kwargs):
        before = triangles.copy()
        xyz, tags, phys = points.copy(), surfaces.copy(), physical.copy()
        def boundary(t):
            edges = Counter(tuple(sorted((int(a), int(b)))) for f in t
                            for a, b in zip(f, np.roll(f, -1)))
            return {k: v for k, v in edges.items() if v != 2}
        result = check(points, triangles, surfaces, physical, groups, wall, axes, **kwargs)
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
    # The count before repair varies with the platform's tessellation; what
    # the repair guarantees is that it starts from real crossings and ends
    # with none.
    assert captured['result']['pairs_before'] > 0
    assert captured['result']['pairs_after'] == 0
    assert captured['result']['changed_facets'] > 0
    assert info.n_triangles == len(captured['before'])
    # Check the actual written facets independently, not the repair's report.
    mesh = meshio.read(path)
    facets = mesh.points[mesh.cells_dict['triangle']]
    tags, groups = captured['tags'], captured['groups']
    bore = facets[np.isin(tags, groups['inner'])]
    shell = facets[~np.isin(tags, groups['inner'])]
    from shell_oracle import strict_pairs
    assert not strict_pairs(bore, shell)


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
    monkeypatch.setattr(sf, 'validate_shell_facets', lambda *args, **kwargs: None)
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


# These are final-validator controls, not broad-phase expectations. In
# particular, shared indices alone never authorize dropping a positive pair.
CONTACTS = {
    'shared-nonplanar-edge': ([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], [[0,1,2],[0,1,3]]),
    'shared-coplanar-edge': ([[0,0,0],[1,0,0],[0,1,0],[0,-1,0]], [[0,1,2],[0,1,3]]),
    'shared-nonplanar-vertex': ([[0,0,0],[1,0,0],[0,1,0],[-1,0,1],[0,-1,1]], [[0,1,2],[0,3,4]]),
    'shared-coplanar-vertex': ([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]], [[0,1,2],[0,3,4]]),
    'coplanar-point-distinct-ids': ([[0,0,0],[1,0,0],[0,1,0],[0,0,0],[-1,0,0],[0,-1,0]], [[0,1,2],[3,4,5]]),
    'coplanar-edge-distinct-ids': ([[0,0,0],[1,0,0],[0,1,0],[0,0,0],[1,0,0],[0,-1,0]], [[0,1,2],[3,4,5]]),
    'coplanar-partial-edge': ([[0,0,0],[1,0,0],[0,1,0],[.2,0,0],[.8,0,0],[.5,-1,0]], [[0,1,2],[3,4,5]]),
    'coplanar-boundary-ray': ([[0,0,0],[1,0,0],[0,1,0],[0,2,0],[-1,0,0]], [[0,1,2],[0,3,4]]),
}


@pytest.mark.parametrize('name', CONTACTS)
def test_allowed_contacts_never_query_cad_or_repair(monkeypatch, name):
    import gmsh
    import hornlab_mesher.shell_facets as sf
    from shell_oracle import strict_pairs
    xyz, faces = CONTACTS[name]
    p, t = np.array(xyz, float), np.array(faces)
    assert not strict_pairs(p[t[:1]], p[t[1:]])
    def unexpected(*args, **kwargs):
        pytest.fail('contact-only validation must not query CAD or repair')
    monkeypatch.setattr(gmsh.model.occ, 'getDistance', unexpected)
    monkeypatch.setattr(sf, 'repair_fitted_bore_facets', unexpected)
    before = t.copy()
    result = sf.validate_shell_facets(p, t, np.array([10,20]), np.ones(2, int),
                                     {'inner':[10], 'outer':[20]}, 3)
    assert result['pairs_before'] == result['pairs_after'] == result['changed_facets'] == 0
    np.testing.assert_array_equal(t, before)


@pytest.mark.parametrize('second', [
    [[.25,.25,-1],[.25,.25,1],[1,1,0]],  # independent strict witness
    [[.1,.1,0],[.2,.1,0],[.1,.2,0]],    # coplanar positive area
    [[0,0,0],[2e-9,2,0],[-2,2e-9,0]],  # narrow shared-vertex overlap
])
def test_prohibited_pairs_survive_contact_filter(second):
    from hornlab_mesher.shell_facets import CrossingIndex
    a = np.array([[[0,0,0],[1,0,0],[0,1,0]]], float)
    b = np.array([second], float)
    assert CrossingIndex(b).hits(a)


@pytest.mark.parametrize('component', ['mouth', 'mouth-bore1', 'mouth-bore2', 'mouth-bore4',
                                        'rear', 'rear_cap', 'throat_disc', 'unlisted'])
def test_real_emitted_joins_are_clean_but_injected_component_crossing_refuses(
        tmp_path, monkeypatch, component):
    import hornlab_mesher.shell_facets as sf
    from shell_oracle import strict_pairs
    config = _config('ICW')
    config.pop('morph')
    config['scale'] = 1
    config['mesh'].update(wall_thickness_mm=2, quadrants='1234')
    config['source']['source_shape'] = 1
    r = resolve_geometry(config)
    check = sf.validate_shell_facets
    observed = []
    def audit(p, t, s, physical, groups, wall, axes, **kwargs):
        before = t.copy()
        result = check(p, t, s, physical, groups, wall, axes, **kwargs)
        assert result['pairs_before'] == result['changed_facets'] == 0
        np.testing.assert_array_equal(t, before)
        ii = np.flatnonzero(np.isin(s, groups['inner']))
        xyz = p[t[ii]]
        if component.startswith('mouth-bore'):
            # Fix the bore quadrant: the global argmin can choose a different
            # equally close axial facet on another platform/dependency build.
            xyz = p[t[ii[s[ii] == groups['inner'][int(component[-1])-1]]]]
        a = xyz[np.argmin(np.abs(xyz.mean(1)[:,2]-60))]
        center = a.mean(0)
        normal = np.cross(a[1]-a[0], a[2]-a[0])
        normal /= np.linalg.norm(normal)
        injected = np.vstack((p, center+.25*normal, center-.25*normal,
                              center+.3*(a[1]-a[0])))
        # Move one real component face to an independent crossing witness.
        mi = np.flatnonzero(np.isin(s, groups['mouth']))[0]
        bad, tags, roles = t.copy(), s.copy(), dict(groups)
        bad[mi] = np.arange(len(p), len(p)+3)
        if not component.startswith('mouth'):
            tags[mi] = 90000
            roles = {k:[tag for tag in v if tag != s[mi]] for k,v in groups.items()}
            if component != 'unlisted':
                roles[component] = [*roles.get(component, []), 90000]
        assert strict_pairs(injected[bad[ii]], injected[bad[mi:mi+1]])
        old = bad.copy()
        with pytest.raises(MesherError, match='bore/shell.*(intersect|facet pairs)'):
            check(injected, bad, tags, physical, roles, wall, axes, **kwargs)
        np.testing.assert_array_equal(bad, old)
        observed.append(True)
        return result
    monkeypatch.setattr(sf, 'validate_shell_facets', audit)
    build_mesh_with_info(r.geometry, r.density, tmp_path/'clean.msh')
    assert observed


def test_default_thin_icw_refusal_names_region_sizes_and_finer_remedy(tmp_path):
    config = _config('ICW')
    config.pop('morph')
    config['scale'] = 1
    config['source']['source_shape'] = 1
    config['mesh'].update(quadrants='1234', wall_thickness_mm=1)
    r = resolve_geometry(config)
    path = tmp_path/'absent.msh'
    with pytest.raises(MesherError) as refusal:
        build_mesh_with_info(r.geometry, r.density, path)
    message = str(refusal.value)
    for text in ('mouth region z=', 'boundary patches', 'wall_thickness_mm=1 mm',
                 'throat_res_mm=4 mm', 'mouth_res_mm=26 mm', 'rear_res_mm=15 mm',
                 'thicker wall_thickness_mm', 'finer (smaller mm)',
                 'mouth_res_mm=8 mm', 'suggested, not a guaranteed remedy'):
        assert text in message
    assert not path.exists() and not list(tmp_path.iterdir())
    # This value is measured for this specific request, not a general promise.
    config['mesh']['mouth_res_mm'] = 8
    r = resolve_geometry(config)
    _, info = build_mesh_with_info(r.geometry, r.density, path)
    assert path.exists() and info.n_triangles > 0
    assert 'shellFacetValidation' not in info.metadata


@pytest.mark.parametrize('gap,trim,reason', [
    (2., False, 'bore/shell tessellation'),
    (.02, False, 'CAD bore and shell'),
    (.02, True, 'bore/shell tessellation'),
])
def test_global_zero_requires_local_interior_cad_evidence(monkeypatch, gap, trim, reason):
    """Global zero and coincident edge tags cannot choose the contact remedy."""
    import gmsh
    import hornlab_mesher.shell_facets as sf
    p = np.array([[0,0,0],[1,0,0],[0,1,0],
                  [.25,.25,-1],[.25,.25,1],[1,1,0]], float)
    t = np.array([[0,1,2],[3,4,5]])
    before = t.copy()
    # Deliberately give both patches the same boundary tag. Local interior
    # contact must still be detected, rather than skipping the entire pair.
    monkeypatch.setattr(gmsh.model, 'getBoundary', lambda *a, **kw: [(1,30)])
    monkeypatch.setattr(gmsh.model.occ, 'getDistance', lambda *a: (0.,)*7)
    queries = []
    def closest(x, y, z, entities, n=1):
        queries.append((x,y,z))
        if entities[0][0] == 1:
            return entities, [0. if trim else 10.], [x,y,z]
        offset = gap/2 if entities[0][1] == 10 else -gap/2
        return entities, [abs(offset)], [x,y,z+offset]
    monkeypatch.setattr(gmsh.model.occ, 'getClosestEntities', closest)
    monkeypatch.setattr(sf, 'repair_fitted_bore_facets', lambda *a, **kw: None)
    with pytest.raises(MesherError, match=reason):
        sf.validate_shell_facets(p,t,np.array([10,20]),np.ones(2,int),
                                 {'inner':[10], 'outer':[20]},1)
    assert queries
    # Every queried witness is on the intersection, rather than a centroid
    # or the global OCC extremum at a distant joined edge.
    assert queries[0][2] == pytest.approx(0)
    assert .25 <= queries[0][0] <= .5 and queries[0][0] == pytest.approx(queries[0][1])
    np.testing.assert_array_equal(t, before)


def test_coplanar_crossing_witness_lies_in_positive_overlap():
    from hornlab_mesher.shell_facets import _crossing_witnesses
    a = np.array([[0,0,0],[1,0,0],[0,1,0]], float)
    b = np.array([[.1,.1,0],[.2,.1,0],[.1,.2,0]], float)
    witness, = _crossing_witnesses(a,b)
    np.testing.assert_allclose(witness,b.mean(0))
