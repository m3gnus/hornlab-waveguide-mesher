"""Preview fidelity, authored-surface coincidence and closure for ATH imports."""
from pathlib import Path

import gmsh
import meshio
import numpy as np
import pytest

from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.config_parser import parse_text_config, ConfigError
from hornlab_mesher.preview import build_preview_geometry, PreviewOptionsV1
from hornlab_mesher.viewport import build_viewport_geometry_from_config
from hornlab_mesher.builders.point_grid_freestanding import _build_ath_import_surfaces
from test_ath_import_surfaces import assert_watertight_after_symmetry

FIXTURES = Path(__file__).parent / 'fixtures/ath_import/ath-v2026-08c'


def config(name):
    return parse_text_config((FIXTURES / (name + '.cfg')).read_text())


def assert_closed(surfaces, geometry):
    positions, triangles, physical = [], [], []
    offset = 0
    for surface in surfaces:
        positions.append(surface.positions)
        cells = surface.indices.reshape(-1, 3)
        triangles.append(cells + offset)
        physical.append(np.full(len(cells), 2 if surface.role == 'source_cap' else 1))
        offset += len(surface.positions)
    points = np.concatenate(positions)
    _, first, inverse = np.unique(np.round(points, 9), axis=0, return_index=True, return_inverse=True)
    mesh = meshio.Mesh(points[first], [('triangle', inverse[np.concatenate(triangles)])],
                       cell_data={'gmsh:physical': [np.concatenate(physical)]})
    assert_watertight_after_symmetry(mesh, geometry)


@pytest.mark.parametrize('name', ['onr', 'rs1', 'coreabec'])
@pytest.mark.parametrize('lod', ['coarse', 'fine', 'inspection'])
def test_preview_and_viewport_use_mesh_patches_at_each_lod(name, lod, monkeypatch):
    import hornlab_mesher.preview.ath_import as renderer
    cfg = config(name)
    geometry = resolve_geometry(cfg).geometry
    captured = []
    boundary_parameters = {}
    original = renderer._initial
    def capture(*args):
        result = original(*args)
        captured.extend(result[1])
        boundary_parameters.update(reference=result[2])
        return result
    monkeypatch.setattr(renderer, '_initial', capture)
    # Independently construct the mesh authority in a caller-owned model.
    # The renderer must leave this model and its options intact.
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add('authored-preview-reference')
        built = _build_ath_import_surfaces(geometry)
        original_option = gmsh.option.getNumber('Mesh.MeshSizeMax')
        output = build_viewport_geometry_from_config(cfg, point_lists=False, lod=lod)
        assert gmsh.model.getCurrent() == 'authored-preview-reference'
        assert gmsh.option.getNumber('Mesh.MeshSizeMax') == original_option
        surfaces = output['surfaces']
        roles = {s.role: s for s in surfaces}
        assert {'horn.inner', 'horn.outer', 'mouth_rim', 'wall.rear_return',
                'wall.rear_cap', 'source_cap'} == set(roles)
        params = output['grid']
        np.testing.assert_array_equal(params['inner_grid'], geometry.inner_points)
        np.testing.assert_array_equal(params['outer_grid'], geometry.ath_rear_points)
        np.testing.assert_array_equal(params['rear_apex'], geometry.ath_rear_apex)
        assert params['rear_grid_n_length'] + 1 == geometry.ath_rear_points.shape[1]
        expected_tags = {tag for tags in built.mesh_surface_groups.values() for tag in tags}
        patch_by_tag = {p['tag']: p for p in captured}
        for surface in surfaces:
            for span in surface.metadata['authoredPatchSpans']:
                tag, start, count = span['surface'], span['start'], span['count']
                assert tag in expected_tags
                patch = patch_by_tag[tag]
                actual = surface.positions[start:start+count]
                # All vertices, including seams, fans and source. The bound is
                # 1e-6 mm, independent of the LOD's triangle chord-error budget.
                expected = np.asarray(gmsh.model.getValue(2, tag, patch['uv'].reshape(-1).tolist())).reshape(-1, 3)
                # GEO rounded-source interiors are projected onto a sphere,
                # whereas their shared spline rim remains the mesh boundary
                # authority. Test seams on those exact shared curves; inverse
                # surface reparametrization alone can miss them by microns.
                boundary = {abs(t) for _, t in gmsh.model.getBoundary([(2, tag)])}
                node_curve = {}
                for (left, right), (curve, tl, tr) in boundary_parameters['reference'].items():
                    if curve in boundary:
                        node_curve[left] = curve, tl
                        node_curve[right] = curve, tr
                for index, node in enumerate(patch['ids']):
                    if node in node_curve:
                        curve, parameter = node_curve[node]
                        expected[index] = gmsh.model.getValue(1, curve, [parameter])
                assert np.linalg.norm(actual-expected, axis=1).max() <= 1e-6
            assert surface.metadata['disagreeingTriangles'] == 0
            assert surface.curvature_mean is not None
            assert np.isfinite(surface.curvature_mean).all()
            fidelity = output['metadata']['fidelity'][surface.role]
            assert fidelity['measurement_complete']
            assert fidelity['max_chord_error_mm'] <= output['metadata']['requested_fidelity']['max_chord_error_mm']
            assert not fidelity['vertex_cap_limited']
            assert not fidelity['iteration_bound_limited']
            assert not fidelity['refinement_stalled']
            assert surface.metadata['foldedTriangles'] == 0
            assert surface.metadata['creaseFeatures']
            assert fidelity['max_normal_step_deg'] <= output['metadata']['requested_fidelity']['max_normal_step_deg']
            # Independent check of every emitted edge within the authored patches.
            # Adjacent patches duplicate seam vertices and retain one-sided normals.
            triangles = surface.indices.reshape(-1, 3)
            n = surface.normals[triangles]
            steps = np.degrees(np.arccos(np.clip(np.sum(n*np.roll(n, -1, axis=1), axis=-1), -1, 1)))
            assert steps.max() <= output['metadata']['requested_fidelity']['max_normal_step_deg']
        assert_closed(surfaces, geometry)
        # The vertex budget bounds runaway refinement; wall-clock time is not
        # asserted because CI runners differ several-fold in speed.
        assert sum(len(s.positions) for s in surfaces) <= {'coarse': 20000, 'fine': 100000, 'inspection': 170000}[lod]
    finally:
        gmsh.finalize()


@pytest.mark.parametrize('topology,preserve', [('acoustic', False), ('legacy', False), ('legacy', True)])
@pytest.mark.parametrize('name', ['onr', 'rs1'])
def test_full_domain_preview_closes_in_each_topology(topology, preserve, name):
    cfg = config(name)
    cfg['mesh'].update(quadrants=1234, topology=topology, preserve_grid=preserve)
    preview = build_preview_geometry(cfg, PreviewOptionsV1(lod='coarse', include_curvature=False))
    assert_closed(preview.surfaces, resolve_geometry(cfg).geometry)
    assert all(s.curvature_mean is None for s in preview.surfaces)


@pytest.mark.parametrize('name', ['p1', 'onr', 'rs1', 'coreabec'])
def test_viewport_flat_list_export_retains_authored_inner_and_rear_counts(name):
    cfg = config(name)
    geometry = resolve_geometry(cfg).geometry
    output = build_viewport_geometry_from_config(cfg)
    grid = output['grid']
    np.testing.assert_array_equal(np.asarray(grid['inner_points']).reshape(geometry.inner_points.shape), geometry.inner_points)
    assert grid['grid_n_length'] + 1 == geometry.inner_points.shape[1]
    if geometry.ath_rear_points is None:
        assert grid['outer_points'] is None
        assert grid['outer_grid_n_length'] is None
    else:
        np.testing.assert_array_equal(np.asarray(grid['outer_points']).reshape(geometry.ath_rear_points.shape), geometry.ath_rear_points)
        assert grid['outer_grid_n_length'] + 1 == geometry.ath_rear_points.shape[1]


def test_wg_inner_export_can_request_bare_core_with_opus_rear_removed():
    cfg = config('coreabec')
    expected = resolve_geometry(cfg).geometry.inner_points
    cfg['mode'] = 'bare'
    cfg['mesh']['wallThickness'] = 0
    for key in ('athRearShape', 'athRearFrontOffset', 'athRearStep'):
        cfg['mesh'].pop(key, None)
    grid = build_viewport_geometry_from_config(cfg, point_lists=False)['grid']
    np.testing.assert_array_equal(grid['inner_grid'], expected)
    assert grid['outer_grid'] is None and grid['rear_apex'] is None


@pytest.mark.parametrize('name', ['onr', 'rs1'])
def test_visibility_preserves_authored_surfaces_and_unprobed_placement_is_refused(name):
    cfg = config(name)
    before = build_preview_geometry(cfg, PreviewOptionsV1(lod='coarse', include_curvature=False))
    after = build_preview_geometry(cfg, PreviewOptionsV1(lod='coarse', include_curvature=False,
        include_source_cap=False, include_rear_cap=False, include_outer=False))
    assert [s.role for s in after.surfaces] == ['horn.inner']
    np.testing.assert_array_equal(after.surfaces[0].positions, before.surfaces[0].positions)
    cfg['mesh']['verticalOffset'] = 23
    for consumer in (resolve_geometry, build_preview_geometry, build_viewport_geometry_from_config):
        with pytest.raises(ConfigError, match='Mesh.VerticalOffset'):
            consumer(cfg)


@pytest.mark.parametrize('value', [-5, -20, -.001])
@pytest.mark.parametrize('transport', ['text', 'dictionary'])
def test_negative_front_offset_has_named_refusal(value, transport):
    if transport == 'text':
        with pytest.raises(ConfigError, match=r'Mesh.RearShape.FrontOffset < 0'):
            parse_text_config(f'OSSE = {{\n L = 100\n}}\nABEC.SimType = 2\nMesh.RearShape = 3\nMesh.RearShape.FrontOffset = {value}\n')
    else:
        cfg = config('onr')
        cfg['mesh']['athRearFrontOffset'] = value
        for consumer in (resolve_geometry, build_preview_geometry, build_viewport_geometry_from_config):
            with pytest.raises(ConfigError, match=r'Mesh.RearShape.FrontOffset < 0'):
                consumer(cfg)


@pytest.mark.parametrize('lod', ['coarse', 'fine', 'inspection'])
@pytest.mark.parametrize('source_shape', [0, 1])
def test_bare_import_lip_keeps_only_its_open_mouth_boundary(lod, source_shape):
    from hornlab_mesher.mesher import _validate_closed_shell_contract
    cfg = config('p1')
    cfg['source'] = {'source_shape': source_shape}
    geometry = resolve_geometry(cfg).geometry
    preview = build_preview_geometry(cfg, PreviewOptionsV1(lod=lod, include_curvature=False))
    assert {s.role for s in preview.surfaces} == {'horn.inner', 'source_cap'}
    for record in preview.metadata['fidelity'].values():
        assert record['max_chord_error_mm'] <= preview.metadata['requested_fidelity']['max_chord_error_mm']
    points, triangles, tags = [], [], []
    offset = 0
    for surface in preview.surfaces:
        points.append(surface.positions)
        triangles.append(surface.indices.reshape(-1, 3) + offset)
        tags.append(np.full(len(surface.indices)//3, 2 if surface.role == 'source_cap' else 1))
        offset += len(surface.positions)
    points = np.concatenate(points)
    _, first, inverse = np.unique(np.round(points, 9), axis=0, return_index=True, return_inverse=True)
    _validate_closed_shell_contract(points[first], inverse[np.concatenate(triangles)], np.concatenate(tags),
        symmetry_snap_axes=geometry.symmetry_planes, symmetry_snap_tol_mm=1e-6, is_open_shell=True)


def test_unrepresentable_preview_vertex_bound_is_explicit():
    with pytest.raises(ValueError, match='max_vertices=8 cannot represent'):
        build_preview_geometry(config('onr'), PreviewOptionsV1(max_vertices=8))


def test_iteration_bound_is_distinct_from_vertex_cap(monkeypatch):
    import hornlab_mesher.preview.ath_import as renderer
    monkeypatch.setattr(renderer, '_MAX_REFINEMENT_ITERATIONS', 1)
    preview = build_preview_geometry(config('coreabec'), PreviewOptionsV1(include_curvature=False))
    for record in preview.metadata['fidelity'].values():
        assert record['iteration_bound_limited']
        assert not record['vertex_cap_limited']
        assert record['refinement_limit'] == 'iterations'


def test_refinement_stops_when_no_nodes_are_added(monkeypatch):
    import hornlab_mesher.preview.ath_import as renderer
    monkeypatch.setattr(renderer, '_refine', lambda *args, **kwargs: None)
    preview = build_preview_geometry(config('coreabec'), PreviewOptionsV1(include_curvature=False))
    for record in preview.metadata['fidelity'].values():
        assert record['refinement_stalled']
        assert not record['vertex_cap_limited']
        assert not record['iteration_bound_limited']
        assert record['refinement_limit'] == 'no-progress'
