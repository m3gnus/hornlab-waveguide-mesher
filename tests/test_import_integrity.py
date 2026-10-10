import numpy as np
import pytest
import meshio

from hornlab_mesher.config_parser import ConfigError, parse_text_config
from hornlab_mesher.config_builder import resolve_geometry, build_from_config
from hornlab_mesher.profile_sampling import build_point_grid_arrays
from hornlab_mesher.edges import build_edge_table

BASE = '''
OSSE = {
 Length = 80
 Throat.Diameter = 20
 Coverage.Angle = 45
 Throat.Angle = 10
 OS.k = 1
 Term.n = 4
 Term.q = 0.995
}
Mesh.AngularSegments = 16
Mesh.LengthSegments = 20
Mesh.WallThickness = 0
Mesh.ThroatResolution = 8
Mesh.MouthResolution = 15
Mesh.RearResolution = 20
Mesh.InterfaceResolution = 8
Mesh.Enclosure = {
 Depth = 120
 EdgeRadius = 0
}
'''


def triangles_and_tags(path):
    mesh = meshio.read(path)
    blocks = [i for i, cell in enumerate(mesh.cells) if cell.type == 'triangle']
    return (np.asarray(mesh.points),
            np.concatenate([mesh.cells[i].data for i in blocks]),
            np.concatenate([mesh.cell_data['gmsh:physical'][i] for i in blocks]))


def test_explicit_zmap_endpoints_through_imported_resolver():
    explicit = resolve_geometry(parse_text_config(BASE + 'Mesh.ZMapPoints = 0,0,1,1\n'))
    implicit = resolve_geometry(parse_text_config(BASE + 'Mesh.ZMapPoints = 0.5,0.5\n'))
    np.testing.assert_allclose(explicit.geometry.inner_points, implicit.geometry.inner_points, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize('controls', ['0,0,0.5,0.1,0.75,0.7,1,1',
                                       '0,0,0.5,0.1,0.75,0.7',
                                       '0.5,0.1,0.75,0.7,1,1'])
def test_explicit_and_implicit_zmap_endpoints_keep_same_physical_profile(controls):
    params = {'type':'OSSE', 'L':100, 'r0':12.7, 'a':45, 'a0':10,
              'lengthSegments':64, 'angularSegments':8, 'samplingMode':'zmap'}
    expected = build_point_grid_arrays({**params, 'zMapPoints':'0.5,0.1,0.75,0.7'})
    actual = build_point_grid_arrays({**params, 'zMapPoints':controls})
    np.testing.assert_allclose(actual['inner_grid'], expected['inner_grid'], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual['slice_map'], expected['slice_map'], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize('controls,message', [
    ('0,0.2,1,1','endpoint'), ('0,0,1,0.9','endpoint'),
    ('0,0,0,0,1,1','strictly increasing'),
    ('0,0,0.7,0.8,0.5,0.9,1,1','strictly increasing'),
    ('0,0,0.5,0.8,0.75,0.7,1,1','non-decreasing'),
])
def test_invalid_explicit_zmap_controls_refused(controls, message):
    with pytest.raises(ValueError, match=message):
        build_point_grid_arrays({'type':'OSSE','L':100,'r0':12.7,'a':45,'a0':10,
                                 'lengthSegments':64,'angularSegments':8,
                                 'samplingMode':'zmap','zMapPoints':controls})


def test_full_zmap_sample_map_unchanged():
    grid = build_point_grid_arrays({'type':'OSSE','L':100,'r0':12.7,'a':45,'a0':10,
                                    'lengthSegments':4,'angularSegments':8,
                                    'samplingMode':'zmap','zMapPoints':[0,0.1,0.3,0.7,1]})
    np.testing.assert_allclose(grid['slice_map'], [0,0.1,0.3,0.7,1], atol=1e-12)


@pytest.mark.parametrize('steps,controls', [(3, '0,0,1,1'), (5, '0,0,0.5,0.2,1,1')])
def test_endpoint_controls_take_precedence_over_coincidental_sample_count(steps, controls):
    from hornlab_mesher.profile_sampling import _custom_zmap

    actual = _custom_zmap(steps, controls, prefer_endpoint_controls=True)
    expected = _custom_zmap(steps, controls, 'controls')
    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.diff(actual) > 0)


@pytest.mark.parametrize('imported,explicit_kind,expected_kind', [
    (False, None, 'samples'),
    (False, 'controls', 'controls'),
    (True, None, 'controls'),
    (True, 'samples', 'samples'),
])
def test_ambiguous_zmap_preserves_native_and_versioned_ath_geometry(
    imported, explicit_kind, expected_kind,
):
    from hornlab_mesher.config_builder import build_geometry_params
    from hornlab_mesher.text_import import TEXT_IMPORT_VERSION, TEXT_IMPORT_VERSION_KEY

    stations = [0, 0, .5, .5, 1, 1]
    config = {'type': 'OSSE', 'L': 100, 'r0': 12.7, 'a': 45, 'a0': 10,
              'lengthSegments': 5, 'angularSegments': 8,
              'samplingMode': 'zmap', 'zMapPoints': stations}
    if imported:
        config[TEXT_IMPORT_VERSION_KEY] = TEXT_IMPORT_VERSION
    if explicit_kind is not None:
        config['zMapKind'] = explicit_kind
    params, _, _ = build_geometry_params(config)
    assert params['zMapKind'] == expected_kind
    actual = build_point_grid_arrays(params)
    expected = build_point_grid_arrays({**params, 'zMapKind': expected_kind})
    np.testing.assert_array_equal(actual['inner_grid'], expected['inner_grid'])
    np.testing.assert_allclose(actual['slice_map'],
                               stations if expected_kind == 'samples' else np.linspace(0, 1, 6),
                               rtol=0, atol=1e-15)
    # Direct parameter callers must use the same versioned sampling authority.
    direct = build_point_grid_arrays(config)
    np.testing.assert_array_equal(direct['slice_map'], actual['slice_map'])


@pytest.mark.parametrize('imported,kind', [(False, 'samples'), (True, 'controls')])
def test_ambiguous_zmap_resolved_geometry_matches_explicit_kind(imported, kind):
    config = {'type': 'OSSE', 'L': 100, 'r0': 12.7, 'a': 45, 'a0': 10,
              'lengthSegments': 5, 'angularSegments': 8,
              'samplingMode': 'zmap', 'zMapPoints': [0, 0, .5, .5, 1, 1]}
    if imported:
        from hornlab_mesher.text_import import TEXT_IMPORT_VERSION, TEXT_IMPORT_VERSION_KEY
        config[TEXT_IMPORT_VERSION_KEY] = TEXT_IMPORT_VERSION
    automatic = resolve_geometry(config)
    explicit = resolve_geometry({**config, 'zMapKind': kind})
    np.testing.assert_array_equal(automatic.geometry.inner_points, explicit.geometry.inner_points)


def test_explicit_legacy_full_samples_keep_repeated_station_behavior():
    from hornlab_mesher.profile_sampling import _custom_zmap

    np.testing.assert_array_equal(_custom_zmap(3, [0,0,1,1], 'samples'), [0,0,1,1])


@pytest.mark.parametrize('offsets,expected,slices', [
    ('0', [0], '9'), ('0', [0], '19'),
    ('0,5', [0,5], '9,19'), ('', [5], '9'),
])
@pytest.mark.parametrize('quadrants', ['1234', '1', '12'])
def test_imported_enclosure_interfaces_emit_requested_planes_and_tags(tmp_path, offsets, expected, slices, quadrants):
    text = BASE + f'Mesh.SubdomainSlices = {slices}\nMesh.Quadrants = {quadrants}\n'
    if offsets:
        text += f'Mesh.InterfaceOffset = {offsets}\n'
    config = parse_text_config(text)
    resolved = resolve_geometry(config)
    specs = resolved.geometry.interfaces
    assert [spec.offset_mm for spec in specs] == expected
    planes = [resolved.geometry.inner_points[:, spec.slice_index, 2].mean() + spec.offset_mm
              for spec in specs]
    result = build_from_config(config, tmp_path/'interfaces.msh')
    points, triangles, tags = triangles_and_tags(result.mesh_path)
    if result.units == "m":
        points *= 1000
    assert {1,2,3,4}.issubset(set(tags))
    assert np.all(build_edge_table(triangles).count <= 2)
    # Shell topology stays closed independently of the detached virtual partition.
    shell = build_edge_table(triangles[tags != 4])
    assert np.all(shell.count <= 2)
    free = shell.count == 1
    if quadrants == '1234':
        assert not np.any(free)
    else:
        on_cut = np.zeros(np.count_nonzero(free), dtype=bool)
        for plane in resolved.geometry.symmetry_planes:
            axis = {'x': 0, 'y': 1, 'z': 2}[plane]
            on_cut |= ((np.abs(points[shell.lo[free], axis]) < 1e-7) &
                       (np.abs(points[shell.hi[free], axis]) < 1e-7))
        assert np.all(on_cut)
    interface = triangles[tags == 4]
    assert interface.size
    vertices = points[interface]
    for plane in planes:
        in_plane = np.all(np.abs(vertices[:,:,2] - plane) < 1e-7, axis=1)
        assert np.count_nonzero(in_plane) > 4
        cap_edges = build_edge_table(interface[in_plane])
        assert np.all(cap_edges.count <= 2)
        assert np.count_nonzero(cap_edges.count == 1) > 4
        cap = interface[in_plane]
        # Connected triangulated disk: Euler characteristic one, with every
        # triangle reachable through the disk's shared vertices.
        assert len(np.unique(cap)) - cap_edges.n_edges + len(cap) == 1
        reachable = {int(cap[0,0])}
        while True:
            adjacent = cap[np.any(np.isin(cap, list(reachable)), axis=1)]
            expanded = set(int(v) for v in adjacent.ravel())
            if expanded == reachable:
                break
            reachable = expanded
        assert len(reachable) == len(np.unique(cap))
        area = np.linalg.norm(np.cross(vertices[in_plane,1]-vertices[in_plane,0],
                                       vertices[in_plane,2]-vertices[in_plane,0]),axis=1).sum()/2
        assert area > 100
    if offsets == '0':
        assert np.max(np.abs(vertices[:,:,2] - planes[0])) < 1e-7


@pytest.mark.parametrize('offset', ['-1', 'nan', 'inf', '0,nan'])
@pytest.mark.parametrize('slices', ['9', ''])
def test_invalid_interface_offset_is_named_refusal(offset, slices):
    with pytest.raises(ConfigError, match='InterfaceOffset.*non-negative'):
        resolve_geometry(parse_text_config(BASE + f'Mesh.SubdomainSlices = {slices}\nMesh.InterfaceOffset = {offset}\n'))


@pytest.mark.parametrize('mode', ['freestanding', 'infinite-baffle'])
def test_zero_interface_preserves_non_enclosure_refusal(mode):
    text = BASE.split('Mesh.Enclosure')[0]
    if mode == 'freestanding':
        text = text.replace('Mesh.WallThickness = 0', 'Mesh.WallThickness = 5')
    else:
        text += 'ABEC.SimType = 1\n'
    with pytest.raises(ConfigError, match='subdomain interfaces.*only supported for enclosure'):
        resolve_geometry(parse_text_config(text + 'Mesh.SubdomainSlices = 9\nMesh.InterfaceOffset = 0\n'))


def test_zero_offset_on_nonplanar_ring_is_named_refusal():
    from hornlab_mesher.builders.point_grid_interfaces import _add_offset_interface_surfaces

    grid = np.zeros((8, 2, 3), dtype=np.float64)
    grid[0, 1, 2] = 0.1
    with pytest.raises(ValueError, match='zero-offset HornInterface requires a planar axial ring'):
        _add_offset_interface_surfaces(grid, slice_index=1, closed=True, offset_mm=0)
