"""Second reliability-review regressions at every public C4 boundary."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from hornlab_mesher.cad import write_step_from_config
from hornlab_mesher.config_builder import build_from_config, build_geometry_params, resolve_geometry
from hornlab_mesher.config_parser import ConfigError, load_config, parse_text_config
from hornlab_mesher.geometry import OsseHornGeometry, RosseHornGeometry
from hornlab_mesher.preview import build_preview_geometry
from hornlab_mesher.preview.contract import PreviewOptionsV1
from hornlab_mesher.profile_formulas import _verify_meridian_self_contact

ENDPOINT = {
    'profile': {'formula': 'R-OSSE', 'R': 200, 'a': 35, 'a0': 3, 'r0': 8.25,
                'k': 4, 'r': .37, 'm': .2, 'b': 0, 'q': 4, 's1': .5, 's2': .2,
                'throatExtLength': 188.70456036837064,
                'throatExtAngle': -45.45862799159362},
    'mode': 'bare',
    'mesh': {'angularSegments': 16, 'lengthSegments': 32, 'surface_fit': 'interpolate',
             'topology': 'acoustic', 'max_triangles': 100000, 'allow_large_mesh': True},
}


def consume(path, config, tmp_path):
    if path == 'resolve':
        return resolve_geometry(config)
    if path in ('coarse', 'fine'):
        return build_preview_geometry(config, PreviewOptionsV1(lod=path))
    if path == 'mesh':
        return build_from_config(config, tmp_path / 'refused.msh')
    return write_step_from_config(config, tmp_path / 'refused.step')


PUBLIC = ['resolve', 'coarse', 'fine', 'mesh', 'step']


@pytest.mark.parametrize('path', PUBLIC)
@pytest.mark.parametrize('key', ['s1', 's2'])
@pytest.mark.parametrize('value', ['0.1-0.2*sin(16*p)^128', '-1+0*p', '10001+0*p',
                                  '1e309-1e309', '1/0', '0*p',
                                  'sin(p)^2-sin(p)^2', '0', '0.5', {'0': .5}, [.5]])
@pytest.mark.parametrize('companion', [0, .2])
def test_no_coefficient_expression_can_reach_a_consumer(path, key, value, companion, tmp_path):
    config = {'mode': 'bare', 'profile': {'L': 80, 'a': 35, 's1': companion, 's2': companion, key: value},
              'mesh': {'angularSegments': 16, 'lengthSegments': 12}}
    with pytest.raises(ConfigError, match='per-azimuth throat stretch is not supported yet'):
        consume(path, config, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('family', ['OSSE', 'R-OSSE'])
@pytest.mark.parametrize('key', ['s1', 's2'])
@pytest.mark.parametrize('value', ['0*p', 'sin(p)^2-sin(p)^2', '1/0', '0.5+cos(p)^2'])
def test_text_expressions_and_native_json_refuse_consistently(family, key, value, tmp_path):
    length = 'L = 160' if family == 'OSSE' else 'R = 200'
    with pytest.raises(ConfigError, match='per-azimuth throat stretch is not supported yet'):
        parse_text_config(f'{family} = {{\n{length}\ns1 = 0\ns2 = 0\n{key} = {value}\n}}')
    config = {'mode': 'bare', 'profile': {'formula': family, key: value}}
    file = tmp_path / 'native.json'
    file.write_text(json.dumps(config))
    with pytest.raises(ConfigError, match='per-azimuth throat stretch is not supported yet'):
        build_geometry_params(load_config(file))


@pytest.mark.parametrize('cls', [OsseHornGeometry, RosseHornGeometry])
@pytest.mark.parametrize('key', ['s1', 's2'])
def test_dataclass_does_not_canonicalize_away_a_dormant_expression(cls, key):
    with pytest.raises(ConfigError, match='per-azimuth throat stretch is not supported yet'):
        cls(**{key: '0*p'})


@pytest.mark.parametrize('path', PUBLIC)
@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), 10001, 10**400, None, True])
def test_invalid_numeric_coefficients_are_config_errors(path, value, tmp_path):
    config = {'profile': {'s1': value, 's2': 0}, 'mode': 'bare'}
    with pytest.raises(ConfigError):
        consume(path, config, tmp_path)


@pytest.mark.parametrize('path', PUBLIC)
@pytest.mark.parametrize('delta', [-1, 0, 1])
def test_source_endpoint_self_contact_is_refused_before_every_consumer(path, delta, tmp_path):
    config = copy.deepcopy(ENDPOINT)
    if delta:
        config['profile']['throatExtLength'] = np.nextafter(
            config['profile']['throatExtLength'], 0 if delta < 0 else np.inf).item()
    with pytest.raises(ConfigError, match='self-contact|self-intersection'):
        consume(path, config, tmp_path)
    assert not list(tmp_path.iterdir())


def test_endpoint_case_stretch_off_still_builds(tmp_path):
    config = copy.deepcopy(ENDPOINT)
    config['profile']['s1'] = 0
    assert build_from_config(config, tmp_path / 'off.msh').n_triangles > 0
    assert write_step_from_config(config, tmp_path / 'off.step')[0].is_file()


@pytest.mark.parametrize('points', [
    [[0, 2], [1, 2], [1, 3], [0, 2]],  # final endpoint at the driver
    [[0, 2], [2, 2], [2, 3], [1, 2]],  # endpoint inside a prefix segment
    [[0, 2], [2, 2], [3, 2], [1, 2]],  # collinear overlap
    [[0, 2], [2, 4], [0, 4], [2, 2]],  # proper crossing
    [[0, 2], [2, 2], [3, 3], [2, 2]],  # nonadjacent endpoint contact
])
def test_composite_contact_guard_includes_endpoints_and_collinear_segments(points):
    with pytest.raises(ConfigError, match='self-contact|self-intersection'):
        _verify_meridian_self_contact(np.array(points, dtype=float))


def test_adjacent_prefix_join_is_permitted():
    _verify_meridian_self_contact(np.array([[0, 2], [1, 2], [2, 2], [3, 3]], dtype=float))


COMPOSITIONS = [
    ('R-OSSE', {'throatExtLength': 12}, {'gcurveType': 1, 'gcurveWidth': 400},
     'GCurve.Type = 1\nGCurve.Width = 400\nThroat.Ext.Length = 12'),
    ('OSSE', {'throatExtLength': 12}, {'gcurveType': 1, 'gcurveWidth': 400},
     'GCurve.Type = 1\nGCurve.Width = 400\nThroat.Ext.Length = 12'),
    ('OSSE', {'rot': 10}, {'gcurveType': 1, 'gcurveWidth': 400},
     'GCurve.Type = 1\nGCurve.Width = 400\nRot = 10'),
    ('OSSE', {'rot': 10, 'throatExtLength': 12}, {}, 'Rot = 10\nThroat.Ext.Length = 12'),
    ('OSSE', {'slotLength': 8}, {}, 'Slot.Length = 8'),
    ('R-OSSE', {'rot': 10}, {}, 'Rot = 10'),
    ('R-OSSE', {'L': 180}, {}, 'Length = 180'),
]


@pytest.mark.parametrize('path', PUBLIC)
@pytest.mark.parametrize('section', ['profile', 'parameters'])
@pytest.mark.parametrize('family,profile,guide,extra', COMPOSITIONS)
def test_text_and_native_composition_refusals_have_identical_errors(path, section, family, profile, guide, extra, tmp_path):
    length = 'L = 160' if family == 'OSSE' else 'R = 200'
    with pytest.raises(ConfigError) as imported:
        parse_text_config(f'{family} = {{\n{length}\ns1 = .5\ns2 = .2\n}}\n{extra}')
    config = {'formula': family, 'mode': 'bare', section: {'s1': .5, 's2': .2, **profile}, 'gcurve': guide}
    with pytest.raises(ConfigError) as native:
        consume(path, config, tmp_path)
    assert str(native.value) == str(imported.value)


@pytest.mark.parametrize('expression', ['0*p', 'sin(p)^2-sin(p)^2', 'p-p', '(p-p)*cos(p)', '0.0',
                                         'sin(p-p)', '0/(1+p^2)', '(p+1)-(1+p)',
                                         'cos(p)^2+sin(p)^2-1'])
def test_zero_slot_expression_retains_base_import_mapping(expression):
    config = parse_text_config(f'OSSE = {{\nL = 160\n}}\nSlot.Length = {expression}')
    if expression == '0.0':
        assert config['profile']['slotLength'] == 0
    else:
        assert config['profile']['slotLength'] == expression
    resolve_geometry({**config, 'mode': 'bare'})


@pytest.mark.parametrize('quadrants', ['1', '12', '14', '1234'])
def test_reduced_morphed_stretched_solve_agrees_with_full_step(quadrants, tmp_path):
    import gmsh
    import meshio
    from hornlab_mesher.mesher import _triangles_and_physical_tags
    from hornlab_mesher.tags import PhysicalGroup

    # Scalar counterpart of the review reproduction, inside the release scope:
    # expression coefficients and the unverified OSSE slot now refuse.
    config = {'mode': 'bare', 'scale': .48,
              'profile': {'L': 80, 'a': 35, 'a0': 5, 'throatExtLength': 8,
                          'throatExtAngle': -3, 's1': .45, 's2': .2},
              'morph': {'morphTarget': 1, 'morphWidth': 150, 'morphHeight': 110,
                        'morphCorner': 15, 'morphFixed': .4},
              'mesh': {'quadrants': quadrants, 'angularSegments': 16, 'lengthSegments': 16,
                       'surface_fit': 'interpolate', 'throat_res_mm': 4, 'mouth_res_mm': 10,
                       'max_triangles': 100000, 'allow_large_mesh': True}}
    mesh = meshio.read(build_from_config(config, tmp_path / 'solve.msh').mesh_path)
    triangles, tags = _triangles_and_physical_tags(mesh)
    wall_nodes = np.unique(triangles[tags == PhysicalGroup.RIGID_WALL])
    assert wall_nodes.size > 0
    points = mesh.points[np.unique(triangles[tags == PhysicalGroup.RIGID_WALL])] * 1000
    assert points.shape == (wall_nodes.size, 3)
    step, _ = write_step_from_config(config, tmp_path / 'full.step')
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.occ.importShapes(str(step), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getEntities(2)
        assert surfaces
        for point in points:
            distance = min(np.linalg.norm(
                np.asarray(gmsh.model.getClosestPoint(2, tag, point.tolist())[0]) - point)
                for _, tag in surfaces)
            assert distance < .1
    finally:
        gmsh.finalize()


@pytest.mark.parametrize('mode', ['freestanding', 'infinite-baffle'])
@pytest.mark.parametrize('quadrants', ['12', '14'])
def test_active_half_model_sector_rims_close_for_mesh_and_step(mode, quadrants, tmp_path):
    config = {'mode': mode, 'scale': .48,
              'profile': {'L': 160, 'r0': 10, 'a': 40, 'a0': 7.9, 's1': .5, 's2': .2,
                          'throatExtLength': 12},
              'morph': {'morphTarget': 1, 'morphWidth': 450, 'morphHeight': 400,
                        'morphCorner': 20, 'morphFixed': .4},
              'mesh': {'quadrants': quadrants, 'angularSegments': 16, 'lengthSegments': 16,
                       'throat_res_mm': 8, 'mouth_res_mm': 16, 'surface_fit': 'interpolate',
                       'allow_large_mesh': True, 'max_triangles': 100000}}
    assert build_from_config(config, tmp_path / 'half.msh').n_triangles > 0
    assert write_step_from_config(config, tmp_path / 'full.step')[0].is_file()


@pytest.mark.parametrize('tmax', [.5, .625, .731, .875, 1])
def test_other_rosse_mouth_endpoint_contacts_are_refused(tmax):
    import math
    from hornlab_mesher.profile_formulas import calculate_rosse

    config = copy.deepcopy(ENDPOINT)
    params = {k: v for k, v in config['profile'].items()
              if k not in ('formula', 'throatExtLength', 'throatExtAngle')}
    x, radius = calculate_rosse(tmax, 0, params)
    assert x < 0
    config['profile'].update(tmax=tmax, throatExtLength=-x,
                             throatExtAngle=math.degrees(math.atan((radius-params['r0'])/x)))
    with pytest.raises(ConfigError, match='self-contact|self-intersection'):
        resolve_geometry(config)


@pytest.mark.parametrize('key', ['s1', 's2'])
def test_text_precedence_cannot_hide_a_coefficient_expression(key):
    for text in [f'OSSE = {{\nL = 160\n}}\n{key} = 0*p',
                 f'OSSE = {{\nL = 160\ns1 = .5\ns2 = .2\n}}\n{key} = 0*p',
                 f'R-OSSE = {{\nR = 200\n}}\nOSSE = {{\nL = 160\n{key} = 0*p\n}}']:
        with pytest.raises(ConfigError, match='per-azimuth throat stretch is not supported yet'):
            parse_text_config(text)


@pytest.mark.parametrize('path', PUBLIC)
@pytest.mark.parametrize('key', ['s1', 's2'])
def test_native_precedence_cannot_hide_a_coefficient_expression(path, key, tmp_path):
    config = {'mode': 'bare', 'profile': {'s1': .5, 's2': .2},
              'parameters': {key: '0*p'}}
    with pytest.raises(ConfigError, match='per-azimuth throat stretch is not supported yet'):
        consume(path, config, tmp_path)


@pytest.mark.parametrize('family', ['FREEFORM', 'ICW'])
@pytest.mark.parametrize('section', ['top', 'profile', 'parameters', 'parameters-only', 'secondary-profile', 'mesh', 'morph', 'source', 'output', 'enclosure', 'gcurve'])
@pytest.mark.parametrize('key', ['s1', 's2'])
@pytest.mark.parametrize('value', [None, 0, .5, '0*p'])
def test_foreign_family_refuses_every_supplied_stretch_entry(family, section, key, value):
    # Fully valid controls ensure removing the guard actually builds a profile.
    profiles = {
        'FREEFORM': {
            'profileH': {'points': [[0, 12.7], [40, 40], [80, 75]],
                         'throatAngleDeg': 15.5, 'mouthAngleDeg': 45},
            'profileV': {'points': [[0, 12.7], [40, 35], [80, 60]],
                         'throatAngleDeg': 15.5, 'mouthAngleDeg': 35}},
        'ICW': {'r0': 12.7, 'a0': 15.5, 'icw_coeffs': [0, 0, 0, 0, 0, 0], 'icw_S': 80},
    }
    config = {'formula': family, 'mode': 'bare', 'profile': profiles[family]}
    build_geometry_params(config)
    if section == 'top':
        config[key] = value
    elif section == 'parameters-only':
        config['parameters'] = config.pop('profile')
        config['parameters'][key] = value
    elif section == 'secondary-profile':
        config['parameters'] = config.pop('profile')
        config['profile'] = {key: value, **profiles[family]}
    else:
        config.setdefault(section, {})[key] = value
    with pytest.raises(ConfigError, match='coefficient keys|shape keys'):
        build_geometry_params(config)


@pytest.mark.parametrize('quadrants', ['1', '12', '14'])
def test_stretched_sector_fit_is_required_for_solve_vs_full_step(quadrants, tmp_path, monkeypatch):
    from dataclasses import replace
    import gmsh
    import meshio
    from hornlab_mesher import config_builder as cb
    from hornlab_mesher.mesher import _triangles_and_physical_tags
    from hornlab_mesher.tags import PhysicalGroup

    # A stretched superellipse at Scale=10 exposes the 0.1 mm failure on
    # every reduced domain, including quadrant 1. Densities scale with it.
    config = {'mode': 'bare', 'scale': 10,
              'profile': {'L': 80, 'r0': 12.7, 'a': 35, 'a0': 5, 's1': .5, 's2': .2},
              'morph': {'morphTarget': 3, 'morphWidth': 180, 'morphHeight': 110,
                        'morphExponent': 4, 'morphFixed': .4},
              'mesh': {'quadrants': quadrants, 'angularSegments': 16, 'lengthSegments': 16,
                       'throat_res_mm': 80, 'mouth_res_mm': 160, 'surface_fit': 'interpolate',
                       'allow_large_mesh': True, 'max_triangles': 100000}}

    def wall_distances(label):
        mesh = meshio.read(cb.build_from_config(config, tmp_path / f'{label}.msh').mesh_path)
        triangles, tags = _triangles_and_physical_tags(mesh)
        wall_nodes = np.unique(triangles[tags == PhysicalGroup.RIGID_WALL])
        assert wall_nodes.size > 0
        points = mesh.points[wall_nodes] * 1000
        assert points.shape == (wall_nodes.size, 3)
        # STEP is the full model even when the solve request is reduced.
        step, _ = write_step_from_config(config, tmp_path / f'{label}.step')
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber('General.Terminal', 0)
            gmsh.model.occ.importShapes(str(step), highestDimOnly=False)
            gmsh.model.occ.synchronize()
            surfaces = gmsh.model.getEntities(2)
            assert surfaces
            distances = np.array([
                min(np.linalg.norm(
                    np.asarray(gmsh.model.getClosestPoint(2, tag, point.tolist())[0]) - point)
                    for _, tag in surfaces)
                for point in points])
            assert distances.shape == (wall_nodes.size,)
            return distances
        finally:
            gmsh.finalize()

    assert wall_distances('active').max() < .1
    original = cb.resolve_geometry

    def disable_fit(*args, **kwargs):
        resolved = original(*args, **kwargs)
        return replace(resolved, geometry=replace(resolved.geometry, quadrant_patch_fit=False))

    monkeypatch.setattr(cb, 'resolve_geometry', disable_fit)
    # Negative control: the same geometry exceeds the feature criterion when
    # both consumers use the previous (unmatched) open/full fitting routes.
    assert wall_distances('disabled').max() > .1
