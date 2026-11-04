"""Sketched design diaphragms use the existing canonical contour and box."""
import copy
import math

import numpy as np
import pytest

from hornlab_mesher import build_from_config
from hornlab_mesher.enclosure_woofers import SketchedShape, EnclosureWoofer
from hornlab_mesher.errors import UserRefusal
from hornlab_mesher.mesher import MesherError
from hornlab_mesher.normals import validate_orientation
from hornlab_mesher.source_contour import cone, dome
from test_enclosure_woofers import assembly_config, arrays


def document(contour, *, motion='axial'):
    return {'contour': contour.to_dict(), 'drive': {'channel_id': 'motor', 'motion': motion,
            'weights': {s.id: (-.25 if s.id == 'surround' else 1.) for s in contour.segments if s.role == 'moving'}}}


def woofer_document():
    return document(cone(30, 10, cap_radius_mm=10, cap_height_mm=3,
                         surround_width_mm=5, surround_depth_mm=-1, physical_source_id='lf-b'))


def test_sketched_woofer_cone_surround_is_closed_and_weighted(tmp_path):
    config = assembly_config()
    config['enclosure']['woofers'] = config['enclosure']['woofers'][:1]
    config['enclosure']['woofers'][0]['shape'] = {'kind': 'sketched', 'diaphragm': woofer_document()}
    result = build_from_config(config, tmp_path/'woofer.msh')
    _, points, triangles, tags, _ = arrays(result)
    assert validate_orientation(points, triangles, tags, require_watertight=True,
                                require_edge_consistency=True, require_positive_volume=True).watertight
    patches = result.metadata['assembly']['patches']
    assert {p['segment_id'] for p in patches} == {'dust-cap', 'cone', 'surround'}
    assert {p['patch_id']: p['weight'] for p in patches} == {
        'lf-b/dust-cap': 1., 'lf-b/cone': 1., 'lf-b/surround': -.25}
    assert all(p['motion'] == 'axial' and p['area_mm2'] > 0 for p in patches)
    assert set(result.metadata['source_tags']) == {'horn-throat', *(p['patch_id'] for p in patches)}


def test_sketched_hf_dome_with_plug_is_closed_and_tagged(tmp_path):
    config = assembly_config()
    config['enclosure'].pop('woofers')
    config['mesh']['max_triangles'] = 25000  # Includes certified passage refinement.
    config['hf_diaphragm'] = {
        'fit_wg0_rim': True, 'diaphragm': document(dome(12.7, 3, physical_source_id='horn-throat'), motion='normal'),
        'phase_plugs': [dict(id='plug', z0_mm=6, z1_mm=15, inner0_mm=0, outer0_mm=2,
                             inner1_mm=0, outer1_mm=3)],
    }
    result = build_from_config(config, tmp_path/'hf.msh')
    _, points, triangles, tags, _ = arrays(result)
    assert validate_orientation(points, triangles, tags, require_watertight=True,
                                require_edge_consistency=True, require_positive_volume=True).watertight
    assert result.metadata['source_tags'] == {'horn-throat/dome': 2}
    patch = result.metadata['assembly']['patches'][0]
    assert patch['weight'] == 1
    assert patch['max_geometry_residual_mm'] <= .025
    print('HF fitted WG0 maximum nodal contour residual (mm):', patch['max_geometry_residual_mm'])
    assert result.metadata['assembly']['phase_plugs'][0]['id'] == 'plug'
    quality = result.metadata['assembly']['passage']['quality']
    assert quality['certified_clearance_mm'] >= .8 * quality['minimum_clearance_mm']


@pytest.mark.parametrize('change,match', [('rim','rim'), ('front','front of baffle'), ('fold','increasing radius')])
def test_invalid_woofer_contour_is_actionable(change, match):
    raw = copy.deepcopy(woofer_document())
    if change == 'rim':
        radius = 36
    else:
        radius = 35
    if change == 'front':
        raw = document(dome(35, 3))
    if change == 'fold':
        raw['contour']['segments'][0].update(kind='line', center_mm=None)
        raw['contour']['points'][1]['r_mm'] = 40
    with pytest.raises(UserRefusal, match=match) as exc:
        EnclosureWoofer('lf-b', 0, -200, radius, SketchedShape.from_document(raw))
    assert exc.value.next_step


@pytest.mark.parametrize('bad', ['depth','clearance'])
def test_bad_hf_plug_refused(tmp_path, bad):
    config = assembly_config()
    config['enclosure'].pop('woofers')
    config['hf_diaphragm'] = {
        'fit_wg0_rim': True, 'diaphragm': document(dome(12.7, 3)),
        'phase_plugs': [dict(id='plug', z0_mm=2 if bad == 'depth' else 6, z1_mm=15,
                             inner0_mm=0, outer0_mm=20 if bad == 'clearance' else 2,
                             inner1_mm=0, outer1_mm=22 if bad == 'clearance' else 3)],
    }
    with pytest.raises((UserRefusal, MesherError), match='0.1 mm'):
        build_from_config(config, tmp_path/'bad.msh')



def test_hf_only_freestanding_body_is_closed(tmp_path):
    config = assembly_config()
    config.pop('enclosure')
    config['mode'] = 'freestanding'
    config['mesh']['wall_thickness_mm'] = 6
    config['hf_diaphragm'] = {'fit_wg0_rim': True, 'diaphragm': document(dome(12.7, 3))}
    result = build_from_config(config, tmp_path/'freestanding.msh')
    _, points, triangles, tags, _ = arrays(result)
    assert validate_orientation(points, triangles, tags, require_watertight=True,
                                require_edge_consistency=True, require_positive_volume=True).watertight


def test_sketched_woofer_depth_refused(tmp_path):
    config = assembly_config()
    config['enclosure']['woofers'][0]['shape'] = {'kind': 'sketched', 'diaphragm': document(
        cone(35, 300, cap_radius_mm=10, cap_height_mm=3))}
    with pytest.raises(MesherError, match='reaches enclosure back') as exc:
        build_from_config(config, tmp_path/'too-deep.msh')
    assert exc.value.user_refusal.next_step


def narrow_passage_config():
    config = assembly_config()
    config['enclosure'].pop('woofers')
    config['profile'].update(L_mm=24, a_deg=.01, a0_deg=0)
    config['mesh']['max_triangles'] = 200000
    config['hf_diaphragm'] = {
        'fit_wg0_rim': True, 'diaphragm': document(dome(12.7, 3)),
        'phase_plugs': [dict(id='vane', z0_mm=6, z1_mm=6.3,
                             inner0_mm=12.2, outer0_mm=12.54, inner1_mm=12.2, outer1_mm=12.54)],
    }
    return config


def test_narrow_passage_survives_tessellation_and_ordinary_sizing_is_refused(tmp_path, monkeypatch):
    from hornlab_mesher import design_passages
    from hornlab_mesher.passage_mesh import annulus_bounds

    config = narrow_passage_config()
    result = build_from_config(config, tmp_path/'narrow.msh')
    passage = result.metadata['assembly']['passage']
    quality = passage['quality']
    assert .1 < quality['minimum_clearance_mm'] < .5
    assert quality['certified_clearance_mm'] >= .8 * quality['minimum_clearance_mm']
    assert {c['id']: c['euler'] for c in quality['components']} == {'enclosure': 2, 'vane': 0}
    _, points, triangles, _, surfaces = arrays(result)
    print('Narrow passage CAD/certified clearance (mm), triangles:', quality['minimum_clearance_mm'],
          quality['certified_clearance_mm'], len(triangles))
    xyz = points[triangles]
    wall = np.isin(surfaces, passage['surfaces']['horn-wall'])
    wall &= (xyz[..., 2].min(axis=1) <= 6.3) & (xyz[..., 2].max(axis=1) >= 6)
    # Independent whole-triangle radial oracle: the bore stays outside the vane.
    # Only radial intrusion matters in this nearly cylindrical fixture.
    # Set each triangle on the reference z plane for the independent oracle.
    flat = xyz[wall].copy()
    flat[..., 2] = 0
    assert len(flat) > 0
    assert annulus_bounds(flat, (0, 0), 0, 12.54, math.inf).max() == 0

    # Recreate the reviewed path's ordinary "inner" sizing for the same CAD.
    # Both bodies remain individually closed; the shared facet certificate
    # must reject the coarse wall before publication.
    def ordinary_density(built, add_field):
        faces = [face for role, faces in built.metadata['assembly']['passage']['surfaces'].items()
                 if role != 'horn-wall' for face in faces]
        add_field('8', faces, [])
        return []
    certificate = design_passages.certify_design_passages
    intersections = []
    def capture_coarse(points, triangles, physical, surfaces, assembly):
        from hornlab_mesher.facet_fit import _FacetIndex
        groups = assembly['passage']['surfaces']
        wall = np.isin(surfaces, groups['horn-wall'])
        passive = np.isin(surfaces, [face for faces in assembly['passive_body_surfaces'].values() for face in faces])
        for component in (passive, ~passive):
            assert validate_orientation(points, triangles[component], physical[component], require_watertight=True,
                                        require_edge_consistency=True, require_positive_volume=True).watertight
        intersections.extend(_FacetIndex(points[triangles[passive]]).hits(points[triangles[wall]]))
        certificate(points, triangles, physical, surfaces, assembly)
    monkeypatch.setattr(design_passages, 'configure_passage_density', ordinary_density)
    monkeypatch.setattr(design_passages, 'certify_design_passages', capture_coarse)
    with pytest.raises(UserRefusal, match='passage (mesh|whole-facet)') as exc:
        build_from_config(config, tmp_path/'coarse.msh')
    assert 'Widen the passage' in exc.value.next_step
    assert not (tmp_path/'coarse.msh').exists()
    assert intersections, 'ordinary sizing must reproduce an actual collapsed CAD passage'


def test_phase_plugs_on_freeform_corner_wall_refuse_with_remedy(tmp_path):
    config = assembly_config('freeform-corners')
    config['enclosure'].pop('woofers')
    config['hf_diaphragm'] = narrow_passage_config()['hf_diaphragm']
    config['hf_diaphragm']['diaphragm'] = document(dome(12.7, 3))
    with pytest.raises((UserRefusal, MesherError), match='authored corners') as exc:
        build_from_config(config, tmp_path/'corners.msh')
    refusal = exc.value if isinstance(exc.value, UserRefusal) else exc.value.user_refusal
    assert 'Remove the phase plugs' in refusal.next_step
    assert not (tmp_path/'corners.msh').exists()
