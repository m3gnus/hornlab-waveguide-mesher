"""Sketched design diaphragms use the existing canonical contour and box."""
import copy

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
