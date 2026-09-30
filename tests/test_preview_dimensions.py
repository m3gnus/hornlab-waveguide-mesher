"""C2 dimensions are full canonical extents, not render-mesh statistics."""
import copy
import json

import numpy as np
import pytest

from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry
from hornlab_mesher.preview.dimensions import canonical_dimensions
from hornlab_mesher.profile_sampling import build_point_grid_arrays
from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry
from test_preview_api import (
    OSSE_FREESTANDING, ROSSE_ENCLOSURE, FREEFORM_FREESTANDING, ICW_FLAT_BAFFLE,
)


@pytest.mark.parametrize('config', [OSSE_FREESTANDING, ROSSE_ENCLOSURE,
                                    FREEFORM_FREESTANDING, ICW_FLAT_BAFFLE])
def test_dimensions_do_not_follow_lod_or_visibility(config):
    coarse = build_preview_geometry(config, PreviewOptionsV1(lod='coarse'))
    fine = build_preview_geometry(config, PreviewOptionsV1(lod='fine', include_inner=False,
        include_outer=False, include_enclosure=False, include_source_cap=False,
        include_rear_cap=False, include_curvature=False))
    assert coarse.metadata['dimensions_mm'] == fine.metadata['dimensions_mm']
    assert coarse.metadata['dimensions_sampling'] == fine.metadata['dimensions_sampling']
    json.dumps(coarse.metadata, allow_nan=False)


@pytest.mark.parametrize('quadrants', ['1', '12', '14', '1234'])
@pytest.mark.parametrize('config', [OSSE_FREESTANDING, ROSSE_ENCLOSURE])
def test_origin_and_symmetry_do_not_change_full_dimensions(config, quadrants):
    changed = copy.deepcopy(config)
    changed.setdefault('mesh', {}).update(quadrants=quadrants, vertical_offset_mm=87.3)
    assert canonical_dimensions(changed) == canonical_dimensions(config)


def test_mouth_is_terminating_opening_and_horn_includes_wall_and_rear_plane():
    config = copy.deepcopy(OSSE_FREESTANDING)
    config['profile'].update(s=0, q=1)
    dims = canonical_dimensions(config)
    no_wall = copy.deepcopy(config)
    no_wall['mesh']['wall_thickness_mm'] = 0
    no_wall['mode'] = 'bare'
    bare = canonical_dimensions(no_wall)
    assert dims['mouth_opening'] == bare['mouth_opening']
    assert dims['horn_overall'][0] > bare['horn_overall'][0]
    assert dims['horn_overall'][1] > bare['horn_overall'][1]
    assert dims['horn_overall'][2] >= 126
    assert 'enclosure_overall' not in dims


def test_rollback_depth_is_maximum_excursion_not_final_z():
    config = copy.deepcopy(ROSSE_ENCLOSURE)
    config.pop('enclosure')
    config['mode'] = 'bare'
    config['mesh'] = {'wall_thickness_mm': 0}
    config['profile'].update(m=0.5, b=0, tmax=1)
    dims = canonical_dimensions(config)
    params, _, _ = build_geometry_params(config)
    grid = build_point_grid_arrays(params)
    points = resolve_geometry(config).geometry.inner_points
    assert dims['horn_overall'][2] == pytest.approx(np.ptp(points[:, :, 2]))
    assert dims['horn_overall'][2] > points[:, -1, 2].max() + 10
    assert dims['mouth_opening'] == pytest.approx([300, 300])


def test_enclosure_extents_include_asymmetric_spacing_and_depth_clamp():
    config = copy.deepcopy(ROSSE_ENCLOSURE)
    config['enclosure'].update(space_l_mm=11, space_r_mm=23,
                               space_b_mm=17, space_t_mm=29, depth_mm=1)
    dims = canonical_dimensions(config)
    assert dims['enclosure_overall'][:2] == pytest.approx([334, 346])
    assert dims['enclosure_overall'][2] > 1


def test_dimensions_use_real_canonical_sampling_for_expression_extrema():
    config = copy.deepcopy(OSSE_FREESTANDING)
    config['mode'] = 'bare'
    config['mesh'].update(wall_thickness_mm=0, angular_segments=512, length_segments=192, topology_mode='legacy')
    config['profile']['a_deg'] = '55+20*sin(128*p)**2'
    params, _, _ = build_geometry_params(config)
    real = build_point_grid_arrays(params)['inner_grid']
    dims = canonical_dimensions(config)
    assert dims['mouth_opening'] == pytest.approx(np.ptp(real[:, -1, :2], axis=0))
    assert dims['mouth_opening'][0] > 800


def test_cached_readout_cannot_be_mutated_or_reused_for_a_new_design():
    config = copy.deepcopy(OSSE_FREESTANDING)
    original = canonical_dimensions(config)
    modified_readout = canonical_dimensions(config)
    modified_readout['horn_overall'][0] = 1
    assert canonical_dimensions(config) == original
    config['profile']['L_mm'] = 240
    changed = canonical_dimensions(config)
    assert changed['horn_overall'] != original['horn_overall']
    assert changed['mouth_opening'] != original['mouth_opening']


def test_preview_keeps_renderable_draft_but_marks_canonical_dimensions_unavailable():
    config = copy.deepcopy(OSSE_FREESTANDING)
    config['mesh']['wall_thickness_mm'] = 0
    preview = build_preview_geometry(config, PreviewOptionsV1(lod='coarse'))
    assert preview.surfaces
    assert preview.metadata['dimensions_mm'] is None
    assert 'freestanding mode requires' in preview.metadata['dimensions_error']
