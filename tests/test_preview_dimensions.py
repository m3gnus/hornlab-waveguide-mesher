"""C2 dimensions are full canonical extents, not render-mesh statistics."""
import copy
import json
import dataclasses
import time
from statistics import median

import numpy as np
import pytest

from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry
from hornlab_mesher.preview.dimensions import canonical_dimensions, DIMENSIONS_SAMPLING
import hornlab_mesher.preview.api as preview_api
from test_offset_envelope import _grooved_config
from hornlab_mesher.profile_sampling import build_point_grid_arrays
from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry
from test_preview_api import (
    OSSE_FREESTANDING, ROSSE_ENCLOSURE, FREEFORM_FREESTANDING, ICW_FLAT_BAFFLE,
)


@pytest.mark.parametrize('config', [OSSE_FREESTANDING, ROSSE_ENCLOSURE,
                                    FREEFORM_FREESTANDING, ICW_FLAT_BAFFLE, _grooved_config()])
def test_dimensions_do_not_follow_lod_or_visibility(config):
    coarse = build_preview_geometry(config, PreviewOptionsV1(lod='coarse'))
    fine = build_preview_geometry(config, PreviewOptionsV1(lod='fine', include_inner=False,
        include_outer=False, include_enclosure=False, include_source_cap=False,
        include_rear_cap=False, include_curvature=False))
    assert coarse.metadata['dimensions_mm'] is not None
    assert fine.metadata['dimensions_mm'] is not None
    tolerance = coarse.metadata['dimensions_sampling']['tolerance_mm']
    for key, value in coarse.metadata['dimensions_mm'].items():
        assert value == pytest.approx(fine.metadata['dimensions_mm'][key], abs=tolerance)
    assert coarse.metadata['dimensions_sampling'] == fine.metadata['dimensions_sampling']
    json.dumps(coarse.metadata, allow_nan=False)


@pytest.mark.parametrize('quadrants', ['1', '12', '14', '1234'])
@pytest.mark.parametrize('config', [OSSE_FREESTANDING, ROSSE_ENCLOSURE,
                                    FREEFORM_FREESTANDING, ICW_FLAT_BAFFLE])
def test_origin_and_symmetry_do_not_change_full_dimensions(config, quadrants):
    changed = copy.deepcopy(config)
    changed.setdefault('mesh', {}).update(quadrants=quadrants, vertical_offset_mm=87.3)
    # A shifted reduced source cap has an existing orientation limitation.
    # Material dimensions exclude the acoustic source cap.
    options = PreviewOptionsV1(lod='coarse', include_source_cap=False)
    baseline = build_preview_geometry(config, options)
    preview = build_preview_geometry(changed, options)
    assert preview.metadata['dimensions_mm'] is not None
    for key, value in baseline.metadata['dimensions_mm'].items():
        assert preview.metadata['dimensions_mm'][key] == pytest.approx(value, abs=1.0)


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
    assert dims['horn_overall'][2] == pytest.approx(np.ptp(points[:, :, 2]), abs=1.0)
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


def test_readout_cannot_be_mutated_or_reused_for_a_new_design():
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


CORPUS = [OSSE_FREESTANDING, ROSSE_ENCLOSURE, FREEFORM_FREESTANDING,
          ICW_FLAT_BAFFLE, _grooved_config()]


@pytest.mark.parametrize('config', CORPUS)
def test_family_dimensions_match_independent_full_model_surfaces(config):
    preview = build_preview_geometry(config, PreviewOptionsV1(lod='fine'))
    dims = preview.metadata['dimensions_mm']
    assert dims is not None
    by_role = {s.role: s for s in preview.surfaces}
    # Emitted grids are axial-major. The last ring is the actual terminating
    # opening even for a curve whose maximum axial excursion precedes it.
    n_phi = preview.metadata['actual_segment_counts']['horn_phi']
    mouth = by_role['horn.inner'].positions[-n_phi:]
    expected_mouth = np.ptp(mouth[:, :2], axis=0)
    horn = np.concatenate([s.positions for s in preview.surfaces
                           if not s.role.startswith('enclosure.') and s.role != 'source_cap'])
    assert dims['mouth_opening'] == pytest.approx(expected_mouth, abs=1.0)
    assert dims['horn_overall'] == pytest.approx(np.ptp(horn, axis=0), abs=1.0)
    if 'enclosure_overall' in dims:
        enclosure = np.concatenate([s.positions for s in preview.surfaces
                                    if s.role.startswith('enclosure.')])
        assert dims['enclosure_overall'] == pytest.approx(np.ptp(enclosure, axis=0), abs=1.0)
    # Additional first-principles checks guard against matching absent or
    # incorrectly selected surfaces on the two previously vacuous families.
    if config['formula'] == 'FREEFORM':
        assert dims['mouth_opening'] == pytest.approx([320, 220])
        assert dims['horn_overall'][2] == pytest.approx(120 + 6)
    if config['formula'] == 'ICW':
        assert dims['mouth_opening'] == pytest.approx([2 * 110, 2 * 110])
        assert dims['horn_overall'][2] == pytest.approx(120 + 6)


@pytest.mark.parametrize('config', CORPUS)
def test_normal_offset_approximation_is_within_published_canonical_tolerance(config):
    # Independent oracle: the solve/CAD control geometry, with wall repair.
    geometry = resolve_geometry(config).geometry
    points = [geometry.inner_points.reshape(-1, 3)]
    if geometry.outer_points is not None:
        points.append(geometry.outer_points.reshape(-1, 3))
        rear = geometry.outer_points[:, 0].copy()
        rear[:, 2] = geometry.inner_points[:, 0, 2].mean() - geometry.wall_thickness_mm
        points.append(rear)
    expected = np.ptp(np.concatenate(points), axis=0)
    for lod in ['coarse', 'fine']:
        preview = build_preview_geometry(config, PreviewOptionsV1(lod=lod))
        assert preview.metadata['dimensions_mm'] is not None
        assert preview.metadata['dimensions_mm']['horn_overall'] == pytest.approx(
            expected, abs=DIMENSIONS_SAMPLING['tolerance_mm'])


def _assert_preview_equal(left, right):
    # All surface fields/arrays and existing metadata must remain unchanged.
    assert len(left.surfaces) == len(right.surfaces)
    for a, b in zip(left.surfaces, right.surfaces):
        for field in dataclasses.fields(a):
            actual, expected = getattr(a, field.name), getattr(b, field.name)
            if isinstance(actual, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
            else:
                assert actual == expected
    def existing(metadata):
        return {k: v for k, v in metadata.items() if k not in
                {'dimensions_mm', 'dimensions_error', 'dimensions_sampling', 'timings_ms'}}
    assert existing(left.metadata) == existing(right.metadata)


@pytest.mark.parametrize('lod', ['coarse', 'fine'])
@pytest.mark.parametrize('config', CORPUS)
def test_measurement_does_not_change_any_surface_or_existing_metadata(config, lod, monkeypatch):
    measured = build_preview_geometry(config, PreviewOptionsV1(lod=lod))
    monkeypatch.setattr(preview_api, 'canonical_dimensions', lambda *a, **kw: None)
    disabled = build_preview_geometry(config, PreviewOptionsV1(lod=lod))
    _assert_preview_equal(measured, disabled)


@pytest.mark.parametrize('kind', ['tiny-density', 'non-mapping-mesh', 'generic'])
def test_measurement_failure_cannot_break_a_renderable_draft(kind, monkeypatch):
    config = copy.deepcopy(OSSE_FREESTANDING)
    if kind == 'tiny-density':
        config['mesh'].update(throat_res_mm=1e-308, mouth_res_mm=1e-308, rear_res_mm=1e-308)
    elif kind == 'non-mapping-mesh':
        config['mesh'] = 6
    else:
        def fail(*args, **kwargs):
            raise RuntimeError('generic dimension failure')
        monkeypatch.setattr(preview_api, 'canonical_dimensions', fail)
    measured = build_preview_geometry(config, PreviewOptionsV1(lod='coarse'))
    assert measured.surfaces
    if kind == 'tiny-density':
        # No acoustic-density fitting remains in the measurement path, so
        # this draft now has dimensions rather than overflowing.
        assert measured.metadata['dimensions_mm'] is not None
    else:
        assert measured.metadata['dimensions_mm'] is None
        assert measured.metadata['dimensions_error']
    monkeypatch.setattr(preview_api, 'canonical_dimensions', lambda *a, **kw: None)
    disabled = build_preview_geometry(config, PreviewOptionsV1(lod='coarse'))
    _assert_preview_equal(measured, disabled)


@pytest.mark.parametrize('exception', [OverflowError, TypeError, RuntimeError])
def test_all_measurement_exceptions_mark_dimensions_unavailable(exception, monkeypatch):
    def fail(*args, **kwargs):
        raise exception('measurement failed')
    monkeypatch.setattr(preview_api, 'canonical_dimensions', fail)
    preview = build_preview_geometry(OSSE_FREESTANDING, PreviewOptionsV1(lod='coarse'))
    assert preview.surfaces
    assert preview.metadata['dimensions_mm'] is None
    assert preview.metadata['dimensions_error'] == 'measurement failed'


@pytest.mark.parametrize('exception', [KeyboardInterrupt, SystemExit])
def test_measurement_does_not_swallow_process_control_exceptions(exception, monkeypatch):
    def fail(*args, **kwargs):
        raise exception()
    monkeypatch.setattr(preview_api, 'canonical_dimensions', fail)
    with pytest.raises(exception):
        build_preview_geometry(OSSE_FREESTANDING, PreviewOptionsV1(lod='coarse'))


@pytest.mark.parametrize('config', CORPUS)
def test_changing_design_measurement_has_small_frame_overhead(config, monkeypatch):
    # Interleaved paired requests with NEW revisions, no warm dimension cache.
    # Median pairs and a generous 35% ratio tolerate scheduling/load noise;
    # the separate benchmark reports the stricter 10% / 10 ms production goal.
    measured, disabled = [], []
    real = preview_api.canonical_dimensions
    options = PreviewOptionsV1(lod='coarse')
    for revision in range(5):
        changed = copy.deepcopy(config)
        changed.setdefault('scale', 1.0)
        changed['scale'] += revision * 0.001
        for enabled in ([True, False] if revision % 2 == 0 else [False, True]):
            monkeypatch.setattr(preview_api, 'canonical_dimensions', real if enabled else lambda *a, **kw: None)
            start = time.perf_counter()
            preview = build_preview_geometry(changed, options)
            elapsed = time.perf_counter() - start
            (measured if enabled else disabled).append(elapsed)
            if enabled:
                assert preview.metadata['dimensions_mm'] is not None
    assert median(measured) <= 1.35 * median(disabled)


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_measurements_are_isolated(value, monkeypatch):
    monkeypatch.setattr(preview_api, 'canonical_dimensions',
                        lambda *a, **kw: {'horn_overall': [value, 1, 1]})
    preview = build_preview_geometry(OSSE_FREESTANDING, PreviewOptionsV1(lod='coarse'))
    assert preview.surfaces
    assert preview.metadata['dimensions_mm'] is None
    assert 'finite' in preview.metadata['dimensions_error']


def test_dimension_work_does_not_repeat_outer_wall_warning(caplog):
    config = _grooved_config()
    preview = build_preview_geometry(config, PreviewOptionsV1(lod='coarse'))
    before = list(caplog.records)
    canonical_dimensions(config)
    assert not [record for record in caplog.records[len(before):]
                if 'outer wall' in record.getMessage()]
    assert preview.metadata['dimensions_mm'] is not None
