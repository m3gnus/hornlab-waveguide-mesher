"""Canonical readouts use independently resolved solve/CAD control geometry."""
import copy
import dataclasses
import json

import numpy as np
import pytest

from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry
import hornlab_mesher.preview.api as preview_api
import hornlab_mesher.preview.dimensions as measurement
from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.builders.enclosure import enclosure_box_bounds
from test_offset_envelope import _grooved_config
from test_freeform_morph import _config_with_morph
from test_preview_morph import ROUNDED_RECT_MORPH
from test_preview_api import OSSE_FREESTANDING, ROSSE_ENCLOSURE, FREEFORM_FREESTANDING, ICW_FLAT_BAFFLE


def freeform_morph():
    config = _config_with_morph(1)
    config['morph']['morphCorner'] = 15
    return config


def folded_morph():
    config = _grooved_config()
    config['morph'] = {'morphTarget': 1, 'morphWidth': 300, 'morphHeight': 300,
                       'morphCorner': 30, 'morphFixed': 0.6}
    return config


CORPUS = [OSSE_FREESTANDING, ROSSE_ENCLOSURE, FREEFORM_FREESTANDING,
          ICW_FLAT_BAFFLE, _grooved_config(), freeform_morph(), folded_morph(), ROUNDED_RECT_MORPH]


@pytest.fixture(autouse=True)
def empty_measurement_cache():
    measurement._DIMENSIONS_CACHE.clear()
    yield
    measurement._DIMENSIONS_CACHE.clear()


def independently_resolved(config):
    full = copy.deepcopy(config)
    full.setdefault('mesh', {}).update(quadrants='1234', vertical_offset_mm=0)
    full['mesh'].pop('verticalOffset', None)
    geometry = resolve_geometry(full).geometry
    inner = geometry.inner_points
    material = [inner.reshape(-1, 3)]
    if geometry.outer_points is not None:
        material.append(geometry.outer_points.reshape(-1, 3))
        rear = geometry.outer_points[:, 0].copy()
        rear[:, 2] = inner[:, 0, 2].mean() - geometry.wall_thickness_mm
        material.append(rear)
    expected = {'mouth_opening': np.ptp(inner[:, -1, :2], axis=0),
                'horn_overall': np.ptp(np.concatenate(material), axis=0)}
    if geometry.enclosure is not None:
        bounds = enclosure_box_bounds(inner, geometry.enclosure, closed=True)
        expected['enclosure_overall'] = np.array([bounds['bx1'] - bounds['bx0'],
            bounds['by1'] - bounds['by0'], bounds['z_front'] - bounds['z_back']])
    return expected


@pytest.mark.parametrize('config', CORPUS)
@pytest.mark.parametrize('scale', [1, 2])
def test_family_dimensions_match_independent_solve_cad_geometry(config, scale):
    config = copy.deepcopy(config)
    config['scale'] = scale
    expected = independently_resolved(config)
    numbers = []
    for quadrant in ['1234', '12', '1']:
        config.setdefault('mesh', {}).update(quadrants=quadrant)
        preview = build_preview_geometry(config, PreviewOptionsV1(lod='fine', include_source_cap=False))
        dims = preview.metadata['dimensions_mm']
        assert dims is not None
        assert preview.metadata['dimensions_status'] == 'current'
        assert set(dims) == set(expected)
        for key, values in expected.items():
            assert dims[key] == pytest.approx(values, abs=1e-9)
        numbers.append(dims)
    assert numbers[0] == numbers[1] == numbers[2]
    if config['formula'] == 'FREEFORM' and 'morph' in config:
        # The main FREEFORM dispatch fix uses the drawn H/V profiles instead
        # of R-OSSE defaults (the old 300 x 300 expectation). With implicit
        # morph axes, the terminal radii 30/20 stay 30/20; rounding the corners
        # changes neither axis. This bare horn spans z=0..100 with no wall.
        assert numbers[0]['mouth_opening'] == [60 * scale, 40 * scale]
        assert numbers[0]['horn_overall'] == [60 * scale, 40 * scale, 100 * scale]


@pytest.mark.parametrize('config', CORPUS)
def test_coarse_is_pending_until_this_exact_design_is_measured(config, monkeypatch):
    real = measurement.canonical_dimensions
    calls = []
    def measure(config):
        calls.append(copy.deepcopy(config))
        return real(config)
    monkeypatch.setattr(measurement, 'canonical_dimensions', measure)
    coarse = build_preview_geometry(config, PreviewOptionsV1(lod='coarse', include_source_cap=False))
    assert coarse.metadata['dimensions_mm'] is None
    assert coarse.metadata['dimensions_status'] == 'pending'
    assert 'dimensions_error' not in coarse.metadata
    assert calls == []
    fine = build_preview_geometry(config, PreviewOptionsV1(lod='fine', include_source_cap=False))
    assert fine.metadata['dimensions_mm'] is not None
    assert len(calls) == 1
    hidden = PreviewOptionsV1(lod='inspection', include_inner=False, include_outer=False,
        include_enclosure=False, include_source_cap=False, include_rear_cap=False, include_curvature=False)
    for options in [hidden, PreviewOptionsV1(lod='coarse', include_source_cap=False), PreviewOptionsV1(lod='fine', include_source_cap=False)]:
        preview = build_preview_geometry(copy.deepcopy(config), options)
        assert preview.metadata['dimensions_mm'] == fine.metadata['dimensions_mm']
        assert preview.metadata['dimensions_status'] == 'current'
    assert len(calls) == 1
    assert fine.metadata['dimensions_sampling'] == {'method': 'resolved-design-geometry', 'lod_independent': True}
    json.dumps(fine.metadata, allow_nan=False)


@pytest.mark.parametrize('section,key,value', [('profile', 'L_mm', 240), ('mesh', 'wall_thickness_mm', 8),
    ('mesh', 'angular_segments', 200), ('morph', 'morphCorner', 15), (None, 'scale', 2)])
def test_full_config_is_cache_identity(section, key, value):
    config = copy.deepcopy(OSSE_FREESTANDING)
    first = measurement.dimension_metadata(config, 'fine')
    assert first['dimensions_status'] == 'current'
    if section is None:
        config[key] = value
    else:
        config.setdefault(section, {})[key] = value
    assert measurement.dimension_metadata(config, 'coarse') == {'dimensions_mm': None, 'dimensions_status': 'pending'}


def test_cached_values_are_owned_by_each_frame():
    first = measurement.dimension_metadata(OSSE_FREESTANDING, 'fine')
    expected = copy.deepcopy(first)
    first['dimensions_mm']['horn_overall'][0] = 1
    assert measurement.dimension_metadata(OSSE_FREESTANDING, 'coarse') == expected


def test_origin_does_not_change_enclosure_rounding():
    config = copy.deepcopy(ROSSE_ENCLOSURE)
    original = measurement.canonical_dimensions(config)
    config.setdefault('mesh', {})['vertical_offset_mm'] = 87.3
    assert measurement.canonical_dimensions(config) == original


def test_enclosure_spacing_and_depth_clamp():
    config = copy.deepcopy(ROSSE_ENCLOSURE)
    config['enclosure'].update(space_l_mm=11, space_r_mm=23, space_b_mm=17, space_t_mm=29, depth_mm=1)
    dims = measurement.canonical_dimensions(config)
    assert dims['enclosure_overall'][:2] == [334, 346]
    assert dims['enclosure_overall'][2] > 1


def test_rollback_depth_is_maximum_excursion():
    config = copy.deepcopy(ROSSE_ENCLOSURE)
    config.pop('enclosure')
    config['mode'] = 'bare'
    config['mesh'] = {'wall_thickness_mm': 0}
    config['profile'].update(m=0.5, b=0, tmax=1)
    dims = measurement.canonical_dimensions(config)
    points = resolve_geometry(config).geometry.inner_points
    assert dims['horn_overall'][2] == pytest.approx(np.ptp(points[:, :, 2]))
    assert dims['horn_overall'][2] > points[:, -1, 2].max() + 10


def _assert_preview_equal(left, right):
    assert len(left.surfaces) == len(right.surfaces)
    for a, b in zip(left.surfaces, right.surfaces):
        for field in dataclasses.fields(a):
            actual, expected = getattr(a, field.name), getattr(b, field.name)
            if isinstance(actual, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
            else:
                assert actual == expected
    added = {'dimensions_mm', 'dimensions_requested_mm', 'dimensions_error', 'dimensions_sampling', 'dimensions_status', 'timings_ms'}
    assert {k: v for k, v in left.metadata.items() if k not in added} == {k: v for k, v in right.metadata.items() if k not in added}


@pytest.mark.parametrize('config', CORPUS)
@pytest.mark.parametrize('lod', ['coarse', 'fine'])
def test_measurement_preserves_all_surfaces_and_existing_metadata(config, lod, monkeypatch):
    options = PreviewOptionsV1(lod=lod, include_source_cap=False)
    measured = build_preview_geometry(config, options)
    monkeypatch.setattr(preview_api, 'dimension_metadata', lambda *a: {})
    disabled = build_preview_geometry(config, options)
    _assert_preview_equal(measured, disabled)


@pytest.mark.parametrize('exception', [OverflowError, TypeError, RuntimeError])
def test_any_measurement_exception_is_isolated_and_cached(exception, monkeypatch):
    def fail(*args):
        raise exception('measurement failed')
    monkeypatch.setattr(measurement, 'canonical_dimensions', fail)
    options = PreviewOptionsV1(lod='fine')
    preview = build_preview_geometry(OSSE_FREESTANDING, options)
    assert preview.metadata['dimensions_mm'] is None
    assert preview.metadata['dimensions_status'] == 'unavailable'
    assert preview.metadata['dimensions_error'] == 'measurement failed'
    monkeypatch.setattr(preview_api, 'dimension_metadata', lambda *a: {})
    _assert_preview_equal(preview, build_preview_geometry(OSSE_FREESTANDING, options))
    assert measurement.dimension_metadata(OSSE_FREESTANDING, 'coarse')['dimensions_status'] == 'unavailable'


@pytest.mark.parametrize('kind', ['mesh', 'density', 'wall'])
def test_unresolvable_draft_keeps_base_preview(kind, monkeypatch):
    config = copy.deepcopy(OSSE_FREESTANDING)
    if kind == 'mesh':
        config['mesh'] = 6
    elif kind == 'density':
        config['mesh'].update(throat_res_mm=1e-308, mouth_res_mm=1e-308, rear_res_mm=1e-308)
    else:
        config['mesh']['wall_thickness_mm'] = 0
    options = PreviewOptionsV1(lod='fine')
    preview = build_preview_geometry(config, options)
    assert preview.surfaces
    assert preview.metadata['dimensions_mm'] is None
    assert preview.metadata['dimensions_status'] == 'unavailable'
    assert preview.metadata['dimensions_error']
    monkeypatch.setattr(preview_api, 'dimension_metadata', lambda *a: {})
    _assert_preview_equal(preview, build_preview_geometry(config, options))


@pytest.mark.parametrize('exception', [KeyboardInterrupt, SystemExit])
def test_process_control_exceptions_propagate(exception, monkeypatch):
    def fail(*args):
        raise exception()
    monkeypatch.setattr(measurement, 'canonical_dimensions', fail)
    with pytest.raises(exception):
        build_preview_geometry(OSSE_FREESTANDING, PreviewOptionsV1(lod='fine'))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), None])
def test_invalid_measurement_result_is_isolated(value, monkeypatch):
    monkeypatch.setattr(measurement, 'canonical_dimensions', lambda *a: None if value is None else {'horn_overall': [value, 1, 1]})
    preview = build_preview_geometry(OSSE_FREESTANDING, PreviewOptionsV1(lod='fine'))
    assert preview.surfaces
    assert preview.metadata['dimensions_status'] == 'unavailable'
    assert preview.metadata['dimensions_mm'] is None
    assert preview.metadata['dimensions_error']


def test_measurement_does_not_repeat_outer_wall_warning(caplog):
    build_preview_geometry(_grooved_config(), PreviewOptionsV1(lod='coarse'))
    before = len(caplog.records)
    measurement.canonical_dimensions(_grooved_config())
    assert not [record for record in caplog.records[before:] if 'outer wall' in record.getMessage()]
