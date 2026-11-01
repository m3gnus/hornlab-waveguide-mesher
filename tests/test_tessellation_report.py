"""Checks for the tessellation report's measurements and failure handling."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from scripts import report_tessellation as report


def test_meridian_distance_projects_to_adjacent_segments_and_keeps_offsets():
    reference = np.array([[0.0, 10.0], [10.0, 10.0], [20.0, 10.0]])
    points = np.array([[10.5, 5.0, 3.0], [9.75, 5.0, 17.0]])
    distance, segments = report._meridian_distance(points, reference, offset=5.0)
    np.testing.assert_allclose(distance, [0.5, 0.25])
    np.testing.assert_array_equal(segments, [0, 1])


def test_sampled_surface_distance_uses_facet_interior_not_vertex_distance():
    corners = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]])
    reference = {"corners": corners, "tree": cKDTree(corners.mean(axis=1))}
    distance = report._sampled_surface_distance(np.array([[3.0, 3.0, 0.2], [12.0, 0.0, 0.0]]), reference)
    np.testing.assert_allclose(distance, [0.2, 2.0])


def test_report_measures_fine_icw_and_selects_solve_wall():
    path = Path('tests/fixtures/tessellation_report/IC01-default.json')
    config = report._load(path)
    before = json.dumps(config, sort_keys=True)
    result = report._measure(config, solve=True, meridian=True)
    inner = result['preview_surfaces']['horn.inner']
    assert inner['triangles'] > 0
    assert 0 < inner['meridian_deviation']['max_mm'] < 0.05
    assert inner['meridian_deviation']['by_arc_decile'][0]['triangles'] > 0
    wall = result['solve']['acoustic_wall']
    assert wall is not None
    assert wall['triangles'] > 100
    assert wall['meridian_deviation']['max_mm'] > inner['meridian_deviation']['max_mm']
    assert result['solve']['acoustic_wall_selection']['unselected_triangles'] > 0
    assert json.dumps(config, sort_keys=True) == before


def test_report_marks_non_axisymmetric_reference_as_approximate():
    config = report._load(Path('tests/fixtures/ath_import/ath-v2026-09a/m3_rosse_morph.cfg'))
    result = report._measure(config, solve=False, meridian=True)
    inner = result['preview_surfaces']['horn.inner']
    assert inner['meridian_deviation'] is None
    assert 'not axisymmetric' in result['meridian_unavailable_reason']
    assert 'triangulation' in inner['sampled_surface_deviation']['method']
    assert inner['sampled_surface_deviation']['by_z_decile']


def test_report_handles_native_terminating_arc():
    config = report._load(Path('tests/fixtures/tessellation_report/OSSE-ARC-forward.json'))
    result = report._measure(config, solve=True, meridian=True)
    assert result['solve']['triangles'] > 0
    inner = result['preview_surfaces']['horn.inner']
    assert inner['triangles'] > 0
    assert inner['sampled_surface_deviation']['max_mm'] >= 0
    json.dumps(result, default=report._json_value, allow_nan=False)


def test_solve_refusal_preserves_preview_and_is_counted(tmp_path, monkeypatch):
    config = report._load(Path('tests/fixtures/tessellation_report/IC01-default.json'))

    def refuse(*_args, **_kwargs):
        raise ValueError('unsupported solve')

    monkeypatch.setattr(report, 'build_from_config', refuse)
    result = report._measure(config, solve=True, meridian=False)
    assert result['preview_surfaces']['horn.inner']['triangles'] > 0
    assert result['solve'] == {'status': 'error', 'error': 'ValueError: unsupported solve'}
    monkeypatch.setattr(report, '_measure', lambda *_args, **_kwargs: result)
    output = tmp_path / 'report.json'
    assert report.main(['tests/fixtures/tessellation_report/IC01-default.json', '--output', str(output)]) == 1
    data = json.loads(output.read_text())
    assert data['cases'][0]['status'] == 'partial'
    assert data['failures'] == 1


def test_numpy_metadata_is_json_encoded_without_allowing_nan(tmp_path, monkeypatch):
    monkeypatch.setattr(report, '_measure', lambda *_args, **_kwargs: {'value': np.float32(1.25), 'count': np.int64(7)})
    output = tmp_path / 'report.json'
    assert report.main(['tests/fixtures/tessellation_report/IC01-default.json', '--output', str(output)]) == 0
    assert json.loads(output.read_text())['cases'][0]['value'] == 1.25
    assert json.loads(output.read_text())['cases'][0]['count'] == 7
    with np.testing.assert_raises(ValueError):
        json.dumps({'value': np.float32(float('nan'))}, default=report._json_value, allow_nan=False)


def test_report_records_errors_and_returns_failure(tmp_path, monkeypatch):
    def fail(_path):
        raise ValueError('bad config')

    monkeypatch.setattr(report, '_load', fail)
    output = tmp_path / 'report.json'
    assert report.main(['missing.json', '--output', str(output)]) == 1
    result = json.loads(output.read_text())
    assert result['failures'] == 1
    assert result['cases'][0]['error'] == 'ValueError: bad config'
