"""Explicit sizing bounds and optional observations survive accuracy opt-in."""
import json
from pathlib import Path

import gmsh
import pytest

from hornlab_mesher import mesher, tessellation
from hornlab_mesher.config_builder import build_from_config, resolve_geometry
from hornlab_mesher.geometry import MeshDensity


def bare_config():
    current = json.loads(Path('tests/fixtures/tessellation_report/OS01-default.json').read_text())
    current['mode'] = 'bare'
    current['mesh'].update(wallThickness=0, throat_res_mm=10, mouth_res_mm=10,
                           rear_res_mm=10)
    return current


@pytest.mark.parametrize('deviation', [None, 0.3])
def test_explicit_minimum_survives_optional_deviation(monkeypatch, deviation):
    current = bare_config()
    resolved = resolve_geometry(current)
    density = MeshDensity(throat_res_mm=10, mouth_res_mm=10, rear_res_mm=10,
                          min_size_mm=5, max_size_mm=20, surface_deviation_mm=deviation)
    observed = []
    configure = mesher.configure_density

    def inspect(geometry, sizing):
        configure(geometry, sizing)
        observed.append((gmsh.option.getNumber('Mesh.MeshSizeMin'),
                         geometry.metadata.get('surface_deviation_sizing')))

    def stop(geometry):
        raise mesher._DensityEstimateReady(geometry.metadata)

    monkeypatch.setattr(mesher, 'configure_density', inspect)
    with pytest.raises(mesher._DensityEstimateReady):
        mesher.build_mesh_with_info(resolved.geometry, density, after_density=stop)
    assert observed[0][0] == 5
    if deviation is not None:
        assert observed[0][1]['curve_size_mm'] >= 5


@pytest.mark.parametrize('deviation', [None, 0.3])
def test_failed_observation_preserves_mesh_and_never_retries_fit(monkeypatch, tmp_path, deviation):
    from hornlab_mesher import config_builder

    current = bare_config()
    current['mesh']['surface_deviation_mm'] = deviation
    before = build_from_config(current, tmp_path / 'before.msh', allow_large_mesh=True)
    calls = []
    build = config_builder.build_mesh_with_info

    def counted(*args, **kwargs):
        calls.append(1)
        return build(*args, **kwargs)

    def unavailable(*args, **kwargs):
        raise RuntimeError('projection unavailable')

    monkeypatch.setattr(config_builder, 'build_mesh_with_info', counted)
    monkeypatch.setattr(tessellation, 'measure_surface_deviation', unavailable)
    after = build_from_config(current, tmp_path / 'after.msh', allow_large_mesh=True)
    assert calls == [1]
    assert before.mesh_path.read_bytes() == after.mesh_path.read_bytes()
    assert after.metadata['surface_deviation']['available'] is False
    assert 'peak_mm' not in after.metadata['surface_deviation']


def test_report_keeps_the_achieved_gap_when_an_explicit_floor_limits_accuracy(tmp_path):
    current = bare_config()
    geometry = resolve_geometry(current).geometry
    density = MeshDensity(throat_res_mm=10, mouth_res_mm=10, rear_res_mm=10,
                          min_size_mm=5, max_size_mm=20, surface_deviation_mm=0.001)
    _, info = mesher.build_mesh_with_info(
        geometry, density, tmp_path / 'limited.msh', surface_deviation_triangle_budget=0,
    )
    assert info.metadata['surface_deviation_mm'] == 0.001
    assert info.metadata['surface_deviation']['peak_mm'] > 1.5 * 0.001
