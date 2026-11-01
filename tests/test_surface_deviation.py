"""Opt-in sizing refines curved walls without changing saved configurations."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hornlab_mesher.config_builder import (
    build_from_config,
    estimate_from_config,
    resolve_geometry,
)


def config():
    return json.loads(Path('tests/fixtures/tessellation_report/RO01-default.json').read_text())


def test_absent_budget_stays_off_and_explicit_budget_reaches_density():
    old = config()
    assert resolve_geometry(old).density.surface_deviation_mm is None
    enabled = copy.deepcopy(old)
    enabled['mesh']['surface_deviation_mm'] = 0.3
    assert resolve_geometry(enabled).density.surface_deviation_mm == 0.3


@pytest.mark.parametrize('value', [True, 0, -0.3, float('inf'), float('nan')])
def test_invalid_deviation_is_rejected(value):
    current = config()
    current['mesh']['surface_deviation_mm'] = value
    with pytest.raises(ValueError, match='surface_deviation_mm'):
        resolve_geometry(current)


def test_budget_refines_rollback_and_estimate_observes_it(tmp_path):
    current = config()
    current['mesh']['wallThickness'] = 0
    current['mode'] = 'bare'
    before = build_from_config(current, tmp_path / 'off.msh', allow_large_mesh=True, surface_deviation_triangle_budget=0)
    current['mesh']['surface_deviation_mm'] = 0.3
    after = build_from_config(current, tmp_path / 'on.msh', allow_large_mesh=True, surface_deviation_triangle_budget=0)
    assert after.n_triangles > before.n_triangles
    assert after.metadata['meshTriangleEstimate'] > before.metadata['meshTriangleEstimate']
    assert after.metadata['surface_deviation']['peak_mm'] <= 0.45
    assert after.metadata['surface_deviation_mm'] == 0.3
    assert before.metadata['surface_deviation_mm'] is None
    assert before.metadata['surface_deviation']['peak_mm'] > 0.45


def test_live_estimate_uses_the_density_pass_without_generating_facets(monkeypatch):
    import gmsh

    def generate(*_args):
        raise AssertionError('an estimate must stop before mesh generation')

    monkeypatch.setattr(gmsh.model.mesh, 'generate', generate)
    current = config()
    before = estimate_from_config(current)
    current['mesh']['surface_deviation_mm'] = 0.3
    after = estimate_from_config(current)
    assert after['meshTriangleEstimate'] > before['meshTriangleEstimate'] > 0
    assert after['surface_deviation_mm'] == 0.3


@pytest.mark.parametrize('name', ['LOOKUP-default', 'RO01-default', 'OS02-morph-enclosure', 'FF02-hv-stations-angles', 'AT07-sc-0p7'])
def test_budget_covers_curved_morph_enclosure_trim_boundaries(tmp_path, name):
    current = json.loads(Path(f'tests/fixtures/tessellation_report/{name}.json').read_text())
    current['mesh']['surface_deviation_mm'] = 0.3
    result = build_from_config(current, tmp_path / 'morph.msh', allow_large_mesh=True,
                               surface_deviation_triangle_budget=0)
    if result.metadata['surface_deviation']['peak_mm'] > 0.45:
        print(json.dumps(result.metadata['surface_deviation']), flush=True)
    assert result.metadata['surface_deviation']['peak_mm'] <= 0.45, result.metadata['surface_deviation']
