"""Deviation estimates leave no post-processing views in a shared session."""
import json
from pathlib import Path

import gmsh

from hornlab_mesher.config_builder import estimate_from_config


def test_native_estimate_removes_its_views_and_preserves_existing_views():
    gmsh.initialize(interruptible=False)
    try:
        retained = gmsh.view.add('retained')
        current = json.loads(Path('tests/fixtures/tessellation_report/OSSE-ADAPTER-CONTROLS.json').read_text())
        current['mesh']['surface_deviation_mm'] = 0.001
        estimate_from_config(current)
        assert list(gmsh.view.getTags()) == [retained]
        estimate_from_config(current)
        assert list(gmsh.view.getTags()) == [retained]
    finally:
        gmsh.finalize()


def test_looser_controls_request_reuses_existing_stronger_certificate(tmp_path):
    from hornlab_mesher.config_builder import build_from_config

    current = json.loads(Path('tests/fixtures/tessellation_report/OSSE-ADAPTER-CONTROLS.json').read_text())
    before = build_from_config(current, tmp_path / 'off.msh', allow_large_mesh=True)
    current['mesh']['surface_deviation_mm'] = 0.3
    after = build_from_config(current, tmp_path / 'on.msh', allow_large_mesh=True)
    assert before.mesh_path.read_bytes() == after.mesh_path.read_bytes()
    report = after.metadata['surface_deviation']
    assert report['peak_is_upper_bound'] is True
    assert report['peak_mm'] <= 0.01
    assert report['sampled_triangle_count'] == 0
    assert report['certified_triangle_count'] == report['triangle_count'] == after.n_triangles
