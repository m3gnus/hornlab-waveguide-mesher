"""Public built boundaries must follow the effective analytic morph outline."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import re
from pathlib import Path

import numpy as np
import pytest

from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.cad import write_step
from hornlab_mesher.mesher import build_mesh_with_info


def _config(formula, corner, *, morph=True):
    profile = (
        {'L_mm': 120., 'r0_mm': 12.7, 'a0_deg': 15.5, 'a_deg': 55., 'k': 1., 'q': .995}
        if formula == 'OSSE' else
        {'L_mm': 120., 'r0_mm': 12.7, 'a0_deg': 18., 'R_mm': 110., 'termination': 'flat_baffle'}
    )
    config = {'formula': formula, 'mode': 'bare', 'profile': profile,
              'source': {'source_shape': 0}, 'mesh': {'allow_large_mesh': True}}
    if morph:
        config['morph'] = {'morphTarget': 1, 'morphWidth': 320., 'morphHeight': 240.,
                           'morphCorner': corner, 'morphAllowShrinkage': 1}
    return config


def _outline_error(points, corner):
    q = np.abs(points[:, :2]) - np.array([160. - corner, 120. - corner])
    return np.abs(np.linalg.norm(np.maximum(q, 0.), axis=1)
                  + np.minimum(np.max(q, axis=1), 0.) - corner)


@pytest.mark.parametrize('formula', ['OSSE', 'ICW'])
@pytest.mark.parametrize('corner', [0., 10., 30., 60.], ids=['sharp', 'r10', 'r30', 'r60'])
def test_default_built_mouth_follows_design(tmp_path, formula, corner):
    """Check the surface between fit stations and the written solve boundary."""
    gmsh = pytest.importorskip('gmsh')
    import meshio
    from scipy.spatial import cKDTree

    resolved = resolve_geometry(_config(formula, corner))
    geometry = resolved.geometry
    z = float(geometry.inner_points[0, -1, 2])
    step = tmp_path / 'mouth.step'
    write_step(geometry, step)
    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber('General.Terminal', 0)
    try:
        gmsh.model.occ.importShapes(str(step))
        gmsh.model.occ.synchronize()
        anchors = cKDTree(geometry.inner_points[:, -1, :2])
        mouth_curves = []
        for _, tag in gmsh.model.getEntities(1):
            lo, hi = gmsh.model.getParametrizationBounds(1, tag)
            points = np.asarray(gmsh.model.getValue(
                1, tag, np.linspace(lo[0], hi[0], 16385))).reshape(-1, 3)
            if np.max(np.abs(points[:, 2] - z)) < 1e-6 and np.max(
                anchors.query(points[[0, -1], :2])[0]
            ) < 1e-6:
                mouth_curves.append(points)
        assert mouth_curves
        step_error = float(_outline_error(np.concatenate(mouth_curves), corner).max())
    finally:
        gmsh.finalize()
    path, _ = build_mesh_with_info(geometry, resolved.density, tmp_path / 'mouth.msh',
                                   scale_to_metres=False)
    points = np.asarray(meshio.read(path).points)
    mouth = points[np.abs(points[:, 2] - z) < 1e-6]
    assert len(mouth)
    node_error = float(_outline_error(mouth, corner).max())
    bound = .6 if corner == 0 else .3
    assert step_error <= bound, f'STEP mouth error {step_error:.6f} mm'
    assert node_error <= bound, f'solve mouth node error {node_error:.6f} mm'


@pytest.mark.parametrize('formula', ['OSSE', 'ICW'])
def test_disabled_morph_preserves_output_bytes(tmp_path, formula):
    """Explicitly disabling the morph must match the original absent setting."""
    configs = [_config(formula, 0., morph=False), _config(formula, 0., morph=False)]
    configs[1]['morph'] = {'morphTarget': 0}
    meshes, steps = [], []
    for i, config in enumerate(configs):
        resolved = resolve_geometry(config)
        path, _ = build_mesh_with_info(resolved.geometry, resolved.density,
                                       tmp_path / f'round-{i}.msh')
        meshes.append(hashlib.sha256(Path(path).read_bytes()).hexdigest())
        path = tmp_path / f'round-{i}.step'
        # Fresh processes reset OCC's process-global product-name counter.
        subprocess.run([sys.executable, '-c',
                        'import json,sys; from hornlab_mesher.config_builder import resolve_geometry; '
                        'from hornlab_mesher.cad import write_step; '
                        'write_step(resolve_geometry(json.loads(sys.argv[1])).geometry,sys.argv[2])',
                        json.dumps(config), str(path)], check=True)
        # OCC inserts the wall clock in FILE_NAME; no geometric bytes may differ.
        data = re.sub(rb"'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d'", b"'TIME'", path.read_bytes())
        steps.append(hashlib.sha256(data).hexdigest())
    assert meshes[0] == meshes[1]
    assert steps[0] == steps[1]


@pytest.mark.parametrize('corner,quadrants,main_triangles', [
    (0., '1234', 3450), (0., '1', 860),
    (10., '1234', 4748), (10., '1', 1204),
])
def test_freestanding_fit_preserves_user_mesh_cost(tmp_path, corner, quadrants,
                                                  main_triangles):
    """Main counts measured at the same default mm resolution and source."""
    config = _config('OSSE', corner)
    config['mode'] = 'freestanding'
    config['mesh'].update(wall_thickness_mm=6., quadrants=quadrants)
    config['morph']['morphAllowShrinkage'] = 0
    resolved = resolve_geometry(config)
    _, info = build_mesh_with_info(resolved.geometry, resolved.density,
                                   tmp_path / 'shell.msh')
    assert abs(info.n_triangles / main_triangles - 1.) < .05


def test_nonconvergent_morph_names_fit_reason(monkeypatch):
    from hornlab_mesher import mouth_fit
    from hornlab_mesher.config_parser import ConfigError

    monkeypatch.setattr(mouth_fit, '_MAX_PASSES', 1)
    config = _config('OSSE', 0.)
    config['profile'].update(s1=.45, s2=.2)
    with pytest.raises(ConfigError, match='morphed mouth cubic fit.*outline tolerance.*limit'):
        resolve_geometry(config)


@pytest.mark.parametrize('case', ['approx-sharp', 'approx-rounded', 'circle', 'circle-rectangle'])
def test_exempt_fit_output_keeps_original_sampling(tmp_path, monkeypatch, case):
    """Round mouths and an explicit approximating fit retain output bytes."""
    from hornlab_mesher import mouth_fit

    config = _config('OSSE', 0. if case == 'approx-sharp' else 10.)
    if case.startswith('approx-'):
        config['mesh']['surface_fit'] = 'approximate'
    else:
        config['morph'].update(morphTarget=2 if case == 'circle' else 1,
                               morphWidth=360., morphHeight=360., morphCorner=180.)
    refine = mouth_fit.refine_mouth_grid
    meshes, points = [], []
    for enabled in [False, True]:
        monkeypatch.setattr(mouth_fit, 'refine_mouth_grid', refine if enabled
                            else lambda params, grid: (grid, {}))
        resolved = resolve_geometry(config)
        points.append(resolved.geometry.inner_points.tobytes())
        path, _ = build_mesh_with_info(resolved.geometry, resolved.density,
                                       tmp_path / f'approximate-{enabled}.msh')
        meshes.append(Path(path).read_bytes())
    assert points[0] == points[1]
    assert meshes[0] == meshes[1]
