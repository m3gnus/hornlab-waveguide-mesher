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
    bound = (.6 if corner == 0 else .15) + 1e-5
    # OCC construction/serialization uses 1e-8 mm tolerances. Allow 10 nm
    # for numeric evaluation and serialization, not additional design error.
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


def _built_outline_error(config, monkeypatch):
    """Measure the requested-domain OCC edges against the accepted target."""
    import gmsh
    from scipy.spatial import cKDTree
    from hornlab_mesher import mouth_fit
    from hornlab_mesher.mesher import _dispatch_builder

    with monkeypatch.context() as before:
        before.setattr(mouth_fit, 'refine_mouth_grid', lambda p, g: (g, {}))
        accepted = resolve_geometry(config).geometry.inner_points
    a, b = np.abs(accepted[:, -1, :2]).max(axis=0)
    radius = min(config['morph']['morphCorner']*config.get('scale', 1), a, b)
    resolved = resolve_geometry(config)
    points = resolved.geometry.inner_points
    np.testing.assert_allclose(np.abs(points[:, -1, :2]).max(axis=0), [a, b], atol=1e-6, rtol=0)
    z = points[0, -1, 2]
    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber('General.Terminal', 0)
    try:
        built = _dispatch_builder(resolved.geometry)
        gmsh.model.occ.synchronize()
        anchors = cKDTree(points[:, -1, :2])
        curves = {tag for dim, tag in gmsh.model.getBoundary(
            [(2, k) for k in built.mesh_surface_groups['inner']],
            combined=False, oriented=False) if dim == 1}
        pieces = []
        for tag in curves:
            lo, hi = gmsh.model.getParametrizationBounds(1, tag)
            ends = np.asarray(gmsh.model.getValue(1, tag, [lo[0], hi[0]])).reshape(-1, 3)
            if np.abs(ends[:, 2]-z).max() > 1e-6 or anchors.query(ends[:, :2])[0].max() > 1e-6:
                continue
            pts = np.asarray(gmsh.model.getValue(
                1, tag, np.linspace(lo[0], hi[0], 32769))).reshape(-1, 3)
            if np.abs(pts[:, 2]-z).max() < 1e-6:
                pieces.append(pts)
        assert pieces
        xy = np.abs(np.concatenate(pieces)[:, :2]) - [a-radius, b-radius]
        error = np.abs(np.linalg.norm(np.maximum(xy, 0), axis=1)
                       + np.minimum(xy.max(axis=1), 0) - radius).max()
    finally:
        gmsh.finalize()
    assert error <= (.6 if radius == 0 else .15) + 1e-5, f'{error:.9f} mm'
    return resolved


@pytest.mark.parametrize('formula,quadrants,width,height', [
    ('OSSE', '1234', 360, 210), ('OSSE', '1', 360, 210),
    ('OSSE', '12', 0, 210), ('OSSE', '14', 360, 0),
    ('R-OSSE', '1', 0, 0), ('LOOKUP', '14', 360, 0),
])
def test_composed_built_outlines(monkeypatch, formula, quadrants, width, height):
    config = _config('OSSE', 23.)
    config['formula'] = formula
    if formula == 'R-OSSE':
        config['profile'] = {'R': 110, 'r0': 12.7, 'a0': 12.7, 'a': 40,
                             'k': 1, 'r': .37, 'm': 1.2, 'b': 0, 'q': .995, 'tmax': 1}
    elif formula == 'LOOKUP':
        config['profile'] = {'lookupProfile': [[0, 12.7], [30, 28], [80, 85], [120, 150]]}
    if formula in ('OSSE', 'R-OSSE'):
        config['profile'].update(s1=.7, s2=.3)
    config['scale'] = 1.6
    config['mesh']['quadrants'] = quadrants
    config['morph'].update(morphWidth=width, morphHeight=height, morphFixed=.35)
    _built_outline_error(config, monkeypatch)


def test_both_implicit_target_stays_accepted(monkeypatch):
    config = {'formula': 'OSSE', 'mode': 'bare',
              'profile': {'L': 310, 'r0': 12.7,
                          'a': '45 - 16*cos(p)^2 - 40*sin(p)^16', 'a0': 15.5,
                          'k': 2, 'q': .993, 'n': 5, 's': .8, 'h': 0},
              'morph': {'morphTarget': 1, 'morphWidth': 0, 'morphHeight': 0,
                        'morphCorner': 100, 'morphRate': 3},
              'mesh': {'allow_large_mesh': True, 'throatResolution': 6,
                       'mouthResolution': 15, 'rearResolution': 40}}
    _built_outline_error(config, monkeypatch)


def _emitted_wall_facets(config, tmp_path, monkeypatch):
    import gmsh
    import meshio
    from scipy.spatial import cKDTree
    import hornlab_mesher.mesher as mesher

    captured = {}
    dispatch = mesher._dispatch_builder
    def capture(geometry):
        built = dispatch(geometry)
        captured['built'] = built
        return built
    monkeypatch.setattr(mesher, '_dispatch_builder', capture)
    resolved = resolve_geometry(config)
    gmsh.initialize(interruptible=False)
    try:
        path, info = build_mesh_with_info(resolved.geometry, resolved.density,
            tmp_path / 'facets.msh', scale_to_metres=False)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(xyz).reshape(-1, 3)
        lookup = {int(tag): i for i, tag in enumerate(tags)}
        raw, roles = [], []
        for role, surfaces in captured['built'].mesh_surface_groups.items():
            for tag in surfaces:
                types, _, sets = gmsh.model.mesh.getElements(2, tag)
                for typ, ns in zip(types, sets, strict=True):
                    assert typ == 2
                    chunk = xyz[np.array([lookup[int(k)] for k in ns]).reshape(-1, 3)]
                    raw.extend(chunk)
                    roles.extend([role]*len(chunk))
        raw, roles = np.asarray(raw), np.asarray(roles)
        mesh = meshio.read(path)
        emitted = mesh.points[mesh.cells_dict['triangle']]
        movement, idx = cKDTree(raw.mean(axis=1)).query(emitted.mean(axis=1))
        assert movement.max() < .01
        facets = {role: emitted[roles[idx] == role] for role in ['inner', 'outer']}
    finally:
        gmsh.finalize()
    return info, facets


def _strict_crossing_count(inner, outer):
    """Independent segment/triangle test; excludes coplanar/shared contacts."""
    from scipy.spatial import cKDTree
    centers = outer.mean(axis=1)
    tree = cKDTree(centers)
    radius = np.linalg.norm(outer-centers[:, None], axis=2).max()
    low, high = outer.min(axis=1), outer.max(axis=1)
    def segment_hits(p, q, triangle):
        e1, e2 = triangle[1]-triangle[0], triangle[2]-triangle[0]
        direction = q-p
        normal = np.cross(e1, e2)
        denominator = normal @ direction
        if abs(denominator) < 1e-10:
            return False
        alpha = (normal @ (triangle[0]-p))/denominator
        if not 1e-8 < alpha < 1-1e-8:
            return False
        uv = np.linalg.lstsq(np.stack([e1, e2], axis=1), p+alpha*direction-triangle[0], rcond=None)[0]
        return min(uv[0], uv[1], 1-uv.sum()) > 1e-8
    count = 0
    for triangle in inner:
        center = triangle.mean(axis=0)
        candidates = tree.query_ball_point(center, np.linalg.norm(triangle-center, axis=1).max()+radius+1e-8)
        for j in candidates:
            if np.any(high[j] < triangle.min(axis=0)) or np.any(low[j] > triangle.max(axis=0)):
                continue
            other = outer[j]
            if any(segment_hits(a[k], a[(k+1) % 3], b)
                   for a, b in [(triangle, other), (other, triangle)] for k in range(3)):
                count += 1
    return count


@pytest.mark.parametrize('quadrants,base_count,base_pairs', [('1', 219, 0), ('1234', 860, 4)])
def test_icw_emitted_facets_preserve_clearance(tmp_path, monkeypatch, quadrants, base_count, base_pairs):
    config = _config('ICW', 0.)
    config.update(mode='freestanding', scale=.48)
    config['mesh'].update(quadrants=quadrants, wall_thickness_mm=3.,
                          throatResolution=4., mouthResolution=26., rearResolution=15.)
    config['morph'].update(morphWidth=360., morphHeight=210., morphFixed=.35)
    info, facets = _emitted_wall_facets(config, tmp_path, monkeypatch)
    assert abs(info.n_triangles/base_count - 1) < .03
    assert _strict_crossing_count(facets['inner'], facets['outer']) <= base_pairs


def test_emitted_intersection_measurement_controls():
    a = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0]]])
    b = np.array([[[.25, .25, -1], [.25, .25, 1], [1, 1, 0]]])
    assert _strict_crossing_count(a, b) == 1
    assert _strict_crossing_count(a, b + [5, 0, 0]) == 0


def test_stretched_half_outline_checks_each_patch(monkeypatch):
    config = {'formula': 'OSSE', 'mode': 'bare', 'scale': .35,
              'profile': {'L': 310, 'r0': 12.7, 'a': 55, 'a0': 15.5, 'k': 2,
                          'q': .993, 'n': 5, 's': .8, 's1': 5, 's2': .2},
              'mesh': {'quadrants': '14', 'allow_large_mesh': True},
              'morph': {'morphTarget': 1, 'morphWidth': 0, 'morphHeight': 210,
                        'morphCorner': 60, 'morphFixed': .96, 'morphAllowShrinkage': 1}}
    _built_outline_error(config, monkeypatch)


@pytest.mark.parametrize('formula', ['OSSE', 'R-OSSE', 'LOOKUP'])
def test_composed_emitted_walls_do_not_cross(tmp_path, monkeypatch, formula):
    config = _config('OSSE', 23.)
    config.update(formula=formula, mode='freestanding', scale=1.6)
    if formula == 'R-OSSE':
        config['profile'] = {'R': 110, 'r0': 12.7, 'a0': 12.7, 'a': 40, 'k': 1,
                             'r': .37, 'm': 1.2, 'b': 0, 'q': .995, 'tmax': 1}
    elif formula == 'LOOKUP':
        config['profile'] = {'lookupProfile': [[0, 12.7], [30, 28], [80, 85], [120, 150]]}
    if formula in ('OSSE', 'R-OSSE'):
        config['profile'].update(s1=.7, s2=.3)
    config['mesh'].update(quadrants='1', wall_thickness_mm=3.)
    config['morph'].update(morphWidth=360., morphHeight=210., morphFixed=.35)
    _, facets = _emitted_wall_facets(config, tmp_path, monkeypatch)
    assert _strict_crossing_count(facets['inner'], facets['outer']) == 0
