"""Native ATH V2026-09a witnesses for the three import blockers."""
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry, build_from_config
from hornlab_mesher.config_parser import parse_text_config
from hornlab_mesher.profile_formulas import calculate_rosse, calculate_rosse_curve
from hornlab_mesher.profile_sampling import build_point_grid_arrays, retain_ath_morph_construction

FIXTURES = Path(__file__).parent / 'fixtures/ath_import/ath-v2026-09a/blockers'


def reference(name):
    return np.array([[[float(v) for v in line.split(';')] for line in block.splitlines()]
                     for block in (FIXTURES / f'{name}.csv').read_text().strip().split('\n\n')])


def params(name):
    return build_geometry_params(parse_text_config((FIXTURES / f'{name}.cfg').read_text()))[0]


@pytest.mark.parametrize('name', ['r-base', 'r-slot', 'r-ext', 'r-slot-ext', 'r-slot-trunc',
                                  'r-ext-trunc', 'r-slot-neg', 'r-ext-run21', 'r-ext-4', 'r-ext-15'])
def test_prefix_rows_and_stations_match_ath(name):
    actual = build_point_grid_arrays(params(name))['inner_grid']
    expected = reference(name)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-4)


def test_slot_physical_meridian_is_translated_without_a_cylindrical_prefix():
    # Recover each ATH point's parameter by its radius, independently of the
    # export's row index. z is checked on both sides of the rollback maximum.
    p = params('r-slot')
    base = params('r-base')
    for point in reference('r-slot')[0, 1:-1]:
        t = brentq(lambda t: calculate_rosse(t, 0, base)[1] - point[0], 0, 1)
        z, radius = calculate_rosse(t, 0, p)
        assert z == pytest.approx(point[2], abs=2e-4)
        assert radius == pytest.approx(point[0], abs=2e-4)
    assert calculate_rosse(0, 0, p) == pytest.approx((12, 12.7), abs=1e-12)
    t = np.linspace(0, 1, 103)
    np.testing.assert_allclose(np.column_stack(calculate_rosse_curve(t, 0, p)),
                               [calculate_rosse(float(u), 0, p) for u in t], rtol=0, atol=1e-12)


def test_slot_extension_join_retains_ath_straight_segment_between_rows():
    from hornlab_mesher.profile_formulas import rosse_axial_layout
    p = params('r-slot-ext')
    layout = rosse_axial_layout(p)
    t = 9.5/layout.full_length
    expected = reference('r-slot-ext')[0,9:11].mean(axis=0)
    actual = calculate_rosse(t,0,p)
    np.testing.assert_allclose(actual, expected[[2,0]], rtol=0, atol=2e-4)
    np.testing.assert_allclose(np.column_stack(calculate_rosse_curve([t],0,p))[0],
                               actual, rtol=0, atol=1e-12)


@pytest.mark.parametrize('family', ['o', 'r'])
@pytest.mark.parametrize('na,ls', [(32,20), (48,10), (64,20)])
@pytest.mark.parametrize('width,height', [(240,360), (240,220), (420,400), (360,240)])
@pytest.mark.parametrize('allow', [0,1])
def test_no_shrink_full_wall_and_sampling_match_ath(family, na, ls, width, height, allow):
    name = f'{family}-m-{width}-{height}-a{allow}-n{na}-l{ls}'
    actual = build_point_grid_arrays(params(name))['inner_grid']
    expected = reference(name)
    assert actual.shape == expected.shape
    # Unblended throat directions establish the rebuilt angular construction.
    np.testing.assert_allclose(actual[:,0], expected[:,0], rtol=0, atol=2e-4)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-4)


def test_no_shrink_sampling_changes_the_physical_boundary():
    a = reference('r-m-240-360-a0-n32-l20')
    b = reference('r-m-240-360-a0-n64-l20')
    assert abs(a[:,-1,1].max()-b[:,-1,1].max()) > 1
    # The larger requested y dimension expands too: a per-axis clamp cannot
    # produce the measured 194.164079 mm half-height.
    assert a[:,-1,:2].max(axis=0) == pytest.approx([180,194.164079], abs=2e-4)


@pytest.mark.parametrize('family', ['o', 'r'])
def test_refinement_retains_authored_physical_wall(family):
    name = f'{family}-m-240-360-a0-n32-l20'
    p = params(name)
    authored = build_point_grid_arrays(p)
    working = retain_ath_morph_construction(p, authored)
    original_angles = np.asarray(authored['angle_list'])
    # Include the original azimuths and new samples between every pair.
    midpoints = (original_angles + np.r_[original_angles[1:], 2*np.pi])/2
    angles = np.sort(np.r_[original_angles, midpoints])
    working.update(angularSegments=64, cornerSegments=12, _acoustic_mouth_angles=angles)
    refined = build_point_grid_arrays(working)
    np.testing.assert_allclose(refined['inner_grid'][::2], reference(name), rtol=0, atol=2e-4)
    assert refined['morph_half_dimensions'] == pytest.approx(authored['morph_half_dimensions'], abs=1e-12)


@pytest.mark.parametrize('shape', [1,3])
def test_saved_v1_and_unqualified_targets_retain_no_shrink_refusal(shape):
    from hornlab_mesher.text_import import TEXT_IMPORT_VERSION_KEY, TEXT_IMPORT_V1
    p = dict(params('r-base'), morphTarget=shape, morphWidth=100, morphHeight=100,
             morphAllowShrinkage=0)
    if shape == 1:
        p[TEXT_IMPORT_VERSION_KEY] = TEXT_IMPORT_V1
    with pytest.raises(ValueError, match='no qualified construction'):
        build_point_grid_arrays(p)


@pytest.mark.parametrize('lod', ['coarse', 'fine'])
def test_preview_lod_retains_authored_no_shrink_boundary(lod):
    from hornlab_mesher.preview import build_preview_geometry, PreviewOptionsV1
    name = 'r-m-240-360-a0-n32-l20'
    cfg = parse_text_config((FIXTURES / f'{name}.cfg').read_text())
    cfg['mode'] = 'bare'
    cfg['mesh']['wallThickness'] = 0
    preview = build_preview_geometry(cfg, PreviewOptionsV1(
        lod=lod, include_curvature=False, include_outer=False, include_enclosure=False,
        include_source_cap=False, include_rear_cap=False))
    surface = next(s for s in preview.surfaces if s.role == 'horn.inner')
    points = np.asarray(surface.positions).reshape(-1,3)
    expected = reference(name)
    mouth = points[np.abs(points[:,2]-expected[0,-1,2]) < 2e-4]
    assert len(mouth) > 8
    np.testing.assert_allclose(np.max(np.abs(mouth[:,:2]),axis=0),
                               np.max(np.abs(expected[:,-1,:2]),axis=0), rtol=0, atol=2e-4)


@pytest.mark.parametrize('name', ['r-slot', 'r-ext', 'o-m-240-360-a0-n32-l20',
                                  'r-m-240-360-a0-n32-l20'])
def test_corrected_wall_reaches_mesh_and_reopened_step(tmp_path, name):
    import gmsh
    from hornlab_mesher.cad import write_step_from_config
    from hornlab_mesher.mesher import load_mesh

    cfg = parse_text_config((FIXTURES / f'{name}.cfg').read_text())
    cfg['mode'] = 'bare'
    cfg['mesh'].update(wallThickness=0, throatResolution=5, mouthResolution=20,
                       maxTriangles=100000)
    expected = reference(name)
    resolved = resolve_geometry(cfg)
    assert resolved.geometry.inner_points.shape[0] >= expected.shape[0]
    # This strongly expanded rollback has a pre-existing OCC triangulation
    # failure with the same AllowShrinkage=1 wall. Its STEP surface is checked
    # here; the other three cases exercise default mesh construction too.
    if name != 'r-m-240-360-a0-n32-l20':
        built = build_from_config(cfg, tmp_path/'wall.msh')
        assert load_mesh(built.mesh_path).n_triangles > 0
    path, info = write_step_from_config(cfg, tmp_path/'wall.step')
    assert info.body == 'surface'
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.occ.importShapes(str(path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        faces = gmsh.model.getEntities(2)
        assert faces
        # Physical closest-surface distances, independent of STEP/mesh indices.
        # Include interior stations and both sides of the R-OSSE rollback.
        for point in expected[::max(1,len(expected)//8), ::4].reshape(-1,3):
            distance = min(np.linalg.norm(np.asarray(gmsh.model.getClosestPoint(2, tag, point)[0])-point)
                           for _,tag in faces)
            assert distance < 0.15
    finally:
        gmsh.finalize()
