"""FREEFORM corner walls qualified at production mesh settings.

``test_freeform_corner_wall`` proves the corner builder at a uniform fine
sizing. These builds use the public entry with a realistic density instead:
5 mm at the throat, 10-20 mm at the mouth, and the default angular, length and
rear settings.

Quality bound. Every corner build is compared with the same profile built
smooth (``without_corners``), which is what the mesher ships today. A corner
build must contain no sliver (``SLIVER_ANGLE_DEG``, the repo's own counting
threshold) and must keep at least half the smooth build's worst-element radius
ratio. Measured over these profiles the corner build kept 0.58-1.29x the smooth
minimum, so the 0.5 floor leaves margin while a single sliver (ratio about 0.03)
still fails it. Steep apexes (+-60-70 degree walls) meet the same bound, so the
corner path asserts quality rather than refusing an apex angle.
"""
from __future__ import annotations

import copy
import warnings
from pathlib import Path

import gmsh
import numpy as np
import pytest
from test_freeform_corner_wall import (  # noqa: F401
    angled_config,
    session,
    without_corners,
)

from hornlab_mesher import build_from_config, build_geometry_params
from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.corner_wall import (
    CornerWall,
    corner_geometry_class,
    cut_indices,
    mitred_outer_grid,
)
from hornlab_mesher.freeform import build_freeform_geometry
from hornlab_mesher.geometry import _MouthFittedPointGridHornGeometry
from hornlab_mesher.profile_sampling import build_point_grid_arrays
from hornlab_mesher.quality import SLIVER_ANGLE_DEG
from hornlab_mesher.tags import PhysicalGroup

CORNERS = 'FREEFORM corner z=0, 60, 120 mm'


def production(config, mouth_res_mm=20.):
    """Realistic density; drop the fine fixture's segment and size overrides."""
    config = copy.deepcopy(config)
    mesh = config['mesh']
    for key in ('angularSegments', 'lengthSegments', 'rear_res_mm', 'enc_front_res_mm',
                'enc_back_res_mm', 'max_triangles', 'allow_large_mesh'):
        mesh.pop(key, None)
    mesh.update(throat_res_mm=5., mouth_res_mm=mouth_res_mm, wall_thickness_mm=2.)
    return config


def build(config, path):
    with warnings.catch_warnings():
        # Differing H/V throat tangents only warn about the source-cap blend.
        warnings.simplefilter('ignore', UserWarning)
        return build_from_config(config, path)


def assert_quality(corner, smooth):
    shape = corner.metadata['quality']['element_shape']
    baseline = smooth.metadata['quality']['element_shape']
    assert shape['sliver_count'] == 0, shape['worst']
    assert shape['min_angle_deg'] >= SLIVER_ANGLE_DEG, shape['worst']
    assert shape['min_radius_ratio'] >= .5 * baseline['min_radius_ratio'], (shape, baseline)


def assert_creases(result):
    corners = result.metadata['freeformReport']['cornerWall']['corners']
    assert corners
    for corner in corners:
        for key in ('sharedEdgeIncidence', 'outerSharedEdgeIncidence'):
            assert all(edge['wallFaces'] == 2 for edge in corner.get(key, [])), corner
        assert len(corner['sharedEdgeIncidence']) > 0
        assert all(row['jumpErrorDeg'] <= .5 for row in corner['mesh'].values()), corner
    return corners


# -- Ignored controls are refused -------------------------------------------

@pytest.mark.parametrize('mesh,named', [
    ({'surface_fit': 'interpolate'}, "mesh.surface_fit='interpolate'"),
    ({'topology': 'legacy'}, "mesh.topology='legacy'"),
    ({'topology': 'legacy', 'preserve_grid': True}, "mesh.topology='legacy', mesh.preserve_grid"),
])
def test_corner_refuses_ignored_controls(mesh, named):
    config = angled_config(30, 15)
    config['mesh'].update(mesh)
    with pytest.raises(ValueError, match=f"{CORNERS}: .*do not support {named.replace('.', '[.]')};"):
        resolve_geometry(config)
    if mesh.get('surface_fit') != 'interpolate':
        # Without a corner the same controls keep their existing meaning.
        assert resolve_geometry(without_corners(config)).geometry.topology_mode == 'legacy'


@pytest.mark.parametrize('mesh', [
    {'surface_fit': 'auto', 'topology': 'acoustic', 'preserve_grid': False},
    {'surface_fit': 'approximate'},
])
def test_corner_accepts_defaults_and_approximate_fit(mesh):
    config = angled_config(30, 15)
    config['mesh'].update(mesh)
    geometry = resolve_geometry(config).geometry
    assert geometry.freeform_wall is not None
    assert geometry.surface_fit == 'approximate'


def test_corner_refuses_outer_clearance_probe():
    params, _, _ = build_geometry_params(angled_config(30, 15, 'freestanding'))
    grid = build_point_grid_arrays(params)
    kernel = build_freeform_geometry(params)
    cls = corner_geometry_class(_MouthFittedPointGridHornGeometry, CornerWall(kernel))
    with pytest.raises(ValueError, match=f'{CORNERS}: .*do not support outer_clearance_points_mm;'):
        cls(inner_points=grid['inner_grid'], outer_points=grid['outer_grid'],
            freeform_report=kernel.report(), wall_thickness_mm=2., source_shape=0,
            outer_clearance_points_mm=grid['outer_grid'])


# -- Production density ------------------------------------------------------

@pytest.mark.parametrize('mouth_res_mm', [10., 20.])
@pytest.mark.parametrize('mode', ['bare', 'freestanding'])
def test_moderate_concave_crease(tmp_path, mode, mouth_res_mm):
    config = production(angled_config(30, 15, mode), mouth_res_mm)
    corner = build(config, tmp_path / 'corner.msh')
    smooth = build(without_corners(config), tmp_path / 'smooth.msh')
    assert_creases(corner)
    assert_quality(corner, smooth)
    if mode == 'freestanding':
        # The mitred 2 mm wall holds on both sides of the production-grid crease.
        geometry = resolve_geometry(config).geometry
        points = geometry.inner_points
        cuts = geometry.freeform_wall.cuts(points)
        outer = mitred_outer_grid(points, 2., full_circle=True, corner_z=cuts)
        j = cut_indices(points, cuts)[1]
        for row in range(0, len(points), max(1, len(points) // 8)):
            for neighbor in (j - 1, j + 1):
                radial = np.hypot(*points[row, j, :2]), np.hypot(*points[row, neighbor, :2])
                tangent = np.array([points[row, neighbor, 2] - points[row, j, 2], radial[1] - radial[0]])
                tangent *= 1 if neighbor > j else -1
                normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
                offset = np.array([outer[row, j, 2] - points[row, j, 2],
                                   np.hypot(*outer[row, j, :2]) - radial[0]])
                assert np.dot(offset, normal) == pytest.approx(2, abs=1e-8), (row, neighbor)


@pytest.mark.parametrize('angles,mode', [
    ((60, -60), 'bare'), ((70, -70), 'bare'), ((70, 0), 'bare'), ((10, 70), 'bare'),
    ((65, -65), 'freestanding'), ((70, 0), 'enclosure'),
])
def test_steep_apex_quality(tmp_path, angles, mode):
    config = production(angled_config(*angles, mode))
    if mode == 'enclosure':
        config['enclosure']['edge'] = 0
    corner = build(config, tmp_path / 'corner.msh')
    smooth = build(without_corners(config), tmp_path / 'smooth.msh')
    assert_creases(corner)
    assert_quality(corner, smooth)


@pytest.mark.parametrize('quadrants', ['1', '12', '14'])
@pytest.mark.parametrize('mode', ['bare', 'enclosure', 'freestanding'])
def test_reduced_domain_roles(session, tmp_path, quadrants, mode):  # noqa: F811
    config = production(angled_config(30, 15, mode))
    config['mesh']['quadrants'] = quadrants
    expected = {PhysicalGroup.RIGID_WALL, PhysicalGroup.PRIMARY_SOURCE}
    if mode == 'enclosure':
        config['enclosure']['edge'] = 0
        config['mesh'].update(subdomain_slices='6', interface_offset_mm=2)
        expected |= {PhysicalGroup.ENCLOSURE_WALL, PhysicalGroup.INTERFACE}
    result = build(config, tmp_path / f'{quadrants}.msh')
    assert_creases(result)
    groups = {tag: set(map(int, gmsh.model.getEntitiesForPhysicalGroup(2, tag)))
              for _, tag in gmsh.model.getPhysicalGroups(2)}
    assert set(groups) == set(map(int, expected))
    assert all(groups.values())
    for a, faces in groups.items():
        for b, others in groups.items():
            assert a == b or not faces & others, (a, b)
    assert set.union(*groups.values()) == {tag for _, tag in gmsh.model.getEntities(2)}


# -- Rectangle morph after 0.2.6 -----------------------------------------------

def morph_config(mode, stations=None):
    def profile(r1, r2):
        return {'points': [{'z': 0, 'r': 12.7, 'join': 'corner'}, {'z': 60, 'r': r1, 'join': 'corner'},
                           {'z': 120, 'r': r2, 'join': 'corner'}]}
    config = {'formula': 'FREEFORM', 'mode': mode,
              'profile': {'profileH': profile(60, 160), 'profileV': profile(45, 110),
                          'crossSections': stations or [{'t': 0, 'shape': 'circle'},
                                                        {'t': 1, 'shape': 'ellipse'}]},
              'morph': {'morphTarget': 1, 'morphCorner': 0., 'morphFixed': 0.},
              'mesh': {'throat_res_mm': 5., 'mouth_res_mm': 20., 'wall_thickness_mm': 2.,
                       'scale_to_metres': False}}
    if mode == 'enclosure':
        config['enclosure'] = {'depth_mm': 150, 'edge_mm': 0}
    return config


@pytest.mark.parametrize('mode', ['bare', 'enclosure', 'freestanding'])
def test_sharp_rectangle_morph_with_corners_builds(tmp_path, mode):
    config = morph_config(mode)
    corner = build(config, tmp_path / 'corner.msh')
    smooth = build(without_corners(config), tmp_path / 'smooth.msh')
    assert_creases(corner)
    assert_quality(corner, smooth)
    # The freestanding shell clears the bore without the facet repair.
    assert 'shellFacetValidation' not in corner.metadata


def test_rounded_rectangle_station_with_sharp_morph_refused_with_corners(tmp_path):
    config = morph_config('bare', [{'t': 0, 'shape': 'circle'},
                                   {'t': .2, 'shape': 'rounded_rectangle', 'cornerRadiusMm': 10.},
                                   {'t': .4, 'shape': 'ellipse'}, {'t': 1, 'shape': 'ellipse'}])
    for candidate in (config, without_corners(config)):
        with pytest.raises(ValueError, match='cannot be combined with a sharp rectangle morph target'):
            build(candidate, Path(tmp_path) / 'refused.msh')
