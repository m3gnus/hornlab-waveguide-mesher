"""FREEFORM corner walls refuse controls their split-patch builder ignores."""
from __future__ import annotations

import pytest
from test_freeform_corner_wall import angled_config, without_corners

from hornlab_mesher import build_geometry_params
from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.corner_wall import CornerWall, corner_geometry_class
from hornlab_mesher.freeform import build_freeform_geometry
from hornlab_mesher.geometry import _MouthFittedPointGridHornGeometry
from hornlab_mesher.profile_sampling import build_point_grid_arrays

CORNERS = 'FREEFORM corner z=0, 60, 120 mm'


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
