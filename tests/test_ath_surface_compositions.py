"""Compositions measured in ATH V2026-09a; authored geometry stays authoritative."""
from pathlib import Path

import numpy as np
import pytest

from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.config_parser import parse_text_config

FIXTURES = Path(__file__).parent / 'fixtures/ath_import/ath-v2026-09a/compositions'
FAMILIES = ['lip', 'opus', 'lip_opus']
VARIANTS = ['base', 'scale', 'scale_z', 'ext', 'ext_angle', 'rot', 'offset', 'source2']


def config(name):
    return parse_text_config((FIXTURES / f'{name}.cfg').read_text())


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('variant', VARIANTS)
def test_composed_grid_and_complete_rear_match_ath(family, variant):
    name = f'{family}_{variant}'
    cfg = config(name)
    geometry = resolve_geometry(cfg).geometry
    expected = np.load(FIXTURES / f'{name}.npz')
    np.testing.assert_allclose(geometry.inner_points, expected['inner'], atol=2e-4, rtol=0)
    actual = np.vstack((geometry.ath_rear_points.reshape(-1, 3), geometry.ath_rear_apex))
    distance = np.linalg.norm(actual[:, None] - expected['rear'][None, :], axis=-1)
    assert distance.min(axis=0).max() <= 2e-3
    assert distance.min(axis=1).max() <= 2e-3
    assert geometry.vertical_offset_mm == (17 if variant == 'offset' else 0)
    assert geometry.source_shape == (0 if variant == 'source2' else 1)  # ATH 2 is a flat disc.
    # The other topology and dictionary transport have the same construction.
    cfg['mesh']['topology'] = 'legacy'
    other = resolve_geometry(cfg).geometry
    np.testing.assert_array_equal(other.inner_points, geometry.inner_points)
    np.testing.assert_array_equal(other.ath_rear_points, geometry.ath_rear_points)


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('variant', VARIANTS[1:])
def test_compositions_preview_and_solve_mesh_close_on_the_same_geometry(tmp_path, family, variant):
    import meshio
    from test_ath_import_surfaces import assert_watertight_after_symmetry

    from hornlab_mesher.mesher import build_mesh
    from hornlab_mesher.preview.api import build_preview_geometry
    from hornlab_mesher.preview.contract import PreviewOptionsV1
    from hornlab_mesher.tags import PhysicalGroup

    cfg = config(f'{family}_{variant}')
    r = resolve_geometry(cfg)
    preview = build_preview_geometry(cfg, PreviewOptionsV1(lod='coarse', include_curvature=False))
    assert {'horn.inner', 'horn.outer', 'source_cap', 'wall.rear_cap'} <= {s.role for s in preview.surfaces}
    assert all(np.isfinite(s.positions).all() for s in preview.surfaces)
    mesh = meshio.read(build_mesh(r.geometry, r.density, tmp_path / 'shell.msh', scale_to_metres=False))
    # Undo the measured placement before reflecting the reduced domain.
    if variant == 'offset':
        mesh.points[:, 1] -= 17
    assert_watertight_after_symmetry(mesh, r.geometry)
    if variant == 'source2':
        source_nodes = np.concatenate([
            cell.data[tags == int(PhysicalGroup.PRIMARY_SOURCE)].reshape(-1)
            for cell, tags in zip(mesh.cells, mesh.cell_data['gmsh:physical'])
            if cell.type == 'triangle'
        ])
        np.testing.assert_allclose(mesh.points[source_nodes, 2], 0, rtol=0, atol=1e-10)
