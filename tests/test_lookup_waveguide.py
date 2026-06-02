from __future__ import annotations

import numpy as np

from hornlab_mesher import (
    LookupHornGeometry,
    MeshDensity,
    build_lookup_waveguide,
    build_mesh,
    compute_lookup_profile_points,
    load_mesh,
)


def test_compute_lookup_profile_points_pchip():
    pts = np.array([[0.0, 10.0], [60.0, 50.0], [120.0, 150.0]], dtype=float)
    geom = LookupHornGeometry(lookup_points=pts, n_axial=5)
    out = compute_lookup_profile_points(geom)
    assert out.shape == (5, 2)
    assert np.isclose(out[0, 1], 10.0)
    assert np.isclose(out[-1, 1], 150.0)
    # Midpoint hits the middle control point exactly.
    assert np.isclose(out[2, 1], 50.0)


def test_build_lookup_waveguide_smoke(tmp_path):
    pts = np.array(
        [
            [0.0, 12.7],
            [30.0, 25.0],
            [80.0, 90.0],
            [150.0, 200.0],
        ],
        dtype=float,
    )
    path = build_mesh(
        LookupHornGeometry(lookup_points=pts, n_phi=24, n_axial=10),
        MeshDensity(throat_res_mm=8.0, mouth_res_mm=30.0),
        tmp_path / "lookup.msh",
    )
    info = load_mesh(path)
    assert info.n_triangles > 0
    assert set(info.physical_groups) >= {1, 2}


def test_lookup_geometry_rejects_invalid_points():
    geom = LookupHornGeometry(
        lookup_points=np.array([[0.0, 10.0]], dtype=float),
        n_axial=5,
    )
    try:
        compute_lookup_profile_points(geom)
    except ValueError:
        return
    raise AssertionError("expected ValueError for single-point lookup")
