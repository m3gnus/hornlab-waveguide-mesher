from __future__ import annotations

import numpy as np

from hornlab_mesher import (
    CrossSection,
    MeshDensity,
    OsseHornGeometry,
    build_mesh,
    compute_osse_profile_points,
    load_mesh,
)
from hornlab_mesher.profiles import profile_points
from hornlab_mesher.builders._occ import superellipse_ring


def test_compute_osse_profile_points_endpoints():
    geom = OsseHornGeometry(
        L_mm=120.0, r0_mm=12.7, a_deg=60.0, a0_deg=15.5,
        k=1.0, n=4.0, q=0.995, s=0.0, n_axial=5,
    )
    points = compute_osse_profile_points(geom)
    assert points.shape == (5, 2)
    assert points[0, 0] == 0.0
    assert points[-1, 0] == 120.0
    assert np.isclose(points[0, 1], 12.7)
    # Mouth radius pinned 2026-05-18 from canonical WG evaluator.
    assert np.isclose(points[-1, 1], 210.25359737777225)


def test_osse_profile_points_ignore_rosse_tmax_key():
    params = {
        "type": "OSSE",
        "L": 120.0,
        "r0": 12.7,
        "a": 60.0,
        "a0": 15.5,
        "k": 1.0,
        "n": 4.0,
        "q": 0.995,
        "tmax": 0.5,
    }

    points = profile_points(params, 5)

    assert points[-1, 0] == 120.0
    assert np.isclose(points[-1, 1], 210.25359737777225)


def test_build_osse_waveguide_smoke(tmp_path):
    path = build_mesh(
        OsseHornGeometry(
            L_mm=120.0,
            r0_mm=12.7,
            a_deg=60.0,
            a0_deg=15.5,
            n_phi=24,
            n_axial=8,
        ),
        MeshDensity(throat_res_mm=8.0, mouth_res_mm=30.0),
        tmp_path / "osse.msh",
    )
    info = load_mesh(path)
    assert info.n_triangles > 0
    assert set(info.physical_groups) >= {1, 2}


def test_build_osse_waveguide_superellipse_cross_section(tmp_path):
    """Non-circular cross-section should still mesh cleanly."""
    path = build_mesh(
        OsseHornGeometry(
            L_mm=100.0,
            r0_mm=15.0,
            a_deg=45.0,
            a0_deg=10.0,
            cross_section=CrossSection(exponent=4.0, aspect_ratio=1.3),
            n_phi=32,
            n_axial=10,
        ),
        MeshDensity(throat_res_mm=6.0, mouth_res_mm=22.0),
        tmp_path / "osse_se.msh",
    )
    info = load_mesh(path)
    assert info.n_triangles > 0


def test_axisymmetric_superellipse_aspect_ratio_matches_point_grid_semantics():
    ring = superellipse_ring(
        z=0.0,
        radius=10.0,
        exponent=2.0,
        aspect_ratio=1.5,
        n_phi=8,
    )

    assert np.isclose(float(np.max(ring[:, 0])), 15.0)
    assert np.isclose(float(np.max(ring[:, 1])), 10.0)
