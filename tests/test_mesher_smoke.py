from __future__ import annotations

import numpy as np

from hornlab_mesher import (
    AxiHornGeometry,
    CabinetGeometry,
    DriverConfig,
    MeshDensity,
    RectHornGeometry,
    build_mesh,
    load_mesh,
)


def test_axisymmetric_smoke(tmp_path):
    path = build_mesh(
        AxiHornGeometry(
            profile_points=np.array(
                [
                    [0.0, 12.7],
                    [40.0, 24.0],
                    [120.0, 80.0],
                ],
                dtype=float,
            ),
            throat_radius_mm=12.7,
            n_phi=24,
        ),
        MeshDensity(throat_res_mm=8.0, mouth_res_mm=30.0),
        tmp_path / "axis.msh",
    )
    info = load_mesh(path)
    assert info.n_triangles > 0
    assert set(info.physical_groups) >= {1, 2}


def test_rectangular_smoke(tmp_path):
    path = build_mesh(
        RectHornGeometry(
            primary_h_deg=45.0,
            primary_v_deg=35.0,
            mouth_width_mm=160.0,
            mouth_height_mm=100.0,
            n_phi=24,
            n_length=6,
        ),
        MeshDensity(throat_res_mm=8.0, mouth_res_mm=30.0),
        tmp_path / "rect.msh",
    )
    info = load_mesh(path)
    assert info.n_triangles > 0
    assert set(info.physical_groups) >= {1, 2}


def test_cabinet_smoke(tmp_path):
    path = build_mesh(
        CabinetGeometry(
            width_mm=300.0,
            depth_mm=180.0,
            height_mm=220.0,
            drivers=[DriverConfig(diameter_mm=80.0, tag=2)],
            aperture_width_mm=120.0,
            aperture_height_mm=40.0,
        ),
        MeshDensity(throat_res_mm=12.0, mouth_res_mm=30.0, rear_res_mm=35.0),
        tmp_path / "cabinet.msh",
    )
    info = load_mesh(path)
    assert info.n_triangles > 0
    assert set(info.physical_groups) >= {1, 2, 3}
