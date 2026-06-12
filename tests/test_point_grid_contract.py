from __future__ import annotations

import math

import meshio
import numpy as np
import pytest

from hornlab_mesher.builders._occ import make_planar_sector_fill_from_ring
from hornlab_mesher.builders.enclosure import _add_curve_loop_from_curves
from hornlab_mesher.builders.point_grid_interfaces import _normalise_interface_specs
from hornlab_mesher.builders.point_grid_surfaces import _rear_rim_points
from hornlab_mesher import HornEnclosure, MeshDensity, MesherError, build_mesh
from hornlab_mesher.cli import build_from_config, parse_ath_config
from hornlab_mesher.geometry import HornInterface, PointGridHornGeometry
from hornlab_mesher.profiles import build_point_grid
from hornlab_mesher.viewport import build_viewport_geometry_from_config


_ASRO2_PARAMS = {
    "type": "R-OSSE",
    "R": "160 * (abs(cos(p)/1.8)^3 + abs(sin(p)/1)^4)^(-1/7)",
    "r": 0.35,
    "b": 0.4,
    "m": 0.84,
    "tmax": 1.0,
    "a": "22 * (abs(cos(p)/1.2)^8 + abs(sin(p)/1)^4)^(-1/4)",
    "a0": 15.5,
    "r0": 12.7,
    "k": "4 * (abs(cos(p)/1.2)^8 + abs(sin(p)/1)^4)^(-1/4)",
    "q": 4.0,
    "angularSegments": 50,
    "lengthSegments": 20,
    "throatResolution": 5.0,
    "mouthResolution": 8.0,
    "quadrants": "1",
    "wallThickness": 6.0,
    "rearResolution": 25.0,
    "encDepth": 0.0,
    "sourceShape": 1,
    "sourceRadius": -1.0,
    "sourceCurv": 0,
}

_ATH_ASRO2_T_VALUES = np.asarray(
    [
        0.0,
        0.031652775,
        0.069285650,
        0.111291038,
        0.158158738,
        0.208217141,
        0.261010634,
        0.315152186,
        0.371049458,
        0.427239696,
        0.483180970,
        0.538366332,
        0.593546216,
        0.647147114,
        0.701376236,
        0.753382922,
        0.804185680,
        0.854976845,
        0.904174233,
        0.953060714,
        1.0,
    ],
    dtype=np.float64,
)

_ATH_ASRO2_PHI0_Z_MM = np.asarray(
    [
        0.0,
        10.55795,
        22.82474,
        36.13612,
        50.49133,
        65.20920,
        79.99040,
        94.28462,
        108.01910,
        120.63596,
        131.82684,
        141.29410,
        148.90752,
        154.19315,
        157.04063,
        157.04101,
        154.16048,
        148.24877,
        139.59949,
        128.26674,
        115.02068,
    ],
    dtype=np.float64,
)

_ATH_ASRO2_PHI0_R_MM = np.asarray(
    [
        12.70000,
        16.16966,
        20.92049,
        26.90634,
        34.29988,
        42.90421,
        52.67341,
        63.37160,
        75.10357,
        87.57512,
        100.63005,
        114.06428,
        127.92751,
        141.62115,
        155.42888,
        168.27181,
        180.00068,
        190.37293,
        198.48906,
        203.91918,
        205.83655,
    ],
    dtype=np.float64,
)


def _make_point_grid(
    *,
    n_phi: int = 32,
    n_length: int = 12,
    length: float = 140.0,
    r0: float = 12.7,
    r1: float = 100.0,
) -> np.ndarray:
    points = np.empty((n_phi, n_length + 1, 3), dtype=np.float64)
    for i in range(n_phi):
        phi = math.tau * i / n_phi
        for j in range(n_length + 1):
            t = j / n_length
            radius = r0 + (r1 - r0) * t
            points[i, j] = (
                radius * math.cos(phi),
                radius * math.sin(phi),
                length * t,
            )
    return points


def _make_quarter_point_grid(
    *,
    n_phi: int = 9,
    n_length: int = 8,
    length: float = 100.0,
    r0: float = 12.7,
    r1: float = 80.0,
) -> np.ndarray:
    points = np.empty((n_phi, n_length + 1, 3), dtype=np.float64)
    for i in range(n_phi):
        phi = (math.pi / 2.0) * i / (n_phi - 1)
        for j in range(n_length + 1):
            t = j / n_length
            radius = r0 + (r1 - r0) * t
            points[i, j] = (
                radius * math.cos(phi),
                radius * math.sin(phi),
                length * t,
            )
    return points


def _triangle_tags(mesh) -> list[int]:
    tags: list[int] = []
    for index, cell_block in enumerate(mesh.cells):
        if cell_block.type not in ("triangle", "triangle3"):
            continue
        phys = mesh.cell_data["gmsh:physical"][index]
        tags.extend(int(v) for v in phys)
    return tags


def _triangles_and_tags(mesh) -> tuple[np.ndarray, np.ndarray]:
    tri_key = "triangle" if "triangle" in mesh.cells_dict else "triangle3"
    triangles = np.asarray(mesh.cells_dict[tri_key], dtype=np.int64)
    tags = None
    for key, by_type in mesh.cell_data_dict.items():
        if "physical" in key and tri_key in by_type:
            tags = np.asarray(by_type[tri_key], dtype=np.int32)
            break
    if tags is None:
        raise AssertionError("mesh has no physical triangle tags")
    return triangles, tags


def _boundary_edges(triangles: np.ndarray) -> dict[tuple[int, int], list[int]]:
    edges: dict[tuple[int, int], list[int]] = {}
    for tri_idx, tri in enumerate(triangles):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = tuple(sorted((int(a), int(b))))
            edges.setdefault(key, []).append(int(tri_idx))
    return {edge: owners for edge, owners in edges.items() if len(owners) == 1}


def _tag_components(triangles: np.ndarray, tags: np.ndarray, tag: int) -> list[int]:
    tri_indices = np.flatnonzero(tags == int(tag))
    local = {int(tri_idx): idx for idx, tri_idx in enumerate(tri_indices)}
    edges: dict[tuple[int, int], list[int]] = {}
    for tri_idx in tri_indices:
        tri = triangles[tri_idx]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = tuple(sorted((int(a), int(b))))
            edges.setdefault(key, []).append(int(tri_idx))
    adjacency = [set() for _ in tri_indices]
    for owners in edges.values():
        for i, lhs in enumerate(owners):
            for rhs in owners[i + 1:]:
                if lhs in local and rhs in local:
                    adjacency[local[lhs]].add(local[rhs])
                    adjacency[local[rhs]].add(local[lhs])
    seen = np.zeros(len(tri_indices), dtype=bool)
    sizes: list[int] = []
    for idx in range(len(tri_indices)):
        if seen[idx]:
            continue
        stack = [idx]
        seen[idx] = True
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nxt in adjacency[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def test_rear_rim_points_extend_local_outer_profile_to_rear_plane():
    n_phi = 32
    outer = np.empty((n_phi, 2, 3), dtype=np.float64)
    for i in range(n_phi):
        phi = math.tau * i / n_phi
        throat_radius = 20.0 + 3.0 * math.cos(2.0 * phi)
        next_radius = 30.0 + 6.0 * math.sin(phi)
        outer[i, 0] = (throat_radius * math.cos(phi), throat_radius * math.sin(phi), 0.0)
        outer[i, 1] = (next_radius * math.cos(phi), next_radius * math.sin(phi), 10.0)

    rear = _rear_rim_points(outer, rear_z=-5.0)

    t = -0.5
    expected = outer[:, 0, :] + (outer[:, 1, :] - outer[:, 0, :]) * t
    expected[:, 2] = -5.0
    assert np.allclose(rear, expected, rtol=0.0, atol=1.0e-9)
    assert np.allclose(rear[:, 2], -5.0)


def test_python_osse_point_grid_full_circle():
    grid = build_point_grid({
        "type": "OSSE",
        "L": 120.0,
        "r0": 12.7,
        "a": 60.0,
        "a0": 15.5,
        "k": 1.0,
        "n": 4.0,
        "q": 0.995,
        "angularSegments": 16,
        "lengthSegments": 6,
        "wallThickness": 5.0,
    })

    assert grid["full_circle"] is True
    assert grid["grid_n_phi"] == 16
    assert grid["grid_n_length"] == 6
    inner = np.asarray(grid["inner_points"], dtype=np.float64).reshape(16, 7, 3)
    outer = np.asarray(grid["outer_points"], dtype=np.float64).reshape(16, 7, 3)
    assert np.isclose(np.linalg.norm(inner[0, 0, :2]), 12.7)
    assert np.isclose(inner[0, -1, 2], 120.0)
    assert np.linalg.norm(outer[0, -1, :2]) > np.linalg.norm(inner[0, -1, :2])


def test_viewport_geometry_from_config_returns_enclosure_rings():
    geometry = build_viewport_geometry_from_config(
        {
            "formula": "OSSE",
            "mode": "enclosure",
            "profile": {
                "L": 120.0,
                "r0": 12.7,
                "a": 60.0,
                "a0": 15.5,
                "k": 1.0,
                "n": 4.0,
                "q": 0.995,
            },
            "mesh": {
                "angularSegments": 16,
                "lengthSegments": 6,
                "wallThickness": 0.0,
            },
            "enclosure": {
                "depth": 150.0,
                "space_l": 25.0,
                "space_t": 25.0,
                "space_r": 25.0,
                "space_b": 25.0,
                "edge": 10.0,
                "edgeType": 1,
            },
        }
    )

    grid = geometry["grid"]
    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    inner = np.asarray(grid["inner_points"], dtype=np.float64).reshape(n_phi, n_length + 1, 3)
    enclosure = geometry["enclosure"]
    assert enclosure is not None

    mouth = np.asarray(enclosure["mouth_points"], dtype=np.float64).reshape(n_phi, 3)
    front = np.asarray(enclosure["front_outer_points"], dtype=np.float64).reshape(-1, 3)
    back = np.asarray(enclosure["back_outer_points"], dtype=np.float64).reshape(-1, 3)
    assert np.allclose(mouth, inner[:, -1, :])
    assert front.shape[0] >= 16
    assert back.shape == front.shape
    assert np.isclose(front[:, 2].max(), inner[:, -1, 2].max())
    assert back[:, 2].max() < front[:, 2].max()
    assert front[:, 0].max() > mouth[:, 0].max()
    assert front[:, 1].max() > mouth[:, 1].max()
    assert enclosure["edge_depth"] > 0
    assert [ring["role"] for ring in enclosure["profile_rings"]] == [
        "front_inset",
        "front_edge",
        "front_edge",
        "side_back_outer",
        "back_edge",
        "back_edge",
    ]
    point_counts = {
        len(np.asarray(ring["points"], dtype=np.float64).reshape(-1, 3))
        for ring in enclosure["profile_rings"]
    }
    assert len(point_counts) == 1


def test_viewport_enclosure_edge_clamps_to_smallest_margin():
    geometry = build_viewport_geometry_from_config(
        {
            "formula": "OSSE",
            "mode": "enclosure",
            "profile": {
                "L": 120.0,
                "r0": 12.7,
                "a": 60.0,
                "a0": 15.5,
                "k": 1.0,
                "n": 4.0,
                "q": 0.995,
            },
            "mesh": {
                "angularSegments": 16,
                "lengthSegments": 6,
                "wallThickness": 0.0,
            },
            "enclosure": {
                "depth": 150.0,
                "space_l": 8.0,
                "space_t": 8.0,
                "space_r": 8.0,
                "space_b": 8.0,
                "edge": 12.0,
                "edgeType": 1,
            },
        }
    )

    assert geometry["enclosure"]["edge_mm"] == 8.0


def test_python_osse_point_grid_ignores_rosse_tmax_key():
    base = {
        "type": "OSSE",
        "L": 120.0,
        "r0": 12.7,
        "a": 60.0,
        "a0": 15.5,
        "k": 1.0,
        "n": 4.0,
        "q": 0.995,
        "angularSegments": 16,
        "lengthSegments": 6,
        "wallThickness": 0.0,
    }

    grid = build_point_grid({**base, "tmax": 0.5})

    inner = np.asarray(grid["inner_points"], dtype=np.float64).reshape(16, 7, 3)
    assert np.isclose(inner[0, -1, 2], 120.0)
    assert np.allclose(
        np.asarray(grid["slice_map"], dtype=np.float64),
        np.linspace(0.0, 1.0, 7, dtype=np.float64),
    )


def test_python_rosse_point_grid_supports_expressions_and_quarter_domain():
    grid = build_point_grid({
        "type": "R-OSSE",
        "R": "150 * (abs(cos(p))^4 + abs(sin(p))^4)^(-1/4)",
        "r0": 12.7,
        "a": "45 + 5*cos(p)",
        "a0": 15.5,
        "k": 1.0,
        "q": 1.0,
        "angularSegments": 16,
        "lengthSegments": 5,
        "quadrants": "1",
    })

    assert grid["full_circle"] is False
    assert grid["grid_n_phi"] == 5
    angles = np.asarray(grid["angle_list"], dtype=np.float64)
    assert np.allclose(angles, np.linspace(0.0, math.pi / 2.0, 5))
    inner = np.asarray(grid["inner_points"], dtype=np.float64).reshape(5, 6, 3)
    assert np.isclose(np.linalg.norm(inner[0, 0, :2]), 12.7)
    assert np.isclose(np.linalg.norm(inner[-1, -1, :2]), 150.0)


def test_python_rosse_point_grid_rejects_guiding_curve():
    with pytest.raises(ValueError, match="guiding curves"):
        build_point_grid({
            "type": "R-OSSE",
            "R": 150.0,
            "r0": 12.7,
            "a": 45.0,
            "a0": 15.5,
            "k": 1.0,
            "q": 1.0,
            "angularSegments": 16,
            "lengthSegments": 5,
            "gcurveType": 1,
            "gcurveWidth": 100.0,
        })


def test_sampling_modes_distinguish_uniform_and_ath_default_zmap():
    default_grid = build_point_grid(_ASRO2_PARAMS)
    ath_grid = build_point_grid({**_ASRO2_PARAMS, "samplingMode": "ath-default-zmap"})

    assert default_grid["sampling_mode"] == "uniform"
    assert np.allclose(
        np.asarray(default_grid["slice_map"], dtype=np.float64),
        np.linspace(0.0, 1.0, 21, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert ath_grid["sampling_mode"] == "ath-default-zmap"
    assert not np.allclose(
        np.asarray(default_grid["slice_map"], dtype=np.float64),
        np.asarray(ath_grid["slice_map"], dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_ath_default_zmap_sampling_matches_asro2_exported_grid():
    grid = build_point_grid({**_ASRO2_PARAMS, "samplingMode": "ath-default-zmap"})

    assert int(grid["grid_n_phi"]) == 15
    assert int(grid["grid_n_length"]) == 20
    assert grid["full_circle"] is False

    uniform_quarter = np.linspace(0.0, math.pi / 2.0, 15, dtype=np.float64)
    angles = np.asarray(grid["angle_list"], dtype=np.float64)
    assert np.allclose(angles, uniform_quarter, rtol=0.0, atol=1.0e-12)

    assert np.allclose(
        np.asarray(grid["slice_map"], dtype=np.float64),
        _ATH_ASRO2_T_VALUES,
        rtol=0.0,
        atol=1.0e-9,
    )

    inner = np.asarray(grid["inner_points"], dtype=np.float64).reshape(15, 21, 3)
    phi0 = inner[0]
    assert np.allclose(phi0[:, 2], _ATH_ASRO2_PHI0_Z_MM, rtol=0.0, atol=2.0e-4)
    assert np.allclose(
        np.linalg.norm(phi0[:, :2], axis=1),
        _ATH_ASRO2_PHI0_R_MM,
        rtol=0.0,
        atol=2.0e-4,
    )


def test_custom_zmap_sampling_interpolates_control_points():
    grid = build_point_grid(
        {
            "type": "OSSE",
            "L": 100.0,
            "r0": 12.7,
            "a": 45.0,
            "a0": 10.0,
            "lengthSegments": 4,
            "angularSegments": 8,
            "samplingMode": "zmap",
            "zMapPoints": "0.5,0.1,0.75,0.7",
        }
    )

    assert grid["sampling_mode"] == "zmap"
    assert np.allclose(
        np.asarray(grid["slice_map"], dtype=np.float64),
        np.asarray([0.0, 0.05, 0.1, 0.7, 1.0], dtype=np.float64),
        rtol=0.0,
        atol=1.0e-12,
    )


def _asro2_ath_cfg_text(*, throat: float = 5.0, mouth: float = 8.0, rear: float = 25.0) -> str:
    return f"""
R-OSSE = {{
  R = {_ASRO2_PARAMS["R"]}
  r = {_ASRO2_PARAMS["r"]}
  b = {_ASRO2_PARAMS["b"]}
  m = {_ASRO2_PARAMS["m"]}
  tmax = {_ASRO2_PARAMS["tmax"]}
  a = {_ASRO2_PARAMS["a"]}
  a0 = {_ASRO2_PARAMS["a0"]}
  r0 = {_ASRO2_PARAMS["r0"]}
  k = {_ASRO2_PARAMS["k"]}
  q = {_ASRO2_PARAMS["q"]}
}}
Mesh = {{
  AngularSegments = 50
  LengthSegments = 20
  WallThickness = 6.0
  Quadrants = 1
  ThroatResolution = {throat}
  MouthResolution = {mouth}
  RearResolution = {rear}
}}
Source = {{
  Shape = 1
  Radius = -1.0
  Curv = 0
}}
"""


def test_ath_config_build_uses_common_resolution_tessellation(tmp_path):
    cfg = parse_ath_config(_asro2_ath_cfg_text())

    assert cfg["mesh"]["samplingMode"] == "ath-default-zmap"
    result = build_from_config(cfg, tmp_path / "asro2-ath.msh")
    mesh = meshio.read(result.mesh_path)
    triangles, tags = _triangles_and_tags(mesh)

    assert int(np.count_nonzero(tags == 1)) > 0
    assert int(np.count_nonzero(tags == 2)) > 0
    assert len(_tag_components(triangles, tags, 1)) == 1


def test_ath_config_tessellation_follows_resolution_inputs(tmp_path):
    coarse = build_from_config(
        parse_ath_config(_asro2_ath_cfg_text(throat=10.0, mouth=16.0, rear=50.0)),
        tmp_path / "coarse.msh",
    )
    fine = build_from_config(
        parse_ath_config(_asro2_ath_cfg_text(throat=3.0, mouth=5.0, rear=12.0)),
        tmp_path / "fine.msh",
    )

    assert fine.n_triangles > coarse.n_triangles


def test_flat_ath_config_keys_build_partial_source_group(tmp_path):
    cfg = parse_ath_config(
        """
R-OSSE = {
R = 160 * (abs(cos(p)/1.8)^3 + abs(sin(p)/1)^4)^(-1/7)
a = 22 * (abs(cos(p)/1.2)^8 + abs(sin(p)/1)^4)^(-1/4)
a0 = 15.5
b = 0.4
k = 4 * (abs(cos(p)/1.2)^8 + abs(sin(p)/1)^4)^(-1/4)
m = 0.84
q = 4
r = 0.35
r0 = 12.7
}
Mesh.AngularSegments = 50
Mesh.LengthSegments = 20
Mesh.MouthResolution = 8
Mesh.Quadrants = 1
Mesh.RearResolution = 25
Mesh.ThroatResolution = 5
Mesh.WallThickness = 6
Source.Shape = 1
Source.Radius = -1
Source.Curv = 0
"""
    )

    assert cfg["mesh"]["angularSegments"] == 50
    assert cfg["mesh"]["lengthSegments"] == 20
    assert cfg["mesh"]["quadrants"] == 1
    assert cfg["source"]["sourceShape"] == 1
    result = build_from_config(cfg, tmp_path / "flat-asro2-ath.msh")
    mesh = meshio.read(result.mesh_path)
    _, tags = _triangles_and_tags(mesh)

    assert int(np.count_nonzero(tags == 1)) > 0
    assert int(np.count_nonzero(tags == 2)) > 0


def test_point_grid_mesh_has_canonical_wall_and_source_tags(tmp_path):
    msh_path = build_mesh(
        PointGridHornGeometry(
            inner_points=_make_point_grid(),
            closed=True,
            wall_thickness_mm=0.0,
            preserve_grid=True,
        ),
        MeshDensity(throat_res_mm=8.0, mouth_res_mm=16.0, rear_res_mm=24.0),
        tmp_path / "horn.msh",
    )

    mesh = meshio.read(msh_path)
    tags = _triangle_tags(mesh)
    assert tags
    assert {1, 2}.issubset(set(tags))


def test_point_grid_flat_source_shape_builds_disc_at_throat_plane(tmp_path):
    msh_path = build_mesh(
        PointGridHornGeometry(
            inner_points=_make_point_grid(),
            closed=True,
            wall_thickness_mm=0.0,
            preserve_grid=True,
            source_shape=0,
        ),
        MeshDensity(throat_res_mm=8.0, mouth_res_mm=16.0, rear_res_mm=24.0),
        tmp_path / "flat-source.msh",
        scale_to_metres=False,
    )

    mesh = meshio.read(msh_path)
    triangles, tags = _triangles_and_tags(mesh)
    source_points = np.unique(triangles[tags == 2])
    source_z = np.asarray(mesh.points, dtype=np.float64)[source_points, 2]
    assert np.ptp(source_z) < 1.0e-9
    assert np.isclose(source_z[0], 0.0, atol=1.0e-9)


def test_point_grid_rounded_source_shape_builds_cap(tmp_path):
    msh_path = build_mesh(
        PointGridHornGeometry(
            inner_points=_make_point_grid(),
            closed=True,
            wall_thickness_mm=0.0,
            preserve_grid=True,
            source_shape=1,
            source_radius_mm=60.0,
        ),
        MeshDensity(throat_res_mm=8.0, mouth_res_mm=16.0, rear_res_mm=24.0),
        tmp_path / "rounded-source.msh",
        scale_to_metres=False,
    )

    mesh = meshio.read(msh_path)
    triangles, tags = _triangles_and_tags(mesh)
    source_points = np.unique(triangles[tags == 2])
    source_z = np.asarray(mesh.points, dtype=np.float64)[source_points, 2]
    assert float(np.max(source_z)) > 1.0
    assert np.isclose(float(np.min(source_z)), 0.0, atol=1.0e-9)


def test_point_grid_unsupported_source_shape_fails_explicitly(tmp_path):
    with pytest.raises(MesherError, match="source_shape=2 is not supported"):
        build_mesh(
            PointGridHornGeometry(
                inner_points=_make_point_grid(),
                closed=True,
                wall_thickness_mm=0.0,
                preserve_grid=True,
                source_shape=2,
            ),
            MeshDensity(throat_res_mm=8.0, mouth_res_mm=16.0, rear_res_mm=24.0),
            tmp_path / "bad-source.msh",
        )


def test_open_quarter_point_grid_mesh_has_source_sector(tmp_path):
    msh_path = build_mesh(
        PointGridHornGeometry(
            inner_points=_make_quarter_point_grid(),
            closed=False,
            wall_thickness_mm=0.0,
            preserve_grid=True,
        ),
        MeshDensity(throat_res_mm=8.0, mouth_res_mm=16.0, rear_res_mm=24.0),
        tmp_path / "quarter.msh",
    )

    mesh = meshio.read(msh_path)
    tags = _triangle_tags(mesh)
    points = np.asarray(mesh.points, dtype=np.float64)
    assert tags
    assert {1, 2}.issubset(set(tags))
    assert min(tags) == 1
    assert np.min(points[:, 0]) >= -1.0e-9
    assert np.min(points[:, 1]) >= -1.0e-9


def test_open_quarter_freestanding_point_grid_is_closed_except_symmetry_planes(tmp_path):
    inner = _make_quarter_point_grid(n_phi=9, n_length=8)
    outer = inner.copy()
    radial = np.linalg.norm(outer[:, :, :2], axis=2)
    scale = (radial + 6.0) / np.maximum(radial, 1.0e-12)
    outer[:, :, 0] *= scale
    outer[:, :, 1] *= scale
    outer[:, 0, 2] = inner[:, 0, 2] - 6.0

    msh_path = build_mesh(
        PointGridHornGeometry(
            inner_points=inner,
            outer_points=outer,
            closed=False,
            wall_thickness_mm=6.0,
            source_shape=1,
            source_radius_mm=-1.0,
            source_auto_angle_deg=15.5,
            preserve_grid=True,
        ),
        MeshDensity(throat_res_mm=8.0, mouth_res_mm=16.0, rear_res_mm=24.0),
        tmp_path / "quarter-freestanding.msh",
        scale_to_metres=False,
    )

    mesh = meshio.read(msh_path)
    triangles, tags = _triangles_and_tags(mesh)
    points = np.asarray(mesh.points, dtype=np.float64)
    boundary = _boundary_edges(triangles)
    assert boundary
    for a, b in boundary:
        pa = points[a]
        pb = points[b]
        on_x = abs(pa[0]) < 1.0e-9 and abs(pb[0]) < 1.0e-9
        on_y = abs(pa[1]) < 1.0e-9 and abs(pb[1]) < 1.0e-9
        assert on_x or on_y
    assert len(_tag_components(triangles, tags, 1)) == 1


def test_open_sector_fill_uses_single_gmsh_surface():
    import gmsh

    initialized_here = False
    try:
        if not gmsh.isInitialized():
            gmsh.initialize()
            initialized_here = True
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add("sector-fill-test")

        surfaces = make_planar_sector_fill_from_ring(
            _make_quarter_point_grid(n_phi=9, n_length=1)[:, 0, :],
            source_axis="z",
        )

        assert len(surfaces) == 1
    finally:
        if initialized_here and gmsh.isInitialized():
            gmsh.finalize()


def test_enclosure_curve_loop_orders_unordered_curves():
    import gmsh

    initialized_here = False
    try:
        if not gmsh.isInitialized():
            gmsh.initialize()
            initialized_here = True
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add("curve-loop-order-test")

        p0 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
        p1 = gmsh.model.occ.addPoint(1.0, 0.0, 0.0)
        p2 = gmsh.model.occ.addPoint(1.0, 1.0, 0.0)
        p3 = gmsh.model.occ.addPoint(0.0, 1.0, 0.0)
        c0 = gmsh.model.occ.addLine(p0, p1)
        c1 = gmsh.model.occ.addLine(p1, p2)
        c2 = gmsh.model.occ.addLine(p2, p3)
        c3 = gmsh.model.occ.addLine(p3, p0)

        loop = _add_curve_loop_from_curves([c2, c0, c3, c1])
        surface = gmsh.model.occ.addPlaneSurface([loop])

        assert surface > 0
    finally:
        if initialized_here and gmsh.isInitialized():
            gmsh.finalize()


def test_point_grid_enclosure_mesh_has_canonical_tags(tmp_path):
    msh_path = build_mesh(
        PointGridHornGeometry(
            inner_points=_make_point_grid(n_phi=24, n_length=10, r1=90.0),
            closed=True,
            preserve_grid=True,
            enclosure=HornEnclosure(
                depth_mm=220.0,
                space_l_mm=45.0,
                space_t_mm=35.0,
                space_r_mm=45.0,
                space_b_mm=35.0,
                edge_mm=8.0,
                edge_type=1,
                plan_type=1,
                plan_n=2.0,
                depth_margin_mm=12.0,
            ),
        ),
        MeshDensity(throat_res_mm=8.0, mouth_res_mm=18.0, rear_res_mm=30.0),
        tmp_path / "enclosed.msh",
    )

    mesh = meshio.read(msh_path)
    tags = _triangle_tags(mesh)
    assert tags
    assert {1, 2}.issubset(set(tags))


def test_point_grid_enclosure_edge_may_equal_margin_and_clamps_above_it(tmp_path):
    for edge_type in (1, 2):
        for edge_mm in (8.0, 12.0):
            msh_path = build_mesh(
                PointGridHornGeometry(
                    inner_points=_make_point_grid(n_phi=24, n_length=10, r1=90.0),
                    closed=True,
                    preserve_grid=True,
                    enclosure=HornEnclosure(
                        depth_mm=220.0,
                        space_l_mm=8.0,
                        space_t_mm=8.0,
                        space_r_mm=8.0,
                        space_b_mm=8.0,
                        edge_mm=edge_mm,
                        edge_type=edge_type,
                        plan_type=1,
                        plan_n=2.0,
                        depth_margin_mm=12.0,
                    ),
                ),
                MeshDensity(throat_res_mm=8.0, mouth_res_mm=18.0, rear_res_mm=30.0),
                tmp_path / f"enclosed_edge_{edge_type}_{edge_mm:g}.msh",
            )

            mesh = meshio.read(msh_path)
            tags = _triangle_tags(mesh)
            assert tags
            assert {1, 2}.issubset(set(tags))


def test_open_quarter_enclosure_uses_symmetry_axis_mouth_endpoints(tmp_path):
    mouth = np.asarray(
        [
            [60.0, 0.0],
            [92.0, 28.0],
            [110.0, 75.0],
            [74.0, 104.0],
            [0.0, 115.0],
        ],
        dtype=np.float64,
    )
    inner = np.zeros((mouth.shape[0], 6, 3), dtype=np.float64)
    for index, scale in enumerate(np.linspace(0.25, 1.0, inner.shape[1])):
        inner[:, index, 0] = mouth[:, 0] * scale
        inner[:, index, 1] = mouth[:, 1] * scale
        inner[:, index, 2] = 120.0 * index / (inner.shape[1] - 1)

    msh_path = build_mesh(
        PointGridHornGeometry(
            inner_points=inner,
            closed=False,
            preserve_grid=True,
            enclosure=HornEnclosure(
                depth_mm=180.0,
                space_l_mm=20.0,
                space_t_mm=20.0,
                space_r_mm=20.0,
                space_b_mm=20.0,
                edge_mm=8.0,
                edge_type=1,
                plan_type=1,
                depth_margin_mm=5.0,
            ),
        ),
        MeshDensity(throat_res_mm=10.0, mouth_res_mm=18.0, rear_res_mm=30.0),
        tmp_path / "bulged-quarter-enclosure.msh",
    )

    mesh = meshio.read(msh_path)
    assert mesh.cells_dict["triangle"].size > 0


def test_point_grid_enclosure_mesh_supports_multiple_interface_slices(tmp_path):
    msh_path = build_mesh(
        PointGridHornGeometry(
            inner_points=_make_point_grid(n_phi=24, n_length=10, r1=90.0),
            closed=True,
            preserve_grid=True,
            interfaces=(
                HornInterface(slice_index=4, offset_mm=8.0),
                HornInterface(slice_index=10, offset_mm=12.0),
            ),
            enclosure=HornEnclosure(
                depth_mm=220.0,
                space_l_mm=45.0,
                space_t_mm=35.0,
                space_r_mm=45.0,
                space_b_mm=35.0,
                edge_mm=8.0,
                edge_type=1,
                plan_type=1,
                plan_n=2.0,
                depth_margin_mm=12.0,
            ),
        ),
        MeshDensity(
            throat_res_mm=8.0,
            mouth_res_mm=18.0,
            rear_res_mm=30.0,
            interface_res_mm=10.0,
        ),
        tmp_path / "enclosed-interfaces.msh",
    )

    mesh = meshio.read(msh_path)
    tags = _triangle_tags(mesh)
    assert tags
    assert 4 in set(tags)


def test_legacy_interface_offset_defaults_to_mouth_ring():
    # ATH's default subdomain interface sits at the end of the profile
    # (solana reference: mouth z 140 + offset 10 = interface plane z 150).
    geometry = PointGridHornGeometry(
        inner_points=_make_point_grid(n_length=10),
        closed=True,
        interface_offset_mm=8.0,
    )

    specs = _normalise_interface_specs(geometry, geometry.inner_points.shape[1])
    assert len(specs) == 1
    assert specs[0].slice_index == 10


def test_explicit_interface_can_still_target_mouth_slice():
    geometry = PointGridHornGeometry(
        inner_points=_make_point_grid(n_length=10),
        closed=True,
        interfaces=(HornInterface(slice_index=10, offset_mm=8.0),),
    )

    specs = _normalise_interface_specs(geometry, geometry.inner_points.shape[1])
    assert len(specs) == 1
    assert specs[0].slice_index == 10


def test_ath_default_zmap_osse_matches_m2_clone_reference_rings():
    # ATH m2-clone (OSSE, Length=150, 32 segments) ring z values from ATH's
    # own mesh.geo, normalized by length. The OSSE default z-map bezier must
    # reproduce them to ~1e-3 of normalized length.
    from hornlab_mesher.profile_sampling import _ath_default_zmap

    ath_rings_mm = np.asarray(
        [
            0.0, 1.151, 2.614, 4.460, 6.816, 9.513, 12.784, 16.682, 20.945,
            26.031, 31.65, 38.153, 44.983, 52.676, 60.799, 69.276, 77.812,
            86.517, 94.876, 102.809, 110.245, 116.775, 122.919, 128.154,
            132.821, 136.663, 140.101, 142.909, 145.147, 147.014, 148.385,
            149.371, 150.0,
        ],
        dtype=np.float64,
    )

    ours = _ath_default_zmap(32, "OSSE")

    assert ours.shape == (33,)
    assert np.max(np.abs(ours - ath_rings_mm / 150.0)) < 1.5e-3


def test_ath_default_zmap_rosse_keeps_exact_reference_table():
    from hornlab_mesher.profile_sampling import _ATH_T_20, _ath_default_zmap

    assert np.array_equal(_ath_default_zmap(20, "R-OSSE"), _ATH_T_20)


def test_symmetry_plane_slivers_are_removed_and_real_defects_raise():
    from hornlab_mesher.mesher import MesherError, _remove_symmetry_plane_slivers

    points = np.asarray(
        [
            # sliver entirely in x=0 (tiny area)
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.05, 0.1],
            # normal triangle off the plane
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            # large triangle entirely in x=0 (real defect)
            [0.0, 0.0, 5.0],
            [0.0, 10.0, 5.0],
            [0.0, 5.0, 15.0],
        ],
        dtype=np.float64,
    )
    sliver_and_normal = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    phys = np.asarray([1, 1], dtype=np.int32)

    kept, kept_phys = _remove_symmetry_plane_slivers(points, sliver_and_normal, phys, ("x", "y"))
    assert len(kept) == 1
    assert kept[0].tolist() == [3, 4, 5]
    assert kept_phys.tolist() == [1]

    with_defect = np.asarray([[3, 4, 5], [6, 7, 8]], dtype=np.int64)
    with pytest.raises(MesherError, match="symmetry plane"):
        _remove_symmetry_plane_slivers(points, with_defect, np.asarray([1, 1], dtype=np.int32), ("x",))

    # no symmetry axes: untouched
    same, _ = _remove_symmetry_plane_slivers(points, with_defect, np.asarray([1, 1], dtype=np.int32), ())
    assert len(same) == 2
