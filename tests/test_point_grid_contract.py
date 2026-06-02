from __future__ import annotations

import math

import meshio
import numpy as np

from hornlab_mesher.builders._occ import make_planar_sector_fill_from_ring
from hornlab_mesher import HornEnclosure, MeshDensity, build_mesh
from hornlab_mesher.cli import build_from_config, parse_ath_config
from hornlab_mesher.geometry import PointGridHornGeometry
from hornlab_mesher.profiles import build_point_grid


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


def test_ath_parity_sampling_matches_asro2_exported_grid():
    default_grid = build_point_grid(_ASRO2_PARAMS)
    parity_grid = build_point_grid({**_ASRO2_PARAMS, "athParitySampling": True})

    assert int(parity_grid["grid_n_phi"]) == 15
    assert int(parity_grid["grid_n_length"]) == 20
    assert parity_grid["full_circle"] is False

    uniform_quarter = np.linspace(0.0, math.pi / 2.0, 15, dtype=np.float64)
    parity_angles = np.asarray(parity_grid["angle_list"], dtype=np.float64)
    assert np.allclose(parity_angles, uniform_quarter, rtol=0.0, atol=1.0e-12)
    assert len(default_grid["angle_list"]) != len(parity_grid["angle_list"])

    assert np.allclose(
        np.asarray(parity_grid["slice_map"], dtype=np.float64),
        _ATH_ASRO2_T_VALUES,
        rtol=0.0,
        atol=1.0e-9,
    )

    inner = np.asarray(parity_grid["inner_points"], dtype=np.float64).reshape(15, 21, 3)
    phi0 = inner[0]
    assert np.allclose(phi0[:, 2], _ATH_ASRO2_PHI0_Z_MM, rtol=0.0, atol=2.0e-4)
    assert np.allclose(
        np.linalg.norm(phi0[:, :2], axis=1),
        _ATH_ASRO2_PHI0_R_MM,
        rtol=0.0,
        atol=2.0e-4,
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

    assert cfg["mesh"]["athParitySampling"] is True
    result = build_from_config(cfg, tmp_path / "asro2-parity.msh")
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
    result = build_from_config(cfg, tmp_path / "flat-asro2-parity.msh")
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
