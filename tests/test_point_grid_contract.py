from __future__ import annotations

import math

import meshio
import numpy as np
import pytest

from hornlab_mesher.builders._occ import make_planar_sector_fill_from_ring
from hornlab_mesher.builders.enclosure import (
    _BAFFLE_CLEARANCE_FRACTION,
    _MIN_BAFFLE_CLEARANCE_MM,
    _add_curve_loop_from_curves,
    _clamp_edge_roundover,
    _reject_front_baffle_wall_intersections,
)
from hornlab_mesher.builders.point_grid_dispatch import build_point_grid as build_point_grid_geometry
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


def test_rear_rim_points_project_outer_throat_ring_straight_back():
    # Straight axial projection (constant x/y): cut-plane rays stay on their
    # plane and the rear cover stays at the outer throat radius like ATH's.
    # The earlier along-ray extrapolation flared past ATH's rear cover and
    # dipped below the y=0 cut plane near the seam of reduced domains.
    n_phi = 32
    outer = np.empty((n_phi, 2, 3), dtype=np.float64)
    for i in range(n_phi):
        phi = math.tau * i / n_phi
        throat_radius = 20.0 + 3.0 * math.cos(2.0 * phi)
        next_radius = 30.0 + 6.0 * math.sin(phi)
        outer[i, 0] = (throat_radius * math.cos(phi), throat_radius * math.sin(phi), 0.0)
        outer[i, 1] = (next_radius * math.cos(phi), next_radius * math.sin(phi), 10.0)

    rear = _rear_rim_points(outer, rear_z=-5.0)

    expected = outer[:, 0, :].copy()
    expected[:, 2] = -5.0
    assert np.allclose(rear, expected, rtol=0.0, atol=1.0e-9)
    # A seam ray on the y=0 plane must stay exactly on it.
    assert rear[0, 1] == 0.0


def test_freestanding_quarter_rear_stays_inside_quadrant(tmp_path):
    """Regression: a coarse rear rim used to chord a triangle into the y=0
    plane (hard build failure) and protrude a node past the cut plane."""
    config = {
        "formula": "OSSE",
        "mode": "freestanding",
        "profile": {"r0": 12.7, "a": 45.0, "a0": 8.0, "L": 90.0},
        "mesh": {
            "angularSegments": 48,
            "lengthSegments": 20,
            "quadrants": 1,
            "throatResolution": 5,
            "mouthResolution": 12,
            "rearResolution": 10.0,
            "scaleToMetres": False,
        },
    }
    out = tmp_path / "quarter-rear.msh"
    build_from_config(config, out)
    mesh = meshio.read(out)
    points = np.asarray(mesh.points)
    assert float(points[:, 0].min()) >= -1.0e-6
    assert float(points[:, 1].min()) >= -1.0e-6


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


@pytest.mark.parametrize(
    ("quadrants", "cut_axis"),
    [("12", 1), ("14", 0)],  # 12 mirrors about xz (y=0); 14 about yz (x=0)
)
def test_freestanding_half_model_boundary_lies_on_single_cut_plane(tmp_path, quadrants, cut_axis):
    # A freestanding half-model is a wall-shell horn cut on one mirror plane.
    # Unlike the quarter model (two cut planes), every open edge must lie on the
    # single cut plane; the wall, mouth rim and rear cap close every other edge.
    cfg = {
        "formula": "OSSE",
        "profile": {"L_mm": 80.0, "r0_mm": 10.0, "a_deg": 40.0, "a0_deg": 0.0},
        "mesh": {"angular_segments": 16, "length_segments": 4, "quadrants": quadrants},
    }
    result = build_from_config(cfg, tmp_path / f"fs-half-{quadrants}.msh")
    assert result.native_symmetry_plane == ("xz" if quadrants == "12" else "yz")

    mesh = meshio.read(result.mesh_path)
    triangles, tags = _triangles_and_tags(mesh)
    points = np.asarray(mesh.points, dtype=np.float64)

    boundary = _boundary_edges(triangles)
    assert boundary
    for a, b in boundary:
        assert abs(points[a][cut_axis]) < 1.0e-7
        assert abs(points[b][cut_axis]) < 1.0e-7
    # The modeled half occupies one side of the cut plane.
    assert points[:, cut_axis].min() >= -1.0e-7
    # The rigid-wall shell is a single connected component.
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


def test_open_quarter_enclosure_default_keeps_fast_inner_wall_grouping():
    gmsh = pytest.importorskip("gmsh")
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

    initialized_here = False
    if not gmsh.isInitialized():
        gmsh.initialize()
        initialized_here = True
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add("open-quarter-fast-topology")
        built = build_point_grid_geometry(
            PointGridHornGeometry(
                inner_points=inner,
                closed=False,
                preserve_grid=False,
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
            )
        )
        assert len(built.mesh_surface_groups["inner"]) == 1
    finally:
        gmsh.clear()
        if initialized_here and gmsh.isInitialized():
            gmsh.finalize()


def test_open_quarter_enclosure_preserves_inner_wall_grid_for_morphed_mouth(tmp_path):
    cfg = {
        "formula": "OSSE",
        "mode": "enclosure",
        "profile": {
            "L": "160",
            "s": "0.8",
            "n": 5,
            "h": 0,
            "a": "45 - 10*cos(1*p)^2 -32*sin(p*1)^12",
            "r0": 12.7,
            "a0": 15.5,
            "k": 0.5,
            "q": 0.993,
            "throatProfile": 1,
            "circArcTermAngle": 1,
        },
        "mesh": {
            "angularSegments": 80,
            "lengthSegments": 20,
            "cornerSegments": 4,
            "quadrants": 1,
            "wallThickness": 6,
            "throatResolution": 5,
            "mouthResolution": 25,
            "rearResolution": 40,
            "preserveGrid": True,
            "scaleToMetres": True,
        },
        "morph": {
            "morphTarget": 1,
            "morphCorner": 18,
            "morphRate": 3,
            "morphFixed": 0,
            "morphAllowShrinkage": 0,
        },
        "source": {
            "sourceShape": 0,
            "sourceRadius": -1,
            "sourceCurv": 0,
        },
        "enclosure": {
            "depth": 500,
            "space_l": 1,
            "space_t": 304,
            "space_r": 1,
            "space_b": 304,
            "edge": 1,
            "edgeType": 1,
            "frontMeshSize": 40,
            "backMeshSize": 40,
        },
    }

    build_from_config(cfg, tmp_path / "superduper-small-5-open-enclosure.msh")
    mesh = meshio.read(tmp_path / "superduper-small-5-open-enclosure.msh")
    triangles, tags = _triangles_and_tags(mesh)

    assert int(np.count_nonzero(tags == 1)) >= 900
    assert int(np.count_nonzero(tags == 2)) > 0
    assert int(np.count_nonzero(tags == 3)) > 0
    assert len(_tag_components(triangles, tags, 1)) == 1


def test_clamp_edge_roundover_leaves_flat_baffle_clearance():
    # Regression for the WG "Superduper small" quarter-symmetry enclosure solve:
    # enc_edge == enc_space (margin) left a ~0 mm flat-baffle ring that OCC dropped,
    # tearing the front-baffle-to-side-wall seam open off the symmetry plane and
    # failing the Metal solver's open-edge guard. The clamp must hold the roundover
    # a real clearance below the smallest margin, not just an exact-tangency epsilon.
    big = 1000.0  # half_w/half_h large enough not to bind

    # edge well below the margin is preserved unchanged.
    assert _clamp_edge_roundover(0.5, 5.0, big, big) == pytest.approx(0.5)

    # edge == margin (the failing case): a real flat-baffle clearance survives.
    margin = 1.0
    clamped = _clamp_edge_roundover(margin, margin, big, big)
    assert 0.0 < clamped < margin
    expected_clearance = max(_MIN_BAFFLE_CLEARANCE_MM, margin * _BAFFLE_CLEARANCE_FRACTION)
    assert margin - clamped >= expected_clearance - 1e-9

    # edge above the margin is clamped below it with the same clearance.
    clamped_over = _clamp_edge_roundover(50.0, margin, big, big)
    assert margin - clamped_over >= expected_clearance - 1e-9

    # half-width / half-height still bind when smaller than the margin.
    assert _clamp_edge_roundover(5.0, 5.0, 0.6, big) == pytest.approx(0.5)

    # a margin smaller than the minimum clearance collapses to a sharp corner.
    assert _clamp_edge_roundover(0.05, 0.05, big, big) == 0.0


@pytest.mark.parametrize(
    ("quadrants", "sym_plane", "cut_axes"),
    [
        ("1", "yz+xz", (0, 1)),  # quarter: rim on x=0 (yz) and/or y=0 (xz)
        ("12", "xz", (1,)),  # half about xz: rim on y=0 only
        ("14", "yz", (0,)),  # half about yz: rim on x=0 only
    ],
)
def test_reduced_enclosure_boundary_lies_on_cut_planes(tmp_path, quadrants, sym_plane, cut_axes):
    # Reduced-domain enclosures (quarter/half) must produce a mesh whose only
    # open edges lie on the symmetry cut plane(s) so the mirrored BEM solve sees
    # a clean reflection. Three distinct defects used to break this: the quarter
    # throat left an off-plane open-edge ring (the source cap and BSpline-patch
    # wall sampled the throat with mismatched phi spans), and both half models
    # built only a single quarter enclosure sector that failed to seal the
    # second quadrant (off-plane / nonmanifold edges, or no rigid-wall group).
    cfg = {
        "formula": "ROSSE",
        "mode": "enclosure",
        "profile": {"R_mm": 150.0, "r0_mm": 12.7, "a_deg": 60.0, "a0_deg": 15.5, "k": 1.0, "q": 1.0},
        "cross_section": {"exponent": 2.0, "aspect_ratio": 1.0},
        "mesh": {
            "angular_segments": 32,
            "length_segments": 16,
            "throat_res_mm": 5.0,
            "mouth_res_mm": 26.0,
            "rear_res_mm": 25.0,
            "quadrants": quadrants,
        },
        "enclosure": {
            "depth_mm": 220.0,
            "space_l_mm": 25.0,
            "space_t_mm": 25.0,
            "space_r_mm": 25.0,
            "space_b_mm": 25.0,
            "edge_mm": 18.0,
            "edge_type": 1,
            "plan_type": 1,
            "plan_n": 2.0,
        },
    }
    result = build_from_config(cfg, tmp_path / f"reduced-enclosure-{quadrants}.msh")
    assert result.native_symmetry_plane == sym_plane

    mesh = meshio.read(result.mesh_path)
    triangles, tags = _triangles_and_tags(mesh)
    points = np.asarray(mesh.points, dtype=np.float64)

    # Canonical groups present: rigid wall (1), source (2), enclosure wall (3).
    assert {1, 2, 3}.issubset({int(t) for t in tags})

    # Every open boundary edge lies on a requested cut plane; an off-plane open
    # edge is a hole the mirrored solve would leak through.
    boundary = _boundary_edges(triangles)
    assert boundary
    for a, b in boundary:
        assert any(
            abs(points[a][axis]) < 1.0e-7 and abs(points[b][axis]) < 1.0e-7
            for axis in cut_axes
        )

    # No nonmanifold edges: every edge is shared by at most two triangles.
    edge_owners: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = tuple(sorted((int(a), int(b))))
            edge_owners[key] = edge_owners.get(key, 0) + 1
    assert max(edge_owners.values()) <= 2

    # The modeled domain occupies the positive side of each cut plane, and the
    # rigid-wall shell is one connected component.
    for axis in cut_axes:
        assert points[:, axis].min() >= -1.0e-7
    assert len(_tag_components(triangles, tags, 1)) == 1


def test_reduced_enclosure_uses_requested_cut_plane_not_offset_minima(tmp_path):
    cfg = {
        "formula": "OSSE",
        "mode": "enclosure",
        "profile": {"L_mm": 80.0, "r0_mm": 10.0, "a_deg": 40.0, "a0_deg": 0.0},
        "mesh": {
            "angular_segments": 16,
            "length_segments": 4,
            "quadrants": "14",
            "vertical_offset_mm": 100.0,
        },
        "enclosure": {
            "depth_mm": 120.0,
            "space_l_mm": 25.0,
            "space_t_mm": 25.0,
            "space_r_mm": 25.0,
            "space_b_mm": 40.0,
            "edge_mm": 0.0,
            "edge_type": 1,
            "plan_type": 1,
            "plan_n": 2.0,
        },
    }

    result = build_from_config(cfg, tmp_path / "shifted-half-yz-enclosure.msh")
    assert result.native_symmetry_plane == "yz"

    mesh = meshio.read(result.mesh_path)
    points = np.asarray(mesh.points, dtype=np.float64)
    # `quadrants="14"` is cut only on x=0. A positive y offset must still apply
    # bottom spacing instead of inventing a physical wall at y=0.
    assert points[:, 0].min() >= -1.0e-7
    assert points[:, 1].min() < -1.0e-4


def test_viewport_reduced_enclosure_uses_requested_cut_plane_not_offset_minima():
    cfg = {
        "formula": "OSSE",
        "mode": "enclosure",
        "profile": {"L_mm": 80.0, "r0_mm": 10.0, "a_deg": 40.0, "a0_deg": 0.0},
        "mesh": {
            "angular_segments": 16,
            "length_segments": 4,
            "quadrants": "14",
            "vertical_offset_mm": 100.0,
        },
        "enclosure": {
            "depth_mm": 120.0,
            "space_l_mm": 25.0,
            "space_t_mm": 25.0,
            "space_r_mm": 25.0,
            "space_b_mm": 40.0,
            "edge_mm": 0.0,
            "edge_type": 1,
            "plan_type": 1,
            "plan_n": 2.0,
        },
    }

    geometry = build_viewport_geometry_from_config(cfg)
    enclosure = geometry["enclosure"]
    assert enclosure is not None
    front = np.asarray(enclosure["front_outer_points"], dtype=np.float64).reshape(-1, 3)

    assert np.isclose(front[:, 0].min(), 0.0)
    assert front[:, 1].min() < 0.0


@pytest.mark.parametrize(
    ("quadrants", "sym_plane", "cut_axes"),
    [
        ("1", "yz+xz", (0, 1)),  # quarter: rim on x=0 (yz) and/or y=0 (xz)
        ("12", "xz", (1,)),  # half about xz: rim on y=0 only
        ("14", "yz", (0,)),  # half about yz: rim on x=0 only
    ],
)
def test_reduced_enclosure_sharp_edge_boundary_lies_on_cut_planes(
    tmp_path, quadrants, sym_plane, cut_axes
):
    # Regression for a reduced-domain enclosure with a perfectly sharp box edge
    # (enc_edge=0, no roundover). The rounded-rectangle sector builder insets the
    # front/back rims by the roundover radius; at radius 0 those inset points
    # collapse onto the outer points, so the roundover/corner "arcs" became
    # zero-length lines OCC rejected with "Could not create line" -- the build
    # raised MesherError. The sector now builds a true sharp box (four faces, no
    # roundover surfaces) and must still seal the domain exactly like the rounded
    # case. This is the opposite extreme from the enc_edge==enc_space sliver that
    # _clamp_edge_roundover guards: there the roundover is too large, here it is
    # absent. (The closed quadrants="1234" path shares the same sector builder
    # and is covered by the canonical-tag enclosure tests.)
    cfg = {
        "formula": "ROSSE",
        "mode": "enclosure",
        "profile": {"R_mm": 150.0, "r0_mm": 12.7, "a_deg": 60.0, "a0_deg": 15.5, "k": 1.0, "q": 1.0},
        "cross_section": {"exponent": 2.0, "aspect_ratio": 1.0},
        "mesh": {
            "angular_segments": 32,
            "length_segments": 16,
            "throat_res_mm": 5.0,
            "mouth_res_mm": 26.0,
            "rear_res_mm": 25.0,
            "quadrants": quadrants,
        },
        "enclosure": {
            "depth_mm": 220.0,
            "space_l_mm": 25.0,
            "space_t_mm": 25.0,
            "space_r_mm": 25.0,
            "space_b_mm": 25.0,
            "edge_mm": 0.0,  # sharp box edge: the regression trigger.
            "edge_type": 1,
            "plan_type": 1,
            "plan_n": 2.0,
        },
    }
    # The build itself must succeed -- it used to raise MesherError here.
    result = build_from_config(cfg, tmp_path / f"reduced-enclosure-sharp-{quadrants}.msh")
    assert result.native_symmetry_plane == sym_plane

    mesh = meshio.read(result.mesh_path)
    triangles, tags = _triangles_and_tags(mesh)
    points = np.asarray(mesh.points, dtype=np.float64)

    # Canonical groups present: rigid wall (1), source (2), enclosure wall (3).
    assert {1, 2, 3}.issubset({int(t) for t in tags})

    # Every open boundary edge lies on a requested cut plane; an off-plane open
    # edge is a hole the mirrored solve would leak through.
    boundary = _boundary_edges(triangles)
    assert boundary
    for a, b in boundary:
        assert any(
            abs(points[a][axis]) < 1.0e-7 and abs(points[b][axis]) < 1.0e-7
            for axis in cut_axes
        )

    # No nonmanifold edges: every edge is shared by at most two triangles. The
    # half models weld two sharp sectors along the off-cut axis; a mismatched
    # seam would surface here as a tripled edge.
    edge_owners: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = tuple(sorted((int(a), int(b))))
            edge_owners[key] = edge_owners.get(key, 0) + 1
    assert max(edge_owners.values()) <= 2

    # The modeled domain occupies the positive side of each cut plane, and the
    # rigid-wall shell is one connected component.
    for axis in cut_axes:
        assert points[:, axis].min() >= -1.0e-7
    assert len(_tag_components(triangles, tags, 1)) == 1


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


def test_remove_degenerate_triangles_drops_needle_slivers():
    from hornlab_mesher.normals import remove_degenerate_triangles

    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0001, 0.0],  # needle apex: 2 mm long, 0.1 um high
            [0.0, 5.0, 0.0],
        ],
        dtype=np.float64,
    )
    tris = np.asarray([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
    tags = np.asarray([1, 1], dtype=np.int32)

    kept, kept_tags, removed = remove_degenerate_triangles(points, tris, tags, min_quality=1.0e-4)
    assert removed == 1
    assert kept.tolist() == [[0, 1, 3]]

    # without the quality threshold the needle survives (area is nonzero)
    kept_all, _t, removed_none = remove_degenerate_triangles(points, tris, tags)
    assert removed_none == 0 and len(kept_all) == 2


def test_weld_near_duplicate_vertices_merges_micrometre_pairs():
    from hornlab_mesher.mesher import _compact_unused_vertices, _weld_near_duplicate_vertices

    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [10.0, 0.0023, 0.0],  # 2.3 um from vertex 1
            [10.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [3, 4, 2]], dtype=np.int64)

    welded = _weld_near_duplicate_vertices(points, triangles, tol_mm=5.0e-3)
    assert welded.tolist() == [[0, 1, 2], [1, 4, 2]]

    compact_points, compact_triangles = _compact_unused_vertices(points, welded)
    assert len(compact_points) == 4
    assert compact_triangles.max() == 3

    # well-separated vertices stay untouched
    same = _weld_near_duplicate_vertices(points[:3], np.asarray([[0, 1, 2]]), tol_mm=5.0e-3)
    assert same.tolist() == [[0, 1, 2]]


def test_lookup_point_grid_follows_pchip_profile():
    """A LOOKUP profile builds a point grid whose throat/mouth radii and
    monotonic radial growth follow the supplied dense [z, r] profile."""
    # Dense, strictly increasing r(z) profile (what the optimizer emits).
    z = np.linspace(0.0, 150.0, 121)
    r = np.interp(z, [0.0, 30.0, 70.0, 110.0, 150.0], [12.7, 28.0, 60.0, 110.0, 170.0])
    profile = [[float(zi), float(ri)] for zi, ri in zip(z, r)]

    params = {
        "type": "LOOKUP",
        "lookupProfile": profile,
        "angularSegments": 48,
        "lengthSegments": 24,
        "throatResolution": 6.0,
        "mouthResolution": 14.0,
        "rearResolution": 28.0,
        "quadrants": "1234",
        "wallThickness": 0.0,
        "encDepth": 0.0,
    }
    grid = build_point_grid(params)
    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    pts = np.asarray(grid["inner_points"], dtype=np.float64).reshape(n_phi, n_length + 1, 3)

    radial = np.hypot(pts[..., 0], pts[..., 1])
    # Throat ring ~= r0, mouth ring ~= mouth radius from the profile.
    assert np.allclose(radial[:, 0], 12.7, atol=1e-6)
    assert np.allclose(radial[:, -1], 170.0, atol=1e-6)
    # Axisymmetric base profile: every azimuth shares the same radius per ring.
    assert np.allclose(radial, radial[0:1, :], atol=1e-9)
    # Radius grows monotonically throat -> mouth for this convex profile.
    assert np.all(np.diff(radial[0]) > 0.0)
    # z spans the profile range.
    assert pts[..., 2].min() == pytest.approx(0.0, abs=1e-9)
    assert pts[..., 2].max() == pytest.approx(150.0, abs=1e-6)


def test_lookup_rejects_missing_profile():
    params = {
        "type": "LOOKUP",
        "angularSegments": 16,
        "lengthSegments": 8,
        "quadrants": "1234",
    }
    with pytest.raises(ValueError, match="lookupProfile"):
        build_point_grid(params)


@pytest.mark.parametrize(
    ("value", "canonical", "planes"),
    [
        # Ath ground truth (ath.exe under Wine, asro2 R-OSSE sweep): Mesh.Quadrants
        # is read as a leading integer; only 1234/12/14 are special, everything else
        # (incl. permutations, trailing junk, empties) is a quarter model.
        ("1", "1", ("x", "y")), ("2", "1", ("x", "y")), ("3", "1", ("x", "y")),
        ("4", "1", ("x", "y")), ("0", "1", ("x", "y")),
        ("12", "12", ("y",)), ("14", "14", ("x",)), ("1234", "1234", ()),
        ("13", "1", ("x", "y")), ("23", "1", ("x", "y")), ("24", "1", ("x", "y")),
        ("34", "1", ("x", "y")), ("123", "1", ("x", "y")), ("234", "1", ("x", "y")),
        ("21", "1", ("x", "y")), ("41", "1", ("x", "y")),  # not reordered to 12/14
        ("1234x", "1234", ()), (" 12 ", "12", ("y",)),     # trailing/leading trim
        ("x1234", "1", ("x", "y")), ("1,2", "1", ("x", "y")),  # no leading digit / stops at comma
        ("99", "1", ("x", "y")), ("foo", "1", ("x", "y")), ("", "1", ("x", "y")),
        (None, "1", ("x", "y")), (12, "12", ("y",)), (1234, "1234", ()),
    ],
)
def test_normalise_quadrants_matches_ath_atoi_rule(value, canonical, planes):
    from hornlab_mesher.profile_common import (
        _normalise_quadrants,
        _symmetry_planes_for_quadrants,
    )

    assert _normalise_quadrants(value) == canonical
    assert _symmetry_planes_for_quadrants(value) == planes


@pytest.mark.parametrize("quadrants", ["13", "foo", "5", "1x", "0", "23", "234", "1,2", ""])
def test_unrecognised_quadrants_default_to_quarter_like_ath(quadrants):
    from hornlab_mesher.config_builder import build_geometry_params

    cfg = {
        "formula": "OSSE",
        "profile": {"L_mm": 80.0, "r0_mm": 10.0, "a_deg": 40.0, "a0_deg": 0.0},
        "mesh": {"angular_segments": 16, "length_segments": 4, "quadrants": quadrants},
    }
    # ATH reads Mesh.Quadrants as a leading integer and silently treats every value it
    # does not recognise (here 13/foo/5/1x/0/23/234/"1,2"/empty) as the quarter default
    # rather than erroring. We reproduce that -- routing it to a well-defined Q1 grid,
    # not the degenerate open full-circle grid the old set-based logic used to build.
    params, _formula, _mode = build_geometry_params(cfg)
    assert params["quadrants"] == "1"


def test_quadrants_are_not_reordered_like_ath():
    from hornlab_mesher.config_builder import build_geometry_params

    # ATH reads the value as a leading integer, so "21" is the number 21 -- an
    # unrecognised value that meshes as a quarter -- NOT the digit set {1, 2} == "12".
    # Verified against ath.exe: Quadrants=21 emits a Sym=xy quarter mesh (1209 pts),
    # not the Sym=y half (2396 pts) that "12" produces. Only 12/14/1234 are special.
    for value, expected in (("21", "1"), ("41", "1"), ("12", "12"), ("14", "14"), ("1234", "1234")):
        cfg = {
            "formula": "OSSE",
            "profile": {"L_mm": 80.0, "r0_mm": 10.0, "a_deg": 40.0, "a0_deg": 0.0},
            "mesh": {"angular_segments": 16, "length_segments": 4, "quadrants": value},
        }
        params, _, _ = build_geometry_params(cfg)
        assert params["quadrants"] == expected, value


@pytest.mark.parametrize("quadrants", ["1", "12"])
def test_vertical_offset_accepted_for_y_cut_reduced_domains(quadrants):
    # ATH parity: a y-cut reduced domain (quadrants 1/12) combined with
    # Mesh.VerticalOffset is accepted, not rejected. The grid is emitted at the
    # origin (its cut edge stays on y=0) and the offset rides along as metadata;
    # it is applied later as a rigid +y translation while the y=0 (xz) symmetry
    # plane stays declared. ATH builds and mirrors exactly this way -- the shifted
    # mesh reconstructs about y=0, not the shifted plane -- so we reproduce it.
    from hornlab_mesher.config_builder import build_geometry_params

    cfg = {
        "formula": "OSSE",
        "profile": {"L_mm": 80.0, "r0_mm": 10.0, "a_deg": 40.0, "a0_deg": 0.0},
        "mesh": {
            "angular_segments": 16,
            "length_segments": 4,
            "quadrants": quadrants,
            "vertical_offset_mm": 10.0,
        },
    }

    params, _formula, _mode = build_geometry_params(cfg)
    assert float(params["verticalOffset"]) == pytest.approx(10.0)

    grid = build_point_grid(
        {
            "type": "OSSE",
            "L": 80.0,
            "r0": 10.0,
            "a": 40.0,
            "a0": 0.0,
            "angularSegments": 16,
            "lengthSegments": 4,
            "quadrants": quadrants,
            "verticalOffset": 10.0,
        }
    )
    # Offset lives in metadata; the y-cut edge is still on y=0 so every downstream
    # cut-plane snap/enclosure step keeps running on the coordinate axes.
    assert grid["vertical_offset_mm"] == pytest.approx(10.0)
    assert "y" in grid["symmetry_planes"]
    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    inner = np.asarray(grid["inner_points"], dtype=np.float64).reshape(n_phi, n_length + 1, 3)
    assert abs(float(inner[:, :, 1].min())) < 1.0e-9


@pytest.mark.parametrize(
    ("quadrants", "sym_plane"),
    [("1", "yz+xz"), ("12", "xz")],
)
def test_vertical_offset_y_cut_reproduces_ath_shifted_reduced_mesh(tmp_path, quadrants, sym_plane):
    # ATH parity for a y-cut reduced enclosure with Mesh.VerticalOffset. ATH builds
    # the reduced model on the axes, translates it by the offset, and still declares
    # the y=0 (xz) mirror -- so the reconstruction mirrors about y=0, leaving the
    # shifted mesh entirely on one side (the "split shell" the analysis flagged). We
    # reproduce it exactly: the finished mesh is placed at y >= offset, the x=0 cut
    # plane (when present) is untouched, and the declared symmetry still names y=0.
    offset_mm = 80.0
    cfg = {
        "formula": "OSSE",
        "mode": "enclosure",
        "profile": {"L_mm": 80.0, "r0_mm": 10.0, "a_deg": 40.0, "a0_deg": 0.0},
        "mesh": {
            "angular_segments": 16,
            "length_segments": 4,
            "quadrants": quadrants,
            "vertical_offset_mm": offset_mm,
        },
        "enclosure": {
            "depth_mm": 120.0,
            "space_l_mm": 25.0,
            "space_t_mm": 25.0,
            "space_r_mm": 25.0,
            "space_b_mm": 25.0,
            "edge_mm": 0.0,
            "edge_type": 1,
            "plan_type": 1,
            "plan_n": 2.0,
        },
    }

    result = build_from_config(cfg, tmp_path / f"shifted-{quadrants}-ycut-enclosure.msh")
    # The y=0 (xz) mirror stays declared even though the mesh is shifted off it.
    assert result.native_symmetry_plane == sym_plane

    mesh = meshio.read(result.mesh_path)
    points = np.asarray(mesh.points, dtype=np.float64)
    # Mesh is written in metres. The whole reduced model sits at y >= offset and
    # never reaches the declared y=0 plane, so mirroring it leaves a gap (the ATH
    # reconstruction defect we are reproducing on purpose).
    assert points[:, 1].min() == pytest.approx(offset_mm * 1.0e-3, abs=2.0e-4)
    assert points[:, 1].min() > 1.0e-4
    # The offset is applied only in y: an x=0 (yz) cut plane, when present
    # (quadrants "1"), stays exactly on x=0.
    if "yz" in sym_plane:
        assert abs(float(points[:, 0].min())) < 2.0e-4


def test_direct_profile_grid_quadrants_share_config_normalisation():
    params = {
        "type": "OSSE",
        "L": 80.0,
        "r0": 10.0,
        "a": 40.0,
        "a0": 0.0,
        "angularSegments": 16,
        "lengthSegments": 4,
        "quadrants": "21",
    }

    # "21" is the integer 21 (see test_quadrants_are_not_reordered_like_ath), an
    # unrecognised value that ATH -- and now build_point_grid -- meshes as a quarter
    # (Q1) model with both cut planes, not the "12" top half.
    grid = build_point_grid(params)

    assert grid["full_circle"] is False
    assert grid["grid_n_phi"] == 5
    assert grid["quadrants"] == "1"
    assert grid["symmetry_planes"] == ["x", "y"]

    # Every other unrecognised value is the same quarter default -- ATH never rejects
    # Mesh.Quadrants, and neither does the mesher.
    for bad in ("13", "foo", "5", "1x", "0", "234"):
        bad_grid = build_point_grid({**params, "quadrants": bad})
        assert bad_grid["quadrants"] == "1"
        assert bad_grid["symmetry_planes"] == ["x", "y"]


def test_front_baffle_guard_catches_in_plane_outside_mouth_contact():
    angles = np.asarray([0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0])
    center = np.asarray([100.0, 0.0])
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    inner_points = np.zeros((angles.size, 3, 3), dtype=np.float64)
    inner_points[:, 0, :2] = center + 5.0 * unit
    inner_points[:, 0, 2] = 0.0
    # This station lies exactly in the baffle plane but outside the mouth loop.
    # On the pi meridian its origin radius is smaller than the shifted mouth
    # point, so the old origin-radius check missed it.
    inner_points[:, 1, :2] = center + 20.0 * unit
    inner_points[:, 1, 2] = 100.0
    inner_points[:, 2, :2] = center + 10.0 * unit
    inner_points[:, 2, 2] = 100.0

    with pytest.raises(NotImplementedError, match="outside the mouth opening"):
        _reject_front_baffle_wall_intersections(inner_points, closed=True)


def test_open_front_baffle_guard_catches_tangential_outside_reduced_mouth_contact():
    mouth_xy = np.asarray(
        [
            [10.0, 0.0],
            [7.0, 7.0],
            [0.0, 10.0],
        ],
        dtype=np.float64,
    )
    inner_points = np.zeros((3, 3, 3), dtype=np.float64)
    inner_points[:, 0, :2] = mouth_xy * 0.4
    inner_points[:, 0, 2] = 0.0
    inner_points[:, 1, :2] = mouth_xy * 0.5
    inner_points[:, 1, 2] = 50.0
    inner_points[:, 2, :2] = mouth_xy
    inner_points[:, 2, 2] = 100.0
    # This contact lies on the baffle plane inside the old meridian ray guard,
    # but outside the actual quarter-domain mouth polygon.
    inner_points[1, 1, :2] = [12.0, 0.0]
    inner_points[1, 1, 2] = 100.0

    with pytest.raises(NotImplementedError, match="outside the mouth opening"):
        _reject_front_baffle_wall_intersections(inner_points, closed=False)


def test_source_shape_zero_builds_flat_disc(tmp_path):
    """sourceShape=0 (flat disc) used to be silently coerced to 1 (cap) by an
    ``or``-default; the driven surface must actually be flat."""
    def _cfg(shape):
        return {
            "formula": "OSSE",
            "mode": "bare",
            "profile": {"r0": 12.7, "a": 45.0, "a0": 8.0, "L": 60.0},
            "mesh": {
                "angularSegments": 32,
                "lengthSegments": 10,
                "quadrants": 1234,
                "wallThickness": 0,
                "throatResolution": 5,
                "mouthResolution": 12,
                "scaleToMetres": False,
            },
            "source": {"sourceShape": shape, "sourceCurv": 1},
        }

    spreads = {}
    for shape in (0, 1):
        out = tmp_path / f"flat-src-{shape}.msh"
        build_from_config(_cfg(shape), out)
        mesh = meshio.read(out)
        triangles, tags = _triangles_and_tags(mesh)
        source_nodes = np.unique(triangles[tags == 2])
        z = mesh.points[source_nodes][:, 2]
        spreads[shape] = float(z.max() - z.min())

    assert spreads[0] < 1e-6, "flat disc source must lie in the throat plane"
    assert spreads[1] > 0.1, "curved cap control should have axial depth"


def test_preserve_grid_quarter_enclosure_has_no_off_plane_open_edges(tmp_path):
    """Regression: preserve_grid enclosure builds paired a faceted wall with a
    B-spline-boundary source cap; the two only met at grid nodes, leaving an
    off-plane open seam ring at the throat."""
    config = {
        "formula": "OSSE",
        "mode": "enclosure",
        "profile": {"r0": 12.7, "a": 45.0, "a0": 8.0, "L": 90.0},
        "mesh": {
            "angularSegments": 32,
            "lengthSegments": 12,
            "quadrants": 1,
            "throatResolution": 5,
            "mouthResolution": 12,
            "preserveGrid": True,
            "scaleToMetres": False,
        },
        "enclosure": {
            "depth": 150.0,
            "space_l": 30.0,
            "space_t": 30.0,
            "space_r": 30.0,
            "space_b": 30.0,
            "edge": 15.0,
            "edgeType": 1,
        },
    }
    out = tmp_path / "preserve-grid-quarter-enclosure.msh"
    build_from_config(config, out)
    mesh = meshio.read(out)
    triangles, _tags = _triangles_and_tags(mesh)
    points = np.asarray(mesh.points, dtype=np.float64)
    off_plane = []
    for a, b in _boundary_edges(triangles):
        pa, pb = points[a], points[b]
        on_x = abs(pa[0]) < 1.0e-9 and abs(pb[0]) < 1.0e-9
        on_y = abs(pa[1]) < 1.0e-9 and abs(pb[1]) < 1.0e-9
        if not (on_x or on_y):
            off_plane.append((pa.round(3).tolist(), pb.round(3).tolist()))
    assert not off_plane, f"{len(off_plane)} off-plane open edges, e.g. {off_plane[:4]}"
