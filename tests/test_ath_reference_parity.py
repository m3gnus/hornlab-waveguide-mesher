from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import meshio

from hornlab_mesher.cli import build_from_config, build_geometry_params, load_config
from hornlab_mesher.cli import _bool, _enclosure_from_config, _reshape_grid, _section
from hornlab_mesher.geometry import PointGridHornGeometry
from hornlab_mesher.mesher import _triangles_and_physical_tags
from hornlab_mesher.profiles import build_point_grid, eval_param
from hornlab_mesher.builders.point_grid import build_point_grid as build_point_grid_geometry


_ATH_REFERENCE_ROOT_TEXT = os.environ.get("ATH_REFERENCE_ROOT")
ATH_REFERENCE_ROOT = Path(_ATH_REFERENCE_ROOT_TEXT) if _ATH_REFERENCE_ROOT_TEXT else Path()
HAS_ATH_REFERENCE_ROOT = bool(_ATH_REFERENCE_ROOT_TEXT) and ATH_REFERENCE_ROOT.exists()


def _read_grid_export_blocks(path: Path) -> list[np.ndarray]:
    blocks: list[np.ndarray] = []
    current: list[list[float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append(np.asarray(current, dtype=np.float64))
                current = []
            continue
        current.append([float(part) for part in line.split(";")])
    if current:
        blocks.append(np.asarray(current, dtype=np.float64))
    return blocks


def _reference_file(case: str, suffix: str) -> Path:
    matches = sorted((ATH_REFERENCE_ROOT / case).glob(f"*_{suffix}_throat.csv"))
    if not matches:
        raise FileNotFoundError(f"no ATH {suffix} export for {case}")
    return matches[0]


def _reference_mesh_file(case: str) -> Path:
    mesh_path = ATH_REFERENCE_ROOT / case / "ABEC_FreeStanding" / f"{case}.msh"
    if not mesh_path.exists():
        raise FileNotFoundError(f"no ATH .msh reference for {case}")
    return mesh_path


def _points_in_mm(mesh: meshio.Mesh) -> np.ndarray:
    points = np.asarray(mesh.points, dtype=np.float64)
    span = float(np.max(np.ptp(points, axis=0)))
    return points * 1000.0 if span < 10.0 else points


def _physical_tag_counts(mesh: meshio.Mesh) -> dict[int, int]:
    _triangles, tags = _triangles_and_physical_tags(mesh)
    unique, counts = np.unique(tags, return_counts=True)
    return {int(tag): int(count) for tag, count in zip(unique, counts)}


def _assert_counts_close(actual: dict[int, int], expected: dict[int, int], *, rtol: float) -> None:
    assert set(actual) == set(expected)
    for tag, expected_count in expected.items():
        actual_count = actual[tag]
        limit = max(2.0, rtol * float(expected_count))
        assert abs(actual_count - expected_count) <= limit, (
            f"tag {tag} count {actual_count} differs from ATH {expected_count} "
            f"by more than {limit:.1f}"
        )


def _built_geometry_from_ath_case(case: str):
    config = load_config(ATH_REFERENCE_ROOT / case / "config.txt")
    params, _formula, _mode = build_geometry_params(config)
    mesh = _section(config, "mesh")
    enclosure = _section(config, "enclosure")
    enclosure_obj = _enclosure_from_config(config, mesh, enclosure)
    grid = build_point_grid(params)
    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    inner = _reshape_grid(grid["inner_points"], n_phi, n_length, "inner_points")
    outer = None
    if grid.get("outer_points") is not None and enclosure_obj is None:
        outer = _reshape_grid(grid["outer_points"], n_phi, n_length, "outer_points")
    geometry = PointGridHornGeometry(
        inner_points=inner,
        outer_points=outer,
        wall_thickness_mm=float(params["wallThickness"] or 0.0),
        preserve_grid=_bool(mesh, names=("preserve_grid", "preserveGrid"), default=False),
        closed=bool(grid.get("full_circle", True)),
        source_shape=int(float(params["sourceShape"])) if params.get("sourceShape") is not None else 1,
        source_radius_mm=float(params["sourceRadius"]) if params.get("sourceRadius") is not None else -1.0,
        source_curv=int(float(params["sourceCurv"])) if params.get("sourceCurv") is not None else 0,
        source_auto_angle_deg=float(eval_param(params.get("a0"), 0.0, 15.5)),
        interface_offset_mm=float(params.get("interfaceOffset", 0.0) or 0.0),
        enclosure=enclosure_obj,
    )
    return build_point_grid_geometry(geometry)


@pytest.mark.skipif(not HAS_ATH_REFERENCE_ROOT, reason="ATH_REFERENCE_ROOT reference archive not available")
@pytest.mark.parametrize("case", ["asro2", "250917asro68", "250917asro68q"])
def test_rosse_point_grid_matches_ath_reference_exports(case: str):
    params, formula, _mode = build_geometry_params(load_config(ATH_REFERENCE_ROOT / case / "config.txt"))
    assert formula == "R-OSSE"

    grid = build_point_grid(params)
    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    inner = np.asarray(grid["inner_points"], dtype=np.float64).reshape(n_phi, n_length + 1, 3)

    # ATH exports are scaled by GridExport.Scale=.1 in these fixtures.
    profiles = _read_grid_export_blocks(_reference_file(case, "profiles"))
    slices = _read_grid_export_blocks(_reference_file(case, "slices"))
    profile_ref = np.stack(profiles[:n_phi], axis=0) / 0.1
    slice_ref = np.stack([block[:n_phi] for block in slices], axis=1) / 0.1

    assert profile_ref.shape == inner.shape
    assert slice_ref.shape == inner.shape
    assert np.allclose(profile_ref, inner, rtol=0.0, atol=2.0e-4)
    assert np.allclose(slice_ref, inner, rtol=0.0, atol=2.0e-4)


@pytest.mark.skipif(not HAS_ATH_REFERENCE_ROOT, reason="ATH_REFERENCE_ROOT reference archive not available")
@pytest.mark.parametrize(
    ("case", "expected_groups"),
    [
        # Outer counts grew by two patches per quadrant when the freestanding
        # outer wall was extended to the rear cap (commit 817aea5).
        (
            "asro2",
            {"inner": 40, "outer": 38, "mouth": 2, "rear": 2, "rear_cap": 2, "throat_disc": 1},
        ),
        (
            "250917asro68",
            {"inner": 160, "outer": 152, "mouth": 8, "rear": 8, "rear_cap": 8, "throat_disc": 4},
        ),
        (
            "250917asro68q",
            {"inner": 40, "outer": 38, "mouth": 2, "rear": 2, "rear_cap": 2, "throat_disc": 1},
        ),
    ],
)
def test_rosse_freestanding_surface_topology_matches_ath_geo(case: str, expected_groups: dict[str, int]):
    gmsh = pytest.importorskip("gmsh")
    initialized_here = False
    if not gmsh.isInitialized():
        gmsh.initialize()
        initialized_here = True
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add(case)
        built = _built_geometry_from_ath_case(case)
        actual_groups = {name: len(tags) for name, tags in built.mesh_surface_groups.items()}
    finally:
        gmsh.clear()
        if initialized_here and gmsh.isInitialized():
            gmsh.finalize()

    assert actual_groups == expected_groups


@pytest.mark.skipif(not HAS_ATH_REFERENCE_ROOT, reason="ATH_REFERENCE_ROOT reference archive not available")
@pytest.mark.parametrize("case", ["250728solana", "250728solana-q"])
def test_solana_point_grid_matches_ath_reference_exports(case: str):
    params, formula, _mode = build_geometry_params(load_config(ATH_REFERENCE_ROOT / case / "config.txt"))
    assert formula == "OSSE"

    grid = build_point_grid(params)
    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    inner = np.asarray(grid["inner_points"], dtype=np.float64).reshape(n_phi, n_length + 1, 3)

    profiles = _read_grid_export_blocks(_reference_file(case, "profiles"))
    slices = _read_grid_export_blocks(_reference_file(case, "slices"))
    profile_ref = np.stack(profiles[:n_phi], axis=0) / 0.1
    slice_ref = np.stack([block[:n_phi] for block in slices], axis=1) / 0.1

    assert profile_ref.shape == inner.shape
    assert slice_ref.shape == inner.shape
    assert np.allclose(profile_ref, inner, rtol=0.0, atol=2.0e-4)
    assert np.allclose(slice_ref, inner, rtol=0.0, atol=2.0e-4)


@pytest.mark.skipif(not HAS_ATH_REFERENCE_ROOT, reason="ATH_REFERENCE_ROOT reference archive not available")
@pytest.mark.parametrize(
    ("case", "expected_groups"),
    [
        (
            "250728solana",
            {
                "inner": 4,
                "throat_disc": 4,
                "interface": 12,
                "enclosure": 44,
                "enclosure_edges_front": 12,
                "enclosure_edges_back": 12,
            },
        ),
        (
            "250728solana-q",
            {
                "inner": 1,
                "throat_disc": 1,
                "interface": 3,
                "enclosure": 11,
                "enclosure_edges_front": 3,
                "enclosure_edges_back": 3,
            },
        ),
    ],
)
def test_solana_enclosure_topology_uses_fast_wall_groups(case: str, expected_groups: dict[str, int]):
    gmsh = pytest.importorskip("gmsh")
    initialized_here = False
    if not gmsh.isInitialized():
        gmsh.initialize()
        initialized_here = True
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add(case)
        built = _built_geometry_from_ath_case(case)
        actual_groups = {name: len(tags) for name, tags in built.mesh_surface_groups.items()}
    finally:
        gmsh.clear()
        if initialized_here and gmsh.isInitialized():
            gmsh.finalize()

    assert actual_groups == expected_groups


@pytest.mark.skipif(not HAS_ATH_REFERENCE_ROOT, reason="ATH_REFERENCE_ROOT reference archive not available")
@pytest.mark.parametrize("case", ["250728solana", "250728solana-q"])
def test_solana_enclosure_mesh_stays_close_to_ath_reference(case: str, tmp_path: Path):
    generated = build_from_config(
        load_config(ATH_REFERENCE_ROOT / case / "config.txt"),
        tmp_path / f"{case}.msh",
    )
    actual_mesh = meshio.read(generated.mesh_path)
    reference_mesh = meshio.read(_reference_mesh_file(case))

    actual_points = _points_in_mm(actual_mesh)
    reference_points = _points_in_mm(reference_mesh)
    assert np.allclose(
        np.min(actual_points, axis=0),
        np.min(reference_points, axis=0),
        rtol=0.0,
        atol=1.0,
    )
    assert np.allclose(
        np.max(actual_points, axis=0),
        np.max(reference_points, axis=0),
        rtol=0.0,
        atol=1.0,
    )

    actual_triangles, _actual_tags = _triangles_and_physical_tags(actual_mesh)
    reference_triangles, _reference_tags = _triangles_and_physical_tags(reference_mesh)
    assert abs(len(actual_triangles) - len(reference_triangles)) <= 0.15 * len(reference_triangles)
    _assert_counts_close(
        _physical_tag_counts(actual_mesh),
        _physical_tag_counts(reference_mesh),
        rtol=0.25,
    )


@pytest.mark.skipif(not HAS_ATH_REFERENCE_ROOT, reason="ATH_REFERENCE_ROOT reference archive not available")
@pytest.mark.parametrize(
    "case",
    ["asro2", "250917asro68", "250917asro68q", "250728solana", "250728solana-q"],
)
def test_ath_reference_configs_build_end_to_end(case: str, tmp_path: Path):
    result = build_from_config(load_config(ATH_REFERENCE_ROOT / case / "config.txt"), tmp_path / f"{case}.msh")

    assert result.n_vertices > 0
    assert result.n_triangles > 0
    assert result.physical_groups[1] == "SD1G0"
    assert result.physical_groups[2] == "SD1D1001"
    if case.startswith("250728solana"):
        assert result.physical_groups[3] == "SD2G0"
        assert result.physical_groups[4] == "I1-2"


@pytest.mark.skipif(not HAS_ATH_REFERENCE_ROOT, reason="ATH_REFERENCE_ROOT reference archive not available")
def test_tritonia_scale_and_vertical_offset_match_ath_reference():
    config = load_config(ATH_REFERENCE_ROOT / "260308tritonia" / "260308tritonia.txt")
    params, formula, _mode = build_geometry_params(config)
    assert formula == "OSSE"

    grid = build_point_grid(params)
    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    inner = np.asarray(grid["inner_points"], dtype=np.float64).reshape(n_phi, n_length + 1, 3)

    # ATH bem_mesh.geo facts: Scale = 0.702 multiplies the geometry (throat
    # radius 18.091 -> 12.700, length 135 -> 94.77). Mesh.VerticalOffset (y = 80)
    # is NOT baked into the grid: it is reported as metadata and applied as a rigid
    # +y translation at the mesh/preview terminals, so the grid stays centred on
    # the axis here.
    assert grid["vertical_offset_mm"] == pytest.approx(80.0)
    throat_ring = inner[:, 0, :]
    throat_radii = np.hypot(throat_ring[:, 0], throat_ring[:, 1])
    assert np.allclose(throat_radii, 12.6999, rtol=0.0, atol=2.0e-3)
    assert abs(float(np.max(inner[:, :, 2])) - 94.77) < 1.0e-2
    # Implicit morph extents are ceiled BEFORE scaling: 208 / 180 raw mm.
    mouth = inner[:, -1, :]
    assert abs(float(np.max(np.abs(mouth[:, 0]))) - 208.0 * 0.702) < 1.0e-6
    assert abs(float(np.max(np.abs(mouth[:, 1]))) - 180.0 * 0.702) < 1.0e-6


def test_tritonia_rounded_rect_angular_grid_matches_current_ath_nodes():
    config = {
        "formula": "OSSE",
        "profile": {
            "L": 129.874,
            "r0": 12.7,
            "a": "49.5128 - 14.8405*cos(p)^2 - 29.2180*sin(p)^8",
            "a0": 8.0307,
            "k": 1.0910,
            "n": 6.04550,
            "q": 0.98349,
            "s": 0.9859,
        },
        "mesh": {
            "angularSegments": 80,
            "cornerSegments": 4,
            "lengthSegments": 32,
            "quadrants": 1,
            "samplingMode": "ath-default-zmap",
        },
        "morph": {
            "morphTarget": 1,
            "morphCorner": 18.0,
            "morphRate": 1.6002,
            "morphWidth": 325.0,
            "morphHeight": 325.0,
            "morphAllowShrinkage": 1,
        },
    }
    params, formula, _mode = build_geometry_params(config)
    assert formula == "OSSE"
    grid = build_point_grid(params)
    inner = np.asarray(grid["inner_points"], dtype=np.float64).reshape(
        int(grid["grid_n_phi"]), int(grid["grid_n_length"]) + 1, 3
    )
    expected_ath_mouth_xy_mm = np.asarray(
        [
            [162.500, 0.000],
            [162.500, 13.152],
            [162.500, 26.477],
            [162.500, 40.159],
            [162.500, 54.399],
            [162.500, 69.432],
            [162.500, 85.542],
            [162.500, 103.086],
            [162.500, 122.530],
            [162.500, 144.500],
            [160.088, 153.500],
            [153.500, 160.088],
            [144.500, 162.500],
            [119.976, 162.500],
            [98.543, 162.500],
            [79.354, 162.500],
            [61.800, 162.500],
            [45.422, 162.500],
            [29.857, 162.500],
            [14.805, 162.500],
            [0.000, 162.500],
        ],
        dtype=np.float64,
    )

    assert int(grid["grid_n_phi"]) == 21
    assert np.allclose(
        inner[:, -1, :2], expected_ath_mouth_xy_mm, rtol=0.0, atol=6.0e-4
    )


def _read_geo_point_grid(path: Path, n_rings: int, n_phi: int) -> np.ndarray:
    import re

    points: list[tuple[int, float, float, float]] = []
    pattern = re.compile(r"Point\((\d+)\)=\{([-\d.]+),([-\d.]+),([-\d.]+),")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            points.append(
                (int(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4)))
            )
    points.sort()
    grid = np.asarray([[x, y, z] for _tag, x, y, z in points], dtype=np.float64)
    assert grid.shape == (n_rings * n_phi, 3)
    return grid.reshape(n_rings, n_phi, 3)


@pytest.mark.skipif(not HAS_ATH_REFERENCE_ROOT, reason="ATH_REFERENCE_ROOT reference archive not available")
def test_m2_clone_point_grid_matches_ath_mesh_geo():
    # The m2-clone case exercises azimuth-dependent Coverage.Angle,
    # Slot.Length, and Term.n expressions plus an implicit rounded-rect morph
    # with corner segments. ATH's mesh.geo is the exact point grid it built.
    case_dir = ATH_REFERENCE_ROOT / "m2-clone"
    reference = _read_geo_point_grid(case_dir / "mesh.geo", n_rings=33, n_phi=104)

    params, formula, mode = build_geometry_params(load_config(case_dir / "config.txt"))
    assert formula == "OSSE"
    assert mode == "infinite-baffle"

    grid = build_point_grid(params)
    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    assert (n_phi, n_length + 1) == (104, 33)

    inner = np.asarray(grid["inner_points"], dtype=np.float64).reshape(n_phi, n_length + 1, 3)
    deviation = np.linalg.norm(inner.transpose(1, 0, 2) - reference, axis=2)
    # Mouth and throat rings are exact; interior rings carry the residual of
    # the fitted default z-map bezier (see _ATH_OSSE_ZMAP_BEZIER).
    assert float(deviation[0].max()) < 2.0e-3
    assert float(deviation[-1].max()) < 2.0e-3
    assert float(deviation.max()) < 0.5


@pytest.mark.skipif(not HAS_ATH_REFERENCE_ROOT, reason="ATH_REFERENCE_ROOT reference archive not available")
def test_m2_clone_infinite_baffle_build_matches_coupled_aperture_contract(tmp_path):
    case_dir = ATH_REFERENCE_ROOT / "m2-clone"
    result = build_from_config(load_config(case_dir / "config.txt"), tmp_path / "m2-clone.msh")

    assert result.mode == "infinite-baffle"
    assert result.native_symmetry_plane is None
    assert result.native_check_open_edges is True
    assert result.physical_groups[1] == "SD1G0"
    assert result.physical_groups[2] == "SD1D1001"
    assert result.physical_groups[12] == "mouth_aperture"
    assert result.metadata["apertureTag"] == 12
    assert 4 not in result.physical_groups

    actual = meshio.read(result.mesh_path)
    points = _points_in_mm(actual)
    assert abs(float(points[:, 2].max())) < 1.0e-6
    assert abs(float(points[:, 2].min()) + 150.0) < 1.0e-3

    actual_triangles, tags = _triangles_and_physical_tags(actual)
    assert len(actual_triangles) > 0
    assert {1, 2, 12}.issubset({int(tag) for tag in tags})

    edge_counts: dict[tuple[int, int], int] = {}
    for tri in actual_triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = tuple(sorted((int(a), int(b))))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    assert not [edge for edge, count in edge_counts.items() if count == 1]
    assert max(edge_counts.values()) <= 2

    aperture = tags == 12
    assert np.all(np.abs(points[actual_triangles[aperture], 2]) < 1.0e-6)
    p0 = points[actual_triangles[aperture, 0]]
    p1 = points[actual_triangles[aperture, 1]]
    p2 = points[actual_triangles[aperture, 2]]
    assert float(np.sum(np.cross(p1 - p0, p2 - p0)[:, 2])) > 0.0
