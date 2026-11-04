"""Imported ATH surfaces keep their authored shape through OCC and STEP."""

from collections import Counter, defaultdict
from pathlib import Path

import gmsh
import numpy as np
import pytest

from hornlab_mesher.cad import write_step_from_config
from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.config_parser import parse_text_config
from hornlab_mesher.step_text import step_records, step_refs

FIXTURES = Path(__file__).parent / "fixtures/ath_import/ath-v2026-09a"


def config(name):
    return parse_text_config((FIXTURES / f"{name}.cfg").read_text())


def reference_points(cfg):
    from hornlab_mesher.builders.point_grid_freestanding import (
        _build_ath_import_surfaces,
    )

    cfg["mesh"]["quadrants"] = "1234"
    geometry = resolve_geometry(cfg).geometry
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("solve-preview-reference")
        built = _build_ath_import_surfaces(geometry)
        points = []
        for role in ("inner", "outer", "mouth", "rear_cap"):
            for tag in built.mesh_surface_groups[role]:
                uv = np.random.default_rng(tag).uniform(0.03, 0.97, (11, 2))
                if len(gmsh.model.getBoundary([(2, tag)])) == 3:
                    uv[:, 1] *= uv[:, 0]
                xyz = np.asarray(gmsh.model.getValue(2, tag, uv.ravel())).reshape(-1, 3)
                if role == "rear_cap":
                    # Opening the throat intentionally removes this central
                    # portion of the acoustic rear fan.
                    radius = np.linalg.norm(
                        geometry.inner_points[:, 0, :2], axis=1
                    ).max()
                    xyz = xyz[np.linalg.norm(xyz[:, :2], axis=1) > radius + 1]
                points.extend(xyz)
        return np.asarray(points)
    finally:
        gmsh.finalize()


def assert_surface_distance(points, tolerance=0.020):
    distances = np.full(len(points), np.inf)
    for _, tag in gmsh.model.getEntities(2):
        box = np.asarray(gmsh.model.getBoundingBox(2, tag))
        selected = np.flatnonzero(
            np.all(
                (points >= box[:3] - tolerance) & (points <= box[3:] + tolerance),
                axis=1,
            )
        )
        if len(selected):
            closest, _ = gmsh.model.getClosestPoint(2, tag, points[selected].ravel())
            distance = np.linalg.norm(
                points[selected] - np.asarray(closest).reshape(-1, 3), axis=1
            )
            distances[selected] = np.minimum(distances[selected], distance)
    assert distances.max() <= tolerance


def assert_single_open_shell(path, n_faces):
    records = step_records(path.read_text())
    shells = [value for value in records.values() if value.startswith("OPEN_SHELL(")]
    assert len(shells) == 1
    assert len(step_refs(shells[0])) == n_faces
    assert not any(
        value.startswith(("CLOSED_SHELL(", "MANIFOLD_SOLID_BREP("))
        for value in records.values()
    )
    assert set(step_refs(shells[0])) == {
        tag for tag, value in records.items() if value.startswith("ADVANCED_FACE(")
    }


@pytest.mark.parametrize("name", ["p1", "onr", "rs1", "coreabec"])
def test_imported_step_round_trip(tmp_path, name):
    cfg = config(name)
    points = reference_points(cfg)
    path, info = write_step_from_config(cfg, tmp_path / "import.step")
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("round-trip")
        gmsh.model.occ.importShapes(str(path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        solids = gmsh.model.getEntities(3)
        if name == "p1":
            assert info.body == "surface" and not solids
            assert_single_open_shell(path, info.n_faces)
        else:
            assert info.body == "solid" and len(solids) == 1
            assert gmsh.model.occ.getMass(*solids[0]) == pytest.approx(
                info.volume_mm3, rel=1e-7
            )
            assert info.throat_opened
            assert not gmsh.model.isInside(3, solids[0][1], [0, 0, -5])
            boundary = gmsh.model.getBoundary(
                gmsh.model.getEntities(2), combined=False, oriented=False
            )
            tags, counts = np.unique([tag for _, tag in boundary], return_counts=True)
            assert all(
                count == 2 or gmsh.model.occ.getMass(1, int(tag)) < 1e-9
                for tag, count in zip(tags, counts)
            )
        assert len(gmsh.model.getEntities(2)) == info.n_faces
        assert np.isfinite(info.bounding_box_mm).all()
        assert_surface_distance(points)
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("faceted, angular_segments", [(False, 8), (False, 16), (True, 8)])
@pytest.mark.parametrize("open_throat", [False, True])
def test_bare_lip_is_one_connected_sheet_with_open_rims(
    tmp_path, faceted, angular_segments, open_throat
):
    cfg = config("p1")
    cfg["source"] = {"source_shape": 0}
    cfg["mesh"]["angularSegments"] = angular_segments
    if faceted:
        cfg["mesh"].update(topology="legacy", preserve_grid=True)
    cfg["mesh"]["quadrants"] = "1234"
    grid = resolve_geometry(cfg).geometry.inner_points
    if faceted:
        # These circular authored rings give planar quadrilaterals. Their
        # centres lie inside the trimmed facets, unlike arbitrary plane UVs.
        adjacent = np.roll(grid, -1, axis=0)
        points = (
            (grid[:, :-1] + grid[:, 1:] + adjacent[:, :-1] + adjacent[:, 1:]) / 4
        ).reshape(-1, 3)
    else:
        points = reference_points(cfg)
    path, info = write_step_from_config(
        cfg, tmp_path / "bare.step", open_throat=open_throat
    )
    assert info.body == "surface" and info.volume_mm3 is None
    assert info.throat_opened == open_throat
    assert_single_open_shell(path, info.n_faces)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("connected-sheet")
        gmsh.model.occ.importShapes(str(path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        assert not gmsh.model.getEntities(3)
        faces = {tag for _, tag in gmsh.model.getEntities(2)}
        assert len(faces) == info.n_faces
        owners = defaultdict(set)
        for face in faces:
            for _, edge in gmsh.model.getBoundary([(2, face)], oriented=False):
                owners[edge].add(face)
        counts = Counter(len(group) for group in owners.values())
        # Both authored rim rings have eight edges; closing the source only
        # removes the throat rim. Every interior edge has two owning faces.
        assert counts[1] == (16 if open_throat else 8)
        assert counts[2] == (408 if open_throat else 424 if faceted else 420)
        assert set(counts) == {1, 2}
        reached = {next(iter(faces))}
        while True:
            expanded = reached | set().union(
                *(group for group in owners.values() if group & reached)
            )
            if expanded == reached:
                break
            reached = expanded
        assert reached == faces
        # The remaining free edges form the mouth ring and, when requested,
        # the throat ring, with no holes elsewhere in the sheet.
        rims = Counter()
        for edge, group in owners.items():
            if len(group) != 1:
                continue
            low, high = gmsh.model.getParametrizationBounds(1, edge)
            xyz = gmsh.model.getValue(1, edge, [(low[0] + high[0]) / 2])
            if abs(xyz[2] - grid[0, 0, 2]) < 1e-7:
                rims["throat"] += 1
            else:
                assert xyz[2] == pytest.approx(grid[0, -1, 2], abs=1e-7)
                rims["mouth"] += 1
        assert rims == ({"mouth": 8, "throat": 8} if open_throat else {"mouth": 8})
        assert_surface_distance(points)
    finally:
        gmsh.finalize()


def test_zero_front_offset_round_trips_without_a_degenerate_band(tmp_path):
    cfg = config("of20")
    cfg["mesh"]["athRearFrontOffset"] = 0
    path, info = write_step_from_config(cfg, tmp_path / "zero.step")
    assert path.is_file() and info.body == "solid" and info.volume_mm3 > 0


def test_disconnected_bare_lip_refuses_without_replacing_the_target(
    tmp_path, monkeypatch
):
    from hornlab_mesher.errors import UserRefusal

    add_loop = gmsh.model.occ.addSurfaceLoop
    monkeypatch.setattr(
        gmsh.model.occ,
        "addSurfaceLoop",
        lambda faces, **kwargs: add_loop(faces, sewing=False),
    )
    path = tmp_path / "kept.step"
    path.write_bytes(b"previous STEP")
    with pytest.raises(UserRefusal) as error:
        write_step_from_config(config("p1"), path)
    assert "one connected sheet" in error.value.user_message
    assert "Export STL" in error.value.next_step
    assert path.read_bytes() == b"previous STEP"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["kept.step"]


@pytest.mark.parametrize("faceted", [False, True])
def test_closed_throat_and_preserve_grid_transport(tmp_path, faceted):
    cfg = config("onr")
    cfg["mesh"].update(angularSegments=16, lengthSegments=8)
    cfg["source"] = {"source_shape": 0}
    if faceted:
        cfg["mesh"].update(topology="legacy", preserve_grid=True)
    path, info = write_step_from_config(
        cfg, tmp_path / "closed.step", open_throat=False
    )
    assert info.body == "solid" and not info.throat_opened
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("closed-throat")
        gmsh.model.occ.importShapes(str(path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        assert len(gmsh.model.getEntities(3)) == 1
    finally:
        gmsh.finalize()


def test_sparse_spherical_closed_source_refuses_with_an_open_throat_remedy(tmp_path):
    from hornlab_mesher.errors import UserRefusal

    cfg = config("onr")
    cfg["mesh"].update(angularSegments=16, lengthSegments=8)
    with pytest.raises(UserRefusal) as error:
        write_step_from_config(cfg, tmp_path / "source.step", open_throat=False)
    assert "spherical source cap" in error.value.user_message
    assert "open throat" in error.value.next_step
    assert not (tmp_path / "source.step").exists()


def test_rosse_opus_uses_the_same_ring_transport(tmp_path):
    cfg = parse_text_config(
        "R-OSSE={\nR=80\nr0=8\na=45\na0=5\n}\nABEC.SimType=2\n"
        "Mesh.AngularSegments=16\nMesh.LengthSegments=8\nMesh.WallThickness=3\n"
        "Mesh.RearShape=3\nMesh.RearShape.FrontOffset=10\n"
    )
    _, info = write_step_from_config(cfg, tmp_path / "rosse.step")
    assert info.body == "solid" and info.throat_opened and info.volume_mm3 > 0


def test_failed_transport_preserves_the_target_and_actionable_error(
    tmp_path, monkeypatch
):
    from hornlab_mesher.builders import ath_cad
    from hornlab_mesher.errors import UserRefusal

    def refuse(*args):
        raise UserRefusal(
            "STEP export",
            "The ATH patch cannot be copied.",
            "Export STL.",
            diagnostic="fitting failure",
        )

    monkeypatch.setattr(ath_cad._PatchTransport, "_copy", refuse)
    path = tmp_path / "kept.step"
    path.write_bytes(b"previous STEP")
    with pytest.raises(UserRefusal) as error:
        write_step_from_config(config("p1"), path)
    assert error.value.next_step == "Export STL."
    assert path.read_bytes() == b"previous STEP"
