"""Imported ATH surfaces keep their authored shape through OCC and STEP."""

from pathlib import Path

import gmsh
import numpy as np
import pytest

from hornlab_mesher.cad import write_step_from_config
from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.config_parser import parse_text_config

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


def test_zero_front_offset_round_trips_without_a_degenerate_band(tmp_path):
    cfg = config("of20")
    cfg["mesh"]["athRearFrontOffset"] = 0
    path, info = write_step_from_config(cfg, tmp_path / "zero.step")
    assert path.is_file() and info.body == "solid" and info.volume_mm3 > 0


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
