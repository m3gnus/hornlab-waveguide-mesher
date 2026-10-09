"""Shared enclosure source geometry, independent topology and surface checks."""

import json
import math
from dataclasses import replace

import meshio
import numpy as np
import pytest
from assembly_oracle import certificate_xyz, independent_surfaces
from hornlab_mesher.assembly_artifact import export_assembly
from hornlab_mesher.source_assembly import SourceAssembly, assembly_channels
from hornlab_mesher.source_contour import ContourDrive, cone, dome, flat


def case(curved=False):
    horn = dome(4, 1) if curved else flat(4)
    horn = replace(horn, physical_source_id="horn", rim_id="horn.rim")
    woofer = (
        cone(
            10,
            4,
            cap_radius_mm=3,
            cap_height_mm=2,
            surround_width_mm=2,
            surround_depth_mm=0.6,
            land_width_mm=1,
        )
        if curved
        else flat(8)
    )
    woofer = replace(woofer, physical_source_id="woofer", rim_id="woofer.rim")
    model = SourceAssembly(
        horn, woofer, 70, 90, 40, 7, (-13, 17), 24, 10, (9, -15), 14 if curved else 8
    )
    drives = [
        ContourDrive(
            "hf", tuple((s.id, 1) for s in horn.segments if s.role == "moving")
        ),
        ContourDrive(
            "lf",
            tuple(
                zip(
                    [s.id for s in woofer.segments if s.role == "moving"],
                    [1, -0.5, 0] if curved else [1],
                )
            ),
            "axial",
        ),
    ]
    return model, drives


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("assembly")
    result = []
    for curved in (False, True):
        model, drives = case(curved)
        for density in (2, 3):
            path = root / f"{curved}-{density}"
            manifest = export_assembly(model, drives, path, mesh_size_mm=density)
            result.append((model, drives, path, manifest))
    return result


@pytest.mark.parametrize("index", range(4))
def test_closed_shared_shell(index, artifacts):
    model, drives, path, manifest = artifacts[index]
    mesh = meshio.read(path / "preview.msh")
    triangles = mesh.get_cells_type("triangle")
    tags = mesh.get_cell_data("gmsh:physical", "triangle")
    xyz = mesh.points[triangles]
    cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    assert np.all(np.linalg.norm(cross, axis=1) > 1e-12)
    assert np.einsum("ij,ij->i", xyz[:, 0], cross).sum() / 6 > 0
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    _, inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    assert set(counts) == {2}
    assert not np.any(
        np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    )
    assert len(manifest["channels"]) == 2
    assert manifest["channels"] == assembly_channels(model, drives)
    assert all(n > 0 for n in manifest["shared_join_edge_counts"])
    assert len(manifest["all_face_indices"]) == len(model.patches) + len(
        model.rigid_areas()
    )
    for patch in manifest["patches"]:
        p = xyz[tags == patch["mesh_tag"]]
        assert len(p)
        if patch["role"] == "moving":
            assert np.all(cross[tags == patch["mesh_tag"], 2] >= -1e-12)
        assert np.max(model.patch_distance(patch["id"], p)) < 1e-6
    step = (path / "geometry.step").read_text()
    assert step.count("SHELL_BASED_SURFACE_MODEL(") == 1
    assert "MANIFOLD_SOLID_BREP(" not in step
    assert manifest["density"]["triangle_count"] == len(triangles)
    moving, rigid = independent_surfaces(model)
    source_tags = {
        p["id"]: p["mesh_tag"] for p in manifest["patches"] if p["role"] == "moving"
    }
    bounds = [
        certificate_xyz(xyz[tags == source_tags[key]], fn) for key, fn in moving.items()
    ]
    bounds.append(
        certificate_xyz(xyz[~np.isin(tags, list(source_tags.values()))], rigid)
    )
    assert max(bounds) <= 0.15


def test_identity_preview_and_atomic(artifacts, tmp_path):
    for a, b in (artifacts[:2], artifacts[2:]):
        model, drives, _, manifest = a
        assert manifest["geometry_sha256"] == b[3]["geometry_sha256"]
        assert manifest["excitation_sha256"] == b[3]["excitation_sha256"]
        assert manifest["mesh_density_sha256"] != b[3]["mesh_density_sha256"]
        assert [p["id"] for p in manifest["patches"]] == [
            p["id"] for p in b[3]["patches"]
        ]
        assert (
            SourceAssembly.from_dict(json.loads(json.dumps(model.to_dict()))) == model
        )
        for key, c, i, origin, _ in model.patches:
            points = model.preview(radial_steps=8, azimuth_steps=16)[key]
            assert np.max(model.patch_distance(key, points)) < 1e-12
        changed = [replace(drives[0], motion="axial"), drives[1]]
        assert assembly_channels(model, changed) != manifest["channels"]
    model, drives, _, _ = artifacts[0]
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep").write_bytes(b"retained")
    with pytest.raises(FileExistsError):
        export_assembly(model, drives, destination)
    assert (destination / "keep").read_bytes() == b"retained"
    with pytest.raises(ValueError, match="budget"):
        export_assembly(model, drives, tmp_path / "budget", triangle_limit=1)
    assert not (tmp_path / "budget").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"horn_xy_mm": (9, -15)},
        {"horn_xy_mm": (25, 0)},
        {"horn_length_mm": 40},
        {"mouth_radius_mm": 4},
        {"aperture_radius_mm": 7},
        {"front_z_mm": math.nan},
        {"width_mm": True},
        {"woofer_xy_mm": (0, math.inf)},
        {"depth_mm": -1},
    ],
)
def test_invalid_assembly(changes):
    model, _ = case()
    with pytest.raises(ValueError):
        replace(model, **changes)


def test_channel_and_source_identity():
    model, drives = case()
    with pytest.raises(ValueError, match="distinct"):
        replace(model, woofer=replace(model.woofer, physical_source_id="horn"))
    with pytest.raises(ValueError, match="distinct"):
        assembly_channels(model, [drives[0], replace(drives[1], channel_id="hf")])
    assert [p[0] for p in model.patches] == ["horn/piston", "woofer/piston"]


def test_local_wall_refinement_is_in_preflight_budget(tmp_path):
    model, drives = case()
    tiny = replace(
        model, horn=replace(flat(0.001), physical_source_id="horn", rim_id="horn.rim")
    )
    with pytest.raises(ValueError, match="estimated triangle budget"):
        export_assembly(tiny, drives, tmp_path / "tiny", mesh_size_mm=10)
    assert not (tmp_path / "tiny").exists()
