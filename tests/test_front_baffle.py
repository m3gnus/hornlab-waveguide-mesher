from dataclasses import replace

import meshio
import numpy as np
import pytest
from test_source_contour import arc_distance, line_distance

from hornlab_mesher.front_baffle import FrontBaffle
from hornlab_mesher.source_contour import ContourDrive, cone, flat
from hornlab_mesher.woofer_artifact import export_woofer


def models():
    return [
        (flat(8), FrontBaffle(30, 40, 12, 8), [1]),
        (
            cone(
                10,
                4,
                cap_radius_mm=3,
                cap_height_mm=2,
                surround_width_mm=2,
                surround_depth_mm=0.6,
                land_width_mm=1,
            ),
            FrontBaffle(44, 54, 12, 14, (3, -4, 7)),
            [1, -0.5, 0],
        ),
        (
            cone(
                10,
                4,
                cap_radius_mm=3,
                cap_height_mm=2,
                surround_width_mm=2,
                surround_depth_mm=-0.6,
                land_width_mm=1,
            ),
            FrontBaffle(44, 54, 12, 13, (-5, 6, -3)),
            [1, 0.4, 0],
        ),
    ]


def meridian(model, index, origin):
    a, b = model.points[index : index + 2]
    s = model.segments[index]
    fn = (
        line_distance((a.r_mm, a.z_mm), (b.r_mm, b.z_mm))
        if s.kind == "line"
        else arc_distance((a.r_mm, a.z_mm), (b.r_mm, b.z_mm), s.center_mm, s.direction)
    )

    def distance(xyz):
        p = xyz - origin
        return fn(np.hypot(p[..., 0], p[..., 1]), p[..., 2])

    return distance


def rigid_distance(model, baffle):
    """Independent finite box planes + circular hole; no production helpers."""
    cx, cy, z = baffle.center_mm
    w, h, d = baffle.width_mm / 2, baffle.height_mm / 2, baffle.depth_mm
    rim = model.points[-1].r_mm
    lands = [
        meridian(model, i, np.asarray(baffle.center_mm))
        for i, s in enumerate(model.segments)
        if s.role == "rigid"
    ]

    def rect(xyz, axis, value, a, b):
        other = [i for i in range(3) if i != axis]
        da = np.maximum(
            np.maximum(a[0] - xyz[..., other[0]], xyz[..., other[0]] - a[1]), 0
        )
        db = np.maximum(
            np.maximum(b[0] - xyz[..., other[1]], xyz[..., other[1]] - b[1]), 0
        )
        return np.sqrt((xyz[..., axis] - value) ** 2 + da**2 + db**2)

    def distance(xyz):
        front = rect(xyz, 2, z, (-w, w), (-h, h))
        radius = np.hypot(xyz[..., 0] - cx, xyz[..., 1] - cy)
        front = np.where(radius < rim, np.hypot(z - xyz[..., 2], rim - radius), front)
        values = [
            front,
            rect(xyz, 2, z - d, (-w, w), (-h, h)),
            rect(xyz, 0, -w, (-h, h), (z - d, z)),
            rect(xyz, 0, w, (-h, h), (z - d, z)),
            rect(xyz, 1, -h, (-w, w), (z - d, z)),
            rect(xyz, 1, h, (-w, w), (z - d, z)),
        ]
        return np.minimum.reduce(values + [f(xyz) for f in lands])

    return distance


def certificate_xyz(facets, distance):
    worst = 0
    for n in (32, 128):
        uv = np.array([(a / n, b / n) for a in range(n + 1) for b in range(n + 1 - a)])
        pending = []
        for batch in np.array_split(facets, max(1, len(facets) // 24 + 1)):
            samples = (
                batch[:, 0, None, :]
                + uv[None, :, 0, None] * (batch[:, 1, None, :] - batch[:, 0, None, :])
                + uv[None, :, 1, None] * (batch[:, 2, None, :] - batch[:, 0, None, :])
            )
            bounds = (
                distance(samples).max(axis=1)
                + np.linalg.norm(batch - np.roll(batch, 1, axis=1), axis=2).max(axis=1)
                / n
            )
            good = bounds <= 0.15
            if good.any():
                worst = max(worst, float(bounds[good].max()))
            pending.extend(batch[~good])
        if not pending:
            return worst
        facets = np.asarray(pending)
    pytest.fail(f"facet certificate exceeds .15 mm: {bounds.max()}")


def topology(mesh):
    tri = mesh.get_cells_type("triangle")
    xyz = mesh.points[tri]
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    _, inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    assert set(counts) == {2}
    assert not np.any(
        np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    )
    assert len(np.unique(np.sort(tri, axis=1), axis=0)) == len(tri)
    assert np.all(
        np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        > 1e-12
    )


@pytest.mark.parametrize("index", range(3))
def test_native_woofer_geometry_and_density(index, tmp_path):
    import gmsh

    from hornlab_mesher.step_mapping import advanced_face_order_for_surfaces

    model, baffle, weights = models()[index]
    drive = ContourDrive(
        "motor",
        tuple(zip([s.id for s in model.segments if s.role == "moving"], weights)),
    )
    manifests = []
    for size in (2, 1):
        path = tmp_path / str(size)
        manifest = export_woofer(model, drive, baffle, path, mesh_size_mm=size)
        manifests.append(manifest)
        mesh = meshio.read(path / "preview.msh")
        topology(mesh)
        tags = mesh.get_cell_data("gmsh:physical", "triangle")
        xyz = mesh.points[mesh.get_cells_type("triangle")]
        moving_tags = []
        for i, s in enumerate(model.segments):
            if s.role == "moving":
                tag = manifest["patches"][i]["mesh_tag"]
                moving_tags.append(tag)
                certificate_xyz(
                    xyz[tags == tag], meridian(model, i, np.asarray(baffle.center_mm))
                )
                assert np.all(
                    np.cross(
                        xyz[tags == tag, 1] - xyz[tags == tag, 0],
                        xyz[tags == tag, 2] - xyz[tags == tag, 0],
                    )[:, 2]
                    >= -1e-9
                )
        certificate_xyz(xyz[~np.isin(tags, moving_tags)], rigid_distance(model, baffle))
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
            gmsh.model.occ.importShapes(str(path / "geometry.step"))
            gmsh.model.occ.synchronize()
            assert not gmsh.model.getEntities(3)
            faces = [t for _, t in gmsh.model.getEntities(2)]
            ids = advanced_face_order_for_surfaces(path / "geometry.step", faces)
            mapped = dict(zip(ids, faces))
            assert sorted(ids) == manifest["all_face_indices"]
            assert len(faces) == len(model.segments) + 6 + (index == 1)
            for p in manifest["patches"] + manifest["baffle_faces"]:
                assert gmsh.model.occ.getMass(
                    2, mapped[p["advanced_face_indices"][0]]
                ) == pytest.approx(p["area_mm2"], rel=1e-7)
            counts = {}
            for face in faces:
                for dim, edge in gmsh.model.getBoundary(
                    [(2, face)], combined=False, oriented=False
                ):
                    if dim == 1 and gmsh.model.occ.getMass(1, edge) > 1e-8:
                        counts[edge] = counts.get(edge, 0) + 1
            # Periodic revolution seams are used twice by one face.
            assert set(counts.values()) == {2}
        finally:
            gmsh.finalize()
        assert (path / "geometry.step").read_text().count("CLOSED_SHELL(") == 1
    assert manifests[0]["geometry_sha256"] == manifests[1]["geometry_sha256"]
    assert manifests[0]["mesh_density_sha256"] != manifests[1]["mesh_density_sha256"]
    assert [(p["id"], p["advanced_face_indices"]) for p in manifests[0]["patches"]] == [
        (p["id"], p["advanced_face_indices"]) for p in manifests[1]["patches"]
    ]


def test_independent_dimensions_preview_and_invalid_attachments(tmp_path):
    model, baffle, _ = models()[1]
    assert FrontBaffle.from_dict(baffle.to_dict()) == baffle
    preview = baffle.preview(model, radial_steps=7, azimuth_steps=9)
    for key, xyz in model.preview(radial_steps=7, azimuth_steps=9).items():
        assert np.allclose(preview[key] - xyz, baffle.center_mm)
    for changed in [
        replace(baffle, aperture_radius_mm=12),
        replace(baffle, center_mm=(9, 0, 7)),
        replace(baffle, depth_mm=4),
    ]:
        with pytest.raises(ValueError):
            changed.validate(model)
    for kwargs in [
        {"width_mm": float("inf")},
        {"center_mm": (0, float("nan"), 0)},
        {"aperture_radius_mm": True},
    ]:
        with pytest.raises(ValueError):
            replace(baffle, **kwargs)
    baffle.validate(
        cone(
            10,
            3,
            cap_radius_mm=4,
            cap_height_mm=1,
            surround_width_mm=2,
            surround_depth_mm=-0.8,
            land_width_mm=1,
        )
    )
    for patch in preview:
        assert patch in {s.id for s in model.segments}
    drive = ContourDrive(
        "motor", tuple((s.id, 1) for s in model.segments if s.role == "moving")
    )
    with pytest.raises(ValueError, match="budget"):
        export_woofer(model, drive, baffle, tmp_path / "over-budget", mesh_size_mm=0.01)
    assert not (tmp_path / "over-budget").exists()


def test_export_preserves_caller_and_publication(tmp_path):
    import gmsh

    model, baffle, weights = models()[0]
    drive = ContourDrive("motor", (("piston", weights[0]),))
    gmsh.initialize()
    try:
        gmsh.model.add("caller")
        before = gmsh.model.list()
        export_woofer(model, drive, baffle, tmp_path / "artifact")
        assert gmsh.model.list() == before
        original = (tmp_path / "artifact/source.json").read_bytes()
        with pytest.raises(FileExistsError):
            export_woofer(model, drive, baffle, tmp_path / "artifact")
        assert (tmp_path / "artifact/source.json").read_bytes() == original
    finally:
        gmsh.finalize()


def test_atomic_worker_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from hornlab_mesher import woofer_artifact

    model, baffle, _ = models()[0]
    monkeypatch.setattr(
        woofer_artifact.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stderr="injected worker failure"),
    )
    with pytest.raises(ValueError, match="injected"):
        export_woofer(
            model, ContourDrive("motor", (("piston", 1),)), baffle, tmp_path / "failed"
        )
    assert not list(tmp_path.iterdir())


def test_identity_dimensions_and_saved_points():
    from hornlab_mesher.source_contour import SourceContour, digest

    model, baffle, _ = models()[1]
    assert SourceContour.from_dict(model.to_dict()) == model
    baseline = digest({"contour": model.to_dict(), "baffle": baffle.to_dict()})
    for other in [
        replace(baffle, aperture_radius_mm=15),
        replace(baffle, center_mm=(-2, 3, 7)),
        replace(baffle, depth_mm=14),
    ]:
        other.validate(model)
        assert (
            digest({"contour": model.to_dict(), "baffle": other.to_dict()}) != baseline
        )
        assert model.to_dict() == SourceContour.from_dict(model.to_dict()).to_dict()
    for changes in [
        {"depth_mm": 3},
        {"cap_radius_mm": 4},
        {"cap_height_mm": 1},
        {"surround_width_mm": 3},
        {"surround_depth_mm": -0.6},
    ]:
        params = {
            "radius_mm": 10,
            "depth_mm": 4,
            "cap_radius_mm": 3,
            "cap_height_mm": 2,
            "surround_width_mm": 2,
            "surround_depth_mm": 0.6,
            "land_width_mm": 1,
        }
        params.update(changes)
        changed = cone(**params)
        baffle.validate(changed)
        assert (
            digest({"contour": changed.to_dict(), "baffle": baffle.to_dict()})
            != baseline
        )
    with pytest.raises(ValueError, match="collar"):
        replace(baffle, aperture_radius_mm=13.01).validate(model)
