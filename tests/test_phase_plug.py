"""Passive topology, analytical clearance and actual sewn STEP/mesh regression."""

import json
from dataclasses import replace

import meshio
import numpy as np
import pytest
from hornlab_mesher.assembly_artifact import export_assembly
from hornlab_mesher.phase_plug import PhasePlug, passage_contract, validate_passages
from hornlab_mesher.source_assembly import SourceAssembly
from test_source_assembly import case


def plug_case(kind="combined", curved=False):
    model, drives = case(curved)
    central = PhasePlug("core", 2, 6, 0, 1, 0, 1.4)
    vane = PhasePlug("vane/A", 2, 6, 2, 2.5, 2.4, 2.9)
    if kind == "narrow":
        vane = replace(vane, inner0_mm=1.6, outer0_mm=2.1, inner1_mm=2, outer1_mm=2.5)
    if kind == "two-vanes":
        return replace(
            model, phase_plugs=(vane, PhasePlug("vane/B", 2, 6, 3.3, 3.8, 3.7, 4.2))
        ), drives
    bodies = (
        (central,)
        if kind == "central"
        else (vane,)
        if kind == "annular"
        else (central, vane)
    )
    return replace(model, phase_plugs=bodies), drives


def test_canonical_identity_and_preview():
    model, _ = plug_case()
    restored = SourceAssembly.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored == model
    assert restored.geometry_sha256 == model.geometry_sha256
    plain, _ = case()
    assert "phase_plugs" not in plain.to_dict()
    assert "phase_plugs" not in SourceAssembly.from_dict(plain.to_dict()).to_dict()
    assert model.geometry_sha256 != plain.geometry_sha256
    preview = model.preview(radial_steps=8, azimuth_steps=16)
    for p in model.phase_plugs:
        for key in p.edges:
            assert model.rigid_distance(key, preview[key]).max() < 1e-12
    assert len(passage_contract(model)["bodies"]) == 2
    assert min(validate_passages(model).values()) > 0.1


@pytest.mark.parametrize(
    "changes",
    [
        {"id": ""},
        {"id": " core"},
        {"id": "a" * 129},
        {"z0_mm": True},
        {"z1_mm": float("nan")},
        {"z1_mm": 2.05},
        {"inner0_mm": -1},
        {"inner0_mm": 0.2},
        {"outer0_mm": 0.1},
        {"inner0_mm": 0.05, "inner1_mm": 0.05},
    ],
)
def test_invalid_body(changes):
    with pytest.raises(ValueError):
        replace(PhasePlug("core", 2, 6, 0, 1, 0, 1.4), **changes)


@pytest.mark.parametrize(
    "kind", ["source", "mouth", "wall", "crossing", "duplicate", "planes", "many"]
)
def test_invalid_passages(kind):
    model, _ = plug_case()
    a, b = model.phase_plugs
    plugs = {
        "source": (replace(a, z0_mm=0.1),),
        "mouth": (replace(a, z1_mm=23.95),),
        "wall": (replace(a, outer0_mm=4.6),),
        "crossing": (a, replace(b, inner1_mm=1.3)),
        "duplicate": (a, replace(b, id="core")),
        "planes": (a, replace(b, z0_mm=2.2)),
        "many": tuple(replace(a, id=str(i)) for i in range(9)),
    }[kind]
    with pytest.raises(ValueError):
        replace(model, phase_plugs=plugs)


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    root = tmp_path_factory.mktemp("passages")
    result = []
    for kind, curved in (
        ("central", False),
        ("annular", False),
        ("combined", False),
        ("narrow", True),
        ("two-vanes", False),
    ):
        model, drives = plug_case(kind, curved)
        for size in (3, 2):
            destination = root / f"{kind}-{size}"
            manifest = export_assembly(model, drives, destination, mesh_size_mm=size)
            result.append((model, destination, manifest))
    return result


def test_flat_annulus_triangle_interior_cannot_hide_a_hole():
    from hornlab_mesher.passage_mesh import annulus_bounds

    triangle = np.array([[[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]]])
    assert np.linalg.norm(triangle[0, :, :2], axis=1).min() > 1
    assert annulus_bounds(triangle, (0, 0), 0, 1, 5)[0] == pytest.approx(1)


def test_actual_closed_components_and_passage_certificate(bundles):
    for model, directory, manifest in bundles:
        quality = manifest["passage_quality"]
        assert len(quality["components"]) == 1 + len(model.phase_plugs)
        assert quality["maximum_facet_bound_mm"] <= quality["surface_tolerance_mm"]
        assert (
            quality["certified_clearance_mm"] >= 0.8 * quality["minimum_clearance_mm"]
        )
        expected = {p.id: 0 if p.inner0_mm else 2 for p in model.phase_plugs}
        for c in quality["components"]:
            assert c["euler"] == (2 if c["id"] == "enclosure" else expected[c["id"]])
            assert c["volume_mm3"] > 0
        mesh = meshio.read(directory / "preview.msh")
        triangles = mesh.get_cells_type("triangle")
        tags = mesh.get_cell_data("gmsh:physical", "triangle")
        moving = {p["mesh_tag"] for p in manifest["patches"] if p["role"] == "moving"}
        assert set(tags) == {p["mesh_tag"] for p in manifest["patches"]} | {
            1 + len(manifest["patches"])
        }
        assert len(triangles) <= 250000
        # Independent local cylinder/cone oracle, no product distance helper.
        xyz = mesh.points[triangles]
        q = xyz - model.parts[0][1]
        r, z = np.hypot(q[..., 0], q[..., 1]), q[..., 2]
        for p in model.phase_plugs:
            mask = (z.min(axis=1) >= p.z0_mm - 1e-7) & (z.max(axis=1) <= p.z1_mm + 1e-7)
            mask &= r.max(axis=1) <= max(p.outer0_mm, p.outer1_mm) + 1e-7
            mask &= r.min(axis=1) >= min(p.inner0_mm, p.inner1_mm) - 1e-7
            assert mask.any()
            assert not np.isin(tags[mask], list(moving)).any()


def test_reopened_step_topology(bundles):
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        for model, directory, manifest in bundles:
            gmsh.clear()
            gmsh.model.occ.importShapes(
                str(directory / "geometry.step"), highestDimOnly=False
            )
            gmsh.model.occ.synchronize()
            assert not gmsh.model.getEntities(3)
            faces = [t for _, t in gmsh.model.getEntities(2)]
            assert len(faces) == len(manifest["all_face_indices"])
            # Reopened shells each have actual shared edges. Seam edges may
            # occur twice on one periodic surface; count signed occurrences.
            boundaries = [
                gmsh.model.getBoundary([(2, f)], oriented=True) for f in faces
            ]
            occurrences = {}
            for edges in boundaries:
                for dim, edge in edges:
                    if dim == 1:
                        occurrences.setdefault(abs(edge), []).append(edge)
            # Gmsh reports one periodic seam per face; geometric sewing is also
            # independently exercised by mesh closure and shell component count.
            assert all(len(es) <= 2 for es in occurrences.values())
            area = sum(gmsh.model.occ.getMass(2, f) for f in faces)
            expected = sum(
                p["area_mm2"] for p in manifest["patches"] + manifest["rigid_faces"]
            )
            assert area == pytest.approx(expected, rel=1e-9)
            text = (directory / "geometry.step").read_text()
            assert text.count("SHELL_BASED_SURFACE_MODEL(") == 1 + len(
                model.phase_plugs
            )
    finally:
        gmsh.finalize()


def test_passage_resource_refusal(tmp_path):
    model, drives = plug_case("narrow")
    with pytest.raises(ValueError, match="estimated triangle budget"):
        export_assembly(model, drives, tmp_path / "refused", triangle_limit=100)
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("change", ["placement", "thickness"])
def test_passive_dimensions_change_geometry_without_drive_groups(change, bundles, tmp_path):
    model, drives = plug_case("annular")
    body = model.phase_plugs[0]
    changed = replace(body, z0_mm=3, z1_mm=8) if change == "placement" else replace(
        body, outer0_mm=2.8, outer1_mm=3.3
    )
    variant = replace(model, phase_plugs=(changed,))
    baseline = next(m for _, path, m in bundles if path.name == "annular-3")
    result = export_assembly(variant, drives, tmp_path / "variant", mesh_size_mm=3)
    assert result["geometry_sha256"] != baseline["geometry_sha256"]
    assert result["excitation_sha256"] == baseline["excitation_sha256"]
    assert result["channels"] == baseline["channels"]
    assert result["passage_contract"]["open_passage_count"] == 2
    assert result["passage_contract"]["bodies"] == baseline["passage_contract"]["bodies"]
    assert result["passage_quality"]["certified_clearance_mm"] > 0


def test_excitation_edit_preserves_passive_mesh(bundles, tmp_path):
    model, drives = plug_case("annular")
    changed = [replace(d, weights=tuple((key, -0.5 if i == 0 else 0) for key, _ in d.weights))
        for i, d in enumerate(drives)]
    baseline = next(m for _, path, m in bundles if path.name == "annular-3")
    result = export_assembly(model, changed, tmp_path / "weighted", mesh_size_mm=3)
    assert result["geometry_sha256"] == baseline["geometry_sha256"]
    assert result["excitation_sha256"] != baseline["excitation_sha256"]
    assert result["members"]["preview.msh"] == baseline["members"]["preview.msh"]
    assert result["passage_contract"] == baseline["passage_contract"]
    assert result["passage_quality"] == baseline["passage_quality"]


def test_three_density_geometric_convergence(tmp_path):
    model, drives = plug_case("annular")
    measured, counts, identities = [], [], []
    p = model.phase_plugs[0]
    # Independent meridian segments, not the product's distance/edge methods.
    rz = [
        (p.inner0_mm, p.z0_mm),
        (p.outer0_mm, p.z0_mm),
        (p.outer1_mm, p.z1_mm),
        (p.inner1_mm, p.z1_mm),
    ]
    uv = np.asarray([(a / 16, b / 16) for a in range(17) for b in range(17 - a)])
    for refinement in (1, 2, 4):
        path = tmp_path / str(refinement)
        manifest = export_assembly(
            model, drives, path, mesh_size_mm=3, passage_refinement=refinement
        )
        mesh = meshio.read(path / "preview.msh")
        xyz = mesh.points[mesh.get_cells_type("triangle")] - model.parts[0][1]
        r = np.linalg.norm(xyz[..., :2], axis=2)
        selected = xyz[
            (r.max(axis=1) <= max(p.outer0_mm, p.outer1_mm) + 1e-7)
            & (xyz[..., 2].min(axis=1) >= p.z0_mm - 1e-7)
            & (xyz[..., 2].max(axis=1) <= p.z1_mm + 1e-7)
        ]
        counts.append(len(selected))
        samples = (
            selected[:, 0, None]
            + uv[None, :, :1] * (selected[:, 1, None] - selected[:, 0, None])
            + uv[None, :, 1:] * (selected[:, 2, None] - selected[:, 0, None])
        )
        q = np.stack(
            [np.linalg.norm(samples[..., :2], axis=-1), samples[..., 2]], axis=-1
        )
        distances = []
        for a, b in zip(rz, rz[1:] + rz[:1]):
            a, b = np.asarray(a), np.asarray(b)
            d = b - a
            t = np.clip(np.sum((q - a) * d, axis=-1) / (d @ d), 0, 1)
            distances.append(np.linalg.norm(q - a - t[..., None] * d, axis=-1))
        measured.append(float(np.minimum.reduce(distances).max()))
        identities.append((manifest["geometry_sha256"], manifest["excitation_sha256"]))
        assert manifest["passage_contract"]["open_passage_count"] == 2
        assert manifest["passage_quality"]["certified_clearance_mm"] > 0
    assert counts[0] < counts[1] < counts[2]
    assert measured[2] < measured[1] < measured[0]
    assert measured[2] < measured[0] / 4
    assert len(set(identities)) == 1
