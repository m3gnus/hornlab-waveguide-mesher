import json
import math
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from hornlab_mesher.contour_artifact import export_contour
from hornlab_mesher.source_contour import (
    ContourDrive,
    ContourPoint,
    ContourSegment,
    SourceContour,
    cone,
    dome,
    flat,
)


def certified_facet_distance(xyz, distance, tolerance=0.15, n=32):
    """Independent whole-facet bound: exact distance is 1-Lipschitz.

    A barycentric lattice covers each facet with radius <= diameter/32.
    The largest sampled distance plus that radius bounds every point, rather
    than reporting only vertices or centroids. Distance uses finite meridians.
    """
    uv = np.array([(a / n, b / n) for a in range(n + 1) for b in range(n + 1 - a)])
    worst = 0
    for batch in np.array_split(xyz, max(1, len(xyz) // 64 + 1)):
        samples = (
            batch[:, 0, None, :]
            + uv[None, :, 0, None] * (batch[:, 1, None, :] - batch[:, 0, None, :])
            + uv[None, :, 1, None] * (batch[:, 2, None, :] - batch[:, 0, None, :])
        )
        r = np.linalg.norm(samples[:, :, :2], axis=2)
        z = samples[:, :, 2]
        diameter = np.max(
            np.linalg.norm(batch - np.roll(batch, 1, axis=1), axis=2), axis=1
        )
        bound = distance(r, z).max(axis=1) + diameter / n
        worst = max(worst, float(bound.max()))
    assert worst <= tolerance, (worst, tolerance)
    return worst


def line_distance(a, b):
    ar, az = a
    br, bz = b
    dr, dz = br - ar, bz - az

    def distance(r, z):
        t = np.clip(((r - ar) * dr + (z - az) * dz) / (dr * dr + dz * dz), 0, 1)
        return np.hypot(r - (ar + t * dr), z - (az + t * dz))

    return distance


def arc_distance(a, b, center, direction=None):
    cr, cz = center
    radius = math.hypot(a[0] - cr, a[1] - cz)
    # These accepted monotone arcs occupy one radial branch. The nearest
    # circle point belongs to the finite arc iff its radial coordinate lies
    # between its endpoints and its z is on the authored branch.
    start = math.atan2(a[1] - cz, a[0] - cr)
    end = math.atan2(b[1] - cz, b[0] - cr)
    delta = (end - start) % (2 * math.pi)
    sweep = (
        delta
        if direction == "ccw"
        else delta - 2 * math.pi
        if direction == "cw"
        else delta
        if delta <= math.pi
        else delta - 2 * math.pi
    )

    def distance(r, z):
        angle = np.arctan2(z - cz, r - cr)
        circle = abs(np.hypot(r - cr, z - cz) - radius)
        ends = np.minimum(np.hypot(r - a[0], z - a[1]), np.hypot(r - b[0], z - b[1]))
        progress = np.mod((angle - start) * math.copysign(1, sweep), 2 * math.pi)
        progress = np.where(abs(progress - 2 * math.pi) < 1e-10, 0, progress)
        return np.where(progress <= abs(sweep) + 1e-10, circle, ends)

    return distance


def cases():
    return [
        flat(24),
        dome(18, 6, surround_width_mm=4, surround_depth_mm=-1, land_width_mm=2),
        cone(
            18,
            8,
            cap_radius_mm=6,
            cap_height_mm=3,
            surround_width_mm=4,
            surround_depth_mm=1,
            land_width_mm=2,
        ),
    ]


@pytest.mark.parametrize("model", cases())
def test_canonical_roundtrip_and_excitation_separation(model):
    assert SourceContour.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    weights = tuple(
        (s.id, [1, 0.4, 0, -0.5][i])
        for i, s in enumerate(model.segments)
        if s.role == "moving"
    )
    normal = ContourDrive("motor", weights)
    normal.validate(model)
    assert replace(normal, motion="axial").excitation_sha256 != normal.excitation_sha256
    assert (
        model.geometry_sha256
        == SourceContour.from_dict(model.to_dict()).geometry_sha256
    )
    for i, s in enumerate(model.segments):
        r, z = np.asarray([model.evaluate(i, t) for t in np.linspace(0, 1, 129)]).T
        assert np.all(np.diff(r) > 0)
        if s.kind == "arc":
            cr, cz = s.center_mm
            R = math.hypot(model.points[i].r_mm - cr, model.points[i].z_mm - cz)
            assert np.max(abs((r - cr) ** 2 + (z - cz) ** 2 - R**2)) < 1e-10


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_invalid_points(bad):
    with pytest.raises(ValueError):
        ContourPoint("p", bad, 0)


def test_bad_contracts():
    with pytest.raises(ValueError):
        ContourDrive("motor", (("a", float("nan")),))
    with pytest.raises(ValueError):
        ContourDrive("motor", (("a", 1), ("a", 2)))
    with pytest.raises(ValueError):
        ContourDrive("motor", (("land", 0),)).validate(dome(18, 6, land_width_mm=2))
    with pytest.raises(ValueError):
        SourceContour(
            "motor",
            "rim",
            (ContourPoint("a", 0, 0), ContourPoint("b", 1, 0)),
            (ContourSegment("a", "a", "b", kind="arc", center_mm=(0.5, -0.5)),),
        )
    with pytest.raises(ValueError):
        dome(18, 19)
    with pytest.raises(ValueError):
        dome(18, 6, surround_width_mm=4, surround_depth_mm=3)


@pytest.mark.parametrize("model", cases(), ids=["flat", "dome", "cone"])
@pytest.mark.parametrize("size", [2, 1])
def test_actual_mesh_and_reopened_step(model, size, tmp_path):
    import gmsh
    import meshio

    from hornlab_mesher.step_mapping import advanced_face_order_for_surfaces

    moving = [s for s in model.segments if s.role == "moving"]
    weights = tuple((s.id, [1, 0.4, 0][i]) for i, s in enumerate(moving))
    artifact = tmp_path / "artifact"
    manifest = export_contour(
        model,
        ContourDrive("motor", weights),
        artifact,
        mouth_radius_mm=48,
        length_mm=70,
        mesh_size_mm=size,
    )
    mesh = meshio.read(artifact / "preview.msh")
    pts = mesh.points
    tri = mesh.get_cells_type("triangle")
    tags = mesh.get_cell_data("gmsh:physical", "triangle")
    assert len(tri) <= 250000
    assert len({tuple(sorted(t)) for t in tri}) == len(tri)
    xyz = pts[tri]
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    assert np.all(np.linalg.norm(normals, axis=1) > 1e-12)
    edges = {}
    for t in tri:
        for a, b in zip(t, np.roll(t, -1)):
            key = tuple(sorted((a, b)))
            edges[key] = edges.get(key, 0) + 1
    assert max(edges.values()) == 2
    boundary = [key for key, n in edges.items() if n == 1]
    assert not boundary
    # A closed consistently oriented surface must traverse shared edges in
    # opposite directions, not just have two coincident triangles there.
    directed = {}
    for t in tri:
        for a, b in zip(t, np.roll(t, -1)):
            key = tuple(sorted((a, b)))
            directed[key] = directed.get(key, 0) + (1 if a < b else -1)
    assert set(directed.values()) == {0}
    for i, s in enumerate(model.segments):
        mask = tags == manifest["patches"][i]["mesh_tag"]
        assert mask.any()
        assert np.min(normals[mask, 2]) >= -1e-9
        p = pts[np.unique(tri[mask])]
        r = np.linalg.norm(p[:, :2], axis=1)
        z = p[:, 2]
        a, b = model.points[i : i + 2]
        endpoints = ((a.r_mm, a.z_mm), (b.r_mm, b.z_mm))
        oracle = (
            line_distance(*endpoints)
            if s.kind == "line"
            else arc_distance(*endpoints, s.center_mm, s.direction)
        )
        certified_facet_distance(xyz[mask], oracle)
        if s.kind == "line":
            expected = a.z_mm + (r - a.r_mm) * (b.z_mm - a.z_mm) / (b.r_mm - a.r_mm)
            assert np.max(abs(z - expected)) < 1e-6
        else:
            cr, cz = s.center_mm
            R = math.hypot(a.r_mm - cr, a.z_mm - cz)
            assert np.max(abs(np.hypot(r - cr, z - cz) - R)) < 1e-6
    rigid_oracles = [
        line_distance((24, 0), (48, 70)),
        line_distance((48, 70), (49, 70)),
        line_distance((49, 70), (49, -15)),
        line_distance((49, -15), (0, -15)),
    ]
    certified_facet_distance(
        xyz[tags == 1 + len(model.segments)],
        lambda r, z: np.minimum.reduce([f(r, z) for f in rigid_oracles]),
    )
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(str(artifact / "geometry.step"))
        gmsh.model.occ.synchronize()
        faces = [tag for _, tag in gmsh.model.getEntities(2)]
        ids = advanced_face_order_for_surfaces(artifact / "geometry.step", faces)
        assert sorted(ids) == manifest["all_face_indices"]
        assert not gmsh.model.getEntities(3)
        mapping = dict(zip(ids, faces))
        for patch in manifest["patches"]:
            assert gmsh.model.occ.getMass(
                2, mapping[patch["advanced_face_indices"][0]]
            ) == pytest.approx(patch["area_mm2"], rel=1e-9)
        rings = [
            {
                abs(tag)
                for dim, tag in gmsh.model.getBoundary(
                    [(2, mapping[p["advanced_face_indices"][0]])], oriented=False
                )
                if dim == 1
            }
            for p in manifest["patches"]
        ]
        assert all(a & b for a, b in pairwise(rings))
    finally:
        gmsh.finalize()


def test_atomic_refusal(tmp_path):
    output = tmp_path / "keep"
    output.mkdir()
    (output / "sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        export_contour(
            flat(24),
            ContourDrive("motor", (("piston", 1),)),
            output,
            mouth_radius_mm=48,
            length_mm=70,
        )
    assert (output / "sentinel").read_text() == "keep"
    with pytest.raises(ValueError, match="budget"):
        export_contour(
            flat(24),
            ContourDrive("motor", (("piston", 1),)),
            tmp_path / "dense",
            mouth_radius_mm=48,
            length_mm=70,
            mesh_size_mm=0.001,
        )
    assert not (tmp_path / "dense").exists()


def test_equivalent_canonical_geometry_and_preview():
    a = flat(24)
    b = replace(flat(24.0), segments=(replace(a.segments[0], direction="cw"),))
    assert a.to_dict() == b.to_dict()
    assert a.geometry_sha256 == b.geometry_sha256
    for model in cases():
        for i, (s, values) in enumerate(
            zip(model.segments, model.preview(8, 16).values())
        ):
            xyz = np.asarray(values)
            p, q = model.points[i : i + 2]
            ends = ((p.r_mm, p.z_mm), (q.r_mm, q.z_mm))
            distance = (
                line_distance(*ends)
                if s.kind == "line"
                else arc_distance(*ends, s.center_mm, s.direction)
            )
            assert distance(np.linalg.norm(xyz[:, :2], axis=1), xyz[:, 2]).max() < 1e-12


@pytest.mark.parametrize(
    "kind",
    [
        "equal-area",
        "reserved-id",
        "many-patches",
        "semicircle-positive",
        "semicircle-negative",
    ],
)
def test_supported_custom_mapping_regressions(kind, tmp_path):
    import meshio

    if kind == "equal-area":
        points = (
            ContourPoint("a", 0, 0),
            ContourPoint("b", 1, 0),
            ContourPoint("c", math.sqrt(2), 0),
        )
        segments = (
            ContourSegment("inner", "a", "b"),
            ContourSegment("outer", "b", "c"),
        )
        model = SourceContour("diaphragm", "horn.throat", points, segments)
    elif kind == "reserved-id":
        model = flat(24)
        model = replace(model, segments=(replace(model.segments[0], id="__rigid0__"),))
    elif kind == "many-patches":
        points = tuple(ContourPoint(f"p{i}", i, 0) for i in range(103))
        segments = tuple(
            ContourSegment(
                f"s{i}", f"p{i}", f"p{i + 1}", role="moving" if i == 0 else "rigid"
            )
            for i in range(102)
        )
        model = SourceContour("diaphragm", "horn.throat", points, segments)
    else:
        model = dome(
            17,
            7,
            surround_width_mm=2,
            surround_depth_mm=1 if kind.endswith("positive") else -1,
        )
    drive = ContourDrive(
        "motor", tuple((s.id, 0.4) for s in model.segments if s.role == "moving")
    )
    path = tmp_path / "artifact"
    manifest = export_contour(
        model,
        drive,
        path,
        mouth_radius_mm=model.points[-1].r_mm + 10,
        length_mm=30,
        mesh_size_mm=10 if kind == "many-patches" else 2,
    )
    tags = meshio.read(path / "preview.msh").get_cell_data("gmsh:physical", "triangle")
    assert set(tags) == {
        1 + len(model.segments),
        *(p["mesh_tag"] for p in manifest["patches"]),
    }
    assert len(manifest["all_face_indices"]) == len(model.segments) + 4
    if kind.startswith("semicircle"):
        mesh = meshio.read(path / "preview.msh")
        xyz = mesh.points[mesh.get_cells_type("triangle")]
        i = 1
        a, b = model.points[i : i + 2]
        certified_facet_distance(
            xyz[tags == manifest["patches"][i]["mesh_tag"]],
            arc_distance(
                (a.r_mm, a.z_mm),
                (b.r_mm, b.z_mm),
                model.segments[i].center_mm,
                model.segments[i].direction,
            ),
        )
