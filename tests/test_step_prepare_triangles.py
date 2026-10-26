"""The shared OCC cutter's opt-in triangle-surface specialization."""

import gmsh
import numpy as np
import pytest

import hornlab_mesher.step_prepare as sp
from test_step_prepare import _gmsh_session


def _triangle_box(n=3):
    occ = gmsh.model.occ
    points = {}
    edges = {}
    faces = []

    def vertex(coordinates):
        key = tuple(coordinates)
        if key not in points:
            points[key] = occ.addPoint(*key)
        return points[key]

    def face(vertices):
        lines = []
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            key = tuple(sorted((a, b)))
            if key not in edges:
                edges[key] = occ.addLine(*key)
            lines.append(edges[key] if a < b else -edges[key])
        return occ.addPlaneSurface([occ.addCurveLoop(lines)])

    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        for side in (-1, 1):
            grid = {}
            for i in range(n + 1):
                for j in range(n + 1):
                    coordinates = np.zeros(3)
                    coordinates[axis] = side
                    coordinates[other] = [-1 + 2 * i / n, -1 + 2 * j / n]
                    grid[i, j] = vertex(coordinates)
            for i in range(n):
                for j in range(n):
                    a, b, c, d = [grid[k] for k in
                                  ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))]
                    faces.extend((face([a, b, c]), face([a, c, d])))
    occ.synchronize()
    return faces


def _box_cut(*, triangle_surface_mesh):
    with _gmsh_session():
        faces = _triangle_box()
        group = sp.OccSurfaceGroup("wall", sp.OccSurfaceSelector(faces), sp.OccSurfaceRole("wall"))
        result = sp.auto_cut_occ_geometry(
            [group], planes=("x0", "y0"), grid=3,
            triangle_surface_mesh=triangle_surface_mesh,
        )
        assert result.planes == ("x0", "y0")
        assert {tag for _, tag in gmsh.model.getEntities(2)} == set(result.group("wall").selector.surface_tags)
        area = sum(gmsh.model.occ.getMass(2, tag) for tag in result.group("wall").selector.surface_tags)
        parent_area = [sum(gmsh.model.occ.getMass(2, child)
                           for child in result.parent_to_children[parent]) for parent in faces]
        for _, curve in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(curve, 2)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 10)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 10)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.model.mesh.generate(2)
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        nodes = {int(tag): p for tag, p in zip(node_tags, np.asarray(coordinates).reshape(-1, 3))}
        triangles = []
        for tag in result.group("wall").selector.surface_tags:
            _, connectivity = gmsh.model.mesh.getElementsByType(2, tag)
            for row in np.asarray(connectivity).reshape(-1, 3):
                triangles.append(tuple(sorted(tuple(np.round(nodes[int(node)], 10)) for node in row)))
        return area, parent_area, sorted(triangles), result.report["cut"]


def test_triangle_cut_matches_whole_model_geometry_and_retained_parents():
    general = _box_cut(triangle_surface_mesh=False)
    triangles = _box_cut(triangle_surface_mesh=True)
    assert triangles[0] == pytest.approx(general[0], abs=1e-10)
    np.testing.assert_allclose(triangles[1], general[1], atol=1e-10, rtol=0)
    assert triangles[2] == general[2]
    assert triangles[3]["surfaces_dropped"] == general[3]["surfaces_dropped"]


def test_triangle_mode_refuses_nontriangular_cad_faces():
    with _gmsh_session():
        face = gmsh.model.occ.addRectangle(-1, -1, 0, 2, 2)
        gmsh.model.occ.synchronize()
        with pytest.raises(ValueError, match="nontriangle"):
            sp._triangle_model([face])


def test_triangle_mode_refuses_intersecting_facets():
    with _gmsh_session():
        occ = gmsh.model.occ
        faces = []
        for coordinates in [((-1, 0, 0), (1, 0, 0), (0, 1, 0)),
                            ((0, -0.5, -1), (0, -0.5, 1), (0, 1, 0))]:
            vertices = [occ.addPoint(*point) for point in coordinates]
            edges = [occ.addLine(vertices[i], vertices[(i + 1) % 3]) for i in range(3)]
            faces.append(occ.addPlaneSurface([occ.addCurveLoop(edges)]))
        occ.synchronize()
        with pytest.raises(ValueError, match="intersecting"):
            sp._triangle_model(faces)


def test_triangle_copy_failure_removes_partially_created_copies(monkeypatch):
    with _gmsh_session():
        faces = _triangle_box(n=1)
        model = sp._triangle_model(faces)
        before = set(gmsh.model.getEntities(2))
        original = gmsh.model.occ.copy

        def fail_after_copy(dimtags):
            original(dimtags[:1])
            raise RuntimeError("injected copy failure")

        monkeypatch.setattr(gmsh.model.occ, "copy", fail_after_copy)
        with pytest.raises(RuntimeError, match="injected copy failure"):
            sp._cut_occ_geometry_to_positive_side(
                ("x0",), bbox=(-1, -1, -1, 1, 1, 1), tolerance=0.001,
                grid=3, triangle_model=model,
            )
        assert set(gmsh.model.getEntities(2)) == before
        assert not gmsh.model.getEntities(3)


def test_triangle_copy_failure_never_removes_a_reported_original(monkeypatch):
    with _gmsh_session():
        faces = _triangle_box(n=1)
        model = sp._triangle_model(faces)
        before = set(gmsh.model.getEntities(2))
        original = gmsh.model.occ.copy

        def incorrect_copy(dimtags):
            result = original(dimtags)
            result[0] = dimtags[0]  # An original instead of its new copy.
            return result

        monkeypatch.setattr(gmsh.model.occ, "copy", incorrect_copy)
        with pytest.raises(RuntimeError, match="unexpected triangle face copies"):
            sp._cut_occ_geometry_to_positive_side(
                ("x0",), bbox=(-1, -1, -1, 1, 1, 1), tolerance=0.001,
                grid=3, triangle_model=model,
            )
        assert set(gmsh.model.getEntities(2)) == before
        assert not gmsh.model.getEntities(3)


def test_triangle_cut_refuses_reordered_batch_copies_before_the_boolean(monkeypatch):
    with _gmsh_session():
        faces = _triangle_box(n=1)
        model = sp._triangle_model(faces)
        before = set(gmsh.model.getEntities(2))
        original = gmsh.model.occ.copy
        monkeypatch.setattr(gmsh.model.occ, "copy", lambda dimtags: list(reversed(original(dimtags))))
        monkeypatch.setattr(gmsh.model.occ, "intersect", lambda *_a, **_k: pytest.fail("boolean ran"))
        with pytest.raises(RuntimeError, match="copies in an unexpected order"):
            sp._cut_occ_geometry_to_positive_side(
                ("x0",), bbox=(-1, -1, -1, 1, 1, 1), tolerance=0.001,
                grid=3, triangle_model=model,
            )
        assert set(gmsh.model.getEntities(2)) == before
        assert not gmsh.model.getEntities(3)


def test_triangle_retry_copy_failure_removes_partially_created_copies(monkeypatch):
    with _gmsh_session():
        faces = _triangle_box(n=1)
        model = sp._triangle_model(faces)
        original = gmsh.model.occ.copy
        calls = 0
        created = []

        def fail_during_retry(dimtags):
            nonlocal calls
            calls += 1
            if calls == 2:
                created.extend(original(dimtags[:1]))
                raise RuntimeError("injected retry copy failure")
            return original(dimtags)

        monkeypatch.setattr(gmsh.model.occ, "copy", fail_during_retry)
        monkeypatch.setattr(sp, "_uncovered_cut_parents",
                            lambda mapping, *_a, **_k: [next(iter(mapping))])
        with pytest.raises(RuntimeError, match="injected retry copy failure"):
            sp._cut_occ_geometry_to_positive_side(
                ("x0",), bbox=(-1, -1, -1, 1, 1, 1), tolerance=0.001,
                grid=3, triangle_model=model,
            )
        assert created
        assert not set(created) & set(gmsh.model.occ.getEntities(2))
        assert not gmsh.model.getEntities(3)


def test_triangle_retry_refuses_reordered_copies_and_cleans_them(monkeypatch):
    with _gmsh_session():
        faces = _triangle_box(n=1)
        model = sp._triangle_model(faces)
        original = gmsh.model.occ.copy
        calls = 0
        retry = []

        def reverse_retry(dimtags):
            nonlocal calls
            calls += 1
            copied = original(dimtags)
            if calls == 2:
                retry.extend(copied)
                return list(reversed(copied))
            return copied

        monkeypatch.setattr(gmsh.model.occ, "copy", reverse_retry)
        monkeypatch.setattr(sp, "_uncovered_cut_parents",
                            lambda mapping, *_a, **_k: [next(iter(mapping))])
        with pytest.raises(RuntimeError, match="copies in an unexpected order"):
            sp._cut_occ_geometry_to_positive_side(
                ("x0",), bbox=(-1, -1, -1, 1, 1, 1), tolerance=0.001,
                grid=3, triangle_model=model,
            )
        assert retry
        assert not set(retry) & set(gmsh.model.occ.getEntities(2))
        assert not gmsh.model.getEntities(3)


def test_default_cad_cut_does_not_enter_triangle_specialization(monkeypatch):
    with _gmsh_session():
        occ = gmsh.model.occ
        volume = occ.addBox(-1, -1, -1, 2, 2, 2)
        occ.synchronize()
        faces = [tag for dim, tag in gmsh.model.getBoundary([(3, volume)], oriented=False)
                 if dim == 2]
        monkeypatch.setattr(sp, "_triangle_model", lambda _faces: pytest.fail("triangle path used"))
        group = sp.OccSurfaceGroup("wall", sp.OccSurfaceSelector(faces), sp.OccSurfaceRole("wall"))
        result = sp.auto_cut_occ_geometry([group], planes=("x0",), grid=3)
        assert result.planes == ("x0",)


def test_triangle_cut_handles_disconnected_halves_without_a_crossing_face():
    with _gmsh_session():
        occ = gmsh.model.occ
        faces = []
        for side in (-1, 1):
            coordinates = [(2 * side, 0, 0), (3 * side, 0, 0),
                           (2 * side, 1, 0), (2 * side, 0, 1)]
            vertices = [occ.addPoint(*point) for point in coordinates]
            edges = {}
            for tri in ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)):
                loop = []
                for a, b in zip(tri, tri[1:] + tri[:1]):
                    key = tuple(sorted((a, b)))
                    if key not in edges:
                        edges[key] = occ.addLine(vertices[key[0]], vertices[key[1]])
                    loop.append(edges[key] if a < b else -edges[key])
                faces.append(occ.addPlaneSurface([occ.addCurveLoop(loop)]))
        occ.synchronize()
        model = sp._triangle_model(faces)
        mapping, undecided, stats = sp._cut_occ_geometry_to_positive_side(
            ("x0",), bbox=(-3, 0, 0, 3, 1, 1), tolerance=0.001,
            grid=3, triangle_model=model,
        )
        assert not undecided
        assert all(mapping[tag] == [] for tag in faces[:4])
        assert all(mapping[tag] == [tag] for tag in faces[4:])
        assert stats["surfaces_after"] == 4
        assert {tag for _, tag in gmsh.model.getEntities(2)} == set(faces[4:])
