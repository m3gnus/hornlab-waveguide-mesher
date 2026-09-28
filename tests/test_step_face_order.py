"""STEP ADVANCED_FACE -> gmsh surface mapping.

gmsh numbers surfaces by walking OCC's shapes (every solid before any free
face, and each shell as the STEP reader rebuilt it), not by STEP record
order. These tests build files whose record order and traversal order
disagree and check that labels land on the geometry they name.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re

import numpy as np
import pytest

from hornlab_mesher.step_import import (
    StepFaceGroup,
    StepFaceOrderError,
    StepLabelSelector,
    advanced_face_order,
    advanced_face_order_for_surfaces,
    gmsh_surface_tags,
    map_step_face_groups,
    named_shell_gmsh_surfaces,
)
from hornlab_mesher.step_prepare import OccSurfaceRole
from hornlab_mesher.step_text import (
    advanced_face_placements_from_text,
    advanced_face_vertices_from_text,
    step_length_unit_mm_from_text,
)


@contextmanager
def _gmsh_session():
    import gmsh

    initialized_here = not gmsh.isInitialized()
    if initialized_here:
        gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        yield gmsh
    finally:
        gmsh.clear()
        if initialized_here:
            gmsh.finalize()


def _write_box_and_sheet(path: Path, *, sheet_first: bool = False) -> None:
    with _gmsh_session() as gmsh:
        if sheet_first:
            gmsh.model.occ.addRectangle(50, 0, 0, 5, 5)
            gmsh.model.occ.addBox(0, 0, 0, 10, 20, 30)
        else:
            gmsh.model.occ.addBox(0, 0, 0, 10, 20, 30)
            gmsh.model.occ.addRectangle(50, 0, 0, 5, 5)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))


def _reverse_first_closed_shell(path: Path) -> None:
    text = path.read_text(encoding="ascii")
    match = re.search(r"CLOSED_SHELL\('',\(([^)]*)\)\)", text)
    assert match is not None
    faces = [face.strip() for face in match.group(1).split(",")]
    reversed_shell = "CLOSED_SHELL('',(%s))" % ",".join(reversed(faces))
    path.write_text(text.replace(match.group(0), reversed_shell), encoding="ascii")


def _move_sheet_records_first(path: Path) -> None:
    """Rewrite the file so the free sheet's part is written before the solid.

    Only the record order in the file changes; ids, references and geometry
    are untouched, so the file is semantically the same model.
    """
    lines = path.read_text(encoding="ascii").split("\n")
    data = lines.index("DATA;")
    body_line = next(i for i, line in enumerate(lines) if "SHELL_BASED_SURFACE_MODEL" in line)
    # The sheet's part starts after the NAUO placing the solid, and runs to
    # the NAUO placing the sheet itself.
    nauo = [i for i, line in enumerate(lines) if "NEXT_ASSEMBLY_USAGE_OCCURRENCE" in line]
    start = max(i for i in nauo if i < body_line) + 1
    end = min(i for i in nauo if i > body_line) + 1
    block = lines[start:end]
    rest = lines[:start] + lines[end:]
    path.write_text("\n".join(rest[: data + 1] + block + rest[data + 1 :]), encoding="ascii")


def _vertex_mean_mm(step_path: Path) -> dict[int, np.ndarray]:
    faces = advanced_face_vertices_from_text(step_path.read_text(encoding="ascii"))
    return {face: np.mean(np.asarray(points), axis=0) for face, points in faces.items()}


def _assert_every_face_lands_on_its_own_geometry(gmsh, step_path: Path, order: list[int]) -> None:
    """Independent oracle: each face here is a rectangle, so its vertex mean
    is the centre of mass OCC reports for the surface it was imported as."""
    means = _vertex_mean_mm(step_path)
    for face, tag in zip(order, gmsh_surface_tags(), strict=True):
        centre = np.asarray(gmsh.model.occ.getCenterOfMass(2, tag))
        assert np.allclose(centre, means[face], atol=1e-6), (face, tag)


def _record_order_is_wrong(gmsh, step_path: Path) -> bool:
    means = _vertex_mean_mm(step_path)
    return any(
        not np.allclose(np.asarray(gmsh.model.occ.getCenterOfMass(2, tag)), means[face], atol=1e-6)
        for face, tag in zip(advanced_face_order(step_path), gmsh_surface_tags())
    )


def test_an_ordinary_export_maps_exactly_as_record_order_did(tmp_path):
    step_path = tmp_path / "box.step"
    _write_box_and_sheet(step_path)
    with _gmsh_session() as gmsh:
        gmsh.model.occ.importShapes(str(step_path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        order = advanced_face_order_for_surfaces(step_path)
        assert order == advanced_face_order(step_path)
        _assert_every_face_lands_on_its_own_geometry(gmsh, step_path, order)


def test_a_shell_listing_its_faces_in_another_order_maps_by_geometry(tmp_path):
    """The reviewer's case: reverse one shell's face list, nothing else."""
    step_path = tmp_path / "reversed.step"
    _write_box_and_sheet(step_path)
    _reverse_first_closed_shell(step_path)
    with _gmsh_session() as gmsh:
        gmsh.model.occ.importShapes(str(step_path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        assert _record_order_is_wrong(gmsh, step_path), "the fixture no longer reorders anything"
        order = advanced_face_order_for_surfaces(step_path)
        _assert_every_face_lands_on_its_own_geometry(gmsh, step_path, order)


def test_a_surface_body_written_before_a_solid_maps_by_geometry(tmp_path):
    """gmsh binds every solid's faces before any free face, whatever the file order."""
    step_path = tmp_path / "sheet-first.step"
    _write_box_and_sheet(step_path, sheet_first=True)
    _move_sheet_records_first(step_path)
    with _gmsh_session() as gmsh:
        gmsh.model.occ.importShapes(str(step_path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        assert _record_order_is_wrong(gmsh, step_path), "the fixture no longer reorders anything"
        order = advanced_face_order_for_surfaces(step_path)
        _assert_every_face_lands_on_its_own_geometry(gmsh, step_path, order)


def test_a_named_sheet_label_lands_on_the_sheet(tmp_path):
    step_path = tmp_path / "named.step"
    _write_box_and_sheet(step_path, sheet_first=True)
    _move_sheet_records_first(step_path)
    text = step_path.read_text(encoding="ascii")
    step_path.write_text(
        text.replace("SHELL_BASED_SURFACE_MODEL('',", "SHELL_BASED_SURFACE_MODEL('driver',"),
        encoding="ascii",
    )
    group = StepFaceGroup(
        name="driver",
        selector=StepLabelSelector("driver"),
        role=OccSurfaceRole("source"),
        tag=2,
    )
    with _gmsh_session() as gmsh:
        gmsh.model.occ.importShapes(str(step_path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        (sheet,) = [tag for _dim, tag in gmsh.model.getEntities(2) if len(gmsh.model.getAdjacencies(2, tag)[0]) == 0]
        assert map_step_face_groups(step_path, [group]).surfaces == {"driver": [sheet]}
        assert named_shell_gmsh_surfaces(step_path, gmsh_surface_tags()) == {"driver": [sheet]}


def test_a_moved_model_needs_the_move_and_refuses_without_it(tmp_path):
    step_path = tmp_path / "reversed.step"
    _write_box_and_sheet(step_path)
    _reverse_first_closed_shell(step_path)
    with _gmsh_session() as gmsh:
        gmsh.model.occ.importShapes(str(step_path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        offset = np.array([0.0, -40.0, 12.5])
        gmsh.model.occ.translate(gmsh.model.getEntities(3) + [(2, 7)], *offset)
        gmsh.model.occ.synchronize()
        with pytest.raises(StepFaceOrderError, match="does not match the file geometry"):
            advanced_face_order_for_surfaces(step_path)
        moved = np.eye(4)
        moved[:3, 3] = offset
        order = advanced_face_order_for_surfaces(step_path, model_from_step=moved)
        means = _vertex_mean_mm(step_path)
        for face, tag in zip(order, gmsh_surface_tags(), strict=True):
            centre = np.asarray(gmsh.model.occ.getCenterOfMass(2, tag))
            assert np.allclose(centre, means[face] + offset, atol=1e-6), (face, tag)


def test_file_units_and_assembly_placements_are_applied(tmp_path):
    step_path = tmp_path / "placed-metres.step"
    _write_box_and_sheet(step_path)
    _reverse_first_closed_shell(step_path)
    text = step_path.read_text(encoding="ascii")
    # Metres instead of millimetres, and the solid's part rotated and moved
    # in the assembly: gmsh applies both on import.
    text = text.replace("SI_UNIT(.MILLI.,.METRE.)", "SI_UNIT($,.METRE.)")
    text = text.replace("#16 = CARTESIAN_POINT('',(0.,0.,0.));", "#16 = CARTESIAN_POINT('',(0.1,0.005,0.));")
    text = text.replace("#17 = DIRECTION('',(0.,0.,1.));", "#17 = DIRECTION('',(1.,0.,0.));")
    text = text.replace("#18 = DIRECTION('',(1.,0.,-0.));", "#18 = DIRECTION('',(0.,1.,0.));")
    step_path.write_text(text, encoding="ascii")
    assert step_length_unit_mm_from_text(text) == 1000.0
    placements = advanced_face_placements_from_text(text)
    assert any(matrix is not None and matrix[0][3] != 0.0 for matrix in placements.values())
    with _gmsh_session() as gmsh:
        gmsh.model.occ.importShapes(str(step_path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        order = advanced_face_order_for_surfaces(step_path)
        means = _vertex_mean_mm(step_path)
        for face, tag in zip(order, gmsh_surface_tags(), strict=True):
            placed = np.asarray(placements[face])
            expected = (placed[:3, :3] @ means[face] + placed[:3, 3]) * 1000.0
            centre = np.asarray(gmsh.model.occ.getCenterOfMass(2, tag))
            assert np.allclose(centre, expected, atol=1e-6), (face, tag)


def test_indistinguishable_faces_with_different_labels_are_refused(tmp_path):
    """Two coincident sheets in two parts, named differently: nothing in the
    file says which imported surface is which, so the mapper must not guess."""
    step_path = tmp_path / "coincident.step"
    with _gmsh_session() as gmsh:
        gmsh.model.occ.addRectangle(0, 0, 0, 5, 5)
        gmsh.model.occ.addRectangle(0, 0, 0, 5, 5)
        gmsh.model.occ.synchronize()
        gmsh.write(str(step_path))
    text = step_path.read_text(encoding="ascii")
    text = text.replace("SHELL_BASED_SURFACE_MODEL('',", "SHELL_BASED_SURFACE_MODEL('left',", 1)
    text = text.replace("SHELL_BASED_SURFACE_MODEL('',", "SHELL_BASED_SURFACE_MODEL('right',", 1)
    step_path.write_text(text, encoding="ascii")
    with _gmsh_session() as gmsh:
        gmsh.model.occ.importShapes(str(step_path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        assert len(gmsh_surface_tags()) == 2
        with pytest.raises(StepFaceOrderError, match="unambiguously"):
            advanced_face_order_for_surfaces(step_path)


def test_indistinguishable_unlabelled_faces_keep_record_order(tmp_path):
    step_path = tmp_path / "coincident.step"
    with _gmsh_session() as gmsh:
        gmsh.model.occ.addRectangle(0, 0, 0, 5, 5)
        gmsh.model.occ.addRectangle(0, 0, 0, 5, 5)
        gmsh.model.occ.synchronize()
        gmsh.write(str(step_path))
    with _gmsh_session() as gmsh:
        gmsh.model.occ.importShapes(str(step_path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        assert advanced_face_order_for_surfaces(step_path) == advanced_face_order(step_path)
        # A caller that will select one of them by id makes the pair distinct;
        # these two are separate bodies in separate parts, so nothing orders them.
        with pytest.raises(StepFaceOrderError, match="unambiguously"):
            advanced_face_order_for_surfaces(
                step_path, addressed_faces=advanced_face_order(step_path)[:1]
            )


def test_a_count_mismatch_is_refused(tmp_path):
    step_path = tmp_path / "box.step"
    _write_box_and_sheet(step_path)
    with _gmsh_session() as gmsh:
        gmsh.model.occ.importShapes(str(step_path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        with pytest.raises(StepFaceOrderError, match="7 ADVANCED_FACE records, 6 gmsh surfaces"):
            advanced_face_order_for_surfaces(step_path, gmsh_surface_tags()[:6])


def test_the_text_parsers_read_units_placements_and_vertices():
    text = """
#1=CARTESIAN_POINT('',(1.,2.,3.));
#2=VERTEX_POINT('',#1);
#3=CARTESIAN_POINT('',(4.,5.,6.));
#4=VERTEX_POINT('',#3);
#5=LINE('',#1,#99);
#6=EDGE_CURVE('',#2,#4,#5,.T.);
#7=ORIENTED_EDGE('',*,*,#6,.T.);
#8=EDGE_LOOP('',(#7));
#9=FACE_OUTER_BOUND('',#8,.T.);
#10=ADVANCED_FACE('named; face',(#9),#11,.T.);
#11=PLANE('',#12);
#12=AXIS2_PLACEMENT_3D('',#13,$,$);
#13=CARTESIAN_POINT('',(100.,100.,100.));
#20=(CONVERSION_BASED_UNIT('INCH',#21)LENGTH_UNIT()NAMED_UNIT(#22));
#21=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#23);
#23=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));
#24=(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNIT_ASSIGNED_CONTEXT((#20))REPRESENTATION_CONTEXT('',''));
/* #30=ADVANCED_FACE('commented out',(#9),#11,.T.); */
"""
    # The surface's own placement point is not a vertex of the face.
    assert advanced_face_vertices_from_text(text) == {10: ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))}
    assert step_length_unit_mm_from_text(text) == pytest.approx(25.4)


def _write_box(path: Path, size=(10.0, 20.0, 30.0)) -> None:
    with _gmsh_session() as gmsh:
        gmsh.model.occ.addBox(0, 0, 0, *size)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))


def _set_root_placement(path: Path, *, origin=None, ref_direction=None) -> None:
    """Change the root representation's own AXIS2_PLACEMENT_3D (item 1)."""
    text = path.read_text(encoding="ascii")
    root = re.search(
        r"ADVANCED_BREP_SHAPE_REPRESENTATION\('',\(#(\d+),", text
    )
    assert root is not None
    placement = re.search(
        rf"#{root.group(1)} = AXIS2_PLACEMENT_3D\('',#(\d+),#(\d+),#(\d+)\);", text
    )
    assert placement is not None
    point, _axis, direction = placement.groups()
    if origin is not None:
        text = re.sub(
            rf"(#{point} = CARTESIAN_POINT\(''),\([^)]*\)\)",
            lambda m: f"{m.group(1)},({','.join(repr(float(v)) for v in origin)}))",
            text,
        )
    if ref_direction is not None:
        text = re.sub(
            rf"(#{direction} = DIRECTION\(''),\([^)]*\)\)",
            lambda m: f"{m.group(1)},({','.join(repr(float(v)) for v in ref_direction)}))",
            text,
        )
    path.write_text(text, encoding="ascii")


@pytest.mark.parametrize(
    "root",
    [
        {"ref_direction": (-1.0, 0.0, 0.0)},  # 180 deg about z: the box is symmetric under it
        {"ref_direction": (0.0, 1.0, 0.0)},  # 90 deg about z
        {"origin": (100.0, 0.0, 0.0)},
        {"origin": (5.0, -7.0, 3.0), "ref_direction": (0.0, -1.0, 0.0)},
    ],
    ids=["rot180", "rot90", "move", "rot-move"],
)
def test_root_representation_placement_is_applied(tmp_path, root):
    """OCC moves the whole import by the root representation's placement."""
    step_path = tmp_path / "box-root.step"
    _write_box(step_path)
    _set_root_placement(step_path, **root)
    with _gmsh_session() as gmsh:
        gmsh.model.occ.importShapes(str(step_path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        order = advanced_face_order_for_surfaces(step_path)
        placements = advanced_face_placements_from_text(step_path.read_text(encoding="ascii"))
        means = _vertex_mean_mm(step_path)
        for face, tag in zip(order, gmsh_surface_tags(), strict=True):
            placed = np.asarray(placements[face])
            expected = placed[:3, :3] @ means[face] + placed[:3, 3]
            centre = np.asarray(gmsh.model.occ.getCenterOfMass(2, tag))
            assert np.allclose(centre, expected, atol=1e-6), (face, tag)
        # gmsh writes faces in traversal order, so the right map is record order.
        assert order == advanced_face_order(step_path)


def test_geometry_free_assembly_root_placement_is_not_applied(tmp_path):
    """OCC leaves a root that holds only placements alone; so must the parser."""
    step_path = tmp_path / "placed-root.step"
    _write_box_and_sheet(step_path)
    text = step_path.read_text(encoding="ascii")
    root = "#10 = SHAPE_REPRESENTATION('',(#11,#15,#19),#23);"
    assert root in text
    text = text.replace(root, "#10 = SHAPE_REPRESENTATION('',(#901,#15,#19),#23);")
    text = text.replace(
        "ENDSEC;\nEND-ISO",
        "#901 = AXIS2_PLACEMENT_3D('',#902,#903,#904);\n"
        "#902 = CARTESIAN_POINT('',(7.,-3.,11.));\n"
        "#903 = DIRECTION('',(0.,0.,1.));\n"
        "#904 = DIRECTION('',(0.,-1.,0.));\n"
        "ENDSEC;\nEND-ISO",
    )
    assert "#901 =" in text
    step_path.write_text(text, encoding="ascii")
    placements = advanced_face_placements_from_text(text)
    assert all(np.allclose(np.asarray(m), np.eye(4)) for m in placements.values())
    with _gmsh_session() as gmsh:
        gmsh.model.occ.importShapes(str(step_path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        assert advanced_face_order_for_surfaces(step_path) == advanced_face_order(step_path)


def test_conflicting_root_placements_are_refused(tmp_path):
    step_path = tmp_path / "two-placements.step"
    _write_box(step_path)
    text = step_path.read_text(encoding="ascii")
    text = text.replace(
        "ADVANCED_BREP_SHAPE_REPRESENTATION('',(#11,#15),",
        "ADVANCED_BREP_SHAPE_REPRESENTATION('',(#901,#911,#15),",
    )
    text = text.replace(
        "ENDSEC;\nEND-ISO",
        "#901 = AXIS2_PLACEMENT_3D('',#902,#903,#904);\n"
        "#902 = CARTESIAN_POINT('',(100.,0.,0.));\n#903 = DIRECTION('',(0.,0.,1.));\n"
        "#904 = DIRECTION('',(1.,0.,0.));\n"
        "#911 = AXIS2_PLACEMENT_3D('',#912,#913,#914);\n"
        "#912 = CARTESIAN_POINT('',(0.,0.,0.));\n#913 = DIRECTION('',(0.,0.,1.));\n"
        "#914 = DIRECTION('',(0.,1.,0.));\n"
        "ENDSEC;\nEND-ISO",
    )
    step_path.write_text(text, encoding="ascii")
    assert all(m is None for m in advanced_face_placements_from_text(text).values())
    with _gmsh_session() as gmsh:
        gmsh.model.occ.importShapes(str(step_path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        with pytest.raises(StepFaceOrderError, match="not determined"):
            advanced_face_order_for_surfaces(step_path)


def _write_two_sheet_bodies(path: Path, offsets: tuple[float, float], order: tuple[int, int]) -> None:
    """Two named surface bodies (a two-face fan each) in ONE representation,
    the bodies listed in ``order``. Written by hand: gmsh writes one part per
    face, so it cannot produce this shape."""
    records: list[str] = []

    def add(text: str) -> int:
        records.append(text)
        return len(records)

    def point(x: float, y: float, z: float) -> int:
        return add(f"CARTESIAN_POINT('',({x:.6f},{y:.6f},{z:.6f}))")

    def direction(x: float, y: float, z: float) -> int:
        return add(f"DIRECTION('',({x:.6f},{y:.6f},{z:.6f}))")

    def edge(a: int, b: int, origin: int, vector: tuple[float, float, float]) -> int:
        vec = add(f"VECTOR('',#{direction(*vector)},1.0)")
        line = add(f"LINE('',#{origin},#{vec})")
        return add(f"EDGE_CURVE('',#{a},#{b},#{line},.T.)")

    app = add("APPLICATION_CONTEXT('automotive design')")
    add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app})")
    pctx = add(f"PRODUCT_CONTEXT('',#{app},'mechanical')")
    prod = add(f"PRODUCT('s','s','',(#{pctx}))")
    form = add(f"PRODUCT_DEFINITION_FORMATION('','',#{prod})")
    dctx = add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')")
    definition = add(f"PRODUCT_DEFINITION('design','',#{form},#{dctx})")
    product = add(f"PRODUCT_DEFINITION_SHAPE('','',#{definition})")
    length = add("( NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) LENGTH_UNIT() )")
    angle = add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
    solid = add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
    tol = add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#{length},'d','')")
    context = add(
        f"( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{tol})) "
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid})) REPRESENTATION_CONTEXT('','') )"
    )
    bodies: list[int] = []
    for name, z in zip("ab", offsets):
        bottom, top = point(0, 0, z), point(0, 0, z + 1)
        v_bottom, v_top = add(f"VERTEX_POINT('',#{bottom})"), add(f"VERTEX_POINT('',#{top})")
        spine = edge(v_bottom, v_top, bottom, (0, 0, 1))
        faces = []
        for k in (1, 2):
            angle = np.pi * k / 3
            dx, dy = float(np.cos(angle)), float(np.sin(angle))
            p_top, p_bottom = point(dx, dy, z + 1), point(dx, dy, z)
            w_top, w_bottom = add(f"VERTEX_POINT('',#{p_top})"), add(f"VERTEX_POINT('',#{p_bottom})")
            loop_edges = [
                spine,
                edge(v_top, w_top, top, (dx, dy, 0)),
                edge(w_top, w_bottom, p_top, (0, 0, -1)),
                edge(w_bottom, v_bottom, p_bottom, (-dx, -dy, 0)),
            ]
            oriented = [add(f"ORIENTED_EDGE('',*,*,#{e},.T.)") for e in loop_edges]
            loop = add("EDGE_LOOP('',(" + ",".join(f"#{o}" for o in oriented) + "))")
            bound = add(f"FACE_OUTER_BOUND('',#{loop},.T.)")
            axis = add(
                f"AXIS2_PLACEMENT_3D('',#{bottom},#{direction(dy, -dx, 0)},#{direction(dx, dy, 0)})"
            )
            plane = add(f"PLANE('',#{axis})")
            faces.append(add(f"ADVANCED_FACE('{name}{k}',(#{bound}),#{plane},.T.)"))
        shell = add(f"OPEN_SHELL('shell{name}',(" + ",".join(f"#{f}" for f in faces) + "))")
        bodies.append(add(f"SHELL_BASED_SURFACE_MODEL('{name}',(#{shell}))"))
    listed = ",".join(f"#{bodies[i]}" for i in order)
    rep = add(f"MANIFOLD_SURFACE_SHAPE_REPRESENTATION('',({listed}),#{context})")
    add(f"SHAPE_DEFINITION_REPRESENTATION(#{product},#{rep})")
    data = "\n".join(f"#{n + 1} = {text};" for n, text in enumerate(records))
    path.write_text(
        "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('t'),'2;1');\n"
        "FILE_NAME('t','2026-01-01T00:00:00',(''),(''),'','','');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n"
        + data
        + "\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="ascii",
    )


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_bodies_of_one_representation_are_walked_in_listed_order(tmp_path, order):
    """The order rule for coincident faces of different bodies rests on this:
    with the bodies apart, gmsh's surfaces follow the representation's item
    order (checked by geometry). Then the same file with the bodies exactly
    coincident maps by that order."""
    apart = tmp_path / "apart.step"
    _write_two_sheet_bodies(apart, (0.0, 10.0), order)
    with _gmsh_session() as gmsh:
        gmsh.model.occ.importShapes(str(apart), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        heights = [
            round(gmsh.model.occ.getCenterOfMass(2, tag)[2])
            for tag in gmsh_surface_tags()
        ]
    first, second = (0, 10) if order == (0, 1) else (10, 0)
    assert heights == [first, first, second, second]

    together = tmp_path / "together.step"
    _write_two_sheet_bodies(together, (0.0, 0.0), order)
    text = together.read_text(encoding="ascii")
    with _gmsh_session() as gmsh:
        gmsh.model.occ.importShapes(str(together), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        mapped = advanced_face_order_for_surfaces(together)
    label = {
        int(m.group(1)): m.group(2)
        for m in re.finditer(r"#(\d+) = ADVANCED_FACE\('([ab])\d'", text)
    }
    listed = "ab" if order == (0, 1) else "ba"
    assert [label[face] for face in mapped] == [listed[0]] * 2 + [listed[1]] * 2
