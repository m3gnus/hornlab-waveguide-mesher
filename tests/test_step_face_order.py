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
