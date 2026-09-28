from __future__ import annotations

import ast
from pathlib import Path
import sys

from hornlab_mesher import step_import, step_text
from hornlab_mesher.step_text import (
    advanced_face_order_from_text,
    count_step_bodies,
    occ_make_solids_is_safe,
    parse_named_shell_faces_from_text,
    parse_solid_brep_faces_from_text,
    record_entity_types,
    step_body_inventory,
    strip_step_comments,
)


_MODULE_PATH = Path(step_text.__file__)


def test_step_text_imports_only_the_standard_library():
    """The whole point of the module: it must run where numpy cannot.

    An embedded CAD Python has no third-party packages, so a single non-stdlib
    import here silently makes the shared body rule unusable on one side of the
    round trip -- which is how the two sides came to count different universes.
    """
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                imported.add(f".{node.module or ''}")
            elif node.module:
                imported.add(node.module.split(".")[0])

    assert imported, "the purity check found no imports at all; it is not looking"
    outside = sorted(name for name in imported if name not in sys.stdlib_module_names)
    assert outside == []


def test_step_import_re_exports_the_text_layer():
    """Existing callers import these from step_import and must keep working."""
    for name in (
        "parse_named_shell_faces",
        "parse_solid_brep_faces",
        "parse_styled_face_groups",
        "advanced_face_order",
        "occ_make_solids_is_safe",
    ):
        assert getattr(step_import, name) is getattr(step_text, name)
    for private, public in (
        ("_step_records", "step_records"),
        ("_step_refs", "step_refs"),
        ("_decode_step_string", "decode_step_string"),
        ("_first_step_string", "first_step_string"),
        ("_parse_named_shell_faces", "parse_named_shell_faces"),
        ("_parse_solid_brep_faces", "parse_solid_brep_faces"),
        ("_parse_styled_face_groups", "parse_styled_face_groups"),
        ("_advanced_face_order", "advanced_face_order"),
    ):
        assert getattr(step_import, private) is getattr(step_text, public)


_TWO_BODY_STEP = """
#10=ADVANCED_FACE('face-a',(),$,.T.);
#11=ADVANCED_FACE('face-b',(),$,.T.);
#12=ADVANCED_FACE('face-c',(),$,.T.);
#20=CLOSED_SHELL('solid shell',(#10,#11));
#21=OPEN_SHELL('sheet shell',(#12));
#30=MANIFOLD_SOLID_BREP('Waveguide body',#20);
#31=SHELL_BASED_SURFACE_MODEL('Exterior sheet',(#21));
"""


def test_body_inventory_names_each_body_and_its_faces():
    inventory = step_body_inventory(_TWO_BODY_STEP)

    assert [(body.record_id, body.entity, body.name, body.kind) for body in inventory] == [
        (30, "MANIFOLD_SOLID_BREP", "Waveguide body", "solid"),
        (31, "SHELL_BASED_SURFACE_MODEL", "Exterior sheet", "surface"),
    ]
    assert inventory[0].face_ids == (10, 11)
    assert inventory[1].face_ids == (12,)


def test_the_count_is_the_length_of_the_inventory():
    """Two body rules that can disagree is the bug; there is only one rule."""
    for text in (
        "",
        _TWO_BODY_STEP,
        _TWO_BODY_STEP + "#40=BREP_WITH_VOIDS('Voided',#20,(#41));",
        "#1=CARTESIAN_POINT('',(0.,0.,0.));",
    ):
        assert count_step_bodies(text) == len(step_body_inventory(text))

    assert count_step_bodies(_TWO_BODY_STEP) == 2


def test_a_body_named_after_an_entity_type_is_not_a_second_body():
    text = "#30=SHELL_BASED_SURFACE_MODEL('MANIFOLD_SOLID_BREP',(#21));"

    inventory = step_body_inventory(text)

    assert count_step_bodies(text) == 1
    assert inventory[0].entity == "SHELL_BASED_SURFACE_MODEL"
    assert inventory[0].name == "MANIFOLD_SOLID_BREP"


def test_a_record_spelled_out_inside_a_comment_is_not_a_body():
    text = "/* #99=MANIFOLD_SOLID_BREP('ghost',#20); */\n" + _TWO_BODY_STEP

    assert count_step_bodies(text) == 2
    assert 99 not in {body.record_id for body in step_body_inventory(text)}


def test_a_comment_delimiter_inside_a_label_does_not_open_a_comment():
    text = "#30=MANIFOLD_SOLID_BREP('a /* b',#20);#31=SHELL_BASED_SURFACE_MODEL('c */ d',(#21));"

    assert count_step_bodies(text) == 2
    assert [body.name for body in step_body_inventory(text)] == ["a /* b", "c */ d"]


def test_a_quote_inside_a_comment_does_not_desynchronise_the_scan():
    text = "/* it's a comment */\n" + _TWO_BODY_STEP

    assert count_step_bodies(text) == 2


def test_a_line_break_inside_the_entity_keyword_still_counts():
    """ISO 10303-21 ed.3 clause 5.6 lets a record be folded anywhere."""
    text = "#30=MANIFOLD_SOLID\n_BREP('folded',#20);\n#20=CLOSED_SHELL('',(#10));"

    inventory = step_body_inventory(text)

    assert count_step_bodies(text) == 1
    assert inventory[0].entity == "MANIFOLD_SOLID_BREP"
    assert inventory[0].name == "folded"


def test_a_semicolon_inside_a_label_does_not_split_a_body_record():
    text = "#30=MANIFOLD_SOLID_BREP('left; right',#20);#20=CLOSED_SHELL('',(#10));"

    assert count_step_bodies(text) == 1
    assert step_body_inventory(text)[0].name == "left; right"


def test_a_voided_solid_reports_its_void_faces_too():
    text = (
        "#10=ADVANCED_FACE('outer',(),$,.T.);"
        "#11=ADVANCED_FACE('void',(),$,.T.);"
        "#20=CLOSED_SHELL('',(#10));"
        "#21=CLOSED_SHELL('',(#11));"
        "#22=ORIENTED_CLOSED_SHELL('',*,#21,.F.);"
        "#30=BREP_WITH_VOIDS('Voided',#20,(#22));"
    )

    inventory = step_body_inventory(text)

    assert count_step_bodies(text) == 1
    assert inventory[0].kind == "solid"
    assert sorted(inventory[0].face_ids) == [10, 11]


# Numbered clear of _TWO_BODY_STEP so the two can be concatenated into one
# file without their record ids colliding.
_VOIDED_SOLID_STEP = (
    "#110=ADVANCED_FACE('outer',(),$,.T.);"
    "#111=ADVANCED_FACE('void',(),$,.T.);"
    "#120=CLOSED_SHELL('',(#110));"
    "#121=CLOSED_SHELL('',(#111));"
    "#122=ORIENTED_CLOSED_SHELL('',*,#121,.F.);"
    "#130=BREP_WITH_VOIDS('Voided',#120,(#122));"
)


def test_a_plain_solid_may_be_rebuilt():
    """Sewing dissolves it and MakeSolids restores it; nothing else changes."""
    assert occ_make_solids_is_safe(_TWO_BODY_STEP) is True


def test_a_body_with_interior_voids_may_never_be_rebuilt():
    """Measured: rebuilding fills the cavity and deletes its inner shell.

    A 40 mm box minus an enclosed r=10 sphere goes from mass 59811.21 to
    64000.00 -- the solid box -- while the volume count stays 1, so no
    counting gate in this family can see it.
    """
    assert occ_make_solids_is_safe(_VOIDED_SOLID_STEP) is False


def test_one_voided_body_disqualifies_a_file_that_also_holds_a_plain_solid():
    """The option is per-file, so the hollow body would be filled anyway."""
    mixed = _TWO_BODY_STEP + _VOIDED_SOLID_STEP

    assert count_step_bodies(mixed) == 3
    assert occ_make_solids_is_safe(mixed) is False


def test_a_file_with_no_solid_body_has_nothing_to_rebuild():
    """Nothing to restore, and coincident sheets would sew into a phantom."""
    surface_only = "#30=SHELL_BASED_SURFACE_MODEL('Exterior sheet',(#21));"

    assert occ_make_solids_is_safe(surface_only) is False
    assert occ_make_solids_is_safe("") is False


def test_being_a_solid_body_and_being_safe_to_rebuild_are_different_questions():
    """The one entity where the two answers differ is the whole point.

    A hollow body is ONE solid body -- so it counts as one, and
    ``SOLID_BODY_ENTITIES`` must keep matching it -- and it is exactly the
    body OCC must not re-solidify. Answering the second question with the
    first turns a loud failure into silent geometry corruption.
    """
    inventory = step_body_inventory(_VOIDED_SOLID_STEP)

    assert [body.kind for body in inventory] == ["solid"]
    assert inventory[0].entity in step_text.SOLID_BODY_ENTITIES
    assert inventory[0].entity in step_text.VOIDED_SOLID_BODY_ENTITIES
    assert occ_make_solids_is_safe(_VOIDED_SOLID_STEP) is False


def test_a_complex_record_still_declares_its_keywords():
    assert record_entity_types("MANIFOLD_SOLID_BREP('x',#20)") == ("MANIFOLD_SOLID_BREP",)
    assert record_entity_types("( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#1)) )") == (
        "GEOMETRIC_REPRESENTATION_CONTEXT",
        "GLOBAL_UNIT_ASSIGNED_CONTEXT",
    )


def test_strip_step_comments_leaves_string_literals_alone():
    assert strip_step_comments("A('keep /* me */ intact');/* drop */") == (
        "A('keep /* me */ intact'); "
    )


def test_the_text_parsers_agree_with_the_path_parsers(tmp_path):
    step_path = tmp_path / "model.step"
    step_path.write_text(_TWO_BODY_STEP, encoding="ascii")

    assert parse_named_shell_faces_from_text(_TWO_BODY_STEP) == step_text.parse_named_shell_faces(
        step_path
    )
    assert parse_solid_brep_faces_from_text(_TWO_BODY_STEP) == step_text.parse_solid_brep_faces(
        step_path
    )
    assert advanced_face_order_from_text(_TWO_BODY_STEP) == step_text.advanced_face_order(step_path)
    assert parse_named_shell_faces_from_text(_TWO_BODY_STEP) == {"Exterior sheet": [12]}
    assert parse_solid_brep_faces_from_text(_TWO_BODY_STEP) == {10, 11}


def test_face_order_ignores_a_record_spelled_out_inside_a_comment():
    """A commented-out face used to count, so a caller comparing the face
    count with the imported surfaces failed with a misleading mismatch."""
    text = _TWO_BODY_STEP + "/* #9999=ADVANCED_FACE('ghost',(),$,.T.); */\n"
    assert advanced_face_order_from_text(text) == [10, 11, 12]
