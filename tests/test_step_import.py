from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import math
import re

import numpy as np
import pytest

from hornlab_mesher.step_import import (
    OCC_HEALING_FALLBACKS,
    RIGID_TAG,
    StepFaceGroup,
    StepLabelSelector,
    detect_symmetry_planes,
    map_step_face_groups,
    occ_make_solids_is_safe,
    run_occ_healing_fallbacks,
)
from hornlab_mesher.step_prepare import OccSurfaceRole


def test_step_face_mapping_matches_the_caller_label_and_keeps_roles_opaque(tmp_path):
    """The mapper matches the string it is handed and never interprets it.

    The role is an arbitrary caller string and must come back untouched -- this
    module has no vocabulary of its own, which is what D4 requires of it.
    """
    step_path = tmp_path / "model.step"
    step_path.write_text("#10=ADVANCED_FACE('',(),$,.T.);\n", encoding="ascii")
    role = OccSurfaceRole("caller-owned-role")
    group = StepFaceGroup(
        name="requested-label",
        selector=StepLabelSelector("requested-label"),
        role=role,
        tag=7,
        resolution_mm=3.5,
    )

    result = map_step_face_groups(
        step_path,
        [group],
        gmsh_surfaces=[101],
        named_faces={},
        styled_faces={"requested-label": [10]},
        face_order=[10],
    )

    assert group.role is role
    assert result.surfaces == {"requested-label": [101]}
    assert result.origins == {"requested-label": "appearance/style"}
    assert result.missing_reasons == {}


def test_step_face_mapping_does_not_fall_back_to_another_label(tmp_path):
    """A label that is absent is MISSING, never quietly satisfied by another.

    Guards the deletion of the PORT_EXIT_L/_R alias: across 471 recorded runs
    that fallback never once fired, so silently resolving a different label is
    behaviour nothing has ever depended on and nobody should reintroduce by
    accident.
    """
    step_path = tmp_path / "model.step"
    step_path.write_text("#10=ADVANCED_FACE('',(),$,.T.);\n", encoding="ascii")
    group = StepFaceGroup(
        name="requested-label",
        selector=StepLabelSelector("requested-label"),
        role=OccSurfaceRole("caller-owned-role"),
        tag=7,
        resolution_mm=3.5,
    )

    result = map_step_face_groups(
        step_path,
        [group],
        skip_missing_groups=True,
        gmsh_surfaces=[101],
        named_faces={},
        styled_faces={"some-other-label": [10]},
        face_order=[10],
    )

    assert result.surfaces == {}
    assert "requested-label" in result.missing_reasons


def test_step_import_keeps_application_vocabulary_out_of_mesher():
    source = (Path(__file__).parents[1] / "hornlab_mesher" / "step_import.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "PORT_EXIT",
        "FEM_MF_AIR",
        "SOURCE_TAG_BASE",
    )
    assert all(name not in source for name in forbidden)
    assert re.search(r"\b(?:LF|MF|HF)\b", source) is None
    assert RIGID_TAG == 1


def test_healing_fallback_returns_rejected_rung_records():
    original = RuntimeError("unhealed failed")
    calls = 0

    def attempt(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("sew rejected")
        return {"mesh_generation_error": None, "mesh": "ok"}

    state, mode, rejected = run_occ_healing_fallbacks(
        attempt,
        original_mesh_error=original,
        original_traceback=original.__traceback__,
        surface_order_reference=[],
    )

    assert state["mesh"] == "ok"
    assert mode == "full"
    assert rejected == [
        {
            "mode": "sew",
            "options": list(OCC_HEALING_FALLBACKS[0][1]),
            "reason": "OCC sew repair rejected before meshing (RuntimeError): sew rejected",
        }
    ]


class _ScopeGateError(ValueError):
    """Stands in for a caller gate that raises a ValueError subclass.

    The Waveguide Generator server's ImportedMeshError is exactly this shape,
    which is why the ladder used to lose it.
    """


def test_a_caller_gate_raising_a_value_error_only_rejects_that_rung():
    """A rejected rung must not kill the ladder, nor replace the real error.

    Before this, ``run_occ_healing_fallbacks`` caught only RuntimeError, so a
    scope-gate ValueError raised inside run_attempt escaped the ladder, skipped
    every remaining rung, and surfaced instead of the unhealed mesh failure the
    user actually needed to see.
    """
    original = RuntimeError("unhealed failed")
    calls = 0

    def attempt(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _ScopeGateError("STEP contains 17 exterior bodies; manifest declares 1")
        return {"mesh_generation_error": None, "mesh": "ok"}

    state, mode, rejected = run_occ_healing_fallbacks(
        attempt,
        original_mesh_error=original,
        original_traceback=original.__traceback__,
        surface_order_reference=[],
    )

    assert calls == 2
    assert state["mesh"] == "ok"
    assert mode == "full"
    assert len(rejected) == 1
    assert rejected[0]["mode"] == "sew"
    assert "_ScopeGateError" in rejected[0]["reason"]
    assert "manifest declares 1" in rejected[0]["reason"]


def test_every_rung_rejected_by_a_value_error_still_raises_the_original():
    """The unhealed mesh failure is the diagnosis; a rejection is a footnote."""
    original = RuntimeError("unhealed failed")

    def attempt(**_kwargs):
        raise _ScopeGateError("scope gate refused")

    with pytest.raises(RuntimeError) as excinfo:
        run_occ_healing_fallbacks(
            attempt,
            original_mesh_error=original,
            original_traceback=original.__traceback__,
            surface_order_reference=[],
        )

    assert excinfo.value is original
    assert any("scope gate refused" in note for note in getattr(original, "__notes__", []))


def test_no_healing_rung_names_make_solids():
    """The rungs are a repair STRATEGY contract, shared across repositories.

    ``Geometry.OCCMakeSolids`` is a conditional undo of a side effect of
    sewing, and whether that undo restores the geometry or corrupts it depends
    on the FILE -- see the two measurements below. A consumer reads a rung as
    "set each of these to 1"
    (``hornlab-fusion-addin/scripts/prepare_step_for_wg_metal.py`` does exactly
    that, immediately after setting this option to 0 on purpose), so an option
    travelling inside a rung silently overrides a decision made by the only
    party that has seen the file. Ask
    :func:`hornlab_mesher.step_text.occ_make_solids_is_safe` instead.
    """
    for mode, options in OCC_HEALING_FALLBACKS:
        assert "Geometry.OCCMakeSolids" not in options, mode


# Every gmsh healing option these tests touch: the union of the rungs plus the
# one deliberately absent from them. Each is set explicitly on every probe and
# reset afterwards, because a gmsh option outlives gmsh.clear() and a leaked
# flag would follow this file into the next one.
_HEALING_OPTIONS = sorted(
    {name for _mode, names in OCC_HEALING_FALLBACKS for name in names}
    | {"Geometry.OCCMakeSolids"}
)
_SEW = "Geometry.OCCSewFaces"
_MAKE_SOLIDS = "Geometry.OCCMakeSolids"


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
        for name in _HEALING_OPTIONS:
            gmsh.option.setNumber(name, 0)  # 0 is the gmsh default for each
        gmsh.clear()
        if initialized_here:
            gmsh.finalize()


def _import_through(step_path: Path, enabled: tuple[str, ...]):
    """Return (volume count, surface count, mass of the first volume or None)."""
    with _gmsh_session() as gmsh:
        for name in _HEALING_OPTIONS:
            gmsh.option.setNumber(name, 1 if name in enabled else 0)
        gmsh.open(str(step_path))
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        surfaces = gmsh.model.getEntities(2)
        mass = gmsh.model.occ.getMass(3, volumes[0][1]) if volumes else None
        return len(volumes), len(surfaces), mass


def test_sewing_dissolves_a_plain_solid_and_rebuilding_restores_its_mass(tmp_path):
    """The case where rebuilding really is a restoration, measured as mass.

    gmsh 4.15.2, a 10x20x30 box written through the OCC writer: one volume
    unhealed, none after sewing, and one volume of the SAME mass once
    MakeSolids is set. The surface count never moves, which is why the
    dissolution went unnoticed -- and why an entity count is not the assertion
    this family of gates needs.
    """
    step_path = tmp_path / "plain-solid.step"
    with _gmsh_session() as gmsh:
        gmsh.model.add("plain-solid")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, 10.0, 20.0, 30.0)
        gmsh.model.occ.synchronize()
        gmsh.write(str(step_path))

    text = step_path.read_text(encoding="ascii", errors="replace")
    assert "MANIFOLD_SOLID_BREP" in text
    assert occ_make_solids_is_safe(text) is True

    assert _import_through(step_path, ()) == (1, 6, pytest.approx(6_000.0, rel=1e-9))
    assert _import_through(step_path, (_SEW,)) == (0, 6, None)
    assert _import_through(step_path, (_SEW, _MAKE_SOLIDS)) == (
        1,
        6,
        pytest.approx(6_000.0, rel=1e-9),
    )


def test_neither_healing_setting_is_safe_for_a_body_with_interior_voids(tmp_path):
    """The measurement no counting assertion in this family can make.

    A 40 mm box minus a fully enclosed r=10 sphere, written through the OCC
    writer, which emits it as BREP_WITH_VOIDS and not MANIFOLD_SOLID_BREP.
    gmsh 4.15.2::

        unhealed          volumes=1  surfaces=7  mass=59811.21
        sew only          volumes=0  surfaces=7  mass=None
        sew + MakeSolids  volumes=1  surfaces=6  mass=64000.00

    64000 is the solid box: MakeSolids FILLS the cavity and deletes its inner
    shell. Both volume counts read 1, so every entity-counting gate in this
    family passes the corrupted body -- and the dropped surface leaves the
    healed import unanchorable against the unhealed reference besides.

    So neither setting repairs this body: without the option it dissolves
    loudly, with it the geometry silently becomes a different shape and
    reaches the solve. Pinning both halves is what keeps the option out of a
    rung.
    """
    step_path = tmp_path / "voided-solid.step"
    with _gmsh_session() as gmsh:
        gmsh.model.add("voided-solid")
        box = gmsh.model.occ.addBox(-20.0, -20.0, -20.0, 40.0, 40.0, 40.0)
        sphere = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 10.0)
        gmsh.model.occ.cut([(3, box)], [(3, sphere)])
        gmsh.model.occ.synchronize()
        gmsh.write(str(step_path))

    text = step_path.read_text(encoding="ascii", errors="replace")
    assert "BREP_WITH_VOIDS" in text
    assert occ_make_solids_is_safe(text) is False

    hollow_mass = 40.0**3 - 4.0 / 3.0 * math.pi * 10.0**3
    unhealed = _import_through(step_path, ())
    assert unhealed == (1, 7, pytest.approx(hollow_mass, rel=1e-6))

    # Sewing alone leaves no volume at all, so there is no mass to compare
    # against: the loud failure.
    assert _import_through(step_path, (_SEW,)) == (0, 7, None)

    # Rebuilding returns a volume of the WRONG mass -- the solid box. This is
    # the assertion the gate family was missing.
    filled_volumes, filled_surfaces, filled_mass = _import_through(
        step_path, (_SEW, _MAKE_SOLIDS)
    )
    assert (filled_volumes, filled_surfaces) == (1, 6)
    assert filled_mass == pytest.approx(40.0**3, rel=1e-9)
    assert filled_mass != pytest.approx(unhealed[2], rel=1e-3)


# A quarter box on the +x/+y side, lifted clear of z=0 so that no rim edge
# lies on two coordinate planes at once except the x0/y0 corner.
_QUARTER_BOX_VERTICES = [
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
    (0.0, 0.0, 2.0),
    (1.0, 0.0, 2.0),
    (1.0, 1.0, 2.0),
    (0.0, 1.0, 2.0),
]
_QUARTER_BOX_FACES = {
    "zlow": [(0, 1, 2), (0, 2, 3)],
    "zhigh": [(4, 5, 6), (4, 6, 7)],
    "y0": [(0, 1, 5), (0, 5, 4)],
    "y1": [(3, 2, 6), (3, 6, 7)],
    "x0": [(0, 3, 7), (0, 7, 4)],
    "x1": [(1, 2, 6), (1, 6, 5)],
}


def _quarter_box_mesh(*open_faces: str):
    triangles = [
        triangle
        for name, face in _QUARTER_BOX_FACES.items()
        if name not in open_faces
        for triangle in face
    ]
    return (
        np.asarray(_QUARTER_BOX_VERTICES, dtype=float),
        np.asarray(triangles, dtype=np.int64),
    )


def test_detect_symmetry_planes_reads_the_cut_back_from_free_edges():
    points, triangles = _quarter_box_mesh("x0", "y0")

    planes, detection = detect_symmetry_planes(points, triangles, tolerance=1.0e-9)

    assert planes == ("x0", "y0")
    assert detection["detected_planes"] == ["x0", "y0"]
    assert detection["plane_free_edge_counts"]["z0"] == 0


def test_detect_symmetry_planes_does_not_report_a_capped_cut_plane():
    """A capped plane is the failure the caller cannot see any other way.

    The geometry was reduced on y0, but the reduced boundary came back closed
    there, so the solver would mirror a rigid baffle instead of a cut.
    """

    points, triangles = _quarter_box_mesh("x0")

    planes, detection = detect_symmetry_planes(points, triangles, tolerance=1.0e-9)

    assert planes == ("x0",)
    assert detection["plane_free_edge_counts"]["y0"] == 0


from hornlab_mesher.step_import import (  # noqa: E402
    _decode_step_string,
    _first_step_string,
    _step_records,
)


def test_a_semicolon_inside_a_label_does_not_truncate_the_record():
    records = _step_records("#1 = STYLED_ITEM('woofer; left',#2,#3);\n")
    assert records[1] == "STYLED_ITEM('woofer; left',#2,#3)"
    assert _first_step_string(records[1]) == "woofer; left"


def test_doubled_quotes_still_decode_to_one_quote():
    assert _first_step_string("X('it''s')") == "it's"


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        (r"M\X2\00E5\X0\ler", "Måler"),
        (r"caf\X\E9", "café"),
        ("\\X4\\0001F600\\X0\\", "\U0001f600"),
    ],
)
def test_iso_10303_escapes_decode(encoded, expected):
    assert _decode_step_string(encoded) == expected


def test_a_malformed_escape_is_left_intact_rather_than_raising():
    malformed = "\\X2\\ZZZZ\\X0\\"
    assert _decode_step_string(malformed) == malformed
