from __future__ import annotations

from pathlib import Path
import re

from hornlab_mesher.step_import import (
    RIGID_TAG,
    StepFaceGroup,
    StepLabelSelector,
    map_step_face_groups,
)
from hornlab_mesher.step_prepare import OccSurfaceRole


def test_step_face_mapping_uses_only_caller_supplied_aliases(tmp_path):
    step_path = tmp_path / "model.step"
    step_path.write_text("#10=ADVANCED_FACE('',(),$,.T.);\n", encoding="ascii")
    role = OccSurfaceRole("caller-owned-role")
    group = StepFaceGroup(
        name="requested-label",
        selector=StepLabelSelector("requested-label", ("available-label",)),
        role=role,
        tag=7,
        resolution_mm=3.5,
    )

    result = map_step_face_groups(
        step_path,
        [group],
        gmsh_surfaces=[101],
        named_faces={},
        styled_faces={"available-label": [10]},
        face_order=[10],
    )

    assert group.role is role
    assert result.surfaces == {"requested-label": [101]}
    assert result.origins == {
        "requested-label": (
            "appearance/style (available-label alias for requested-label)"
        )
    }
    assert result.missing_reasons == {}


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
