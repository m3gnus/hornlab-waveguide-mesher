"""ATH controls use the native exact cubic/conic authority."""
import json
from pathlib import Path

import numpy as np
import pytest

from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.config_parser import ConfigError, parse_text_config

TEXT = """OSSE = {
L = 120
r0 = 18
a = 40
a0 = 8
k = 1
s = 0
}
Throat.Ext.Ctrl = -38,17.78,-27.4,18.6,-12.8,21.441923768883
Source.Shape = 2
ABEC.SimType = 2
Mesh.WallThickness = 0
Mesh.Quadrants = 1234
Mesh.AngularSegments = 16
Mesh.LengthSegments = 32
"""


def test_import_uses_native_authority_and_matches_archived_ath_cubic_and_body():
    config = parse_text_config(TEXT)
    assert config["formula"] == "OSSE-ADAPTER-CONTROLS"
    model = resolve_geometry(config).geometry.controls_meridian
    root = Path(__file__).parent / "fixtures/throat_adapter/ath-v2025-12"
    case = json.loads((root / "a9-osse-N512-omitted.json").read_text())
    poles = np.asarray(case["adapter"]["control_points_join_local_mm"])
    poles[:, 0] += 38
    np.testing.assert_array_equal(model.cubic_poles, poles)
    for block in (root / "a9-osse-N512-omitted.csv").read_text().strip().split("\n\n"):
        points = np.asarray([[float(v) for v in row.split(";")] for row in block.splitlines()])
        meridian = np.stack((points[:, 2], np.hypot(points[:, 0], points[:, 1])), axis=-1)
        np.testing.assert_allclose(model.cubic(np.linspace(0, 1, 11))[0], meridian[:11], atol=1e-6, rtol=0)
        t = (meridian[10:, 0] - 38) / 120
        # Projection and inversion each use six-decimal exported coordinates.
        np.testing.assert_allclose(model.body(np.clip(t, 0, 1))[0], meridian[10:], atol=2e-6, rtol=0)
    assert abs(model.join_jump_deg) > 1  # The supplied corner is retained.


def test_import_builds_preview_mesh_and_step_with_one_construction(tmp_path):
    from hornlab_mesher.cad import write_step_from_config
    from hornlab_mesher.config_builder import build_from_config
    from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry

    config = parse_text_config(TEXT + "Scale = .005\n")
    model = resolve_geometry(config).geometry.controls_meridian
    preview = build_preview_geometry(config, PreviewOptionsV1(lod="coarse"))
    wall = next(s for s in preview.surfaces if s.role == "horn.inner")
    assert len(wall.metadata["branchRanges"]) == 2
    result = build_from_config(config, tmp_path / "imported.msh")
    assert result.metadata["controlsMeshChordBoundMm"] <= .01
    assert result.metadata["construction_fingerprint"] == model.fingerprint
    _, info = write_step_from_config(config, tmp_path / "imported.step", open_throat=False)
    assert info.body == "surface" and info.n_faces == 9


@pytest.mark.parametrize("change", [
    ("OSSE =", "R-OSSE ="), ("s = 0", "s = .5"),
    ("a = 40", "a = 40+sin(p)"), ("Source.Shape = 2", "Source.Shape = 1"),
    ("ABEC.SimType = 2", "ABEC.SimType = 1"),
    ("Mesh.WallThickness = 0", "Mesh.WallThickness = 2"),
    ("Mesh.Quadrants = 1234", "Mesh.Quadrants = 12"),
    ("-38,17.78,-27.4,18.6,-12.8,21.441923768883", "-38,17.78,-27.4,18.6,-12.8"),
    ("17.78", "17.78+1"), ("17.78", "nan"),
])
def test_unqualified_or_malformed_import_refuses(change):
    with pytest.raises(ConfigError):
        parse_text_config(TEXT.replace(*change))


def test_malformed_coordinates_report_controls_guidance_in_public_refusal():
    from hornlab_mesher.errors import UserRefusal

    with pytest.raises(UserRefusal) as caught:
        parse_text_config(TEXT.replace("17.78", "nan"))
    assert "Throat.Ext.Ctrl" in caught.value.user_message
    assert "six finite coordinates" in caught.value.user_message
    assert "block" not in caught.value.user_message.lower()
    assert "import" in caught.value.next_step.lower()


@pytest.mark.parametrize("setting", [
    "Throat.Ext.Length = 2", "Slot.Length = 2", "Rot = 2", "Scale.Z = 2",
    "Mesh.ThroatExtSegments = 8", "Morph.TargetShape = 1", "GCurve.Type = 1",
])
def test_unqualified_compositions_refuse(setting):
    with pytest.raises(ConfigError):
        parse_text_config(TEXT + setting + "\n")


@pytest.mark.parametrize("version", ["V2025-06", "V2025-07", "V2025-11"])
def test_unqualified_older_ath_version_refuses(version):
    with pytest.raises(ConfigError, match="pre-December-2025"):
        parse_text_config(f"; Ath version {version}\n" + TEXT)
