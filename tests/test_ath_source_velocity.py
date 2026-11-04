"""Source motion is solver metadata, independent of the meshed cap or disc."""

import pytest

from hornlab_mesher.config_builder import build_from_config
from hornlab_mesher.config_parser import parse_text_config
from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry


@pytest.mark.parametrize("shape", [1, 2])
def test_axial_velocity_keeps_preview_and_mesh_bytes(shape, tmp_path):
    text = f"""OSSE = {{
L = 60
r0 = 12.7
a = 45
a0 = 7
}}
ABEC.SimType = 2
Source.Shape = {shape}
Source.Radius = 30
Source.Curv = -1
Mesh.AngularSegments = 16
Mesh.LengthSegments = 6
Mesh.ThroatResolution = 12
Mesh.MouthResolution = 30
"""
    normal = parse_text_config(text + "Source.Velocity = 1\n")
    axial = parse_text_config(text + "Source.Velocity = 2\n")
    assert axial["source"]["sourceVelocity"] == 2
    options = PreviewOptionsV1(lod="coarse")
    first = build_preview_geometry(normal, options)
    second = build_preview_geometry(axial, options)
    assert len(first.surfaces) == len(second.surfaces)
    for left, right in zip(first.surfaces, second.surfaces):
        assert left.role == right.role
        assert left.positions.tobytes() == right.positions.tobytes()
        assert left.indices.tobytes() == right.indices.tobytes()
        assert left.normals.tobytes() == right.normals.tobytes()
    before = build_from_config(normal, tmp_path / "normal.msh")
    after = build_from_config(axial, tmp_path / "axial.msh")
    assert before.mesh_path.read_bytes() == after.mesh_path.read_bytes()
