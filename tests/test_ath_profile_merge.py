"""Repeated ATH declarations retain members and replace later assignments.

WG's retained ath_integrity/repeated.cfg and ATH-written merged.cfg witness
L=160 -> 80 and a=40 -> 30 while a0/k/s/n/q survive the second declaration.
"""

import pytest

from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.config_parser import ConfigError, parse_text_config
from hornlab_mesher.profile_sampling import build_point_grid_arrays


@pytest.fixture(params=["OSSE", "R-OSSE", "ROSSE"])
def profile_configs(request):
    family = request.param
    dimension = "L" if family == "OSSE" else "R"
    extra = "s = 0.7\n n = 4\n q = 0.996" if family == "OSSE" else "m = 0.8\n b = 0.2\n r = 0.4\n q = 3.4"
    original = f"""{family} = {{
 {dimension} = 160
 r0 = 10
 a0 = 7.9
 k = 1.38
 a = 40
 {extra}
}}
"""
    # Three populated copies, plus an empty copy that must not reset anything.
    repeated = (original + f"{family} = {{\n {dimension} = 80\n a = 30+5*sin(p)^2\n}}\n"
                + f"{family} = {{\n r0 = 11\n}}\n{family} = {{\n}}\n")
    merged = original.replace(f"{dimension} = 160", f"{dimension} = 80").replace(
        "a = 40", "a = 30+5*sin(p)^2"
    ).replace("r0 = 10", "r0 = 11")
    sampling = "ABEC.SimType = 2\nMesh.AngularSegments = 8\nMesh.LengthSegments = 20\nMesh.WallThickness = 2\n"
    return [parse_text_config(text + sampling) for text in (repeated, merged)]


def test_profile_members_merge_in_source_order_with_identical_geometry(profile_configs):
    actual, expected = profile_configs
    assert actual == expected
    grids = [build_point_grid_arrays(build_geometry_params(config)[0]) for config in (actual, expected)]
    for key in ("inner_grid", "outer_grid"):
        assert grids[0][key].shape == grids[1][key].shape
        assert grids[0][key].tobytes() == grids[1][key].tobytes()


def test_merged_profile_reaches_preview_and_mesh_with_identical_bytes(profile_configs, tmp_path):
    from hornlab_mesher.config_builder import build_from_config
    from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry

    previews, meshes = [], []
    for config in profile_configs:
        preview = build_preview_geometry(config, PreviewOptionsV1(lod="coarse"))
        previews.append([(surface.role, surface.positions.tobytes(), surface.indices.tobytes())
                         for surface in preview.surfaces])
        built = build_from_config(config, tmp_path / "merged.msh")
        meshes.append(built.mesh_path.read_bytes())
    assert previews[0] == previews[1]
    assert meshes[0] == meshes[1]


@pytest.mark.parametrize("family", ["OSSE", "R-OSSE", "ROSSE"])
def test_repeated_block_rotation_stays_ignored(family):
    dimension = "L" if family == "OSSE" else "R"
    plain = f"{family} = {{\n {dimension} = 120\n r0 = 10\n a = 40\n}}\nRot = 10\n"
    repeated = plain + f"{family} = {{\n Rot = 30\n}}\n{family} = {{\n Rot = 0\n}}\n"
    assert parse_text_config(repeated) == parse_text_config(plain)


def test_merging_does_not_hide_unsupported_earlier_members():
    text = "OSSE = {\n L = 120\n Bogus = 2\n}\nOSSE = {\n a = 30\n}\n"
    with pytest.raises(ConfigError, match="OSSE.Bogus"):
        parse_text_config(text)


def test_different_profile_names_are_still_refused():
    text = "R-OSSE = {\n R = 150\n}\nROSSE = {\n a = 30\n}\n"
    with pytest.raises(ConfigError, match="second profile"):
        parse_text_config(text)
