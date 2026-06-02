from __future__ import annotations

from hornlab_mesher.cli import build_from_config, build_geometry_params, load_config


ROSSE_CFG = """
R-OSSE = {
  R = 160 * (abs(cos(p)/1.8)^3 + abs(sin(p)/1)^4)^(-1/7)
  r = 0.35
  b = 0.4
  m = 0.84
  a = 22
  a0 = 15.5
  r0 = 12.7
  k = 4
  q = 4
}
Mesh = {
  AngularSegments = 80
  LengthSegments = 20
  WallThickness = 6
}
"""

OSSE_CFG = """
OSSE = {
  Length = 80
  Throat.Diameter = 20
  Coverage.Angle = 45 * cos(p)
  Throat.Angle = 10
  OS.k = 1
  Term.n = 4
  Term.q = 0.995
}
MORPH = {
  TargetShape = 1
}
Mesh = {
  AngularSegments = 12
  LengthSegments = 4
  WallThickness = 5
}
"""

OSSE_ENCLOSURE_CFG = """
OSSE = {
  Length = 80
  Throat.Diameter = 20
  Coverage.Angle = 45
  Throat.Angle = 10
  OS.k = 1
  Term.n = 4
  Term.q = 0.995
}
Mesh = {
  AngularSegments = 12
  LengthSegments = 4
  WallThickness = 0
}
Mesh.Enclosure = {
  Depth = 120
  EdgeRadius = 6
}
"""


def test_build_geometry_params_accepts_rosse_alias():
    params, formula, mode = build_geometry_params(
        {
            "formula": "ROSSE",
            "profile": {"R_mm": 130.0, "r0_mm": 12.7},
            "mesh": {"angular_segments": 16, "length_segments": 6},
        }
    )

    assert formula == "R-OSSE"
    assert mode == "freestanding"
    assert params["type"] == "R-OSSE"
    assert params["R"] == 130.0
    assert params["wallThickness"] == 6.0


def test_load_config_accepts_ath_cfg_fixture(tmp_path):
    cfg_path = tmp_path / "rosse-simple.cfg"
    cfg_path.write_text(ROSSE_CFG, encoding="utf-8")

    config = load_config(cfg_path)
    params, formula, mode = build_geometry_params(config)

    assert formula == "R-OSSE"
    assert mode == "freestanding"
    assert params["type"] == "R-OSSE"
    assert "cos(p)" in params["R"]
    assert params["angularSegments"] == 80


def test_load_config_accepts_ath_txt_extension(tmp_path):
    txt_path = tmp_path / "osse-simple.txt"
    txt_path.write_text(OSSE_CFG, encoding="utf-8")

    config = load_config(txt_path)
    params, formula, mode = build_geometry_params(config)

    assert formula == "OSSE"
    assert mode == "freestanding"
    assert "cos(p)" in params["a"]
    assert params["morphTarget"] == 1


def test_build_from_config_osse_freestanding(tmp_path):
    result = build_from_config(
        {
            "formula": "OSSE",
            "mode": "freestanding",
            "profile": {
                "L_mm": 80.0,
                "r0_mm": 10.0,
                "a_deg": 45.0,
                "a0_deg": 10.0,
            },
            "mesh": {
                "angular_segments": 12,
                "length_segments": 4,
                "wall_thickness_mm": 5.0,
                "throat_res_mm": 8.0,
                "mouth_res_mm": 30.0,
                "rear_res_mm": 30.0,
            },
        },
        tmp_path / "osse.msh",
    )

    assert result.formula == "OSSE"
    assert result.mode == "freestanding"
    assert result.n_vertices > 0
    assert result.n_triangles > 0
    assert {1, 2}.issubset(result.physical_groups)


def test_build_from_config_rosse_enclosure(tmp_path):
    result = build_from_config(
        {
            "formula": "ROSSE",
            "profile": {
                "R_mm": 90.0,
                "r0_mm": 10.0,
                "a_deg": 50.0,
                "a0_deg": 12.0,
                "k": 1.0,
                "q": 1.0,
            },
            "mesh": {
                "angular_segments": 16,
                "length_segments": 6,
                "throat_res_mm": 8.0,
                "mouth_res_mm": 35.0,
                "rear_res_mm": 40.0,
            },
            "enclosure": {
                "depth_mm": 120.0,
                "space_l_mm": 20.0,
                "space_t_mm": 20.0,
                "space_r_mm": 20.0,
                "space_b_mm": 20.0,
                "edge_mm": 6.0,
            },
        },
        tmp_path / "rosse-enclosure.msh",
    )

    assert result.formula == "R-OSSE"
    assert result.mode == "enclosure"
    assert result.n_vertices > 0
    assert result.n_triangles > 0
    assert {1, 2}.issubset(result.physical_groups)


def test_build_from_ath_enclosure_fixture(tmp_path):
    cfg_path = tmp_path / "osse-with-enclosure.cfg"
    cfg_path.write_text(OSSE_ENCLOSURE_CFG, encoding="utf-8")

    result = build_from_config(
        load_config(cfg_path),
        tmp_path / "osse-with-enclosure.msh",
    )

    assert result.formula == "OSSE"
    assert result.mode == "enclosure"
    assert result.n_vertices > 0
    assert result.n_triangles > 0
    assert {1, 2}.issubset(result.physical_groups)
