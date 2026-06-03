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
    assert params["samplingMode"] == "ath-default-zmap"


def test_load_config_accepts_ath_txt_extension(tmp_path):
    txt_path = tmp_path / "osse-simple.txt"
    txt_path.write_text(OSSE_CFG, encoding="utf-8")

    config = load_config(txt_path)
    params, formula, mode = build_geometry_params(config)

    assert formula == "OSSE"
    assert mode == "freestanding"
    assert "cos(p)" in params["a"]
    assert params["morphTarget"] == 1


def test_parse_ath_config_preserves_gcurve_metadata(tmp_path):
    cfg_path = tmp_path / "gcurve.cfg"
    cfg_path.write_text(
        """
OSSE = {
  Length = 80
}
GCurve.Type = 2
GCurve.Width = 75
GCurve.SF = 1,1,8,0.6,5,2
Mesh.LengthSegments = 4
Mesh.SubdomainSlices = 2,4
Mesh.InterfaceOffset = 10
Mesh.InterfaceResolution = 12
""",
        encoding="utf-8",
    )

    config = load_config(cfg_path)
    params, formula, _mode = build_geometry_params(config)

    assert formula == "OSSE"
    assert params["gcurveType"] == 2
    assert params["gcurveWidth"] == 75
    assert params["gcurveSF"] == "1,1,8,0.6,5,2"
    assert params["subdomainSlices"] == "2,4"
    assert params["interfaceOffset"] == 10
    assert params["interfaceResolution"] == 12


def test_parse_ath_config_preserves_interface_arrays(tmp_path):
    cfg_path = tmp_path / "interfaces.cfg"
    cfg_path.write_text(
        """
OSSE = {
  Length = 80
}
Mesh.SubdomainSlices = 2,4
Mesh.InterfaceOffset = 6,9
Mesh.InterfaceResolution = 12
""",
        encoding="utf-8",
    )

    config = load_config(cfg_path)
    params, formula, _mode = build_geometry_params(config)

    assert formula == "OSSE"
    assert params["subdomainSlices"] == "2,4"
    assert params["interfaceOffset"] == "6,9"
    assert params["samplingMode"] == "ath-default-zmap"


def test_parse_ath_config_preserves_zmap_points(tmp_path):
    cfg_path = tmp_path / "zmap.cfg"
    cfg_path.write_text(
        """
OSSE = {
  Length = 80
}
Mesh.LengthSegments = 4
Mesh.ZMapPoints = 0.5,0.1,0.75,0.7
""",
        encoding="utf-8",
    )

    config = load_config(cfg_path)
    params, formula, _mode = build_geometry_params(config)

    assert formula == "OSSE"
    assert config["mesh"]["samplingMode"] == "zmap"
    assert config["mesh"]["zMapPoints"] == "0.5,0.1,0.75,0.7"
    assert params["samplingMode"] == "zmap"
    assert params["zMapPoints"] == "0.5,0.1,0.75,0.7"


def test_build_from_config_accepts_ath_morph(tmp_path):
    result = build_from_config(
        {
            "formula": "OSSE",
            "profile": {
                "L": 80.0,
                "r0": 12.7,
                "a": 45.0,
                "a0": 10.0,
            },
            "morph": {
                "morphTarget": 1,
                "morphWidth": 320,
                "morphHeight": 320,
            },
            "mesh": {
                "angularSegments": 12,
                "lengthSegments": 4,
            },
        },
        tmp_path / "morph.msh",
    )

    assert result.n_vertices > 0


def test_build_from_config_accepts_ath_gcurve(tmp_path):
    result = build_from_config(
        {
            "formula": "OSSE",
            "profile": {
                "L": 80.0,
                "r0": 12.7,
                "a": 45.0,
                "a0": 10.0,
            },
            "gcurve": {
                "gcurveType": 2,
                "gcurveWidth": 75,
            },
            "mesh": {
                "angularSegments": 12,
                "lengthSegments": 4,
            },
        },
        tmp_path / "gcurve.msh",
    )

    assert result.n_vertices > 0


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
