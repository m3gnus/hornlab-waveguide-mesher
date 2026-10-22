"""ATH LFSource import against the measured ten-case corpus (3a oracle)."""
import pytest
from hornlab_mesher import build_from_config
from hornlab_mesher.config_parser import parse_text_config, ConfigError
from hornlab_mesher.config_builder import resolve_geometry
from test_enclosure_woofers import FIXTURES, REFERENCE, assert_corpus_apertures


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["case_id"])
def test_imported_corpus(tmp_path, case):
    config = parse_text_config((FIXTURES/"ath_cfg"/(case["case_id"]+".cfg")).read_text())
    config["mesh"].update(scale_to_metres=False, max_triangles=30000)
    geometry = resolve_geometry(config).geometry
    for woofer, expected in zip(geometry.enclosure.woofers, case["woofers"], strict=True):
        assert [woofer.x_mm, woofer.y_mm, geometry.inner_points[:,-1,2].max()] == pytest.approx(expected["aperture_center_mm"], abs=.1)
        assert woofer.radius_mm == pytest.approx(expected["radius_mm"], abs=.1)
    result = build_from_config(config, tmp_path/"corpus.msh")
    assert_corpus_apertures(result, case, radiator_prefix="ath-lf-")


@pytest.mark.parametrize("dim", ["1,2,3", "1,2,3,0", "1,2,3,nan"])
def test_invalid_dim(dim):
    text = (FIXTURES/"ath_cfg/b-dim4.cfg").read_text().replace("250,400,270,450", dim)
    with pytest.raises((ConfigError, ValueError), match="Dim|dim_mm"):
        resolve_geometry(parse_text_config(text))


def test_asymmetric_dim_half_refused():
    text = (FIXTURES/"ath_cfg/b-dim4.cfg").read_text().replace("Mesh.Quadrants = 1234", "Mesh.Quadrants = 14")
    with pytest.raises(ValueError, match="symmetric.*(left/right|enclosure Dim)"):
        resolve_geometry(parse_text_config(text))


def test_symmetric_dim_overrides_asymmetric_unused_spacing():
    text = (FIXTURES/"ath_cfg/b-dim4.cfg").read_text().replace("250,400,270,450", "270,400,270,450").replace("40,40,40,220", "40,40,80,220").replace("Mesh.Quadrants = 1234", "Mesh.Quadrants = 14")
    assert resolve_geometry(parse_text_config(text)).geometry.enclosure.dim_mm == (270,400,270,450)


def test_lf_without_depth_uses_measured_default():
    text = (FIXTURES/"ath_cfg/b-base.cfg").read_text().replace("Depth = 200\n", "")
    config = parse_text_config(text)
    assert config["enclosure"]["depth_mm"] == pytest.approx(220)
    assert len(resolve_geometry(config).geometry.enclosure.woofers) == 1


@pytest.mark.parametrize("q", [1,12,14])
def test_asymmetric_dim_without_lf_refused(q):
    config = parse_text_config((FIXTURES/"ath_cfg/b-dim4.cfg").read_text())
    config["enclosure"]["woofers"] = []
    config["mesh"]["quadrants"] = q
    with pytest.raises(ConfigError, match="symmetric enclosure Dim"):
        resolve_geometry(config)
