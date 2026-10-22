"""ATH LFSource import against the measured ten-case corpus (3a oracle)."""
import re

import pytest
from hornlab_mesher import build_from_config
from hornlab_mesher.config_parser import parse_text_config, ath_enclosure_sources, ConfigError
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


@pytest.mark.parametrize("sides", ["Dim = 250,400,270,450", "Spacing = 40,40,40,220"])
def test_enclosure_without_lf_or_depth_uses_measured_default(sides):
    text = (FIXTURES/"ath_cfg/b-dim4.cfg").read_text()
    text = re.sub(r"LFSource.B = \{.*?\}\n", "", text, flags=re.S)
    for line in ("Depth = 200", "Spacing = 40,40,40,220", "Dim = 250,400,270,450"):
        text = text.replace(line + "\n", "")
    text = text.replace("Mesh.Enclosure = {", "Mesh.Enclosure = {\n" + sides)
    config = parse_text_config(text)
    enclosure = resolve_geometry(config).geometry.enclosure
    assert enclosure is not None
    assert enclosure.depth_mm == pytest.approx(220)
    assert not enclosure.woofers
    if sides.startswith("Dim"):
        assert enclosure.dim_mm == (250, 400, 270, 450)


def test_sid_zero_imports_unassigned_with_note(caplog):
    text = (FIXTURES/"ath_cfg/b-dim4.cfg").read_text().replace("SID = 1", "SID = 0")
    config = parse_text_config(text)
    assert len(config["enclosure"]["woofers"]) == 1
    rows, notes, override = ath_enclosure_sources({"LFSource.B.Radius": "50", "LFSource.B.SID": "0"})
    assert "sid" not in rows[0]["ath"]
    assert any("SID 0 is reserved" in note for _, _, note in notes)
    assert override is None
    assert "SID 0 is reserved for the waveguide; LF SID is left unset" in caplog.text


@pytest.mark.parametrize("reverse_blocks", [False, True])
def test_mtm_sids_keeps_first_ath_source(reverse_blocks, caplog):
    text = (FIXTURES/"ath_cfg/mtm-sids.cfg").read_text()
    if reverse_blocks:
        text = text.replace("LFSource.A", "LFSource.TEMP").replace("LFSource.B", "LFSource.A").replace("LFSource.TEMP", "LFSource.B")
    # Mapping/declaration order must not override ATH's A-before-B order.
    rows, notes, _ = ath_enclosure_sources({
        "LFSource.B.Radius": "50", "LFSource.B.SID": "1",
        "LFSource.A.Radius": "50", "LFSource.A.SID": "1",
    })
    assert [row["ath"] for row in rows] == [{"key": "A", "sid": 1}, {"key": "B"}]
    note, = [note for note in notes if note[0].endswith(".SID")]
    assert note == ("Mesh.Enclosure.LFSource.B.SID", "1",
                    "SID 1 already selects LFSource.A (ATH's first match); LFSource.B SID is left unset.")
    config = parse_text_config(text)
    assert len(config["enclosure"]["woofers"]) == 2
    assert len(resolve_geometry(config).geometry.enclosure.woofers) == 2
    assert note[2] in caplog.text


@pytest.mark.parametrize("sid", ["-1", "1.5", "nan", "inf", "bad"])
def test_invalid_sid_has_named_refusal(sid):
    text = (FIXTURES/"ath_cfg/mtm-sids.cfg").read_text().replace("SID = 1", f"SID = {sid}", 1)
    with pytest.raises(ConfigError, match=r"^LFSource\.A\.SID must be (finite|a nonnegative integer)"):
        parse_text_config(text)


@pytest.mark.parametrize("q", [1, 12, "1.0"])
def test_quadrants_keep_independent_sources(q, caplog):
    text = (FIXTURES/"ath_cfg/mtm-q14.cfg").read_text().replace("Mesh.Quadrants = 14", f"Mesh.Quadrants = {q}")
    config = parse_text_config(text)
    assert config["mesh"]["quadrants"] == 1234
    assert len(config["enclosure"]["woofers"]) == 2
    assert "ATH drops B and mirrors A" in caplog.text
