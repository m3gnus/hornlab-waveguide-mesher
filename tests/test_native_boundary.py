"""Native controls refuse coercion before legacy precedence and rewriting."""
import copy

import pytest

from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.native_boundary import validate_native_boundary


MARKERS=("OSSE-AXIAL","OSSE-ADAPTER","OSSE-ROUNDOVER")


@pytest.mark.parametrize("marker",MARKERS)
@pytest.mark.parametrize("key",["allow_large_mesh","allowLargeMesh","scale_to_metres","scaleToMetres"])
@pytest.mark.parametrize("value",[0,1,"false",None])
def test_native_boolean_controls_require_actual_booleans(marker,key,value):
    with pytest.raises(ConfigError,match="boolean"):
        validate_native_boundary({"formula":marker,"mesh":{key:value}})


@pytest.mark.parametrize("marker",MARKERS)
@pytest.mark.parametrize("value",[1,"1234garbage",True,1234.0,"1234 "])
def test_native_coverage_requires_exact_full_circle(marker,value):
    with pytest.raises(ConfigError,match="quadrants"):
        validate_native_boundary({"formula":marker,"quadrants":value,"mesh":{"quadrants":"1234"}})


@pytest.mark.parametrize("marker",MARKERS)
@pytest.mark.parametrize("mesh",[
    {"allow_large_mesh":True,"allowLargeMesh":False},
    {"scale_to_metres":True,"scaleToMetres":False},
    {"vertical_offset_mm":7,"verticalOffset":8},
    {"angular_segments":64,"angularSegments":32},
])
def test_native_duplicate_aliases_must_agree(marker,mesh):
    with pytest.raises(ConfigError,match="agree"):
        validate_native_boundary({"formula":marker,"mesh":mesh})


@pytest.mark.parametrize("marker",MARKERS)
@pytest.mark.parametrize("key,value",[("angularSegments",0),("length_segments",1.5),("maxTriangles",True),
    ("verticalOffset",float("nan")),("vertical_offset_mm","7")])
def test_native_numeric_coercions_refuse(marker,key,value):
    with pytest.raises(ConfigError):
        validate_native_boundary({"formula":marker,"mesh":{key:value}})


@pytest.mark.parametrize("marker",MARKERS)
def test_native_all_formula_aliases_and_placement_are_checked(marker):
    with pytest.raises(ConfigError,match="formula"):
        validate_native_boundary({"formula":marker,"profile":{"type":"ICW"}})
    with pytest.raises(ConfigError,match="agree"):
        validate_native_boundary({"formula":marker,"verticalOffset":7,"mesh":{"vertical_offset_mm":8}})
    with pytest.raises(ConfigError,match="boolean"):
        validate_native_boundary({"formula":marker},allow_large_mesh=1)


@pytest.mark.parametrize("marker",MARKERS)
def test_valid_native_boundary_preserves_input(marker):
    c={"formula":marker,"profile":{"type":marker.lower()},"quadrants":1234,"verticalOffset":7,
        "mesh":{"quadrants":"1234","vertical_offset_mm":7,"verticalOffset":7,
                "allow_large_mesh":False,"allowLargeMesh":False,"scale_to_metres":True,"scaleToMetres":True,
                "angularSegments":64,"length_segments":32,"maxTriangles":100000}}
    before=copy.deepcopy(c)
    assert validate_native_boundary(c,allow_large_mesh=True)==marker
    assert c==before


def test_ordinary_boundary_remains_delegated_without_new_validation():
    c={"formula":"OSSE","quadrants":"1234garbage","mesh":{"scale_to_metres":"false","length_segments":1.5}}
    before=copy.deepcopy(c)
    assert validate_native_boundary(c,allow_large_mesh=1) is None
    assert c==before


@pytest.mark.parametrize("marker",MARKERS)
def test_whitespace_markers_cannot_bypass_boolean_or_coverage_guards(marker):
    for controls in ({"scaleToMetres":"false"},{"quadrants":"1234garbage"}):
        with pytest.raises(ConfigError):
            validate_native_boundary({"formula":"  "+marker.lower()+" ","mesh":controls})


@pytest.mark.parametrize("marker",MARKERS)
def test_nonstring_marker_discriminant_cannot_bypass_native_guards(marker):
    class StringifiesToMarker:
        def __str__(self):return " "+marker+" "
    with pytest.raises(ConfigError,match="formula"):
        validate_native_boundary({"formula":StringifiesToMarker(),"mesh":{"scaleToMetres":"false"}})
