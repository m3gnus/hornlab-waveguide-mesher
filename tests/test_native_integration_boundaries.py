"""Combined native features validate authored intent before terminal rewrites."""
import copy

import pytest

from hornlab_mesher.config_builder import build_geometry_params, build_from_config, resolve_geometry
from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.preview import build_preview_geometry, PreviewOptionsV1
from hornlab_mesher.preview.dimensions import canonical_dimensions
from hornlab_mesher.cad import write_step_from_config


def native_config(marker):
    c={"formula":marker,"mode":"bare","profile":{"L_mm":120,"r0_mm":12.7,
        "a_deg":40,"a0_deg":10,"k":1,"s":0},"source":{"source_shape":0},"mesh":{}}
    if marker=="OSSE-AXIAL":c["axial_scale"]=1.2
    elif marker=="OSSE-ROUNDOVER":
        c["mode"]="freestanding"
        c["mesh"].update(wall_thickness_mm=5,mouth_roundover_radius_mm=25)
    else:
        c["throat_adapter"]={"mode":"authored","contract_revision":1,
            "driver_exit_diameter_mm":25.4,"exit_half_angle_deg":10,"length_mm":40,
            "join_t":.137,"driver_handle_mm":10,"body_handle_mm":15}
    return c


def invoke(entry,c,path):
    if entry=="params":return build_geometry_params(c)
    if entry=="resolve":return resolve_geometry(c)
    if entry=="dimensions":return canonical_dimensions(c)
    if entry=="preview":return build_preview_geometry(c,PreviewOptionsV1(lod="coarse"))
    return write_step_from_config(c,path,full_model=entry=="cad-full")


@pytest.mark.parametrize("marker",["OSSE-ADAPTER","OSSE-ROUNDOVER","OSSE-AXIAL"])
@pytest.mark.parametrize("entry",["params","resolve","dimensions","preview","cad-full","cad-raw"])
@pytest.mark.parametrize("patch",[
    {"quadrants":"1234garbage"},
    {"vertical_offset_mm":7,"mesh":{"vertical_offset_mm":8}},
    {"mesh":{"scale_to_metres":"false"}},
    {"mesh":{"allow_large_mesh":True,"allowLargeMesh":False}},
    {"mesh":{"angular_segments":32,"angularSegments":64}},
    {"profile":{"type":"OSSE"}},
])
def test_raw_native_intent_refuses_before_rewrites(marker,entry,patch,tmp_path):
    c=native_config(marker)
    for key,value in patch.items():
        if isinstance(value,dict):c.setdefault(key,{}).update(value)
        else:c[key]=value
    before=copy.deepcopy(c)
    output=tmp_path/"preserved.step";output.write_bytes(b"owned output")
    with pytest.raises(ConfigError):invoke(entry,c,output)
    assert c==before
    assert output.read_bytes()==b"owned output"


@pytest.mark.parametrize("marker",["OSSE-ADAPTER","OSSE-ROUNDOVER","OSSE-AXIAL"])
@pytest.mark.parametrize("override",[0,1,"true",[],{}])
@pytest.mark.parametrize("entry",["resolve","build"])
def test_api_override_refuses_before_native_early_return(marker,override,entry,tmp_path):
    c=native_config(marker);before=copy.deepcopy(c)
    with pytest.raises(ConfigError):
        if entry=="resolve":resolve_geometry(c,allow_large_mesh=override)
        else:build_from_config(c,tmp_path/"refused.msh",allow_large_mesh=override)
    assert c==before
    assert not (tmp_path/"refused.msh").exists()


@pytest.mark.parametrize("key",["formula","type"])
@pytest.mark.parametrize("axis",[False,True])
@pytest.mark.parametrize("entry",["params","resolve","dimensions","preview","cad-full","cad-raw"])
def test_hidden_parameters_axis_marker_refuses_without_legacy_fallback(key,axis,entry,tmp_path):
    c=native_config("OSSE-AXIAL")
    c.pop("formula");c["parameters"]=c.pop("profile");c["parameters"][key]="OSSE-AXIAL"
    if not axis:c.pop("axial_scale")
    output=tmp_path/"refused.step"
    with pytest.raises(ConfigError):invoke(entry,c,output)
    assert not output.exists()
