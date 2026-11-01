"""Only the new v3 refusal boundary; no geometry predicate is changed."""
import math
import pickle
import pytest
from hornlab_mesher.config_parser import parse_text_config
from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.errors import UserRefusal
from hornlab_mesher.text_import import TEXT_IMPORT_VERSION_KEY

BASE = 'ABEC.SimType = 2\nOSSE = {\nL = 100\nr0 = 25\na0 = 10\na = 45\n}\n'

@pytest.mark.parametrize('text,key', [
    (BASE + 'Scale.Z = -1\n', 'Scale.Z'),
    (BASE + 'Mesh.ThroatExtSegments = 0\n', 'Mesh.ThroatExtSegments'),
    ('; Waveguide Generator geometry-interpretation: ath-2026-09a-v2\n' + BASE + 'Mesh.ThroatExtSegments = 4\n', 'Mesh.ThroatExtSegments'),
    ('; Ath version V2025-06\n' + BASE + 'Mesh.ThroatExtSegments = 4\n', 'Mesh.ThroatExtSegments'),
    (BASE + 'Scale.Z = 2\nMesh.Roundover = {\nRadius = 20\nSegments = 6\n}\n', 'Scale.Z'),
])
def test_new_ath_parser_refusals_are_typed_and_picklable(text, key):
    with pytest.raises(UserRefusal) as caught:
        parsed = parse_text_config(text)
        build_geometry_params(parsed)
    refusal = caught.value
    assert key in str(refusal)
    assert f'“{key}”' in refusal.user_message
    assert refusal.next_step
    restored = pickle.loads(pickle.dumps(refusal))
    assert (type(restored), str(restored), restored.field, restored.user_message, restored.next_step) == (
        type(refusal), str(refusal), refusal.field, refusal.user_message, refusal.next_step)

@pytest.mark.parametrize('extra,key', [
    ({TEXT_IMPORT_VERSION_KEY: 'ath-2026-09a-v2', 'athAxialScale': 2}, 'Scale.Z'),
    ({'athAxialScale': -1}, 'Scale.Z'),
    ({TEXT_IMPORT_VERSION_KEY: 'ath-2026-09a-v2', 'mesh': {'athThroatExtSegments': 4}}, 'Mesh.ThroatExtSegments'),
    ({'mesh': {'athThroatExtSegments': 0}}, 'Mesh.ThroatExtSegments'),
])
def test_new_dictionary_refusals_keep_the_same_gate(extra, key):
    cfg = {**parse_text_config(BASE), **extra}
    with pytest.raises(UserRefusal) as caught:
        build_geometry_params(cfg)
    assert key in str(caught.value)
    assert f'“{key}”' in caught.value.user_message
    assert caught.value.next_step

def test_direct_lip_composition_uses_the_same_refusal_owner():
    from hornlab_mesher.ath_import_surfaces import validate_combinations
    with pytest.raises(UserRefusal) as caught:
        validate_combinations({TEXT_IMPORT_VERSION_KEY: 'ath-2026-09a-v3', 'athAxialScale': 2})
    assert '“Scale.Z”' in caught.value.user_message
    assert caught.value.next_step

def test_direct_rotation_refusal_is_user_affectable():
    from hornlab_mesher.profile_formulas import _rotate_meridian
    with pytest.raises(UserRefusal) as caught:
        _rotate_meridian(1., 2., math.nan, 0., 1.)
    assert str(caught.value) == 'Rot must evaluate to a finite angle'
    assert caught.value.next_step


def test_new_rosse_rotation_builder_refusal_is_typed():
    cfg = {**parse_text_config(BASE.replace('OSSE =', 'R-OSSE =').replace('L = 100', 'R = 200')), 'rot': math.nan}
    with pytest.raises(UserRefusal) as caught:
        build_geometry_params(cfg)
    assert str(caught.value) == 'Rot must be finite'
    assert caught.value.next_step
