"""Release regressions against identities/imports captured from 5c8ea4dc."""
import copy
from dataclasses import asdict, astuple, fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from hornlab_mesher import config_builder as cb
from hornlab_mesher.builders.osse_waveguide import _osse_params
from hornlab_mesher.builders.rosse_waveguide import _rosse_params
from hornlab_mesher.config_parser import ConfigError, parse_text_config
from hornlab_mesher.geometry import OsseHornGeometry, RosseHornGeometry
from hornlab_mesher.profile_formulas import (
    _icw_cache_key, _stretch_x, _stretch_x_curve,
    calculate_osse, calculate_osse_curve, calculate_rosse, calculate_rosse_curve,
    rosse_axial_layout,
)
from hornlab_mesher.throat_stretch import STRETCH_COEFFICIENT_MAX

BASE = json.loads((Path(__file__).parent / 'fixtures/throat_stretch/base-compatibility.json').read_text())
CLASSES = {'OSSE': (OsseHornGeometry, _osse_params), 'R-OSSE': (RosseHornGeometry, _rosse_params)}
DORMANT = [{}, {'s1': 0, 's2': 0.2}, {'s1': 0.5, 's2': 0}, {'s1': 0}, {'s2': 0.2}]


@pytest.mark.parametrize('name', [k for k in BASE['identities'] if not k.endswith('-config')])
@pytest.mark.parametrize('dormant', DORMANT)
def test_dataclass_legacy_identity_and_serialization(name, dormant):
    captured = BASE['identities'][name]
    cls, adapter = CLASSES[captured['family']]
    geometry = cls(**captured['kwargs'], **dormant)
    legacy = cls(**captured['kwargs'])
    assert geometry == legacy and hash(geometry) == hash(legacy)
    assert asdict(geometry) == captured['asdict']
    assert astuple(geometry) == astuple(legacy)
    assert repr(geometry) == captured['repr']
    assert adapter(geometry) == captured['adapter']
    # None's hash is interpreter-dependent on older supported Python versions.
    # The literal base hashes were captured with Python 3.13 on a 64-bit build.
    if sys.version_info[:2] == (3, 13) and sys.hash_info.width == 64:
        assert hash(geometry) == captured['hash']
    active = replace(geometry, s1=0.5, s2=0.2)
    assert active != geometry and hash(active) != hash(geometry)
    assert asdict(active)['s1'] == 0.5 and adapter(active)['s2'] == 0.2
    assert replace(active, s1=0) == legacy


@pytest.mark.parametrize('family', list(CLASSES))
@pytest.mark.parametrize('dormant', DORMANT)
def test_native_and_text_normalization_equal_captured_base(family, dormant):
    captured = BASE['identities'][family + '-config']
    native = copy.deepcopy(captured['native'])
    native['profile'].update(dormant)
    assert cb.build_geometry_params(native)[0] == captured['native_params']
    text = captured['text'].replace('\n}', ''.join(f'\n{k} = {v}' for k, v in dormant.items()) + '\n}')
    parsed = parse_text_config(text)
    assert parsed == parse_text_config(captured['text'])
    assert cb.build_geometry_params(parsed)[0] == captured['text_params']


@pytest.mark.parametrize('dormant', DORMANT)
def test_dormant_icw_seed_key_matches_base(dormant):
    assert _icw_cache_key({'icw_seed': {'type': 'OSSE', 'L': 160, **dormant}}) == BASE['icw_disabled_key']


@pytest.mark.parametrize('case', BASE['imports'], ids=lambda c: c['id'])
def test_import_corpus_equal_base_except_two_approved_corrections(case):
    if 'text' in case:
        text = case['text']
    else:
        root = os.environ.get('ATH_REFERENCE_ROOT')
        if not root:
            if os.environ.get('HORNLAB_ATH_PARITY') == 'required':
                pytest.fail('required import corpus needs ATH_REFERENCE_ROOT')
            pytest.skip('ATH archive not configured')
        matches = [p for p in Path(root).glob('*/config.txt')
                   if hashlib.sha256(p.read_text().encode()).hexdigest() == case['sha256']]
        assert len(matches) == 1, 'captured archive config missing or changed'
        text = matches[0].read_text()
    # Only these two existing imports may differ. In-block Rot still wins
    # over a top-level Rot when both are supplied, exactly as at the base.
    osse_block = text.startswith('OSSE = {')
    block_text = text.split('}', 1)[0]
    approved_slot = osse_block and 'Slot.Length = 8' in text
    approved_rot = osse_block and 'Rot = 10' in text and 'Rot = 5' not in block_text
    if approved_slot:
        with pytest.raises(ConfigError, match='OSSE block with nonzero Slot.Length'):
            parse_text_config(text)
        return
    if 'error' in case:
        with pytest.raises(ConfigError, match=case['message']):
            parse_text_config(text)
        return
    parsed = parse_text_config(text)
    params = cb.build_geometry_params(parsed)[0]
    expected_parsed, expected_params = copy.deepcopy(case['parsed']), dict(case['params'])
    if approved_rot:
        expected_parsed['profile']['rot'] = 10
        expected_params['rot'] = 10
    assert parsed == expected_parsed
    assert params == expected_params


@pytest.mark.parametrize('prefix', [{'throatExtLength': 20}, {'slotLength': 8}, {'throatExtLength': 20, 'slotLength': 8}])
@pytest.mark.parametrize('angle', [0, -20])
def test_review_discontinuous_and_self_intersecting_native_design_refused(prefix, angle, tmp_path, monkeypatch):
    params = dict(L=160, r0=10, a=40, a0=7.9, k=1.38, n=4, q=0.996, s=0.7,
                  rot=10, s1=0.5, s2=0.2, throatExtAngle=angle, **prefix)
    # The review's exact crossing stations, plus the join and driver station.
    for station in [0, 20, 20 + 1e-9, 20.21105824482577, 19.639715632211413]:
        with pytest.raises(ConfigError, match='Rot.*prefix.*unverified'):
            calculate_osse(station, 0, params)
    with pytest.raises(ConfigError, match='Rot.*prefix.*unverified'):
        calculate_osse_curve(np.array([0, 20, 20 + 1e-9]), 0, params)
    def no_mesh(*args, **kwargs):
        pytest.fail('invalid profile reached mesh construction')
    monkeypatch.setattr(cb, 'build_mesh_with_info', no_mesh)
    config = {'profile': {'formula': 'OSSE', **params}, 'mode': 'bare',
              'mesh': {'angularSegments': 16, 'lengthSegments': 16, 'wallThickness': 0}}
    with pytest.raises(ConfigError, match='Rot.*prefix.*unverified'):
        cb.resolve_geometry(config)
    with pytest.raises(ConfigError, match='Rot.*prefix.*unverified'):
        cb.build_from_config(config, tmp_path / 'broken.msh')
    assert not (tmp_path / 'broken.msh').exists()
    disabled = {**params, 's1': 0}
    calculate_osse_curve(np.array([0, 20, 20 + 1e-9]), 0, disabled)


@pytest.mark.parametrize('family', list(CLASSES))
@pytest.mark.parametrize('prefix', [{'throatExtLength': 20}, {'slotLength': 8}, {'throatExtLength': 20, 'slotLength': 8}])
def test_stretched_composite_junction_is_continuous_and_foldback_preserved(family, prefix):
    if family == 'OSSE':
        prefix = {k: v for k, v in prefix.items() if k != 'slotLength'}
    params = dict(r0=10, L=160, R=200, a=40, a0=7.9, k=1.38, q=4, s1=0.5, s2=0.2, **prefix)
    if family == 'OSSE':
        scalar, vector = calculate_osse, calculate_osse_curve
        join = sum(prefix.values())
        stations = np.array([join, join + 1e-10, join + 1e-9])
    else:
        scalar, vector = calculate_rosse, calculate_rosse_curve
        join = sum(prefix.values()) / rosse_axial_layout(params).full_length
        stations = np.array([join, join + 1e-12, join + 1e-11])
    result = np.column_stack(vector(stations, 0, params))
    np.testing.assert_allclose(result, [scalar(float(t), 0, params) for t in stations], rtol=0, atol=1e-12)
    assert np.max(np.abs(result[1:] - result[0])) < 1e-7


@pytest.mark.parametrize('family', list(CLASSES))
def test_radially_broken_junction_refused_even_when_axial_join_is_valid(family):
    params = dict(r0=10, L=160, R=200, k=-1, throatExtLength=20, s1=0.5, s2=0.2)
    scalar, vector = (calculate_osse, calculate_osse_curve) if family == 'OSSE' else (calculate_rosse, calculate_rosse_curve)
    for evaluate, station in [(scalar, 0), (vector, np.array([0]))]:
        with pytest.raises(ConfigError, match='continuous.*prefix/main junction'):
            evaluate(station, 0, params)


@pytest.mark.parametrize('family', list(CLASSES))
@pytest.mark.parametrize('key', ['s1', 's2'])
@pytest.mark.parametrize('value', [1e300, 1e307, STRETCH_COEFFICIENT_MAX + 1])
def test_unsupported_magnitudes_fail_with_coefficient_reason(family, key, value):
    profile = {'formula': family, 's1': 0.5, 's2': 0.2, key: value}
    scalar, vector = (calculate_osse, calculate_osse_curve) if family == 'OSSE' else (calculate_rosse, calculate_rosse_curve)
    for evaluate, station in [(scalar, 0), (vector, np.array([0]))]:
        with pytest.raises(ConfigError, match=key + '.*<= 10000'):
            evaluate(station, 0, profile)
    with pytest.raises(ConfigError, match=key + '.*<= 10000'):
        cb.resolve_geometry({'profile': profile, 'mode': 'bare'})


def test_bound_keeps_map_finite_for_extreme_finite_coordinates():
    x = np.array([-sys.float_info.max, -1e300, 0, 1e300, sys.float_info.max])
    actual = _stretch_x_curve(x, STRETCH_COEFFICIENT_MAX, STRETCH_COEFFICIENT_MAX)
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, [_stretch_x(float(t), STRETCH_COEFFICIENT_MAX, STRETCH_COEFFICIENT_MAX) for t in x])


def test_float_collapse_of_monotone_axial_map_is_refused():
    # Distinct finite inputs can round to the same output near maximum stretch.
    with pytest.raises(ConfigError, match='axial map.*monotone'):
        _stretch_x_curve(np.array([1.0, np.nextafter(1.0, 2.0)]), 10000, 10000)


@pytest.mark.parametrize('family', list(CLASSES))
def test_long_tapered_extension_rounding_discontinuity_is_refused(family):
    params = dict(r0=10, L=160, R=200, throatExtLength=1e17,
                  throatExtAngle=-45, s1=0.5, s2=0.2)
    scalar, vector = (calculate_osse, calculate_osse_curve) if family == 'OSSE' else (calculate_rosse, calculate_rosse_curve)
    for evaluate, station in [(scalar, 0), (vector, np.array([0]))]:
        with pytest.raises(ConfigError, match='continuous.*prefix/main junction'):
            evaluate(station, 0, params)


@pytest.mark.parametrize('q', [0, -1])
def test_rosse_noncontinuous_or_undefined_main_throat_is_refused(q):
    params = dict(R=200, r0=10, q=q, throatExtLength=20, s1=0.5, s2=0.2)
    for evaluate, station in [(calculate_rosse, 0), (calculate_rosse_curve, np.array([0]))]:
        with pytest.raises(ConfigError, match='continuous.*prefix/main junction'):
            evaluate(station, 0, params)


@pytest.mark.parametrize('cls', [OsseHornGeometry, RosseHornGeometry])
@pytest.mark.parametrize('dormant', DORMANT)
def test_legacy_instance_state_survives_deepcopy_pickle_and_replace(cls, dormant):
    import pickle
    original = cls(**dormant)
    legacy = cls()
    assert 's1' not in vars(original) and 's2' not in vars(original)
    for cloned in [copy.deepcopy(original), pickle.loads(pickle.dumps(original)), replace(original, k=1.0)]:
        assert cloned == legacy and asdict(cloned) == asdict(legacy)
        assert vars(cloned) == vars(legacy)
        assert hash(cloned) == hash(legacy)


def test_stretch_cannot_move_rosse_foldback_through_tapered_prefix(tmp_path, monkeypatch):
    params = dict(R=200, a=35, a0=3, r0=8.25, k=4, r=0.37, m=0.2, b=0,
                  q=4, s1=0.5, s2=0.2, throatExtLength=100, throatExtAngle=-60)
    # Adversarial reproduction: the unstretched meridian is clear, whereas
    # stretch crosses near composite t=0.6072754, x=50.748 mm, r=93.557 mm.
    stations = np.linspace(0, 1, 10001)
    x, radius = calculate_rosse_curve(stations, 0, {**params, 's1': 0})
    in_prefix = (x >= 0) & (x < 100)
    distance = radius - 8.25 - (x - 100) * math.tan(math.radians(-60))
    assert np.all(distance[in_prefix] > -1e-12)
    for evaluate, t in [(calculate_rosse, 0), (calculate_rosse, 0.6072754),
                        (calculate_rosse_curve, np.array([0])),
                        (calculate_rosse_curve, stations)]:
        with pytest.raises(ConfigError, match='intersects.*prefix|self-contact|self-intersection'):
            evaluate(t, 0, params)
    def no_mesh(*args, **kwargs):
        pytest.fail('crossing reached mesh construction')
    monkeypatch.setattr(cb, 'build_mesh_with_info', no_mesh)
    config = {'profile': {'formula': 'R-OSSE', **params}, 'mode': 'bare'}
    with pytest.raises(ConfigError, match='intersects.*prefix|self-contact|self-intersection'):
        cb.build_from_config(config, tmp_path / 'broken.msh')
    assert not (tmp_path / 'broken.msh').exists()
    # Changing any active coefficient/prefix input must invalidate validation.
    calculate_rosse_curve(np.array([0, 1]), 0, {**params, 's1': 0.001})
    calculate_rosse_curve(np.array([0, 1]), 0, {**params, 'throatExtAngle': -20})


def _resolved_value(value):
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {key: _resolved_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resolved_value(item) for item in value]
    return value


@pytest.mark.parametrize('family', ['OSSE', 'R-OSSE', 'FREEFORM', 'ICW'])
def test_absent_stretch_resolved_asdict_fields_state_and_repr_equal_base(family):
    captured = BASE['resolved'][family]
    resolved = cb.resolve_geometry(captured['config'])
    assert _resolved_value(asdict(resolved)) == captured['asdict']
    assert [f.name for f in fields(resolved.geometry)] == captured['geometry_fields']
    assert sorted(vars(resolved.geometry)) == captured['geometry_state_keys']
    assert repr(resolved) == captured['repr']


@pytest.mark.parametrize('case', BASE['zero_imports'], ids=lambda c: c['text'].split('Slot.Length = ')[1])
def test_evaluated_zero_import_result_equals_captured_base(case):
    assert parse_text_config(case['text']) == case['parsed']
