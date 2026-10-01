"""Release regressions against identities/imports captured from 5c8ea4dc."""
import copy
from dataclasses import asdict, astuple, fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
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


def _assert_base_equal(actual, expected):
    """Keep captured structure exact; allow only cross-platform float drift."""
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        if np.issubdtype(expected.dtype, np.floating):
            zero = expected == 0
            np.testing.assert_allclose(actual[~zero], expected[~zero], rtol=1e-12, atol=0)
            np.testing.assert_allclose(actual[zero], expected[zero], rtol=0, atol=1e-12)
        else:
            np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, (float, np.floating)):
        assert isinstance(actual, (int, float, np.floating)) and not isinstance(actual, bool)
        if isinstance(actual, int):
            assert actual == expected
        else:
            assert math.isclose(actual, expected, rel_tol=1e-12,
                                abs_tol=1e-12 if expected == 0 else 0)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys()
        for key in expected:
            _assert_base_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_base_equal(a, e)
    else:
        # ATH may normalize an integer token such as 0 to 0.0. Preserve the
        # original exact numeric equality without admitting boolean aliases.
        if isinstance(expected, int) and not isinstance(expected, bool):
            assert isinstance(actual, (int, float)) and not isinstance(actual, bool)
        else:
            assert type(actual) is type(expected)
        assert actual == expected


# Consume quoted strings and identifiers whole so their numeric-looking text
# (including field names and dtypes) can never receive float tolerance.
_REPR_TOKEN = re.compile(
    r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|[A-Za-z_]\w*|"
    r'(?P<float>(?<![\w.])[+-]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])'
    r'|(?<![\w.])[+-]?\d+[eE][+-]?\d+(?![\w.]))'
)


def _assert_base_repr_equal(actual, expected):
    """Compare every character outside float tokens to the captured repr."""
    def split(text):
        chunks, floats, start = [], [], 0
        for token in _REPR_TOKEN.finditer(text):
            if token.group('float') is not None:
                chunks.append(text[start:token.start()])
                floats.append(float(token.group()))
                start = token.end()
        chunks.append(text[start:])
        return chunks, floats

    actual_text, actual_floats = split(actual)
    expected_text, expected_floats = split(expected)
    assert actual_text == expected_text
    assert len(actual_floats) == len(expected_floats)
    for actual_float, expected_float in zip(actual_floats, expected_floats):
        assert math.isclose(actual_float, expected_float, rel_tol=1e-12, abs_tol=0)


def test_captured_base_float_tolerance_includes_only_expected_zero_atol():
    _assert_base_equal({'x': np.nextafter(1., 2.).item(), 'zero': 5e-13},
                       {'x': 1., 'zero': 0.})
    _assert_base_equal(np.array([np.nextafter(1., 2.), 5e-13]), np.array([1., 0.]))
    _assert_base_equal(np.float32(1.), np.float32(1.))
    for actual, expected in [(0., 1e-300), (1. + 2e-12, 1.), (2e-12, 0.),
                             (np.array([0.]), np.array([1e-300]))]:
        with pytest.raises(AssertionError):
            _assert_base_equal(actual, expected)


@pytest.mark.parametrize('actual,expected', [
    ({'other': 1.}, {'x': 1.}), ([1.], [1., 2.]), ('1.000000000001', '1'),
    (True, 1), (1, True), (1. + 5e-13, 1),
    (np.ones(2), np.ones(3)), (np.ones(2, dtype=np.float32), np.ones(2)),
    (np.array([10**12 + 1], dtype=np.int64), np.array([10**12], dtype=np.int64)),
    (np.array([2**60 + 1], dtype=np.int64), np.array([2**60], dtype=np.int64)),
    (np.array([True]), np.array([False])),
    (np.array(['changed']), np.array(['capture'])),
])
def test_captured_base_non_float_structure_remains_exact(actual, expected):
    with pytest.raises(AssertionError):
        _assert_base_equal(actual, expected)


def test_captured_base_repr_allows_only_float_token_drift():
    _assert_base_repr_equal('Geometry(x=1.0000000000000002, y=-2e-3, z=.0)',
                            'Geometry(x=1.0, y=-0.002, z=0.0)')
    # Array whitespace, signs, scientific notation and multiple lines survive.
    _assert_base_repr_equal('array([[ +1.0000000000000002, -2.0],\n [ 3e0, 4.]])',
                            'array([[ +1.0, -2.],\n [ 3.0, 4.0]])')


@pytest.mark.parametrize('actual,expected', [
    ('Xeometry(x=1.0)', 'Geometry(x=1.0)'),
    ('Geometry(x=1.0) ', 'Geometry(x=1.0)'),
    ('Geometry(x =1.0)', 'Geometry(x=1.0)'),
    ('Geometry(x=1.000000000002)', 'Geometry(x=1.0)'),
    ('Geometry(x=0.0)', 'Geometry(x=1e-300)'),
    ('Geometry(x=1e-300)', 'Geometry(x=0.0)'),
    ('Geometry(n=1000000000001)', 'Geometry(n=1000000000000)'),
    ('Geometry(flag=False)', 'Geometry(flag=True)'),
    ("Geometry(name='1.0000000000000002')", "Geometry(name='1.0')"),
    ('Geometry(name="1.0000000000000002")', 'Geometry(name="1.0")'),
    (r"Geometry(name='it\'s 1.0000000000000002')", r"Geometry(name='it\'s 1.0')"),
    ('Geometry(field1=1.0)', 'Geometry(field2=1.0)'),
    ('array([1.0], dtype=float32)', 'array([1.0], dtype=float64)'),
    ('Geometry(x=1)', 'Geometry(x=1.0)'),
    ('Geometry(x=1.0, y=2.0)', 'Geometry(x=1.0)'),
])
def test_captured_base_repr_non_float_text_remains_exact(actual, expected):
    with pytest.raises(AssertionError):
        _assert_base_repr_equal(actual, expected)


@pytest.mark.parametrize('name', [k for k in BASE['identities'] if not k.endswith('-config')])
@pytest.mark.parametrize('dormant', DORMANT)
def test_dataclass_legacy_identity_and_serialization(name, dormant):
    captured = BASE['identities'][name]
    cls, adapter = CLASSES[captured['family']]
    geometry = cls(**captured['kwargs'], **dormant)
    legacy = cls(**captured['kwargs'])
    assert geometry == legacy and hash(geometry) == hash(legacy)
    _assert_base_equal(asdict(geometry), captured['asdict'])
    assert astuple(geometry) == astuple(legacy)
    assert repr(geometry) == repr(legacy)
    _assert_base_repr_equal(repr(geometry), captured['repr'])
    _assert_base_equal(adapter(geometry), captured['adapter'])
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
    _assert_base_equal(cb.build_geometry_params(native)[0], captured['native_params'])
    text = captured['text'].replace('\n}', ''.join(f'\n{k} = {v}' for k, v in dormant.items()) + '\n}')
    parsed = parse_text_config(text)
    assert parsed == parse_text_config(captured['text'])
    _assert_base_equal(cb.build_geometry_params(parsed)[0], captured['text_params'])


@pytest.mark.parametrize('dormant', DORMANT)
def test_dormant_icw_seed_key_matches_base(dormant):
    assert _icw_cache_key({'icw_seed': {'type': 'OSSE', 'L': 160, **dormant}}) == BASE['icw_disabled_key']


# These synthetic variants previously dropped nonzero R-OSSE Rot. Refuse it
# until ATH parity establishes the geometry, as documented in config-schema.
# No archive case is exempted from the captured import/parameter comparison.
UNVERIFIED_ROSSE_ROT_VARIANTS = {
    f'variant-{number:02d}' for number in (22, 23, 24, 26, 28, 30, 31, 32, 33, 34, 35, 36)
}


@pytest.mark.parametrize('case', BASE['imports'], ids=lambda c: c['id'])
def test_import_corpus_equal_base_except_documented_rotation_changes(case):
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
    # Only top-level Rot beside an OSSE block may differ. In-block Rot still wins
    # over a top-level Rot when both are supplied, exactly as at the base.
    osse_block = text.startswith('OSSE = {')
    block_text = text.split('}', 1)[0]
    approved_rot = osse_block and 'Rot = 10' in text and 'Rot = 5' not in block_text
    if 'error' in case:
        with pytest.raises(ConfigError, match=case['message']):
            parse_text_config(text)
        return
    if case['id'] in UNVERIFIED_ROSSE_ROT_VARIANTS:
        with pytest.raises(ConfigError, match=r'unsupported item.*\bRot\b'):
            parse_text_config(text)
        return
    parsed = parse_text_config(text)
    params = cb.build_geometry_params(parsed)[0]
    expected_parsed, expected_params = copy.deepcopy(case['parsed']), dict(case['params'])
    if approved_rot:
        expected_parsed['profile']['rot'] = 10
        expected_params['rot'] = 10
    _assert_base_equal(parsed, expected_parsed)
    _assert_base_equal(params, expected_params)


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
    params = dict(r0=10, a=40, a0=7.9, k=1.38, q=4, s1=0.5, s2=0.2, **prefix)
    params.update({'L': 160} if family == 'OSSE' else {'R': 200})
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
    params = dict(r0=10, k=-1, throatExtLength=20, s1=0.5, s2=0.2)
    params.update({'L': 160} if family == 'OSSE' else {'R': 200})
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
        with pytest.raises(ConfigError, match=key + '.*<= 10'):
            evaluate(station, 0, profile)
    with pytest.raises(ConfigError, match=key + '.*<= 10'):
        cb.resolve_geometry({'profile': profile, 'mode': 'bare'})


def test_bound_keeps_map_finite_for_extreme_finite_coordinates():
    x = np.array([-sys.float_info.max, -1e300, 0, 1e300, sys.float_info.max])
    actual = _stretch_x_curve(x, STRETCH_COEFFICIENT_MAX, STRETCH_COEFFICIENT_MAX)
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, [_stretch_x(float(t), STRETCH_COEFFICIENT_MAX, STRETCH_COEFFICIENT_MAX) for t in x])


def test_adjacent_stations_use_scalar_monotonicity_rule():
    stations = np.array([0.5, np.nextafter(0.5, 1.0)])
    np.testing.assert_allclose(_stretch_x_curve(stations, 10, 10),
                               [_stretch_x(float(t), 10, 10) for t in stations],
                               rtol=1e-12, atol=0)


@pytest.mark.parametrize('family', list(CLASSES))
def test_long_tapered_extension_rounding_discontinuity_is_refused(family):
    params = dict(r0=10, throatExtLength=1e17,
                  throatExtAngle=-45, s1=0.5, s2=0.2)
    params.update({'L': 160} if family == 'OSSE' else {'R': 200})
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
        return value
    if isinstance(value, dict):
        return {key: _resolved_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resolved_value(item) for item in value]
    return value


def _captured_resolved_value(value, path, arrays):
    if isinstance(value, dict):
        if set(value) == {'shape', 'dtype', 'sha256'}:
            array = arrays[path]
            assert list(array.shape) == value['shape'] and str(array.dtype) == value['dtype']
            # The portable numeric capture must match the original base bytes.
            assert hashlib.sha256(array.tobytes()).hexdigest() == value['sha256']
            return array
        return {key: _captured_resolved_value(item, path + '.' + key, arrays)
                for key, item in value.items()}
    return value


@pytest.mark.parametrize('family', ['OSSE', 'R-OSSE', 'FREEFORM', 'ICW'])
def test_absent_stretch_resolved_asdict_fields_state_and_repr_equal_base(family):
    captured = BASE['resolved'][family]
    resolved = cb.resolve_geometry(captured['config'])
    with np.load(Path(__file__).parent / 'fixtures/throat_stretch/base-resolved-arrays.npz') as arrays:
        expected = _captured_resolved_value(captured['asdict'], family, arrays)
        _assert_base_equal(_resolved_value(asdict(resolved)), expected)
    assert [f.name for f in fields(resolved.geometry)] == captured['geometry_fields']
    assert sorted(vars(resolved.geometry)) == captured['geometry_state_keys']
    _assert_base_repr_equal(repr(resolved), captured['repr'])


@pytest.mark.parametrize('case', BASE['zero_imports'], ids=lambda c: c['text'].split('Slot.Length = ')[1])
def test_evaluated_zero_import_result_equals_captured_base(case):
    _assert_base_equal(parse_text_config(case['text']), case['parsed'])


@pytest.mark.parametrize('case', BASE['composition_imports'])
@pytest.mark.parametrize('dormant', DORMANT)
def test_inactive_composition_imports_equal_true_base(case, dormant):
    text = case['text'].replace('\n}', ''.join(f'\n{k} = {v}' for k, v in dormant.items()) + '\n}')
    expected = copy.deepcopy(case['parsed'])
    if 'top_level_rot' in case:
        expected['profile']['rot'] = case['top_level_rot']
    _assert_base_equal(parse_text_config(text), expected)


@pytest.mark.parametrize('key', ['s1', 's2'])
@pytest.mark.parametrize('formula', ['OSSE', 'R-OSSE'])
def test_negative_coefficient_is_refused_with_its_own_reason(key, formula):
    profile = {'L': 80} if formula == 'OSSE' else {'formula': 'R-OSSE', 'R': 150}
    profile.update({'s1': 0.5, 's2': 0.2, key: -0.05})
    with pytest.raises(ConfigError, match=key + ' must not be negative.*fold the profile back'):
        cb.resolve_geometry({'mode': 'bare', 'profile': profile})
