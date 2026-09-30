"""Numeric composition boundaries, independent of any azimuth zero probe."""
import itertools
import math
import sys

import numpy as np
import pytest

from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.config_parser import ConfigError, load_config, parse_text_config
from hornlab_mesher.profile_formulas import (
    calculate_osse, calculate_osse_curve, calculate_rosse, calculate_rosse_curve,
)
from hornlab_mesher.throat_stretch import COMPOSITION_PROFILE_KEYS, COMPOSITION_GUIDE_KEYS
from test_throat_stretch_review import PUBLIC, consume

EXPRESSIONS = ['0*p', 'sin(p-p)', '0/(1+p^2)', '(p+1)-(1+p)',
               'cos(p)^2+sin(p)^2-1', '10000*(cos(p)^2+sin(p)^2-1)',
               '20*sin(720*p)^2', '20 if abs(p-pi/64)<0.001 else 0', '1/0']
KEYS = {**COMPOSITION_PROFILE_KEYS, **COMPOSITION_GUIDE_KEYS}


def text_config(family, key, value):
    dimension = 'L = 160' if family == 'OSSE' else 'R = 200'
    return f'{family} = {{\n{dimension}\ns1 = .5\ns2 = .2\n}}\n{KEYS[key][0]} = {value}'


@pytest.mark.parametrize('family', ['OSSE', 'R-OSSE'])
@pytest.mark.parametrize('key', KEYS)
@pytest.mark.parametrize('expression', EXPRESSIONS)
def test_active_composition_expressions_refuse_from_native_text_and_evaluators(family, key, expression):
    p = {'L': 160, 'R': 200, 's1': .5, 's2': .2, key: expression}
    if family == 'R-OSSE':
        p.pop('L')
    config = {'formula': family, 'profile': {k: v for k, v in p.items() if k not in COMPOSITION_GUIDE_KEYS},
              'gcurve': {k: v for k, v in p.items() if k in COMPOSITION_GUIDE_KEYS}}
    scalar, vector = (calculate_osse, calculate_osse_curve) if family == 'OSSE' else (calculate_rosse, calculate_rosse_curve)
    errors = []
    for call in [lambda: parse_text_config(text_config(family, key, expression)),
                 lambda: build_geometry_params(config),
                 lambda: scalar(.25, math.pi/1440, p),
                 lambda: vector(np.array([.25, .5]), math.pi/1440, p)]:
        with pytest.raises(ConfigError, match=f'throat stretch does not support per-azimuth {KEYS[key][0]} yet') as error:
            call()
        errors.append(str(error.value))
    assert len(set(errors)) == 1


@pytest.mark.parametrize('key', KEYS)
@pytest.mark.parametrize('value', ['0', '', None, True, float('inf'), float('nan')])
def test_active_compositions_require_plain_finite_native_inputs(key, value):
    section = 'gcurve' if key in COMPOSITION_GUIDE_KEYS else 'profile'
    config = {'profile': {'L': 160, 's1': .5, 's2': .2}, section: {key: value}}
    config.setdefault('profile', {}).update(L=160, s1=.5, s2=.2)
    with pytest.raises(ConfigError, match='plain finite number'):
        build_geometry_params(config)


@pytest.mark.parametrize('expression', EXPRESSIONS[-4:-1])
@pytest.mark.parametrize('key', ['rot', 'slotLength'])
@pytest.mark.parametrize('path', PUBLIC)
@pytest.mark.parametrize('entry', ['native', 'json', 'toml', 'text'])
def test_review_expressions_refuse_before_every_consumer(expression, key, path, entry, tmp_path):
    p = {'L': 160, 's1': .5, 's2': .2, 'throatExtLength': 20, key: expression}
    config = {'profile': p, 'mode': 'bare'}
    expected = f'throat stretch does not support per-azimuth {KEYS[key][0]} yet'
    if entry == 'text':
        with pytest.raises(ConfigError, match=expected):
            parse_text_config(text_config('OSSE', key, expression) + '\nThroat.Ext.Length = 20')
        return
    if entry in ('json', 'toml'):
        import json
        file = tmp_path / ('config.' + entry)
        content = json.dumps(config) if entry == 'json' else 'mode = "bare"\n[profile]\n' + '\n'.join(f'{k} = {json.dumps(v)}' for k, v in p.items())
        file.write_text(content)
        config = load_config(file)
    with pytest.raises(ConfigError, match=expected):
        consume(path, config, tmp_path)
    assert not list(tmp_path.glob('*.msh')) and not list(tmp_path.glob('*.step'))


@pytest.mark.parametrize('family,prefix,rot,guide', list(itertools.product(
    ['OSSE', 'R-OSSE'], [{}, {'throatExtLength': 12, 'throatExtAngle': -3},
                          {'slotLength': 8}, {'throatExtLength': 12, 'slotLength': 8}],
    [0, 10], [False, True])))
@pytest.mark.parametrize('phi', [0, math.pi/1440, math.pi/64, math.pi/2])
def test_scalar_vector_acceptance_and_values_agree_for_compositions(family, prefix, rot, guide, phi):
    p = {'r0': 10, 'a': 40, 'a0': 7.9, 'k': 1.38, 'q': 4, 's1': .5, 's2': .2,
         'rot': rot, **prefix}
    p.update({'L': 160, 'n': 4, 's': .7} if family == 'OSSE' else {'R': 200, 'm': .84, 'r': .37, 'b': .2})
    if guide:
        p.update(gcurveType=1, gcurveWidth=400, gcurveAspectRatio=1, gcurveDist=1)
    scalar, vector = (calculate_osse, calculate_osse_curve) if family == 'OSSE' else (calculate_rosse, calculate_rosse_curve)
    stations = np.linspace(0, 160 if family == 'OSSE' else 1, 33)
    refused = (family == 'OSSE' and prefix.get('slotLength', 0) != 0 or
               rot != 0 and (family == 'R-OSSE' or bool(prefix)) or
               guide and (bool(prefix) or rot != 0))
    if refused:
        with pytest.raises(ConfigError) as verror:
            vector(stations, phi, p)
        for station in stations:
            with pytest.raises(ConfigError) as serror:
                scalar(float(station), phi, p)
            assert str(serror.value) == str(verror.value)
    else:
        actual = np.column_stack(vector(stations, phi, p))
        expected = np.array([scalar(float(station), phi, p) for station in stations])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize('key', ['rot', 'slotLength', 'gcurveType', 'gcurveWidth'])
def test_tiny_numeric_compositions_use_exact_zero(key):
    p = {'L': 160, 's1': .5, 's2': .2, 'throatExtLength': 12,
         'gcurveType': 1 if key == 'gcurveWidth' else 0,
         'gcurveWidth': 400 if key == 'gcurveType' else 0, key: 1e-13}
    with pytest.raises(ConfigError, match='unverified'):
        calculate_osse(20, 0, p)
    p[key] = 0
    calculate_osse(20, 0, p)


def test_active_rosse_numeric_zero_rotation_matches_text():
    imported = parse_text_config('R-OSSE = {\nR = 200\ns1 = .5\ns2 = .2\n}\nRot = 0')
    native = {'formula': 'R-OSSE', 'profile': {'R': 200, 's1': .5, 's2': .2, 'rot': 0, '_athLengthMode': 'total'},
              'mesh': imported['mesh'], 'simType': imported['simType']}
    assert build_geometry_params(native) == build_geometry_params(imported)


@pytest.mark.parametrize('dimension', ['L', 'L_mm', 'Length'])
def test_direct_rosse_length_composition_refuses_in_both_evaluators(dimension):
    p = {'R': 200, 's1': .5, 's2': .2, dimension: 0}
    with pytest.raises(ConfigError) as scalar:
        calculate_rosse(.5, 0, p)
    with pytest.raises(ConfigError) as vector:
        calculate_rosse_curve([.5], 0, p)
    assert str(scalar.value) == str(vector.value)


@pytest.mark.parametrize('key,alias', [(key, alias) for key, (_, aliases) in KEYS.items() for alias in aliases])
def test_native_composition_alias_cannot_hide_an_expression(key, alias):
    config = {'profile': {'L': 160, 's1': .5, 's2': .2}}
    config.setdefault('gcurve' if key in COMPOSITION_GUIDE_KEYS else 'profile', {})[alias] = '0*p'
    with pytest.raises(ConfigError, match=f'per-azimuth {KEYS[key][0]} yet'):
        build_geometry_params(config)


@pytest.mark.parametrize('family', ['OSSE', 'R-OSSE'])
@pytest.mark.parametrize('s1,s2', list(itertools.product(
    [0, sys.float_info.min, 1e-13, .05, .2, 1, 5, np.nextafter(10., 0), 10.], repeat=2)))
@pytest.mark.parametrize('station', [0., sys.float_info.min, .25, .5, .999])
def test_scalar_vector_acceptance_at_adjacent_stations_across_coefficient_range(family, s1, s2, station):
    p = {('L' if family == 'OSSE' else 'R'): 160, 's1': s1, 's2': s2}
    scalar, vector = (calculate_osse, calculate_osse_curve) if family == 'OSSE' else (calculate_rosse, calculate_rosse_curve)
    stations = np.array([station, np.nextafter(station, math.inf)])
    expected = np.array([scalar(float(t), 0, p) for t in stations])
    actual = np.column_stack(vector(stations, 0, p))
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize('family', ['OSSE', 'R-OSSE'])
@pytest.mark.parametrize('key', ['s1', 's2'])
@pytest.mark.parametrize('value', [np.nextafter(10., math.inf), 11., 10000.])
def test_above_ten_refused_by_both_evaluators_and_native_normalization(family, key, value):
    value = float(value)
    p = {('L' if family == 'OSSE' else 'R'): 160, 's1': .5, 's2': .2, key: value}
    scalar, vector = (calculate_osse, calculate_osse_curve) if family == 'OSSE' else (calculate_rosse, calculate_rosse_curve)
    errors = []
    for call in [lambda: scalar(.5, 0, p), lambda: vector([.5], 0, p),
                 lambda: build_geometry_params({'formula': family, 'profile': p}),
                 lambda: parse_text_config(f'{family} = {{\n{key} = {value!r}\n}}')]:
        with pytest.raises(ConfigError, match=key + r'.*<= 10(?:,|$)') as error:
            call()
        errors.append(str(error.value))
    assert len(set(errors[:3])) == 1
