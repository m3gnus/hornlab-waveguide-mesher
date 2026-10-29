"""Fresh-process mesh and STEP identity against the S1-curve base.

Pin the writer's timestamp *before writing* with Geometry.OCCSTEPTimeStamp;
only the documented writer product counter is normalized in the comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import pytest
from identity_harness import archive_tree, assert_mesh_and_step_bytes

BASE = 'cf21494a94937f368e5b3c9f706f98e0a27c2667'
ROUNDED_BASE = 'd740c28ea4ee3dfb8662544d62613e16e98e76ed'
ROOT = Path(__file__).resolve().parents[1]


def identity_cases():
    data = json.loads((ROOT / 'tests/fixtures/morph_identity/baseline.json').read_text())
    configs = [(case['id'], case['config']) for case in data['cases'] if case['config']['formula'] == 'FREEFORM']
    for family in ('OSSE', 'R-OSSE'):
        case = next(case for case in data['cases'] if case['config']['formula'] == family)
        configs.append((case['id'], case['config']))
    data = json.loads((ROOT / 'tests/fixtures/throat_stretch/base-compatibility.json').read_text())
    configs.append(('LOOKUP-composed-outline-profile', {'formula':'LOOKUP','mode':'bare',
        'profile':{'lookupProfile':[[0,12.7],[30,28],[80,85],[120,150]]},
        'mesh':{'allow_large_mesh':True}}))
    configs.append(('FREEFORM-stretch-compatibility', data['resolved']['FREEFORM']['config']))
    configs.append(('FREEFORM-example', tomllib.loads((ROOT / 'examples/freeform-bare.toml').read_text())))
    from hornlab_mesher import freeform_insert_anchor
    handles = freeform_insert_anchor({'points':[[0,12.7],[40,30],[100,60]]}, z=20,
                                      default_throat_angle_deg=15.5)
    configs.append(('FREEFORM-handles-C1', {'formula':'FREEFORM','mode':'bare',
        'profile':{'profileH':handles,'profileV':handles},
        'mesh':{'throat_res_mm':15,'mouth_res_mm':15,'allow_large_mesh':True}}))
    from test_freeform_corner_wall import case_config
    for mode in ('freestanding', 'enclosure', 'infinite-baffle'):
        for quadrants in ('1234', '12'):
            config = case_config('a', mode)
            config['mesh']['quadrants'] = quadrants
            if mode == 'enclosure':
                config['enclosure']['edge'] = 0
            for plane in ('profileH', 'profileV'):
                for point in config['profile'][plane]['points']:
                    point.pop('join')
            configs.append((f'FREEFORM-{mode}-{quadrants}', config))
    assert len(configs) == 20
    return configs


@pytest.fixture(scope='module')
def base_tree(tmp_path_factory):
    return archive_base(tmp_path_factory, BASE)


@pytest.fixture(scope='module')
def rounded_base_tree(tmp_path_factory):
    return archive_base(tmp_path_factory, ROUNDED_BASE)


def archive_base(tmp_path_factory, base):
    # The historical base is immutable; a rebase changes ancestry, not the
    # compatibility promise. Missing/unreadable objects fail git archive.
    return archive_tree(tmp_path_factory, base)


@pytest.mark.parametrize('name,config', identity_cases(), ids=[item[0] for item in identity_cases()])
def test_base_mesh_and_step_bytes(base_tree, tmp_path, name, config):
    assert_mesh_and_step_bytes(base_tree, tmp_path, name, config)


@pytest.mark.parametrize('family', ['FREEFORM', 'OSSE'])
def test_rounded_enclosure_mesh_and_step_bytes(rounded_base_tree, tmp_path, family):
    from test_freeform_corner_wall import rounded_enclosure_config, without_corners

    config = without_corners(rounded_enclosure_config(case='a'))
    if family == 'OSSE':
        data = json.loads((ROOT / 'tests/fixtures/morph_identity/baseline.json').read_text())
        osse = next(case['config'] for case in data['cases'] if case['config']['formula'] == family)
        config['formula'] = family
        config['profile'] = osse['profile']
    assert_mesh_and_step_bytes(rounded_base_tree, tmp_path, family, config)
