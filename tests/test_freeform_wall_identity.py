"""Fresh-process mesh and STEP identity against the S1-curve base.

Pin the writer's timestamp *before writing* with Geometry.OCCSTEPTimeStamp;
only the documented writer product counter is normalized in the comparison.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import pytest

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
    if not (ROOT / '.git').exists():
        pytest.skip(f'S1-wall slice gate requires a Git checkout with base {base[:7]}')
    ancestry = subprocess.run(['git', 'merge-base', '--is-ancestor', base, 'HEAD'],
                              cwd=ROOT, capture_output=True)
    if ancestry.returncode != 0:
        pytest.skip(f'S1-wall slice gate applies only while {base[:7]} is an ancestor of HEAD')
    archive = subprocess.run(['git','archive',base], cwd=ROOT, check=True, capture_output=True).stdout
    tree = tmp_path_factory.mktemp('wall-identity-base')
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tree, filter='data')
    return tree


WORKER = '''
import json, sys
from pathlib import Path
import gmsh
from hornlab_mesher import build_from_config, write_step_from_config
root, source, destination = map(Path, sys.argv[1:])
import hornlab_mesher
assert Path(hornlab_mesher.__file__).resolve().is_relative_to(root.resolve())
config=json.loads(source.read_text())
gmsh.initialize(interruptible=False)
gmsh.option.setString('Geometry.OCCSTEPTimeStamp','2000-01-01T00:00:00')
try:
    build_from_config(config, destination/'out.msh')
    write_step_from_config(config, destination/'out.step')
finally:
    gmsh.finalize()
'''


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


def assert_mesh_and_step_bytes(base_tree, tmp_path, name, config):
    source = tmp_path/'config.json'
    source.write_text(json.dumps(config))
    artifacts=[]
    for label, tree in (('base', base_tree), ('current', ROOT)):
        target=tmp_path/label
        target.mkdir()
        with (target/'build.log').open('w') as log:
            subprocess.run([sys.executable, '-c', WORKER, str(tree), str(source), str(target)],
                cwd=tree, env={**os.environ, 'PYTHONPATH': str(tree)}, stdout=log, stderr=subprocess.STDOUT,
                check=True, timeout=50)
        artifacts.append(((target/'out.msh').read_bytes(), (target/'out.step').read_bytes()))
    assert artifacts[0][0] == artifacts[1][0], name
    normalize=lambda value: re.sub(rb'Open CASCADE STEP translator 7\.8 \d+', b'Open CASCADE STEP translator 7.8 N', value)
    assert normalize(artifacts[0][1]) == normalize(artifacts[1][1]), name
