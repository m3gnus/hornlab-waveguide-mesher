"""Saved V2 designs retain 746c6b1 values and artifact bytes, without migration."""
from pathlib import Path
import copy,hashlib,json
import os
import subprocess
import sys

import numpy as np
import pytest

from identity_harness import archive_tree, assert_mesh_and_step_bytes
from hornlab_mesher.config_builder import build_geometry_params, resolve_geometry
from hornlab_mesher.config_parser import ConfigError, parse_text_config
from hornlab_mesher.profile_sampling import build_point_grid_arrays
from hornlab_mesher.profile_formulas import calculate_rosse
from hornlab_mesher.builders.point_grid_sources import _source_cap_radius, _source_cap_height, _throat_radius
from hornlab_mesher.text_import import TEXT_IMPORT_VERSION, TEXT_IMPORT_VERSION_KEY

TEXT_IMPORT_V2='ath-2026-09a-v2'

ROOT=Path(__file__).parent/'fixtures/ath_import/saved-v2-controls'
MANIFEST=json.loads((ROOT/'manifest.json').read_text())
NAMES=[row['name'] for row in MANIFEST['cases']]


def saved(name):
    return parse_text_config((ROOT/(name+'.cfg')).read_text())


def test_saved_v2_fixtures_come_from_the_immutable_base():
    assert MANIFEST['base']=='746c6b1ca6b8782d018d5b86b0f7c5e5cadca2bd'
    for name,digest in MANIFEST['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name


@pytest.fixture(scope='module')
def same_runtime_saved_reference(old_tree, tmp_path_factory):
    # Exact saved identity is against the immutable implementation on this host.
    # Windows captures remain exact secondary witnesses on their authored platform.
    destination = tmp_path_factory.mktemp('saved-control-reference')
    worker = r"""
import json,sys
from pathlib import Path
import numpy as np
import hornlab_mesher
from hornlab_mesher.config_parser import parse_text_config
from hornlab_mesher.config_builder import build_geometry_params,resolve_geometry
from hornlab_mesher.profile_sampling import build_point_grid_arrays
from hornlab_mesher.profile_formulas import calculate_rosse
from hornlab_mesher.builders.point_grid_sources import _source_cap_radius,_source_cap_height,_throat_radius
root,fixtures,destination=map(Path,sys.argv[1:])
assert Path(hornlab_mesher.__file__).resolve().is_relative_to(root.resolve())
manifest=json.loads((fixtures/'manifest.json').read_text())
arrays={};sources={}
for version in [None,'ath-2026-08c-v1','ath-2026-09a-v2']:
    params=dict(manifest['scalar_rosse_params'])
    if version:params['_textImportVersion']=version
    arrays['scalar_'+str(version)]=np.array([calculate_rosse(float(t),0,params) for t in np.linspace(0,1,51)])
for row in manifest['cases']:
    name=row['name']
    # The baseline owns parsing too: candidate-derived params could hide parser drift.
    config=parse_text_config((fixtures/(name+'.cfg')).read_text())
    arrays[name]=build_point_grid_arrays(build_geometry_params(config)[0])['inner_grid']
    geometry=resolve_geometry(config).geometry
    radius=_throat_radius(np.asarray(geometry.inner_points),closed=geometry.closed)
    sources[name]=[geometry.source_auto_angle_deg,_source_cap_radius(radius,geometry),_source_cap_height(radius,geometry)]
np.savez(destination/'arrays.npz',**arrays)
(destination/'sources.json').write_text(json.dumps(sources))
"""
    subprocess.run([sys.executable, '-c', worker, str(old_tree), str(ROOT), str(destination)],
                   cwd=old_tree, env={**os.environ, 'PYTHONPATH': str(old_tree),
                                      'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
                   stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True, timeout=90)
    with np.load(destination/'arrays.npz') as values:
        arrays={key:values[key].copy() for key in values.files}
    return arrays,json.loads((destination/'sources.json').read_text())


@pytest.mark.parametrize('version',[None,'ath-2026-08c-v1',TEXT_IMPORT_V2])
def test_scalar_rosse_keeps_original_arithmetic_association(version, same_runtime_saved_reference):
    params=dict(MANIFEST['scalar_rosse_params'])
    if version:params[TEXT_IMPORT_VERSION_KEY]=version
    actual=np.array([calculate_rosse(float(t),0,params) for t in np.linspace(0,1,51)])
    np.testing.assert_array_equal(actual,same_runtime_saved_reference[0]['scalar_'+str(version)])
    if sys.platform=='win32':
        with np.load(ROOT/'scalar-rosse.npz') as data:
            np.testing.assert_array_equal(actual,data[version or 'native-v1'])


@pytest.mark.parametrize('name',NAMES)
def test_saved_v2_grid_and_source_are_exactly_unchanged(name, same_runtime_saved_reference):
    cfg=saved(name)
    expected=next(row for row in MANIFEST['cases'] if row['name']==name)
    old=same_runtime_saved_reference[0][name]
    if sys.platform=='win32':
        with np.load(ROOT/'grids.npz') as grids:
            np.testing.assert_array_equal(old,grids[name])
    for config in (cfg,json.loads(json.dumps(cfg))):
        assert config[TEXT_IMPORT_VERSION_KEY]==TEXT_IMPORT_V2
        params=build_geometry_params(config)[0]
        np.testing.assert_array_equal(build_point_grid_arrays(params)['inner_grid'],old)
        geometry=resolve_geometry(config).geometry
        radius=_throat_radius(np.asarray(geometry.inner_points),closed=geometry.closed)
        assert [geometry.source_auto_angle_deg,_source_cap_radius(radius,geometry),
                _source_cap_height(radius,geometry)]==same_runtime_saved_reference[1][name]
        if sys.platform=='win32':
            assert same_runtime_saved_reference[1][name]==expected['source_cap']


@pytest.fixture(scope='module')
def old_tree(tmp_path_factory):
    return archive_tree(tmp_path_factory,MANIFEST['base'])


@pytest.mark.parametrize('name',[
    'OSSE-n9-ext1-rot0','OSSE-n37-ext1-rot1',
    'R-OSSE-n12-ext0-rot0','R-OSSE-n20-ext1-rot0','OSSE-unit-axis-global3',
])
def test_saved_v2_mesh_and_normalized_step_are_identical(old_tree,tmp_path,name):
    cfg=saved(name)
    # A rolled-back R-OSSE mouth is not an admissible infinite-baffle plane
    # even at the base. Exercise its supported inner-wall JSON transport.
    cfg['mode']='bare'
    assert_mesh_and_step_bytes(old_tree,tmp_path,name,cfg)


@pytest.mark.parametrize('name',['OSSE-n9-ext1-rot0','R-OSSE-n12-ext0-rot0'])
def test_explicit_new_stamp_selects_the_correction_without_mutating_saved_input(name):
    old=saved(name)
    before=copy.deepcopy(old)
    fresh=copy.deepcopy(old);fresh[TEXT_IMPORT_VERSION_KEY]=TEXT_IMPORT_VERSION
    a=build_point_grid_arrays(build_geometry_params(old)[0])['inner_grid']
    b=build_point_grid_arrays(build_geometry_params(fresh)[0])['inner_grid']
    assert not np.array_equal(a,b)
    assert old==before
    unversioned=(ROOT/(name+'.cfg')).read_text().split('\n',1)[1]
    assert parse_text_config(unversioned)[TEXT_IMPORT_VERSION_KEY]==TEXT_IMPORT_VERSION


@pytest.mark.parametrize('key',['Scale.Z = 2','Mesh.ThroatExtSegments = 4','Rot = 5'])
def test_saved_v2_does_not_gain_previously_refused_controls(key):
    with pytest.raises(ConfigError):
        parse_text_config((ROOT/'R-OSSE-n12-ext0-rot0.cfg').read_text()+key+'\n')


def test_saved_v2_unit_axis_scale_keeps_the_previous_noop():
    old=saved('OSSE-unit-axis-global3')
    assert 'athAxialScale' not in old
    text=(ROOT/'OSSE-unit-axis-global3.cfg').read_text().replace(TEXT_IMPORT_V2,TEXT_IMPORT_VERSION)
    assert parse_text_config(text)['athAxialScale']==1
