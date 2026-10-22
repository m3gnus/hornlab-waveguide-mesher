"""Byte identity for every legacy native marker against the v0.2.6 release."""
from __future__ import annotations

import subprocess

import pytest
from identity_harness import ROOT, archive_tree, assert_mesh_and_step_bytes
from test_absolute_axial_scale import config as axial_config
from test_adapter_axial_scale import config as adapter_axial_config
from test_native_adapter_controls import config as controls_config
from test_native_mouth_roundover import config as roundover_config
from test_native_terminating_arc import config as arc_config
from test_native_throat_adapter import config as adapter_config
from test_standalone_source_body import config as source_config


def identity_cases():
    controls = controls_config()
    controls['scale'] = .01
    roundover = roundover_config()
    roundover['profile'].update(L_mm=12., r0_mm=1.27)
    roundover['mesh'].update(wall_thickness_mm=.2, mouth_roundover_radius_mm=1.)
    # Small finished dimensions keep the native facet certificates affordable;
    # increasing mm resolutions alone cannot coarsen their hard error bounds.
    cases = [
        ('OSSE-ADAPTER', adapter_config()),
        ('OSSE-ADAPTER-CONTROLS', controls),
        ('OSSE-ADAPTER-AXIAL', adapter_axial_config()),
        ('OSSE-AXIAL', axial_config()),
        ('OSSE-ARC-forward', arc_config(60.)),
        ('OSSE-ARC-returning', arc_config(100.)),
        ('OSSE-ROUNDOVER', roundover),
        ('SOURCE-DISK', source_config(size_mm=5.)),
    ]
    for name, config in cases:
        if name != 'SOURCE-DISK':
            config['mesh'].update(throat_res_mm=5., mouth_res_mm=15., rear_res_mm=10.)
    roundover['mesh'].update(
        throat_res_mm=.2, mouth_res_mm=.4, rear_res_mm=.4, max_triangles=100000,
    )
    return cases


@pytest.fixture(scope='module')
def native_base_tree(tmp_path_factory):
    if not (ROOT / '.git').exists():
        pytest.skip('Native identity gate requires a Git checkout and tag v0.2.6')
    tag = subprocess.run(
        ['git', 'rev-parse', '--verify', 'v0.2.6^{commit}'], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    if tag.returncode != 0:
        # Distinguish an absent tag from a broken/unreadable existing ref.
        absent = subprocess.run(
            ['git', 'show-ref', '--verify', '--quiet', 'refs/tags/v0.2.6'], cwd=ROOT,
            check=False,
        )
        if absent.returncode == 1:
            pytest.skip('Native identity gate requires tag v0.2.6, which is absent')
        tag.check_returncode()
    return archive_tree(tmp_path_factory, tag.stdout.strip())


@pytest.mark.parametrize('name,config', identity_cases(), ids=[name for name, _ in identity_cases()])
def test_native_single_feature_identity(native_base_tree, tmp_path, name, config):
    # All selected v0.2.6 fixtures support all five artifacts. Unexpected
    # refusals fail the worker; no exception is accepted as a baseline artifact.
    assert_mesh_and_step_bytes(native_base_tree, tmp_path, name, config, native=True)
