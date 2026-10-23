"""Byte identity for every legacy native marker against the v0.2.6 release."""
from __future__ import annotations

import pytest
from identity_harness import assert_mesh_and_step_bytes, release_tree

# Literal fixtures: editing a feature's own test module must not move the gate.
# Small finished dimensions keep the native facet certificates affordable;
# increasing mm resolutions alone cannot coarsen their hard error bounds.
_ADAPTER = {'mode': 'authored', 'contract_revision': 1, 'driver_exit_diameter_mm': 25.4,
            'exit_half_angle_deg': 10.0, 'length_mm': 40.0, 'join_t': 0.137,
            'driver_handle_mm': 10.0, 'body_handle_mm': 15.0}
_RES = {'throat_res_mm': 5.0, 'mouth_res_mm': 15.0, 'rear_res_mm': 10.0}


def _arc(end):
    return {'formula': 'OSSE-ARC', 'mode': 'bare', 'scale': 1.0,
            'profile': {'L_mm': 120.0, 'r0_mm': 12.7, 'a_deg': 40.0, 'a0_deg': 15.5, 'k': 1.0, 's': 0},
            'terminating_arc': {'contract_revision': 1, 'join_t': 0.05, 'end_tangent_deg': end},
            'mesh': {'wall_thickness_mm': 0, 'vertical_offset_mm': 7.0, 'allow_large_mesh': True, **_RES},
            'source': {'source_shape': 0}}


def identity_cases():
    return [
        ('OSSE-ADAPTER', {
            'formula': 'OSSE-ADAPTER', 'mode': 'bare',
            'profile': {'L': 120.0, 'r0': 12.7, 'a': 40.0, 'a0': 15.5, 'k': 1.0, 's': 0.0},
            'source': {'source_shape': 0},
            'mesh': {'angular_segments': 32, 'length_segments': 13, **_RES},
            'throat_adapter': dict(_ADAPTER)}),
        ('OSSE-ADAPTER-CONTROLS', {
            'formula': 'OSSE-ADAPTER-CONTROLS', 'mode': 'bare', 'scale': 0.01,
            'profile': {'L_mm': 137, 'r0_mm': 17, 'a_deg': 37, 'a0_deg': 6, 'k': 1.25, 's': 0},
            'throat_adapter': {'mode': 'controls', 'contract_revision': 1, 'join_t': 0,
                               'control_points_mm': [[-31.7, 16.2], [-22.1, 16.9], [-9.3, 20.4]]},
            'source': {'source_shape': 0},
            'mesh': {'wall_thickness_mm': 0, **_RES, 'max_triangles': 250000,
                     'allow_large_mesh': True, 'scale_to_metres': False}}),
        ('OSSE-ADAPTER-AXIAL', {
            'formula': 'OSSE-ADAPTER-AXIAL', 'scale': 2.0, 'axial_scale': 1.0, 'vertical_offset_mm': 7.0,
            'profile': {'L_mm': 120.0, 'r0_mm': 12.7, 'a_deg': 40.0, 'a0_deg': 15.5, 'k': 1.0},
            'throat_adapter': dict(_ADAPTER), 'mesh': dict(_RES)}),
        ('OSSE-AXIAL', {
            'formula': 'OSSE-AXIAL', 'axial_scale': 1.2, 'scale': 2,
            'profile': {'L_mm': 120, 'r0_mm': 18, 'a_deg': 40, 'a0_deg': 8, 'k': 1, 's': 0},
            'mesh': {'wall_thickness_mm': 0, 'vertical_offset_mm': 7, **_RES},
            'source': {'source_shape': 0}}),
        ('OSSE-ARC-forward', _arc(60.0)),
        ('OSSE-ARC-returning', _arc(100.0)),
        ('OSSE-ROUNDOVER', {
            'formula': 'OSSE-ROUNDOVER', 'mode': 'freestanding',
            'profile': {'L_mm': 12.0, 'r0_mm': 1.27, 'a_deg': 45, 'a0_deg': 10, 'k': 1, 's': 0},
            'mesh': {'wall_thickness_mm': 0.2, 'mouth_roundover_radius_mm': 1.0, 'scale_to_metres': False,
                     'throat_res_mm': 0.2, 'mouth_res_mm': 0.4, 'rear_res_mm': 0.4, 'max_triangles': 100000}}),
        ('SOURCE-DISK', {
            'formula': 'SOURCE-DISK', 'mode': 'standalone-source',
            'source_body': {'radius_mm': 18.0, 'outer_radius_mm': 22.0, 'depth_mm': 8.0},
            'mesh': {'size_mm': 5.0}}),
    ]


@pytest.fixture(scope='module')
def native_base_tree(tmp_path_factory):
    return release_tree(tmp_path_factory, 'v0.2.6')


@pytest.mark.parametrize('name,config', identity_cases(), ids=[name for name, _ in identity_cases()])
def test_native_single_feature_identity(native_base_tree, tmp_path, name, config):
    # All selected v0.2.6 fixtures support all five artifacts. Unexpected
    # refusals fail the worker; no exception is accepted as a baseline artifact.
    assert_mesh_and_step_bytes(native_base_tree, tmp_path, name, config, native=True)
