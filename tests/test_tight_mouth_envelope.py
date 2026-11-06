"""Tight mouth bends: the default R-OSSE, WG's Custom conversion of it, and siblings.

The JSON fixture is design_to_mesher_config(convert_design(
DesignConfig.model_validate({'formula': 'R-OSSE'}), 'fixture').design), generated
by WG's server/preview/freeform_conversion.py. WG preserves a 5 mm default;
the mesher-only default R-OSSE configuration requests 6 mm.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pytest

from hornlab_mesher import build_from_config, build_geometry_params
from hornlab_mesher.freeform import validate_outer_offset_grid
from hornlab_mesher.preview.api import build_preview_geometry
from hornlab_mesher.profile_sampling import build_point_grid_arrays
from test_offset_envelope import _raw_offset
from test_freeform_builds import _assert_closed_valid_mesh

CONVERTED = json.loads((Path(__file__).parent / 'fixtures/rosse-converted-freeform.json').read_text())
ROSSE = {'formula': 'R-OSSE', 'mode': 'freestanding'}
ROSSE_12 = {**ROSSE, 'mesh': {'wall_thickness_mm': 12}}


def corner_config(wall=12.0):
    """Corner FREEFORM whose concave 5->70 degree mitre folds at 12 mm."""
    radius, points = 12.7, [{'z': 0, 'r': 12.7, 'join': 'corner'}]
    for (z0, z1), angle in zip(((0, 5), (5, 120)), (5, 70)):
        radius += math.tan(math.radians(angle)) * (z1 - z0)
        points.append({'z': z1, 'r': radius, 'join': 'corner'})
    profile = {'points': points}
    return {'formula': 'FREEFORM', 'mode': 'freestanding',
            'profile': {'profileH': profile, 'profileV': profile,
                        'crossSections': [{'t': 0, 'shape': 'circle'}, {'t': 1, 'shape': 'circle'}]},
            'mesh': {'angularSegments': 64, 'lengthSegments': 24, 'samplingMode': 'uniform',
                     'throat_res_mm': 8, 'mouth_res_mm': 8, 'rear_res_mm': 5, 'wall_thickness_mm': wall,
                     'max_triangles': 100000, 'allow_large_mesh': True, 'source_shape': 0,
                     'scale_to_metres': False}}


def _wall(config):
    params, _, _ = build_geometry_params(config)
    return float(params['wallThickness'])


@pytest.mark.parametrize('config', [ROSSE, CONVERTED], ids=['R-OSSE', 'converted-FREEFORM'])
def test_tight_mouth_envelope_preserves_inner_and_removes_fold(config):
    # R-OSSE folds at its rolled lip. The converted wall turns at
    # wall*kappa=0.505 at the throat, past the 0.4 FREEFORM used to refuse:
    # both now get the exterior envelope instead of a fold or a refusal.
    params, _, _ = build_geometry_params(config)
    grid = build_point_grid_arrays(params)
    acoustic = build_point_grid_arrays({**params, 'wallThickness': 0.0})
    assert grid['inner_grid'].tobytes() == acoustic['inner_grid'].tobytes()
    assert grid['outer_offset_fold'] is None and grid['outer_envelope']
    validate_outer_offset_grid(grid['inner_grid'], grid['outer_grid'], full_circle=grid['full_circle'])
    assert not np.array_equal(grid['outer_grid'], _raw_offset(grid, params['wallThickness']))


@pytest.mark.parametrize('config', [ROSSE, CONVERTED], ids=['R-OSSE', 'converted-FREEFORM'])
def test_tight_mouth_designs_build_closed_meshes(config, tmp_path):
    result = build_from_config(config, tmp_path / 'tight-mouth.msh')
    assert not result.metadata.get('outerOffsetFold')
    _assert_closed_valid_mesh(result, require_positive_volume=True)
    # P5: the fitted faces themselves keep Wall thickness; the envelope
    # allowance keeps the wall within about 0.1 mm of what was asked.
    certificate = result.metadata['outerEnvelopeClearance']
    assert _wall(config) <= certificate['lowerBoundMm'] <= _wall(config) + 0.11
    assert result.metadata['outerEnvelopeMeshClearance']['minimumMm'] > 0


def test_default_rosse_preview_stays_fast():
    # P1: the envelope must not move the live preview off its normal budget
    # (0.2-1.0 s before envelopes; 23-46 s with the first envelope).
    build_preview_geometry(ROSSE)  # import and cache warm-up
    start = time.perf_counter()
    build_preview_geometry(ROSSE)
    assert time.perf_counter() - start < 3.0


@pytest.mark.parametrize('config', [ROSSE, ROSSE_12, CONVERTED],
                         ids=['wall-6', 'wall-12', 'converted-FREEFORM'])
def test_preview_and_solve_agree_on_the_envelope(config, tmp_path, caplog):
    # P2: one verdict. The dense preview master used to fold its envelope at
    # the rolled lip and refuse (12 mm) while the solve built.
    with caplog.at_level(logging.WARNING, logger='hornlab_mesher'):
        preview = build_preview_geometry(config)
    assert any(surface.role.startswith('horn.outer') for surface in preview.surfaces)
    assert not [r for r in caplog.records if 'self-intersects' in r.getMessage()]
    result = build_from_config(config, tmp_path / 'agree.msh')
    assert not result.metadata.get('outerOffsetFold')
    assert result.metadata['outerEnvelopeClearance']['lowerBoundMm'] >= _wall(config)


# Lips curling back in both radius and z (tmax > 1). tmax 1.2 and 1.5 and a60
# 1.3 and 1.5 failed their solve after a clean preview (nonmanifold edges, or
# the throat and rear caps filled across a mouth ring below them); the others
# built a folded wall.
CURLS = {
    'tmax1.1': {'tmax': 1.1}, 'tmax1.2': {'tmax': 1.2}, 'tmax1.3': {'tmax': 1.3},
    'tmax1.5': {'tmax': 1.5},
    'a60-tmax1.2': {'tmax': 1.2, 'a': 60, 'r': 0.5, 'b': 0.4},
    'a60-tmax1.3': {'tmax': 1.3, 'a': 60, 'r': 0.5, 'b': 0.4},
    'a60-tmax1.5': {'tmax': 1.5, 'a': 60, 'r': 0.5, 'b': 0.4},
}


def _meridian_clearance(inner, outer):
    """Exact 2D distance from each outer station to the inner meridian polyline."""
    from hornlab_mesher.preview.fidelity import _point_segment_squared

    def meridian(grid):
        return np.stack((np.linalg.norm(grid[0, :, :2], axis=1), grid[0, :, 2]), axis=1)
    m, o = meridian(inner), meridian(outer)
    return np.sqrt(np.min(_point_segment_squared(o[:, None], m[None, :-1], m[None, 1:]), axis=1))


@pytest.mark.parametrize('profile', CURLS.values(), ids=CURLS.keys())
def test_curled_rosse_gets_its_meridian_envelope(profile):
    # A curl has no axial or radial ray schedule, but on a surface of
    # revolution the exterior is the meridian's trimmed one-sided offset.
    params, _, _ = build_geometry_params({**ROSSE, 'profile': profile})
    grid = build_point_grid_arrays(params)
    acoustic = build_point_grid_arrays({**params, 'wallThickness': 0.0})
    assert grid['inner_grid'].tobytes() == acoustic['inner_grid'].tobytes()
    assert grid['outer_offset_fold'] is None and grid['outer_envelope']
    validate_outer_offset_grid(grid['inner_grid'], grid['outer_grid'], full_circle=grid['full_circle'])
    clearance = _meridian_clearance(grid['inner_grid'], grid['outer_grid'])
    wall = params['wallThickness']
    # Chords between the 5-degree arc points of a vertex ball sag 0.006 mm.
    assert clearance.min() >= wall + 0.1 - 0.01
    # Stations stay on the envelope: none is left deeper than the kink's
    # sampling allows (the raw normal offset sat about 2 mm inside the wall).
    assert np.median(clearance) <= wall + 0.1 + 1e-6


@pytest.mark.parametrize('profile', CURLS.values(), ids=CURLS.keys())
def test_curled_rosse_preview_and_solve_agree(profile, tmp_path, caplog):
    config = {**ROSSE, 'profile': profile}
    with caplog.at_level(logging.WARNING, logger='hornlab_mesher'):
        preview = build_preview_geometry(config)
    assert any(surface.role.startswith('horn.outer') for surface in preview.surfaces)
    assert not [r for r in caplog.records if 'self-intersects' in r.getMessage()
                or 'normal outer offset' in r.getMessage()]
    result = build_from_config(config, tmp_path / 'curl.msh')
    assert not result.metadata.get('outerOffsetFold')
    _assert_closed_valid_mesh(result, require_positive_volume=True)
    certificate = result.metadata['outerEnvelopeClearance']
    assert certificate['certified'] and certificate['lowerBoundMm'] >= _wall(config)


def test_corner_freeform_envelope_is_certified(tmp_path):
    # P4: the mitred grid's own envelope must reach the certificate.
    start = time.perf_counter()
    result = build_from_config(corner_config(), tmp_path / 'corner.msh')
    elapsed = time.perf_counter() - start
    assert result.metadata['outerEnvelope'] is True
    assert result.metadata['outerEnvelopeClearance']['lowerBoundMm'] >= 12.0
    _assert_closed_valid_mesh(result, require_positive_volume=True)
    assert elapsed < 60.0


def test_step_export_certifies_the_fitted_faces(tmp_path, monkeypatch):
    import hornlab_mesher.wall_distance as distance
    from hornlab_mesher.cad import write_step_from_config

    seen = []
    certify = distance.certify_outer_envelope
    monkeypatch.setattr(distance, 'certify_outer_envelope',
                        lambda built, wall: seen.append(certify(built, wall)) or seen[-1])
    path, _ = write_step_from_config(ROSSE, tmp_path / 'rosse.step')
    assert path.exists() and len(seen) == 1 and seen[0]['lowerBoundMm'] >= 6.0


def test_measured_shortfall_rebuilds_and_never_fails(tmp_path, monkeypatch, caplog):
    # P0: a shortfall rebuilds the wall with more clearance; if it persists
    # the design still builds and exports, with the shortfall recorded.
    import hornlab_mesher.wall_distance as distance
    from hornlab_mesher.cad import write_step_from_config
    from hornlab_mesher.offset_envelope import envelope_allowance_mm

    seen = []

    def short(*args, **kwargs):
        seen.append(envelope_allowance_mm(6.0))
        return {'lowerBoundMm': 5.9, 'method': 'forced'}
    monkeypatch.setattr(distance, 'fitted_face_clearance', short)
    with caplog.at_level(logging.WARNING, logger='hornlab_mesher'):
        result = build_from_config(ROSSE, tmp_path / 'short.msh')
    assert result.metadata['outerEnvelopeClearance']['certified'] is False
    attempts = sorted(set(seen))
    assert len(attempts) == 1 + distance.ENVELOPE_REBUILDS and attempts[1] > attempts[0] + 0.1
    assert any('shortfall recorded' in r.getMessage() for r in caplog.records)
    path, _ = write_step_from_config(ROSSE, tmp_path / 'short.step')
    assert path.exists()


@pytest.mark.parametrize('wall', [2, 3, 4, 5, 8, 10])
def test_rosse_walls_build_certified(wall, tmp_path):
    # P0: the default R-OSSE at WG's 5 mm (and 4 mm) measured short of the
    # fixed allowance; every wall must build, certified by a rebuild.
    config = {**ROSSE, 'mesh': {'wall_thickness_mm': wall}}
    result = build_from_config(config, tmp_path / f'wall-{wall}.msh')
    certificate = result.metadata.get('outerEnvelopeClearance')
    if certificate is not None:
        assert certificate['certified'] and certificate['lowerBoundMm'] >= wall
    _assert_closed_valid_mesh(result, require_positive_volume=True)


@pytest.mark.parametrize('profile', [{'k': 2}, {'r': 0.3}, {'b': 0.5}, {'m': 0.9}],
                         ids=['k2', 'r0.3', 'b0.5', 'm0.9'])
def test_rosse_variants_build_certified(profile, tmp_path):
    result = build_from_config({**ROSSE, 'profile': profile}, tmp_path / 'variant.msh')
    certificate = result.metadata.get('outerEnvelopeClearance')
    if certificate is not None:
        assert certificate['certified'] and certificate['lowerBoundMm'] >= 6.0


@pytest.mark.parametrize('quadrants', [12, 1])
def test_symmetry_sector_preview_stays_fast(quadrants):
    # P1: a sector of a surface of revolution shares the full circle's
    # one-row envelope (21 s / 9 s on the dense master before).
    config = {**ROSSE, 'mesh': {'quadrants': quadrants}}
    build_preview_geometry(config)
    start = time.perf_counter()
    build_preview_geometry(config)
    assert time.perf_counter() - start < 3.0


AZIMUTH_R = {**ROSSE, 'profile': {'R': '140 + 20*cos(2*p)'}}


def test_azimuth_varying_preview_is_fast_and_agrees_with_solve(tmp_path, caplog):
    # P1: the envelope is drawn on the preview's own display grid, not its
    # dense master (27 s there, ending in a fold warning the solve lacks).
    build_preview_geometry(AZIMUTH_R)
    with caplog.at_level(logging.WARNING, logger='hornlab_mesher'):
        start = time.perf_counter()
        preview = build_preview_geometry(AZIMUTH_R)
        elapsed = time.perf_counter() - start
    assert elapsed < 4.0
    assert any(surface.role.startswith('horn.outer') for surface in preview.surfaces)
    assert not [r for r in caplog.records if 'self-intersects' in r.getMessage()]
    result = build_from_config(AZIMUTH_R, tmp_path / 'azimuth.msh')
    assert not result.metadata.get('outerOffsetFold')
    assert result.metadata['outerEnvelopeClearance']['lowerBoundMm'] >= 6.0


def _counting_certificates(monkeypatch):
    import hornlab_mesher.wall_distance as distance
    calls = []
    certify = distance.certify_outer_envelope

    def counted(built, wall):
        try:
            result = certify(built, wall)
        except distance.EnvelopeShortfall as exc:
            calls.append(('short', exc.shortfall_mm))
            raise
        calls.append(('ok', result.get('lowerBoundMm')))
        return result
    monkeypatch.setattr(distance, 'certify_outer_envelope', counted)
    return calls


def _identity(geometry):
    """The geometry's field-wise identity, as WG hashes it (dataclass fields)."""
    import dataclasses
    import hashlib

    def plain(value):
        if dataclasses.is_dataclass(value):
            return plain(dataclasses.asdict(value))
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if hasattr(value, 'tolist'):
            return plain(value.tolist())
        return getattr(value, 'value', value) if not isinstance(value, (int, float, str)) else value
    return hashlib.sha256(json.dumps(plain(geometry), sort_keys=True, default=str).encode()).hexdigest()


@pytest.mark.parametrize('config,wall', [
    ({**ROSSE, 'mesh': {'wall_thickness_mm': 5}}, 5.0),
    (corner_config(), 12.0),
], ids=['R-OSSE-5mm', 'corner-FREEFORM'])
def test_wglink_export_carries_a_certified_envelope(config, wall, tmp_path, monkeypatch):
    # Round-3 P1: WG resolves the geometry and hands it to write_wglink; the
    # export certifies (and rebuilds) the faces it writes. Round 4: the
    # resolve itself stays cheap and its identity never depends on that.
    from hornlab_mesher import write_wglink
    from hornlab_mesher.config_builder import resolve_geometry

    calls = _counting_certificates(monkeypatch)
    resolved = resolve_geometry(config)
    assert calls == []  # identity-only resolves measure nothing
    identity = _identity(resolved.geometry)
    product = write_wglink(resolved.geometry, tmp_path / 'horn.wglink')
    assert product.manifest_path.exists()
    assert calls and calls[-1][0] == 'ok' and calls[-1][1] >= wall
    if wall == 5.0:
        # WG's default wall measured short at the base allowance: rebuilt.
        assert calls[0][0] == 'short'
    assert _identity(resolved.geometry) == identity
    assert _identity(resolve_geometry(config).geometry) == identity


def test_identity_is_the_same_on_every_path(tmp_path):
    # Preview identity, solve, STEP and WGLink all hash resolve_geometry's
    # geometry; a rebuild changes only the faces built from it.
    from hornlab_mesher.cad import write_step_from_config
    from hornlab_mesher.config_builder import resolve_geometry

    config = {**ROSSE, 'mesh': {'wall_thickness_mm': 5}}
    before = _identity(resolve_geometry(config).geometry)
    result = build_from_config(config, tmp_path / 'solve.msh')
    assert result.metadata['outerEnvelopeClearance']['lowerBoundMm'] >= 5.0
    write_step_from_config(config, tmp_path / 'part.step')
    assert _identity(resolve_geometry(config).geometry) == before


@pytest.mark.parametrize('config', [ROSSE, {**ROSSE, 'mesh': {'wall_thickness_mm': 5}},
                                    {**ROSSE, 'profile': {'k': 2}}, AZIMUTH_R],
                         ids=['6mm', '5mm', 'k2', 'azimuth-R'])
def test_identity_resolves_stay_cheap(config):
    # Round 4: WG's identity, freshness and front-view resolves run on
    # interactive paths; they must cost what they did before envelopes.
    from hornlab_mesher.config_builder import resolve_geometry

    resolve_geometry(config)
    start = time.perf_counter()
    for _ in range(5):
        resolve_geometry(config)
    assert (time.perf_counter() - start) / 5 < 0.25


def test_nested_rebuilds_inherit_the_outer_clearance():
    from hornlab_mesher.offset_envelope import envelope_allowance_mm, extra_envelope_allowance
    from hornlab_mesher.wall_distance import with_envelope_retries

    with extra_envelope_allowance(0.3):
        assert with_envelope_retries(lambda: envelope_allowance_mm(6.0)) == pytest.approx(
            envelope_allowance_mm(0.0) + 0.1 + 0.3)


def test_a_failed_measurement_never_blocks_a_solve(tmp_path, monkeypatch):
    # Round 4: a fault inside the certificate is recorded; the solve carries
    # on and never reports a STEP export failure.
    import hornlab_mesher.wall_distance as distance
    from hornlab_mesher.errors import InternalInvariantError

    def broken(*args, **kwargs):
        raise InternalInvariantError("forced sampling fault")
    monkeypatch.setattr(distance, 'fitted_face_clearance', broken)
    result = build_from_config(ROSSE, tmp_path / 'solve.msh')
    record = result.metadata['outerEnvelopeClearance']
    assert record['certified'] is False and 'forced sampling fault' in record['unmeasured']


@pytest.mark.parametrize('meridian,radius,side', [
    ([[50, 0], [50.232, 7.757], [57.089, 5.084], [49.38, -0.629], [45.634, -2.675]], 5.8195, 1.0),
    ([[50, 0], [45.281, 4.002], [39.727, 5.921], [40.031, 4.245], [40.045, 2.993],
      [44.858, 2.049], [45.716, -0.034]], 6.9925, -1.0),
], ids=['throat-station-inside', 'mouth-tip-inside'])
def test_curled_envelope_refuses_stations_inside_the_wall(meridian, radius, side):
    # Review P1: the walk splits only at self-crossings, so an end station could
    # sit inside another part's reach. Such an envelope must fall back (raise).
    from hornlab_mesher.offset_envelope import _curled_meridian_envelope

    inner = np.asarray(meridian, float)
    steps = np.diff(inner, axis=0)
    normal = side * np.stack((-steps[:, 1], steps[:, 0]), axis=1) / np.linalg.norm(steps, axis=1)[:, None]
    raw = inner + radius * np.vstack((normal, normal[-1:]))
    with pytest.raises(ValueError, match='inside the wall'):
        _curled_meridian_envelope(inner, raw, radius)
