"""Reuse physical enclosure resolution only for equivalent dimension inputs."""

import copy
from pathlib import Path

import pytest
from test_point_grid_contract import rectangular_enclosure
from test_preview_enclosure_edges import sampling_sensitive_enclosure

from hornlab_mesher import config_builder
from hornlab_mesher.preview import api, dimensions


@pytest.fixture(autouse=True)
def clear_dimensions_cache():
    dimensions._DIMENSIONS_CACHE.clear()
    yield
    dimensions._DIMENSIONS_CACHE.clear()


def track_resolutions(monkeypatch):
    calls = []
    original = config_builder.resolve_geometry

    def track(config, *args, **kwargs):
        result = original(config, *args, **kwargs)
        calls.append(result)
        return result

    for module in (config_builder, api, dimensions):
        monkeypatch.setattr(module, 'resolve_geometry', track)
    return calls


@pytest.mark.parametrize('lod', ['coarse', 'fine', 'inspection'])
@pytest.mark.parametrize('hidden', [False, True])
def test_full_flush_preview_resolves_physical_geometry_once(lod, hidden, monkeypatch):
    config = sampling_sensitive_enclosure()
    expected = dimensions.canonical_dimensions(config)
    calls = track_resolutions(monkeypatch)
    options = api.PreviewOptionsV1(
        lod=lod, include_curvature=False, include_enclosure=not hidden,
    )
    preview = api.build_preview_geometry(config, options)
    assert len(calls) == 1
    if lod == 'coarse':
        assert preview.metadata['dimensions_status'] == 'pending'
    else:
        assert preview.metadata['dimensions_status'] == 'current'
        assert preview.metadata['dimensions_mm'] == expected
    assert preview.surfaces


@pytest.mark.parametrize('variant', ['reduced', 'placed', 'baffle', 'preserved'])
def test_dimension_reuse_keeps_distinct_physical_inputs_separate(variant, monkeypatch):
    config = sampling_sensitive_enclosure()
    if variant == 'reduced':
        config = rectangular_enclosure(margin=0, quadrants='1')
    elif variant == 'placed':
        config['mesh']['verticalOffset'] = 7.25
    elif variant == 'baffle':
        config['enclosure'].update({f'space_{s}': .1 for s in 'lrtb'})
    else:
        config['mesh']['preserve_grid'] = True
    original_config = copy.deepcopy(config)
    expected = dimensions.canonical_dimensions(config)
    calls = track_resolutions(monkeypatch)
    preview = api.build_preview_geometry(
        config, api.PreviewOptionsV1(lod='fine', include_curvature=False),
    )
    assert len(calls) == (3 if variant in ('reduced', 'placed') else 1)
    assert preview.metadata['dimensions_mm'] == expected
    assert config == original_config


def test_shared_enclosure_bounds_still_run_the_existing_validator(monkeypatch):
    from hornlab_mesher.builders import enclosure

    config = sampling_sensitive_enclosure()
    resolved = config_builder.resolve_geometry(config)
    expected = config_builder.resolve_enclosure_bounds(config, warn_prefix=None)
    original = enclosure.enclosure_box_bounds
    calls = []

    def validate(points, box, **kwargs):
        assert points is resolved.geometry.inner_points
        assert box is resolved.geometry.enclosure
        calls.append(kwargs)
        return original(points, box, **kwargs)

    monkeypatch.setattr(enclosure, 'enclosure_box_bounds', validate)
    resolutions = track_resolutions(monkeypatch)
    assert config_builder.resolve_enclosure_bounds(
        config, resolved=resolved, warn_prefix=None,
    ) == expected
    assert not resolutions
    assert len(calls) == 1
    assert calls[0]['warn_prefix'] is None


@pytest.mark.parametrize('difference', ['scale', 'retained_construction', 'transient_data'])
def test_measurement_rejects_different_resolved_construction(difference, monkeypatch):
    if difference in ('scale', 'transient_data'):
        config = sampling_sensitive_enclosure()
        other = copy.deepcopy(config)
        if difference == 'scale':
            other['scale'] = 2
        else:
            other['_transient_construction'] = object()
    else:
        from hornlab_mesher.config_parser import parse_text_config

        fixture = (Path(__file__).parent / 'fixtures/ath_import/ath-v2026-09a/'
                   'blockers/r-m-240-360-a0-n32-l20.cfg')
        config = parse_text_config(fixture.read_text())
        config['mode'] = 'bare'
        config['mesh']['wallThickness'] = 0
        other = config_builder.retain_ath_import_construction(config)
        assert dimensions._config_key(other) != dimensions._config_key(config)
    expected = dimensions.canonical_dimensions(config)
    resolved = config_builder.resolve_geometry(other)
    bounds = config_builder.resolve_enclosure_bounds(other, resolved=resolved)
    shared = dimensions._ResolvedDimensionInput(other, resolved, bounds or {})
    calls = track_resolutions(monkeypatch)
    assert dimensions.canonical_dimensions(config, shared) == expected
    assert len(calls) == (1 if difference == 'retained_construction' else 2)
