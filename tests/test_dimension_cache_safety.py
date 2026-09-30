"""Measurement ownership, threaded reuse and failure isolation regressions."""
import copy
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor

import pytest

import hornlab_mesher.preview.dimensions as measurement
import hornlab_mesher.preview.api as api
from hornlab_mesher.preview import PreviewOptionsV1, build_preview_geometry
from test_preview_api import OSSE_FREESTANDING, ROSSE_ENCLOSURE
from test_preview_dimensions import _assert_preview_equal


@pytest.fixture(autouse=True)
def clear_cache():
    measurement._DIMENSIONS_CACHE.clear()
    yield
    measurement._DIMENSIONS_CACHE.clear()


def test_input_mutation_during_measurement_cannot_poison_original_design(monkeypatch):
    config = copy.deepcopy(OSSE_FREESTANDING)
    expected = measurement.canonical_dimensions(config)
    entered, resume = threading.Event(), threading.Event()
    real = measurement.canonical_dimensions
    def delayed(snapshot):
        entered.set()
        assert resume.wait(5)
        return real(snapshot)
    monkeypatch.setattr(measurement, 'canonical_dimensions', delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(build_preview_geometry, config, PreviewOptionsV1(lod='fine'))
        assert entered.wait(5)
        config['profile']['L_mm'] = 240
        resume.set()
        result = future.result(timeout=5)
    assert result.metadata['dimensions_mm'] == expected
    assert measurement.dimension_metadata(OSSE_FREESTANDING, 'coarse')['dimensions_mm'] == expected


def test_four_settled_requests_share_one_resolve_and_owned_results(monkeypatch):
    barrier = threading.Barrier(4)
    calls = []
    def measure(snapshot):
        calls.append(snapshot)
        # Leave enough time for the other requests to enter the cache path.
        threading.Event().wait(0.1)
        return {'mouth_opening': [1, 2], 'horn_overall': [3, 4, 5]}
    monkeypatch.setattr(measurement, 'canonical_dimensions', measure)
    def request():
        barrier.wait(timeout=5)
        return measurement.dimension_metadata(OSSE_FREESTANDING, 'fine')
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: request(), range(4)))
    assert len(calls) == 1
    results[0]['dimensions_mm']['mouth_opening'][0] = 999
    assert all(result['dimensions_mm']['mouth_opening'] == [1, 2] for result in results[1:])


@pytest.mark.parametrize('warm', [False, True])
def test_cache_copy_failure_preserves_complete_preview(warm, monkeypatch):
    if warm:
        measurement.dimension_metadata(OSSE_FREESTANDING, 'fine')
    real = copy.deepcopy
    def fail_current(value, *args, **kwargs):
        if isinstance(value, dict) and value.get('dimensions_status') == 'current':
            raise MemoryError('cache-copy')
        return real(value, *args, **kwargs)
    monkeypatch.setattr(measurement.copy, 'deepcopy', fail_current)
    options = PreviewOptionsV1(lod='fine')
    preview = build_preview_geometry(OSSE_FREESTANDING, options)
    assert preview.metadata['dimensions_status'] == 'unavailable'
    assert preview.metadata['dimensions_mm'] is None
    monkeypatch.setattr(api, 'dimension_metadata', lambda *args: {})
    _assert_preview_equal(preview, build_preview_geometry(OSSE_FREESTANDING, options))


def test_measurement_does_not_repeat_enclosure_clamp_warnings(caplog):
    config = copy.deepcopy(ROSSE_ENCLOSURE)
    config['enclosure'].update(depth_mm=1, edge_mm=0)
    build_preview_geometry(config, PreviewOptionsV1(lod='fine'))
    for name in ['enc_depth']:
        records = [record for record in caplog.records if name in record.getMessage()]
        assert len(records) == 1
        assert 'viewport' in records[0].getMessage()


@pytest.mark.parametrize('scale', [1, 2])
@pytest.mark.parametrize('aliases', [False, True])
def test_requested_morph_size_is_distinct_from_effective_no_shrink_size(scale, aliases):
    config = copy.deepcopy(OSSE_FREESTANDING)
    config['Scale'] = scale
    config['mesh']['quadrants'] = '1'
    config['morph'] = ({'morph_target': 1, 'morph_width_mm': 320, 'morph_height_mm': 240, 'morph_corner_mm': 0}
                       if aliases else {'morphTarget': 1, 'morphWidth': 320, 'morphHeight': 240, 'morphCorner': 0})
    result = measurement.dimension_metadata(config, 'fine')
    assert result['dimensions_requested_mm'] == {'mouth_opening': [320 * scale, 240 * scale]}
    assert result['dimensions_mm']['mouth_opening'] == pytest.approx([348.5794722023637 * scale] * 2)
    assert measurement.dimension_metadata(config, 'coarse') == result


def test_inactive_morph_has_no_requested_size():
    config = copy.deepcopy(OSSE_FREESTANDING)
    config['morph'] = {'morphTarget': 0, 'morphWidth': 320, 'morphHeight': 240}
    assert 'dimensions_requested_mm' not in measurement.dimension_metadata(config, 'fine')


def test_mutation_before_first_render_read_preserves_surfaces_and_numbers(monkeypatch):
    config = copy.deepcopy(OSSE_FREESTANDING)
    options = PreviewOptionsV1(lod='fine')
    expected = build_preview_geometry(config, options)
    measurement._DIMENSIONS_CACHE.clear()
    entered, resume = threading.Event(), threading.Event()
    real = api.build_geometry_params

    seen = []

    def delayed(snapshot):
        seen.append(snapshot)
        entered.set()
        assert resume.wait(10)
        return real(snapshot)

    monkeypatch.setattr(api, 'build_geometry_params', delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(build_preview_geometry, config, options)
        try:
            assert entered.wait(10)
            config['profile']['L_mm'] = 240
        finally:
            resume.set()
        preview = pending.result(timeout=10)
    _assert_preview_equal(expected, preview)
    assert seen[0] is not config
    assert preview.metadata['dimensions_mm'] == expected.metadata['dimensions_mm']
    assert measurement.dimension_metadata(OSSE_FREESTANDING, 'coarse')['dimensions_mm'] == expected.metadata['dimensions_mm']


def test_snapshot_failure_keeps_renderer_config_handling(monkeypatch):
    options = PreviewOptionsV1(lod='fine')
    expected = build_preview_geometry(OSSE_FREESTANDING, options)
    real = api.copy.deepcopy

    failed = False

    def fail_snapshot(value, *args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise MemoryError('snapshot')
        return real(value, *args, **kwargs)

    monkeypatch.setattr(api.copy, 'deepcopy', fail_snapshot)
    preview = build_preview_geometry(OSSE_FREESTANDING, options)
    _assert_preview_equal(expected, preview)
    assert preview.metadata['dimensions_status'] == 'unavailable'
    assert preview.metadata['dimensions_error'] == 'snapshot'


@pytest.mark.parametrize('operation', ['publication', 'eviction'])
@pytest.mark.parametrize('exception', [MemoryError, KeyboardInterrupt, SystemExit])
def test_cache_publication_failure_releases_owner_waiters_and_allows_retry(operation, exception, monkeypatch):
    class BrokenCache(OrderedDict):
        broken = True

        def __setitem__(self, key, value):
            if self.broken and operation == 'publication':
                raise exception('cache-publication')
            return super().__setitem__(key, value)

        def popitem(self, *args, **kwargs):
            if self.broken:
                raise exception('cache-eviction')
            return super().popitem(*args, **kwargs)

    cache = BrokenCache()
    if operation == 'eviction':
        for key in range(measurement._CACHE_LIMIT):
            OrderedDict.__setitem__(cache, str(key), {})
    options = PreviewOptionsV1(lod='fine')
    expected = build_preview_geometry(OSSE_FREESTANDING, options)
    measurement._DIMENSIONS_CACHE.clear()
    entered, resume, waiting = threading.Event(), threading.Event(), threading.Event()
    waiter_lock = threading.Lock()
    waiter_count = 0
    calls = []
    real = measurement.canonical_dimensions

    class ObservedFuture(Future):
        def result(self, *args, **kwargs):
            nonlocal waiter_count
            with waiter_lock:
                waiter_count += 1
                if waiter_count == 3:
                    waiting.set()
            return super().result(*args, **kwargs)

    def delayed(snapshot):
        calls.append(snapshot)
        entered.set()
        assert resume.wait(10)
        return real(snapshot)

    def request():
        try:
            return build_preview_geometry(OSSE_FREESTANDING, options)
        except BaseException as exc:
            return exc

    monkeypatch.setattr(measurement, '_DIMENSIONS_CACHE', cache)
    monkeypatch.setattr(measurement, 'Future', ObservedFuture)
    monkeypatch.setattr(measurement, 'canonical_dimensions', delayed)
    with ThreadPoolExecutor(max_workers=4) as pool:
        owner = pool.submit(request)
        try:
            assert entered.wait(10)
            waiters = [pool.submit(request) for _ in range(3)]
            assert waiting.wait(10)
        finally:
            resume.set()
        results = [future.result(timeout=10) for future in [owner, *waiters]]
    assert len(calls) == 1
    assert not measurement._IN_FLIGHT
    if exception is MemoryError:
        for preview in results:
            _assert_preview_equal(expected, preview)
            assert preview.metadata['dimensions_status'] == 'unavailable'
            assert preview.metadata['dimensions_mm'] is None
            assert preview.metadata['dimensions_error'] == f'cache-{operation}'
        assert [p.metadata['dimensions_error'] for p in results] == [results[0].metadata['dimensions_error']] * 4
    else:
        assert all(isinstance(result, exception) for result in results)
    cache.broken = False
    if exception is MemoryError:
        assert measurement.dimension_metadata(OSSE_FREESTANDING, 'coarse')['dimensions_status'] == 'pending'
    monkeypatch.setattr(measurement, 'canonical_dimensions', real)
    retry = build_preview_geometry(OSSE_FREESTANDING, options)
    _assert_preview_equal(expected, retry)
    assert retry.metadata['dimensions_status'] == 'current'
