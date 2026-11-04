"""Reuse the dense inversion table without sharing mutable sampled maps."""

import numpy as np
import pytest

import hornlab_mesher.profile_sampling as sampling


def _uncached_map(steps, controls):
    (x1, y1), (x2, y2) = controls
    s = np.linspace(0.0, 1.0, 100001)
    one_minus = 1.0 - s
    bx = 3.0 * x1 * s * one_minus**2 + 3.0 * x2 * s * s * one_minus + s**3
    by = 3.0 * y1 * s * one_minus**2 + 3.0 * y2 * s * s * one_minus + s**3
    out = np.interp(np.linspace(0.0, 1.0, max(1, steps) + 1), bx, by)
    out[0], out[-1] = 0.0, 1.0
    return out


def test_refinement_builds_one_dense_table_for_different_station_counts(monkeypatch):
    cached_table = getattr(sampling, "_bezier_zmap_table", None)
    if cached_table is not None:
        cached_table.cache_clear()
    original = np.linspace
    dense_calls = []

    def counted(start, stop, num=50, *args, **kwargs):
        if num == 100001:
            dense_calls.append(num)
        return original(start, stop, num, *args, **kwargs)

    monkeypatch.setattr(np, "linspace", counted)
    for steps in (8, 16, 32, 47, 64):
        sampling._ath_default_zmap(steps)
    assert len(dense_calls) == 1


@pytest.mark.parametrize("steps", [0, 1, 8, 9, 32, 47, 128, 513])
@pytest.mark.parametrize("controls", [sampling._ATH_OSSE_ZMAP_BEZIER,
                                     ((0.25, 0.15), (0.75, 0.85))])
def test_cached_table_retains_exact_interpolation_and_independent_outputs(steps, controls):
    expected = _uncached_map(steps, controls)
    first = sampling._bezier_zmap(steps, controls)
    assert first.tobytes() == expected.tobytes()
    first[:] = -1.0
    second = sampling._bezier_zmap(steps, controls)
    assert second.tobytes() == expected.tobytes()
    assert not np.shares_memory(first, second)
