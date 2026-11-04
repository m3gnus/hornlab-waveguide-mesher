"""Resolver bytes captured at 63b62c2 and the internal smooth-profile cost probe.

Each fixture freezes the control polygon, polynomial coefficients and handle
factors, including recovery cases whose factors are below one.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pytest

from hornlab_mesher import freeform as ff

CASES = json.loads((Path(__file__).parent / 'fixtures/freeform-resolution-identity.json').read_text())


@pytest.mark.parametrize('case', CASES)
def test_resolver_bytes(case):
    profile = case['profile']
    parsed = ff._parse_anchors(profile, 'H')
    angles = ff._parse_endpoint_values(profile, parsed, 'H', 15.5)
    plane = ff._resolve_plane('H', parsed, *angles, force_extended=case['extended'])
    arrays = [ff._plane_beziers(plane), plane.spline.c, plane.speed_factors]
    assert [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays] == case['hashes']


@pytest.mark.parametrize('extended', [False, True])
def test_256_anchor_internal_smooth_resolution(extended):
    # Construct the internal parsed input directly. The public 64-anchor guard
    # remains in force; this stress probe does not replace or patch that guard.
    z = np.linspace(0, 600, 256)
    profile = {'points': np.c_[z, 12.7 + z * (150-12.7)/600].tolist()}
    with pytest.raises(ValueError, match='64'):
        ff._parse_anchors(profile, 'H')
    seed = ff._parse_anchors({'points': profile['points'][:2]}, 'H')
    from dataclasses import replace
    parsed = replace(seed, points=np.asarray(profile['points']), angles_deg=np.full(256, np.nan),
                     joins=('smooth',) * 256, handles_in=(None,) * 256,
                     handles_out=(None,) * 256, angles_out_deg=(None,) * 256)
    angle = float(np.degrees(np.arctan((150-12.7)/600)))
    start = time.perf_counter()
    plane = ff._resolve_plane('H', parsed, angle, angle, force_extended=extended)
    elapsed = time.perf_counter() - start
    assert len(plane.anchors) == 256
    assert elapsed < 10, elapsed
