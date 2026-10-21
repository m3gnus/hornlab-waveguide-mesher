"""Exercise legacy/default-object identity on every successful FREEFORM fixture.

This observes the normal fixture builds in test_freeform_* and test_surface_fit,
so new fixtures automatically join the gate without a second fixture catalog.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def legacy_gate_all_existing_freeform_fixtures(request, monkeypatch):
    if not (request.node.path.name.startswith("test_freeform_") or
            request.node.path.name == "test_surface_fit.py"):
        return
    from hornlab_mesher import freeform

    original_key = freeform._freeform_cache_key
    original_store = freeform._cache_store
    definitions = {}
    checking = False

    def remember(params):
        key = original_key(params)
        if not checking:
            definitions[key] = params
        return key

    def check(key, geometry):
        nonlocal checking
        original_store(key, geometry)
        if checking or key not in definitions:
            return
        params = definitions[key]
        if any(freeform._parse_anchors(params[k], p).extended
               for k, p in (("profileH", "H"), ("profileV", "V"))):
            return
        checking = True
        try:
            # These validated inputs may contain immutable maps; only copy the
            # profile blocks that are rewritten, leaving callable morphs intact.
            converted = dict(params)
            for name in ("profileH", "profileV"):
                converted[name] = dict(params[name])
                converted[name]["points"] = [
                    dict(row) if isinstance(row, Mapping) else
                    {"z": row[0], "r": row[1], "join": "smooth",
                     **({"angleDeg": row[2]} if len(row) == 3 else {})}
                    for row in params[name]["points"]]
            other = freeform.build_freeform_geometry(converted)
            assert type(other._profile_h) is freeform._PlaneSpline
            assert type(other._profile_v) is freeform._PlaneSpline
            z = np.linspace(geometry._profile_h.anchors[0, 0],
                            geometry._profile_h.anchors[-1, 0], 2001)
            for old, new in zip(geometry.evaluate_radii(z), other.evaluate_radii(z)):
                assert (old == new).all(), request.node.nodeid
            assert geometry.report() == other.report(), request.node.nodeid
        finally:
            checking = False

    monkeypatch.setattr(freeform, "_freeform_cache_key", remember)
    monkeypatch.setattr(freeform, "_cache_store", check)
