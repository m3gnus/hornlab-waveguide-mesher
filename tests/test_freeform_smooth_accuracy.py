"""The smooth FREEFORM wall follows the authored surface at every azimuth."""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from hornlab_mesher import build_geometry_params, build_from_config
from hornlab_mesher.config_builder import resolve_geometry
from hornlab_mesher.corner_wall import CornerWall
from hornlab_mesher.freeform import build_freeform_geometry
from hornlab_mesher.mesher import _dispatch_builder
from test_freeform_corner_accuracy import accuracy_config, sample_wall_accuracy
from test_freeform_corner_wall import session  # noqa: F401


SMOOTH_CASES = ['research-cone', 'cone', 'circle', 'superellipse', 'rounded-rectangle', 'quadratic',
                'morph', 'enclosure', 'baffle', 'outer-wall']


def smooth_config(name):
    if name == 'research-cone':
        angle = math.degrees(math.atan((150. - 12.7) / 600.))
        plane = {'points': [[0., 12.7], [600., 150.]],
                 'throatAngleDeg': angle, 'mouthAngleDeg': angle}
        return {'formula': 'FREEFORM', 'mode': 'bare',
                'profile': {'profileH': plane, 'profileV': plane,
                            'crossSections': [{'t': 0, 'shape': 'circle'}, {'t': 1, 'shape': 'circle'}]}}
    config = accuracy_config('endpoint' if name == 'cone' else name)
    for key in ('profileH', 'profileV'):
        profile = config['profile'][key]
        for point in profile['points']:
            point.pop('join', None)
        if name == 'cone':
            rows = profile['points']
            slope = (rows[-1]['r'] - rows[0]['r']) / (rows[-1]['z'] - rows[0]['z'])
            profile['throatAngleDeg'] = profile['mouthAngleDeg'] = math.degrees(math.atan(slope))
    return config


@pytest.mark.parametrize('name', SMOOTH_CASES)
def test_smooth_wall_sampled_accuracy(session, tmp_path, name):
    config = smooth_config(name)
    started = time.perf_counter()
    geometry = resolve_geometry(config).geometry
    resolved_s = time.perf_counter() - started
    recipe = CornerWall(build_freeform_geometry(build_geometry_params(config)[0]))
    started = time.perf_counter()
    built = _dispatch_builder(geometry)
    build_s = time.perf_counter() - started
    points = np.asarray(getattr(built, 'freeform_wall_points_mm', geometry.inner_points)).copy()
    points[..., 2] += built.axial_bounds_mm[0] - points[0, 0, 2]
    outer = None
    if name == 'outer-wall':
        # Independent dense normal-offset meridian. The corner helper's
        # piecewise-linear reference is exact only for straight corner spans.
        from hornlab_mesher.profile_sampling import _outer_offset_shell
        z = np.linspace(*built.axial_bounds_mm, 4097)
        r = recipe.profile.evaluate_radii(z)[0]
        phi = np.arange(4) * np.pi / 2
        reference = np.stack([np.c_[r * np.cos(p), r * np.sin(p), z] for p in phi])
        outer = _outer_offset_shell(reference, geometry.wall_thickness_mm, full_circle=True)
    errors = sample_wall_accuracy(geometry, built, points=points, recipe=recipe, outer_reference=outer)
    (tmp_path / f'{name}.json').write_text(json.dumps(
        {'errors': errors, 'resolve_s': resolved_s, 'occ_build_s': build_s}, indent=2))
    assert max(errors.values()) <= .01, errors


@pytest.mark.parametrize('name', SMOOTH_CASES)
def test_smooth_public_build_cost(tmp_path, name):
    started = time.perf_counter()
    result = build_from_config(smooth_config(name), tmp_path / f'{name}.msh')
    elapsed = time.perf_counter() - started
    (tmp_path / f'{name}-public.json').write_text(json.dumps(
        {'elapsed_s': elapsed, 'triangles': result.n_triangles}, indent=2))
    assert result.n_triangles > 0
    assert elapsed < 30, elapsed
