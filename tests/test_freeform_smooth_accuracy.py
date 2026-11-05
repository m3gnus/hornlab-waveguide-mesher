"""The smooth FREEFORM wall follows the authored surface at every azimuth."""
from __future__ import annotations

import json
import math
import time

import gmsh
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


def flared_mouth_config(quadrants='1234'):
    """Waveguide Generator's wall-fold fixture: ellipse meridians reaching 85/90 degrees."""
    def plane(mouth_r, mouth_angle):
        return {'points': [[0., 12.7], [120., mouth_r]], 'throatAngleDeg': 15.5, 'mouthAngleDeg': mouth_angle}
    return {'formula': 'FREEFORM', 'mode': 'freestanding',
            'profile': {'profileH': plane(140., 85.), 'profileV': plane(200., 90.),
                        'crossSections': [{'t': 0, 'shape': 'ellipse'}, {'t': 1, 'shape': 'ellipse'}],
                        'inflectionPolicy': 'warn'},
            'mesh': {'angularSegments': 40, 'lengthSegments': 20, 'cornerSegments': 4,
                     'quadrants': quadrants, 'wallThickness': 5, 'throatResolution': 35,
                     'mouthResolution': 40, 'rearResolution': 40, 'maxTriangles': 50000}}


def test_flared_mouth_wall_builds_within_budget(session):
    """A radial wall at the mouth is measured by a probe that crosses it.

    1a53231 built this design. A fixed-z radial probe runs along the flat
    mouth flange, where a micron of normal error reads as most of a millimetre,
    so 8c47e63 refined to its cap and refused.
    """
    from hornlab_mesher.corner_wall import WALL_FIT_TOLERANCE_MM, wall_probe_error
    geometry = resolve_geometry(flared_mouth_config()).geometry
    built = _dispatch_builder(geometry)
    points = built.freeform_wall_points_mm
    recipe = geometry.freeform_wall
    assert built.metadata['cornerWall']['sampledFitErrorMm'] <= WALL_FIT_TOLERANCE_MM
    worst = 0.
    for tag in built.mesh_surface_groups['inner']:
        lo, hi = map(np.asarray, gmsh.model.getParametrizationBounds(2, tag))
        u, v = np.meshgrid(np.linspace(lo[0], hi[0], 129), np.linspace(lo[1], hi[1], 257))
        xyz = np.asarray(gmsh.model.getValue(2, tag, np.c_[u.ravel(), v.ravel()].ravel())).reshape(257, 129, 3)
        for row in xyz:
            error = wall_probe_error(recipe, np.arctan2(row[:, 1], row[:, 0]), float(row[0, 2]),
                                     np.linalg.norm(row[:, :2], axis=1), points)
            worst = max(worst, float(np.max(error)))
    assert worst <= .01, worst
    # Independent of that measure: the built meridian at a sampled row lies
    # in that row's half-plane; take its point-to-segment distance to a dense
    # authored meridian polyline, densest at the flange.
    from scipy.optimize import brentq
    z0, z1 = float(points[0, 0, 2]), float(points[0, -1, 2])
    stations = np.unique(np.r_[np.linspace(z0, z1, 2001), z1 - np.geomspace(2., 1e-6, 2001), z1])
    rows = np.arctan2(points[:, 0, 1], points[:, 0, 0])
    distance, meridians = 0., 0
    for phi in rows[[0, len(rows) // 3, len(rows) // 2, -1]]:
        reference = np.c_[stations, [float(recipe.radius(phi, float(z), points)) for z in stations]]
        start, direction = reference[:-1], np.diff(reference, axis=0)
        for tag in built.mesh_surface_groups['inner']:
            lo, hi = map(np.asarray, gmsh.model.getParametrizationBounds(2, tag))
            azimuth = lambda u: math.remainder(math.atan2(*gmsh.model.getValue(2, tag, [u, hi[1]])[1::-1]) - phi, math.tau)
            if azimuth(lo[0]) * azimuth(hi[0]) > 0:
                continue
            u = brentq(azimuth, lo[0], hi[0], xtol=1e-14)
            meridians += 1
            v = np.r_[np.linspace(lo[1], hi[1], 2001), hi[1] - np.geomspace(.05, 1e-7, 401) * (hi[1] - lo[1])]
            xyz = np.asarray(gmsh.model.getValue(2, tag, np.c_[np.full_like(v, u), v].ravel())).reshape(-1, 3)
            built_zr = np.c_[xyz[:, 2], np.linalg.norm(xyz[:, :2], axis=1)]
            for point in built_zr:
                t = np.clip(np.einsum('ij,ij->i', point - start, direction)
                            / np.einsum('ij,ij->i', direction, direction), 0., 1.)
                distance = max(distance, float(np.min(np.linalg.norm(start + t[:, None] * direction - point, axis=1))))
    assert meridians >= 4 and distance <= .01, (meridians, distance)


@pytest.mark.parametrize('quadrants', ['1', '1234'])
def test_flared_mouth_public_build(tmp_path, quadrants):
    result = build_from_config(flared_mouth_config(quadrants), tmp_path / f'flared-{quadrants}.msh')
    assert result.n_triangles > 0
