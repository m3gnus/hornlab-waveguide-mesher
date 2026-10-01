"""Refine interpolating rectangle mouth fits independently of mesh density."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.interpolate import make_interp_spline

from .builders._occ import _averaged_chord_parameters
from .builders.point_grid_freestanding import _restored_outer_throat_points
from .config_parser import ConfigError
from . import profile_sampling as sampling

# CAD/STEP mouth curves and solve nodes use these built-mm tolerances.
# Linear solve edges between nodes still chord corners at coarse resolution;
# these are not whole-wall or emitted-edge bounds and do not set element sizes.
_ROUNDED_TOLERANCE_MM = 0.15
_SHARP_TOLERANCE_MM = 0.6
_ANGLE_KEY = '_acoustic_mouth_angles'
_MAX_PASSES = 12
_MAX_PROFILES = 1024


def _rectangle_distance(xy, half_width, half_height, radius):
    q = np.abs(xy) - np.array([half_width - radius, half_height - radius])
    return (np.linalg.norm(np.maximum(q, 0), axis=-1)
            + np.minimum(np.max(q, axis=-1), 0) - radius)


def refine_mouth_grid(params: Mapping[str, Any], grid: dict[str, Any]):
    """Pin corners/tangencies and test the realized cubic, including sector fits.

    Preserve the accepted axial map and effective target. Recomputing implicit
    dimensions on a different angular net can change ATH's rounding/no-shrink
    result, rather than just improving the fit of the same design.
    """
    if params.get('type') == 'FREEFORM' or params.get('morphTarget') != 1:
        return grid, {}
    if any(not isinstance(params.get(k, 0), (int, float))
           for k in ('morphWidth', 'morphHeight', 'morphCorner')):
        return grid, {}
    points = np.asarray(grid['inner_points']).reshape(
        grid['grid_n_phi'], grid['grid_n_length'] + 1, 3)
    a, b = np.max(np.abs(points[:, -1, :2]), axis=0)
    scale = float(params.get('scale', 1))
    radius = min(float(params.get('morphCorner', 0)) * scale, a, b)
    if (abs(a-b) < 1e-9 and abs(radius-a) < 1e-9) or np.max(np.abs(
            _rectangle_distance(points[:, -1, :2], a, b, radius))) > 1e-6:
        return grid, {}
    tolerance = _SHARP_TOLERANCE_MM if radius == 0 else _ROUNDED_TOLERANCE_MM
    working = {**params, 'morphWidth': 2*a/scale, 'morphHeight': 2*b/scale,
               'morphAllowShrinkage': 1}
    angles = np.asarray(grid['angle_list'])
    q = np.unique(np.round(np.arctan2(np.abs(np.sin(angles)),
                                     np.abs(np.cos(angles))), 13))
    pins = [0., math.pi/2]
    if radius == 0:
        pins.append(math.atan2(b, a))
    else:
        pins.extend([math.atan2(b-radius, a), math.atan2(b, a-radius)])
        arc = np.linspace(0, math.pi/2, 7)
        pins.extend(np.arctan2(b-radius+radius*np.sin(arc),
                              a-radius+radius*np.cos(arc)))
    for pin in pins:
        q = np.r_[q[np.abs(q-pin) > 1e-10], pin]

    def deduplicate(values):
        values = np.sort(values)
        values = values[np.r_[True, np.diff(values) > 1e-10]]
        values[0], values[-1] = 0., math.pi/2
        return values

    q = deduplicate(q)
    for iteration in range(_MAX_PASSES):
        working[_ANGLE_KEY] = sampling._mirror_quadrant_angles(q).tolist()
        selected, _ = sampling._restrict_to_quadrants(
            np.asarray(working[_ANGLE_KEY]),
            sampling._normalise_quadrants(working.get('quadrants', '1234')))
        if len(selected) > _MAX_PROFILES:
            raise ConfigError(
                f'rectangular morphed mouth cubic fit needs {len(selected)} profiles; '
                f'the station limit is {_MAX_PROFILES}; coarsen throat/mouth '
                'resolution (increase their mm values) or round the mouth corners')
        candidate = sampling.build_point_grid(working)
        points = np.asarray(candidate['inner_points']).reshape(
            candidate['grid_n_phi'], candidate['grid_n_length']+1, 3)
        n = len(points)
        count = len(q)-1
        groups = [list(range(n)) + ([0] if candidate['full_circle'] else [])]
        if candidate['full_circle']:
            groups.extend([[(k*count+i) % n for i in range(count+1)]
                           for k in range(4)])
        elif n == 2*count+1:
            groups.extend([list(range(count+1)), list(range(count, n))])
        bad = []
        worst = 0.
        for cols in groups:
            patch = points[cols].transpose(1, 0, 2)
            u = _averaged_chord_parameters(patch, 1)
            curve = make_interp_spline(u, patch[-1], k=min(3, len(cols)-1))
            for j in range(len(cols)-1):
                xy = curve(np.linspace(u[j], u[j+1], 65))[:, :2]
                error = float(np.max(np.abs(_rectangle_distance(xy, a, b, radius))))
                worst = max(worst, error)
                if error > tolerance:
                    start, end = np.asarray(candidate['angle_list'])[cols[j:j+2]]
                    if end < start:
                        end += math.tau
                    mid = (start+end)/2
                    bad.append(math.atan2(abs(math.sin(mid)), abs(math.cos(mid))))
        if worst <= tolerance:
            # Clearance sizing is a discretisation-scale probe, not a fit
            # tolerance. Adding arbitrarily close fit stations must not turn
            # its minimum three-point radius into a global shell size floor.
            if grid.get('outer_points') is not None:
                baseline_inner = np.asarray(grid['inner_points']).reshape(
                    grid['grid_n_phi'], grid['grid_n_length']+1, 3)
                baseline_outer = np.asarray(grid['outer_points']).reshape(baseline_inner.shape)
                candidate['outer_clearance_points'] = _restored_outer_throat_points(
                    baseline_inner, baseline_outer,
                    wall_thickness_mm=float(params.get('wallThickness', 0)))
            return candidate, {'mouthFitToleranceMm': tolerance,
                               'mouthFitMeasuredErrorMm': worst,
                               'mouthFitIterations': iteration+1}
        if not bad or n > _MAX_PROFILES:
            break
        q = deduplicate(np.r_[q, bad])
    raise ConfigError(
        f'rectangular morphed mouth cubic fit did not reach {tolerance:g} mm '
        f'outline tolerance (measured {worst:g} mm) within the station/iteration '
        'limit; reduce the morph transition or use a rounded mouth corner')
