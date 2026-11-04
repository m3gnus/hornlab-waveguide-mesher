"""Bounded ATH throat cut; the ordinary R-OSSE formula remains its sole body owner."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from .errors import ath_refusal

KEY = 'athRosseTruncMm'


def _fail(reason):
    raise ath_refusal(f'R-OSSE.trunc {reason}. Remove trunc or the incompatible setting, then re-import.')


def validate(params, *, mode=None):
    """Refuse unmeasured compositions before preview, mesh or export starts."""
    from .profile_formulas import _rosse_main_coefficients
    from .text_import import (
        uses_current_ath_axial_sampling,
        uses_v3_text_import_geometry,
    )

    if KEY not in params:
        return
    if params.get('type') != 'R-OSSE' or not uses_v3_text_import_geometry(params):
        _fail('requires a current R-OSSE import')
    if not uses_current_ath_axial_sampling(params):
        _fail('requires the current ATH axial sampling rule')
    value = params[KEY]
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        _fail('must be a positive finite number of axial millimetres')
    # Numeric circular meridians are the measured domain. Do not let an
    # azimuth expression silently produce a different split/source per profile.
    for key in ('R', 'r0', 'a', 'a0', 'k', 'r', 'm', 'b', 'q', 'tmax', 's1', 's2'):
        if key in params and (type(params[key]) not in (int, float) or not math.isfinite(params[key])):
            _fail(f'requires finite scalar profile coefficients ({key})')
    if params.get('tmax', 1) != 1:
        _fail('with tmax other than 1 is unqualified: ATH can fail its split search')
    for key, default, required, label in (
        ('scale', 1, 1, 'Scale'), ('athAxialScale', 1, 1, 'Scale.Z'),
        ('rot', 0, 0, 'Rot'), ('throatExtLength', 0, 0, 'Throat.Ext.Length'),
        ('throatExtAngle', 0, 0, 'Throat.Ext.Angle'), ('slotLength', 0, 0, 'Slot.Length'),
        ('sourceShape', 1, 0, 'Source.Shape (use 2 for a flat disc)'),
        ('verticalOffset', 0, 0, 'Mesh.VerticalOffset'),
        ('wallThickness', 0, 0, 'Mesh.WallThickness'), ('morphTarget', 0, 0, 'Morph'),
        ('gcurveType', 0, 0, 'GCurve'),
    ):
        if params.get(key, default) != required:
            _fail(f'with {label} is not yet qualified')
    section = params.get('profileSystem', {}).get('crossSection', {})
    if section.get('exponent', 2) != 2 or section.get('aspectRatio', 1) != 1:
        _fail('requires a circular cross-section')
    if params.get('athRoundover') or params.get('athRearShape', 1) != 1 or params.get('throat_adapter'):
        _fail('with an ATH lip, shaped rear or throat adapter is not yet qualified')
    if params.get('encDepth', 0) or (mode is not None and mode not in {'bare', 'freestanding'}):
        _fail('requires a free-standing horn without an enclosure')
    c = _rosse_main_coefficients(0, params)
    # X is concave up to m in this domain; the positive endpoint bracket
    # selects the forward crossing even when a second root exists after m.
    if not (c.L > 0 and c.r > 0 and .5 <= c.m <= 1 and c.b >= 0 and c.q >= 1
            and c.k > 0 and c.r0 > 0 and c.R > c.r0):
        _fail('requires R > r0 > 0, k > 0, r > 0, 0.5 <= m <= 1, b >= 0 and q >= 1')
    _split(c, float(value))


@lru_cache(maxsize=128)
def _split(coefficients, distance):
    from .profile_formulas import _rosse_main_curve

    def z(t):
        return float(_rosse_main_curve(np.asarray(t), coefficients)[0])

    if not 0 < distance < z(coefficients.m):
        _fail('must lie on the qualified forward branch, below the axial depth at t=m')
    return brentq(lambda t: z(t)-distance, 0, coefficients.m, xtol=1e-14)


def curve(t, p, params):
    """Evaluate the cut body in the retained parameter, using the existing formula."""
    from .profile_formulas import _rosse_main_coefficients, _rosse_main_curve

    validate(params)
    c = _rosse_main_coefficients(p, params)
    distance = float(params[KEY])
    join = _split(c, distance)
    t = np.asarray(t, dtype=np.float64)
    z, radius = _rosse_main_curve(join+(1-join)*t, c)
    return np.where(t == 0, 0., z-distance), radius
