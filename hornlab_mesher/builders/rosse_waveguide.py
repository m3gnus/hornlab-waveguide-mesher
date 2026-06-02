"""R-OSSE curve evaluator (no full mesh builder).

ROSSE's axial coordinate is not monotonic in normalised ``t`` for typical
parameter ranges (the curve folds back on itself near the mouth), so it
cannot be fed straight into :func:`build_axisymmetric`. This module
exposes only :func:`compute_rosse_profile_points` for callers that want
the (z, r) curve for plotting, analysis, or custom surface authoring.

If a future builder handles folded curves (e.g. a free-form sweep that
does not assume monotonic z), it can live alongside this evaluator.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..geometry import RosseHornGeometry
from ..geometry_client import GeometryClient, get_default_client


def _rosse_params(geometry: RosseHornGeometry) -> dict[str, float]:
    params: dict[str, float] = dict(
        R=float(geometry.R_mm),
        r0=float(geometry.r0_mm),
        a=float(geometry.a_deg),
        a0=float(geometry.a0_deg),
        k=float(geometry.k),
        q=float(geometry.q),
    )
    if geometry.m is not None:
        params["m"] = float(geometry.m)
    if geometry.r is not None:
        params["r"] = float(geometry.r)
    if geometry.b is not None:
        params["b"] = float(geometry.b)
    return params


def compute_rosse_profile_points(
    geometry: RosseHornGeometry,
    *,
    client: Optional[GeometryClient] = None,
) -> np.ndarray:
    """Return an ``(n_axial, 2)`` array of ``(x_mm, y_mm)`` curve points.

    Note that the ``x`` (axial-like) coordinate is generally not monotonic
    in ROSSE: for typical parameter ranges the curve peaks short of the
    mouth and folds back. Consumers should not assume it is a single-valued
    ``r(z)`` profile.
    """
    gc = client or get_default_client()
    t_values = np.linspace(0.0, 1.0, int(geometry.n_axial))
    x, y, _L = gc.compute_rosse_profile(t_values, phi=0.0, **_rosse_params(geometry))
    return np.column_stack([x, y])
