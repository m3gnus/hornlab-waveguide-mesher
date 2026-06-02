"""R-OSSE curve evaluator."""

from __future__ import annotations

import numpy as np

from ..geometry import RosseHornGeometry
from ..profiles import profile_points


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
) -> np.ndarray:
    """Return an ``(n_axial, 2)`` array of ``(x_mm, y_mm)`` curve points.

    Note that the ``x`` (axial-like) coordinate is generally not monotonic
    in ROSSE: for typical parameter ranges the curve peaks short of the
    mouth and folds back. Consumers should not assume it is a single-valued
    ``r(z)`` profile.
    """
    return profile_points({"type": "R-OSSE", **_rosse_params(geometry)}, int(geometry.n_axial), phi=0.0)
