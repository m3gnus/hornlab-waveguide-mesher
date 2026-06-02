"""Lookup-table waveguide builder.

Evaluates a horn profile from explicit ``(z, r)`` control points using
PCHIP interpolation (via the geometry-cli), then delegates the mesh build
to :func:`build_axisymmetric`. The lookup form covers any profile that
does not fit OSSE / R-OSSE closed-form expressions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..geometry import AxiHornGeometry, BuiltGeometry, LookupHornGeometry
from ..geometry_client import GeometryClient, get_default_client
from .axisymmetric import build_axisymmetric


def compute_lookup_profile_points(
    geometry: LookupHornGeometry,
    *,
    client: Optional[GeometryClient] = None,
) -> np.ndarray:
    """Return an ``(n_axial, 2)`` array of ``(z_mm, r_mm)`` points."""
    pts = np.asarray(geometry.lookup_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError("LookupHornGeometry.lookup_points must be (N, 2) with at least two rows")
    gc = client or get_default_client()
    t_values = np.linspace(0.0, 1.0, int(geometry.n_axial))
    x, y, _L = gc.compute_lookup_profile(t_values, pts.tolist())
    return np.column_stack([x, y])


def build_lookup_waveguide(
    geometry: LookupHornGeometry,
    *,
    client: Optional[GeometryClient] = None,
) -> BuiltGeometry:
    """Build a lookup-table waveguide horn surface."""
    profile = compute_lookup_profile_points(geometry, client=client)
    axi = AxiHornGeometry(
        profile_points=profile,
        throat_radius_mm=float(profile[0, 1]),
        cross_section=geometry.cross_section,
        enclosure=geometry.enclosure,
        n_phi=geometry.n_phi,
    )
    return build_axisymmetric(axi)
