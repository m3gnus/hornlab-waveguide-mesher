"""OSSE waveguide builder."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..geometry import _AxiHornGeometry, BuiltGeometry, OsseHornGeometry
from ..profiles import build_point_grid, profile_points
from .axisymmetric import _build_axisymmetric


def compute_osse_profile_points(
    geometry: OsseHornGeometry,
) -> np.ndarray:
    """Return an ``(n_axial, 2)`` array of ``(z_mm, r_mm)`` points.

    ``n_axial`` controls the number of sample points along the profile,
    uniformly in normalised ``t``.
    """
    return profile_points(_osse_params(geometry), int(geometry.n_axial), phi=0.0)


def build_osse_waveguide(
    geometry: OsseHornGeometry,
) -> BuiltGeometry:
    """Build an OSSE waveguide horn surface via the canonical evaluator."""
    profile = compute_osse_profile_points(geometry)
    axi = _AxiHornGeometry(
        profile_points=profile,
        throat_radius_mm=float(profile[0, 1]),
        cross_section=geometry.cross_section,
        enclosure=geometry.enclosure,
        n_phi=geometry.n_phi,
    )
    return _build_axisymmetric(axi)


def compute_osse_inner_points(
    geometry: OsseHornGeometry,
    *,
    quadrants: str = "1234",
) -> dict[str, Any]:
    """Return a flat OSSE ``inner_points`` grid for *geometry*."""
    if geometry.n_axial < 2:
        raise ValueError("OsseHornGeometry.n_axial must be at least 2 for inner-point grids")
    params = _osse_params(geometry)
    params.update({
        "angularSegments": int(geometry.n_phi),
        "lengthSegments": int(geometry.n_axial - 1),
        "quadrants": quadrants,
        "wallThickness": 0.0,
        "encDepth": 0.0,
    })
    return build_point_grid(params)


def _osse_params(geometry: OsseHornGeometry) -> dict[str, Any]:
    return {
        "type": "OSSE",
        "L": float(geometry.L_mm),
        "r0": float(geometry.r0_mm),
        "a": float(geometry.a_deg),
        "a0": float(geometry.a0_deg),
        "k": float(geometry.k),
        "n": float(geometry.n),
        "q": float(geometry.q),
        "s": float(geometry.s),
        "throatExtLength": float(geometry.throat_ext_length_mm),
        "throatExtAngle": float(geometry.throat_ext_angle_deg),
        "slotLength": float(geometry.slot_length_mm),
        "rot": float(geometry.rot_deg),
        "profileSystem": {
            "crossSection": {
                "exponent": float(geometry.cross_section.exponent),
                "aspectRatio": float(geometry.cross_section.aspect_ratio),
            },
        },
    }
