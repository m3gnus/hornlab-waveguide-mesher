"""OSSE waveguide builder.

Evaluates the OSSE profile through the canonical WG JS pipeline (via the
geometry-cli subprocess), then delegates the mesh build to
an internal axial loft helper. This keeps a single source of truth for
acoustic-waveguide formulas (WG owns the math) while reusing one surface
authoring path.

:func:`compute_osse_inner_points` produces the same flat 3D point grid the
WG OCC builder consumes, so downstream tools that speak the WG point-grid
contract can construct an OSSE wall directly from parameters without needing
a JS frontend.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..geometry import _AxiHornGeometry, BuiltGeometry, OsseHornGeometry
from ..geometry_client import GeometryClient, get_default_client
from .axisymmetric import _build_axisymmetric


def compute_osse_profile_points(
    geometry: OsseHornGeometry,
    *,
    client: Optional[GeometryClient] = None,
) -> np.ndarray:
    """Return an ``(n_axial, 2)`` array of ``(z_mm, r_mm)`` points.

    Calls the geometry-cli's ``compute_osse_profile`` op for the OSSE
    parameters carried in *geometry*. ``n_axial`` controls the number of
    sample points along the profile (uniform in normalised ``t``).
    """
    gc = client or get_default_client()
    t_values = np.linspace(0.0, 1.0, int(geometry.n_axial))
    x, y, _total_length = gc.compute_osse_profile(
        t_values,
        phi=0.0,
        L=geometry.L_mm,
        r0=geometry.r0_mm,
        a=geometry.a_deg,
        a0=geometry.a0_deg,
        k=geometry.k,
        n=geometry.n,
        q=geometry.q,
        s=geometry.s,
        throatExtLength=geometry.throat_ext_length_mm,
        throatExtAngle=geometry.throat_ext_angle_deg,
        slotLength=geometry.slot_length_mm,
        rot=geometry.rot_deg,
    )
    return np.column_stack([x, y])


def build_osse_waveguide(
    geometry: OsseHornGeometry,
    *,
    client: Optional[GeometryClient] = None,
) -> BuiltGeometry:
    """Build an OSSE waveguide horn surface via the canonical evaluator."""
    profile = compute_osse_profile_points(geometry, client=client)
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
    client: Optional[GeometryClient] = None,
) -> dict[str, Any]:
    """Return the flat WG ``inner_points`` grid for *geometry*.

    The grid follows the WG OCC payload contract: a flat
    ``[x, y, axial, x, y, axial, ...]`` list of length
    ``grid_n_phi * (grid_n_length + 1) * 3`` with X/Y transverse and Z
    along the horn axis from throat to mouth. The cross-section exponent
    and aspect ratio carried on *geometry* flow through unchanged.

    ``quadrants`` follows ATH semantics (``"1234"`` = full circle).

    Returns a dict with ``inner_points`` (list of floats), ``grid_n_phi``,
    ``grid_n_length``, ``full_circle``, and ``angle_list`` — exactly what
    the Waveguide Generator mesh endpoint consumes.
    """
    if geometry.n_axial < 2:
        raise ValueError("OsseHornGeometry.n_axial must be at least 2 for inner-point grids")
    gc = client or get_default_client()
    params: dict[str, Any] = {
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
        "morphTarget": 0,
        "angularSegments": int(geometry.n_phi),
        "lengthSegments": int(geometry.n_axial - 1),
        "quadrants": quadrants,
        # Mesh-resolution fields are required by the WG params normaliser but
        # are not used for inner-point extraction. Sane defaults keep the
        # pipeline happy without leaking arbitrary numbers to callers.
        "throatResolution": 4.0,
        "mouthResolution": 26.0,
        "rearResolution": 25.0,
        "wallThickness": 6.0,
        "encDepth": 0.0,
        "profileSystem": {
            "crossSection": {
                "exponent": float(geometry.cross_section.exponent),
                "aspectRatio": float(geometry.cross_section.aspect_ratio),
            },
        },
    }
    return gc.build_inner_points(params)
