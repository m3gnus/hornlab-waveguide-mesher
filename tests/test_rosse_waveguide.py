from __future__ import annotations

import numpy as np

from hornlab_mesher import RosseHornGeometry, compute_rosse_profile_points


def test_compute_rosse_profile_points_endpoints():
    geom = RosseHornGeometry(
        R_mm=150.0, r0_mm=12.7, a_deg=60.0, a0_deg=15.5, k=1.0, q=1.0, n_axial=5,
    )
    points = compute_rosse_profile_points(geom)
    assert points.shape == (5, 2)
    assert np.isclose(points[0, 1], 12.7)
    assert np.isclose(points[-1, 1], 150.0)


def test_compute_rosse_explicit_mrb_overrides_defaults():
    """``m``/``r``/``b`` defaults from WG (M=0.85, R=0.4, B=0.2) vs. explicit overrides."""
    t = 5
    default_geom = RosseHornGeometry(
        R_mm=150.0, r0_mm=12.7, a_deg=60.0, a0_deg=15.5, k=1.0, q=1.0, n_axial=t,
    )
    override_geom = RosseHornGeometry(
        R_mm=150.0, r0_mm=12.7, a_deg=60.0, a0_deg=15.5,
        k=1.0, q=1.0, m=0.5, r=0.5, b=1.0, n_axial=t,
    )
    pts_default = compute_rosse_profile_points(default_geom)
    pts_override = compute_rosse_profile_points(override_geom)
    # Default and explicit (m=0.5, r=0.5, b=1) give materially different curves.
    assert not np.allclose(pts_default, pts_override)
