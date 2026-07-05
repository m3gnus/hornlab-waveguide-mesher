from __future__ import annotations

import numpy as np

from hornlab_mesher import RosseHornGeometry, compute_rosse_profile_points
from hornlab_mesher.profiles import calculate_rosse, profile_points, rosse_total_length


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


def test_calculate_rosse_applies_throat_extension_and_slot():
    params = {
        "type": "R-OSSE",
        "R": 150.0,
        "r0": 10.0,
        "a": 45.0,
        "a0": 12.0,
        "k": 1.0,
        "q": 1.0,
        "throatExtLength": 12.0,
        "throatExtAngle": 20.0,
        "slotLength": 5.0,
    }
    total = rosse_total_length(params)
    # ATH convention (verified via GridExport): r0 anchors the MAIN throat and
    # the extension tapers BACK to the driver end; the main curve and mouth are
    # unchanged by the extension, which adds to the total length.
    no_ext = {k: v for k, v in params.items() if k not in ("throatExtLength", "throatExtAngle", "slotLength")}
    assert np.isclose(total, rosse_total_length(no_ext) + 12.0 + 5.0)
    driver_r = 10.0 - 12.0 * np.tan(np.deg2rad(20.0))

    z, radius = calculate_rosse(0.0, 0.0, params)
    assert np.isclose(z, 0.0)
    assert np.isclose(radius, driver_r)

    z, radius = calculate_rosse((6.0 / total), 0.0, params)
    assert np.isclose(z, 6.0)
    assert np.isclose(radius, driver_r + 6.0 * np.tan(np.deg2rad(20.0)))

    z, radius = calculate_rosse(((12.0 + 2.5) / total), 0.0, params)
    assert np.isclose(z, 14.5)
    assert np.isclose(radius, 10.0)

    z, radius = calculate_rosse(1.0, 0.0, params)
    assert z > 17.0
    assert np.isclose(radius, 150.0)


def test_compute_rosse_profile_points_accepts_throat_extension_dataclass_fields():
    geom = RosseHornGeometry(
        R_mm=150.0,
        r0_mm=10.0,
        a_deg=45.0,
        a0_deg=12.0,
        k=1.0,
        q=1.0,
        throat_ext_length_mm=12.0,
        throat_ext_angle_deg=20.0,
        slot_length_mm=5.0,
        n_axial=5,
    )

    points = compute_rosse_profile_points(geom)
    # Taper-back convention: the profile starts at the DRIVER radius
    # (r0 - ext*tan(angle)) and reaches r0 at the main throat.
    assert np.isclose(points[0, 1], 10.0 - 12.0 * np.tan(np.deg2rad(20.0)))
    assert np.isclose(points[-1, 1], 150.0)
    assert points[-1, 0] > 17.0


def test_rosse_tmax_truncates_total_profile_when_extension_is_enabled():
    params = {
        "type": "R-OSSE",
        "R": 150.0,
        "r0": 10.0,
        "a": 45.0,
        "a0": 12.0,
        "k": 1.0,
        "q": 1.0,
        "throatExtLength": 12.0,
        "throatExtAngle": 20.0,
        "slotLength": 5.0,
        "tmax": 0.5,
    }

    points = profile_points(params, 5)
    expected = calculate_rosse(0.5, 0.0, params)
    assert np.allclose(points[-1], expected)
    assert points[-1, 1] < 150.0


def test_rosse_nine_segment_zmap_uses_rosse_table_not_osse_solana():
    """The 9-segment early return was formula-blind and handed R-OSSE the OSSE
    solana table; R-OSSE must interpolate its own T20 reference."""
    from hornlab_mesher.profile_sampling import _ath_default_zmap, _ATH_T_20
    import numpy as np

    nine = _ath_default_zmap(9, "R-OSSE")
    positions = (np.arange(1, 9) / 9.0) * (len(_ATH_T_20) - 1)
    expected = np.interp(positions, np.arange(len(_ATH_T_20)), _ATH_T_20)
    assert np.allclose(nine[1:9], expected)
    # neighbours stay monotone-consistent: first interior ring between 8- and 10-seg values
    eight = _ath_default_zmap(8, "R-OSSE")[1]
    ten = _ath_default_zmap(10, "R-OSSE")[1]
    assert min(eight, ten) <= nine[1] <= max(eight, ten)
    # OSSE keeps the exact solana table at 9 segments
    from hornlab_mesher.profile_sampling import _ATH_T_9
    assert np.array_equal(_ath_default_zmap(9, "OSSE"), _ATH_T_9)
