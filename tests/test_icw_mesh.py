"""Mesher-integration tests for the Intrinsic-Curvature Waveguide (ICW).

These exercise the ICW *adapter* and dispatch wiring (Phase 1): the ICW kernel
(``hornlab_mesher.icw``) feeding ``profile_points`` / ``build_point_grid`` /
``build_from_config`` along the SAME branch path as R-OSSE. The kernel itself is
covered separately by ``test_icw.py``; here we assert the real mesher can build
an ICW waveguide and that an ICW seeded from an OSSE profile reproduces the OSSE
meridian (the migration-parity guarantee).

Mirrors ``test_osse_waveguide.py`` / ``test_rosse_waveguide.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher import build_from_config, load_mesh
from hornlab_mesher.profiles import build_point_grid, profile_points


# OSSE reference dict (matches the one in test_icw.py) used as the ICW seed for
# the meridian-parity test.
OSSE_PARAMS = {
    "type": "OSSE", "r0": 12.7, "a0": 18, "a": 35, "k": 1,
    "L": 120, "s": 0.8, "n": 4, "q": 0.9,
}

# R-OSSE reference dict (matches the one in test_icw.py) -- a rolled-back mouth
# (theta crosses 90 deg, non-monotone x) used for the rollback-parity test.
ROSSE_PARAMS = {
    "type": "R-OSSE", "R": 150, "r0": 12.7, "k": 1, "q": 1, "m": 0.85,
    "r": 0.4, "b": 0.2, "a": 60, "a0": 15.5,
}


def _inner_grid(grid: dict) -> np.ndarray:
    """Reshape the flat ``inner_points`` list to ``(n_phi, n_length + 1, 3)``."""
    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    return np.asarray(grid["inner_points"], dtype=np.float64).reshape(n_phi, n_length + 1, 3)


def _ring_radii(inner: np.ndarray, column: int) -> np.ndarray:
    return np.hypot(inner[:, column, 0], inner[:, column, 1])


# =====================================================================================
# Flat-baffle end-to-end
# =====================================================================================
def test_icw_flat_baffle_point_grid_shape_and_endpoints():
    params = {
        "type": "ICW",
        "r0": 12.7,
        "a0": 18,
        "termination": "flat_baffle",
        "L": 120,
        "R": 110,
        "lengthSegments": 32,
        "angularSegments": 48,
    }
    grid = build_point_grid(params)
    inner = _inner_grid(grid)

    n_phi, n_cols, _ = inner.shape
    assert n_cols == 33  # lengthSegments + 1
    assert n_phi >= 48  # full circle, normalised angular count
    assert grid["sampling_mode"] == "uniform"  # ICW forces uniform axial sampling

    assert np.all(np.isfinite(inner))

    # Throat ring radius ~= r0; mouth ring radius ~= R (flat-baffle size targets).
    throat = _ring_radii(inner, 0)
    mouth = _ring_radii(inner, -1)
    assert np.allclose(throat, 12.7, atol=1e-3)
    assert np.allclose(mouth, 110.0, atol=0.05)

    # Radius is monotone non-decreasing along the axis for every azimuth.
    radii = np.hypot(inner[:, :, 0], inner[:, :, 1])
    assert np.all(np.diff(radii, axis=1) >= -1e-6)


def test_icw_flat_baffle_builds_full_mesh(tmp_path):
    """The same config-driven path as the OSSE smoke test, with formula=ICW."""
    config = {
        "profile": {
            "formula": "ICW",
            "r0_mm": 12.7,
            "a0_deg": 18,
            "termination": "flat_baffle",
            "L_mm": 120,
            "R_mm": 110,
        },
        "mesh": {
            "length_segments": 32,
            "angular_segments": 48,
            "throat_res_mm": 6.0,
            "mouth_res_mm": 22.0,
        },
    }
    result = build_from_config(config, tmp_path / "icw_flat.msh")
    assert result.formula == "ICW"
    assert result.n_triangles > 0

    info = load_mesh(result.mesh_path)
    assert info.n_triangles > 0
    assert set(info.physical_groups) >= {1, 2}


# =====================================================================================
# Meridian parity (the key test): ICW seeded from OSSE == OSSE meridian
# =====================================================================================
def test_icw_seed_reproduces_osse_meridian():
    """ICW with ``icw_seed`` = an OSSE param dict reproduces the OSSE meridian.

    Compared as CURVES (interpolate r at matching x), not index-by-index, since
    the ICW sampler uses a uniform-sigma zmap that differs from OSSE's axial
    distribution. Deviation must stay well under 0.05 mm.
    """
    n = 200
    osse_pts = profile_points(OSSE_PARAMS, n)
    icw_pts = profile_points({"type": "ICW", "icw_seed": OSSE_PARAMS}, n)

    x_o, r_o = osse_pts[:, 0], osse_pts[:, 1]
    x_i, r_i = icw_pts[:, 0], icw_pts[:, 1]

    # Both meridians share the same throat/mouth axial span (sub-micron ends).
    assert x_i[0] == pytest.approx(x_o[0], abs=1e-6)
    assert x_i[-1] == pytest.approx(x_o[-1], abs=1e-3)

    # Resample the ICW radius onto the OSSE axial stations and compare as curves.
    r_i_on_o = np.interp(x_o, x_i, r_i)
    max_dev = float(np.max(np.abs(r_i_on_o - r_o)))
    assert max_dev < 0.05, f"ICW-seeded meridian deviates {max_dev:.4f} mm from OSSE"


def test_icw_seed_builds_mesh_matching_osse_extents(tmp_path):
    """An OSSE-seeded ICW config builds a valid mesh with the OSSE mouth radius."""
    config = {
        "profile": {
            "formula": "ICW",
            "icw_seed": OSSE_PARAMS,
        },
        "mesh": {
            "length_segments": 32,
            "angular_segments": 48,
            "throat_res_mm": 6.0,
            "mouth_res_mm": 22.0,
        },
    }
    result = build_from_config(config, tmp_path / "icw_seed.msh")
    assert result.formula == "ICW"
    assert result.n_triangles > 0

    # Mouth radius of the seeded ICW grid should match the OSSE mouth radius.
    osse_mouth_r = float(profile_points(OSSE_PARAMS, 200)[-1, 1])
    grid = build_point_grid({"type": "ICW", "icw_seed": OSSE_PARAMS, "lengthSegments": 32, "angularSegments": 48})
    inner = _inner_grid(grid)
    assert np.allclose(_ring_radii(inner, -1), osse_mouth_r, atol=0.05)


# =====================================================================================
# Rollback end-to-end (non-monotone x, handled like R-OSSE)
# =====================================================================================
def test_icw_rollback_builds_valid_grid():
    params = {
        "type": "ICW",
        "r0": 12.7,
        "a0": 12,
        "termination": "rollback",
        "theta1": 160,
        "R": 110,
        "depth": 90,
        "lengthSegments": 40,
        "angularSegments": 32,
    }
    grid = build_point_grid(params)
    inner = _inner_grid(grid)

    n_phi, n_cols, _ = inner.shape
    assert n_cols == 41  # lengthSegments + 1
    assert np.all(np.isfinite(inner))

    # Rollback: terminal axial depth x(1) ~= depth target, but the wall bulges
    # forward past it before curling back (theta passes 90deg), so the axial
    # coordinate is NON-monotone -- exactly the R-OSSE-like parametric case.
    x_axis = inner[0, :, 2]
    assert x_axis[-1] == pytest.approx(90.0, abs=0.5)
    assert float(np.max(x_axis)) > x_axis[-1] + 1.0  # bulges past the tip plane

    # Radius stays monotone non-decreasing even through the rollback.
    radii = np.hypot(inner[:, :, 0], inner[:, :, 1])
    assert np.all(np.diff(radii, axis=1) >= -1e-6)


def test_icw_rollback_builds_full_mesh(tmp_path):
    config = {
        "profile": {
            "formula": "ICW",
            "r0_mm": 12.7,
            "a0_deg": 12,
            "termination": "rollback",
            "theta1_deg": 160,
            "R_mm": 110,
            "depth": 90,
        },
        "mesh": {
            "length_segments": 40,
            "angular_segments": 32,
            "throat_res_mm": 6.0,
            "mouth_res_mm": 22.0,
        },
    }
    result = build_from_config(config, tmp_path / "icw_rollback.msh")
    assert result.formula == "ICW"
    assert result.n_triangles > 0


# =====================================================================================
# Infeasible target raises a clear ValueError (never a silent bad mesh)
# =====================================================================================
def test_icw_infeasible_target_raises_value_error():
    """A mouth narrower than the throat is geometrically impossible for ICW."""
    params = {
        "type": "ICW",
        "r0": 50.0,
        "a0": 18,
        "termination": "flat_baffle",
        "L": 120,
        "R": 20.0,  # r_mouth < r0 -> infeasible
    }
    with pytest.raises(ValueError, match="infeasible"):
        profile_points(params, 20)


def test_icw_infeasible_target_raises_in_build_point_grid():
    params = {
        "type": "ICW",
        "r0": 50.0,
        "a0": 18,
        "termination": "flat_baffle",
        "L": 120,
        "R": 20.0,
        "lengthSegments": 16,
        "angularSegments": 16,
    }
    with pytest.raises(ValueError, match="infeasible"):
        build_point_grid(params)


def test_icw_infeasible_target_raises_through_build_from_config(tmp_path):
    """An infeasible ICW target surfaces as a clear ValueError through the FULL
    config-driven build path (not just profile_points / build_point_grid)."""
    config = {
        "profile": {
            "formula": "ICW",
            "r0_mm": 50.0,
            "a0_deg": 18,
            "termination": "flat_baffle",
            "L_mm": 120,
            "R_mm": 20.0,  # r_mouth < r0 -> geometrically impossible
        },
        "mesh": {"length_segments": 16, "angular_segments": 16},
    }
    with pytest.raises(ValueError, match="infeasible"):
        build_from_config(config, tmp_path / "icw_infeasible.msh")


# =====================================================================================
# Curve-cache correctness (P1-1): differing coeff arrays must NOT collide
# =====================================================================================
def test_icw_cache_distinguishes_arrays_differing_at_one_interior_index():
    """Two ICW DIRECT-mode grids whose ``icw_coeffs`` differ only at a single
    interior index by 1e-9 must produce DIFFERENT meridians.

    Guards the P1-1 fix: the old ``json.dumps(..., default=str)`` cache key
    stringified numpy arrays lossily, so distinct coefficient arrays could hash
    to the same key and the memo returned a stale curve. The key now hashes
    array-likes losslessly, so no stale hit can occur.
    """
    n = 2000
    coeffs_a = np.linspace(-0.01, 0.01, n)
    coeffs_b = coeffs_a.copy()
    coeffs_b[n // 2] += 1e-9  # one interior index, tiny perturbation

    base = {
        "type": "ICW",
        "r0": 12.7,
        "a0": 18.0,
        "icw_S": 150.0,
        "lengthSegments": 24,
        "angularSegments": 16,
    }
    grid_a = build_point_grid({**base, "icw_coeffs": coeffs_a})
    grid_b = build_point_grid({**base, "icw_coeffs": coeffs_b})

    inner_a = _inner_grid(grid_a)
    inner_b = _inner_grid(grid_b)
    # A stale cache hit would make these byte-identical; the lossless key prevents it.
    assert not np.array_equal(inner_a, inner_b), (
        "differing icw_coeffs collided -> stale cache hit (P1-1 regression)"
    )
    # Same arrays must still share a curve (no spurious miss): identical grids match.
    grid_a2 = build_point_grid({**base, "icw_coeffs": coeffs_a.copy()})
    assert np.array_equal(inner_a, _inner_grid(grid_a2))


# =====================================================================================
# ICW + morph (rectangular morph target) builds a valid grid
# =====================================================================================
def test_icw_with_rectangular_morph_builds_valid_grid():
    """ICW feeds the same morph path as OSSE/R-OSSE: a rounded-rectangular morph
    target (morphTarget=1) deforms the mouth ring to a non-circular section."""
    params = {
        "type": "ICW",
        "r0": 12.7,
        "a0": 18,
        "termination": "flat_baffle",
        "L": 120,
        "R": 110,
        "lengthSegments": 24,
        "angularSegments": 48,
        "morphTarget": 1,  # rounded rectangle
        "morphWidth": 200.0,
        "morphHeight": 120.0,
        "morphCorner": 20.0,
    }
    grid = build_point_grid(params)
    inner = _inner_grid(grid)
    assert np.all(np.isfinite(inner))

    # Throat ring stays circular at r0; mouth ring is non-circular (rectangular morph).
    throat = _ring_radii(inner, 0)
    assert np.allclose(throat, 12.7, atol=1e-3)
    mouth = _ring_radii(inner, -1)
    assert (mouth.max() - mouth.min()) > 1.0  # azimuthal radius varies -> morphed


def test_icw_morph_builds_full_mesh(tmp_path):
    config = {
        "profile": {
            "formula": "ICW",
            "r0_mm": 12.7,
            "a0_deg": 18,
            "termination": "flat_baffle",
            "L_mm": 120,
            "R_mm": 110,
        },
        "mesh": {"length_segments": 24, "angular_segments": 48,
                 "throat_res_mm": 6.0, "mouth_res_mm": 22.0},
        "morph": {
            "morph_target": 1,
            "morph_width_mm": 200.0,
            "morph_height_mm": 120.0,
            "morph_corner_mm": 20.0,
        },
    }
    result = build_from_config(config, tmp_path / "icw_morph.msh")
    assert result.formula == "ICW"
    assert result.n_triangles > 0


# =====================================================================================
# ICW + enclosure builds a valid mesh
# =====================================================================================
def test_icw_with_enclosure_builds_valid_mesh(tmp_path):
    """ICW drives the enclosure build path (inner waveguide wall + enclosure box)."""
    config = {
        "formula": "ICW",
        "mode": "enclosure",
        "profile": {
            "formula": "ICW",
            "r0_mm": 12.7,
            "a0_deg": 18,
            "termination": "flat_baffle",
            "L_mm": 120,
            "R_mm": 110,
        },
        "mesh": {"length_segments": 24, "angular_segments": 48,
                 "throat_res_mm": 6.0, "mouth_res_mm": 22.0},
        "enclosure": {
            "depth": 300, "space_l": 1, "space_t": 150, "space_r": 1, "space_b": 150,
            "edge": 1, "edgeType": 1, "frontMeshSize": 40, "backMeshSize": 40,
        },
    }
    result = build_from_config(config, tmp_path / "icw_enclosure.msh")
    assert result.formula == "ICW"
    assert result.n_triangles > 0

    info = load_mesh(result.mesh_path)
    assert info.n_triangles > 0
    # Enclosure adds physical groups beyond the bare interior/throat pair.
    assert set(info.physical_groups) >= {1, 2, 3}


# =====================================================================================
# Rollback parity: ICW seeded from an R-OSSE dict reproduces the R-OSSE meridian
# =====================================================================================
def test_icw_seed_reproduces_rosse_rollback_meridian():
    """ICW with ``icw_seed`` = an R-OSSE param dict reproduces the R-OSSE meridian
    as a curve, including the rolled-back lip (theta crossing 90 deg, non-monotone
    x). Compared by normalised arc length (not index-by-index) since the rollback
    makes x non-monotone. Measured max deviation is ~0.009 mm (well under 0.1 mm).
    """
    n = 300
    rosse_pts = profile_points(ROSSE_PARAMS, n)
    icw_pts = profile_points({"type": "ICW", "icw_seed": ROSSE_PARAMS}, n)

    x_o, r_o = rosse_pts[:, 0], rosse_pts[:, 1]
    x_i, r_i = icw_pts[:, 0], icw_pts[:, 1]

    # The R-OSSE reference genuinely rolls back (non-monotone x) -- this is the case
    # the index-by-index parity test cannot use, so compare as arc-length curves.
    assert not np.all(np.diff(x_o) >= 0.0), "reference R-OSSE is not a rollback"

    s_o = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x_o), np.diff(r_o)))])
    s_o /= s_o[-1]
    s_i = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x_i), np.diff(r_i)))])
    s_i /= s_i[-1]
    x_i_on = np.interp(s_o, s_i, x_i)
    r_i_on = np.interp(s_o, s_i, r_i)

    max_dev = float(np.max(np.hypot(x_i_on - x_o, r_i_on - r_o)))
    assert max_dev < 0.1, f"ICW-seeded meridian deviates {max_dev:.4f} mm from R-OSSE"


# =====================================================================================
# Dispatch / validation wiring
# =====================================================================================
def test_icw_config_rejects_osse_rosse_shape_keys():
    """OSSE-only (s/n/rot) and R-OSSE-only (m/r/b/tmax) keys are rejected at the
    top level for an ICW config (but remain valid nested inside icw_seed)."""
    from hornlab_mesher.config_builder import build_geometry_params
    from hornlab_mesher.config_parser import ConfigError

    for bad_key, value in (("n", 4), ("s", 0.8), ("m", 0.85), ("tmax", 0.5)):
        config = {
            "profile": {"formula": "ICW", "r0_mm": 12.7, "a0_deg": 18, bad_key: value},
            "mesh": {},
        }
        with pytest.raises(ConfigError):
            build_geometry_params(config)


def test_icw_seed_nested_osse_keys_are_allowed():
    """OSSE shape keys nested inside icw_seed must NOT trip the top-level guard."""
    from hornlab_mesher.config_builder import build_geometry_params

    config = {
        "profile": {"formula": "ICW", "icw_seed": OSSE_PARAMS},
        "mesh": {},
    }
    common, formula, mode = build_geometry_params(config)
    assert formula == "ICW"
    assert common["icw_seed"] == OSSE_PARAMS
