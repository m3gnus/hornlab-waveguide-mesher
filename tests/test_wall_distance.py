"""Wall clearance: exact triangle distances and the fitted-face lower bound."""
import gmsh
import numpy as np
import pytest

from hornlab_mesher.offset_envelope import exterior_outer_offset, regularize_outer_offset
from hornlab_mesher.profile_sampling import _outer_offset_shell
from hornlab_mesher.wall_distance import (
    _paired_triangle_squared, fitted_face_clearance, minimum_wall_distance)


@pytest.mark.parametrize('scale', [0.01, 1, 100])
@pytest.mark.parametrize('other,expected', [
    ([[0, 0, 2], [1, 0, 2], [0, 1, 2]], 2),  # parallel faces
    ([[0.2, 0.2, 2], [0.3, 0.2, 3], [0.2, 0.3, 3]], 2),  # vertex/face interior
    ([[0.4, -1, 2], [0.4, 1, 2], [0.5, -1, 3]], 2),  # skew edge interiors
    ([[0.2, 0.2, -1], [0.2, 0.2, 1], [0.3, 0.2, 1]], 0),  # crossing faces
    ([[0.1, 0.1, 0], [0.2, 0.1, 0], [0.1, 0.2, 0]], 0),  # coplanar containment
    ([[0, 0, 2], [0, 0, 2], [0, 0, 2]], 2),  # degenerate triangle
])
def test_analytic_triangle_distances(other, expected, scale):
    inner = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float) * scale
    outer = np.array([other], dtype=float) * scale
    minimum, guard = minimum_wall_distance(inner, outer)
    assert minimum == pytest.approx(expected * scale, abs=guard)
    translated, _ = minimum_wall_distance(inner + np.array([31, -72, 113]) * scale,
                                          outer + np.array([31, -72, 113]) * scale)
    assert translated == pytest.approx(minimum, abs=guard)


def test_acceleration_matches_all_triangle_pairs():
    rng = np.random.default_rng(317)
    inner = rng.normal(size=(23, 3, 3))
    outer = rng.normal(size=(17, 3, 3)) + [0, 0, 4]
    a = np.repeat(inner, len(outer), axis=0)
    b = np.tile(outer, (len(inner), 1, 1))
    expected = np.sqrt(_paired_triangle_squared(a, b).min())
    minimum, guard = minimum_wall_distance(inner, outer)
    assert minimum == pytest.approx(expected, abs=guard)


def _cylinder_faces(gap, *, across=False):
    """Two coaxial cylindrical side faces, radii 20 and 20+gap.

    Along z the meridian map is exact; across (axis along x, off the z axis)
    it is loose and the sampled triangles must decide.
    """
    gmsh.model.add('clearance')
    if across:
        inner = gmsh.model.occ.addCylinder(0, 60, 0, 30, 0, 0, 20)
        outer = gmsh.model.occ.addCylinder(-5, 60, 0, 40, 0, 0, 20 + gap)
    else:
        inner = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 30, 20)
        outer = gmsh.model.occ.addCylinder(0, 0, -5, 0, 0, 40, 20 + gap)
    gmsh.model.occ.synchronize()

    def side(volume):
        faces = [tag for dim, tag in gmsh.model.getBoundary([(3, volume)], oriented=False)]
        return [tag for tag in faces if gmsh.model.getType(2, tag) == 'Cylinder']
    return side(inner), side(outer)


@pytest.mark.parametrize('across,method', [(False, 'meridian map of'), (True, 'projected blocks')])
def test_fitted_face_clearance_is_a_tight_lower_bound(across, method):
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        inner, outer = _cylinder_faces(6.0, across=across)
        exact = 6.0
        result = fitted_face_clearance(inner, outer, target_mm=0.05, required_mm=exact - 0.1)
        assert method in result['method']
        assert exact - 0.1 <= result['lowerBoundMm'] <= exact
    finally:
        gmsh.finalize()


def test_healthy_outer_is_returned_unchanged(monkeypatch):
    import hornlab_mesher.offset_envelope as owner
    phi = np.arange(16) * 2 * np.pi / 16
    z = np.linspace(0, 20, 9)
    r = 12 + z
    inner = np.stack((np.cos(phi)[:, None] * r, np.sin(phi)[:, None] * r,
                      np.broadcast_to(z, (16, 9))), axis=2)
    outer = _outer_offset_shell(inner, 1, full_circle=True)
    monkeypatch.setattr(owner, 'regularize_outer_offset',
                        lambda *a, **k: pytest.fail('healthy offsets must stay byte-identical'))
    result, enveloped = owner.exterior_outer_offset(inner, outer, 1, full_circle=True)
    assert result is outer and not enveloped


def test_both_coordinate_reversal_keeps_the_caller_fold_handling():
    phi = np.arange(8) * 2 * np.pi / 8
    z, r = np.array([0., 2., 1., 3.]), np.array([10., 12., 11., 13.])
    inner = np.stack((np.cos(phi)[:, None] * r, np.sin(phi)[:, None] * r,
                      np.broadcast_to(z, (8, 4))), axis=2)
    outer = inner.copy()
    result, enveloped = exterior_outer_offset(inner, outer, 2, full_circle=True)
    assert result is outer and not enveloped
    # A FREEFORM offset known to fold has no unchanged fallback: it keeps
    # the refusal it always had, naming the control and Mounting's Bare shell.
    from hornlab_mesher.errors import UserRefusal
    with pytest.raises(UserRefusal) as error:
        exterior_outer_offset(inner, outer, 2, full_circle=True, force=True)
    assert error.value.field == 'Wall thickness' and 'in Mounting' in error.value.next_step


def test_revolution_reuse_matches_every_row():
    phi = np.arange(24) * 2 * np.pi / 24
    z = np.linspace(0, 30, 13)
    r = 12 + 0.02 * z ** 2
    inner = np.stack((np.cos(phi)[:, None] * r, np.sin(phi)[:, None] * r,
                      np.broadcast_to(z, (24, 13))), axis=2)
    raw = _outer_offset_shell(inner, 4, full_circle=True)
    shared = regularize_outer_offset(inner, raw, 4, full_circle=True, revolution=True)
    each = regularize_outer_offset(inner, raw, 4, full_circle=True)
    np.testing.assert_allclose(shared, each, rtol=0, atol=1e-9)
