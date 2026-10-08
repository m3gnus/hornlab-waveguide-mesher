"""Signed curvatures follow the emitted oriented fundamental forms."""
import math

import numpy as np
import pytest

from hornlab_mesher.preview import build_preview_geometry, PreviewOptionsV1
from test_native_mouth_roundover import config


def fundamental_forms(position, first, second, normal):
    """Independent meridian/azimuth fundamental forms at azimuth zero."""
    z, radius = position
    dz, dr = first
    ddz, ddr = second
    t = np.array([dr, 0., dz])
    phi = np.array([0., radius, 0.])
    tt = np.array([ddr, 0., ddz])
    pp = np.array([-radius, 0., 0.])
    metric = np.diag([t @ t, phi @ phi])
    second_form = np.diag([tt @ normal, pp @ normal])
    eigenvalues = np.linalg.eigvalsh(np.linalg.solve(metric, second_form))
    principal = eigenvalues[1] if abs(eigenvalues[1]) >= abs(eigenvalues[0]) else eigenvalues[0]
    return eigenvalues.mean(), principal


@pytest.mark.parametrize("role", ["horn.inner", "horn.outer", "wall.rear_return"])
def test_body_and_rear_curvatures_match_oriented_fundamental_forms(role):
    c = config()
    preview = build_preview_geometry(c, PreviewOptionsV1(include_curvature=True))
    surface = next(s for s in preview.surfaces if s.role == role)
    normal = surface.normals[0]
    position = surface.positions[0]
    if role == "wall.rear_return":
        first, second = (1., 0.), (0., 0.)
    else:
        p = c["profile"]
        slope = math.tan(math.radians(p["a0_deg"]))
        second_radius = (math.tan(math.radians(p["a_deg"]))**2-slope**2)/p["r0_mm"]
        first = np.array([1., slope])
        second = np.array([0., second_radius])
        if role == "horn.outer":
            # Normal offset: P_outer=P+w*(1,-slope)/sqrt(1+slope²)
            # in radial/axial order. Differentiate its tangent with respect
            # to base z; only its normal component enters the second form.
            curvature = second_radius/(1+slope*slope)**1.5
            factor = 1-c["mesh"]["wall_thickness_mm"]*curvature
            first *= factor
            second *= factor
    expected = fundamental_forms((position[2],position[0]),first,second,normal)
    np.testing.assert_allclose([surface.curvature_mean[0],surface.curvature_principal[0]],expected,rtol=1e-12,atol=1e-14)


@pytest.mark.parametrize("role,outer", [("horn.inner",False),("horn.outer",True)])
@pytest.mark.parametrize("station", ["turn", "forward", "backward", "end"])
def test_lip_curvatures_match_oriented_fundamental_forms(role,outer,station):
    c = config()
    preview = build_preview_geometry(c, PreviewOptionsV1(include_curvature=True))
    surface = next(s for s in preview.surfaces if s.role == role)
    radius = c["mesh"]["mouth_roundover_radius_mm"]-(c["mesh"]["wall_thickness_mm"] if outer else 0)
    p = c["profile"]
    a, a0 = math.tan(math.radians(p["a_deg"])), math.tan(math.radians(p["a0_deg"]))
    root = math.sqrt(p["r0_mm"]**2+2*p["r0_mm"]*p["L_mm"]*a0+p["L_mm"]**2*a*a)
    beta = math.atan((p["r0_mm"]*a0+p["L_mm"]*a*a)/root)
    circle_radius = c["mesh"]["mouth_roundover_radius_mm"]
    center = np.array([p["L_mm"]-circle_radius*math.sin(beta),root+circle_radius*math.cos(beta)])
    rz = surface.positions[:,[2,0]]
    azimuth_zero = (abs(surface.positions[:,1])<1e-12)&(surface.positions[:,0]>0)
    on_circle = abs(np.linalg.norm(rz-center,axis=1)-radius)<1e-9
    candidates = np.flatnonzero(azimuth_zero&on_circle)
    angle = np.arctan2(rz[candidates,0]-center[0],-(rz[candidates,1]-center[1]))
    target = {"turn":(beta+math.pi/2)/2,"forward":math.pi/2,"backward":3*math.pi/4,"end":math.pi}[station]
    selected = np.argmin(abs(angle-target))
    index, theta = candidates[selected], angle[selected]
    position,normal = surface.positions[index],surface.normals[index]
    expected = fundamental_forms((position[2],position[0]),
        (radius*math.cos(theta),radius*math.sin(theta)),
        (-radius*math.sin(theta),radius*math.cos(theta)),normal)
    np.testing.assert_allclose([surface.curvature_mean[index],surface.curvature_principal[index]],expected,rtol=1e-12,atol=1e-14)
    assert surface.curvature_principal[index] < 0 if not outer else surface.curvature_principal[index] > 0


@pytest.mark.parametrize("radius,slope,second", [(2.,0.,.8),(10.,.5,.01)])
def test_normal_reversal_reverses_signed_curvatures(radius,slope,second):
    from hornlab_mesher.preview.mouth_roundover import _signed_curvatures
    normal = np.array([-1.,0.,slope])/math.sqrt(1+slope*slope)
    curvature = second/(1+slope*slope)**1.5
    mean,principal = _signed_curvatures(curvature,normal[0],radius,inward=True)
    reverse_mean,reverse_principal = _signed_curvatures(curvature,-normal[0],radius,inward=False)
    oracle = fundamental_forms((0.,radius),(1.,slope),(0.,second),normal)
    reverse_oracle = fundamental_forms((0.,radius),(1.,slope),(0.,second),-normal)
    np.testing.assert_allclose([mean,principal],oracle,rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose([reverse_mean,reverse_principal],reverse_oracle,rtol=1e-14,atol=1e-14)
    np.testing.assert_array_equal([reverse_mean,reverse_principal],[-mean,-principal])
    if radius==2.:assert principal<0


def test_flat_caps_and_annulus_curvature_stays_zero():
    preview = build_preview_geometry(config(), PreviewOptionsV1(include_curvature=True))
    for surface in preview.surfaces:
        if surface.role in {"source_cap","wall.rear_cap","mouth_rim"}:
            np.testing.assert_array_equal(surface.curvature_mean,0.)
            np.testing.assert_array_equal(surface.curvature_principal,0.)


@pytest.mark.parametrize("inward", [True,False])
def test_equal_opposite_principal_magnitudes_choose_positive(inward):
    from hornlab_mesher.preview.mouth_roundover import _signed_curvatures
    normal = np.array([-1. if inward else 1.,0.,0.])
    mean,principal = _signed_curvatures(.5,normal[0],2.,inward=inward)
    expected = fundamental_forms((0.,2.),(1.,0.),(0.,.5),normal)
    np.testing.assert_array_equal([mean,principal],expected)
    assert mean==0 and principal==.5
