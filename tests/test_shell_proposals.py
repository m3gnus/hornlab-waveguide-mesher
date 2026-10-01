"""Independent OCC obstacles protect all emitted diagonal proposals."""
import numpy as np
import pytest

from hornlab_mesher.mesher import MesherError
import hornlab_mesher.shell_facets as sf
from shell_oracle import strict_pairs
from test_facet_fit_guards import SHELL, gmsh_session


def _plane(gmsh, xyz, tag):
    occ = gmsh.model.occ
    vertices = [occ.addPoint(*p) for p in xyz]
    edges = [occ.addLine(vertices[k], vertices[(k+1)%3]) for k in range(3)]
    occ.addPlaneSurface([occ.addCurveLoop(edges)], tag=tag)


def _saddle(gmsh, obstacle=True):
    bore = np.array([[-1,-1,0],[1,-1,1],[-1,1,1],[1,1,0]], float)
    shell = SHELL[[1,2,4] if obstacle else [1,2]].copy()
    p = np.concatenate((bore, shell.reshape(-1,3)))
    t = np.concatenate(([[0,1,2],[3,2,1]], np.arange(4,len(p)).reshape(-1,3)))
    s = np.array([10,10,20,21]+([30] if obstacle else []))
    occ = gmsh.model.occ
    occ.addBSplineSurface([occ.addPoint(*v) for v in bore], 2, tag=10, degreeU=1, degreeV=1)
    for tag, face in zip(s[2:], shell):
        _plane(gmsh, face, int(tag))
    occ.synchronize()
    return p, t, s


def test_occ_disjoint_closure_obstacle_blocks_swap_before_final_scan(gmsh_session, monkeypatch):
    p, t, s = _saddle(gmsh_session)
    groups = {'inner':[10], 'outer':[20,21], 'mouth':[30]}
    before = t.copy()
    assert strict_pairs(p[t[:2]], p[t[2:4]])
    assert not strict_pairs(p[t[:2]], p[t[4:]])
    gap = min(gmsh_session.model.occ.getDistance(2,10,2,tag)[0] for tag in groups['outer'])
    assert .05 < gap < .5
    # Observe the actual accepted state inside repair. Final all-boundary
    # refusal/rollback alone would let removal of the proposal guard survive.
    repair = sf.repair_fitted_bore_facets
    observed = []
    def audit(*args, **kwargs):
        result = repair(*args, **kwargs)
        observed.append(True)
        assert not strict_pairs(p[t[:2]], p[t[4:]]), 'proposal introduced a closure crossing'
        np.testing.assert_array_equal(t, before)
        return result
    monkeypatch.setattr(sf, 'repair_fitted_bore_facets', audit)
    with pytest.raises(MesherError, match='remaining.*facet pairs'):
        sf.validate_shell_facets(p,t,s,np.ones(len(t),int),groups,.5)
    assert observed
    np.testing.assert_array_equal(t, before)


@pytest.mark.parametrize('constraint', ['none', 'physical', 'symmetry'])
def test_occ_physical_and_symmetry_constraints_on_real_proposals(gmsh_session, constraint):
    p, t, s = _saddle(gmsh_session, obstacle=False)
    if constraint == 'symmetry':
        # Map one swapped triangle's plane to x=0. Both old triangles remain
        # off that plane. Rigid transforms preserve independent crossings.
        a = p[[0,1,3]]
        normal = np.cross(a[1]-a[0],a[2]-a[0]); normal /= np.linalg.norm(normal)
        tangent = a[1]-a[0]; tangent /= np.linalg.norm(tangent)
        rotation = np.column_stack((normal,tangent,np.cross(normal,tangent)))
        p = (p-a[0]) @ rotation
        # Rebuild the real OCC fixture after the same rigid transform.
        gmsh_session.model.occ.remove(gmsh_session.model.getEntities(2), recursive=True)
        gmsh_session.model.occ.synchronize()
        occ = gmsh_session.model.occ
        occ.addBSplineSurface([occ.addPoint(*v) for v in p[:4]], 2, tag=10, degreeU=1, degreeV=1)
        for tag,face in zip(s[2:],p[t[2:]]):
            _plane(gmsh_session,face,int(tag))
        occ.synchronize()
        assert np.allclose(p[[0,1,3],0],0)
        assert not np.any(np.all(np.abs(p[t[:2],0])<1e-9,axis=1))
    before, xyz, tags = t.copy(), p.copy(), s.copy()
    physical = np.array([1,2,1,1]) if constraint == 'physical' else np.ones(len(t),int)
    assert strict_pairs(p[t[:2]],p[t[2:]])
    args = (p,t,s,physical,{'inner':[10],'outer':[20,21]},.5)
    if constraint == 'none':
        result = sf.validate_shell_facets(*args)
        assert result['changed_facets'] > 0 and result['pairs_after'] == 0
        assert not strict_pairs(p[t[:2]],p[t[2:]])
    else:
        with pytest.raises(MesherError,match='remaining.*facet pairs'):
            sf.validate_shell_facets(*args,symmetry_axes=('x',) if constraint=='symmetry' else ())
        np.testing.assert_array_equal(t,before)
    np.testing.assert_array_equal(p,xyz)
    np.testing.assert_array_equal(s,tags)


@pytest.mark.parametrize('neighbor,allowed', [
    ([[0,0,0],[-1,0,1],[0,-1,1]], True),  # shared vertex, opposite cone
    ([[0,0,0],[1,0,0],[0,0,1]], True),    # shared edge, distinct planes
    ([[0,0,0],[.3,.3,-1],[.3,.3,1]], False), # shared vertex, strict crossing
    ([[0,0,0],[1,0,0],[.25,.25,0]], False), # shared edge, coplanar overlap
    ([[.25,.25,-1],[.25,.25,1],[1,1,0]], False), # disjoint obstacle
])
def test_occ_neighbor_contacts_and_overlap_on_proposals(gmsh_session, neighbor, allowed):
    a = np.array([[0,0,0],[1,0,0],[0,1,0]],float)
    b = np.array(neighbor,float)
    p = a.copy(); ids=[]
    for vertex in b:
        same = np.flatnonzero(np.all(p==vertex,axis=1))
        if len(same):ids.append(int(same[0]))
        else:
            ids.append(len(p));p=np.vstack((p,vertex))
    # One replaced face and one neighboring obstacle on distinct CAD patches.
    t = np.array([[0,1,2],ids]);s=np.array([10,30])
    _plane(gmsh_session,a,10);_plane(gmsh_session,b,30)
    gmsh_session.model.occ.synchronize()
    if not allowed:
        if np.ptp(b[:,2]):
            assert strict_pairs(a[None],b[None])
        else:
            # Independent barycentric witness for the coplanar edge overlap.
            u,v=np.linalg.lstsq(np.column_stack((a[1]-a[0],a[2]-a[0])),b[2]-a[0],rcond=None)[0]
            assert u>0 and v>0 and u+v<1
    else:
        assert not strict_pairs(a[None],b[None])
    assert sf._proposal_guard(p,t,s,np.ones(2,int),(),set(),np.array([0]),t[:1]) == allowed
