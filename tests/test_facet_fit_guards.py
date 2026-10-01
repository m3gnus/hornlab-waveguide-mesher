"""Guards of the fitted-wall diagonal swap (facet_fit), on real OCC patches.

Both cases came from the independent review of the swap repair: a swap that
postprocessing would then drop as a sliver, and a swap that lowers the total
crossing count while crossing a shell facet the old pair did not cross.
"""

from __future__ import annotations

import numpy as np
import pytest

from hornlab_mesher.facet_fit import _FacetIndex, repair_fitted_bore_facets
from hornlab_mesher.mesh_repair import DEGENERATE_MIN_QUALITY, _remove_degenerate_triangles

# Five shell facets: the first four cross only the original diagonal's pair,
# the fifth crosses only the swapped pair.
SHELL = np.array([
    [-0.42700055369572043, -0.031480835157483816, 0.7257360876040699],
    [-0.6879880417547125, -0.3892476411390369, 0.30337541110265365],
    [-0.6022759707130669, -0.05061071901882963, 0.6819011993292481],
    [0.5452786045331668, -0.03261970657245997, 0.6915516205461479],
    [0.6079549019578252, 0.013521825691366295, 0.6086414554642766],
    [0.513360122866079, -0.02318951983641619, 0.83908252009809],
    [0.00012400533345220156, 0.22085591041383526, 1.0522583932150174],
    [0.11160494090528499, 0.2487705623778963, 0.8286417186764065],
    [-0.3042536321075045, 0.44819091295221036, 0.6696166344343025],
    [-0.327218091060351, -0.4781619220741312, 0.5325009499269099],
    [-0.34405660380054376, -0.4515281051948513, 0.38712009475859077],
    [-0.6383968204095207, -0.34148361364828844, 0.5864567112227973],
    [0.35119525223703213, -0.11266678619896882, 0.11249093951541134],
    [0.21344358867887758, -0.12819105901346906, 0.27847639243060684],
    [0.27758101833521515, -0.11299686005149912, 0.35764192337006556]
]).reshape(-1, 3, 3)


@pytest.fixture
def gmsh_session():
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        yield gmsh
    finally:
        gmsh.finalize()


def test_a_swap_is_never_a_facet_postprocessing_drops(gmsh_session):
    gmsh = gmsh_session
    p = np.array([[-1, 0, 0], [0, .0003, 0], [0, -1, 0], [1, 0, 0],
                  [-.1, -.25, -1], [.1, -.25, -1], [0, -.25, 1]], float)
    t = np.array([[0, 2, 1], [3, 1, 2], [4, 5, 6]])
    tags = np.array([10, 10, 20])
    occ = gmsh.model.occ
    points = [occ.addPoint(*xyz) for xyz in p[:4]]
    lines = [occ.addLine(points[i], points[j]) for i, j in [(0, 2), (2, 3), (3, 1), (1, 0)]]
    occ.addPlaneSurface([occ.addCurveLoop(lines)], tag=10)
    occ.synchronize()

    repair_fitted_bore_facets(p, t, tags, {"inner": [10], "outer": [20]})

    # The only available swap makes a needle below the postprocessing
    # threshold, so the pair must be left as it was and survive intact.
    np.testing.assert_array_equal(t[:2], [[0, 2, 1], [3, 1, 2]])
    _, _, removed = _remove_degenerate_triangles(p, t, tags, min_quality=DEGENERATE_MIN_QUALITY)
    assert removed == 0


def test_a_swap_never_crosses_a_shell_facet_the_old_pair_did_not(gmsh_session):
    gmsh = gmsh_session
    bore = np.array([[-1, -1, 0], [1, -1, 1], [-1, 1, 1], [1, 1, 0]], float)
    occ = gmsh.model.occ
    occ.addBSplineSurface([occ.addPoint(*xyz) for xyz in bore], 2, tag=10, degreeU=1, degreeV=1)
    occ.synchronize()
    points = np.concatenate([bore, SHELL.reshape(-1, 3)])
    triangles = np.concatenate([[[0, 1, 2], [3, 2, 1]], np.arange(4, len(points)).reshape(-1, 3)])
    tags = np.array([10, 10] + [20] * len(SHELL))
    index = _FacetIndex(SHELL)
    before = {j for _, j in index.hits(points[triangles[:2]])}
    assert before == {0, 1, 2, 3}

    repair_fitted_bore_facets(points, triangles, tags, {"inner": [10], "outer": [20]})

    after = {j for _, j in index.hits(points[triangles[:2]])}
    assert after <= before
