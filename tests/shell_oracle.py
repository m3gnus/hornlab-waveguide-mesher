"""Independent plane/segment oracle for shell tests; no production SAT calls.

Both segment directions must be tested. Strict crossings exclude coplanar
faces and endpoints/triangle edges; contact and overlap controls are explicit
fixtures in test_shell_facets. Candidate search uses one global radius.
"""
import numpy as np
from scipy.spatial import cKDTree


def strict_pairs(a, b):
    if not len(a) or not len(b):
        return []
    centers = b.mean(1)
    radius = np.linalg.norm(b-centers[:, None], axis=2).max()
    tree = cKDTree(centers)
    low, high = b.min(1), b.max(1)
    hits = []
    for i, face in enumerate(a):
        c = face.mean(0)
        js = tree.query_ball_point(c, np.linalg.norm(face-c, axis=1).max()+radius+1e-9)
        for j in js:
            if np.any(high[j] < face.min(0)-1e-9) or np.any(low[j] > face.max(0)+1e-9):
                continue
            if strict_crossing(face, b[j]):
                hits.append((i, j))
    return hits


def strict_crossing(a, b):
    for source, target in ((a, b), (b, a)):
        basis = np.column_stack((target[1]-target[0], target[2]-target[0]))
        normal = np.cross(basis[:, 0], basis[:, 1])
        length = np.linalg.norm(normal)
        if length == 0:
            continue
        normal /= length
        d = (source-target[0]) @ normal
        for k in range(3):
            l = (k+1) % 3
            # A plane crossing has endpoints strictly on opposite sides.
            if not ((d[k] < -1e-9 and d[l] > 1e-9) or (d[l] < -1e-9 and d[k] > 1e-9)):
                continue
            hit = source[k] + d[k]/(d[k]-d[l])*(source[l]-source[k])
            u, v = np.linalg.lstsq(basis, hit-target[0], rcond=None)[0]
            if u > 1e-9 and v > 1e-9 and u+v < 1-1e-9:
                return True
    return False
