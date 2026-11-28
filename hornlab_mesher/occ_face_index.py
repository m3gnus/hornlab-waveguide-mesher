"""Forward tessellation of trimmed OCC faces, for fast certified distance bounds.

The mirror test asks, for thousands of points, how far each lies from the
trimmed faces of an imported model. OCC answers that one point and one face
at a time (``getClosestPoint`` and a trim query), which is what made the test
take seconds. This module answers most of those questions without OCC, and
says so only when it can prove the answer:

* Each face is tessellated forward, from its own parameter grid
  (``getValue`` is a fraction of a microsecond per point), refined per
  direction where its chords sag, and the chord error of every cell is
  *measured* (not proven): the surface is evaluated again at the cell's
  centre, its two triangles' centroids and two edge midpoints (and, on a
  non-analytic face, a 3 x 3 lattice per cell); the largest gap from the
  planar triangles, doubled and raised to its neighbours', is that cell's
  ``delta``.
* The trim comes from the face's own boundary curves, reparametrized onto the
  face (its p-curves), as polygons in parameter space with a measured margin.
  A grid cell is *inside* only when its four corners are inside and no
  boundary curve passes within the margin of it -- so a hole smaller than a
  cell still makes its cells boundary cells. The polygons must agree with the
  area OCC reports for the face, or the face keeps no inside cells.

For a point ``m``, a triangle at distance ``d``, its cell's ``delta`` and
``nudge`` (how far the mirror test's trim query, which nudges a projection by
1e-3 of the parameter range, can move a point in space there):

* ``d - delta - nudge > tolerance`` for every triangle of the face: a miss
  (by the measured ``delta``: an error there can only reject, never accept).
* On a **plane** only, a hit is proven: the triangle's closest point is the
  perpendicular foot itself, within tolerance, and certainly inside the trim
  (the caller also has OCC count the feet inside).
* A curved face within reach is only a candidate: OCC's projection onto
  that face confirms a hit (``step_prepare``).
* Anything else is left to OCC's exact projection.

gmsh and scipy are imported lazily, as everywhere in this package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._lazy_gmsh import LazyGmsh

gmsh: Any = LazyGmsh(globals())

#: Target chord error, as a fraction of the mirror tolerance.
DELTA_TARGET_FRACTION = 1.0 / 4.0
#: Largest cell, as a fraction of the model diagonal.
CELL_SIZE_REL = 1.0 / 48.0
#: Cells per parameter direction, before and after refinement.
MIN_CELLS = 12
MAX_CELLS = 1024
#: Cells per face at most (the grid stops refining there).
MAX_FACE_CELLS = 1 << 16
#: Points per boundary curve for the trim polygons (refined once if coarse).
CURVE_SAMPLES = (33, 129, 513)
#: Safety factor on the measured chord error.
DELTA_SAFETY = 2.0
#: Surface types whose chord error the 5-probe measure follows (smooth,
#: closed-form curvature); every other face is re-sampled on a lattice.
ANALYTIC_TYPES = frozenset({"Plane", "Cylinder", "Cone", "Sphere", "Torus"})
#: Lattice of extra chord probes per cell on non-analytic faces (cell-local).
LATTICE = (1.0 / 6.0, 0.5, 5.0 / 6.0)


# -- planar point-in-polygon in parameter space --------------------------------


def _segments(loops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    starts = [loop for loop in loops if len(loop) >= 2]
    if not starts:
        return np.zeros((0, 2)), np.zeros((0, 2))
    a = np.concatenate([loop for loop in starts])
    b = np.concatenate([np.roll(loop, -1, axis=0) for loop in starts])
    return a, b


def _crossings(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Even-odd inside flags for ``points`` against closed segment soup ``a -> b``."""
    inside = np.zeros(len(points), dtype=bool)
    if not len(a):
        return inside
    chunk = max(1, 2_000_000 // max(1, len(a)))
    ay, by = a[:, 1], b[:, 1]
    dy = by - ay
    safe = np.where(dy == 0.0, 1.0, dy)
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk]
        py = p[:, 1:2]
        straddle = (ay > py) != (by > py)
        x_at = a[:, 0] + (b[:, 0] - a[:, 0]) * (py - ay) / safe
        hit = straddle & (p[:, 0:1] < x_at)
        inside[start:start + chunk] = (np.count_nonzero(hit, axis=1) % 2) == 1
    return inside


def segment_pairs(points: np.ndarray, a: np.ndarray, b: np.ndarray, reach: float,
                  tree: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every (point, segment) pair within ``reach``, with its distance.

    Works in any dimension. Segments are found through a KD-tree of their
    midpoints (``tree``, built here when not given) searched to ``reach``
    plus half the longest segment, so no segment within ``reach`` is missed.
    Returns point indices, segment indices and distances.
    """
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0))
    finite = np.all(np.isfinite(points), axis=1) if len(points) else np.zeros(0, dtype=bool)
    if not len(a) or not np.any(finite):
        return empty
    from scipy.spatial import cKDTree

    ab = b - a
    length = np.sqrt(np.einsum("ij,ij->i", ab, ab))
    if tree is None:
        tree = cKDTree(0.5 * (a + b))
    query = np.where(finite[:, None], points, 0.0)
    rows = tree.query_ball_point(query, (reach + 0.5 * float(np.max(length))) * (1.0 + 1e-9))
    sizes = np.fromiter((len(row) if keep else 0 for keep, row in zip(finite, rows)),
                        dtype=np.int64, count=len(rows))
    if not sizes.sum():
        return empty
    pid = np.repeat(np.arange(len(points)), sizes)
    sid = np.fromiter((i for keep, row in zip(finite, rows) if keep for i in row),
                      dtype=np.int64, count=int(sizes.sum()))
    seg = ab[sid]
    den = np.einsum("ij,ij->i", seg, seg)
    rel = points[pid] - a[sid]
    t = np.clip(np.einsum("ij,ij->i", rel, seg) / np.where(den > 0, den, 1.0), 0.0, 1.0)
    gap = rel - t[:, None] * seg
    return pid, sid, np.sqrt(np.einsum("ij,ij->i", gap, gap))


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray, reach: float,
                      tree: Any = None) -> np.ndarray:
    """Distance from each point to the nearest segment, exact up to ``reach``
    (``inf`` beyond it)."""
    out = np.full(len(points), np.inf)
    pid, _sid, dist = segment_pairs(points, a, b, reach, tree)
    np.minimum.at(out, pid, dist)
    return out


def _crossings_grid(u: np.ndarray, v: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``_crossings`` for every node of the grid ``u x v``, row by row.

    The same even-odd rule: a node is inside when an odd number of segment
    crossings of its row lie strictly to its right.
    """
    inside = np.zeros((len(u), len(v)), dtype=bool)
    if not len(a):
        return inside
    ay, by = a[:, 1], b[:, 1]
    dy = by - ay
    safe = np.where(dy == 0.0, 1.0, dy)
    for j, row in enumerate(v):
        straddle = (ay > row) != (by > row)
        if not np.any(straddle):
            continue
        x_at = np.sort(a[straddle, 0] + (b[straddle, 0] - a[straddle, 0])
                       * (row - ay[straddle]) / safe[straddle])
        right = len(x_at) - np.searchsorted(x_at, u, side="right")
        inside[:, j] = (right % 2) == 1
    return inside


@dataclass
class FaceTrim:
    """A face's trim as parameter-space polygons, with an honest margin.

    ``classify`` is certain only farther than ``margin`` from every polygon
    edge: within it, the polygon's own approximation of the p-curves could
    put a point on the wrong side.
    """

    surface: int
    loops: list[np.ndarray]
    margin: float
    ok: bool
    #: Set once the polygons have matched OCC's area for the face.
    verified: bool = False

    def segments(self) -> tuple[np.ndarray, np.ndarray, Any]:
        """The polygon edges ``a -> b`` and a KD-tree of their midpoints (cached)."""
        cached = getattr(self, "_segments_cache", None)
        if cached is None:
            a, b = _segments(self.loops)
            tree = None
            if len(a):
                from scipy.spatial import cKDTree

                tree = cKDTree(0.5 * (a + b))
            cached = (a, b, tree)
            self._segments_cache = cached
        return cached

    @classmethod
    def build(cls, surface: int) -> FaceTrim:
        try:
            _loop_tags, loop_curves = gmsh.model.occ.getCurveLoops(surface)
        except Exception:
            return cls(surface, [], math.inf, False)
        loops: list[np.ndarray] = []
        error = 0.0
        for curves in loop_curves:
            variants: list[list[np.ndarray]] = []
            for curve in curves:
                curve = abs(int(curve))
                options, curve_error = _pcurve_options(curve, surface)
                if not options:
                    return cls(surface, [], math.inf, False)
                error = max(error, curve_error)
                variants.append(options)
            if not variants:
                continue
            loops.append(_chain(variants))
        if not loops:
            return cls(surface, [], math.inf, False)
        span = _uv_span(surface)
        margin = 4.0 * error + 1.0e-9 * span
        return cls(surface, loops, margin, True)

    def classify_grid(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``classify`` for every node of the grid ``u x v``: arrays (len(u), len(v))."""
        if not self.ok:
            shape = (len(u), len(v))
            return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
        a, b, tree = self.segments()
        inside = _crossings_grid(u, v, a, b)
        nodes = np.stack(np.meshgrid(u, v, indexing="ij"), axis=-1).reshape(-1, 2)
        certain = ~(_segment_distance(nodes, a, b, self.margin, tree) <= self.margin)
        return inside, certain.reshape(len(u), len(v))

    def classify(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(inside, certain)`` flags for parameter points ``(N, 2)``."""
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        if not self.ok:
            return np.zeros(len(uv), dtype=bool), np.zeros(len(uv), dtype=bool)
        a, b, tree = self.segments()
        inside = _crossings(uv, a, b)
        certain = ~(_segment_distance(uv, a, b, self.margin, tree) <= self.margin)
        return inside, certain


def _uv_span(surface: int) -> float:
    try:
        lo, hi = gmsh.model.getParametrizationBounds(2, surface)
        return float(math.hypot(hi[0] - lo[0], hi[1] - lo[1]))
    except Exception:
        return 1.0


def _pcurve_options(curve: int, surface: int) -> tuple[list[np.ndarray], float]:
    """The curve's p-curve(s) on the face as polylines, and their chord error.

    A seam curve has two p-curves on a periodic face (``which`` 0 and 1);
    both are offered and the loop chaining picks the one that connects.
    """
    try:
        lo, hi = gmsh.model.getParametrizationBounds(1, curve)
        t0, t1 = float(lo[0]), float(hi[0])
    except Exception:
        return [], math.inf
    options: list[np.ndarray] = []
    error = 0.0
    for count in CURVE_SAMPLES:
        params = np.linspace(t0, t1, count)
        mids = 0.5 * (params[1:] + params[:-1])
        options, error = [], 0.0
        for which in (0, 1):
            try:
                uv = np.asarray(gmsh.model.reparametrizeOnSurface(
                    1, curve, params.tolist(), surface, which), dtype=float).reshape(-1, 2)
                uv_mid = np.asarray(gmsh.model.reparametrizeOnSurface(
                    1, curve, mids.tolist(), surface, which), dtype=float).reshape(-1, 2)
            except Exception:
                continue
            if not (np.all(np.isfinite(uv)) and np.all(np.isfinite(uv_mid))):
                continue
            # Chord error: each true mid-point against its polyline segment.
            seg_a, seg_b = uv[:-1], uv[1:]
            ab = seg_b - seg_a
            denom = np.einsum("ij,ij->i", ab, ab)
            t = np.clip(np.einsum("ij,ij->i", uv_mid - seg_a, ab) / np.where(denom > 0, denom, 1.0), 0, 1)
            gap = np.linalg.norm(uv_mid - (seg_a + t[:, None] * ab), axis=1)
            error = max(error, float(np.max(gap)) if len(gap) else 0.0)
            if not any(np.array_equal(uv, other) for other in options):
                options.append(uv)
        if options and error <= 1.0e-4 * _uv_span(surface):
            break
    return options, error


def _chain(variants: list[list[np.ndarray]]) -> np.ndarray:
    """Join a loop's curves end to end, choosing each one's direction and p-curve.

    OCC lists a loop's curves in order but not their direction on the face,
    so each curve is placed in whichever direction (and, for a seam, on
    whichever side) starts nearest the previous curve's end. Both directions
    of the first curve are tried and the tighter chain is kept. A gap that
    remains (a degenerate edge gmsh does not list, such as a sphere's pole)
    is closed by a straight segment -- which is that edge's image.
    """
    best: tuple[float, list[np.ndarray]] | None = None
    first_options = [option[::step] for option in variants[0] for step in (1, -1)]
    for first in first_options:
        chain = [first]
        gap = 0.0
        for options in variants[1:]:
            end = chain[-1][-1]
            candidates = [option[::step] for option in options for step in (1, -1)]
            distances = [float(np.linalg.norm(candidate[0] - end)) for candidate in candidates]
            pick = int(np.argmin(distances))
            gap += distances[pick]
            chain.append(candidates[pick])
        gap += float(np.linalg.norm(chain[-1][-1] - chain[0][0]))
        if best is None or gap < best[0]:
            best = (gap, chain)
    assert best is not None
    return np.concatenate(best[1])


# -- trim membership with OCC as the fallback ----------------------------------


def params_inside(
    surface: int, uv: list[float], trim: FaceTrim | None, exact_one: Any
) -> list[bool]:
    """Trim flags for concatenated ``(u, v)`` pairs, as OCC would give them.

    The polygon answers every point it is certain of; ``exact_one(surface, u,
    v)`` (OCC, one point) answers the rest. Polygons not yet checked against
    OCC's area for the face must also match OCC's own inside count for the
    whole batch -- one call -- or every point is asked of OCC one at a time,
    exactly as without this shortcut.
    """
    count = len(uv) // 2
    if trim is not None and trim.ok and trim.verified:
        # The polygons already matched OCC's area for this face.
        flags = trim.classify(np.asarray(uv, dtype=float).reshape(-1, 2))
        answer = flags[0].tolist()
        for index in np.flatnonzero(~flags[1]):
            answer[index] = exact_one(surface, uv[2 * index], uv[2 * index + 1])
        return answer
    try:
        total = int(gmsh.model.isInside(2, surface, uv, parametric=True))
    except Exception:
        total = -1
    if total == count:
        return [True] * count
    if total == 0:
        return [False] * count
    if trim is not None and trim.ok and total > 0:
        pairs = np.asarray(uv, dtype=float).reshape(-1, 2)
        inside, certain = trim.classify(pairs)
        flags = inside.tolist()
        for index in np.flatnonzero(~certain):
            flags[index] = exact_one(surface, uv[2 * index], uv[2 * index + 1])
        if sum(flags) == total:
            return flags
        trim.ok = False
    return [exact_one(surface, uv[2 * index], uv[2 * index + 1]) for index in range(count)]


# -- forward tessellation -------------------------------------------------------


@dataclass
class FaceMesh:
    """One face's tessellation and, per triangle, the bounds that certify it.

    ``delta`` is the *measured* gap between a triangle and the surface patch
    it stands for (sampled, doubled, and taken over the cell's neighbours;
    on a non-analytic face also re-sampled on a 3 x 3 lattice per cell). It
    is not a proof, so it only ever decides misses -- a too small ``delta``
    can turn a hit into a miss, never a miss into a hit;
    ``nudge`` bounds how far the trim query's parameter nudge can move a
    point of that patch in space.
    """

    surface: int
    triangles: np.ndarray        # (T, 3, 3), inside and boundary cells
    inside: np.ndarray           # (T,) cell fully inside the trim
    delta: np.ndarray            # (T,)
    nudge: np.ndarray            # (T,)
    uv: np.ndarray               # (T, 3, 2) parameter corners
    trim: FaceTrim | None = None
    planar: bool = False


def _grid_values(surface: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    uu, vv = np.meshgrid(u, v, indexing="ij")
    flat = np.column_stack([uu.ravel(), vv.ravel()]).ravel().tolist()
    return np.asarray(gmsh.model.getValue(2, surface, flat), dtype=float).reshape(len(u), len(v), 3)


def _point_values(surface: int, uv: np.ndarray) -> np.ndarray:
    return np.asarray(
        gmsh.model.getValue(2, surface, uv.ravel().tolist()), dtype=float
    ).reshape(uv.shape[:-1] + (3,))


def _dilate_max(values: np.ndarray) -> np.ndarray:
    """Each cell's value raised to the largest among its 8 neighbours."""
    nu, nv = values.shape
    padded = np.full((nu + 2, nv + 2), -np.inf)
    padded[1:-1, 1:-1] = values
    out = values.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            out = np.maximum(out, padded[1 + di:nu + 1 + di, 1 + dj:nv + 1 + dj])
    return out


def _trim_cells(trim: FaceTrim, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(inside, keep)`` per cell of the grid ``u x v``.

    Inside: four corners certainly inside and no boundary curve within the
    polygon margin of the cell. Kept: anything not certainly outside.
    """
    nu, nv = len(u) - 1, len(v) - 1
    if not trim.ok:
        return np.zeros((nu, nv), dtype=bool), np.ones((nu, nv), dtype=bool)
    u0, v0 = float(u[0]), float(v[0])
    du, dv = float(u[1] - u[0]), float(v[1] - v[0])
    if trim.margin < 0.25 * min(du, dv):
        # A node within the margin of a boundary curve has that curve in one
        # of its four cells, which are then crossed: the even-odd rule alone
        # decides every node that matters.
        a, b, _tree = trim.segments()
        corner_in = _crossings_grid(u, v, a, b)
        corner_certain = np.ones_like(corner_in)
    else:
        corner_in, corner_certain = trim.classify_grid(u, v)
    crossed = np.zeros((nu, nv), dtype=bool)
    step = 0.25 * min(du, dv)
    reach = trim.margin + 1.0e-9 * (abs(float(u[-1])) + abs(float(v[-1])) + du + dv)
    for loop in trim.loops:
        closed = np.vstack([loop, loop[:1]])
        seg = np.diff(closed, axis=0)
        pieces = np.maximum(1, np.ceil(np.linalg.norm(seg, axis=1) / step)).astype(int)
        owner = np.repeat(np.arange(len(seg)), pieces)
        offset = np.arange(int(pieces.sum())) - np.repeat(np.cumsum(pieces) - pieces, pieces)
        dense = closed[owner] + (offset / pieces[owner])[:, None] * seg[owner]
        # Each polyline point marks the cells within the polygon's margin of
        # it: the true p-curve is no farther than that from its polyline.
        # Every cell the reach box touches, however wide it is.
        iu_lo = np.clip(np.floor((dense[:, 0] - reach - u0) / du).astype(int), 0, nu - 1)
        iu_hi = np.clip(np.floor((dense[:, 0] + reach - u0) / du).astype(int), 0, nu - 1)
        iv_lo = np.clip(np.floor((dense[:, 1] - reach - v0) / dv).astype(int), 0, nv - 1)
        iv_hi = np.clip(np.floor((dense[:, 1] + reach - v0) / dv).astype(int), 0, nv - 1)
        # Mark every cell of each reach box at once (a 2D difference array:
        # linear in points and cells however wide the boxes).
        marks = np.zeros((nu + 1, nv + 1), dtype=np.int64)
        np.add.at(marks, (iu_lo, iv_lo), 1)
        np.add.at(marks, (iu_hi + 1, iv_lo), -1)
        np.add.at(marks, (iu_lo, iv_hi + 1), -1)
        np.add.at(marks, (iu_hi + 1, iv_hi + 1), 1)
        crossed |= marks.cumsum(axis=0).cumsum(axis=1)[:nu, :nv] > 0
    sure_in = corner_in & corner_certain
    sure_out = ~corner_in & corner_certain
    all_in = sure_in[:-1, :-1] & sure_in[1:, :-1] & sure_in[:-1, 1:] & sure_in[1:, 1:]
    all_out = sure_out[:-1, :-1] & sure_out[1:, :-1] & sure_out[:-1, 1:] & sure_out[1:, 1:]
    return all_in & ~crossed, ~(all_out & ~crossed)


def build_face_mesh(
    surface: int, trim: FaceTrim, *, cell_size: float, delta_target: float,
    area: float | None = None, nudge_rel: float,
) -> FaceMesh | None:
    """Tessellate one face; ``None`` when OCC cannot evaluate it.

    The grid starts at ``cell_size`` in space and is refined (each direction
    doubled, up to ``MAX_CELLS``) while the measured chord error of most
    cells is above ``delta_target`` or while too little of the face's area
    lies in inside cells (a narrow face needs cells narrower than itself).
    """
    try:
        lo, hi = gmsh.model.getParametrizationBounds(2, surface)
        u0, v0, u1, v1 = float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])
    except Exception:
        return None
    if not (np.isfinite([u0, v0, u1, v1]).all() and u1 > u0 and v1 > v0):
        return None
    try:
        coarse = _grid_values(surface, np.linspace(u0, u1, 9), np.linspace(v0, v1, 9))
        kind = gmsh.model.getType(2, surface)
    except Exception:
        return None
    planar = kind == "Plane"
    length_u = float(np.max(np.sum(np.linalg.norm(np.diff(coarse, axis=0), axis=2), axis=0)))
    length_v = float(np.max(np.sum(np.linalg.norm(np.diff(coarse, axis=1), axis=2), axis=1)))
    nu = int(min(MAX_CELLS, max(MIN_CELLS, math.ceil(length_u / cell_size))))
    nv = int(min(MAX_CELLS, max(MIN_CELLS, math.ceil(length_v / cell_size))))
    thin_rounds = 0
    while True:
        u = np.linspace(u0, u1, nu + 1)
        v = np.linspace(v0, v1, nv + 1)
        du, dv = (u1 - u0) / nu, (v1 - v0) / nv
        try:
            nodes = _grid_values(surface, u, v)
            uu, vv = np.meshgrid(u[:-1] + 0.5 * du, v[:-1] + 0.5 * dv, indexing="ij")
            probe = np.stack([
                np.stack([uu, vv], axis=-1),                         # cell centre
                np.stack([uu + du / 6.0, vv - dv / 6.0], axis=-1),   # centroid of (00, 10, 11)
                np.stack([uu - du / 6.0, vv + dv / 6.0], axis=-1),   # centroid of (00, 11, 01)
                np.stack([uu, vv - 0.5 * dv], axis=-1),              # mid of edge 00-10
                np.stack([uu - 0.5 * du, vv], axis=-1),              # mid of edge 00-01
            ])
            actual = _point_values(surface, probe)
        except Exception:
            return None
        p00, p10, p01, p11 = nodes[:-1, :-1], nodes[1:, :-1], nodes[:-1, 1:], nodes[1:, 1:]
        planar_guess = np.stack([
            0.5 * (p00 + p11), (p00 + p10 + p11) / 3.0, (p00 + p11 + p01) / 3.0,
            0.5 * (p00 + p10), 0.5 * (p00 + p01),
        ])
        gaps = np.linalg.norm(actual - planar_guess, axis=-1)
        gap = np.max(gaps, axis=0)   # (nu, nv)
        sag_u, sag_v = gaps[3], gaps[4]
        if not np.all(np.isfinite(gap)):
            return None
        cell_in, cell_keep = _trim_cells(trim, u, v)
        cell_area = 0.5 * (
            np.linalg.norm(np.cross(p10 - p00, p11 - p00), axis=-1)
            + np.linalg.norm(np.cross(p11 - p00, p01 - p00), axis=-1)
        )
        # Narrow faces need cells narrower than themselves (twice at most).
        thin = (trim.ok and area is not None and area > 0.0 and thin_rounds < 2
                and float(np.sum(cell_area[cell_in])) < 0.5 * area)
        thin_rounds += int(thin)
        grow_u = grow_v = thin
        if not planar and np.any(cell_keep):
            # Refine while most cells' chord error is above target, in the
            # direction(s) whose chords sag most (a few bad cells keep their
            # own larger delta).
            limit = delta_target / DELTA_SAFETY
            if float(np.percentile(gap[cell_keep], 95)) > limit:
                su = float(np.percentile(sag_u[cell_keep], 95))
                sv = float(np.percentile(sag_v[cell_keep], 95))
                grow_u |= su >= 0.5 * sv
                grow_v |= sv >= 0.5 * su
        grow_u &= nu < MAX_CELLS and 2 * nu * nv <= MAX_FACE_CELLS
        grow_v &= nv < MAX_CELLS and 2 * nu * nv <= MAX_FACE_CELLS
        if grow_u or grow_v:
            nu, nv = (2 * nu if grow_u else nu), (2 * nv if grow_v else nv)
            continue
        break

    if not planar and kind not in ANALYTIC_TYPES:
        # Features narrower than a cell can hide between the five probes: on
        # a non-analytic face sample each cell again on a 3 x 3 lattice.
        try:
            for a in LATTICE:
                for b in LATTICE:
                    lattice = np.stack([u[:-1, None] + a * du + 0.0 * v[None, :-1],
                                        0.0 * u[:-1, None] + v[None, :-1] + b * dv], axis=-1)
                    value = _point_values(surface, lattice)
                    if a >= b:      # triangle (00, 10, 11)
                        guess = p00 + a * (p10 - p00) + b * (p11 - p10)
                    else:           # triangle (00, 11, 01)
                        guess = p00 + b * (p01 - p00) + a * (p11 - p01)
                    gap = np.maximum(gap, np.linalg.norm(value - guess, axis=-1))
        except Exception:
            return None
        if not np.all(np.isfinite(gap)):
            return None
    delta = np.zeros_like(gap) if planar else DELTA_SAFETY * _dilate_max(gap)
    # The trim query nudges a projection by ``nudge_rel`` of the parameter
    # range in u and in v; bound that step in space from the local edges.
    speed_u = np.maximum(np.linalg.norm(p10 - p00, axis=-1), np.linalg.norm(p11 - p01, axis=-1)) / du
    speed_v = np.maximum(np.linalg.norm(p01 - p00, axis=-1), np.linalg.norm(p11 - p10, axis=-1)) / dv
    nudge = 1.5 * nudge_rel * _dilate_max(speed_u * (u1 - u0) + speed_v * (v1 - v0)) + delta
    if not (np.all(np.isfinite(delta)) and np.all(np.isfinite(nudge))):
        return None
    if not (area is not None and np.isfinite(area) and area > 0.0):
        # Unverifiable polygons: no inside cells (no certified hit here).
        trim.ok = False
    if trim.ok:
        # The trim polygons must agree with OCC's own area for the face: the
        # inside cells cannot cover more, nor all kept ones less.
        assert area is not None
        slack = 1.0e-3 * area + 4.0 * float(np.max(delta)) * (length_u + length_v)
        if (float(np.sum(cell_area[cell_in])) > area + slack
                or float(np.sum(cell_area[cell_keep])) < area - slack):
            trim.ok = False
        else:
            trim.verified = True
    if not trim.ok:
        cell_in = np.zeros_like(cell_in)
        cell_keep = np.ones_like(cell_keep)

    eps = 64.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(nodes))))
    keep = cell_keep.ravel()
    tri_a = np.stack([p00, p10, p11], axis=2).reshape(-1, 3, 3)[keep]
    tri_b = np.stack([p00, p11, p01], axis=2).reshape(-1, 3, 3)[keep]
    q00 = np.stack(np.meshgrid(u[:-1], v[:-1], indexing="ij"), axis=-1)
    q10, q01, q11 = q00 + [du, 0.0], q00 + [0.0, dv], q00 + [du, dv]
    uv_a = np.stack([q00, q10, q11], axis=2).reshape(-1, 3, 2)[keep]
    uv_b = np.stack([q00, q11, q01], axis=2).reshape(-1, 3, 2)[keep]
    def twice(values: np.ndarray) -> np.ndarray:
        kept = values.ravel()[keep]
        return np.concatenate([kept, kept])

    return FaceMesh(
        surface, np.concatenate([tri_a, tri_b]), twice(cell_in),
        twice(delta) + eps, twice(nudge) + eps, np.concatenate([uv_a, uv_b]), trim, planar,
    )


# -- point to triangle distances -----------------------------------------------


def point_triangle_distance(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Euclidean distance from points ``p`` (K, 3) to triangles ``tri`` (K, 3, 3)."""
    return np.linalg.norm(p - closest_point_on_triangle(p, tri), axis=1)


def closest_point_on_triangle(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """The closest point of each triangle to each point (Ericson's regions)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    # Default: interior (barycentric).
    denom = va + vb + vc
    with np.errstate(divide="ignore", invalid="ignore"):
        v = vb / denom
        w = vc / denom
        closest = a + ab * v[:, None] + ac * w[:, None]
        # Edge BC
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        on_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        closest = np.where(on_bc[:, None], b + (c - b) * t_bc[:, None], closest)
        # Edge AC
        t_ac = d2 / (d2 - d6)
        on_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        closest = np.where(on_ac[:, None], a + ac * t_ac[:, None], closest)
        # Vertex C, then edge AB, vertex B, vertex A: Ericson's order, with
        # the first matching region winning (so applied last here).
        closest = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, closest)
        t_ab = d1 / (d1 - d3)
        on_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        closest = np.where(on_ab[:, None], a + ab * t_ab[:, None], closest)
    closest = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, closest)
    closest = np.where(((d1 <= 0) & (d2 <= 0))[:, None], a, closest)
    # Degenerate triangles: fall back to the nearest vertex or edge point.
    bad = ~np.all(np.isfinite(closest), axis=1)
    if np.any(bad):
        closest[bad] = _degenerate_closest(p[bad], a[bad], b[bad], c[bad])
    return closest


def _barycentric(q: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Barycentric weights (K, 3) of points ``q`` lying on triangles ``tri``."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1, v2 = b - a, c - a, q - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    den = d00 * d11 - d01 * d01
    # A degenerate triangle (a pole's collapsed cell) has no barycentric
    # coordinates: its rows come back NaN, and the caller treats them as
    # unproven.
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        return np.column_stack([1.0 - v - w, v, w])


def _degenerate_closest(p, a, b, c):
    best = a.copy()
    best_d = np.linalg.norm(p - a, axis=1)
    for s, e in ((a, b), (b, c), (c, a)):
        se = e - s
        den = np.einsum("ij,ij->i", se, se)
        t = np.clip(np.einsum("ij,ij->i", p - s, se) / np.where(den > 0, den, 1.0), 0.0, 1.0)
        q = s + t[:, None] * se
        d = np.linalg.norm(p - q, axis=1)
        better = d < best_d
        best[better] = q[better]
        best_d[better] = d[better]
    return best


class FaceIndex:
    """Every tessellated face of a model, behind centroid KD-trees."""

    def __init__(self, meshes: dict[int, FaceMesh]) -> None:
        self.meshes = meshes
        parts = list(meshes.values())
        if parts:
            self.triangles = np.concatenate([m.triangles for m in parts])
            self.faces = np.concatenate([np.full(len(m.triangles), m.surface, dtype=np.int64)
                                         for m in parts])
            self.inside = np.concatenate([m.inside for m in parts])
            self.planar = np.concatenate([np.full(len(m.triangles), m.planar) for m in parts])
            self.delta = np.concatenate([m.delta for m in parts])
            self.nudge = np.concatenate([m.nudge for m in parts])
            self.uv = np.concatenate([m.uv for m in parts])
        else:
            self.uv = np.zeros((0, 3, 2))
            self.triangles = np.zeros((0, 3, 3))
            self.faces = np.zeros(0, dtype=np.int64)
            self.inside = self.planar = np.zeros(0, dtype=bool)
            self.delta = self.nudge = np.zeros(0)
        self.centroids = self.triangles.mean(axis=1) if len(self.triangles) else np.zeros((0, 3))
        radii = (np.max(np.linalg.norm(self.triangles - self.centroids[:, None, :], axis=2), axis=1)
                 if len(self.triangles) else np.zeros(0))
        # Triangles are searched in classes of similar size, each with its own
        # largest radius, so one big triangle does not widen every search.
        self._classes: list[tuple[np.ndarray, float]] = []
        if len(radii):
            top = max(float(np.max(radii)), 1.0e-300)
            level = np.floor(np.log2(np.maximum(radii, top * 2.0 ** -20) / top)).astype(int)
            for value in np.unique(level):
                members = np.flatnonzero(level == value)
                self._classes.append((members, float(np.max(radii[members]))))
        normals = (np.cross(self.triangles[:, 1] - self.triangles[:, 0],
                            self.triangles[:, 2] - self.triangles[:, 0])
                   if len(self.triangles) else np.zeros((0, 3)))
        lengths = np.linalg.norm(normals, axis=1)
        self.normals = normals / np.where(lengths > 0, lengths, 1.0)[:, None]
        self.normal_ok = lengths > 0
        self.max_margin = float(np.max(self.delta + self.nudge)) if len(self.delta) else 0.0
        self._trees_cache = None

    def _trees(self):
        if self._trees_cache is None:
            from scipy.spatial import cKDTree

            self._trees_cache = [(members, radius, cKDTree(self.centroids[members]))
                                 for members, radius in self._classes]
        return self._trees_cache

    def _foot_inside(self, closest: np.ndarray, tri_ids: np.ndarray) -> np.ndarray:
        """Whether each plane foot's (u, v) -- affine on a plane, so exact up
        to rounding -- is certainly inside its face's trim."""
        weights = _barycentric(closest, self.triangles[tri_ids])
        uv = np.einsum("kj,kjl->kl", weights, self.uv[tri_ids])
        ok = np.all(np.isfinite(uv), axis=1)
        faces = self.faces[tri_ids]
        for face in np.unique(faces[ok]):
            trim = self.meshes[int(face)].trim
            rows = np.flatnonzero(ok & (faces == face))
            if trim is None or not (trim.ok and trim.verified):
                ok[rows] = False
                continue
            inside, certain = trim.classify(uv[rows])
            ok[rows] = inside & certain
        return ok

    def near(self, points: np.ndarray, tolerance: float) -> NearFaces:
        """Certifying bounds from each point to every face near it.

        A face not listed for a point has no triangle within ``tolerance``
        plus its largest margin of the point: certainly farther than
        ``tolerance``, nudge included.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        finite = np.all(np.isfinite(points), axis=1)
        reach = (tolerance + self.max_margin) * (1.0 + 1.0e-9)
        pid_parts, tid_parts = [], []
        query = np.where(finite[:, None], points, 0.0)
        for members, radius, tree in (self._trees() if len(points) else ()):
            rows = tree.query_ball_point(query, (reach + radius) * (1.0 + 1.0e-9))
            lengths = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
            lengths[~finite] = 0
            total = int(lengths.sum())
            if not total:
                continue
            pid_parts.append(np.repeat(np.arange(len(points)), lengths))
            tid_parts.append(members[np.fromiter(
                (t for keep, row in zip(finite, rows) if keep for t in row),
                dtype=np.int64, count=total)])
        if not pid_parts:
            return NearFaces.empty()
        point_ids = np.concatenate(pid_parts)
        tri_ids = np.concatenate(tid_parts)
        mirrored = points[point_ids]
        triangles = self.triangles[tri_ids]
        closest = closest_point_on_triangle(mirrored, triangles)
        dist = np.linalg.norm(mirrored - closest, axis=1)
        delta = self.delta[tri_ids]
        lower = dist - delta - self.nudge[tri_ids]
        within = dist + delta <= tolerance
        # A plane is the one case the tessellation proves a hit: the triangle
        # lies in the carrier, so its closest point is where OCC projects
        # (the perpendicular foot) when the foot falls in the triangle, and
        # that foot must be certainly inside the trim. A curved face within
        # tolerance is only a *candidate*: OCC confirms it.
        planar = self.planar[tri_ids]
        # The perpendicular foot on the triangle's plane: a plane hit needs
        # the triangle's closest point to be that very foot (compared as
        # points: comparing distances would let a lateral offset through as
        # its square), and the foot certainly inside the trim.
        normals = self.normals[tri_ids]
        signed = np.einsum("ij,ij->i", mirrored - triangles[:, 0], normals)
        foot_point = mirrored - signed[:, None] * normals
        scale = 1.0e-9 * (1.0 + np.max(np.abs(mirrored), axis=1))
        foot = (planar & self.normal_ok[tri_ids] & within
                & (np.linalg.norm(closest - foot_point, axis=1) <= scale))
        certified = foot & self.inside[tri_ids]
        check = np.flatnonzero(foot & ~certified)
        if len(check):
            certified[check] = self._foot_inside(foot_point[check], tri_ids[check])
        upper = np.where(certified, dist, np.inf)
        foot_uv = np.full((len(tri_ids), 2), np.nan)
        rows = np.flatnonzero(certified)
        if len(rows):
            weights = _barycentric(foot_point[rows], triangles[rows])
            foot_uv[rows] = np.einsum("kj,kjl->kl", weights, self.uv[tri_ids[rows]])
        candidate = within & ~planar
        face = self.faces[tri_ids]
        # Group by (point, face), nearest triangle first.
        order = np.lexsort((dist, face, point_ids))
        point_ids, face, tri_ids = point_ids[order], face[order], tri_ids[order]
        dist, lower, upper, candidate = dist[order], lower[order], upper[order], candidate[order]
        foot_uv = foot_uv[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = (point_ids[1:] != point_ids[:-1]) | (face[1:] != face[:-1])
        starts = np.flatnonzero(first)
        nearest_tri = tri_ids[starts]
        # Per group, the (u, v) of its proven hit with the least distance.
        best_upper = np.minimum.reduceat(upper, starts)
        group = np.cumsum(first) - 1
        best_uv = np.full((len(starts), 2), np.nan)
        rows = np.flatnonzero(np.isfinite(upper) & (upper == best_upper[group]))
        best_uv[group[rows[::-1]]] = foot_uv[rows[::-1]]
        return NearFaces(
            point_ids[starts], face[starts], np.minimum.reduceat(lower, starts),
            best_upper, dist[starts], np.logical_or.reduceat(candidate, starts),
            self.normals[nearest_tri], self.normal_ok[nearest_tri], best_uv,
        )


@dataclass
class NearFaces:
    """Per (point, face) pair: ``lower`` -- by the measured chord error,
    every point of the trimmed face (and of its carrier within the trim
    query's nudge) is farther than this; ``upper`` -- a proven hit's
    distance (planes only, else ``inf``); ``estimate`` -- the least triangle
    distance; ``candidate`` -- a curved face the tessellation puts within
    tolerance, for OCC to confirm; and the unit normal of the nearest
    triangle (exact on a plane)."""

    point: np.ndarray
    face: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    estimate: np.ndarray
    candidate: np.ndarray
    normal: np.ndarray
    normal_ok: np.ndarray
    uv: np.ndarray               # (u, v) of the proven hit's foot, NaN if none

    @classmethod
    def empty(cls) -> NearFaces:
        return cls(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0),
                   np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool), np.zeros((0, 3)),
                   np.zeros(0, dtype=bool), np.zeros((0, 2)))


def build_face_index(
    surfaces: list[int], *, diagonal: float, tolerance: float, nudge_rel: float,
    trims: dict[int, FaceTrim] | None = None,
) -> tuple[FaceIndex, dict[int, FaceTrim]]:
    """Trim polygons and tessellations for ``surfaces``; faces OCC cannot
    evaluate are simply absent (the caller treats them as unknown)."""
    trims = dict(trims or {})
    meshes: dict[int, FaceMesh] = {}
    cell = max(diagonal * CELL_SIZE_REL, 1.0e-12)
    delta_target = max(tolerance * DELTA_TARGET_FRACTION, 0.0)
    for surface in surfaces:
        trim = trims.get(surface)
        if trim is None:
            trim = trims[surface] = FaceTrim.build(surface)
        try:
            area = float(gmsh.model.occ.getMass(2, surface))
        except Exception:
            area = None
        mesh = build_face_mesh(surface, trim, cell_size=cell, delta_target=delta_target,
                               area=area, nudge_rel=nudge_rel)
        if mesh is not None:
            meshes[surface] = mesh
    return FaceIndex(meshes), trims
