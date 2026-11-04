"""Clearance between an outer wall and the acoustic wall it encloses.

``minimum_wall_distance`` is the exact minimum distance between two sets of
linear triangles (vertex/face, edge/edge and crossing contacts); the build
records it for the written mesh.
``fitted_face_clearance`` bounds the distance between two sets of fitted CAD
faces from below: each face is sampled on a parameter grid whose every cell
carries an interpolation allowance, and 1-Lipschitz maps (the meridian map,
then line projections) settle most of the wall before exact sample-triangle
distances settle the rest. ``certify_outer_envelope`` holds an envelope's
fitted faces, the ones a mesh or STEP export uses, to Wall thickness.
"""
from __future__ import annotations

import contextvars
import logging
import math

import numpy as np
from scipy.spatial import cKDTree

from .preview.fidelity import _point_segment_squared, _point_triangle_squared
from .errors import InternalInvariantError
from .preview.intersections import _triangles_intersect

logger = logging.getLogger(__name__)


def _paired_triangle_squared(a, b):
    """Squared distance between paired solid triangles ``a[i]`` and ``b[i]``."""
    result = np.full(len(a), np.inf)
    for i in range(3):
        result = np.minimum(result, _point_triangle_squared(a[:, i], b[:, 0], b[:, 1], b[:, 2]))
        result = np.minimum(result, _point_triangle_squared(b[:, i], a[:, 0], a[:, 1], a[:, 2]))
        u = a[:, (i + 1) % 3] - a[:, i]
        aa = np.sum(u * u, axis=1)
        for j in range(3):
            v = b[:, (j + 1) % 3] - b[:, j]
            w = a[:, i] - b[:, j]
            bb, cc = np.sum(u * v, axis=1), np.sum(v * v, axis=1)
            dd, ee = np.sum(u * w, axis=1), np.sum(v * w, axis=1)
            # |u x v|^2 avoids subtracting two almost equal squared lengths.
            den = np.sum(np.cross(u, v) ** 2, axis=1)
            usable = den > 0
            s = (bb * ee - cc * dd) / np.where(usable, den, 1)
            t = (aa * ee - bb * dd) / np.where(usable, den, 1)
            # Interior skew-edge minima only; endpoint cases are point/triangle.
            valid = usable & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
            distance = np.sum((w + s[:, None] * u - t[:, None] * v) ** 2, axis=1)
            result = np.minimum(result, np.where(valid, distance, np.inf))
    # Crossing face interiors need not contain a closest vertex or edge pair.
    result[_triangles_intersect(a, b)] = 0
    return result


def minimum_wall_distance(inner, outer):
    """Return ``(minimum, numerical_guard)`` over all facet points, in input units.

    ``minimum`` is the exact distance between the two triangle sets (zero
    spatial sampling error). Centroid spheres and boxes only prune pairs that
    cannot improve a known upper bound. ``numerical_guard`` is a
    floating-point allowance of 1e-8 of the model extent, not an
    interval-arithmetic proof.
    """
    inner, outer = np.asarray(inner, dtype=float), np.asarray(outer, dtype=float)
    if not len(inner) or not len(outer):
        raise ValueError('wall clearance requires inner and outer triangles')
    origin = inner[0, 0].copy()
    inner, outer = inner - origin, outer - origin
    scale = max(float(np.max(np.abs(inner))), float(np.max(np.abs(outer))))
    guard = 1e-8 * scale
    centers = inner.mean(axis=1)
    radii = np.linalg.norm(inner - centers[:, None], axis=2).max(axis=1)
    tree = cKDTree(centers)
    oc = outer.mean(axis=1)
    # Distance between two surface centroids is a legitimate upper bound.
    best = float(np.min(tree.query(oc)[0]))
    lo, hi = inner.min(axis=1), inner.max(axis=1)
    reach = float(radii.max())
    for start in range(0, len(outer), 256):
        batch = outer[start:start + 256]
        bc = oc[start:start + 256]
        br = np.linalg.norm(batch - bc[:, None], axis=2).max(axis=1)
        neighbors = tree.query_ball_point(bc, best + br + reach + guard)
        oi = np.repeat(np.arange(len(batch)), [len(ns) for ns in neighbors])
        if not len(oi):
            continue
        ii = np.fromiter((i for ns in neighbors for i in ns), dtype=int, count=len(oi))
        delta = np.maximum(0, np.maximum(lo[ii] - batch.max(axis=1)[oi], batch.min(axis=1)[oi] - hi[ii]))
        keep = np.sum(delta ** 2, axis=1) <= (best + guard) ** 2
        oi, ii = oi[keep], ii[keep]
        sphere_gap = np.linalg.norm(bc[oi] - centers[ii], axis=1) - br[oi] - radii[ii]
        keep = sphere_gap <= best + guard
        oi, ii = oi[keep], ii[keep]
        for first in range(0, len(ii), 8192):
            sl = slice(first, first + 8192)
            distances = np.sqrt(_paired_triangle_squared(batch[oi[sl]], inner[ii[sl]]))
            if len(distances):
                best = min(best, float(distances.min()))
    return best, guard


# Sampled second-derivative maxima are widened by this factor before they
# bound a cell: they are evaluated at the cell corners, not proven suprema.
_CURVATURE_WIDENING = 1.25


def _arclength_stations(gmsh, tag, lo, hi, other, along_u, spacing, probes=4097):
    """Parameter stations about ``spacing`` apart in space along one direction.

    Fitted walls can be parameterized very unevenly (an axial parameter that
    follows z barely moves across a mouth that turns to 90 degrees), so
    stations follow arclength, measured on lines across the other direction.
    """
    t = np.linspace(lo, hi, probes)
    length = np.zeros(probes)
    for fixed in other:
        uv = np.stack((t, np.full(probes, fixed)) if along_u else (np.full(probes, fixed), t),
                      axis=1).reshape(-1)
        xyz = np.asarray(gmsh.model.getValue(2, tag, uv)).reshape(probes, 3)
        length = np.maximum(length, np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
    count = max(2, math.ceil(length[-1] / spacing))
    stations = np.interp(np.linspace(0.0, length[-1], count + 1), length, t)
    return np.unique(np.r_[lo, stations, hi])


def _sample_face(gmsh, tag, target, spacing=2.0, max_samples=800_000):
    """Sample one face: ``(points (nu, nv, 3), per-cell allowance (nu-1, nv-1))``.

    The allowance bounds how far the face leaves either sample triangle of a
    cell, from second derivatives evaluated at the cell corners (widened);
    cells are bisected until every allowance is at most ``target``.
    """
    (u0, v0), (u1, v1) = gmsh.model.getParametrizationBounds(2, tag)

    def second(us, vs):
        uv = np.stack(np.meshgrid(us, vs, indexing='ij'), axis=-1).reshape(-1)
        d = np.asarray(gmsh.model.getSecondDerivative(2, tag, uv)).reshape(len(us), len(vs), 3, 3)
        return np.linalg.norm(d, axis=-1)  # uu, vv, uv norms

    us = _arclength_stations(gmsh, tag, u0, u1, (v0, (v0 + v1) / 2, v1), True, spacing)
    vs = _arclength_stations(gmsh, tag, v0, v1, (u0, (u0 + u1) / 2, u1), False, spacing)
    for _ in range(40):
        if len(us) * len(vs) > max_samples:
            raise InternalInvariantError(f"fitted face {tag} needs more than {max_samples} clearance samples")
        m = second(us, vs)
        corner = np.maximum.reduce((m[:-1, :-1], m[1:, :-1], m[:-1, 1:], m[1:, 1:])) * _CURVATURE_WIDENING
        du, dv = np.diff(us)[:, None], np.diff(vs)[None, :]
        # Linear interpolation error on either triangle of a du x dv cell:
        # |S - I S| <= (Muu du^2 + 2 Muv du dv + Mvv dv^2) / 8.
        terms = (corner[..., 0] * du ** 2, 2 * corner[..., 2] * du * dv, corner[..., 1] * dv ** 2)
        allowance = sum(terms) / 8
        # The meridian bound also pays each cell's |xy| sag; keep it in budget.
        uv = np.stack(np.meshgrid(us, vs, indexing='ij'), axis=-1).reshape(-1)
        points = np.asarray(gmsh.model.getValue(2, tag, uv)).reshape(len(us), len(vs), 3)
        # Cells far from revolution (or at the axis) are left to the
        # triangle fallback rather than refined without end.
        sag = _radius_sag(points)
        over = (allowance > target) | ((allowance + sag > target) & (sag < 10 * target))
        if not over.any():
            break
        # Bisect the direction whose own term dominates an oversized cell;
        # for the sag, the direction whose cell edges turn further in azimuth.
        angle = np.arctan2(points[..., 1], points[..., 0])
        turn_u = np.abs(np.diff(angle, axis=0))[:, :-1]
        turn_v = np.abs(np.diff(angle, axis=1))[:-1, :]
        split_u = np.where(allowance >= sag, terms[0] >= terms[2], turn_u >= turn_v)
        rows = np.flatnonzero((over & split_u).any(axis=1))
        cols = np.flatnonzero((over & ~split_u).any(axis=0))
        us = np.unique(np.r_[us, (us[rows] + us[rows + 1]) / 2])
        vs = np.unique(np.r_[vs, (vs[cols] + vs[cols + 1]) / 2])
    else:
        raise InternalInvariantError(f"fitted face {tag} clearance samples did not reach {target:g} mm")
    return points, allowance


def _radius_sag(points):
    """Per cell, how far |xy| can fall below its linear interpolant.

    The Hessian of |xy| at y is (I - y^ y^T)/|y|: only a step's component
    across the local radius bends it. Within a cell that component is at most
    (largest radius + span) * azimuth turn, and within a triangle
    sum(lambda_k |x_k - y|^2) <= (longest edge)^2 / 3.
    """
    xy = points[..., :2]
    corners = (xy[:-1, :-1], xy[1:, :-1], xy[:-1, 1:], xy[1:, 1:])
    pairs = [(a, b) for k, a in enumerate(corners) for b in corners[k + 1:]]
    span = np.sqrt(np.maximum.reduce([np.sum((a - b) ** 2, axis=-1) for a, b in pairs]))
    turn = np.maximum.reduce([np.abs(np.arctan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
                                                np.sum(a * b, axis=-1))) for a, b in pairs])
    radius = np.minimum.reduce([np.linalg.norm(c, axis=-1) for c in corners])
    largest = np.maximum.reduce([np.linalg.norm(c, axis=-1) for c in corners])
    across = np.minimum(span, (largest + span) * turn)
    return np.where(radius > 0, across ** 2 / (6 * np.maximum(radius, 1e-300)), np.inf)


def _meridian_boxes(points, allowance, axis):
    """Oriented 2D boxes covering ``m(x) = (|xy|, z)`` of each cell strip.

    ``axis`` 0 groups the cells of each v-interval (strips across u), 1 the
    cells of each u-interval. Each box covers its strip's mapped samples; its
    allowance adds the face-to-sample-triangle allowance and the inward sag of
    |xy| across a cell (|xy| is convex with Hessian norm 1/|xy|, so its linear
    interpolant exceeds it by at most (1/2|xy|) sum(lambda_k |x_k - y|^2)).
    """
    grid = points if axis == 0 else points.transpose(1, 0, 2)
    cells = allowance if axis == 0 else allowance.T
    planar = np.stack((np.linalg.norm(grid[..., :2], axis=-1), grid[..., 2]), axis=-1)
    sag = _radius_sag(grid)
    strip_allow = (cells + sag).max(axis=0)
    # Strip k holds stations k and k+1 of the second index, all of the first.
    pts = np.stack((planar[:, :-1], planar[:, 1:]), axis=1)  # (n0, 2, n1-1, 2)
    pts = pts.transpose(2, 0, 1, 3).reshape(planar.shape[1] - 1, -1, 2)
    center = pts.mean(axis=1)
    rel = pts - center[:, None]
    cov = np.einsum('kni,knj->kij', rel, rel)
    _, vectors = np.linalg.eigh(cov)
    proj = np.einsum('kni,kij->knj', rel, vectors)
    low, high = proj.min(axis=1), proj.max(axis=1)
    box = np.stack([center + np.einsum('kj,kij->ki', np.stack((a, b), axis=1), vectors)
                    for a, b in ((low[:, 0], low[:, 1]), (high[:, 0], low[:, 1]),
                                 (high[:, 0], high[:, 1]), (low[:, 0], high[:, 1]))], axis=1)
    return box, strip_allow


def _box_distances(a, b):
    """Distance between paired convex quadrilaterals (0 where they overlap)."""
    separated = np.zeros(len(a), dtype=bool)
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=1) - poly
        normals = np.stack((-edges[..., 1], edges[..., 0]), axis=-1)
        for k in range(4):
            n = normals[:, k]
            pa, pb = np.einsum('nij,nj->ni', a, n), np.einsum('nij,nj->ni', b, n)
            separated |= (pa.max(axis=1) < pb.min(axis=1)) | (pb.max(axis=1) < pa.min(axis=1))
    squared = np.full(len(a), np.inf)
    for p, q in ((a, b), (b, a)):
        for i in range(4):
            for j in range(4):
                squared = np.minimum(squared, _point_segment_squared(
                    p[:, i], q[:, j], q[:, (j + 1) % 4]))
    return np.where(separated, np.sqrt(squared), 0.0)


def _thinner_boxes(points, allowance):
    """Strip boxes in the direction whose boxes are thinner, and that axis."""
    def width(boxes):
        box, allow = boxes
        sides = np.linalg.norm(np.diff(box[:, :3], axis=1), axis=2)
        return float(np.max(sides.min(axis=1) + 2 * allow))
    options = [(_meridian_boxes(points, allowance, axis), axis) for axis in (0, 1)]
    return min(options, key=lambda option: width(option[0]))


def _meridian_strip_bounds(inner, outer, threshold):
    """Lower bound per outer strip; pairs not examined are at least ``threshold``.

    Returns ``(bounds, strips)``; ``strips`` names each outer strip as
    ``(face, axis, index)`` so that failing strips can be refined.
    """
    ib = [_thinner_boxes(p, a)[0] for p, a in inner]
    ob = [_thinner_boxes(p, a) for p, a in outer]
    ibox, iallow = np.concatenate([b for b, _ in ib]), np.concatenate([a for _, a in ib])
    obox = np.concatenate([b for (b, _), _ in ob])
    oallow = np.concatenate([a for (_, a), _ in ob])
    strips = [(face, axis, index) for face, ((b, _), axis) in enumerate(ob) for index in range(len(b))]
    ilo, ihi = ibox.min(axis=1), ibox.max(axis=1)
    olo, ohi = obox.min(axis=1), obox.max(axis=1)
    tree = cKDTree((ilo + ihi) / 2)
    ihalf = float(np.max(np.linalg.norm(ihi - ilo, axis=1))) / 2
    # A nearby pair's center distance less allowances bounds the minimum from
    # above, so no pair beyond that needs examining to find it.
    nearest_distance, nearest = tree.query((olo + ohi) / 2)
    threshold = max(threshold, float(np.min(nearest_distance - oallow - iallow[nearest])))
    ohalf = np.linalg.norm(ohi - olo, axis=1) / 2
    neighbors = tree.query_ball_point((olo + ohi) / 2, threshold + ohalf + ihalf + oallow + iallow.max())
    oi = np.repeat(np.arange(len(obox)), [len(n) for n in neighbors])
    ii = np.fromiter((i for n in neighbors for i in n), dtype=int, count=len(oi))
    gap = np.maximum(0, np.maximum(ilo[ii] - ohi[oi], olo[oi] - ihi[ii]))
    keep = np.linalg.norm(gap, axis=1) - oallow[oi] - iallow[ii] < threshold
    oi, ii = oi[keep], ii[keep]
    bounds = np.full(len(obox), threshold)
    for first in range(0, len(oi), 65536):
        sl = slice(first, first + 65536)
        values = _box_distances(obox[oi[sl]], ibox[ii[sl]]) - oallow[oi[sl]] - iallow[ii[sl]]
        np.minimum.at(bounds, oi[sl], values)
    return bounds, strips


def _units(points, allowance, size):
    """Blocks of ``size`` x ``size`` cells: corner points, allowance, cell ranges.

    Corners are ordered row-major over the block's sample grid, so a single
    cell's are (0,0), (0,1), (1,0), (1,1).
    """
    nu, nv = allowance.shape
    bu, bv = np.arange(0, nu, size), np.arange(0, nv, size)
    iu = np.minimum(bu[:, None] + np.arange(size + 1)[None, :], nu)
    iv = np.minimum(bv[:, None] + np.arange(size + 1)[None, :], nv)
    corners = points[iu[:, None, :, None], iv[None, :, None, :]].reshape(len(bu) * len(bv), -1, 3)
    allow = np.maximum.reduceat(np.maximum.reduceat(allowance, bu, axis=0), bv, axis=1).reshape(-1)
    ranges = [(i, min(i + size, nu), j, min(j + size, nv)) for i in bu for j in bv]
    return corners, allow, ranges


def _unit_normals(corners):
    """Approximate unit normals; any direction keeps the bound valid."""
    normal = np.cross(corners[:, -1] - corners[:, 0], corners[:, 1] - corners[:, -2])
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    return np.where(length > 0, normal / np.where(length > 0, length, 1.0), [[0.0, 0.0, 1.0]])


def _projected_gaps(o, i, normal):
    """Lower bounds of the distance between paired point hulls ``o`` and ``i``.

    Projection onto a line never increases distances, so the gap between the
    projected hulls bounds the hulls' distance: along the outer unit's normal
    (tight for facing walls) and along the line through both centers (tight
    for units beside each other). The larger is kept.
    """
    center = o.mean(axis=1) - i.mean(axis=1)
    length = np.linalg.norm(center, axis=1, keepdims=True)
    line = center / np.where(length > 0, length, 1.0)
    best = np.zeros(len(o))
    for direction in (normal, line):
        po, pi = np.einsum('nkj,nj->nk', o, direction), np.einsum('nkj,nj->nk', i, direction)
        best = np.maximum(best, np.maximum(pi.min(axis=1) - po.max(axis=1),
                                           po.min(axis=1) - pi.max(axis=1)))
    return best


def _unit_pairs(ocorners, oallow, icorners, iallow, threshold):
    """Pairs that could fall below ``threshold``, with projected lower bounds."""
    ic = icorners.mean(axis=1)
    ir = np.linalg.norm(icorners - ic[:, None], axis=2).max(axis=1)
    oc = ocorners.mean(axis=1)
    orad = np.linalg.norm(ocorners - oc[:, None], axis=2).max(axis=1)
    neighbors = cKDTree(ic).query_ball_point(
        oc, threshold + orad + oallow + float(ir.max() + iallow.max()))
    oi = np.repeat(np.arange(len(oc)), [len(n) for n in neighbors])
    ii = np.fromiter((i for n in neighbors for i in n), dtype=int, count=len(oi))
    sphere = np.linalg.norm(oc[oi] - ic[ii], axis=1) - orad[oi] - ir[ii] - oallow[oi] - iallow[ii]
    oi, ii = oi[sphere < threshold], ii[sphere < threshold]
    normals = _unit_normals(ocorners)
    values = np.empty(len(oi))
    for first in range(0, len(oi), 32768):
        sl = slice(first, first + 32768)
        values[sl] = (_projected_gaps(ocorners[oi[sl]], icorners[ii[sl]], normals[oi[sl]])
                      - oallow[oi[sl]] - iallow[ii[sl]])
    return oi, ii, values


def _cell_triangle_bound(ocells, oallow, icells, iallow):
    """Exact sample-triangle distances of paired cells, less both allowances."""
    def split(c):
        return c[:, [0, 2, 3]], c[:, [0, 3, 1]]
    result = np.full(len(ocells), np.inf)
    for to in split(ocells):
        for ti in split(icells):
            result = np.minimum(result, np.sqrt(_paired_triangle_squared(to, ti)))
    return result - oallow - iallow


REFINEMENT_BLOCK = 2


def _refined_lower_bound(inner, outer, failing, threshold, block=REFINEMENT_BLOCK):
    """Bound the outer cells of failing strips: blocks, then cells, then triangles."""
    def gather(faces, select=None):
        corners, allow, owners = [], [], []
        for face, (points, allowance) in enumerate(faces):
            c, a, r = _units(points, allowance, block)
            mask = np.ones(len(r), dtype=bool) if select is None else np.array(
                [select(face, rng) for rng in r], dtype=bool)
            corners.append(c[mask])
            allow.append(a[mask])
            owners.extend((face, rng) for rng, m in zip(r, mask) if m)
        return np.concatenate(corners), np.concatenate(allow), owners

    def cells(faces):
        units = [_units(points, allowance, 1) for points, allowance in faces]
        offsets = np.cumsum([0] + [len(u[1]) for u in units])
        return (np.concatenate([u[0] for u in units]), np.concatenate([u[1] for u in units]),
                offsets, [allowance.shape[1] for _, allowance in faces])

    def members(owner, offsets, widths):
        face, (a0, a1, b0, b1) = owner
        rows, cols = np.arange(a0, a1), np.arange(b0, b1)
        return offsets[face] + (rows[:, None] * widths[face] + cols[None, :]).reshape(-1)

    ocorners, oallow, oowners = gather(outer, failing)
    if not len(oallow):
        return threshold
    icorners, iallow, iowners = gather(inner)
    oi, ii, values = _unit_pairs(ocorners, oallow, icorners, iallow, threshold)
    lower = float(values[values >= threshold].min(initial=threshold))
    bad = np.flatnonzero(values < threshold)
    if not len(bad):
        return lower
    # Every cell pair inside a failing block pair, bounded by projections.
    oc, oca, ooff, owidth = cells(outer)
    ic, ica, ioff, iwidth = cells(inner)
    normals = _unit_normals(oc)
    ocenter, icenter = oc.mean(axis=1), ic.mean(axis=1)
    oradius = np.linalg.norm(oc - ocenter[:, None], axis=2).max(axis=1) + oca
    iradius = np.linalg.norm(ic - icenter[:, None], axis=2).max(axis=1) + ica
    pairs_o, pairs_i = [], []
    for k in bad:
        mo, mi = members(oowners[oi[k]], ooff, owidth), members(iowners[ii[k]], ioff, iwidth)
        pairs_o.append(np.repeat(mo, len(mi)))
        pairs_i.append(np.tile(mi, len(mo)))
    po, pi = np.concatenate(pairs_o), np.concatenate(pairs_i)
    # Bounding spheres settle most cell pairs of a block pair outright.
    sphere = np.linalg.norm(ocenter[po] - icenter[pi], axis=1) - oradius[po] - iradius[pi]
    lower = min(lower, float(sphere[sphere >= threshold].min(initial=threshold)))
    po, pi = po[sphere < threshold], pi[sphere < threshold]
    for first in range(0, len(po), 65536):
        so, si = po[first:first + 65536], pi[first:first + 65536]
        bound = _projected_gaps(oc[so], ic[si], normals[so]) - oca[so] - ica[si]
        lower = min(lower, float(bound[bound >= threshold].min(initial=threshold)))
        tight = bound < threshold
        if tight.any():
            exact = _cell_triangle_bound(oc[so[tight]], oca[so[tight]], ic[si[tight]], ica[si[tight]])
            lower = min(lower, float(exact.min()))
    return lower


def fitted_face_clearance(inner_tags, outer_tags, *, target_mm, required_mm=None):
    """Lower bound of the distance from fitted outer faces to inner faces.

    Both face sets are sampled so that each sample triangle stays within its
    own (widened, sampled) second-derivative allowance of the face, at most
    ``target_mm``. First, ``m(x) = (|xy|, z)`` is 1-Lipschitz, so the 2D
    distance between mapped strips bounds their 3D distance from below:
    tight for walls of revolution and cheap. Strips that do not reach
    ``required_mm`` are refined: blocks and then single cells bounded by
    1-Lipschitz projections, and finally the exact distance between their
    sample triangles, each less both allowances.
    """
    from .builders._occ import require_gmsh

    gmsh = require_gmsh()
    inner = [_sample_face(gmsh, int(tag), target_mm) for tag in inner_tags]
    outer = [_sample_face(gmsh, int(tag), target_mm) for tag in outer_tags]
    samples = int(sum(p.shape[0] * p.shape[1] for p, _ in inner + outer))
    interpolation = float(max(a.max() for _, a in inner + outer))
    guard = 1e-8 * max(float(np.max(np.abs(p))) for p, _ in inner + outer)
    threshold = -np.inf if required_mm is None else required_mm + guard
    # Strips are bounded up to a little above the requirement, so the
    # reported bound is the measured one, not the requirement it cleared.
    bounds, strips = _meridian_strip_bounds(inner, outer, threshold + REPORT_HEADROOM_MM)
    lower = float(bounds.min())
    method = "meridian map of sampled fitted faces"
    resolved = True
    if required_mm is not None and lower < threshold:
        failing = {}
        for (face, axis, index), value in zip(strips, bounds):
            if value < threshold:
                failing.setdefault(face, []).append((axis, index))

        def in_failing_strip(face, rng):
            a0, a1, b0, b1 = rng
            return any(b0 <= index < b1 if axis == 0 else a0 <= index < a1
                       for axis, index in failing.get(face, ()))

        refined = _refined_lower_bound(inner, outer, in_failing_strip, threshold)
        lower = min(float(bounds[bounds >= threshold].min(initial=threshold)), refined)
        # Refined pairs are settled once they clear the requirement, so a
        # passing refined bound states "at least Wall thickness".
        resolved = lower < threshold
        method = "meridian map, then projected blocks, cells and sample triangles"
    return {"lowerBoundMm": lower - guard, "lowerBoundResolved": resolved,
            "numericalAllowanceMm": guard, "interpolationAllowanceMm": interpolation,
            "samples": samples, "method": method}


# How far above Wall thickness the meridian bound is resolved for reporting.
REPORT_HEADROOM_MM = 0.25

# Fitted faces are sampled finely enough that each sample triangle stays within
# this distance of its face. The envelope's construction allowance
# (offset_envelope.envelope_allowance_mm) must exceed the certificate's
# slack, which is about twice this plus the grid's own chord deficit.
FACE_SAMPLE_ALLOWANCE_MM = 0.02


def _face_tags(entries):
    return [int(entry[1]) if isinstance(entry, (tuple, list)) else int(entry) for entry in entries]


class EnvelopeShortfall(InternalInvariantError):
    """Fitted envelope faces measured closer than Wall thickness."""

    def __init__(self, message: str, shortfall_mm: float):
        super().__init__(message)
        self.shortfall_mm = float(shortfall_mm)


_RECORD_SHORTFALL = contextvars.ContextVar("record_envelope_shortfall", default=False)
# Only a caller that can rebuild is told about a shortfall; any other caller
# (an export handed a resolved geometry) keeps the wall and the record.
_RETRYING = contextvars.ContextVar("envelope_rebuild_available", default=False)


def certify_outer_envelope(built, wall_thickness_mm):
    """Prove the fitted outer faces keep Wall thickness from the acoustic faces.

    Runs on the model a mesh or STEP export is about to use. The bound is the
    fitted faces' own (see ``fitted_face_clearance``), not the input grid's.
    A shortfall raises :class:`EnvelopeShortfall`, which
    :func:`with_envelope_retries` answers by rebuilding with more clearance.
    """
    groups = built.mesh_surface_groups
    try:
        certificate = fitted_face_clearance(
            _face_tags(groups['inner']), _face_tags(groups['outer']),
            target_mm=FACE_SAMPLE_ALLOWANCE_MM, required_mm=wall_thickness_mm)
    except InternalInvariantError as exc:
        # The measurement itself could not be completed: a fault for the
        # developer diagnostics, never a reason to fail the user's build.
        logger.warning("[hornlab-mesher] outer envelope clearance not measured: %s", exc)
        return {"certified": False, "wallThicknessMm": wall_thickness_mm, "unmeasured": str(exc)}
    certificate["wallThicknessMm"] = wall_thickness_mm
    shortfall = wall_thickness_mm - certificate["lowerBoundMm"]
    certificate["certified"] = shortfall <= 0.0
    if shortfall > 0.0:
        message = (f"fitted outer envelope clearance {certificate['lowerBoundMm']:.6g} mm is below "
                   f"Wall thickness {wall_thickness_mm:g} mm ({certificate['method']})")
        if _RETRYING.get() and not _RECORD_SHORTFALL.get():
            raise EnvelopeShortfall(message, shortfall)
        logger.warning("[hornlab-mesher] %s; the wall is kept and the shortfall recorded", message)
    return certificate


# Rebuilds after a measured shortfall: each adds twice the shortfall plus
# 0.05 mm, which also covers the measurement's own sampling slack.
ENVELOPE_REBUILDS = 2


def with_envelope_retries(build):
    """Run ``build()``; after a fitted-face shortfall, rebuild with more clearance.

    A design that built before envelopes must keep building, so after the last
    rebuild a remaining shortfall is recorded (``certified: False``) rather
    than raised. A nested call starts from the clearance its caller already
    added.
    """
    from .offset_envelope import _EXTRA_ALLOWANCE, extra_envelope_allowance

    extra = _EXTRA_ALLOWANCE.get()
    retrying = _RETRYING.set(True)
    try:
        for _ in range(ENVELOPE_REBUILDS):
            try:
                with extra_envelope_allowance(extra):
                    return build()
            except EnvelopeShortfall as exc:
                extra += 2.0 * exc.shortfall_mm + 0.05
                logger.info("[hornlab-mesher] rebuilding the outer envelope with %.3g mm more clearance", extra)
        token = _RECORD_SHORTFALL.set(True)
        try:
            with extra_envelope_allowance(extra):
                return build()
        finally:
            _RECORD_SHORTFALL.reset(token)
    finally:
        _RETRYING.reset(retrying)
