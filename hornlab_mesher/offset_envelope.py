"""Exterior of a constant-distance offset of a sampled radial surface."""

from __future__ import annotations

import collections
import contextlib
import contextvars
import dataclasses
import hashlib
import threading
import weakref

import numpy as np


def _triangles(grid: np.ndarray, closed: bool) -> np.ndarray:
    rows = grid if closed else grid[:-1]
    following = np.roll(grid, -1, axis=0) if closed else grid[1:]
    corners = (rows[:, :-1], following[:, :-1], following[:, 1:], rows[:, 1:])
    center = sum(corners) / 4.0
    # A fixed diagonal biases a non-planar quad toward one azimuthal direction
    # and breaks reflection symmetry. A center fan is invariant under either
    # direction and agrees when a symmetric full grid is cut into quadrants.
    return np.concatenate(
        [
            np.stack((corners[index], corners[(index + 1) % 4], center), axis=-2)
            for index in range(4)
        ],
        axis=0,
    ).reshape(-1, 3, 3)


def _offset_intersections(
    triangles: np.ndarray, z: np.ndarray, direction: np.ndarray, wall: float
) -> np.ndarray:
    """Farthest intersection for each paired radial ray and dilated triangle.

    The boundary of each dilated triangle consists of its two offset faces,
    three finite edge cylinders and three vertex spheres. Taking their union
    removes normal-offset loops, including loops away from the throat.
    """
    relative = triangles.copy()
    relative[:, :, 2] -= z[:, None]
    along = np.einsum("...j,j->...", relative, direction)
    perpendicular_squared = np.maximum(
        np.einsum("ijk,ijk->ij", relative, relative) - along * along, 0.0
    )
    discriminant = wall * wall - perpendicular_squared
    sphere = np.where(
        discriminant >= 0.0, along + np.sqrt(np.maximum(discriminant, 0.0)), -np.inf
    )
    farthest = np.max(sphere, axis=1)

    for index in range(3):
        start = relative[:, index]
        edge = relative[:, (index + 1) % 3] - start
        squared = np.einsum("ij,ij->i", edge, edge)
        edge_along = np.einsum("ij,j->i", edge, direction)
        start_along = np.einsum("ij,j->i", start, direction)
        start_edge = np.einsum("ij,ij->i", start, edge)
        valid_edge = squared > 0.0
        squared = np.where(valid_edge, squared, 1.0)
        a = np.maximum(1.0 - edge_along * edge_along / squared, 0.0)
        b = -2.0 * (start_along - edge_along * start_edge / squared)
        c = (
            np.einsum("ij,ij->i", start, start)
            - start_edge * start_edge / squared
            - wall * wall
        )
        disc = b * b - 4.0 * a * c
        radius = (-b + np.sqrt(np.maximum(disc, 0.0))) / np.where(
            a > 1e-14, 2.0 * a, 1.0
        )
        parameter = (radius * edge_along - start_edge) / squared
        valid = (
            valid_edge
            & (a > 1e-14)
            & (disc >= 0.0)
            & (parameter >= 0.0)
            & (parameter <= 1.0)
        )
        farthest = np.maximum(farthest, np.where(valid, radius, -np.inf))

    a = relative[:, 0]
    ab = relative[:, 1] - a
    ac = relative[:, 2] - a
    normal = np.cross(ab, ac)
    norm = np.linalg.norm(normal, axis=1)
    unit = normal / np.where(norm > 0.0, norm, 1.0)[:, None]
    denominator = np.einsum("ij,j->i", unit, direction)
    plane = np.einsum("ij,ij->i", unit, a)
    ab2 = np.einsum("ij,ij->i", ab, ab)
    ac2 = np.einsum("ij,ij->i", ac, ac)
    abac = np.einsum("ij,ij->i", ab, ac)
    determinant = ab2 * ac2 - abac * abac
    usable = (norm > 0.0) & (np.abs(denominator) > 1e-14) & (determinant > 0.0)
    for side in (-1.0, 1.0):
        radius = (plane + side * wall) / np.where(usable, denominator, 1.0)
        foot = radius[:, None] * direction - side * wall * unit - a
        fab = np.einsum("ij,ij->i", foot, ab)
        fac = np.einsum("ij,ij->i", foot, ac)
        u = (ac2 * fab - abac * fac) / np.where(usable, determinant, 1.0)
        v = (ab2 * fac - abac * fab) / np.where(usable, determinant, 1.0)
        valid = usable & (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1.0 + 1e-12)
        farthest = np.maximum(farthest, np.where(valid, radius, -np.inf))
    return farthest


def _ray_offset_radius(
    triangles: np.ndarray, z: float, direction: np.ndarray, wall: float
) -> float:
    intersections = _offset_intersections(
        triangles, np.full(len(triangles), z), direction, wall
    )
    farthest = float(np.max(intersections, initial=-np.inf))
    if not np.isfinite(farthest) or farthest <= 0.0:
        raise ValueError("outer offset envelope has no positive radial intersection")
    return farthest


def _candidate_ray_intervals(triangles, z_values, direction, wall, lower, margin):
    """Conservatively prune axial ray ranges before constructing feature pairs."""
    along = np.einsum("ijk,k->ij", triangles, direction).max(axis=1)
    transverse_direction = np.array([-direction[1], direction[0], 0.0])
    transverse = np.einsum("ijk,k->ij", triangles, transverse_direction)
    across = np.maximum(
        0.0, np.maximum(transverse.min(axis=1), -transverse.max(axis=1))
    )
    z_low, z_high = triangles[:, :, 2].min(axis=1), triangles[:, :, 2].max(axis=1)
    starts = np.searchsorted(z_values, z_low - wall - margin, side="left")
    ends = np.searchsorted(z_values, z_high + wall + margin, side="right")

    # A sparse range-minimum table keeps storage O(n log n), including offline
    # control grids much longer than the usual preview meridian.
    n = len(z_values)
    levels = n.bit_length()
    minima = np.full((levels, n), np.inf)
    minima[0] = lower
    for level in range(1, levels):
        half = 1 << (level - 1)
        count = n - 2 * half + 1
        minima[level, :count] = np.minimum(
            minima[level - 1, :count], minima[level - 1, half : half + count]
        )
    for _ in range(32):
        counts = ends - starts
        powers = np.floor(np.log2(np.maximum(counts, 1))).astype(np.intp)
        interval_minimum = np.minimum(
            minima[powers, np.minimum(starts, n - 1)],
            minima[powers, np.maximum(ends - np.left_shift(1, powers), 0)],
        )
        interval_minimum = np.where(counts > 0, interval_minimum, np.inf)
        # To exceed even the smallest known exterior radius in this interval,
        # a ball must spend at least need² + across² in the transverse plane.
        # Only the remaining radius can reach forward/backward in z. Repeating
        # after shrinking an interval strengthens the bound without changing it.
        need = np.maximum(0.0, interval_minimum - along - margin)
        remaining = (wall + margin) ** 2 - across**2 - need**2
        reach = np.sqrt(np.maximum(remaining, 0.0)) + margin
        new_starts = np.maximum(
            starts, np.searchsorted(z_values, z_low - reach, side="left")
        )
        new_ends = np.minimum(
            ends, np.searchsorted(z_values, z_high + reach, side="right")
        )
        new_starts = np.minimum(new_starts, new_ends)
        new_ends = np.where(remaining >= 0.0, new_ends, new_starts)
        if np.array_equal(new_starts, starts) and np.array_equal(new_ends, ends):
            break
        starts, ends = new_starts, new_ends
    return starts, ends


def _equivalent_azimuth_rows(inner, target_z, angles, *, full_circle, margin,
                             revolution=False):
    """Reuse a rotation only after the entire sampled surface proves it.

    ``revolution`` additionally accepts every uniform row step of a full,
    uniformly sampled surface of revolution: its center-fan triangulation
    then maps onto itself under each step, so one row's rays are every row's.
    """
    representatives = np.arange(len(angles))
    if not full_circle:
        return representatives
    if revolution and _uniform_revolution(inner, target_z, angles, margin):
        return np.zeros_like(representatives)
    period = 2.0 * np.pi
    for turn in (np.pi / 2.0, np.pi):
        desired = (angles + turn - angles[0]) % period + angles[0]
        right = np.searchsorted(angles, desired) % len(angles)
        left = (right - 1) % len(angles)
        right_error = np.abs((angles[right] - desired + np.pi) % period - np.pi)
        left_error = np.abs((angles[left] - desired + np.pi) % period - np.pi)
        matching = np.where(left_error < right_error, left, right)
        if np.any(np.minimum(right_error, left_error) > 64 * np.finfo(float).eps):
            continue
        rotated = inner.copy()
        cosine, sine = np.cos(turn), np.sin(turn)
        rotated[:, :, 0] = cosine * inner[:, :, 0] - sine * inner[:, :, 1]
        rotated[:, :, 1] = sine * inner[:, :, 0] + cosine * inner[:, :, 1]
        if not np.allclose(rotated, inner[matching], rtol=0.0, atol=margin):
            continue
        if not np.allclose(target_z, target_z[matching], rtol=0.0, atol=4 * margin):
            continue
        for _ in range(4):
            representatives = np.minimum(representatives, representatives[matching])
    return representatives


def _uniform_revolution(inner, target_z, angles, margin):
    """Whether every row is the same meridian at uniformly stepped azimuths."""
    steps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
    if np.any(np.abs(steps - steps[0]) > 64 * np.finfo(float).eps * 2.0 * np.pi):
        return False
    radius = np.linalg.norm(inner[:, :, :2], axis=2)
    direction = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    off_meridian = (inner[:, :, 0] * direction[:, None, 1]
                    - inner[:, :, 1] * direction[:, None, 0])
    return bool(
        np.all(np.abs(radius - radius[:1]) <= margin)
        and np.all(np.abs(inner[:, :, 2] - inner[:1, :, 2]) <= margin)
        and np.all(np.abs(off_meridian) <= margin)
        # Station schedules derived from per-row normals differ by roundoff
        # only; every row then shares the representative's schedule.
        and np.all(np.abs(target_z - target_z[:1]) <= 1e-9 * max(1.0, float(np.max(np.abs(inner)))))
    )


def _revolution_band(inner, outer, wall, *, full_circle):
    """A synthetic azimuth band that settles row 0 of a surface of revolution.

    Returns ``(inner_band, outer_band, center, closed)`` or None. Both grids
    must be one meridian (a full circle or any symmetry sector, at any
    azimuths). The band repeats row 0's meridian at the grid's own step (the
    finest step, rounded to divide the circle, for uneven rows) on both
    sides, as far as a dilated triangle can reach (asin(wall/r) past its
    vertices), or around the whole circle. A sector's cut planes therefore
    never truncate it: every sector shares the full surface's envelope.
    """
    angles = np.unwrap(np.arctan2(inner[:, 0, 1], inner[:, 0, 0]))
    if len(angles) < 2:
        return None
    steps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi] if full_circle else angles)
    if np.any(steps <= 0.0):
        return None
    step = float(steps.min())
    if np.any(np.abs(steps - step) > 64 * np.finfo(float).eps * 2.0 * np.pi):
        # Unevenly spaced rows (an adaptive display grid): the band takes the
        # finest spacing, rounded to divide the circle.
        step = 2.0 * np.pi / np.ceil(2.0 * np.pi / step - 1e-9)
    scale = max(1.0, float(np.max(np.abs(inner))), float(np.max(np.abs(outer))))
    margin = 64 * np.finfo(float).eps * max(scale, wall)
    inner_m = np.stack((np.linalg.norm(inner[:, :, :2], axis=2), inner[:, :, 2]), axis=2)
    outer_m = np.stack((np.linalg.norm(outer[:, :, :2], axis=2), outer[:, :, 2]), axis=2)
    direction = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    off_meridian = inner[:, :, 0] * direction[:, None, 1] - inner[:, :, 1] * direction[:, None, 0]
    if (np.any(np.abs(inner_m - inner_m[:1]) > margin) or np.any(np.abs(off_meridian) > margin)
            or np.any(np.abs(outer_m - outer_m[:1]) > 1e-9 * scale)):
        return None
    radius = inner_m[0, :, 0]
    turns = 2.0 * np.pi / step
    count = (int(np.ceil((float(np.max(np.arcsin(wall / radius))) + 2.0 * step) / step))
             if np.all(radius > 2.0 * wall) else None)
    closed = count is None or 2 * count + 1 >= turns
    if closed:
        # The wall reaches around the axis: the band is the whole circle.
        if abs(turns - round(turns)) > 1e-9 * turns:
            step = 2.0 * np.pi / np.ceil(turns)
            turns = 2.0 * np.pi / step
        count = 0
        phi = angles[0] + step * np.arange(int(round(turns)))
    else:
        phi = angles[0] + step * np.arange(-count, count + 1)

    def band(meridian):
        out = np.empty((len(phi), meridian.shape[0], 3))
        out[:, :, 0] = np.cos(phi)[:, None] * meridian[None, :, 0]
        out[:, :, 1] = np.sin(phi)[:, None] * meridian[None, :, 0]
        out[:, :, 2] = meridian[None, :, 1]
        return out
    return band(inner_m[0]), band(outer_m[0]), count, closed


def _ray_pairs(selected, starts, ends):
    """Expand per-triangle column intervals into (triangle, column) pairs."""
    counts = ends - starts
    indices = np.repeat(selected, counts)
    groups = np.repeat(np.cumsum(counts) - counts, counts)
    columns = np.repeat(starts, counts) + np.arange(len(indices)) - groups
    return indices, columns


def _accumulate_farthest(values, columns, intersections):
    """Maximum of feature intersections per ray, in bounded temporary chunks."""
    for start in range(0, len(columns), 16384):
        selection = slice(start, start + 16384)
        np.maximum.at(values, columns[selection], intersections(selection))


def _axial_outer_envelope(inner, outer, wall, *, full_circle, revolution=False, rows=None):
    """Expose the backward-facing exterior of a radial-monotone rollback.

    A rolled mouth is multi-valued in z, so horizontal radial rays would lose
    a branch of the wall. Axial rays through an ordered radius schedule retain
    it. Face, cylinder and sphere intersections remain owned by
    _offset_intersections; only the ray origins and candidate bounds differ.
    """
    radii = np.linalg.norm(inner[:, :, :2], axis=2)
    if np.any(np.diff(radii, axis=1) <= 0.0):
        raise ValueError(
            "rolled-back offset requires increasing inner radii; both axial "
            "and radial ray schedules reverse at this bend"
        )
    endpoints = np.linalg.norm(outer[:, (0, -1), :2], axis=2)
    # The last ray runs through the mouth edge itself, so the shell ends on
    # the bottom of that edge's ball and its final cells turn with the lip.
    # A normal-offset end inside the rollback would leave them at right
    # angles to the lip's last cells.
    endpoints[:, 1] = np.maximum(endpoints[:, 1], radii[:, -1])
    if np.any(endpoints[:, 1] <= endpoints[:, 0]):
        raise ValueError("rolled-back offset has reversed throat/mouth radii")
    # Spread rays by meridian arclength, not by radius: a rolled lip adds
    # length while its radius barely changes, and rays bunched at one radius
    # leave outer cells too short to keep the inner cells' orientation.
    length = np.concatenate((np.zeros((len(inner), 1)), np.cumsum(
        np.linalg.norm(np.diff(inner, axis=1), axis=2), axis=1)), axis=1)
    progress = length / length[:, -1:]
    targets = endpoints[:, :1] + progress * (endpoints[:, 1:] - endpoints[:, :1])
    triangles = _triangles(inner, full_circle)
    centers = triangles.mean(axis=1)
    bounds = wall + np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
    angles = np.unwrap(np.arctan2(inner[:, 0, 1], inner[:, 0, 0]))
    margin = 64 * np.finfo(float).eps * max(float(np.max(np.abs(inner))), wall)
    representatives = _equivalent_azimuth_rows(
        inner, targets, angles, full_circle=full_circle, margin=margin,
        revolution=revolution,
    )
    origin_z = float(np.max(inner[:, :, 2])) + 2 * wall
    direction = np.array([0.0, 0.0, -1.0])
    # Each dilated triangle lies within asin(wall/r) of its vertices'
    # azimuths: sorted by azimuth, a row only examines its own window.
    vertex_angles = np.unwrap(np.arctan2(triangles[:, :, 1], triangles[:, :, 0]), axis=1)
    vertex_radii = np.linalg.norm(triangles[:, :, :2], axis=2)
    spread = np.arcsin(np.minimum(wall / np.maximum(vertex_radii, wall), 1.0))
    low = np.min(vertex_angles - spread, axis=1)
    high = np.max(vertex_angles + spread, axis=1)
    near_axis = np.any(vertex_radii <= wall, axis=1) | (high - low >= np.pi)
    middle = ((low + high) / 2.0) % (2.0 * np.pi)
    order = np.argsort(middle)
    sorted_middle = middle[order]
    half_width = float(np.max(((high - low) / 2.0)[~near_axis], initial=0.0)) + 1e-9
    always = np.flatnonzero(near_axis)
    result = np.empty_like(outer)
    for row, angle in enumerate(angles):
        if rows is not None and row not in rows:
            continue
        radial = np.array([np.cos(angle), np.sin(angle), 0.0])
        result[row] = targets[row, :, None] * radial
        if representatives[row] < row:
            result[row, :, 2] = result[representatives[row], :, 2]
            continue
        transverse = np.array([-radial[1], radial[0], 0.0])
        a = angle % (2.0 * np.pi)
        windows = [(a - half_width, a + half_width)]
        if a - half_width < 0.0:
            windows.append((a - half_width + 2.0 * np.pi, 2.0 * np.pi))
        if a + half_width > 2.0 * np.pi:
            windows.append((0.0, a + half_width - 2.0 * np.pi))
        nearby = np.concatenate([always] + [
            order[np.searchsorted(sorted_middle, lo):np.searchsorted(sorted_middle, hi, side="right")]
            for lo, hi in windows])
        across = triangles[nearby] @ transverse
        selected = nearby[
            (across.min(axis=1) <= wall + margin)
            & (across.max(axis=1) >= -wall - margin)
        ]
        along = triangles[selected] @ radial
        starts = np.searchsorted(targets[row], along.min(axis=1) - wall - margin)
        ends = np.searchsorted(targets[row], along.max(axis=1) + wall + margin,
                               side="right")
        indices, columns = _ray_pairs(selected, starts, ends)
        # This meridian's own vertex balls belong to the offset body, so their
        # lowest points bound each ray's exterior from below; a triangle whose
        # dilation cannot reach below that bound cannot be the boundary.
        meridian_r = inner[row] @ radial
        reach = wall * wall - (targets[row][:, None] - meridian_r[None, :]) ** 2
        distances = np.max(np.where(
            reach >= 0.0, origin_z - inner[row, None, :, 2] + np.sqrt(np.maximum(reach, 0.0)),
            -np.inf), axis=1)
        # A centroid sphere containing the triangle, enlarged by the wall,
        # bounds how deep its dilation can reach on each vertical ray.
        offset = centers[indices, :2] - targets[row, columns, None] * radial[:2]
        reach = bounds[indices] ** 2 - np.sum(offset * offset, axis=1)
        deepest = origin_z - centers[indices, 2] + np.sqrt(np.maximum(reach, 0.0))
        keep = (reach >= -margin) & (deepest + margin >= distances[columns])
        indices, columns = indices[keep], columns[keep]

        def axial(pick, row=row):
            shifted = triangles[indices[pick]] - targets[row, columns[pick], None, None] * radial
            return _offset_intersections(
                shifted, np.full(len(shifted), origin_z), direction, wall
            )

        _accumulate_farthest(distances, columns, axial)
        if np.any(~np.isfinite(distances)):
            raise ValueError("rolled-back offset has no axial envelope intersection")
        result[row, :, 2] = _follow_absorbed_lip(
            inner[row] @ radial, inner[row, :, 2], targets[row], origin_z - distances)
    return result


# Outer cells must turn with the acoustic cells they enclose; this keeps
# their meridian steps within 75 degrees.
_LIP_ALIGNMENT = np.cos(np.radians(75.0))


def _follow_absorbed_lip(inner_r, inner_z, outer_r, outer_z):
    """Lower outer stations where a lip inside the wall turns down past them.

    A short lip that the wall swallows whole has no exterior beside it: the
    axial rays meet the flat back of the wall while the acoustic cells run
    straight down, and the cells would meet at right angles. Every point
    below an axial ray's farthest intersection is outside the dilated wall,
    so lowering a station keeps it at least the envelope radius from the
    acoustic surface; each station is lowered just enough for its step to
    follow the lip's.
    """
    z = outer_z.copy()
    for j in range(len(z) - 1):
        dr_i, dz_i = inner_r[j + 1] - inner_r[j], inner_z[j + 1] - inner_z[j]
        if dz_i >= 0.0:
            continue
        dt, dz = outer_r[j + 1] - outer_r[j], z[j + 1] - z[j]
        inner_length = np.hypot(dr_i, dz_i)
        if dt * dr_i + dz * dz_i >= _LIP_ALIGNMENT * np.hypot(dt, dz) * inner_length:
            continue
        # Smallest drop with (dt, dz')·d_i >= cos * |(dt, dz')| * |d_i|.
        unit = np.array([dr_i, dz_i]) / inner_length
        normal = np.array([-unit[1], unit[0]])
        along = np.array([dt, dz]) @ normal
        # Rotate the step toward the lip direction at the alignment angle,
        # keeping its radial part dt fixed.
        tangent = np.sqrt(1.0 - _LIP_ALIGNMENT ** 2)
        for sign in (1.0, -1.0):
            direction = _LIP_ALIGNMENT * unit + sign * tangent * normal
            if direction[0] * dt > 0.0 and direction[1] < 0.0:
                z[j + 1] = min(z[j + 1], z[j] + dt * direction[1] / direction[0])
                break
        else:
            z[j + 1] = min(z[j + 1], z[j] + dz_i / inner_length * max(abs(dt), abs(along)))
    return z


def _curls_back(meridian_r, meridian_z) -> bool:
    """Whether a meridian turns back in both radius and z (a curled lip)."""
    return bool(np.any(np.diff(meridian_r) <= 0.0) and np.any(np.diff(meridian_z) <= 0.0))


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


# Largest turn between the offset points drawn around a vertex ball.
_ARC_STEP = np.radians(5.0)


def _meridian_offset_polyline(meridian, side, radius):
    """The untrimmed one-sided offset of a 2D polyline, and each vertex's point.

    Convex vertices get an arc of their vertex ball; at concave ones the two
    offset segments cross and the loop they form is trimmed later.
    """
    steps = np.diff(meridian, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("curled meridian has repeated stations")
    tangent = steps / lengths[:, None]
    normal = side * np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    points = [meridian[0] + radius * normal[0]]
    anchors = [0]
    for k in range(1, len(meridian) - 1):
        a, b = normal[k - 1], normal[k]
        turn = float(np.arctan2(_cross2(a, b), a @ b))
        anchors.append(len(points))
        if side * _cross2(tangent[k - 1], tangent[k]) < 0.0:
            # Turning away from the offset side: the ball's arc joins them.
            count = int(np.ceil(abs(turn) / _ARC_STEP))
            base = np.arctan2(a[1], a[0]) + turn * np.arange(count + 1) / max(count, 1)
            arc = meridian[k] + radius * np.stack((np.cos(base), np.sin(base)), axis=1)
            anchors[-1] += count // 2
            points.extend(arc if count else arc[:1])
        else:
            points.append(meridian[k] + radius * a)
            if turn != 0.0:
                points.append(meridian[k] + radius * b)
    anchors.append(len(points))
    # The mouth vertex's ball, from its normal round to its tip: a lip that
    # ends inside the wall is wrapped there instead of leaving no exterior.
    count = int(np.ceil(0.5 * np.pi / _ARC_STEP))
    base = (np.arctan2(normal[-1, 1], normal[-1, 0])
            - side * 0.5 * np.pi * np.arange(count + 1) / count)
    points.extend(meridian[-1] + radius * np.stack((np.cos(base), np.sin(base)), axis=1))
    return np.asarray(points), np.asarray(anchors)


def _meridian_clearance(points, meridian):
    """Distance from each 2D point to the meridian polyline."""
    from .preview.fidelity import _point_segment_squared

    return np.sqrt(np.min(_point_segment_squared(
        points[:, None], meridian[None, :-1], meridian[None, 1:]), axis=1))


def _trimmed_offset(polyline, meridian, radius):
    """Drop the offset's loops: follow the pieces at least ``radius`` away.

    The polyline is split at all its self-crossings. A piece whose midpoint
    or interior vertex lies closer than ``radius`` to the meridian is inside
    the wall. From the start, the walk takes at each crossing the outside
    branch, jumping forward over a loop; outside slivers it never reaches are
    not part of the sheet. Returns the walked polyline, its arclength, each
    input vertex's arclength on it (a skipped vertex moves to the next join)
    and the joins' arclengths.
    """
    def clearance(points):
        return _meridian_clearance(points, meridian)

    start, step = polyline[:-1], np.diff(polyline, axis=0)
    count = len(step)
    seglen = np.linalg.norm(step, axis=1)
    denominator = _cross2(step[:, None], step[None])
    gap = start[None] - start[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross2(gap, step[None]) / denominator
        u = _cross2(gap, step[:, None]) / denominator
        index = np.arange(count)
        crossing = ((denominator != 0.0) & (t > 0.0) & (t < 1.0) & (u > 0.0) & (u < 1.0)
                    & (np.abs(index[:, None] - index[None]) > 1))
        vertex_at = np.concatenate(([0.0], np.cumsum(seglen)))
        here = (vertex_at[:-1, None] + t * seglen[:, None])[crossing]
        there = (vertex_at[None, :-1] + u * seglen[None, :])[crossing]
    cuts = np.unique(np.concatenate((vertex_at[[0, -1]], here)))

    def point_at(at):
        k = np.clip(np.searchsorted(vertex_at, at, side="right") - 1, 0, count - 1)
        return start[k] + ((at - vertex_at[k]) / seglen[k])[:, None] * step[k]

    floor = radius * (1.0 - 1e-6)
    # A midpoint on a chord of a vertex ball's arc sags inside by at most this.
    keep = clearance(point_at(0.5 * (cuts[:-1] + cuts[1:]))) >= floor * np.cos(0.5 * _ARC_STEP)
    interior = vertex_at[1:-1]
    holds = (interior[None] > cuts[:-1, None]) & (interior[None] < cuts[1:, None])
    keep &= ~np.any(holds & (clearance(polyline[1:-1]) < floor)[None], axis=1)
    if not (keep[0] and keep[-1]):
        raise ValueError("a curled wall's offset ends inside the wall")
    # Each crossing's cut and the cut where the other branch passes it.
    tolerance = 1e-9 * max(1.0, float(vertex_at[-1]))
    partner = {}
    for a, b in zip(np.searchsorted(cuts, here - tolerance), np.searchsorted(cuts, there - tolerance)):
        partner.setdefault(int(a), []).append(int(b))
    first, last = [cuts[0]], []
    piece = 0
    while piece < len(keep) - 1:
        cut = piece + 1
        ahead = [b for b in partner.get(cut, ()) if b > cut and b < len(keep) and keep[b]]
        if ahead:
            last.append(cuts[cut])
            piece = max(ahead)
            first.append(cuts[piece])
        elif keep[cut]:
            piece = cut
        else:
            raise ValueError("a curled wall's offset breaks into separate sheets")
    last.append(cuts[-1])
    first, last = np.asarray(first), np.asarray(last)
    before = np.concatenate(([0.0], np.cumsum(last - first)))

    def trimmed(at):
        k = np.clip(np.searchsorted(first, at, side="right") - 1, 0, len(first) - 1)
        return np.where(at <= last[k], before[k] + np.maximum(at - first[k], 0.0),
                        before[np.minimum(k + 1, len(first) - 1)])

    inside = np.any((interior[:, None] >= first) & (interior[:, None] <= last), axis=1)
    at = np.unique(np.concatenate((first, last, interior[inside])))
    arclength = trimmed(at)
    order = np.argsort(arclength, kind="stable")
    return point_at(at)[order], arclength[order], trimmed(vertex_at), before[1:-1]


def _stations_around_joins(stations, meridian, curve, arclength, joins):
    """Move the stations a trimmed loop collapsed onto a join (the wall's kink).

    Each collapsed run is split at the kink by direction: a station's step
    stays on the flank it follows, so outer cells turn with the acoustic
    cells. The run borrows as many neighbouring stations on each side for
    room, spaced like the acoustic stations they face.
    """
    s = stations.copy()
    steps = np.diff(meridian, axis=0)
    along = np.concatenate(([0.0], np.cumsum(np.linalg.norm(steps, axis=1))))
    tolerance = 1e-9 * max(1.0, float(arclength[-1]))
    probe = 1e-3 * float(arclength[-1])

    def at(where):
        return np.array([np.interp(where, arclength, curve[:, k]) for k in range(2)])

    for join in joins:
        run = np.flatnonzero(np.abs(s - join) <= tolerance)
        if not run.size:
            continue
        a, b = int(run[0]), int(run[-1])
        if a == 0 or b == len(s) - 1:
            raise ValueError("a curled wall's offset loop reaches its end")
        here = at(join)
        incoming, outgoing = here - at(join - probe), at(join + probe) - here
        faces = steps[a - 1:b + 1]
        kink = a - 1 + int(np.count_nonzero(faces @ incoming / np.linalg.norm(incoming)
                                            >= faces @ outgoing / np.linalg.norm(outgoing)))
        low = max(0, a - 1 - max(1, kink - a + 1))
        high = min(len(s) - 1, b + 1 + max(1, b - kink))
        for lo, hi, s0, s1 in ((low, kink, s[low], join), (kink, high, join, s[high])):
            weight = along[lo:hi + 1] - along[lo]
            s[lo:hi + 1] = s0 + (s1 - s0) * weight / weight[-1]
    if np.any(np.diff(s) <= tolerance):
        raise ValueError("a curled wall's outer stations collapse")
    return s


def _curled_meridian_envelope(inner_m, outer_m, radius):
    """Exterior envelope of a meridian that turns back in radius and z.

    On a surface of revolution the distance from a meridian-plane point to the
    surface is its 2D distance to the meridian, so the constant-distance
    exterior is the meridian's one-sided offset with its loops trimmed. Where
    the wall fills a tight curl the exterior has a kink; the stations the
    loop swallowed are spread on both sides of it.
    """
    steps = np.diff(inner_m, axis=0)
    # The raw normal offset says which side the wall is on.
    sides = np.sum(_cross2(steps, outer_m[:-1] - inner_m[:-1]))
    if not abs(sides) > 1e-9 * radius * float(np.sum(np.linalg.norm(steps, axis=1))):
        raise ValueError("the offset does not say which side of a curl the wall is on")
    side = 1.0 if sides > 0.0 else -1.0
    polyline, anchors = _meridian_offset_polyline(inner_m, side, radius)
    curve, arclength, vertex_s, joins = _trimmed_offset(polyline, inner_m, radius)
    s = vertex_s[anchors]
    if np.any(np.abs(joins - s[-1]) <= 1e-9 * max(1.0, float(arclength[-1]))):
        # The mouth's normal point is inside the wall: end on the tip of its ball.
        s[-1] = arclength[-1]
    s = _stations_around_joins(s, inner_m, curve, arclength, joins)
    result = np.stack([np.interp(s, arclength, curve[:, k]) for k in range(2)], axis=1)
    # The walk splits pieces only where the offset crosses itself, so a
    # station can still sit inside another part's reach (the throat end has
    # no ball): every station must keep the radius, up to an arc chord's sag.
    if np.any(_meridian_clearance(result, inner_m) < radius * (1.0 - 1e-6) * np.cos(0.5 * _ARC_STEP)):
        raise ValueError("a curled wall's envelope passes inside the wall")
    if np.any(result[:, 0] <= 0.0):
        raise ValueError("a curled wall's envelope crosses the axis")
    return result


def regularize_outer_offset(
    inner: np.ndarray, outer: np.ndarray, wall: float, *, full_circle: bool,
    rollback: bool = False, revolution: bool = False, rows=None,
) -> np.ndarray:
    """Resample the exterior offset where the normal correspondence has folded.

    An offset thicker than a local concavity's radius is still well-defined:
    its exterior is the envelope, while the normal parameterization contains
    internal loops. Radial rays retain the input grid's angular order instead
    of trying to mesh those loops. The acoustic grid is never changed.

    ``rollback`` lets a radius-monotone profile that turns back in z use
    axial rays instead of refusing; ``revolution`` reuses one row's rays on a
    uniformly sampled surface of revolution. OSSE uses neither. ``rows``
    limits the work to those rows; the others are left unset.
    """
    if revolution and rows is None:
        window = _revolution_band(inner, outer, wall, full_circle=full_circle)
        if window is not None:
            # Only triangles within the wall's angular reach can touch row
            # 0's rays: solve that open band, then turn its row to every row.
            inner_band, outer_band, center, closed = window
            inner_r = np.linalg.norm(inner[0, :, :2], axis=1)
            if rollback and _curls_back(inner_r, inner[0, :, 2]):
                # Neither ray schedule follows a curl; its meridian does.
                radius, z = _curled_meridian_envelope(
                    np.stack((inner_r, inner[0, :, 2]), axis=1),
                    np.stack((np.linalg.norm(outer[0, :, :2], axis=1), outer[0, :, 2]), axis=1),
                    wall).T
            else:
                meridian = regularize_outer_offset(
                    inner_band, outer_band, wall, full_circle=closed,
                    rollback=rollback, rows=(center,))[center]
                radius, z = np.linalg.norm(meridian[:, :2], axis=1), meridian[:, 2]
            angles = np.arctan2(inner[:, 0, 1], inner[:, 0, 0])
            result = np.empty_like(outer)
            result[:, :, 0] = np.cos(angles)[:, None] * radius
            result[:, :, 1] = np.sin(angles)[:, None] * radius
            result[:, :, 2] = z
            return result
    if np.any(np.diff(inner[:, :, 2], axis=1) <= 0.0):
        if not rollback:
            raise ValueError("outer offset envelope requires monotone axial input stations")
        return _axial_outer_envelope(inner, outer, wall, full_circle=full_circle,
                                     revolution=revolution, rows=rows)
    if np.any(outer[:, -1, 2] <= outer[:, 0, 2]):
        raise ValueError(
            "outer offset envelope requires ordered throat and mouth stations"
        )
    triangles = _triangles(inner, full_circle)
    margin = 64.0 * np.finfo(np.float64).eps * max(float(np.max(np.abs(inner))), wall)
    centers = triangles.mean(axis=1)
    bounds = wall + np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
    squared_margin = margin * max(wall, float(np.max(bounds)))
    center_radial_squared = np.sum(centers[:, :2] ** 2, axis=1)
    vertex_angles = np.unwrap(
        np.arctan2(triangles[:, :, 1], triangles[:, :, 0]), axis=1
    )
    vertex_radii = np.linalg.norm(triangles[:, :, :2], axis=2)
    spread = np.arcsin(np.minimum(wall / np.maximum(vertex_radii, wall), 1.0))
    angle_min = np.min(vertex_angles - spread, axis=1)
    angle_max = np.max(vertex_angles + spread, axis=1)
    angular_center = ((angle_min + angle_max) / 2.0) % (2.0 * np.pi)
    angular_half_width = (angle_max - angle_min) / 2.0
    angular_half_width += 64.0 * np.finfo(np.float64).eps
    angular_half_width[np.any(vertex_radii <= wall, axis=1)] = np.pi
    result = np.empty_like(outer)
    angles = np.unwrap(np.arctan2(inner[:, 0, 1], inner[:, 0, 0]))
    target_z = outer[:, :, 2].copy()
    rollback = np.any(np.diff(target_z, axis=1) <= margin, axis=1)
    if np.any(rollback):
        progress = (inner[rollback, :, 2] - inner[rollback, 0:1, 2]) / (
            inner[rollback, -1:, 2] - inner[rollback, 0:1, 2]
        )
        target_z[rollback] = target_z[rollback, 0:1] + progress * (
            target_z[rollback, -1:] - target_z[rollback, 0:1]
        )
    representatives = _equivalent_azimuth_rows(
        inner, target_z, angles, full_circle=full_circle, margin=margin,
        revolution=revolution,
    )
    radius_rows = np.empty(inner.shape[:2])
    for row, angle in enumerate(angles):
        if rows is not None and row not in rows:
            continue
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        # A dilated triangle is the convex hull of its three vertex balls.
        # Their angular tangents conservatively bound its whole silhouette,
        # avoiding two full triangle-coordinate projections for every ray row.
        angular_distance = np.abs(angular_center - angle % (2.0 * np.pi))
        candidates = (angular_distance <= angular_half_width) | (
            angular_distance >= 2.0 * np.pi - angular_half_width
        )
        z_values = target_z[row]
        if representatives[row] < row:
            radii = radius_rows[representatives[row]]
            radius_rows[row] = radii
            result[row] = radii[:, None] * direction
            # Share the proven-equivalent station schedule as well as radii.
            # Derivative roundoff can shift z by a few ulps; using a radius
            # evaluated at a different z is unsafe near a vertical tangent.
            result[row, :, 2] = target_z[representatives[row]]
            continue
        # Vertex balls on this meridian are part of the offset body, so their
        # farthest intersections are guaranteed lower bounds for its exterior.
        # Dense throat grids otherwise test the same deeply internal features
        # millions of times per row even though they cannot be the boundary.
        meridian = inner[row]
        meridian_along = np.einsum("ij,j->i", meridian, direction)
        meridian_transverse_squared = np.maximum(
            np.sum(meridian[:, :2] ** 2, axis=1) - meridian_along**2, 0.0
        )
        sphere_disc = (
            wall * wall
            - meridian_transverse_squared[None, :]
            - (z_values[:, None] - meridian[None, :, 2]) ** 2
        )
        radii = np.max(
            np.where(
                sphere_disc >= 0.0,
                meridian_along[None, :] + np.sqrt(np.maximum(sphere_disc, 0.0)),
                -np.inf,
            ),
            axis=1,
        )
        # Pair only triangles whose expanded axial bounds contain each ray.
        # Doing one vectorized intersection pass per azimuth avoids a Python /
        # NumPy call stack for every sample in a dense live-preview grid.
        selected = np.flatnonzero(candidates)
        starts, ends = _candidate_ray_intervals(
            triangles[selected], z_values, direction, wall, radii, margin
        )
        triangle_indices, columns = _ray_pairs(selected, starts, ends)
        center_along = np.einsum("ij,j->i", centers, direction)
        center_transverse_squared = np.maximum(
            center_radial_squared - center_along**2, 0.0
        )
        # A centroid sphere containing the whole triangle, enlarged by wall,
        # bounds every possible face/edge/vertex intersection from above.
        # Rejecting one that cannot beat a known vertex ball is exact pruning.
        disc = (
            bounds[triangle_indices] ** 2
            - center_transverse_squared[triangle_indices]
            - (centers[triangle_indices, 2] - z_values[columns]) ** 2
        )
        upper = center_along[triangle_indices] + np.sqrt(np.maximum(disc, 0.0))
        keep = (disc >= -squared_margin) & (upper + margin >= radii[columns])
        triangle_indices, columns = triangle_indices[keep], columns[keep]
        # Bound temporary feature arrays even for unusually dense inspection
        # grids or a wall thick enough to cover most of a meridian.
        _accumulate_farthest(radii, columns, lambda selection: _offset_intersections(
            triangles[triangle_indices[selection]],
            z_values[columns[selection]],
            direction,
            wall,
        ))
        if np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
            raise ValueError(
                "outer offset envelope has no positive radial intersection"
            )
        result[row] = radii[:, None] * direction
        result[row, :, 2] = z_values
        radius_rows[row] = radii
    return result


# Added to an envelope's ball radius beyond Wall thickness. Grid vertices sit
# exactly on the enlarged envelope; this covers the CAD fit's and the grid
# chords' departure from it, which the build then measures on the fitted
# faces (wall_distance.certify_outer_envelope) before meshing or export.
ENVELOPE_ALLOWANCE_MM = 0.1


# Extra clearance a build adds after its fitted faces measured short
# (wall_distance.with_envelope_retries); zero outside such a rebuild.
_EXTRA_ALLOWANCE = contextvars.ContextVar("envelope_extra_allowance_mm", default=0.0)


@contextlib.contextmanager
def extra_envelope_allowance(mm: float):
    """Build envelopes inside this block with ``mm`` more clearance."""
    token = _EXTRA_ALLOWANCE.set(float(mm))
    try:
        yield
    finally:
        _EXTRA_ALLOWANCE.reset(token)


def envelope_allowance_mm(wall: float) -> float:
    """Clearance added to an envelope's ball radius beyond Wall thickness."""
    return ENVELOPE_ALLOWANCE_MM + _EXTRA_ALLOWANCE.get() if wall > 0.0 else 0.0


# Identity and freshness checks resolve the same design many times; the
# envelope is a pure function of its inputs, so repeats reuse the result.
_ENVELOPE_CACHE: "collections.OrderedDict[tuple, np.ndarray]" = collections.OrderedDict()
_ENVELOPE_CACHE_SIZE = 32
# Previews and resolves run concurrently in worker threads; the lock keeps the
# lookup, insert and eviction atomic without holding it while computing.
_ENVELOPE_CACHE_LOCK = threading.Lock()


def _cached_envelope(inner, outer, radius, *, full_circle):
    key = (hashlib.blake2b(np.ascontiguousarray(inner).tobytes(), digest_size=16).digest(),
           hashlib.blake2b(np.ascontiguousarray(outer).tobytes(), digest_size=16).digest(),
           inner.shape, float(radius), bool(full_circle))
    with _ENVELOPE_CACHE_LOCK:
        cached = _ENVELOPE_CACHE.get(key)
        if cached is not None:
            _ENVELOPE_CACHE.move_to_end(key)
    if cached is None:
        cached = regularize_outer_offset(inner, outer, radius, full_circle=full_circle,
                                         rollback=True, revolution=True)
        cached.setflags(write=False)
        with _ENVELOPE_CACHE_LOCK:
            _ENVELOPE_CACHE[key] = cached
            while len(_ENVELOPE_CACHE) > _ENVELOPE_CACHE_SIZE:
                _ENVELOPE_CACHE.popitem(last=False)
    return cached.copy()


def exterior_outer_offset(inner, outer, wall, *, full_circle, label="FREEFORM",
                          force=False, allowance=0.0):
    """Return ``(outer, enveloped)``: a healthy offset unchanged, else its exterior.

    A healthy normal or mitred offset is returned as the same object, so its
    bytes cannot change. A folded one is replaced by the constant-distance
    exterior at ``wall + allowance`` when that envelope is a valid grid.
    When it is not (a profile turning back in both radius and z, or an
    envelope that itself folds), the input is returned unchanged with
    ``enveloped=False`` and the caller keeps its existing fold handling: the
    same warning or refusal a design received before envelopes existed.
    ``force`` marks an offset known to fold between samples (FREEFORM's
    curvature check); without an envelope it is refused, as it always was.
    """
    from .errors import control_refusal
    from .freeform import validate_outer_offset_grid

    if not force:
        try:
            validate_outer_offset_grid(inner, outer, full_circle=full_circle, label=label)
            return outer, False
        except ValueError:
            pass
    try:
        candidate = _cached_envelope(inner, outer, wall + allowance, full_circle=full_circle)
        validate_outer_offset_grid(inner, candidate, full_circle=full_circle, label=label)
    except ValueError as exc:
        if force:
            raise control_refusal(
                f"{label} outer wall envelope could not be built where Wall thickness "
                f"reaches the bend radius: {exc}", "Wall thickness",
                "Reduce Wall thickness, soften the profile bend, or choose Bare shell in Mounting.",
                message="The outer wall folds over itself at a tight bend of the profile.") from exc
        return outer, False
    return candidate, True


# Geometries whose outer grid is an envelope, recorded when they are resolved.
# Not a dataclass field: a geometry's identity hash (its fields) stays the
# same whatever clearance its built faces later need.
_ENVELOPED: dict[int, weakref.ref] = {}


def mark_outer_envelope(geometry) -> None:
    """Record that ``geometry``'s outer grid is a constant-distance envelope."""
    key = id(geometry)
    _ENVELOPED[key] = weakref.ref(geometry, lambda _, key=key: _ENVELOPED.pop(key, None))


def has_outer_envelope(geometry) -> bool:
    ref = _ENVELOPED.get(id(geometry))
    return ref is not None and ref() is geometry


def envelope_variant(geometry):
    """``geometry`` with its envelope rebuilt at the current extra clearance.

    Inside a rebuild (``wall_distance.with_envelope_retries``) the envelope
    radius grows by the extra clearance. A FREEFORM wall builder refits its
    own envelope at that clearance, so it gets the geometry unchanged; a
    point-grid wall is re-enveloped from its acoustic grid. The resolved
    geometry itself, and so its identity, never changes.
    """
    extra = _EXTRA_ALLOWANCE.get()
    if extra <= 0.0 or not has_outer_envelope(geometry):
        return geometry
    if getattr(geometry, "freeform_wall", None) is not None:
        return geometry
    from .freeform import validate_outer_offset_grid

    inner = np.asarray(geometry.inner_points, dtype=float)
    outer = np.asarray(geometry.outer_points, dtype=float)
    wall = float(geometry.wall_thickness_mm)
    try:
        rebuilt = regularize_outer_offset(
            inner, outer, wall + envelope_allowance_mm(wall), full_circle=bool(geometry.closed),
            rollback=True, revolution=True)
        validate_outer_offset_grid(inner, rebuilt, full_circle=bool(geometry.closed))
    except ValueError:
        return geometry
    variant = dataclasses.replace(geometry, outer_points=rebuilt)
    mark_outer_envelope(variant)
    return variant
