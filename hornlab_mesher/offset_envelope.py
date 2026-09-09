"""Exterior of a constant-distance offset of a sampled radial surface."""

from __future__ import annotations

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


def _equivalent_azimuth_rows(inner, target_z, angles, *, full_circle, margin):
    """Reuse a rotation only after the entire sampled surface proves it."""
    representatives = np.arange(len(angles))
    if not full_circle:
        return representatives
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


def regularize_outer_offset(
    inner: np.ndarray, outer: np.ndarray, wall: float, *, full_circle: bool
) -> np.ndarray:
    """Resample the exterior offset where the normal correspondence has folded.

    An offset thicker than a local concavity's radius is still well-defined:
    its exterior is the envelope, while the normal parameterization contains
    internal loops. Radial rays retain the input grid's angular order instead
    of trying to mesh those loops. The acoustic grid is never changed.
    """
    if np.any(np.diff(inner[:, :, 2], axis=1) <= 0.0):
        raise ValueError("outer offset envelope requires monotone axial input stations")
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
        inner, target_z, angles, full_circle=full_circle, margin=margin
    )
    radius_rows = np.empty(inner.shape[:2])
    for row, angle in enumerate(angles):
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
        counts = ends - starts
        triangle_indices = np.repeat(selected, counts)
        groups = np.repeat(np.cumsum(counts) - counts, counts)
        columns = np.repeat(starts, counts) + np.arange(len(triangle_indices)) - groups
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
        for start in range(0, len(columns), 16384):
            selection = slice(start, start + 16384)
            intersections = _offset_intersections(
                triangles[triangle_indices[selection]],
                z_values[columns[selection]],
                direction,
                wall,
            )
            np.maximum.at(radii, columns[selection], intersections)
        if np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
            raise ValueError(
                "outer offset envelope has no positive radial intersection"
            )
        result[row] = radii[:, None] * direction
        result[row, :, 2] = z_values
        radius_rows[row] = radii
    return result
