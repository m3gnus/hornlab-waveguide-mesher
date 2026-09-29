"""Fidelity measurement and error-bounded preview-grid refinement."""

from __future__ import annotations

import math
import time

import numpy as np
from numpy.typing import NDArray


def _angle_degrees(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    dots = np.sum(left * right, axis=-1)
    minimum_dot = float(np.min(dots))
    if minimum_dot < -1.0:
        clipped = -1.0
    elif minimum_dot > 1.0:
        clipped = 1.0
    else:
        clipped = minimum_dot
    return math.degrees(math.acos(clipped))


def _axis_interval_error(
    points: NDArray[np.float64],
    normals: NDArray[np.float64],
    first: int,
    last: int,
    *,
    axis: int,
    wrapped_length: int | None = None,
    coordinates: NDArray[np.float64] | None = None,
) -> tuple[float, float, int | None, bool]:
    """Measure every available true sample against one parameter chord."""

    if wrapped_length is None:
        candidates = np.arange(first + 1, last, dtype=np.int64)
        span = last - first
        endpoint_last = last
    else:
        span = (last - first) % wrapped_length
        candidates = (first + np.arange(1, span, dtype=np.int64)) % wrapped_length
        endpoint_last = last % wrapped_length
    if span <= 0:
        return 0.0, 0.0, None, False

    if axis == 0:
        normal_step = _angle_degrees(normals[first], normals[endpoint_last])
        if not len(candidates):
            return 0.0, normal_step, None, False
        if coordinates is None:
            weights = np.arange(1, span, dtype=np.float64) / span
        else:
            parameter = np.asarray(coordinates, dtype=np.float64)
            if parameter.shape != (points.shape[0],):
                raise ValueError("axial coordinates do not match the surface grid")
            denominator = parameter[endpoint_last] - parameter[first]
            if not math.isfinite(float(denominator)) or denominator <= 0.0:
                raise ValueError("axial coordinates must be finite and increasing")
            weights = (parameter[candidates] - parameter[first]) / denominator
        weights = weights[:, None, None]
        chord = points[first][None, :, :] * (1.0 - weights) + points[endpoint_last][
            None, :, :
        ] * weights
        deltas = points[candidates] - chord
    else:
        normal_step = _angle_degrees(normals[:, first], normals[:, endpoint_last])
        if not len(candidates):
            return 0.0, normal_step, None, False
        if coordinates is None:
            weights = np.broadcast_to(
                np.arange(1, span, dtype=np.float64)[None, :] / span,
                (points.shape[0], len(candidates)),
            )
        else:
            parameter = np.asarray(coordinates, dtype=np.float64)
            if parameter.shape != points.shape[:2]:
                raise ValueError("azimuth coordinates do not match the surface grid")
            first_values = parameter[:, first]
            last_values = parameter[:, endpoint_last].copy()
            if wrapped_length is not None and endpoint_last <= first:
                last_values += math.tau
            candidate_values = parameter[:, candidates].copy()
            if wrapped_length is not None:
                candidate_values[:, candidates <= first] += math.tau
            denominator = last_values - first_values
            if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
                raise ValueError("azimuth coordinates must be finite and increasing")
            weights = (candidate_values - first_values[:, None]) / denominator[:, None]
        weights = weights[:, :, None]
        chord = points[:, first][:, None, :] * (1.0 - weights) + points[:, endpoint_last][
            :, None, :
        ] * weights
        deltas = points[:, candidates] - chord

    squared_deviations = (deltas * deltas).sum(axis=-1)
    flat_index = int(np.argmax(squared_deviations))
    maximum_deviation = float(
        np.sqrt(squared_deviations.reshape(-1)[flat_index])
    )
    candidate_axis = np.unravel_index(flat_index, squared_deviations.shape)[axis]
    split = int(candidates[candidate_axis])
    if normal_step > 0.0 and maximum_deviation <= 1.0e-14:
        split = int(candidates[len(candidates) // 2])
    return maximum_deviation, normal_step, split, True


def _intervals(indices: list[int], size: int, closed: bool) -> list[tuple[int, int]]:
    ordered = sorted(set(indices))
    result = list(zip(ordered[:-1], ordered[1:]))
    if closed and len(ordered) > 1:
        result.append((ordered[-1], ordered[0]))
    return result


class _IntervalErrors:
    """Memo of one refinement's interval measurements.

    Refinement re-measures every interval after every round, but a round only
    changes the intervals it split -- the rest return the same numbers, since
    the candidate grid and both parameter coordinate arrays are fixed for the
    whole call. On the seed R-OSSE design roughly two of every three
    measurements were repeats of one already taken.
    """

    def __init__(
        self,
        points: NDArray[np.float64],
        normals: NDArray[np.float64],
        *,
        t_coordinates: NDArray[np.float64] | None,
        phi_coordinates: NDArray[np.float64] | None,
    ) -> None:
        self._points = points
        self._normals = normals
        self._coordinates = (t_coordinates, phi_coordinates)
        self._cache: dict[tuple[int, int, int, bool], tuple[float, float, int | None, bool]] = {}

    def __call__(
        self, axis: int, first: int, last: int, *, closed: bool
    ) -> tuple[float, float, int | None, bool]:
        key = (axis, first, last, closed)
        measurement = self._cache.get(key)
        if measurement is None:
            measurement = _axis_interval_error(
                self._points,
                self._normals,
                first,
                last,
                axis=axis,
                wrapped_length=self._points.shape[axis] if closed else None,
                coordinates=self._coordinates[axis],
            )
            self._cache[key] = measurement
        return measurement


def _worst_axis_interval(
    points: NDArray[np.float64],
    measure: "_IntervalErrors",
    indices: list[int],
    *,
    axis: int,
    closed: bool,
    chord_target: float,
    normal_target: float,
) -> tuple[float, float, float, int | None, int]:
    worst_score = -1.0
    worst_chord = 0.0
    worst_normal = 0.0
    worst_split: int | None = None
    unmeasured = 0
    size = points.shape[axis]
    for first, last in _intervals(indices, size, closed):
        chord, normal, split, measured = measure(axis, first, last, closed=closed)
        if not measured:
            unmeasured += 1
        score = max(chord / max(chord_target, 1.0e-15), normal / normal_target)
        if score > worst_score + 1.0e-15:
            worst_score = score
            worst_chord = chord
            worst_normal = normal
            worst_split = split
    return worst_score, worst_chord, worst_normal, worst_split, unmeasured


def _point_segment_squared(
    points: NDArray[np.float64],
    start: NDArray[np.float64],
    end: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Squared distance from each point to the segment ``start``-``end``."""

    edge = end - start
    length = np.einsum("...i,...i->...", edge, edge)
    offset = points - start
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.einsum("...i,...i->...", offset, edge) / length
    fraction = np.where(length > 0.0, np.clip(fraction, 0.0, 1.0), 0.0)
    delta = offset - fraction[..., None] * edge
    return np.einsum("...i,...i->...", delta, delta)


def _triangle_plane_squared(
    points: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Squared plane distance, and whether each point projects inside ``abc``.

    A degenerate (zero-area) triangle contains no projection.
    """

    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    area2 = np.einsum("...i,...i->...", normal, normal)
    offset = points - a
    height = np.einsum("...i,...i->...", offset, normal)
    usable = area2 > 0.0
    ratio = np.divide(height, area2, out=np.zeros_like(height), where=usable)
    projected = offset - ratio[..., None] * normal
    inside = usable.copy()
    # Edge-side tests on the projected point, each relative to its edge's start.
    for edge, relative in (
        (ab, projected),
        (c - b, projected - ab),
        (-ac, projected - ac),
    ):
        side = np.einsum("...i,...i->...", np.cross(edge, relative), normal)
        inside &= side >= 0.0
    plane = np.where(usable, height * ratio, np.inf)
    return plane, inside


def _point_triangle_squared(
    points: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Squared distance from each point to the solid triangle ``abc``.

    The plane distance applies where the point projects inside the triangle,
    and the nearest edge everywhere else. A degenerate (zero-area) triangle is
    its edges.
    """

    plane, inside = _triangle_plane_squared(points, a, b, c)
    edges = np.minimum(
        _point_segment_squared(points, a, b),
        np.minimum(
            _point_segment_squared(points, b, c),
            _point_segment_squared(points, c, a),
        ),
    )
    return np.where(inside, np.minimum(plane, edges), edges)


def _quad_triangle_squared(
    points: NDArray[np.float64],
    p00: NDArray[np.float64],
    p01: NDArray[np.float64],
    p10: NDArray[np.float64],
    p11: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Squared distance to the two triangles a grid quad is emitted as.

    ``p00``/``p01`` are the quad's lower-``t`` row (``phi`` increasing) and
    ``p10``/``p11`` its upper one. The preview splits every quad on the
    ``p00``-``p11`` diagonal (``preview.api._grid_indices``), so these are the
    planar faces a renderer draws -- not the bilinear patch through the same
    four corners, which a twisted quad departs from.

    A point that projects into one of the two triangles takes that plane's
    distance (the smaller one if it projects into both); only the rest pay for
    the five edges. This can only read high, never low: a point inside one
    triangle is charged that triangle even if the other is marginally nearer.
    """

    shape = np.broadcast_shapes(points.shape, p00.shape)
    points, p00, p01, p10, p11 = (
        np.broadcast_to(value, shape).reshape(-1, 3)
        for value in (points, p00, p01, p10, p11)
    )
    plane_a, inside_a = _triangle_plane_squared(points, p00, p01, p11)
    plane_b, inside_b = _triangle_plane_squared(points, p00, p11, p10)
    squared = np.full(len(points), np.inf, dtype=np.float64)
    squared[inside_a] = plane_a[inside_a]
    squared[inside_b] = np.minimum(squared[inside_b], plane_b[inside_b])
    outside = ~(inside_a | inside_b)
    if np.any(outside):
        q = points[outside]
        a, b, c, d = p00[outside], p01[outside], p11[outside], p10[outside]
        edges = _point_segment_squared(q, a, b)
        for start, end in ((b, c), (c, d), (d, a), (a, c)):
            edges = np.minimum(edges, _point_segment_squared(q, start, end))
        squared[outside] = edges
    return squared.reshape(shape[:-1])


def _cell_lookup(
    selected: list[int] | NDArray[np.int64], size: int, *, closed: bool
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Map every candidate index to the selected interval containing it.

    Returns the interval number per candidate and a mask of candidates that
    lie inside the selection (an open selection need not span every sample).
    Candidates on a selected station belong to the interval that starts there
    (the last one closes the final interval of an open selection).
    """

    stations = np.asarray(sorted(set(int(value) for value in selected)), dtype=np.int64)
    if not closed and len(stations) < 2:
        raise ValueError("an open selection needs at least two stations")
    candidates = np.arange(size, dtype=np.int64)
    cell = np.searchsorted(stations, candidates, side="right") - 1
    if closed:
        cell = np.where(cell < 0, len(stations) - 1, cell)
        return cell, np.ones(size, dtype=bool)
    inside = (candidates >= stations[0]) & (candidates <= stations[-1])
    cell = np.clip(cell, 0, len(stations) - 2)
    return cell, inside


def emitted_triangle_errors(
    points: NDArray[np.float64],
    t_indices: list[int] | NDArray[np.int64],
    phi_indices: list[int] | NDArray[np.int64],
    *,
    closed_phi: bool,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
    """Distance of every true-surface sample from the triangles emitted for it.

    ``points`` is the ``(t, phi, xyz)`` candidate grid and the index lists are
    the stations the render grid keeps. A sample strictly inside a quad is
    measured against the quad's two triangles -- the samples neither
    parameter-direction chord sees; a sample on a kept row or column is
    measured against that grid line's segment, which is the triangles' shared
    edge; a sample on both is a vertex. Returns the distances, the ``(t, phi)``
    quad of every sample, and a mask of the samples the selection covers.
    """

    grid = np.asarray(points, dtype=np.float64)
    t_stations = np.asarray(sorted(set(int(v) for v in t_indices)), dtype=np.int64)
    phi_stations = np.asarray(sorted(set(int(v) for v in phi_indices)), dtype=np.int64)
    n_t, n_phi = grid.shape[:2]
    t_cell, t_inside = _cell_lookup(t_stations, n_t, closed=False)
    phi_cell, phi_inside = _cell_lookup(phi_stations, n_phi, closed=closed_phi)
    covered = t_inside[:, None] & phi_inside[None, :]
    on_t = np.zeros(n_t, dtype=bool)
    on_t[t_stations] = True
    on_phi = np.zeros(n_phi, dtype=bool)
    on_phi[phi_stations] = True
    lower_t = t_stations[t_cell]
    upper_t = t_stations[t_cell + 1]
    lower_phi = phi_stations[phi_cell]
    upper_phi = phi_stations[(phi_cell + 1) % len(phi_stations)]

    squared = np.zeros((n_t, n_phi), dtype=np.float64)
    interior = covered & ~on_t[:, None] & ~on_phi[None, :]
    rows, cols = np.nonzero(interior)
    if len(rows):
        squared[rows, cols] = _quad_triangle_squared(
            grid[rows, cols],
            grid[lower_t[rows], lower_phi[cols]],
            grid[lower_t[rows], upper_phi[cols]],
            grid[upper_t[rows], lower_phi[cols]],
            grid[upper_t[rows], upper_phi[cols]],
        )
    # On a kept column, between kept rows: the column's own segment.
    rows, cols = np.nonzero(covered & ~on_t[:, None] & on_phi[None, :])
    if len(rows):
        squared[rows, cols] = _point_segment_squared(
            grid[rows, cols], grid[lower_t[rows], cols], grid[upper_t[rows], cols]
        )
    # On a kept row, between kept columns: the row's own segment.
    rows, cols = np.nonzero(covered & on_t[:, None] & ~on_phi[None, :])
    if len(rows):
        squared[rows, cols] = _point_segment_squared(
            grid[rows, cols], grid[rows, lower_phi[cols]], grid[rows, upper_phi[cols]]
        )
    return np.sqrt(squared), t_cell, phi_cell, covered


def adaptive_grid_indices(
    points: NDArray[np.float64],
    normals: NDArray[np.float64],
    initial_t: NDArray[np.int64] | list[int],
    initial_phi: NDArray[np.int64] | list[int],
    *,
    max_chord_error_mm: float,
    max_normal_step_deg: float,
    max_vertices: int | None,
    closed_phi: bool,
    t_coordinates: NDArray[np.float64] | None = None,
    phi_coordinates: NDArray[np.float64] | None = None,
    refinement_deadline: float | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64], dict[str, float | int | bool | None]]:
    """Largest-error-first refinement of a canonical candidate grid.

    Each interval is checked at every available interior candidate sample.  A
    chord test uses the true midpoint/interior point against its endpoint chord;
    the normal test uses the analytic normals at the interval endpoints.  The
    two parameter directions compete for the next vertex allocation by their
    normalized worst error.  Using one shared phi-index set is the union-grid
    alternative allowed by P1.2 and retains FREEFORM row correspondence.
    An optional monotonic deadline stops at refinement/measurement checkpoints;
    the current valid grid is returned with incomplete fidelity, never a false
    chord bound. Sampling and surface assembly do not check the deadline.
    """

    sample_points = np.asarray(points, dtype=np.float64)
    sample_normals = _normalise(np.asarray(normals, dtype=np.float64))
    axial_parameter = (
        None if t_coordinates is None else np.asarray(t_coordinates, dtype=np.float64)
    )
    azimuth_parameter = (
        None
        if phi_coordinates is None
        else np.unwrap(np.asarray(phi_coordinates, dtype=np.float64), axis=1)
    )
    t_indices = sorted({int(value) for value in initial_t})
    phi_indices = sorted({int(value) for value in initial_phi})
    if len(t_indices) < 2 or len(phi_indices) < 3:
        raise ValueError("adaptive preview grid needs at least 2x3 initial stations")

    cap = None if max_vertices is None else max(6, int(max_vertices))
    cap_limited = cap is not None and len(t_indices) * len(phi_indices) > cap
    candidate_starved = False
    if cap_limited:
        # Semantic/end stations have already been inserted.  Retain the axial
        # endpoints and a deterministic spread of the remaining stations.
        max_phi = max(3, cap // 2)
        if len(phi_indices) > max_phi:
            take = np.linspace(0, len(phi_indices) - 1, max_phi, dtype=np.int64)
            phi_indices = [phi_indices[int(index)] for index in take]
        max_t = max(2, cap // len(phi_indices))
        if len(t_indices) > max_t:
            take = np.linspace(0, len(t_indices) - 1, max_t, dtype=np.int64)
            t_indices = [t_indices[int(index)] for index in take]

    def expired() -> bool:
        return refinement_deadline is not None and time.perf_counter() >= refinement_deadline

    def time_limited_result():
        # No further reference scans: endpoint normal steps are cheap and exact
        # for the retained grid, but its triangle chord error is unmeasured.
        selected = sample_normals[np.ix_(t_indices, phi_indices)]
        normal = _angle_degrees(selected[:-1], selected[1:])
        normal = max(normal, _angle_degrees(selected[:, :-1], selected[:, 1:]))
        if closed_phi:
            normal = max(normal, _angle_degrees(selected[:, -1], selected[:, 0]))
        return (
            np.asarray(t_indices, dtype=np.int64),
            np.asarray(phi_indices, dtype=np.int64),
            {
                "max_chord_error_mm": None,
                "max_normal_step_deg": normal,
                "vertex_cap_limited": cap_limited,
                "refinement_time_limited": True,
                "measurement_complete": False,
                "unmeasured_intervals": len(t_indices) - 1 + len(phi_indices) - (not closed_phi),
                "candidate_starved": candidate_starved,
            },
        )

    # Directional chord bounds add under bilinear interpolation, hence each
    # direction receives half the requested surface-error allowance.
    directional_chord = max_chord_error_mm * 0.5
    measure = _IntervalErrors(
        sample_points,
        sample_normals,
        t_coordinates=axial_parameter,
        phi_coordinates=azimuth_parameter,
    )
    while True:
        if expired():
            return time_limited_result()
        candidates: list[tuple[float, int, int | None]] = []
        saw_unmeasured = False
        for axis, indices, closed in (
            (0, t_indices, False),
            (1, phi_indices, closed_phi),
        ):
            for first, last in _intervals(indices, sample_points.shape[axis], closed):
                if expired():
                    return time_limited_result()
                chord, normal, split, measured = measure(
                    axis, first, last, closed=closed
                )
                saw_unmeasured = saw_unmeasured or not measured
                score = max(
                    chord / max(directional_chord, 1.0e-15),
                    normal / max_normal_step_deg,
                )
                if score > 1.0 + 1.0e-12:
                    candidates.append((score, axis, split))
        if not candidates:
            cap_limited = cap_limited or saw_unmeasured
            break
        added = False
        for _score, axis, split in sorted(
            candidates, key=lambda item: (-item[0], item[1], item[2] or -1)
        ):
            if split is None:
                # This interval misses the target and the candidate grid holds
                # nothing between its endpoints, so refinement cannot answer it
                # at this density. Distinct from the vertex cap, which is a
                # budget the caller chose: a denser candidate grid would let
                # refinement continue, and that is what ``candidate_starved``
                # tells the caller.
                cap_limited = True
                candidate_starved = True
                continue
            target = t_indices if axis == 0 else phi_indices
            if split in target:
                continue
            projected = (
                (len(t_indices) + 1) * len(phi_indices)
                if axis == 0
                else len(t_indices) * (len(phi_indices) + 1)
            )
            if cap is not None and projected > cap:
                cap_limited = True
                continue
            target.append(split)
            added = True
        t_indices.sort()
        phi_indices.sort()
        if not added:
            cap_limited = True
            break

    # The directional chords bound the grid LINES; the renderer draws each quad
    # as two planar triangles, and a twisted quad departs from those between its
    # lines even when every line is exact (P(u,v) = (100u, 100v, 4uv) mm is
    # straight along both parameters and ~1 mm off its triangles in the middle).
    # Measure every candidate sample against the triangles actually emitted for
    # it and split any quad that still misses the whole chord budget.
    triangle_error = 0.0
    while True:
        if expired():
            return time_limited_result()
        distances, t_cell, phi_cell, covered = emitted_triangle_errors(
            sample_points, t_indices, phi_indices, closed_phi=closed_phi
        )
        triangle_error = float(np.max(distances[covered])) if np.any(covered) else 0.0
        failing = covered & (distances > max_chord_error_mm * (1.0 + 1.0e-9))
        if not np.any(failing):
            break
        added = False
        t_set = set(t_indices)
        phi_set = set(phi_indices)
        rows, cols = np.nonzero(failing)
        order = np.argsort(-distances[rows, cols], kind="stable")
        seen_cells: set[tuple[int, int]] = set()
        t_sorted = sorted(t_set)
        phi_sorted = sorted(phi_set)

        def t_stations_of(interval: int) -> tuple[int, int]:
            return t_sorted[interval], t_sorted[interval + 1]

        def phi_stations_of(interval: int) -> tuple[int, int]:
            return phi_sorted[interval], phi_sorted[(interval + 1) % len(phi_sorted)]

        for flat in order:
            if expired():
                t_indices.sort()
                phi_indices.sort()
                return time_limited_result()
            row, col = int(rows[flat]), int(cols[flat])
            cell = (int(t_cell[row]), int(phi_cell[col]))
            if cell in seen_cells:
                continue
            seen_cells.add(cell)
            # The worst sample of this quad names the split. Along a grid line
            # it can only split that line's own direction; strictly inside the
            # quad either direction halves the twist, so split across the
            # quad's longer extent and keep its triangles from turning into
            # slivers.
            split_t = row not in t_set
            split_phi = col not in phi_set
            if split_t and split_phi:
                lower_t = t_stations_of(cell[0])
                lower_phi, upper_phi = phi_stations_of(cell[1])
                corners = sample_points[np.ix_(lower_t, (lower_phi, upper_phi))]
                extent_t = float(
                    np.sum(np.linalg.norm(corners[1] - corners[0], axis=-1))
                )
                extent_phi = float(
                    np.sum(np.linalg.norm(corners[:, 1] - corners[:, 0], axis=-1))
                )
                if extent_t >= extent_phi:
                    split_phi = False
                else:
                    split_t = False
            if split_t:
                projected = (len(t_indices) + 1) * len(phi_indices)
                if cap is not None and projected > cap:
                    cap_limited = True
                    continue
                t_indices.append(row)
                t_set.add(row)
                added = True
            elif split_phi:
                projected = len(t_indices) * (len(phi_indices) + 1)
                if cap is not None and projected > cap:
                    cap_limited = True
                    continue
                phi_indices.append(col)
                phi_set.add(col)
                added = True
        t_indices.sort()
        phi_indices.sort()
        if not added:
            cap_limited = True
            break

    if expired():
        return time_limited_result()
    t_error = _worst_axis_interval(
        sample_points,
        measure,
        t_indices,
        axis=0,
        closed=False,
        chord_target=directional_chord,
        normal_target=max_normal_step_deg,
    )
    if expired():
        return time_limited_result()
    phi_error = _worst_axis_interval(
        sample_points,
        measure,
        phi_indices,
        axis=1,
        closed=closed_phi,
        chord_target=directional_chord,
        normal_target=max_normal_step_deg,
    )
    unmeasured_intervals = int(t_error[4] + phi_error[4])
    if expired():
        return time_limited_result()
    measurement_complete = unmeasured_intervals == 0
    # The emitted triangles' measured deviation from every candidate sample
    # they cover, rather than the sum of the two directional line chords: that
    # sum is not a bound for planar triangles (it misses the twist) and it
    # charges tangential parameter drift that moves no surface.
    achieved_chord = (
        max(np.finfo(np.float64).eps, triangle_error)
        if measurement_complete
        else None
    )
    achieved_normal = max(t_error[2], phi_error[2])
    limited = bool(
        cap_limited
        or not measurement_complete
        or (
            achieved_chord is not None
            and achieved_chord > max_chord_error_mm * (1.0 + 1.0e-9)
        )
        or achieved_normal > max_normal_step_deg * (1.0 + 1.0e-9)
    )
    return (
        np.asarray(t_indices, dtype=np.int64),
        np.asarray(phi_indices, dtype=np.int64),
        {
            "max_chord_error_mm": achieved_chord,
            "max_normal_step_deg": achieved_normal,
            "vertex_cap_limited": limited,
            "measurement_complete": measurement_complete,
            "unmeasured_intervals": unmeasured_intervals,
            # True when refinement stopped because the candidate grid ran out
            # of interior samples on an interval that still misses the target.
            # A caller that owns the candidate grid can answer this by making
            # it denser; nothing else it does will help.
            "candidate_starved": candidate_starved,
        },
    )


def _normalise(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 1.0e-14):
        raise ValueError("analytic surface produced a non-finite or zero normal")
    return vectors / lengths


def _phi_derivative(
    values: NDArray[np.float64],
    *,
    closed_phi: bool,
    phi_coordinates: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Differentiate ``(t, phi, ...)`` values along their azimuth rows."""

    samples = np.asarray(values, dtype=np.float64)
    n_t, n_phi = samples.shape[:2]
    if phi_coordinates is None:
        if closed_phi:
            step = math.tau / n_phi
            derivative = np.empty_like(samples)
            if samples.flags.c_contiguous:
                flat_samples = samples.reshape((-1,) + samples.shape[2:])
                flat_derivative = derivative.reshape((-1,) + samples.shape[2:])
                flat_derivative[1:-1] = flat_samples[2:] - flat_samples[:-2]
            else:
                derivative[:, 1:-1] = samples[:, 2:] - samples[:, :-2]
            derivative[:, 0] = samples[:, 1] - samples[:, -1]
            derivative[:, -1] = samples[:, 0] - samples[:, -2]
            derivative /= 2.0 * step
            return derivative
        coordinates = np.linspace(0.0, 1.0, n_phi, dtype=np.float64)
        return np.gradient(samples, coordinates, axis=1, edge_order=2)

    phi = np.asarray(phi_coordinates, dtype=np.float64)
    if phi.shape != (n_t, n_phi):
        raise ValueError("azimuth coordinates do not match the surface grid")
    if closed_phi:
        unwrapped = np.unwrap(phi, axis=1)
        steps = unwrapped[:, 1:] - unwrapped[:, :-1]
        h_previous = np.empty_like(unwrapped)
        h_previous[:, 1:] = steps
        h_previous[:, 0] = unwrapped[:, 0] - (unwrapped[:, -1] - math.tau)
        h_next = np.empty_like(unwrapped)
        h_next[:, :-1] = steps
        h_next[:, -1] = (unwrapped[:, 0] + math.tau) - unwrapped[:, -1]
        if (
            np.any(~np.isfinite(h_previous))
            or np.any(~np.isfinite(h_next))
            or np.any(h_previous <= 0.0)
            or np.any(h_next <= 0.0)
        ):
            raise ValueError("azimuth coordinates must be finite and increasing")
        coefficient_previous = -h_next / (
            h_previous * (h_previous + h_next)
        )
        coefficient_center = (h_next - h_previous) / (h_previous * h_next)
        coefficient_next = h_previous / (h_next * (h_previous + h_next))
        trailing = (1,) * (samples.ndim - 2)
        previous = coefficient_previous.reshape((n_t, n_phi) + trailing)
        center = coefficient_center.reshape((n_t, n_phi) + trailing)
        following = coefficient_next.reshape((n_t, n_phi) + trailing)
        derivative = np.empty_like(samples)
        if samples.flags.c_contiguous:
            flat_samples = samples.reshape((-1,) + samples.shape[2:])
            flat_derivative = derivative.reshape((-1,) + samples.shape[2:])
            flat_previous = previous.reshape((-1,) + trailing)
            flat_center = center.reshape((-1,) + trailing)
            flat_following = following.reshape((-1,) + trailing)
            flat_derivative[1:-1] = (
                flat_previous[1:-1] * flat_samples[:-2]
                + flat_center[1:-1] * flat_samples[1:-1]
                + flat_following[1:-1] * flat_samples[2:]
            )
        else:
            derivative[:, 1:-1] = (
                previous[:, 1:-1] * samples[:, :-2]
                + center[:, 1:-1] * samples[:, 1:-1]
                + following[:, 1:-1] * samples[:, 2:]
            )
        derivative[:, 0] = (
            previous[:, 0] * samples[:, -1]
            + center[:, 0] * samples[:, 0]
            + following[:, 0] * samples[:, 1]
        )
        derivative[:, -1] = (
            previous[:, -1] * samples[:, -2]
            + center[:, -1] * samples[:, -1]
            + following[:, -1] * samples[:, 0]
        )
        return derivative
    derivative = np.empty_like(samples)
    for jt in range(n_t):
        derivative[jt] = np.gradient(
            samples[jt], phi[jt], axis=0, edge_order=2
        )
    return derivative


def _t_derivative(
    values: NDArray[np.float64], t_coordinates: NDArray[np.float64] | None
) -> NDArray[np.float64]:
    samples = np.asarray(values, dtype=np.float64)
    coordinates = (
        np.asarray(t_coordinates, dtype=np.float64)
        if t_coordinates is not None
        else np.arange(samples.shape[0], dtype=np.float64)
    )
    if coordinates.shape != (samples.shape[0],):
        raise ValueError("axial coordinates do not match the surface grid")
    return np.gradient(
        samples,
        coordinates,
        axis=0,
        edge_order=2 if samples.shape[0] >= 3 else 1,
    )


def analytic_grid_normals(
    reference: NDArray[np.float64],
    *,
    closed_phi: bool,
    t_coordinates: NDArray[np.float64] | None = None,
    phi_coordinates: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Normals from central differences of true-surface reference samples.

    Input order is ``(t, phi, xyz)``.  The cross-product convention is exactly
    ``dP/dphi x dP/dt``.  Closed surfaces use periodic central differences in
    phi; endpoints in t use second-order one-sided differences where possible.
    No triangle or triangle normal participates in this calculation.
    """

    points = np.asarray(reference, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("reference surface must have shape (n_t, n_phi, 3)")
    if points.shape[0] < 2 or points.shape[1] < 3:
        raise ValueError("reference surface needs at least 2x3 parameter samples")

    d_phi = _phi_derivative(
        points, closed_phi=closed_phi, phi_coordinates=phi_coordinates
    )
    d_t = _t_derivative(points, t_coordinates)
    return _normalise(np.cross(d_phi, d_t))


def analytic_grid_curvature(
    reference: NDArray[np.float64],
    *,
    closed_phi: bool,
    t_coordinates: NDArray[np.float64] | None = None,
    phi_coordinates: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return signed mean and largest-magnitude principal curvature in 1/mm.

    Input order is ``(t, phi, xyz)``. The first and second fundamental forms
    use the same ``dP/dphi x dP/dt`` normal convention as
    :func:`analytic_grid_normals`; reversing that normal reverses both returned
    signs. ``curvaturePrincipal`` is whichever of ``H +/- sqrt(H^2-K)`` has the
    larger magnitude (the FRAME-SPEC section 5 ``f32[V]`` convention).

    Central differences are used internally, with periodic azimuth derivatives
    for closed grids and second-order one-sided endpoint derivatives where the
    sample count permits. Degenerate metric determinants emit 0.0, never NaN or
    infinity.
    """

    points = np.asarray(reference, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("reference surface must have shape (n_t, n_phi, 3)")
    if points.shape[0] < 2 or points.shape[1] < 3:
        raise ValueError("reference surface needs at least 2x3 parameter samples")

    d_t = _t_derivative(points, t_coordinates)
    d_phi = _phi_derivative(
        points, closed_phi=closed_phi, phi_coordinates=phi_coordinates
    )
    normals = _normalise(np.cross(d_phi, d_t))
    d_tt = _t_derivative(d_t, t_coordinates)
    d_phi_phi = _phi_derivative(
        d_phi, closed_phi=closed_phi, phi_coordinates=phi_coordinates
    )
    # Averaging the two finite-difference orders makes the discrete mixed term
    # symmetric without changing the analytic quantity they approximate.
    d_t_phi = 0.5 * (
        _phi_derivative(
            d_t, closed_phi=closed_phi, phi_coordinates=phi_coordinates
        )
        + _t_derivative(d_phi, t_coordinates)
    )

    e = np.einsum("...i,...i->...", d_t, d_t)
    f = np.einsum("...i,...i->...", d_t, d_phi)
    g = np.einsum("...i,...i->...", d_phi, d_phi)
    l = np.einsum("...i,...i->...", d_tt, normals)
    m = np.einsum("...i,...i->...", d_t_phi, normals)
    n = np.einsum("...i,...i->...", d_phi_phi, normals)
    determinant = e * g - f * f
    determinant_scale = np.maximum(e * g, f * f)
    valid = (
        np.isfinite(determinant)
        & np.isfinite(determinant_scale)
        & (determinant > 64.0 * np.finfo(np.float64).eps * determinant_scale)
    )
    mean = np.zeros(points.shape[:2], dtype=np.float64)
    gaussian = np.zeros_like(mean)
    mean[valid] = (
        e[valid] * n[valid]
        - 2.0 * f[valid] * m[valid]
        + g[valid] * l[valid]
    ) / (2.0 * determinant[valid])
    gaussian[valid] = (
        l[valid] * n[valid] - m[valid] * m[valid]
    ) / determinant[valid]
    root = np.sqrt(np.maximum(0.0, mean * mean - gaussian))
    first = mean + root
    second = mean - root
    principal = np.where(np.abs(first) >= np.abs(second), first, second)
    mean[~np.isfinite(mean)] = 0.0
    principal[~np.isfinite(principal)] = 0.0
    mean[np.abs(mean) <= 64.0 * np.finfo(np.float64).eps] = 0.0
    principal[np.abs(principal) <= 64.0 * np.finfo(np.float64).eps] = 0.0
    return mean, principal


def _coordinate_grid(
    n_t: int,
    n_phi: int,
    t_coordinates: NDArray[np.float64] | None,
    phi_coordinates: NDArray[np.float64] | None,
    *,
    closed_phi: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    t = (
        np.asarray(t_coordinates, dtype=np.float64)
        if t_coordinates is not None
        else np.linspace(0.0, 1.0, n_t)
    )
    if phi_coordinates is None:
        base_phi = (
            np.arange(n_phi, dtype=np.float64) * math.tau / n_phi
            if closed_phi
            else np.linspace(0.0, 1.0, n_phi)
        )
        phi = np.broadcast_to(base_phi, (n_t, n_phi))
    else:
        phi = np.asarray(phi_coordinates, dtype=np.float64)
    if t.shape != (n_t,) or phi.shape != (n_t, n_phi):
        raise ValueError("parameter coordinate shapes do not match the surface grid")
    return t, phi


def _periodic_row_interp(
    row: NDArray[np.float64],
    source_phi: NDArray[np.float64],
    target_phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    order = np.argsort(source_phi)
    phi = np.unwrap(source_phi[order])
    values = row[order]
    phi_ext = np.concatenate((phi[-1:] - math.tau, phi, phi[:1] + math.tau))
    values_ext = np.vstack((values[-1], values, values[0]))
    target = phi[0] + np.mod(target_phi - phi[0], math.tau)
    result = np.empty((len(target), row.shape[1]), dtype=np.float64)
    for component in range(row.shape[1]):
        result[:, component] = np.interp(target, phi_ext, values_ext[:, component])
    return result


def _open_row_interp(
    row: NDArray[np.float64],
    source_phi: NDArray[np.float64],
    target_phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    order = np.argsort(source_phi)
    phi = source_phi[order]
    values = row[order]
    result = np.empty((len(target_phi), row.shape[1]), dtype=np.float64)
    for component in range(row.shape[1]):
        result[:, component] = np.interp(target_phi, phi, values[:, component])
    return result


def resample_parametric_grid(
    source: NDArray[np.float64],
    out_shape: tuple[int, int],
    *,
    source_t: NDArray[np.float64] | None = None,
    source_phi: NDArray[np.float64] | None = None,
    target_t: NDArray[np.float64] | None = None,
    target_phi: NDArray[np.float64] | None = None,
    normalise: bool = False,
    closed_phi: bool = True,
) -> NDArray[np.float64]:
    """Interpolate a grid by its true coordinates, periodically when closed."""

    values = np.asarray(source, dtype=np.float64)
    src_t, src_phi_count, components = values.shape
    out_t, out_phi_count = out_shape
    source_t_values, source_phi_values = _coordinate_grid(
        src_t, src_phi_count, source_t, source_phi, closed_phi=closed_phi
    )
    target_t_values, target_phi_values = _coordinate_grid(
        out_t, out_phi_count, target_t, target_phi, closed_phi=closed_phi
    )
    result = np.empty((out_t, out_phi_count, components), dtype=np.float64)
    for jt, t_value in enumerate(target_t_values):
        upper = int(np.searchsorted(source_t_values, t_value, side="right"))
        upper = min(max(upper, 1), src_t - 1)
        lower = upper - 1
        span = source_t_values[upper] - source_t_values[lower]
        weight = 0.0 if abs(span) <= 1.0e-15 else (t_value - source_t_values[lower]) / span
        interpolate_row = _periodic_row_interp if closed_phi else _open_row_interp
        row_lower = interpolate_row(
            values[lower], source_phi_values[lower], target_phi_values[jt]
        )
        row_upper = interpolate_row(
            values[upper], source_phi_values[upper], target_phi_values[jt]
        )
        result[jt] = row_lower * (1.0 - weight) + row_upper * weight
    return _normalise(result) if normalise else result


def resample_grid_vectors(
    vectors: NDArray[np.float64],
    out_shape: tuple[int, int],
    *,
    closed_phi: bool,
) -> NDArray[np.float64]:
    """Bilinearly resample a dense ``(t, phi, 3)`` vector grid."""

    source = np.asarray(vectors, dtype=np.float64)
    out_t, out_phi = out_shape
    src_t, src_phi, _ = source.shape
    t_coords = np.linspace(0.0, src_t - 1.0, out_t)
    if closed_phi:
        phi_coords = np.arange(out_phi, dtype=np.float64) * src_phi / out_phi
    else:
        phi_coords = np.linspace(0.0, src_phi - 1.0, out_phi)

    t0 = np.minimum(np.floor(t_coords).astype(np.int64), src_t - 1)
    t1 = np.minimum(t0 + 1, src_t - 1)
    wt = t_coords - t0
    phi_floor = np.floor(phi_coords)
    p0 = phi_floor.astype(np.int64)
    if closed_phi:
        p0 %= src_phi
        p1 = (p0 + 1) % src_phi
    else:
        p0 = np.minimum(p0, src_phi - 1)
        p1 = np.minimum(p0 + 1, src_phi - 1)
    wp = phi_coords - phi_floor
    phi_weight = wp[None, :, None]
    a = (
        source[t0[:, None], p0[None, :]] * (1.0 - phi_weight)
        + source[t0[:, None], p1[None, :]] * phi_weight
    )
    b = (
        source[t1[:, None], p0[None, :]] * (1.0 - phi_weight)
        + source[t1[:, None], p1[None, :]] * phi_weight
    )
    t_weight = wt[:, None, None]
    result = a * (1.0 - t_weight) + b * t_weight
    return _normalise(result)


def _reference_cells(
    coarse_shape: tuple[int, int],
    reference_shape: tuple[int, int],
    *,
    closed_phi: bool,
    coarse_t: NDArray[np.float64] | None,
    coarse_phi: NDArray[np.float64] | None,
    reference_t: NDArray[np.float64] | None,
    reference_phi: NDArray[np.float64] | None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """The coarse quad containing each reference sample, in parameter space.

    Returns the quad's lower ``t`` row per reference row and its lower ``phi``
    column per reference sample ``(ref_t, ref_phi)``. Without coordinates both
    grids span the same parameter range in index space, exactly as
    ``resample_grid_vectors`` maps them; with coordinates each reference row
    is located between the two coarse rows around it, on their blended azimuths.
    """

    coarse_rows, coarse_columns = coarse_shape
    ref_rows, ref_columns = reference_shape
    last_row = max(coarse_rows - 2, 0)
    last_column = max(coarse_columns - 2, 0)
    if all(
        value is None for value in (coarse_t, coarse_phi, reference_t, reference_phi)
    ):
        t_coords = np.linspace(0.0, coarse_rows - 1.0, ref_rows)
        if closed_phi:
            phi_coords = (
                np.arange(ref_columns, dtype=np.float64) * coarse_columns / ref_columns
            )
        else:
            phi_coords = np.linspace(0.0, coarse_columns - 1.0, ref_columns)
        t_cell = np.minimum(np.floor(t_coords).astype(np.int64), last_row)
        phi_cell = np.floor(phi_coords).astype(np.int64)
        if closed_phi:
            phi_cell %= coarse_columns
        else:
            phi_cell = np.minimum(phi_cell, last_column)
        return t_cell, np.broadcast_to(phi_cell, (ref_rows, ref_columns)).copy()

    source_t, source_phi = _coordinate_grid(
        coarse_rows, coarse_columns, coarse_t, coarse_phi, closed_phi=closed_phi
    )
    target_t, target_phi = _coordinate_grid(
        ref_rows, ref_columns, reference_t, reference_phi, closed_phi=closed_phi
    )
    t_cell = np.clip(np.searchsorted(source_t, target_t, side="right") - 1, 0, last_row)
    phi_cell = np.empty((ref_rows, ref_columns), dtype=np.int64)
    for row in range(ref_rows):
        lower, upper = int(t_cell[row]), int(t_cell[row]) + 1
        span = float(source_t[upper] - source_t[lower])
        weight = 0.0 if abs(span) <= 1.0e-15 else float(target_t[row] - source_t[lower]) / span
        blended = (1.0 - weight) * np.unwrap(source_phi[lower]) + weight * np.unwrap(
            source_phi[upper]
        )
        if np.any(np.diff(blended) < 0.0):
            # searchsorted needs an increasing array; a surface that runs
            # against its azimuth would be assigned the wrong quads silently.
            raise ValueError("azimuth coordinates must not decrease along a row")
        target = np.asarray(target_phi[row], dtype=np.float64)
        if closed_phi:
            target = blended[0] + np.mod(target - blended[0], math.tau)
            column = np.searchsorted(blended, target, side="right") - 1
            phi_cell[row] = np.mod(column, coarse_columns)
        else:
            column = np.searchsorted(blended, target, side="right") - 1
            phi_cell[row] = np.clip(column, 0, last_column)
    return t_cell, phi_cell


def estimate_grid_fidelity(
    coarse: NDArray[np.float64],
    reference: NDArray[np.float64],
    normals: NDArray[np.float64],
    *,
    closed_phi: bool,
    coarse_t: NDArray[np.float64] | None = None,
    coarse_phi: NDArray[np.float64] | None = None,
    reference_t: NDArray[np.float64] | None = None,
    reference_phi: NDArray[np.float64] | None = None,
) -> dict[str, float]:
    """Return measured chord and adjacent analytic-normal errors."""

    coarse_points = np.asarray(coarse, dtype=np.float64)
    reference_points = np.asarray(reference, dtype=np.float64)
    t_cell, phi_cell = _reference_cells(
        coarse_points.shape[:2],
        reference_points.shape[:2],
        closed_phi=closed_phi,
        coarse_t=coarse_t,
        coarse_phi=coarse_phi,
        reference_t=reference_t,
        reference_phi=reference_phi,
    )
    n_phi = coarse_points.shape[1]
    t0 = t_cell[:, None]
    t1 = t0 + 1
    p0 = phi_cell
    p1 = (phi_cell + 1) % n_phi if closed_phi else phi_cell + 1
    # Measured against the planar triangles the grid is emitted as, not the
    # bilinear patch through the same corners (see ``_quad_triangle_squared``).
    squared = _quad_triangle_squared(
        reference_points,
        coarse_points[t0, p0],
        coarse_points[t0, p1],
        coarse_points[t1, p0],
        coarse_points[t1, p1],
    )
    chord_error = float(np.sqrt(np.max(squared)))

    unit = _normalise(np.asarray(normals, dtype=np.float64))
    dot_t = np.sum(unit[:-1] * unit[1:], axis=2)
    if closed_phi:
        dot_phi = np.sum(unit * np.roll(unit, -1, axis=1), axis=2)
    else:
        dot_phi = np.sum(unit[:, :-1] * unit[:, 1:], axis=2)
    dots = np.concatenate((dot_t.reshape(-1), dot_phi.reshape(-1)))
    normal_step = float(np.degrees(np.arccos(np.clip(np.min(dots), -1.0, 1.0))))
    return {
        "max_chord_error_mm": max(chord_error, np.finfo(np.float64).eps),
        "max_normal_step_deg": normal_step,
        "reference_density_multiplier": 4,
    }


__all__ = [
    "analytic_grid_curvature",
    "analytic_grid_normals",
    "emitted_triangle_errors",
    "estimate_grid_fidelity",
    "resample_parametric_grid",
    "resample_grid_vectors",
]
