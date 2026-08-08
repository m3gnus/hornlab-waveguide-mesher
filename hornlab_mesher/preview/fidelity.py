"""Fidelity measurement and error-bounded preview-grid refinement."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def _angle_degrees(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    dots = np.sum(left * right, axis=-1)
    return float(np.degrees(np.arccos(np.clip(np.min(dots), -1.0, 1.0))))


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
        deviations = np.linalg.norm(points[candidates] - chord, axis=2)
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
        deviations = np.linalg.norm(points[:, candidates] - chord, axis=2)

    flat_index = int(np.argmax(deviations))
    candidate_axis = np.unravel_index(flat_index, deviations.shape)[axis]
    split = int(candidates[candidate_axis])
    if normal_step > 0.0 and float(np.max(deviations)) <= 1.0e-14:
        split = int(candidates[len(candidates) // 2])
    return float(np.max(deviations)), normal_step, split, True


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
) -> tuple[NDArray[np.int64], NDArray[np.int64], dict[str, float | int | bool | None]]:
    """Largest-error-first refinement of a canonical candidate grid.

    Each interval is checked at every available interior candidate sample.  A
    chord test uses the true midpoint/interior point against its endpoint chord;
    the normal test uses the analytic normals at the interval endpoints.  The
    two parameter directions compete for the next vertex allocation by their
    normalized worst error.  Using one shared phi-index set is the union-grid
    alternative allowed by P1.2 and retains FREEFORM row correspondence.
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
        candidates: list[tuple[float, int, int | None]] = []
        saw_unmeasured = False
        for axis, indices, closed in (
            (0, t_indices, False),
            (1, phi_indices, closed_phi),
        ):
            for first, last in _intervals(indices, sample_points.shape[axis], closed):
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
                cap_limited = True
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

    t_error = _worst_axis_interval(
        sample_points,
        measure,
        t_indices,
        axis=0,
        closed=False,
        chord_target=directional_chord,
        normal_target=max_normal_step_deg,
    )
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
    measurement_complete = unmeasured_intervals == 0
    achieved_chord = (
        max(np.finfo(np.float64).eps, t_error[1] + phi_error[1])
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
            return (np.roll(samples, -1, axis=1) - np.roll(samples, 1, axis=1)) / (
                2.0 * step
            )
        coordinates = np.linspace(0.0, 1.0, n_phi, dtype=np.float64)
        return np.gradient(samples, coordinates, axis=1, edge_order=2)

    phi = np.asarray(phi_coordinates, dtype=np.float64)
    if phi.shape != (n_t, n_phi):
        raise ValueError("azimuth coordinates do not match the surface grid")
    if closed_phi:
        unwrapped = np.unwrap(phi, axis=1)
        previous_phi = np.roll(unwrapped, 1, axis=1)
        previous_phi[:, 0] -= math.tau
        next_phi = np.roll(unwrapped, -1, axis=1)
        next_phi[:, -1] += math.tau
        h_previous = unwrapped - previous_phi
        h_next = next_phi - unwrapped
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
        return (
            coefficient_previous.reshape((n_t, n_phi) + trailing)
            * np.roll(samples, 1, axis=1)
            + coefficient_center.reshape((n_t, n_phi) + trailing) * samples
            + coefficient_next.reshape((n_t, n_phi) + trailing)
            * np.roll(samples, -1, axis=1)
        )
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

    result = np.empty((out_t, out_phi, 3), dtype=np.float64)
    for jt, t_coord in enumerate(t_coords):
        t0 = min(int(math.floor(t_coord)), src_t - 1)
        t1 = min(t0 + 1, src_t - 1)
        wt = t_coord - t0
        for ip, phi_coord in enumerate(phi_coords):
            p0 = int(math.floor(phi_coord))
            if closed_phi:
                p0 %= src_phi
                p1 = (p0 + 1) % src_phi
            else:
                p0 = min(p0, src_phi - 1)
                p1 = min(p0 + 1, src_phi - 1)
            wp = phi_coord - math.floor(phi_coord)
            a = source[t0, p0] * (1.0 - wp) + source[t0, p1] * wp
            b = source[t1, p0] * (1.0 - wp) + source[t1, p1] * wp
            result[jt, ip] = a * (1.0 - wt) + b * wt
    return _normalise(result)


def _bilinear_coarse_points(
    coarse: NDArray[np.float64],
    reference_shape: tuple[int, int],
    *,
    closed_phi: bool,
) -> NDArray[np.float64]:
    """Evaluate the coarse chord surface at the dense reference parameters."""

    coarse_t, coarse_phi, _ = coarse.shape
    ref_t, ref_phi = reference_shape
    t_coords = np.linspace(0.0, coarse_t - 1.0, ref_t)
    if closed_phi:
        phi_coords = np.arange(ref_phi, dtype=np.float64) * coarse_phi / ref_phi
    else:
        phi_coords = np.linspace(0.0, coarse_phi - 1.0, ref_phi)
    out = np.empty((ref_t, ref_phi, 3), dtype=np.float64)
    for jt, t_coord in enumerate(t_coords):
        t0 = min(int(math.floor(t_coord)), coarse_t - 1)
        t1 = min(t0 + 1, coarse_t - 1)
        wt = t_coord - t0
        for ip, phi_coord in enumerate(phi_coords):
            p_floor = math.floor(phi_coord)
            p0 = int(p_floor)
            if closed_phi:
                p0 %= coarse_phi
                p1 = (p0 + 1) % coarse_phi
            else:
                p0 = min(p0, coarse_phi - 1)
                p1 = min(p0 + 1, coarse_phi - 1)
            wp = phi_coord - p_floor
            a = coarse[t0, p0] * (1.0 - wp) + coarse[t0, p1] * wp
            b = coarse[t1, p0] * (1.0 - wp) + coarse[t1, p1] * wp
            out[jt, ip] = a * (1.0 - wt) + b * wt
    return out


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
    if any(
        value is not None
        for value in (coarse_t, coarse_phi, reference_t, reference_phi)
    ):
        chord_surface = resample_parametric_grid(
            coarse_points,
            reference_points.shape[:2],
            source_t=coarse_t,
            source_phi=coarse_phi,
            target_t=reference_t,
            target_phi=reference_phi,
            closed_phi=closed_phi,
        )
    else:
        chord_surface = _bilinear_coarse_points(
            coarse_points, reference_points.shape[:2], closed_phi=closed_phi
        )
    chord_error = float(np.max(np.linalg.norm(reference_points - chord_surface, axis=2)))

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
    "estimate_grid_fidelity",
    "resample_parametric_grid",
    "resample_grid_vectors",
]
