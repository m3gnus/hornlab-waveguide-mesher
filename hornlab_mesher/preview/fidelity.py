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
) -> tuple[float, float, int | None]:
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
        return 0.0, 0.0, None

    if axis == 0:
        normal_step = _angle_degrees(normals[first], normals[endpoint_last])
        if not len(candidates):
            return 0.0, normal_step, None
        weights = np.arange(1, span, dtype=np.float64)[:, None, None] / span
        chord = points[first][None, :, :] * (1.0 - weights) + points[endpoint_last][
            None, :, :
        ] * weights
        deviations = np.linalg.norm(points[candidates] - chord, axis=2)
    else:
        normal_step = _angle_degrees(normals[:, first], normals[:, endpoint_last])
        if not len(candidates):
            return 0.0, normal_step, None
        weights = np.arange(1, span, dtype=np.float64)[None, :, None] / span
        chord = points[:, first][:, None, :] * (1.0 - weights) + points[:, endpoint_last][
            :, None, :
        ] * weights
        deviations = np.linalg.norm(points[:, candidates] - chord, axis=2)

    flat_index = int(np.argmax(deviations))
    candidate_axis = np.unravel_index(flat_index, deviations.shape)[axis]
    split = int(candidates[candidate_axis])
    if normal_step > 0.0 and float(np.max(deviations)) <= 1.0e-14:
        split = int(candidates[len(candidates) // 2])
    return float(np.max(deviations)), normal_step, split


def _intervals(indices: list[int], size: int, closed: bool) -> list[tuple[int, int]]:
    ordered = sorted(set(indices))
    result = list(zip(ordered[:-1], ordered[1:]))
    if closed and len(ordered) > 1:
        result.append((ordered[-1], ordered[0]))
    return result


def _worst_axis_interval(
    points: NDArray[np.float64],
    normals: NDArray[np.float64],
    indices: list[int],
    *,
    axis: int,
    closed: bool,
    chord_target: float,
    normal_target: float,
) -> tuple[float, float, float, int | None]:
    worst_score = -1.0
    worst_chord = 0.0
    worst_normal = 0.0
    worst_split: int | None = None
    size = points.shape[axis]
    for first, last in _intervals(indices, size, closed):
        chord, normal, split = _axis_interval_error(
            points,
            normals,
            first,
            last,
            axis=axis,
            wrapped_length=size if closed else None,
        )
        score = max(chord / max(chord_target, 1.0e-15), normal / normal_target)
        if score > worst_score + 1.0e-15:
            worst_score = score
            worst_chord = chord
            worst_normal = normal
            worst_split = split
    return worst_score, worst_chord, worst_normal, worst_split


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
) -> tuple[NDArray[np.int64], NDArray[np.int64], dict[str, float | bool]]:
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
    while True:
        candidates: list[tuple[float, int, int | None]] = []
        for axis, indices, closed in (
            (0, t_indices, False),
            (1, phi_indices, closed_phi),
        ):
            for first, last in _intervals(indices, sample_points.shape[axis], closed):
                chord, normal, split = _axis_interval_error(
                    sample_points,
                    sample_normals,
                    first,
                    last,
                    axis=axis,
                    wrapped_length=sample_points.shape[axis] if closed else None,
                )
                score = max(
                    chord / max(directional_chord, 1.0e-15),
                    normal / max_normal_step_deg,
                )
                if score > 1.0 + 1.0e-12:
                    candidates.append((score, axis, split))
        if not candidates:
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
        sample_normals,
        t_indices,
        axis=0,
        closed=False,
        chord_target=directional_chord,
        normal_target=max_normal_step_deg,
    )
    phi_error = _worst_axis_interval(
        sample_points,
        sample_normals,
        phi_indices,
        axis=1,
        closed=closed_phi,
        chord_target=directional_chord,
        normal_target=max_normal_step_deg,
    )
    achieved_chord = max(np.finfo(np.float64).eps, t_error[1] + phi_error[1])
    achieved_normal = max(t_error[2], phi_error[2])
    limited = bool(
        cap_limited
        or achieved_chord > max_chord_error_mm * (1.0 + 1.0e-9)
        or achieved_normal > max_normal_step_deg * (1.0 + 1.0e-9)
    )
    return (
        np.asarray(t_indices, dtype=np.int64),
        np.asarray(phi_indices, dtype=np.int64),
        {
            "max_chord_error_mm": achieved_chord,
            "max_normal_step_deg": achieved_normal,
            "vertex_cap_limited": limited,
        },
    )


def _normalise(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 1.0e-14):
        raise ValueError("analytic surface produced a non-finite or zero normal")
    return vectors / lengths


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

    if closed_phi and phi_coordinates is None:
        d_phi = 0.5 * (np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1))
    elif closed_phi:
        phi = np.asarray(phi_coordinates, dtype=np.float64)
        d_phi = np.empty_like(points)
        for jt in range(points.shape[0]):
            unwrapped = np.unwrap(phi[jt])
            extended_phi = np.concatenate(
                ([unwrapped[-1] - math.tau], unwrapped, [unwrapped[0] + math.tau])
            )
            extended_points = np.vstack((points[jt, -1], points[jt], points[jt, 0]))
            d_phi[jt] = np.gradient(
                extended_points, extended_phi, axis=0, edge_order=2
            )[1:-1]
    else:
        d_phi = np.gradient(points, axis=1, edge_order=2)
    t_axis = (
        np.asarray(t_coordinates, dtype=np.float64)
        if t_coordinates is not None
        else np.arange(points.shape[0], dtype=np.float64)
    )
    d_t = np.gradient(
        points, t_axis, axis=0, edge_order=2 if points.shape[0] >= 3 else 1
    )
    return _normalise(np.cross(d_phi, d_t))


def _coordinate_grid(
    n_t: int,
    n_phi: int,
    t_coordinates: NDArray[np.float64] | None,
    phi_coordinates: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    t = (
        np.asarray(t_coordinates, dtype=np.float64)
        if t_coordinates is not None
        else np.linspace(0.0, 1.0, n_t)
    )
    if phi_coordinates is None:
        phi = np.broadcast_to(
            np.arange(n_phi, dtype=np.float64) * math.tau / n_phi, (n_t, n_phi)
        )
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


def resample_parametric_grid(
    source: NDArray[np.float64],
    out_shape: tuple[int, int],
    *,
    source_t: NDArray[np.float64] | None = None,
    source_phi: NDArray[np.float64] | None = None,
    target_t: NDArray[np.float64] | None = None,
    target_phi: NDArray[np.float64] | None = None,
    normalise: bool = False,
) -> NDArray[np.float64]:
    """Interpolate a grid by its true t/phi coordinates, periodically in phi."""

    values = np.asarray(source, dtype=np.float64)
    src_t, src_phi_count, _ = values.shape
    out_t, out_phi_count = out_shape
    source_t_values, source_phi_values = _coordinate_grid(
        src_t, src_phi_count, source_t, source_phi
    )
    target_t_values, target_phi_values = _coordinate_grid(
        out_t, out_phi_count, target_t, target_phi
    )
    result = np.empty((out_t, out_phi_count, 3), dtype=np.float64)
    for jt, t_value in enumerate(target_t_values):
        upper = int(np.searchsorted(source_t_values, t_value, side="right"))
        upper = min(max(upper, 1), src_t - 1)
        lower = upper - 1
        span = source_t_values[upper] - source_t_values[lower]
        weight = 0.0 if abs(span) <= 1.0e-15 else (t_value - source_t_values[lower]) / span
        row_lower = _periodic_row_interp(
            values[lower], source_phi_values[lower], target_phi_values[jt]
        )
        row_upper = _periodic_row_interp(
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
    if closed_phi and any(
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
    "analytic_grid_normals",
    "estimate_grid_fidelity",
    "resample_parametric_grid",
    "resample_grid_vectors",
]
