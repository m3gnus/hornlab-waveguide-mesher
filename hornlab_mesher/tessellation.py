"""Shared OCC surface sizing and measured facet deviation, in millimetres."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

_SAGITTA_FLAT_CURVATURE = 1.0e-9
_SAGITTA_GRID_SAMPLES = 24
_SAGITTA_QUANTISE_LOG = math.log(1.05)


class TessellationError(ValueError):
    """A surface could not supply the requested tessellation field."""


def parametric_trim_mask(gmsh: Any, surface: int, uv: np.ndarray) -> np.ndarray:
    """Per-sample in-trim mask for a face, given concatenated (u, v) pairs.

    ``gmsh.model.isInside`` reads like a predicate but is not one: it returns
    *how many* of the points handed to it fall inside the trimmed entity, a
    single integer, not one flag per point. Treating its result as an array
    yields a length-1 array, which is why the guard this replaces silently fell
    back to "everything is inside" on every face it was ever given.

    The batch answer is still worth asking for first, because it settles the
    two common cases outright -- an untrimmed face returns ``n``, a face whose
    parameter rectangle misses it entirely returns ``0`` -- and only a
    genuinely partial face has to pay for the per-point loop.
    """

    count = len(uv) // 2
    if count <= 0:
        return np.ones(0, dtype=bool)
    try:
        total = int(gmsh.model.isInside(2, surface, uv.tolist(), parametric=True))
    except Exception:  # noqa: BLE001 - retain samples when OCC cannot classify them
        # A representation that cannot classify its own points is not evidence
        # that the samples are outside it; keep the face rather than drop it.
        return np.ones(count, dtype=bool)
    if total >= count:
        return np.ones(count, dtype=bool)
    if total <= 0:
        return np.zeros(count, dtype=bool)
    pairs = uv.reshape(-1, 2)
    mask = np.zeros(count, dtype=bool)
    for index in range(count):
        try:
            mask[index] = bool(
                gmsh.model.isInside(2, surface, pairs[index].tolist(), parametric=True)
            )
        except Exception:  # noqa: BLE001 - retain samples when OCC cannot classify them
            mask[index] = True
    return mask


def sagitta_background_view(
    gmsh: Any,
    *,
    deviation_mm: float,
    size_min_mm: float,
    size_max_mm: float,
    samples: int = _SAGITTA_GRID_SAMPLES,
    size_report: dict[int, float] | None = None,
    cover_trim_boundary: bool = False,
    surface_views: dict[int, int] | None = None,
    minimum_size_report: dict[int, float] | None = None,
    end_bias: bool = False,
) -> int:
    """Build a background size view giving every panel the same chord sagitta.

    ``Mesh.MeshSizeFromCurvature = N`` is a segments-per-2pi rule: it targets
    h ~= 2*pi*R/N, so element size scales with the curvature radius and chord
    error grows linearly with R. That over-refines small fillets (an R=5 mm
    fillet gets ~0.04 mm of sagitta) while under-refining large sweeps (R~115 mm
    gets ~0.84 mm), and on the reference return it put 52% of the triangles on
    1.6% of the area. A constant-sagitta rule, h = 2*sqrt(2*R*delta), equalises
    the chord error instead.

    The size field is sampled on each surface's own parameter grid, which
    matters for two independent reasons:

    * Cost. The obvious implementation, a per-query ``setSizeCallback``, needs
      the surface parameters at an arbitrary xyz; that inverse projection
      measures 36 us/point and barely vectorises (1.3x), costing ~18 s of extra
      build time. Going forward from (u,v) instead is free, and
      ``getPrincipalCurvatures`` is 0.7 us/point when handed the whole array.
    * Determinism. Sampling a throwaway coarse *mesh* instead would make the
      field depend on absolute position, because gmsh lays that mesh out in
      world space. A CAD return meshed before recentring then came out with a
      different triangle count than the same body meshed after it, which breaks
      the invariant that moving a design cannot change its acoustics. A
      parameter grid is intrinsic to the surface and carries no such dependency.
    """

    view = gmsh.view.add("sagitta_size")
    payloads: list[np.ndarray] = []
    for _, surface in gmsh.model.getEntities(2):
        first_payload = len(payloads)
        surface_levels = []
        lower, upper = gmsh.model.getParametrizationBounds(2, surface)
        if len(lower) < 2 or len(upper) < 2:
            continue
        u_lo, v_lo = float(lower[0]), float(lower[1])
        u_hi, v_hi = float(upper[0]), float(upper[1])
        if not all(math.isfinite(value) for value in (u_lo, u_hi, v_lo, v_hi)):
            continue
        if u_hi <= u_lo or v_hi <= v_lo:
            continue
        u = np.linspace(u_lo, u_hi, samples)
        v = np.linspace(v_lo, v_hi, samples)
        if end_bias:
            # Short axial returns can occupy less than one uniform grid cell.
            # Cluster the same number of samples at both parameter endpoints.
            t = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, samples)))
            v = v_lo + (v_hi - v_lo) * t
        uu, vv = np.meshgrid(u, v, indexing="ij")
        uv = np.column_stack([uu.ravel(), vv.ravel()]).ravel()
        xyz = np.asarray(gmsh.model.getValue(2, surface, uv.tolist()), dtype=float).reshape(-1, 3)
        cmax, cmin, _, _ = gmsh.model.getPrincipalCurvatures(surface, uv.tolist())
        curvature = np.maximum(np.abs(np.asarray(cmax)), np.abs(np.asarray(cmin)))
        sizes = np.where(
            curvature < _SAGITTA_FLAT_CURVATURE,
            size_max_mm,
            2.0 * np.sqrt(2.0 * deviation_mm / np.maximum(curvature, _SAGITTA_FLAT_CURVATURE)),
        )
        sizes = np.clip(sizes, size_min_mm, size_max_mm)

        # A trimmed face's parameter rectangle extends past the face itself.
        # Those out-of-trim samples still sit on the underlying surface, but
        # they sit over whatever geometry actually occupies that region, and
        # the view joins a ``Min`` field -- so a sample from a tightly curved
        # face that projects outside its own trim can only pull its neighbour's
        # size DOWN, spending triangles on a surface that never asked for them.
        inside = parametric_trim_mask(gmsh, surface, uv)

        grid_xyz = xyz.reshape(samples, samples, 3)
        grid_size = sizes.reshape(samples, samples)
        grid_inside = inside.reshape(samples, samples)
        # Two triangles per parameter cell. Trimmed-away corners still lie on
        # the underlying surface and carry that surface's own sizing, so a
        # trimmed face is covered rather than left with holes in its field.
        a = (slice(0, -1), slice(0, -1))
        b = (slice(1, None), slice(0, -1))
        c = (slice(1, None), slice(1, None))
        d = (slice(0, -1), slice(1, None))
        for corners in ((a, b, c), (a, c, d)):
            pts = [grid_xyz[i].reshape(-1, 3) for i in corners]
            vals = [grid_size[i].reshape(-1) for i in corners]
            # One quantised value per cell rather than three interpolated
            # corner values. Translating a body cannot be bit-exact, and a
            # linearly interpolated field turns that sub-micron coordinate
            # noise into occasional different meshing decisions -- the same
            # design came out with a different triangle count purely for having
            # been recentred. A piecewise-constant field snapped to discrete
            # levels is flat over each cell, so the noise has nothing to tip.
            # See below for why one value per cell is not the same thing as
            # quantising each corner.
            # Parametric opt-in sizing also covers cells crossing a face trim.
            # Requiring all corners leaves an unsized strip at a curved join.
            # CAD retains its existing all-corners rule and byte identity.
            classify = np.logical_or if cover_trim_boundary else np.logical_and
            keep = classify.reduce([grid_inside[i].reshape(-1) for i in corners])
            if not keep.any():
                continue
            pts = [point[keep] for point in pts]
            vals = [value[keep] for value in vals]
            # The cell takes the finest of its own corners before quantising,
            # so a curvature peak caught at one corner is not averaged away by
            # its flatter neighbours, and the three vertices then carry the
            # same value -- which is what makes the field flat over the cell.
            # Quantising each corner separately instead (as this did) leaves
            # three different levels on a cell whose curvature varies, and gmsh
            # interpolates across them, reintroducing exactly the continuous
            # response to sub-micron coordinate noise the ladder exists to
            # remove.
            cell = np.minimum.reduce(vals)
            level = np.clip(
                size_min_mm
                * np.exp(
                    np.floor(np.log(cell / size_min_mm) / _SAGITTA_QUANTISE_LOG)
                    * _SAGITTA_QUANTISE_LOG
                ),
                size_min_mm,
                size_max_mm,
            )
            surface_levels.append(level)
            payloads.append(
                np.column_stack(
                    [pts[0][:, 0], pts[1][:, 0], pts[2][:, 0],
                     pts[0][:, 1], pts[1][:, 1], pts[2][:, 1],
                     pts[0][:, 2], pts[1][:, 2], pts[2][:, 2],
                     level, level, level]
                )
            )
        if size_report is not None and surface_levels:
            levels = np.concatenate(surface_levels)
            size_report[int(surface)] = float(1.0 / np.sqrt(np.mean(1.0 / levels**2)))
        if minimum_size_report is not None and surface_levels:
            minimum_size_report[int(surface)] = float(min(np.min(level) for level in surface_levels))
        if surface_views is not None and len(payloads) > first_payload:
            surface_view = view if not surface_views else gmsh.view.add("sagitta_size")
            data = np.vstack(payloads[first_payload:])
            gmsh.view.addListData(surface_view, "ST", len(data), data.ravel().tolist())
            surface_views[int(surface)] = int(surface_view)

    if not payloads:
        raise TessellationError("mesh sizing: no surface could be sampled for the size field")
    if surface_views is None:
        data = np.vstack(payloads)
        gmsh.view.addListData(view, "ST", len(data), data.ravel().tolist())
    return int(view)


# Barycentric sample lattice for deviation measurement, vertices excluded --
# a mesh vertex lies on the surface by construction, so it measures nothing.
# Edge midpoints and the centroid are where a flat facet departs furthest from
# a curved face, so a small lattice already finds the peak.
_DEVIATION_BARYCENTRIC = (
    (0.5, 0.5, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
    (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
    (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
)


def measure_surface_deviation(
    gmsh: Any, *, triangle_budget: int = 0
) -> dict[str, Any]:
    """Measure how far the generated facets sit from the CAD surfaces they came from.

    This is the achieved value beside the requested one. Nothing else in the
    pipeline measures the geometric fidelity of a solver mesh at all: a sizing
    rule is asked for a deviation and never asked what it delivered.

    Each triangle is projected onto the OCC face it was meshed from -- gmsh
    still holds both, so the association is exact and no nearest-surface search
    is needed. Points are sampled inside the facet, never at its vertices,
    which are on the surface by construction.

    Called with the model still loaded, immediately after ``generate(2)``.
    ``triangle_budget`` of 0 measures every facet, which is what a comparison
    between sizing rules needs: a per-surface stride costs each face at least
    one sample, so it over-weights the many small well-resolved faces, and two
    rules that distribute triangles differently are then biased differently.
    A budget strides deterministically for when the cost matters.
    """

    _, coordinates, _ = gmsh.model.mesh.getNodes()
    nodes = np.asarray(coordinates, dtype=float).reshape(-1, 3)
    tags, _, _ = gmsh.model.mesh.getNodes()
    index = {int(tag): position for position, tag in enumerate(np.asarray(tags).ravel())}

    per_surface: list[tuple[int, np.ndarray]] = []
    total = 0
    for _, surface in gmsh.model.getEntities(2):
        kinds, _, connectivity = gmsh.model.mesh.getElements(2, surface)
        for kind, nodes_flat in zip(kinds, connectivity):
            if int(kind) != 2:  # 3-node triangles only
                continue
            facets = np.asarray(nodes_flat, dtype=np.int64).reshape(-1, 3)
            if facets.size:
                per_surface.append((int(surface), facets))
                total += len(facets)
    if not total:
        return {"triangle_count": 0, "sampled_triangle_count": 0}

    stride = 1 if triangle_budget <= 0 else max(1, math.ceil(total / triangle_budget))
    distances: list[np.ndarray] = []
    worst: tuple[float, int, float] | None = None
    peak_point = None
    peak_type = None
    sampled = 0
    for surface, facets in per_surface:
        chosen = facets[::stride]
        if not len(chosen):
            continue
        sampled += len(chosen)
        corners = np.stack(
            [nodes[[index[int(tag)] for tag in chosen[:, column]]] for column in range(3)]
        )
        points = np.concatenate(
            [
                weights[0] * corners[0] + weights[1] * corners[1] + weights[2] * corners[2]
                for weights in _DEVIATION_BARYCENTRIC
            ]
        )
        projected = np.asarray(
            gmsh.model.getClosestPoint(2, surface, points.ravel().tolist())[0], dtype=float
        ).reshape(-1, 3)
        gaps = np.linalg.norm(points - projected, axis=1)
        distances.append(gaps)
        # Remember the facet that produced this surface's worst sample, so the
        # report can say whether the peak was the sizing rule's doing or the
        # size cap's. A facet already at ``Mesh.MeshSizeMax`` is as fine as the
        # user's rigid target allows, and no deviation dial can improve it.
        local = int(np.argmax(gaps))
        if worst is None or gaps[local] > worst[0]:
            facet = corners[:, local % len(chosen)]
            edges = [
                float(np.linalg.norm(facet[0] - facet[1])),
                float(np.linalg.norm(facet[1] - facet[2])),
                float(np.linalg.norm(facet[2] - facet[0])),
            ]
            worst = (float(gaps[local]), int(surface), max(edges))
            peak_point = points[local].tolist()
            peak_type = gmsh.model.getType(2, surface)

    if not distances:
        return {"triangle_count": total, "sampled_triangle_count": 0}
    everything = np.concatenate(distances)
    return {
        "method": "facet barycentric lattice projected to its originating OCC face",
        "triangle_count": int(total),
        "sampled_triangle_count": int(sampled),
        "sample_count": int(everything.size),
        "peak_mm": float(np.max(everything)),
        "p95_mm": float(np.percentile(everything, 95.0)),
        "rms_mm": float(np.sqrt(np.mean(np.square(everything)))),
        "mean_mm": float(np.mean(everything)),
        "peak_surface": None if worst is None else worst[1],
        "peak_point_mm": peak_point,
        "peak_surface_type": peak_type,
        "peak_facet_longest_edge_mm": None if worst is None else worst[2],
    }
