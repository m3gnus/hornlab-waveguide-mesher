"""Mesh quality measurement: element shape and chord deviation.

Nothing in this pipeline measured either. Only outright degeneracy was caught,
at ``areas <= 1e-15``, so both a sliver with finite area and a facet that chords
a whole roundover in one triangle passed every check on the way to the solver.

The module reports **two independent measures**, because a single one cannot see
both defects:

*Element shape* -- minimum interior angle and the radius ratio ``2*r_in/r_circ``.
The radius ratio rather than an aspect ratio because an aspect ratio cannot tell
a needle from a cap. This is the sliver measure.

*Chord deviation* -- how far the faceting departs from the smooth surface it
approximates, estimated from the dihedral turn across each interior edge and
reported as a length. **A triangle that chords a 3.6 mm roundover radius in one
26 mm facet can be perfectly equilateral**, so every element-shape statistic
passes it while the surface sits millimetres out of position. Measured on the
mesh alone: for two facets meeting across an edge with a turn ``theta`` over a
span ``L``, the arc through them stands off its chord by ``(L/2) * tan(theta/4)``.

Percentiles, not extremes. One bad triangle in 40,000 is not the same problem as
500 of them, and the extreme statistic cannot tell those apart -- see
``docs/mesh-quality.md`` for the measurement that makes this concrete.

Both reports locate their worst elements in ``(z, radius)`` cylindrical
coordinates about the axis, which is how the mouth-rim investigation located its
crossings and is what lets a user find the offending region on the model.

**What this module does not do.** It was commissioned on the hypothesis that
slivers are what makes some meshes hard for an iterative solver, and measured
across the ATH reference archive that hypothesis does not hold: the archive's
worst single triangle, at 2.05 degrees, belongs to a mesh that converges in 19 to
32 iterations at every frequency tried, while a mesh with no triangle under 16
degrees stagnates at three frequencies of five. Removing one mesh's slivers
outright, with its geometry held, cured one of its two stagnating frequencies and
one of six across the archive. So these measures are a statement about the mesh --
the triangles are shaped like triangles, the faceting is where the surface is --
and never a prediction of solver behaviour. ``docs/mesh-quality.md`` carries the
measurement, the controls, and the thresholds each number rests on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


#: Interior angle, in degrees, below which a triangle is counted as a sliver.
#: A counting threshold, not a gate: the gate is on the percentile below.
SLIVER_ANGLE_DEG = 10.0

#: 1st-percentile smallest interior angle at which a mesh is reported. **This
#: number is the gap in the measured population, not a preference.** Over the
#: ATH reference archive the 1st percentile runs 7.35, then 12.00, then 19.70
#: and up; over the application's own mesh library it runs 7.67 (seven meshes),
#: then 20.71 and up. Nothing in either population lies between 8 and 12
#: degrees, so 10 sits on a plateau and it does not matter where in the gap it
#: lands. See ``docs/mesh-quality.md``.
FAIL_P1_ANGLE_DEG = 10.0

#: The advisory band above the gate. **Not** a measured separation -- there is
#: only one gap in the data and ``FAIL_P1_ANGLE_DEG`` is on it. This exists so a
#: mesh drifting toward the bad population is visible before it crosses, and it
#: is documented as a judgement rather than a measurement.
WARN_P1_ANGLE_DEG = 15.0

#: Chord deviation, in millimetres, at which a mesh is reported. Also a measured
#: gap: over the application's mesh library the maximum chord deviation runs
#: 0.29 to 4.23 mm and then jumps to 8.38, with nothing between. The known mouth
#: rollback defect measures 10.74 mm at ATH's default mouth resolution and 1.28
#: mm once the rollback is resolved, so the gap brackets the defect and its own
#: remedy.
FAIL_CHORD_DEVIATION_MM = 5.0

#: The advisory band above the chord gate. A judgement, like ``WARN_P1_ANGLE_DEG``.
WARN_CHORD_DEVIATION_MM = 3.0

#: Retained under the old names so a caller that wants the reporting threshold
#: rather than the gate keeps working.
POOR_P1_ANGLE_DEG = WARN_P1_ANGLE_DEG
POOR_CHORD_DEVIATION_MM = WARN_CHORD_DEVIATION_MM

_UNIT_TO_MM = {"m": 1000.0, "mm": 1.0, "cm": 10.0}


@dataclass(frozen=True)
class ElementLocation:
    """Where a reported element sits, in cylindrical coordinates about the axis."""

    z: float
    radius: float
    min_angle_deg: float
    radius_ratio: float


@dataclass(frozen=True)
class EdgeLocation:
    """Where a reported chord deviation sits."""

    z: float
    radius: float
    deviation_mm: float
    turn_deg: float


@dataclass(frozen=True)
class ElementShapeReport:
    """Distribution of triangle shape over a mesh."""

    triangle_count: int
    excluded_count: int
    min_angle_deg: float | None
    p1_angle_deg: float | None
    p5_angle_deg: float | None
    median_angle_deg: float | None
    min_radius_ratio: float | None
    p1_radius_ratio: float | None
    mean_radius_ratio: float | None
    sliver_count: int
    sliver_fraction: float
    sliver_angle_deg: float
    worst: tuple[ElementLocation, ...] = ()

    @property
    def measured(self) -> bool:
        return self.triangle_count > 0


@dataclass(frozen=True)
class ChordDeviationReport:
    """How far the faceting stands off the surface it approximates."""

    edge_count: int
    max_deviation_mm: float | None
    p99_deviation_mm: float | None
    median_deviation_mm: float | None
    max_turn_deg: float | None
    above_threshold_count: int
    above_threshold_fraction: float
    deviation_threshold_mm: float
    worst: tuple[EdgeLocation, ...] = ()

    @property
    def measured(self) -> bool:
        return self.edge_count > 0


@dataclass(frozen=True)
class QualityGateResult:
    """Verdict of the quality gate, with the reason in the user's own units."""

    passed: bool
    warnings: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()


def _as_arrays(vertices: Any, triangles: Any) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    points = np.asarray(vertices, dtype=float)
    faces = np.asarray(triangles, dtype=np.int64)
    if points.ndim == 1 and points.size % 3 == 0:
        points = points.reshape((-1, 3))
    if faces.ndim == 1 and faces.size % 3 == 0:
        faces = faces.reshape((-1, 3))
    if points.ndim != 2 or points.shape[1:] != (3,):
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        return points, np.empty((0, 3), dtype=np.int64)
    in_range = np.all((faces >= 0) & (faces < len(points)), axis=1)
    return points, faces[in_range]


def _mm_per_unit(vertex_units: str) -> float:
    try:
        return _UNIT_TO_MM[vertex_units]
    except KeyError:
        raise ValueError(
            f"vertex_units must be one of {sorted(_UNIT_TO_MM)}, got {vertex_units!r}"
        ) from None


def _cylindrical(points: NDArray[np.float64], axis: int) -> tuple[NDArray, NDArray]:
    """Split points into an along-axis coordinate and a distance from the axis."""

    others = [index for index in range(3) if index != axis]
    z = points[:, axis]
    radius = np.hypot(points[:, others[0]], points[:, others[1]])
    return z, radius


def element_shape_report(
    vertices: Any,
    triangles: Any,
    *,
    sliver_angle_deg: float = SLIVER_ANGLE_DEG,
    axis: int = 2,
    sample_limit: int = 8,
) -> ElementShapeReport:
    """Report the distribution of triangle shape, not merely whether it exists.

    Computed from the corner coordinates alone rather than through gmsh, so it
    applies equally to a generated mesh, an imported CAD mesh, and one read back
    from an archived ``.msh``. Degenerate faces are excluded from the statistics
    and counted separately: a face with zero area has no meaningful shape, and
    folding it in would drag every reported statistic to zero for a reason the
    caller already has from the topology report.
    """

    points, faces = _as_arrays(vertices, triangles)
    empty = ElementShapeReport(
        triangle_count=0,
        excluded_count=int(len(np.asarray(triangles, dtype=np.int64).reshape(-1, 3)))
        if np.asarray(triangles).size % 3 == 0
        else 0,
        min_angle_deg=None,
        p1_angle_deg=None,
        p5_angle_deg=None,
        median_angle_deg=None,
        min_radius_ratio=None,
        p1_radius_ratio=None,
        mean_radius_ratio=None,
        sliver_count=0,
        sliver_fraction=0.0,
        sliver_angle_deg=float(sliver_angle_deg),
    )
    if not len(faces):
        return empty

    corners = points[faces]
    # Side lengths opposite each corner, so they pair with the angles below.
    a = np.linalg.norm(corners[:, 2] - corners[:, 1], axis=1)
    b = np.linalg.norm(corners[:, 0] - corners[:, 2], axis=1)
    c = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    perimeter = a + b + c
    product = a * b * c
    measurable = (
        np.isfinite(area)
        & np.isfinite(product)
        & (area > 1.0e-15)
        & (product > 0.0)
        & (perimeter > 0.0)
    )
    excluded = int(len(np.asarray(triangles).reshape(-1, 3)) - len(faces))
    excluded += int(np.count_nonzero(~measurable))
    if not np.any(measurable):
        return ElementShapeReport(**{**asdict(empty), "excluded_count": excluded})

    kept = faces[measurable]
    a, b, c = a[measurable], b[measurable], c[measurable]
    area = area[measurable]
    perimeter, product = perimeter[measurable], product[measurable]

    # gamma = 2 * r_in / r_circ, with r_in = 2A/p and r_circ = abc/(4A).
    radius_ratio = 16.0 * area * area / (perimeter * product)
    # Law of cosines on the two shortest sides gives the smallest angle; clip
    # because rounding can push a near-degenerate cosine just outside [-1, 1].
    sides = np.sort(np.stack((a, b, c), axis=1), axis=1)
    shortest, middle, longest = sides[:, 0], sides[:, 1], sides[:, 2]
    cosine = (middle * middle + longest * longest - shortest * shortest) / (
        2.0 * middle * longest
    )
    min_angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    centroids = points[kept].mean(axis=1)
    z, radius = _cylindrical(centroids, axis)
    order = np.argsort(min_angle)[: max(int(sample_limit), 0)]
    sliver_count = int(np.count_nonzero(min_angle < sliver_angle_deg))
    return ElementShapeReport(
        triangle_count=int(len(min_angle)),
        excluded_count=excluded,
        min_angle_deg=float(min_angle.min()),
        p1_angle_deg=float(np.percentile(min_angle, 1.0)),
        p5_angle_deg=float(np.percentile(min_angle, 5.0)),
        median_angle_deg=float(np.median(min_angle)),
        min_radius_ratio=float(radius_ratio.min()),
        p1_radius_ratio=float(np.percentile(radius_ratio, 1.0)),
        mean_radius_ratio=float(radius_ratio.mean()),
        sliver_count=sliver_count,
        sliver_fraction=float(sliver_count / len(min_angle)),
        sliver_angle_deg=float(sliver_angle_deg),
        worst=tuple(
            ElementLocation(
                z=float(z[index]),
                radius=float(radius[index]),
                min_angle_deg=float(min_angle[index]),
                radius_ratio=float(radius_ratio[index]),
            )
            for index in order
        ),
    )


def _interior_edges(faces: NDArray[np.int64]) -> tuple[NDArray, NDArray, NDArray]:
    """Interior edges as (edge vertex pairs, left face, right face).

    Only edges shared by exactly two triangles are returned. A boundary edge has
    no dihedral, and a non-manifold edge has no single one; both are the topology
    report's business rather than this one's.
    """

    corners = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    owner = np.tile(np.arange(len(faces)), 3)
    keys = np.sort(corners, axis=1)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    keys, owner, corners = keys[order], owner[order], corners[order]
    same = np.all(keys[1:] == keys[:-1], axis=1)
    # A run of exactly two is a manifold interior edge. Runs of one or three or
    # more are excluded by requiring the pair not to extend either side.
    pair = np.zeros(len(keys), dtype=bool)
    pair[:-1] = same
    left_of = np.zeros(len(keys), dtype=bool)
    left_of[1:] = same
    exact = pair.copy()
    exact[1:] &= ~same  # the second of a triple cannot open a new pair
    exact[:-1] &= ~np.concatenate(([False], same[:-1]))
    index = np.flatnonzero(exact)
    return keys[index], owner[index], owner[index + 1]


def chord_deviation_report(
    vertices: Any,
    triangles: Any,
    *,
    vertex_units: str = "m",
    deviation_threshold_mm: float = POOR_CHORD_DEVIATION_MM,
    axis: int = 2,
    sample_limit: int = 8,
) -> ChordDeviationReport:
    """Report how far the faceting departs from the surface it approximates.

    Two facets meeting across an interior edge sample a surface that turns
    through the dihedral angle ``theta`` over the span ``L`` between their far
    corners. The arc through them stands off its own chord by
    ``(L/2) * tan(theta/4)``, which is the sagitta and has the units of a length
    the user can act on: a 3.1 mm deviation means the surface is 3.1 mm out of
    position, whatever the element shapes are.

    This measure is deliberately blind to element shape and element shape is
    deliberately blind to it. A facet that swallows a whole roundover can be
    equilateral, and a sliver can lie exactly on the true surface.
    """

    mm = _mm_per_unit(vertex_units)
    points, faces = _as_arrays(vertices, triangles)
    empty = ChordDeviationReport(
        edge_count=0,
        max_deviation_mm=None,
        p99_deviation_mm=None,
        median_deviation_mm=None,
        max_turn_deg=None,
        above_threshold_count=0,
        above_threshold_fraction=0.0,
        deviation_threshold_mm=float(deviation_threshold_mm),
    )
    if len(faces) < 2:
        return empty

    edges, left, right = _interior_edges(faces)
    if not len(edges):
        return empty

    corners = points[faces]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    length = np.linalg.norm(normal, axis=1)
    good = length > 0.0
    unit = np.zeros_like(normal)
    unit[good] = normal[good] / length[good, None]

    keep = good[left] & good[right]
    edges, left, right = edges[keep], left[keep], right[keep]
    if not len(edges):
        return empty

    cosine = np.clip(np.einsum("ij,ij->i", unit[left], unit[right]), -1.0, 1.0)
    turn = np.arccos(cosine)

    # The span the turn is taken over: the two corners not on the shared edge.
    def opposite(face_index: NDArray[np.int64]) -> NDArray[np.int64]:
        tri = faces[face_index]
        on_edge = (tri == edges[:, 0:1]) | (tri == edges[:, 1:2])
        # Exactly one corner per triangle is off the shared edge.
        return tri[~on_edge].reshape(-1)

    far_left, far_right = opposite(left), opposite(right)
    span = np.linalg.norm(points[far_left] - points[far_right], axis=1)
    # Sagitta of the arc that turns by ``turn`` across the chord ``span``.
    deviation_mm = 0.5 * span * np.tan(np.clip(turn, 0.0, np.pi * 0.999) / 4.0) * mm

    midpoints = 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])
    z, radius = _cylindrical(midpoints, axis)
    order = np.argsort(deviation_mm)[::-1][: max(int(sample_limit), 0)]
    above = int(np.count_nonzero(deviation_mm > deviation_threshold_mm))
    return ChordDeviationReport(
        edge_count=int(len(deviation_mm)),
        max_deviation_mm=float(deviation_mm.max()),
        p99_deviation_mm=float(np.percentile(deviation_mm, 99.0)),
        median_deviation_mm=float(np.median(deviation_mm)),
        max_turn_deg=float(np.degrees(turn.max())),
        above_threshold_count=above,
        above_threshold_fraction=float(above / len(deviation_mm)),
        deviation_threshold_mm=float(deviation_threshold_mm),
        worst=tuple(
            EdgeLocation(
                z=float(z[index]),
                radius=float(radius[index]),
                deviation_mm=float(deviation_mm[index]),
                turn_deg=float(np.degrees(turn[index])),
            )
            for index in order
        ),
    )


def gmsh_sicn(gmsh: Any, dimension: int = 2) -> dict[str, Any]:
    """Read gmsh's own signed inverse condition number for the current model.

    Only available while a gmsh model is live, which is why it is separate from
    the two measures above rather than folded into them: an archived ``.msh``,
    an imported CAD mesh and a solver-side triage all have vertices and
    triangles, and none of them has a gmsh handle.
    """

    types, tags, _nodes = gmsh.model.mesh.getElements(dimension)
    values: list[float] = []
    for element_tags in tags:
        if not len(element_tags):
            continue
        values.extend(
            gmsh.model.mesh.getElementQualities(
                [int(tag) for tag in element_tags], "minSICN"
            )
        )
    if not values:
        return {"element_count": 0, "min_sicn": None, "p1_sicn": None, "mean_sicn": None}
    array = np.asarray(values, dtype=float)
    return {
        "element_count": int(len(array)),
        "min_sicn": float(array.min()),
        "p1_sicn": float(np.percentile(array, 1.0)),
        "mean_sicn": float(array.mean()),
    }


def mesh_quality_report(
    vertices: Any,
    triangles: Any,
    *,
    vertex_units: str = "m",
    sliver_angle_deg: float = SLIVER_ANGLE_DEG,
    deviation_threshold_mm: float = POOR_CHORD_DEVIATION_MM,
    axis: int = 2,
    sample_limit: int = 8,
) -> dict[str, Any]:
    """Both measures, as plain data ready to ride beside the integrity report."""

    shape = element_shape_report(
        vertices,
        triangles,
        sliver_angle_deg=sliver_angle_deg,
        axis=axis,
        sample_limit=sample_limit,
    )
    chord = chord_deviation_report(
        vertices,
        triangles,
        vertex_units=vertex_units,
        deviation_threshold_mm=deviation_threshold_mm,
        axis=axis,
        sample_limit=sample_limit,
    )
    return {
        "measured": shape.measured,
        "element_shape": asdict(shape),
        "chord_deviation": asdict(chord),
    }


def evaluate_quality_gate(
    report: dict[str, Any],
    *,
    strict: bool = False,
    warn_p1_angle_deg: float = WARN_P1_ANGLE_DEG,
    fail_p1_angle_deg: float = FAIL_P1_ANGLE_DEG,
    warn_chord_deviation_mm: float = WARN_CHORD_DEVIATION_MM,
    fail_chord_deviation_mm: float = FAIL_CHORD_DEVIATION_MM,
) -> QualityGateResult:
    """Turn a quality report into warnings, and in strict mode into failures.

    **What this gate is and is not.** It is a statement about the mesh: the
    triangles are shaped like triangles, and the faceting is where the surface
    is. It is **not** a prediction of solver behaviour, and it must never be
    described as one. Measured across the ATH reference archive, element shape
    does not sort the meshes an iterative solver struggles with from the ones it
    does not: the archive's worst single triangle, at 2.05 degrees, belongs to a
    mesh that converges in 19 to 32 iterations at every frequency tried, while a
    mesh with no triangle under 16 degrees stagnates at three frequencies of
    five. See ``docs/mesh-quality.md`` for the measurement and the controls.

    The gate is on the **1st-percentile** minimum angle rather than the worst
    triangle, because those two statistics rank the population differently and
    the extreme one ranks it wrongly -- one bad triangle in 40,000 is not the
    same defect as 500 of them.

    Chord deviation is gated on the maximum rather than a percentile, because a
    single facet that chords a whole roundover puts that part of the surface in
    the wrong place however few facets do it. It carries one known blind spot:
    a genuine sharp crease has no arc to deviate from, so a coarse mesh of a
    boxy body can report a deviation that is not a defect. In the measured
    library it does not -- the 90 degree enclosure corners report 0.29 to 1.24
    mm because their facets are small -- but the metric cannot tell the two
    apart on its own, and a caller that meshes a large flat-faced body coarsely
    should read the location before acting on the number.
    """

    warnings: list[str] = []
    failures: list[str] = []

    shape = report.get("element_shape") or {}
    p1_angle = shape.get("p1_angle_deg")
    if p1_angle is not None:
        count = int(shape.get("sliver_count") or 0)
        fraction = float(shape.get("sliver_fraction") or 0.0)
        if p1_angle < warn_p1_angle_deg:
            message = (
                f"Element shape: the 1st-percentile smallest interior angle is "
                f"{p1_angle:.1f} degrees, below {warn_p1_angle_deg:.1f}. "
                f"{count} triangles ({100.0 * fraction:.2f}% of the mesh) are "
                f"under {float(shape.get('sliver_angle_deg') or 0.0):.0f} degrees."
            )
            worst = shape.get("worst") or []
            if worst:
                first = worst[0]
                message += (
                    f" Worst at z={1000.0 * float(first['z']):.1f} mm, "
                    f"r={1000.0 * float(first['radius']):.1f} mm."
                )
            warnings.append(message)
        if p1_angle < fail_p1_angle_deg:
            text = (
                f"Element shape: the 1st-percentile smallest interior angle is "
                f"{p1_angle:.1f} degrees, below the limit of "
                f"{fail_p1_angle_deg:.1f}."
            )
            (failures if strict else warnings).append(text)

    chord = report.get("chord_deviation") or {}
    deviation = chord.get("max_deviation_mm")
    if deviation is not None:
        if deviation > warn_chord_deviation_mm:
            message = (
                f"Chord deviation: the faceting stands {deviation:.2f} mm off the "
                f"surface it approximates, above {warn_chord_deviation_mm:.2f} mm."
            )
            worst = chord.get("worst") or []
            if worst:
                first = worst[0]
                message += (
                    f" Worst at z={1000.0 * float(first['z']):.1f} mm, "
                    f"r={1000.0 * float(first['radius']):.1f} mm, over a "
                    f"{float(first['turn_deg']):.0f} degree turn."
                )
            warnings.append(message)
        if deviation > fail_chord_deviation_mm:
            text = (
                f"Chord deviation: the faceting stands {deviation:.2f} mm off the "
                f"surface, above the limit of {fail_chord_deviation_mm:.2f} mm."
            )
            (failures if strict else warnings).append(text)

    return QualityGateResult(
        passed=not failures,
        warnings=tuple(warnings),
        failures=tuple(failures),
    )
