"""Surface-mesh repair: weld, degenerate removal, winding and orientation.

:func:`postprocess_mesh` is the entry point; it repairs the mesh and returns
the validation reports from :mod:`mesh_validation` alongside.
"""

from __future__ import annotations

from collections import defaultdict, deque

import meshio
import numpy as np

from .mesh_validation import (
    _edge_direction_stats,
    _edge_on_expected_plane,
    _signed_volume,
    _signed_volume_noise_floor,
    _topology_stats,
)
from .normals import open_shell_bore_alignment
from .step_mapping import RIGID_TAG, StepFaceGroup
from .step_prepare import millimetres_to_step_units, snap_symmetry_plane_vertices

WELD_TOLERANCE_MM = 5.0e-3  # 5 micrometres; closes near-duplicate OCC patch nodes
DEGENERATE_MIN_QUALITY = 1.0e-4  # drops needle slivers that make dense solves singular

def _mesh_triangle_data(mesh: meshio.Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "triangle" not in mesh.cells_dict:
        raise RuntimeError("mesh has no triangle cells")
    triangles = np.asarray(mesh.cells_dict["triangle"], dtype=np.int64)
    points = np.asarray(mesh.points, dtype=np.float64)
    try:
        tags = np.asarray(mesh.cell_data_dict["gmsh:physical"]["triangle"], dtype=np.int32)
    except KeyError as exc:
        raise RuntimeError("mesh has no gmsh:physical triangle tags") from exc
    return points, triangles, tags


def _remove_degenerate_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
    tags: np.ndarray,
    *,
    eps: float = 1e-18,
    min_quality: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop zero-area triangles and, optionally, needle slivers.

    ``min_quality`` is a scale-invariant shape threshold: a triangle whose area
    falls below ``min_quality * longest_edge**2`` is removed. Fine OCC meshes
    carry micrometre-wide needles bridging near-duplicate patch-boundary nodes
    whose quadrature-degenerate rows make the dense metal-bem solve singular
    (LAPACK info > 0). Ported from hornlab_mesher.normals (commit a5539de).
    """
    if len(triangles) == 0:
        return triangles, tags, 0
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    area2 = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    keep = area2 > eps
    if min_quality > 0.0:
        longest_sq = np.maximum(
            np.maximum(
                np.sum((p1 - p0) ** 2, axis=1),
                np.sum((p2 - p1) ** 2, axis=1),
            ),
            np.sum((p0 - p2) ** 2, axis=1),
        )
        keep &= (0.5 * area2) > (min_quality * longest_sq)
    return triangles[keep], tags[keep], int(np.count_nonzero(~keep))


def _weld_near_duplicate_vertices(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    tol_mm: float = WELD_TOLERANCE_MM,
) -> np.ndarray:
    """Remap triangles so vertices closer than ``tol_mm`` coincide.

    Spatial hash with cells of the weld tolerance; clusters merge to the lowest
    vertex index via union-find. Closes the near-duplicate boundary nodes
    (micrometres apart) that OCC leaves between sewn patches on fine meshes,
    which otherwise seed singular slivers and spurious free edges. Ported from
    hornlab_mesher.mesher (commit a8c2648).
    """
    if len(points) == 0 or len(triangles) == 0:
        return triangles
    cells = np.floor(points / tol_mm).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(map(tuple, cells)):
        buckets.setdefault(key, []).append(index)

    parent = np.arange(len(points))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    tol_sq = tol_mm * tol_mm
    neighbor_offsets = [
        (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    ]
    for key, indices in buckets.items():
        candidates: list[int] = []
        for dx, dy, dz in neighbor_offsets:
            candidates.extend(buckets.get((key[0] + dx, key[1] + dy, key[2] + dz), ()))
        for i in indices:
            pi = points[i]
            for j in candidates:
                if j <= i:
                    continue
                delta = points[j] - pi
                if float(delta @ delta) <= tol_sq:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)

    roots = np.fromiter((find(i) for i in range(len(points))), dtype=np.int64, count=len(points))
    if np.array_equal(roots, np.arange(len(points))):
        return triangles
    return roots[triangles]


def _compact_unused_vertices(
    points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop unreferenced vertices and renumber triangles to the survivors."""
    if len(triangles) == 0:
        return points, triangles
    used = np.unique(triangles)
    if len(used) == len(points):
        return points, triangles
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return points[used], remap[triangles]


def _source_normal_projections(
    points: np.ndarray,
    triangles: np.ndarray,
    tags: np.ndarray,
    source_specs: list[StepFaceGroup],
) -> dict[str, dict[str, object]]:
    projections: dict[str, dict[str, object]] = {}
    for spec in source_specs:
        mask = tags == spec.tag
        if not np.any(mask):
            continue
        tri = triangles[mask]
        p0 = points[tri[:, 0]]
        p1 = points[tri[:, 1]]
        p2 = points[tri[:, 2]]
        vector = np.sum(np.cross(p1 - p0, p2 - p0), axis=0)
        projections[spec.name] = {
            "tag": int(spec.tag),
            "triangle_count": int(len(tri)),
            "vector_step_units2": [float(v) for v in vector],
            "projection_x_step_units2": float(vector[0]),
            "projection_y_step_units2": float(vector[1]),
            "projection_z_step_units2": float(vector[2]),
        }
    return projections


def _free_edges_on_expected_planes(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    symmetry_planes: tuple[str, ...],
    tolerance: float,
) -> bool:
    """True when every free edge lies wholly on a declared symmetry plane."""
    edge_count: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = tuple(sorted((int(a), int(b))))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    return all(
        _edge_on_expected_plane(points, edge, symmetry_planes, tolerance)
        for edge, count in edge_count.items()
        if count == 1
    )


#: Why :func:`_symmetry_source_projection_detail` declined to judge a component.
#: The distinction is load-bearing: ``no_source`` means there is no evidence and
#: the caller must abstain, while ``no_open_axis`` and ``degenerate_projection``
#: mean a cap exists but this particular measure cannot read it, so a different
#: oracle may still be able to.
_SYMMETRY_PROJECTION_OK = "ok"
_SYMMETRY_PROJECTION_NO_SOURCE = "no_source"
_SYMMETRY_PROJECTION_NO_OPEN_AXIS = "no_open_axis"
_SYMMETRY_PROJECTION_DEGENERATE = "degenerate_projection"
_SYMMETRY_PROJECTION_EMPTY = "empty_component"

#: How far the throat-collar bore fraction must sit from 0.5 before it counts as
#: a reading. At exactly half the collar reports that as much wall area faces
#: the bore as faces away, which is not a winding verdict at any threshold.
#:
#: Measured on flared half-horns, shallow waveguides and straight ducts, a
#: single-cap component reads exactly 1.0 wound correctly and exactly 0.0
#: inverted -- the intermediate values all come from geometries where the
#: radial reference is not the bore axis, and those are shapes to decline
#: rather than close calls to decide. 0.25 is therefore loose, not tight; it
#: exists to reject an even split, not to discriminate near one.
#:
#: It is deliberately NOT waveguide-generator's 0.9 bare-shell threshold
#: (server/mesh/integrity.py). That gate judges a whole finished mesh and can
#: demand near-perfection; this one picks a direction for one component and
#: must not turn a readable component into an unjudged one.
_BORE_ALIGNMENT_MARGIN = 0.25


def _symmetry_source_projection_detail(
    points: np.ndarray,
    triangles: np.ndarray,
    tags: np.ndarray,
    *,
    source_tags: set[int],
    symmetry_planes: tuple[str, ...],
) -> tuple[float | None, str]:
    """The source cap's normal projection on the open axis, and why it abstained.

    ``None`` means the component cannot be judged without guessing: it has no
    tagged source cap, does not have exactly two distinct principal cut
    planes, or its cap has no resolvable projection on the remaining axis.

    Collapsing those outcomes onto a bare ``None`` is what let a caller treat
    "this component has no source at all" and "this component has a source I
    cannot project" as the same situation, and apply one fallback to both.
    They need opposite responses, so the reason travels with the value.
    """
    if len(triangles) == 0 or len(tags) != len(triangles):
        return None, _SYMMETRY_PROJECTION_EMPTY

    cut_axes = {
        {"x0": 0, "y0": 1, "z0": 2}[plane]
        for plane in symmetry_planes
    }
    open_axes = sorted({0, 1, 2} - cut_axes)

    source_mask = np.isin(tags, tuple(source_tags))
    if not np.any(source_mask):
        # Checked before the axis count so that a component with no cap reports
        # the reason that actually governs the caller's decision, whatever its
        # cut geometry happens to be.
        return None, _SYMMETRY_PROJECTION_NO_SOURCE

    if len(cut_axes) != 2 or len(open_axes) != 1:
        return None, _SYMMETRY_PROJECTION_NO_OPEN_AXIS

    source_triangles = triangles[source_mask]
    p0 = points[source_triangles[:, 0]]
    p1 = points[source_triangles[:, 1]]
    p2 = points[source_triangles[:, 2]]
    area_vectors = np.cross(p1 - p0, p2 - p0)
    total_area_vector = np.sum(area_vectors, axis=0)
    total_area = float(np.sum(np.linalg.norm(area_vectors, axis=1)))
    projection = float(total_area_vector[open_axes[0]])
    if total_area <= 0.0 or abs(projection) <= 1.0e-12 * total_area:
        return None, _SYMMETRY_PROJECTION_DEGENERATE
    return projection, _SYMMETRY_PROJECTION_OK


def _bore_alignment_verdict(
    points: np.ndarray,
    triangles: np.ndarray,
    tags: np.ndarray,
    *,
    source_tags: set[int],
) -> bool | None:
    """Is this component wound with its walls facing the bore?

    ``True`` keep, ``False`` flip, ``None`` no verdict.

    One cap at a time, and they must agree. ``open_shell_bore_alignment``
    derives its axis and radial origin from the caps it is given, so handing it
    several at once averages them into a reference that need not lie in the
    body at all: measured on a flared half-horn with a 25 mm throat cap and a
    400 mm mouth cap both declared, the combined centroid sat 169 mm off axis
    and an INVERTED mesh read 0.867 -- confident, and wrong. Asked about the
    throat cap alone the same mesh reads 0.0.

    So each declared cap present in the component is asked separately. A cap
    that cannot judge (no collar, a cancelling net area, or a fraction too near
    an even split to be a reading) simply does not vote; the mouth cap of a
    horn usually abstains this way. Disagreement between two caps that both
    voted is itself a reason to decline -- one of them is being measured about
    the wrong axis and this function cannot tell which.
    """
    present = [
        int(tag)
        for tag in np.unique(tags).tolist()
        if int(tag) in source_tags
    ]
    if not present:
        return None
    wall_tags = set(np.unique(tags).tolist()) - source_tags

    votes: set[bool] = set()
    for tag in present:
        alignment = open_shell_bore_alignment(
            points,
            triangles,
            tags,
            source_tags={tag},
            wall_tags=wall_tags,
        )
        if alignment is None or abs(alignment - 0.5) <= _BORE_ALIGNMENT_MARGIN:
            continue
        votes.add(alignment > 0.5)
    if len(votes) != 1:
        return None
    return votes.pop()


#: Which rule :func:`_source_anchored_verdict` used. Each is counted separately
#: by :func:`_repair_triangle_winding`, because they carry different confidence
#: and consumers warn about some and not others.
_SOURCE_RULE_PROJECTION = "projection"
_SOURCE_RULE_NO_SOURCE = "no_source"
_SOURCE_RULE_UNRESOLVED = "unresolved"
_SOURCE_RULE_BORE_ALIGNMENT = "bore_alignment"
_SOURCE_RULE_VOLUME_FALLBACK = "volume_fallback"


def _source_anchored_verdict(
    points: np.ndarray,
    triangles: np.ndarray,
    tags: np.ndarray,
    *,
    source_tags: set[int],
    symmetry_planes: tuple[str, ...],
) -> tuple[bool | None, str]:
    """Orient one symmetry-reduced component from its tagged source cap.

    ``True`` keep, ``False`` flip, ``None`` no verdict -- together with the
    rule that decided, which the caller counts.

    Positive source-normal projection on the one non-cut principal axis is the
    Metal aperture contract, and it is tried first. A component with no source
    cap at all is left unjudged, because nothing in it says which side the
    fluid is on. A component whose cap cannot be projected is judged by its
    throat collar, and the signed volume is only the last resort.
    """
    projection, projection_reason = _symmetry_source_projection_detail(
        points,
        triangles,
        tags,
        source_tags=source_tags,
        symmetry_planes=symmetry_planes,
    )
    if projection is not None:
        return projection > 0.0, _SOURCE_RULE_PROJECTION
    if projection_reason == _SYMMETRY_PROJECTION_NO_SOURCE:
        # No source cap at all, so nothing establishes which side of this
        # surface the fluid is on. Signed volume answers a different question
        # -- which way is out of the region the cut planes cap -- and the two
        # agree only where the meshed region is the solid. For an acoustic bare
        # shell they are opposed: ``normals.open_shell_bore_alignment`` records
        # that the volume "reports the *wrong* sign for correctly wound rollback
        # profiles", and the consuming add-in pins a reduced component whose
        # walls face the bore, and whose volume is therefore negative, as
        # correct (hornlab-fusion-addin,
        # test_symmetry_reduced_source_anchor_wins_when_signed_volume
        # _disagrees). Orienting that outward inverts every normal.
        #
        # Abstention rather than a guess is the decision this contract was
        # built on: hornlab-fusion-addin commit 3cf7d31 deleted a signed-volume
        # predicate here and replaced it with the source anchor. A component
        # can also reach this branch source-less because its STEP label did not
        # match and the caller passed ``skip_missing_groups`` -- recorded in
        # ``missing_reasons``, so not silent, but arriving here
        # indistinguishable from a body that never had a cap. Count it and let
        # the caller say so.
        return None, _SOURCE_RULE_NO_SOURCE
    # This component DOES carry a source cap, but the projection abstained
    # anyway: the cut planes leave no unique non-cut axis to project onto (a
    # single-plane cut, the common Fusion case, or a three-plane octant), or
    # the cap's net normal along that axis is degenerate.
    #
    # Degeneracy first, and on the volume, exactly as before. A component
    # enclosing a volume indistinguishable from zero is not a body this
    # function can orient by any means, and the collar below will still return
    # a confident fraction for one -- measured: the 10 x 10 x 1e-8 sliver in the
    # tests reads 0.0, not None, because the wall normals of a collapsed box
    # are perfectly well defined even though the box is not. Keeping this guard
    # in front preserves an abstention that predates this change rather than
    # quietly spending it.
    volume = _signed_volume(points, triangles)
    if abs(volume) <= _signed_volume_noise_floor(points, triangles):
        return None, _SOURCE_RULE_UNRESOLVED

    # A cap is still an anchor even when it cannot be projected onto a single
    # axis, so ask the throat collar instead of the volume. The collar measure
    # builds both its references out of the cap itself -- the cap's
    # area-weighted centroid and its net area vector -- so it needs no axis
    # from the cut planes and survives translation, rotation and reduced
    # domains. Where the volume oracle and this one disagree, this one is right
    # for an acoustic component: the volume answers "outward from the region
    # the cut planes cap", which inverts a bore-facing acoustic component, and
    # that inversion was live on exactly this path.
    verdict = _bore_alignment_verdict(
        points,
        triangles,
        tags,
        source_tags=source_tags,
    )
    if verdict is not None:
        return verdict, _SOURCE_RULE_BORE_ALIGNMENT
    # The collar could not judge it -- no wall band, a cancelling cap, or a
    # fraction too close to half to be a reading rather than a verdict. Fall
    # back to the signed volume, already known to clear its noise floor. It
    # remains the wrong question for an acoustic component, so it is counted
    # separately and the caller reports it.
    return volume > 0.0, _SOURCE_RULE_VOLUME_FALLBACK


#: How :func:`_repair_triangle_winding` orients a symmetry-reduced component.
#:
#: ``source-anchor`` (the default) trusts the tagged source cap, which is the
#: contract of the Fusion add-in: its reduced components may enclose FLUID
#: (a bore-facing acoustic shell), and for those the signed volume is inverted.
#:
#: ``mirrored-parent`` orients the component as its mirrored, watertight parent
#: would be oriented by the closed branch: outward from the enclosed region.
#: That is the right contract for a caller whose meshed regions are solids and
#: whose full-domain mesh is oriented by signed volume -- because it is the
#: only one under which the reduced domain and the same body meshed whole
#: always agree. A tagged source facing away from the open axis (the rear face
#: of a throat plug, say) is correct in the full domain and would be inverted,
#: with every other triangle of the component, by the source anchor.
REDUCED_ORIENTATION_SOURCE_ANCHOR = "source-anchor"
REDUCED_ORIENTATION_MIRRORED_PARENT = "mirrored-parent"
REDUCED_ORIENTATIONS = (
    REDUCED_ORIENTATION_SOURCE_ANCHOR,
    REDUCED_ORIENTATION_MIRRORED_PARENT,
)


def _repair_triangle_winding(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    tags: np.ndarray | None = None,
    source_tags: set[int] | None = None,
    symmetry_planes: tuple[str, ...] = (),
    tolerance: float = 0.0,
    reduced_orientation: str = REDUCED_ORIENTATION_SOURCE_ANCHOR,
) -> tuple[np.ndarray, dict[str, int]]:
    """Repair manifold winding and orient components with a valid anchor.

    Watertight components retain the signed-volume outwardness contract.

    A symmetry-reduced open component -- every free edge on a cut plane, so
    its mirrored parent is watertight -- is oriented according to
    ``reduced_orientation`` (see ``REDUCED_ORIENTATIONS``):

    * ``source-anchor``, the default, uses the tagged source cap as
      :func:`_source_anchored_verdict` describes: the source projection on the
      one non-cut axis, then the throat collar, then the signed volume, and no
      verdict at all for a component without a cap.
    * ``mirrored-parent`` uses the parent's signed volume, which is the
      component's own origin-based signed volume (see below), and falls back to
      the source anchor only when that volume is within its noise floor.

    Both verdicts are taken for every reduced component, and a disagreement is
    counted in ``symmetry_source_parent_conflicts`` whichever mode wins. The
    two disagree where the source-anchored rules misread a solid -- a tagged
    source facing away from the open axis, or a throat collar read on the
    outside of a wall -- or where the component encloses fluid rather than
    solid. Which of them is right there depends on what the meshed region is,
    and only the caller knows that. A flip against the other contract is never
    silent.
    """
    if reduced_orientation not in REDUCED_ORIENTATIONS:
        raise ValueError(
            f"reduced_orientation must be one of {REDUCED_ORIENTATIONS}, "
            f"not {reduced_orientation!r}"
        )
    repaired = triangles.copy()
    stats = {
        "flipped_consistency": 0,
        "flipped_global": 0,
        "unjudged_symmetry_components": 0,
        "unjudged_symmetry_no_source": 0,
        "symmetry_volume_fallback_flipped": 0,
        "symmetry_volume_fallback_kept": 0,
        "unresolved_symmetry_components": 0,
        "bore_alignment_flipped": 0,
        "bore_alignment_kept": 0,
        "symmetry_parent_volume_flipped": 0,
        "symmetry_parent_volume_kept": 0,
        "symmetry_source_parent_conflicts": 0,
    }
    if len(repaired) == 0:
        return repaired, stats
    component_tags = (
        np.asarray(tags, dtype=np.int32)
        if tags is not None
        else np.full(len(repaired), RIGID_TAG, dtype=np.int32)
    )
    if len(component_tags) != len(repaired):
        raise ValueError("triangle and physical-tag counts differ")
    declared_source_tags = set(source_tags or ())

    edge_to_triangles: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for tri_idx, tri in enumerate(repaired):
        for start, end in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a = int(start)
            b = int(end)
            if a == b:
                continue
            if a < b:
                edge_to_triangles[(a, b)].append((tri_idx, 1))
            else:
                edge_to_triangles[(b, a)].append((tri_idx, -1))

    neighbours: list[list[tuple[int, bool]]] = [[] for _ in range(len(repaired))]
    for uses in edge_to_triangles.values():
        if len(uses) != 2:
            continue
        (tri_a, dir_a), (tri_b, dir_b) = uses
        must_differ = dir_a == dir_b
        neighbours[tri_a].append((tri_b, must_differ))
        neighbours[tri_b].append((tri_a, must_differ))

    flip = np.zeros(len(repaired), dtype=bool)
    seen = np.zeros(len(repaired), dtype=bool)
    components: list[np.ndarray] = []
    for seed in range(len(repaired)):
        if seen[seed]:
            continue
        seen[seed] = True
        queue: deque[int] = deque([seed])
        component: list[int] = []
        while queue:
            tri_idx = queue.popleft()
            component.append(tri_idx)
            for other, must_differ in neighbours[tri_idx]:
                required = bool(flip[tri_idx]) ^ bool(must_differ)
                if seen[other]:
                    continue
                flip[other] = required
                seen[other] = True
                queue.append(other)
        components.append(np.asarray(component, dtype=np.int64))

    if np.any(flip):
        repaired[flip] = repaired[flip][:, [0, 2, 1]]
        stats["flipped_consistency"] = int(np.count_nonzero(flip))

    for component in components:
        component_triangles = repaired[component]
        edge_stats = _edge_direction_stats(component_triangles)
        closed = (
            edge_stats["boundary_edges"] == 0
            and edge_stats["nonmanifold_edges"] == 0
        )
        symmetry_reduced = (
            edge_stats["boundary_edges"] > 0
            and edge_stats["nonmanifold_edges"] == 0
            and _free_edges_on_expected_planes(
                points,
                component_triangles,
                symmetry_planes=symmetry_planes,
                tolerance=tolerance,
            )
        )
        if edge_stats["inconsistent_edges"] != 0:
            continue
        if closed:
            if _signed_volume(points, component_triangles) < 0.0:
                repaired[component] = component_triangles[:, [0, 2, 1]]
                stats["flipped_global"] += int(len(component))
            continue
        if not symmetry_reduced:
            continue

        source_keep, source_rule = _source_anchored_verdict(
            points,
            component_triangles,
            component_tags[component],
            source_tags=declared_source_tags,
            symmetry_planes=symmetry_planes,
        )
        # What the mirrored parent says. Every free edge of this component lies
        # on a cut plane through the origin, so mirroring closes it: the parent
        # is watertight. The cut-plane caps a closed piece would need add
        # nothing to the origin-based volume sum either, because r . n vanishes
        # on a plane through the origin. This component's own signed volume is
        # therefore exactly 1/2^k of its parent's, sign included -- the verdict
        # the closed branch above gives the same body meshed whole.
        volume = _signed_volume(points, component_triangles)
        parent_keep = (
            None
            if abs(volume) <= _signed_volume_noise_floor(points, component_triangles)
            else volume > 0.0
        )
        if (
            source_keep is not None
            and parent_keep is not None
            and source_keep != parent_keep
        ):
            stats["symmetry_source_parent_conflicts"] += 1

        if (
            reduced_orientation == REDUCED_ORIENTATION_MIRRORED_PARENT
            and parent_keep is not None
        ):
            if parent_keep:
                stats["symmetry_parent_volume_kept"] += 1
            else:
                repaired[component] = component_triangles[:, [0, 2, 1]]
                stats["flipped_global"] += int(len(component))
                stats["symmetry_parent_volume_flipped"] += 1
            continue

        if source_rule != _SOURCE_RULE_PROJECTION:
            stats["unjudged_symmetry_components"] += 1
        if source_rule == _SOURCE_RULE_NO_SOURCE:
            stats["unjudged_symmetry_no_source"] += 1
        elif source_rule == _SOURCE_RULE_UNRESOLVED:
            stats["unresolved_symmetry_components"] += 1
        elif source_rule == _SOURCE_RULE_BORE_ALIGNMENT:
            stats["bore_alignment_kept" if source_keep else "bore_alignment_flipped"] += 1
        elif source_rule == _SOURCE_RULE_VOLUME_FALLBACK:
            stats[
                "symmetry_volume_fallback_kept"
                if source_keep
                else "symmetry_volume_fallback_flipped"
            ] += 1
        if source_keep is False:
            repaired[component] = component_triangles[:, [0, 2, 1]]
            stats["flipped_global"] += int(len(component))

    return repaired, stats


def _normalize_to_positive_side(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    symmetry_planes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Reflect the reduced domain onto the positive side of each cut plane.

    The Metal native symmetry solve requires the reduced mesh on the positive
    side of its symmetry planes. A model cut to a negative quadrant in CAD is
    the mirror image of the equivalent positive-quadrant model, so reflecting
    it (and flipping winding once per reflection to keep normals outward)
    changes nothing about the represented full-domain geometry.
    """
    axis_for_plane = {"x0": 0, "y0": 1, "z0": 2}
    reflected_axes: list[str] = []
    points = points.copy()
    triangles = triangles.copy()
    for plane in symmetry_planes:
        axis = axis_for_plane[plane]
        coords = points[:, axis]
        if -float(coords.min()) > float(coords.max()):
            points[:, axis] = -points[:, axis]
            reflected_axes.append("xyz"[axis])
    if len(reflected_axes) % 2 == 1:
        triangles = triangles[:, [0, 2, 1]]
    normalization = {
        "symmetry_planes": list(symmetry_planes),
        "reflected_axes": reflected_axes,
    }
    return points, triangles, normalization


def postprocess_mesh(
    mesh: meshio.Mesh,
    source_specs: list[StepFaceGroup],
    *,
    symmetry_planes: tuple[str, ...],
    tolerance: float,
    symmetry_snap_tolerance: float | None = None,
    reduced_orientation: str = REDUCED_ORIENTATION_SOURCE_ANCHOR,
    unit_scale_to_m: float = 1.0e-3,
) -> tuple[meshio.Mesh, dict[str, object], dict[str, object]]:
    """Repair and validate a tagged surface mesh without interpreting roles.

    ``tolerance`` and ``symmetry_snap_tolerance`` are in the mesh's own units;
    the fixed near-duplicate weld (``WELD_TOLERANCE_MM``) is converted with
    ``unit_scale_to_m`` (default: the mesh is in millimetres).

    ``reduced_orientation`` selects how a symmetry-reduced component is
    oriented; see ``REDUCED_ORIENTATIONS`` and :func:`_repair_triangle_winding`.
    The mode used is echoed in the repair report.
    """
    points, triangles, tags = _mesh_triangle_data(mesh)
    # A copy: welding reads and the symmetry snap writes these coordinates,
    # and the caller's mesh must come back as it was handed in.
    points = np.array(points, dtype=np.float64, copy=True)
    before_edge_stats = _edge_direction_stats(triangles)
    before_signed_volume = _signed_volume(points, triangles)
    distinct_before = int(len(np.unique(triangles))) if len(triangles) else 0
    triangles = _weld_near_duplicate_vertices(
        points,
        triangles,
        tol_mm=millimetres_to_step_units(WELD_TOLERANCE_MM, unit_scale_to_m),
    )
    welded_vertices = max(0, distinct_before - (int(len(np.unique(triangles))) if len(triangles) else 0))
    triangles, tags, degenerate_removed = _remove_degenerate_triangles(
        points, triangles, tags, min_quality=DEGENERATE_MIN_QUALITY
    )

    symmetry_tolerance = (
        symmetry_snap_tolerance
        if symmetry_snap_tolerance is not None
        else tolerance
    )
    if isinstance(symmetry_planes, str):
        # "auto" guessed the cut from open rims; its only user was the frozen
        # legacy pipeline, which carries its own copy. Callers declare planes.
        raise ValueError(
            "symmetry_planes must be a tuple of plane names such as ('x0', 'y0'); "
            f"{symmetry_planes!r} is not supported (detect_symmetry_planes reports rims)"
        )
    repaired_triangles, repair_stats = _repair_triangle_winding(
        points,
        triangles,
        tags=tags,
        source_tags={spec.tag for spec in source_specs},
        symmetry_planes=symmetry_planes,
        tolerance=symmetry_tolerance,
        reduced_orientation=reduced_orientation,
    )
    repair_stats["welded_vertices"] = int(welded_vertices)
    after_edge_stats = _edge_direction_stats(repaired_triangles)
    if symmetry_snap_tolerance is not None:
        snap_symmetry_plane_vertices(
            points,
            symmetry_planes=symmetry_planes,
            tolerance=symmetry_snap_tolerance,
        )

    points, repaired_triangles, axis_normalization = _normalize_to_positive_side(
        points,
        repaired_triangles,
        symmetry_planes=symmetry_planes,
    )

    # Drop vertices orphaned by welding/degenerate removal so the written node
    # count matches the live mesh the solver assembles.
    points, repaired_triangles = _compact_unused_vertices(points, repaired_triangles)
    after_signed_volume = _signed_volume(points, repaired_triangles)

    repaired_mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", repaired_triangles)],
        cell_data={
            "gmsh:physical": [tags.astype(np.int32, copy=False)],
            "gmsh:geometrical": [tags.astype(np.int32, copy=False)],
        },
        field_data=mesh.field_data,
    )
    topology = _topology_stats(
        points,
        repaired_triangles,
        symmetry_planes=symmetry_planes,
        tolerance=tolerance,
    )
    topology["axis_normalization"] = axis_normalization
    topology["signed_volume_step_units3"] = after_signed_volume
    topology["source_normal_projections"] = _source_normal_projections(
        points,
        repaired_triangles,
        tags,
        source_specs,
    )
    repair = {
        "degenerate_triangles_removed": int(degenerate_removed),
        "reduced_orientation": reduced_orientation,
        **repair_stats,
        "before": {
            **before_edge_stats,
            "signed_volume_step_units3": before_signed_volume,
        },
        "after": {
            **after_edge_stats,
            "signed_volume_step_units3": after_signed_volume,
        },
    }
    return repaired_mesh, repair, topology
