"""Admission of every passive component and both sides of narrow passages."""

import math

import numpy as np

from .phase_plug import line_distance, validate_passages


def annulus_bounds(facets, center, z, inner, outer):
    """Conservative exact flat-annulus error for whole triangles.

    Norm is convex: its maximum is at a vertex; its minimum is the closest
    point on the triangle, including its interior. No vertex-only hole test.
    """
    xy = facets[..., :2] - np.asarray(center)
    a, b = xy, np.roll(xy, -1, axis=1)
    d = b - a
    denominator = np.einsum("ijk,ijk->ij", d, d)
    t = np.divide(
        -np.einsum("ijk,ijk->ij", a, d),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    closest = a + np.clip(t, 0, 1)[..., None] * d
    minimum = np.linalg.norm(closest, axis=2).min(axis=1)
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    inside = (cross >= 0).all(axis=1) | (cross <= 0).all(axis=1)
    minimum = np.where(inside, 0, minimum)
    maximum = np.linalg.norm(xy, axis=2).max(axis=1)
    radial = np.maximum.reduce(
        [inner - minimum, maximum - outer, np.zeros(len(facets))]
    )
    return np.hypot(np.abs(facets[..., 2] - z).max(axis=1), radial)


def enclosure_planar_facets(assembly, facets):
    """Prove box-plane facets against convex rectangles; holes stay separate."""
    w, h, z, depth = (
        assembly.width_mm / 2,
        assembly.height_mm / 2,
        assembly.front_z_mm,
        assembly.depth_mm,
    )
    bounds = [(-w, w), (-h, h), (z - depth, z)]
    mask = np.zeros(len(facets), dtype=bool)
    for axis, plane in ((0, -w), (0, w), (1, -h), (1, h), (2, z - depth), (2, z)):
        covered = np.abs(facets[..., axis] - plane).max(axis=1) <= 1e-9
        for k, (low, high) in enumerate(bounds):
            if k != axis:
                covered &= (facets[..., k].min(axis=1) >= low - 1e-9) & (
                    facets[..., k].max(axis=1) <= high + 1e-9
                )
        if axis == 2 and plane == z:
            for center, radius in (
                (assembly.horn_xy_mm, assembly.mouth_radius_mm),
                (assembly.woofer_xy_mm, assembly.aperture_radius_mm),
            ):
                covered &= annulus_bounds(facets, center, z, radius, math.inf) <= 1e-9
        mask |= covered
    return mask


def facet_bound(facets, distance, tolerance):
    """Whole-facet bound using exact finite surfaces and a Lipschitz cover."""
    if not len(facets):
        raise ValueError("passage surface is empty")
    worst = 0.0
    for n in (8, 32, 128, 256):
        uv = np.asarray(
            [(a / n, b / n) for a in range(n + 1) for b in range(n + 1 - a)]
        )
        pending = []
        for batch in np.array_split(facets, max(1, len(facets) // 16 + 1)):
            samples = (
                batch[:, 0, None]
                + uv[None, :, :1] * (batch[:, 1, None] - batch[:, 0, None])
                + uv[None, :, 1:] * (batch[:, 2, None] - batch[:, 0, None])
            )
            sampled = distance(samples).max(axis=1)
            if np.any(sampled > tolerance):
                raise ValueError("passage mesh exceeds its fixed geometric tolerance")
            diameter = np.linalg.norm(batch - np.roll(batch, 1, axis=1), axis=2).max(
                axis=1
            )
            bounds = sampled + diameter / n
            good = bounds <= tolerance
            if good.any():
                worst = max(worst, float(bounds[good].max()))
            pending.extend(batch[~good])
        if not pending:
            return worst
        facets = np.asarray(pending)
    raise ValueError("passage whole-facet tolerance cannot be certified")


def certify_passage_mesh(assembly, points_mm, triangles, source_tags):
    """Check closed components, genus, orientation and finite-surface error.

    source_tags is one boolean per facet identifying moving facets. Passive
    components cannot contain them. Bounds are tied to each body independently,
    so the union of nearby surfaces cannot conceal a lost or inverted vane.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    points_mm, triangles = np.asarray(points_mm), np.asarray(triangles)
    if not assembly.phase_plugs:
        raise ValueError("passage certification requires passive bodies")
    if not np.isfinite(points_mm).all():
        raise ValueError("passage mesh vertices must be finite")
    xyz = points_mm[triangles]
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    unique, inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    winding = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    if np.any(counts != 2) or np.any(winding != 0):
        raise ValueError("passage components must be closed and consistently wound")
    if np.any(
        np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        <= 1e-12
    ):
        raise ValueError("passage mesh contains degenerate facets")
    graph = coo_matrix(
        (np.ones(len(unique)), (unique[:, 0], unique[:, 1])),
        shape=(len(points_mm), len(points_mm)),
    )
    _, vertex_labels = connected_components(graph, directed=False)
    labels = vertex_labels[triangles[:, 0]]
    actual = np.unique(labels)
    if len(actual) != 1 + len(assembly.phase_plugs):
        raise ValueError("passage mesh component inventory contradicts recipe")
    active_labels = np.unique(labels[np.asarray(source_tags, dtype=bool)])
    if len(active_labels) != 1:
        raise ValueError("all moving sources must belong to the enclosure component")
    origin = assembly.parts[0][1]
    remaining = list(assembly.phase_plugs)
    clearances = validate_passages(assembly)
    gap = min(clearances.values())
    tolerance = min(0.15, gap / 10)
    components, maximum = [], 0.0
    for label in actual:
        selected = labels == label
        tris, facets = triangles[selected], xyz[selected]
        vertices = np.unique(tris)
        edge_count = len(
            np.unique(
                np.sort(
                    np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]),
                    axis=1,
                ),
                axis=0,
            )
        )
        euler = len(vertices) - edge_count + len(tris)
        # Subtract a nearby origin to avoid cancellation at displaced placements.
        local = facets - np.asarray(origin)
        volume = float(
            np.einsum("ij,ij->i", local[:, 0], np.cross(local[:, 1], local[:, 2])).sum()
            / 6
        )
        if volume <= 1e-10:
            raise ValueError(
                "every passage component must have outward orientation and positive volume"
            )
        if label == active_labels[0]:
            identifier, expected_euler = "enclosure", 2
            rigid_fns = [
                lambda p, role=role: assembly.rigid_distance(role, p)
                for role in assembly.rigid_areas()
                if not role.startswith("plug/")
            ]
            rigid_fns += [
                lambda p, key=key: assembly.patch_distance(key, p)
                for key, c, i, _, _ in assembly.patches
                if c.segments[i].role == "rigid"
            ]
            rigid_facets = xyz[selected & ~np.asarray(source_tags, dtype=bool)]
            # Convex rectangles admit a direct whole-facet proof. Reserve the
            # adaptive finite-surface certificate for holes and curved walls.
            rigid_facets = rigid_facets[
                ~enclosure_planar_facets(assembly, rigid_facets)
            ]
            bound = facet_bound(
                rigid_facets,
                lambda p, fns=rigid_fns: np.minimum.reduce([fn(p) for fn in fns]),
                tolerance,
            )
            for key, c, i, _, _ in assembly.patches:
                if c.segments[i].role != "moving":
                    continue
                # Identify the patch by its exact finite surface, then certify;
                # native/consumer admission separately binds moving tag selectors.
                candidates = facets[
                    np.max(assembly.patch_distance(key, facets), axis=1) <= 1e-6
                ]
                a, b = c.points[i : i + 2]
                origin_patch = next(p[3] for p in assembly.patches if p[0] == key)
                if c.segments[i].kind == "line" and a.z_mm == b.z_mm:
                    patch_bound = float(
                        annulus_bounds(
                            candidates,
                            origin_patch[:2],
                            origin_patch[2] + a.z_mm,
                            a.r_mm,
                            b.r_mm,
                        ).max()
                    )
                    if patch_bound > tolerance:
                        raise ValueError("moving annulus exceeds passage tolerance")
                else:
                    try:
                        patch_bound = facet_bound(
                            candidates,
                            lambda p, key=key: assembly.patch_distance(key, p),
                            tolerance,
                        )
                    except ValueError as exc:
                        raise ValueError(f"passage moving patch {key}: {exc}") from exc
                bound = max(bound, patch_bound)
        else:
            matches = []
            for body in remaining:
                fns = [
                    lambda p, a=a, b=b: line_distance(a, b, p, origin)
                    for a, b in body.edges.values()
                ]
                if (
                    np.max(np.minimum.reduce([fn(points_mm[vertices]) for fn in fns]))
                    <= 1e-6
                ):
                    matches.append((body, fns))
            if len(matches) != 1:
                raise ValueError(
                    "passage component cannot be uniquely bound to a passive body"
                )
            body, fns = matches[0]
            remaining.remove(body)
            identifier, expected_euler = body.id, 0 if body.inner0_mm else 2
            bound = facet_bound(
                facets,
                lambda p, fns=fns: np.minimum.reduce([fn(p) for fn in fns]),
                tolerance,
            )
            expected_volume = (
                math.pi
                * (body.z1_mm - body.z0_mm)
                / 3
                * (
                    body.outer0_mm**2
                    + body.outer0_mm * body.outer1_mm
                    + body.outer1_mm**2
                    - body.inner0_mm**2
                    - body.inner0_mm * body.inner1_mm
                    - body.inner1_mm**2
                )
            )
            # Whole-facet approximation gives a shell error; retain measured
            # volume rather than impose a mesh-independent relative threshold.
        if euler != expected_euler:
            raise ValueError("passage component genus contradicts canonical topology")
        maximum = max(maximum, bound)
        components.append(
            {
                "id": identifier,
                "euler": euler,
                "volume_mm3": volume,
                "facet_bound_mm": bound,
            }
        )
        if identifier != "enclosure":
            components[-1]["canonical_volume_mm3"] = expected_volume
    if remaining or gap - 2 * maximum <= 0:
        raise ValueError(
            "passage clearance cannot survive both facing approximation errors"
        )
    return {
        "version": 1,
        "components": components,
        "surface_tolerance_mm": tolerance,
        "minimum_clearance_mm": gap,
        "certified_clearance_mm": gap - 2 * maximum,
        "maximum_facet_bound_mm": maximum,
    }
