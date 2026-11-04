"""Passage sizing and admission for the design builder's existing CAD wall."""
from types import SimpleNamespace

import numpy as np

from .errors import InternalInvariantError, control_refusal
from .passage_mesh import PassageToleranceError, certify_passage_mesh, facet_bound
from .phase_plug import PhasePlug, surface_targets, validate_passages


def passage_context(recipe):
    """Adapt the design's clearance proof to the shared passage contract."""
    return SimpleNamespace(
        phase_plugs=tuple(PhasePlug(**p) for p in recipe["plugs"]),
        horn=SimpleNamespace(points=[SimpleNamespace(r_mm=recipe["throat_radius_mm"])],
                             axial_bounds_mm=(0, recipe["source_z_max_mm"])),
        horn_wall=SimpleNamespace(poles_mm=recipe["wall_poles_mm"],
                                 minimum_body_clearance=lambda _: recipe["wall_clearance_mm"]),
        horn_length_mm=recipe["length_mm"], mouth_radius_mm=recipe["mouth_radius_mm"],
        rigid_areas=lambda: {}, parts=[(None, recipe["origin_mm"])],
    )


def configure_passage_density(built, add_field):
    """Feed shared clearance targets into the ordinary design density fields."""
    passage = built.metadata.get("assembly", {}).get("passage")
    if passage is None:
        return []
    context = passage_context(passage["recipe"])
    targets = surface_targets(context, float("inf"))
    gap = min(validate_passages(context).values())
    z0 = context.phase_plugs[0].z0_mm + passage["recipe"]["origin_mm"][2] - gap
    z1 = context.phase_plugs[0].z1_mm + passage["recipe"]["origin_mm"][2] + gap
    from .density import _collect_boundary_curves

    for role, faces in passage["surfaces"].items():
        target = targets[role]
        # Only the facing wall band needs this density. Grade back to the
        # user's ordinary horn density outside it; remote cabinet faces stay coarse.
        formula = (f"{target:.12g} + 0.5*max(0, {z0:.12g}-z, z-{z1:.12g})"
                   if role == "horn-wall" else f"{target:.12g}")
        add_field(formula, faces, _collect_boundary_curves(faces))
    passage["targets_mm"] = targets
    return list(targets.values())


def certify_design_passages(points, triangles, physical, surfaces, assembly):
    """Certify plug components and whole facing wall facets before placement."""
    import gmsh
    from scipy.spatial import cKDTree

    passage = assembly["passage"]
    recipe = passage["recipe"]
    context = passage_context(recipe)
    gap = min(validate_passages(context).values())
    origin_z = recipe["origin_mm"][2]
    z0, z1 = context.phase_plugs[0].z0_mm + origin_z, context.phase_plugs[0].z1_mm + origin_z
    xyz = points[triangles]

    def enclosure_bound(component, tolerance):
        bound = 0.
        facing_count = 0
        for face in passage["surfaces"]["horn-wall"]:
            selected = (component & (surfaces == face) & (xyz[..., 2].max(axis=1) >= z0 - gap)
                        & (xyz[..., 2].min(axis=1) <= z1 + gap))
            if not selected.any():
                continue
            facing_count += int(selected.sum())
            low, high = map(np.asarray, gmsh.model.getParametrizationBounds(2, face))
            facets = xyz[selected]
            vertices = np.unique(facets.reshape(-1, 3), axis=0)
            parameters = np.asarray(gmsh.model.getParametrization(2, face, vertices.ravel())).reshape(-1, 2)
            tree = cKDTree(vertices)

            def sample_distance(batch, grid, samples):
                _, indices = tree.query(batch)
                uv = parameters[indices]
                uv = (uv[:, 0, None] + grid[None, :, :1] * (uv[:, 1, None] - uv[:, 0, None])
                      + grid[None, :, 1:] * (uv[:, 2, None] - uv[:, 0, None]))
                uv = np.clip(uv, low, high)
                flat = samples.reshape(-1, 3)
                # Attachment requires these horn faces to remain unsplit.
                # Their rectangular parameter domains are the actual finite CAD
                # patches. Any evaluated point gives a distance upper bound;
                # facet_bound adds its independent Lipschitz cover for interiors.
                on_wall = np.asarray(gmsh.model.getValue(2, face, uv.ravel())).reshape(-1, 3)
                return np.linalg.norm(flat - on_wall, axis=1).reshape(samples.shape[:-1])

            bound = max(bound, facet_bound(facets, None, tolerance, sample_distance=sample_distance))
        if facing_count == 0:
            raise InternalInvariantError("passage certification requires facing horn wall facets")
        # Convex triangle interpolation cannot cross either axial end bound.
        hf_faces = [face for patch in assembly["patches"] if patch["radiator_id"] == "horn-throat"
                    for face in patch["surface_tags"]]
        hf = np.isin(surfaces, hf_faces)
        if not hf.any() or xyz[hf, :, 2].max() > origin_z + recipe["source_z_max_mm"] + 1e-7:
            raise InternalInvariantError("HF mesh exceeds its certified source clearance plane")
        return bound

    active_tags = [patch["physical_tag"] for patch in assembly["patches"] if patch["role"] != "rigid"]
    try:
        passage["quality"] = certify_passage_mesh(
            context, points, triangles, np.isin(physical, active_tags), enclosure_bound=enclosure_bound)
    except PassageToleranceError as exc:
        raise control_refusal(str(exc), "Phase plug and annular vanes",
                              "Widen the passage or use finer throat and mouth resolutions, then rebuild.",
                              message="This mesh could not preserve the phase-plug passage clearance.") from exc
