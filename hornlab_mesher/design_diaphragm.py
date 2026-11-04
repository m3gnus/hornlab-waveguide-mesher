"""Opt-in sketched HF source on the design builder's own WG0 throat edge."""
from __future__ import annotations

import math
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np

from .enclosure_woofers import (
    NAMESPACE, SketchedShape, _certify_occ_partition, _fragment_with_glue,
    _remap_groups, validate_aperture_contour,
)
from .errors import control_refusal
from .phase_plug import PhasePlug
from .source_contour_occ import revolve_contour, fit_contour_rim


FEATURE = "design-sketched-diaphragms-v1"


def _refuse(detail, fix):
    raise control_refusal(detail, "Source attachment", fix, message=detail + ".")


def attach_hf_diaphragm(built, geometry):
    """Replace only the classic source faces; the design builder owns the horn."""
    import gmsh

    payload = geometry.hf_diaphragm
    shape = SketchedShape.from_document(payload["diaphragm"])
    contour = shape.diaphragm
    if not geometry.closed:
        _refuse("A sketched HF diaphragm requires the full model", "Choose Full model in Symmetry.")
    if built.open_shell_wall_points_mm is not None:
        _refuse("A sketched HF diaphragm needs a closed horn body",
                "Choose Thickened waveguide (freestanding) or Waveguide in enclosure in Mounting.")
    if set(built.surface_groups) - {1, 2, 3, *built.metadata.get("source_tags", {}).values()}:
        _refuse("The HF sketch cannot include subdomain interfaces", "Remove the subdomain interfaces, or revert the HF sketch to Classic.")
    ring = geometry.inner_points[:, 0, :]
    radius = float(np.linalg.norm(ring[:, :2], axis=1).mean())
    if (np.ptp(ring[:, 2]) > 1e-7 or
            np.max(abs(np.linalg.norm(ring[:, :2], axis=1) - radius)) > 1e-7):
        _refuse("The HF sketch needs a circular planar throat", "Use a circular planar throat before attaching the HF sketch.")
    validate_aperture_contour(contour, radius)
    occ = gmsh.model.occ
    occ.synchronize()
    original = set(built.surface_groups[2])
    if not original.issubset({t for _, t in occ.getEntities(2)}):
        _refuse("This horn cannot attach an HF sketch", "Choose an acoustic surface fit for this horn.")
    from .builders._occ import extreme_boundary_loop_curves
    rim_curves = extreme_boundary_loop_curves([(2, t) for t in built.mesh_surface_groups["inner"]],
                                              source_axis="z", use_min=True)
    fitted_rim = any(gmsh.model.getType(1, abs(t)) != "Circle" for t in rim_curves)
    if not rim_curves:
        _refuse("The horn has no attachable WG0 edge", "Revert the HF sketch to Classic.")
    if fitted_rim and not payload.get("fit_wg0_rim", False):
        _refuse("The fitted WG0 rim cannot join an exact revolved HF contour",
                "Revert the HF sketch to Classic pending qualification of the fitted diaphragm attachment.")
    center = (0.0, 0.0, float(ring[0, 2]))
    if fitted_rim:
        faces = fit_contour_rim(occ, contour, rim_curves, center)
    else:
        _, exact = revolve_contour(occ, contour, center)
        faces = {segment: [face] for segment, face in exact.items()}
    occ.remove([(2, tag) for tag in original], recursive=False)
    for groups in (built.surface_groups, built.mesh_surface_groups):
        for key, values in groups.items():
            groups[key] = sorted(set(values) - original)
    assembly = built.metadata.setdefault("assembly", {
        "patches": [], "contours": {}, "half_x": False,
        "coordinate_frame": "horn-throat-pre-placement-mm",
        "vertical_offset_mm": geometry.vertical_offset_mm,
    })
    assembly["contours"]["horn-throat"] = contour.to_dict()
    sources = built.metadata.setdefault("source_tags", {})
    sources.pop("horn-throat", None)
    tag_map = built.metadata.setdefault("tag_map", {
        "1": {"source_id": None, "instance_id": None, "role": "rigid"},
    })
    tag_map.pop("2", None)
    names = built.metadata.setdefault("assemblyPhysicalNames", {1: NAMESPACE + "|rigid"})
    names.pop(2, None)
    used = set(built.surface_groups) | set(sources.values())
    weights = dict(shape.drive.weights)
    for segment in contour.segments:
        moving = segment.role == "moving"
        physical = (2 if 2 not in sources.values() else max(used | {100}) + 1) if moving else 1
        used.add(physical)
        patch_id = "horn-throat/" + quote(segment.id, safe="")
        if moving:
            sources[patch_id] = physical
            tag_map[str(physical)] = {"source_id": patch_id, "instance_id": None, "role": "primary"}
            names[physical] = f"{NAMESPACE}|tag={physical}|source_id={patch_id}|instance_id=null|role=primary"
        segment_faces = faces[segment.id]
        built.surface_groups.setdefault(physical, []).extend(segment_faces)
        role = "woofer:" + patch_id  # The existing per-patch density owner.
        built.mesh_surface_groups[role] = segment_faces
        assembly["patches"].append({
            "patch_id": patch_id, "radiator_id": "horn-throat", "segment_id": segment.id,
            "role": "primary" if moving else "rigid", "physical_tag": physical,
            "surface_tags": segment_faces, "center_mm": list(center), "radius_mm": radius,
            "recess_mm": 0, "resolution_mm": 2.0,
            "density_role": role, "weight": weights.get(segment.id, 0.0), "motion": shape.drive.motion,
            **({"surface_tolerance_mm": .025} if fitted_rim and segment == contour.segments[-1] else {}),
        })
    occ.synchronize()
    # The exact OCC intersection must leave each diaphragm and wall face whole.
    # A crossing away from the common WG0 rim splits one of those faces.
    before = gmsh.model.getEntities(2)
    _, mapping = _fragment_with_glue(gmsh, before, [])
    remap = {old: [t for d, t in new if d == 2] for (_, old), new in zip(before, mapping)}
    if any(len(new) != 1 for new in remap.values()):
        _refuse("The HF diaphragm intersects the horn or enclosure", "Reduce the diaphragm depth or height until it clears the horn wall.")
    _remap_groups(built.surface_groups, remap)
    _remap_groups(built.mesh_surface_groups, remap)
    for patch in assembly["patches"]:
        patch["surface_tags"] = sorted({t for old in patch["surface_tags"] for t in remap[old]})
    occ.synchronize()
    plugs = tuple(PhasePlug(**p) for p in payload.get("phase_plugs", []))
    if len(plugs) > 8 or len({p.id for p in plugs}) != len(plugs):
        _refuse("Phase plugs need distinct identities and at most eight bodies", "Remove extra bodies or give each a distinct identity.")
    if plugs:
        if getattr(built, "freeform_wall_recipe", None) is not None:
            _refuse("Phase plugs are not qualified for FREEFORM profiles with authored corners",
                    "Remove the phase plugs, or use a smooth circular profile without corner anchors.")
        # Rational Bezier/B-spline meridians stay in their pole convex hulls.
        # The adapter builder revolves these same meridians: this radial bound
        # certifies separation without slow OCC surface-distance queries.
        adapter = getattr(geometry, "adapter_meridian", None)
        if adapter is None:
            from .general_horn import GeneralHornWall
            wall = GeneralHornWall.from_geometry(geometry, "OSSE")

            class Clearance:
                poles_mm = wall.poles_mm

                def minimum_body_clearance(self, plug):
                    return wall.minimum_body_clearance(plug) - wall.circle_correction_bound_mm
        else:
            min_radius = min(np.asarray(adapter.cubic)[:, 1].min(),
                             np.asarray(adapter.body_poles)[:, 1].min()) * geometry.adapter_scale

            class Clearance:
                poles_mm = [(min_radius, 0)]

                def minimum_body_clearance(self, plug):
                    return min_radius - max(plug.outer0_mm, plug.outer1_mm)

        from .phase_plug import validate_passages
        context = SimpleNamespace(phase_plugs=plugs,
                          horn=SimpleNamespace(points=contour.points, axial_bounds_mm=(contour.axial_bounds_mm[0],
                                               contour.axial_bounds_mm[1] + (.025 if fitted_rim else 0))),
                          horn_wall=Clearance(),
                          horn_length_mm=float(geometry.inner_points[:, -1, 2].min() - center[2]),
                          mouth_radius_mm=float(np.linalg.norm(geometry.inner_points[:, -1, :2], axis=1).min()))
        clearances = validate_passages(context)
        passage_surfaces = {"horn-wall": list(built.mesh_surface_groups["inner"])}
        assembly["passage"] = {
            "recipe": {"plugs": [p.to_dict() for p in plugs], "origin_mm": list(center),
                       "throat_radius_mm": radius, "source_z_max_mm": context.horn.axial_bounds_mm[1],
                       "length_mm": context.horn_length_mm, "mouth_radius_mm": context.mouth_radius_mm,
                       "wall_poles_mm": list(Clearance.poles_mm),
                       "wall_clearance_mm": clearances["horn-wall"]},
            "surfaces": passage_surfaces,
        }
    passive_faces = {}
    for plug in plugs:
        # Reuse the canonical passive body's meridian edges, independently of
        # the retired source-assembly builder and its enclosure.
        surfaces = []
        for role, (a, b) in plug.edges.items():
            pa = occ.addPoint(a[0], 0, center[2] + a[1])
            pb = occ.addPoint(b[0], 0, center[2] + b[1])
            edge = occ.addLine(pa, pb)
            faces = [t for d, t in occ.revolve([(1, edge)], 0, 0, center[2], 0, 0, 1, 2 * math.pi) if d == 2]
            surfaces.extend(faces)
            passage_surfaces[role] = faces
        occ.synchronize()
        passive_faces[plug.id] = surfaces
        built.surface_groups[1].extend(surfaces)
    if plugs:
        occ.synchronize()
        before = gmsh.model.getEntities(2)
        _, mapping = _fragment_with_glue(gmsh, before, [])
        remap = {old: [t for d, t in new if d == 2] for (_, old), new in zip(before, mapping)}
        _remap_groups(built.surface_groups, remap)
        _remap_groups(built.mesh_surface_groups, remap)
        _remap_groups(passive_faces, remap)
        _remap_groups(passage_surfaces, remap)
        for patch in assembly["patches"]:
            patch["surface_tags"] = sorted({t for old in patch["surface_tags"] for t in remap[old]})
    assembly["phase_plugs"] = [p.to_dict() for p in plugs]
    assembly["passive_body_surfaces"] = passive_faces
    built.surface_groups[1] = sorted(set(built.surface_groups[1] + built.surface_groups.pop(3, [])))
    assembly["face_tags"] = {str(k): v for k, v in built.surface_groups.items() if v}
    built.surface_groups = {k: v for k, v in built.surface_groups.items() if v}
    built.metadata.update(tag_namespace=NAMESPACE, rigid_tag=1)
    occ.synchronize()
    _certify_occ_partition(built, gmsh)
