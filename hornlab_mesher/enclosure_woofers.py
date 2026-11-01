"""Resolved design-enclosure radiators and their OCC/mesh admission contract.

Coordinates are finished, pre-placement millimetres in the horn throat frame.
The ordinary enclosure builder remains the only owner of the box and baffle.
"""
from __future__ import annotations

from .errors import UserRefusal, control_refusal

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any
from dataclasses import dataclass, replace
import math
from typing import Literal

import numpy as np

from .source_body import _number
from .source_contour import SourceContour, cone

NAMESPACE = "wg-assembly-v1"
CLEARANCE_MM = 0.1
ATH_JOIN_RADIUS = 0.35
ATH_JOIN_DEPTH = 0.4
ATH_SPHERE_RADIUS = 0.525


@dataclass(frozen=True)
class PistonShape:
    recess_mm: float = 0.0
    kind: Literal["piston"] = "piston"

    def __post_init__(self):
        if self.kind != "piston":
            raise ValueError("invalid piston kind")
        _number(self.recess_mm, "woofer recess")
        if self.recess_mm < 0:
            raise control_refusal("woofer recess must be nonnegative", 'Recess', 'Enter a nonnegative Recess in Sources and baffle.', message='Woofer recess is negative.')
        if 0 < self.recess_mm < .01:
            raise control_refusal("woofer recess must be zero or at least 0.01 mm (vertex weld tolerance)", 'Recess', 'Set Recess to zero or at least 0.01 mm in Sources and baffle.', message='Woofer recess is too small to form a distinct surface.')


@dataclass(frozen=True)
class AthConeShape:
    kind: Literal["ath_cone"] = "ath_cone"

    def __post_init__(self):
        if self.kind != "ath_cone":
            raise ValueError("invalid ATH cone kind")


@dataclass(frozen=True)
class EnclosureWoofer:
    radiator_id: str
    x_mm: float
    y_mm: float
    radius_mm: float
    shape: PistonShape | AthConeShape
    resolution_mm: float | None = None
    active: bool = True

    def __post_init__(self):
        if (not isinstance(self.radiator_id, str) or not self.radiator_id.strip()
                or self.radiator_id.strip() != self.radiator_id
                or any(c in self.radiator_id for c in '/|"\\\n\r')
                or not self.radiator_id.isprintable() or self.radiator_id == "horn-throat"):
            raise ValueError("woofer radiator_id must be a nonempty trimmed identity without '/' and not horn-throat")
        for name in ("x_mm", "y_mm", "radius_mm"):
            _number(getattr(self, name), "woofer " + name)
        if self.radius_mm <= 0:
            raise control_refusal("woofer radius_mm must be positive", 'Radius', 'Enter a positive Radius in Sources and baffle.', message='Woofer radius must be positive.')
        if not isinstance(self.shape, (PistonShape, AthConeShape)):
            raise ValueError("woofer shape must be piston or ath_cone")
        if type(self.active) is not bool:
            raise ValueError("woofer active must be boolean")
        if self.resolution_mm is not None:
            _number(self.resolution_mm, "woofer resolution_mm")
            if self.resolution_mm <= 0:
                raise control_refusal("woofer resolution_mm must be positive", 'Resolution', 'Enter a positive Resolution in Sources and baffle.', message='Woofer resolution must be positive.')


def resolve_woofer_placements(
    woofers: Sequence[Mapping[str, Any]], *, mouth_bottom_mm: float, mouth_top_mm: float
) -> list[dict[str, Any]]:
    """Resolve in the horn-throat pre-placement frame, independent of list order.

    A spacing relation uses the ray from its resolved target towards this
    woofer's anchor position. Coincident anchors have no defined direction.
    Disabled targets still own placement; only enabled woofers are emitted.
    """
    by_id = {w["radiator_id"]: w for w in woofers}
    if len(by_id) != len(woofers):
        raise ValueError("duplicate radiator identity")
    resolved: dict[str, tuple[float, float]] = {}
    visiting: set[str] = set()

    def position(key: str) -> tuple[float, float]:
        if key in resolved:
            return resolved[key]
        if key in visiting:
            raise control_refusal("cyclic spacing_to relation", 'Spacing to', 'Clear Spacing to on one woofer in the loop.', message='The woofer spacing references form a loop.')
        if key not in by_id:
            raise control_refusal("spacing_to target is missing", 'Spacing to', 'Choose a current woofer in Spacing to, or clear it.', message='The referenced woofer no longer exists.')
        visiting.add(key)
        w = by_id[key]
        p = w["placement"]
        if p["anchor"] == "free":
            x, y = p["x_mm"], p["y_mm"]
        elif p["anchor"] == "below":
            x, y = 0.0, mouth_bottom_mm - p["gap_mm"] - w["radius_mm"]
        elif p["anchor"] == "above":
            x, y = 0.0, mouth_top_mm + p["gap_mm"] + w["radius_mm"]
        else:
            raise ValueError("unknown woofer anchor")
        if spacing := p.get("spacing_to"):
            tx, ty = position(spacing["radiator_id"])
            dx, dy = x - tx, y - ty
            length = math.hypot(dx, dy)
            if length == 0:
                raise control_refusal("spacing_to requires a nonzero direction from target", 'Spacing to', 'Move one woofer centre before setting Spacing to.', message='The two woofer centres coincide, so their spacing direction is undefined.')
            distance = spacing["distance_mm"]
            if not math.isfinite(distance) or distance <= 0:
                raise control_refusal("spacing_to distance must be finite and positive", 'Spacing distance', 'Enter a positive Spacing distance in Sources and baffle.', message='Woofer spacing distance must be positive.')
            x, y = tx + distance * dx / length, ty + distance * dy / length
        if not all(math.isfinite(v) for v in (x, y)):
            raise control_refusal("placement must be finite", 'Woofer position', 'Enter finite Horizontal and Vertical positions in Sources and baffle.', message='A woofer position is not finite.')
        visiting.remove(key)
        resolved[key] = (x, y)
        return x, y

    for key, woofer in by_id.items():
        if woofer["enabled"]:
            position(key)
    return [
        dict(
            radiator_id=w["radiator_id"],
            x_mm=resolved[w["radiator_id"]][0],
            y_mm=resolved[w["radiator_id"]][1],
            radius_mm=w["radius_mm"],
            shape=dict(w["shape"]),
            active=w["drive"]["kind"] != "rigid",
            **({"resolution_mm": w["resolution_mm"]} if w.get("resolution_mm") else {}),
        )
        for w in woofers
        if w["enabled"]
    ]



def mouth_y_edges(inner_points):
    """Final cardinal samples, including the lip; never spline overshoot."""
    ring = np.asarray(inner_points)[:, -1, :]
    lower, upper = ring[ring[:, 1] < 0], ring[ring[:, 1] > 0]
    return (float(lower[np.argmin(abs(lower[:, 0])), 1]),
            float(upper[np.argmin(abs(upper[:, 0])), 1]))


def woofer_from_config(row):
    try:
        values = dict(row)
        shape = dict(values.pop("shape"))
        kind = shape.pop("kind")
        if kind == "piston":
            resolved = PistonShape(**shape)
        elif kind == "ath_cone":
            resolved = AthConeShape(**shape)
        else:
            raise ValueError("woofer shape kind must be piston or ath_cone")
        return EnclosureWoofer(**values, shape=resolved)
    except (KeyError, TypeError) as exc:
        if isinstance(exc, UserRefusal):
            raise
        raise ValueError(f"invalid woofer configuration: {exc}") from exc


def ath_contour(w):
    """ATH's scale-invariant meridian; sphere R=.525r, centre=-.791311896r."""
    r = w.radius_mm
    cap_height = (ATH_SPHERE_RADIUS - math.sqrt(ATH_SPHERE_RADIUS**2 - ATH_JOIN_RADIUS**2))*r
    contour = cone(r, ATH_JOIN_DEPTH*r, cap_radius_mm=ATH_JOIN_RADIUS*r,
                   cap_height_mm=cap_height, physical_source_id=w.radiator_id, rim_id="rim")
    # Preserve the assembly's existing public patch identity.
    return replace(contour, segments=(replace(contour.segments[0], id="cap"), contour.segments[1]))


def _on_x_cut(gmsh, dim, tag):
    lo, hi = gmsh.model.getParametrizationBounds(dim, tag)
    values = gmsh.model.getValue(dim, tag, np.linspace(lo[0], hi[0], 5).tolist())
    return np.max(abs(np.asarray(values).reshape(-1, 3)[:, 0])) < 1e-7


def _disc(occ, w, z, half):
    if not half:
        return occ.addDisk(w.x_mm, w.y_mm, z, w.radius_mm, w.radius_mm)
    a = occ.addPoint(0, w.y_mm-w.radius_mm, z)
    b = occ.addPoint(0, w.y_mm+w.radius_mm, z)
    # A point on the +X branch selects the intended semicircle.
    mid = occ.addPoint(w.radius_mm, w.y_mm, z)
    arc = occ.addCircleArc(a, mid, b, center=False)
    return occ.addPlaneSurface([occ.addCurveLoop([arc, occ.addLine(b, a)])])


def _remap_groups(groups, remap):
    for key, tags in groups.items():
        groups[key] = sorted({new for old in tags for new in remap.get(old, [old])})


def _certify_occ_partition(built, gmsh):
    counts = Counter(t for tags in built.surface_groups.values() for t in tags)
    faces = {t for _, t in gmsh.model.getEntities(2)}
    if set(counts) != faces or any(n != 1 for n in counts.values()):
        raise ValueError("assembly faces must be classified exactly once")
    for patch in built.metadata["assembly"]["patches"]:
        area = sum(gmsh.model.occ.getMass(2, t) for t in patch["surface_tags"])
        if not math.isfinite(area) or area <= 0:
            raise ValueError(f"assembly patch {patch['patch_id']} has no positive area")
        patch["area_mm2"] = area


def _symmetric_enclosure_x(enclosure):
    if enclosure.dim_mm is not None:
        return enclosure.dim_mm[0] == enclosure.dim_mm[2]
    return enclosure.space_l_mm == enclosure.space_r_mm


def validate_woofer_domain(params, enclosure):
    """A reduction must preserve the complete requested horn and enclosure."""
    from .profile_common import _normalise_quadrants
    if enclosure is None or not enclosure.woofers:
        return
    quadrants = _normalise_quadrants(params.get("quadrants", "1234"))
    if quadrants == "1234":
        return
    if quadrants != "14" or any(w.x_mm != 0 for w in enclosure.woofers):
        raise control_refusal("woofers require full domain or x=0 half with every woofer centred on x=0", 'Symmetry', 'Choose Full model, or centre every woofer on the vertical symmetry plane and use that half model.', message='This woofer arrangement cannot use the chosen symmetry.')
    if not _symmetric_enclosure_x(enclosure):
        raise control_refusal("woofer x=0 half requires symmetric left/right enclosure margins", 'Enclosure margins', 'Use equal Left and Right margins, or choose Full model.', message='The enclosure margins differ across the woofer symmetry plane.')
    from scipy.spatial import cKDTree
    from .profile_sampling import build_point_grid_arrays
    full = build_point_grid_arrays({**params, "quadrants": "1234"})
    points = np.asarray(full["inner_grid"]).reshape(-1, 3)
    reflected = points * [-1, 1, 1]
    if np.max(cKDTree(points).query(reflected)[0]) > 1e-6:
        raise control_refusal("woofer x=0 half requires a horn symmetric about x=0", 'Symmetry', 'Choose Full model in Symmetry.', message='The horn is not symmetric across the woofer symmetry plane.')


def cut_woofers(built, geometry):
    """Cut the existing flat baffle and attach exact diaphragm surfaces."""
    from .builders._occ import require_gmsh
    from .source_contour_occ import revolve_contour

    woofers = geometry.enclosure.woofers
    if not woofers:
        return
    half = not geometry.closed
    if half and (geometry.symmetry_planes != ("x",) or any(w.x_mm != 0 for w in woofers)):
        raise control_refusal("woofers require full domain or x=0 half with every woofer centred on x=0", 'Symmetry', 'Choose Full model, or centre every woofer on the vertical symmetry plane and use that half model.', message='This woofer arrangement cannot use the chosen symmetry.')
    if half and not _symmetric_enclosure_x(geometry.enclosure):
        raise control_refusal("woofer x=0 half requires symmetric left/right enclosure margins", 'Enclosure margins', 'Use equal Left and Right margins, or choose Full model.', message='The enclosure margins differ across the woofer symmetry plane.')
    if set(built.surface_groups) - {1, 2, 3}:
        raise control_refusal("woofer assembly does not support subdomain interfaces", 'Subdomain interfaces', 'Clear Subdomain interfaces before solving this assembly.', message='Woofer assemblies cannot include subdomain interfaces.')
    gmsh = require_gmsh()
    occ = gmsh.model.occ
    occ.synchronize()
    z = built.enclosure_bounds["z_front"]
    front = [(2, t) for t in built.mesh_surface_groups["enclosure"]
             if gmsh.model.getType(2, t) == "Plane" and abs(occ.getCenterOfMass(2, t)[2]-z) < 1e-6]
    if not front:
        raise ValueError("no planar front baffle found")
    # Enclosure sectors can have coincident, independently authored seam
    # curves. Fuse copies solely for the boundary-distance query.
    clearance_front = occ.copy(front)
    if len(clearance_front) > 1:
        clearance_front, _ = occ.fuse(clearance_front[:1], clearance_front[1:])
    occ.synchronize()
    boundary = [abs(t) for d, t in gmsh.model.getBoundary(clearance_front, combined=True, oriented=True)
                if d == 1 and not (half and _on_x_cut(gmsh, 1, abs(t)))]
    discs = []
    for i, w in enumerate(woofers):
        depth = w.shape.recess_mm if isinstance(w.shape, PistonShape) else ATH_JOIN_DEPTH*w.radius_mm
        if depth >= z - built.enclosure_bounds["z_back"]:
            raise control_refusal("woofer diaphragm reaches enclosure back", 'Enclosure depth', 'Increase Enclosure depth or reduce the woofer depth.', message='A woofer diaphragm reaches the back of the enclosure.')
        for other in woofers[:i]:
            if math.hypot(w.x_mm-other.x_mm, w.y_mm-other.y_mm) < w.radius_mm+other.radius_mm+CLEARANCE_MM-1e-8:
                raise control_refusal("woofer apertures overlap or lack 0.1 mm clearance", 'Woofer position', 'Move the woofers farther apart or reduce Radius in Sources and baffle.', message='Two woofer openings overlap or have less than 0.1 mm clearance.')
        disc = _disc(occ, w, z, half)
        for curve in boundary:
            distance = occ.getDistance(2, disc, 1, curve)[0]
            if not math.isfinite(distance) or distance < CLEARANCE_MM-1e-8:
                raise control_refusal("woofer aperture lacks 0.1 mm clearance to mouth/flat baffle boundary", 'Woofer position', 'Move the woofer or increase the enclosure margins to leave at least 0.1 mm clearance.', message='A woofer opening is too close to the horn mouth or baffle edge.')
        discs.append((2, disc))
    occ.remove(clearance_front, recursive=True)
    _, maps = occ.fragment(front, discs)
    front_result = {t for mapping in maps[:len(front)] for d, t in mapping if d == 2}
    patches = [{t for d, t in mapping if d == 2} for mapping in maps[len(front):]]
    for w, patch in zip(woofers, patches):
        area = sum(occ.getMass(2, t) for t in patch & front_result)
        if patch-front_result or not math.isclose(area, math.pi*w.radius_mm**2/(2 if half else 1), rel_tol=1e-7):
            raise control_refusal("woofer aperture is not wholly inside the flat front baffle (mouth/edge collision)", 'Woofer position', 'Move the woofer inward, reduce Radius or increase the enclosure margins.', message='A woofer opening extends outside the flat front baffle.')
    old_front = {t for _, t in front}
    replacements = front_result-set().union(*patches)
    for groups in (built.surface_groups, built.mesh_surface_groups):
        for key, tags in groups.items():
            groups[key] = sorted((set(tags)-old_front) | (replacements if set(tags)&old_front else set()))
    sources = {"horn-throat": 2}
    tag_map = {"1": {"source_id": None, "instance_id": None, "role": "rigid"},
               "2": {"source_id": "horn-throat", "instance_id": None, "role": "primary"}}
    records = []
    contours = {}
    names = {1: NAMESPACE+"|rigid", 2: NAMESPACE+"|tag=2|source_id=horn-throat|instance_id=null|role=primary"}
    for w, top in zip(woofers, patches):
        collar = set()
        if isinstance(w.shape, AthConeShape):
            contour = ath_contour(w)
            contours[w.radiator_id] = contour.to_dict()
            _, diaphragm = revolve_contour(occ, contour, (w.x_mm, w.y_mm, z), half_x=half)
            segments = {s: {t} for s, t in diaphragm.items()}
            occ.remove([(2, t) for t in top], recursive=False)
        elif w.shape.recess_mm:
            occ.synchronize()
            boundary_curves = {abs(t) for d, t in gmsh.model.getBoundary(
                [(2, t) for t in top], combined=True, oriented=True)
                if d == 1 and not (half and _on_x_cut(gmsh, 1, abs(t)))}
            lower = set()
            for t in sorted(top):
                rim = {abs(e) for d, e in gmsh.model.getBoundary([(2, t)], oriented=True) if d == 1}
                out = occ.extrude([(2, t)], 0, 0, -w.shape.recess_mm)
                lower.add(out[0][1])
                occ.synchronize()
                for d, side in out[2:]:
                    if d != 2:
                        continue
                    curves = {abs(e) for dim, e in gmsh.model.getBoundary([(2, side)], oriented=False) if dim == 1}
                    if curves & rim & boundary_curves:
                        collar.add(side)
                occ.remove([dt for dt in out if dt[0] == 3], recursive=False)
                internal = [(d, t) for d, t in out[2:] if d == 2 and t not in collar]
                if internal:
                    occ.remove(internal, recursive=False)
            occ.remove([(2, t) for t in top], recursive=False)
            segments = {"piston": lower}
        else:
            segments = {"piston": top}
        # Exact OCC distances also detect deep collars/cones crossing rollback
        # walls. Closure alone cannot detect these overlapping closed surfaces.
        for face in set().union(*segments.values()) | collar:
            for wall in built.surface_groups[1] + built.surface_groups[2]:
                distance = occ.getDistance(2, face, 2, wall)[0]
                if not math.isfinite(distance) or distance < 1e-7:
                    raise control_refusal("woofer diaphragm/collar intersects horn wall/source", 'Woofer position', 'Move the woofer away from the horn or reduce Recess.', message='A woofer diaphragm or collar intersects the horn.')
        built.surface_groups.setdefault(3, []).extend(sorted(collar))
        built.mesh_surface_groups["enclosure"].extend(sorted(collar))
        for segment, faces in segments.items():
            patch_id = f"{w.radiator_id}/{segment}"
            physical = 101+len(sources)-1 if w.active else 1
            if w.active:
                sources[patch_id] = physical
                tag_map[str(physical)] = {"source_id": patch_id, "instance_id": None, "role": "secondary"}
                names[physical] = f"{NAMESPACE}|tag={physical}|source_id={patch_id}|instance_id=null|role=secondary"
            built.surface_groups.setdefault(physical, []).extend(sorted(faces))
            # Separate density role: the per-radiator size is independent of
            # horn and enclosure resolutions.
            role = "woofer:"+patch_id
            built.mesh_surface_groups[role] = sorted(faces)
            size = w.resolution_mm or 10.0
            if segment == "cap":
                size = min(size, math.sqrt(8*contour.arc(0)[0]*.025))
            if isinstance(w.shape, AthConeShape) and geometry.enclosure.ath_lf_envelope:
                # Imported ATH cones also bound circumferential chord error.
                # Keep generic assembly density and its budgets unchanged.
                size = min(size, math.sqrt(8*ATH_JOIN_RADIUS*w.radius_mm*.025))
            records.append({"patch_id": patch_id, "radiator_id": w.radiator_id,
                            "segment_id": segment, "role": "secondary" if w.active else "rigid",
                            "physical_tag": physical, "surface_tags": sorted(faces),
                            "center_mm": [w.x_mm, w.y_mm, z], "radius_mm": w.radius_mm,
                            "recess_mm": getattr(w.shape, "recess_mm", 0),
                            "resolution_mm": size, "density_role": role})
    built.surface_groups[1] = sorted(set(built.surface_groups[1]+built.surface_groups.pop(3, [])))
    # Unify coincident rim/join edges while retaining face ancestry. Only this
    # opt-in path performs the assembly boolean; horn-only output is untouched.
    occ.synchronize()
    faces = gmsh.model.getEntities(2)
    _, mapping = occ.fragment(faces, [])
    remap = {old: [t for d, t in new if d == 2] for (_, old), new in zip(faces, mapping)}
    _remap_groups(built.surface_groups, remap)
    _remap_groups(built.mesh_surface_groups, remap)
    for record in records:
        record["surface_tags"] = sorted({new for old in record["surface_tags"] for new in remap[old]})
    built.metadata.update(source_tags=sources, tag_map=tag_map, tag_namespace=NAMESPACE,
                          rigid_tag=1, assembly={"patches": records, "contours": contours,
                              "face_tags": {str(k): v for k, v in built.surface_groups.items()},
                              "half_x": half,
                              "coordinate_frame": "horn-throat-pre-placement-mm",
                              "vertical_offset_mm": geometry.vertical_offset_mm},
                          assemblyBaffleZMm=z, assemblyPhysicalNames=names)
    occ.synchronize()
    _certify_occ_partition(built, gmsh)


def certify_arrays(points, triangles, physical, surfaces, assembly):
    """Certify every declared patch after welding, filtering and orientation."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from .normals import validate_orientation

    owners = {face: int(tag) for tag, faces in assembly["face_tags"].items() for face in faces}
    if any(owners.get(int(face)) != int(tag) for face, tag in zip(surfaces, physical)):
        raise ValueError("assembly facet classification contradicts OCC identity map")
    # Check each body separately: a large positive body must not hide an
    # inverted or open disconnected component.
    graph = coo_matrix((np.ones(3*len(triangles)),
                       (triangles.ravel(), np.roll(triangles, -1, axis=1).ravel())),
                      shape=(len(points), len(points)))
    _, labels = connected_components(graph, directed=False)
    facet_labels = labels[triangles[:, 0]]
    for label in np.unique(facet_labels):
        selected = facet_labels == label
        validate_orientation(points, triangles[selected], physical[selected],
                             require_watertight=not assembly["half_x"],
                             require_edge_consistency=True, require_positive_volume=True)
    xyz = points[triangles]
    normals = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    seen = set()
    for patch in assembly["patches"]:
        selected = np.isin(surfaces, patch["surface_tags"])
        if not np.any(selected) or np.any(lengths[selected] <= 1e-12):
            raise ValueError(f"assembly patch {patch['patch_id']} has no positive mesh area")
        if seen.intersection(patch["surface_tags"]):
            raise ValueError("assembly patches must classify faces exactly once")
        seen.update(patch["surface_tags"])
        if np.any(physical[selected] != patch["physical_tag"]):
            raise ValueError("assembly patch physical tag disagrees with identity map")
        vertices = xyz[selected]
        center = np.asarray(patch["center_mm"])
        local = vertices-center
        radial = np.linalg.norm(local[:, :, :2], axis=2)
        r = patch["radius_mm"]
        segment = patch["segment_id"]
        expected = np.zeros((selected.sum(), 3))
        if segment == "piston":
            residual = abs(local[:, :, 2]+patch["recess_mm"])
            expected[:, 2] = 1
        else:
            contour = SourceContour.from_dict(assembly["contours"][patch["radiator_id"]])
            index = next(i for i, s in enumerate(contour.segments) if s.id == segment)
            authored = contour.segments[index]
            if authored.kind == "line":
                a, b = contour.points[index:index+2]
                slope = (b.z_mm-a.z_mm)/(b.r_mm-a.r_mm)
                residual = abs(local[:, :, 2] - (a.z_mm + (radial-a.r_mm)*slope))
                expected[:, :2] = -local[:, :, :2].mean(axis=1)*slope
                expected[:, 2] = np.linalg.norm(local[:, :, :2].mean(axis=1), axis=1)
            else:
                radius, _, _ = contour.arc(index)
                cr, cz = authored.center_mm
                relative = np.stack((radial-cr, local[:, :, 2]-cz), axis=2)
                residual = abs(np.linalg.norm(relative, axis=2)-radius)
                radial_unit = np.divide(local[:, :, :2], radial[:, :, None],
                                        out=np.zeros_like(local[:, :, :2]), where=radial[:, :, None] > 0)
                expected[:, :2] = (radial_unit*(radial-cr)[:, :, None]).mean(axis=1)
                expected[:, 2] = (local[:, :, 2]-cz).mean(axis=1)
        if np.max(residual) > 1e-5 or np.any(radial > r+1e-5):
            raise ValueError(f"assembly patch {patch['patch_id']} deviates from declared geometry")
        if np.any(np.einsum("ij,ij->i", normals[selected], expected) <= 0):
            raise ValueError(f"assembly patch {patch['patch_id']} has inward normals")
    # Every emitted facet has exactly one nonempty declared physical owner.
    if np.any(physical <= 0) or len(set(map(tuple, np.sort(triangles, axis=1)))) != len(triangles):
        raise ValueError("assembly facets must be classified exactly once")
