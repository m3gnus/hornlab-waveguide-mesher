"""Atomic two-source export with one sewn scattering shell, never shell stacking."""

import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

from .phase_plug import FEATURE as PLUG_FEATURE
from .phase_plug import passage_contract, rigid_edges, surface_targets, wall_edges
from .source_assembly import FEATURE, SourceAssembly, assembly_channels
from .source_contour import FEATURE as CONTOUR_FEATURE
from .source_contour import ContourDrive, canonical, digest


def export_assembly(
    assembly,
    drives,
    destination,
    *,
    mesh_size_mm=2.0,
    triangle_limit=250000,
    passage_refinement=1,
):
    channels = assembly_channels(assembly, drives)
    if (
        type(mesh_size_mm) not in (int, float)
        or not math.isfinite(mesh_size_mm)
        or mesh_size_mm <= 0
    ):
        raise ValueError("mesh size must be finite and positive")
    if type(triangle_limit) is not int or not 1 <= triangle_limit <= 250000:
        raise ValueError("triangle budget must be between 1 and 250000")
    estimate = 4 * sum(assembly.rigid_areas().values()) / mesh_size_mm**2
    if type(passage_refinement) is not int or passage_refinement not in (1, 2, 4):
        raise ValueError("passage refinement must be 1, 2 or 4")
    targets = surface_targets(assembly, mesh_size_mm, passage_refinement)
    estimate += sum(
        4 * assembly.rigid_areas()[role] * (1 / target**2 - 1 / mesh_size_mm**2)
        for role, target in targets.items()
    )
    for _, c, i, _, _ in assembly.patches:
        a, b = c.points[i : i + 2]
        target = mesh_size_mm
        length = math.hypot(b.r_mm - a.r_mm, b.z_mm - a.z_mm)
        if c.segments[i].kind == "arc":
            R, _, sweep = c.arc(i)
            target = min(target, math.sqrt(8 * 0.02 * R))
            length = R * abs(sweep)
        estimate += 4 * 2 * math.pi * b.r_mm * length / target**2
    if math.ceil(estimate) > triangle_limit:
        raise ValueError("assembly estimated triangle budget exceeded")
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError("assembly publication requires a new destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "recipe": assembly.to_dict(),
        "drives": [asdict(d) for d in drives],
        "channels": channels,
        "mesh_size_mm": float(mesh_size_mm),
        "triangle_limit": triangle_limit,
        "passage_refinement": passage_refinement,
    }
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=".assembly-"
    ) as temporary:
        stage = Path(temporary)
        (stage / "request.json").write_bytes(canonical(request))
        run = subprocess.run(
            [sys.executable, "-m", "hornlab_mesher.assembly_artifact", str(stage)],
            capture_output=True,
            text=True,
            check=False,
        )
        if run.returncode:
            raise ValueError("native assembly export failed: " + run.stderr)
        (stage / "request.json").unlink()
        stage.rename(destination)
    return json.loads((destination / "source.json").read_text())


def _worker(stage):
    import gmsh
    import numpy as np

    from .cad import _entity_closure, _prune_to
    from .step_mapping import advanced_face_order_for_surfaces

    request = json.loads((stage / "request.json").read_text())
    assembly = SourceAssembly.from_dict(request["recipe"])
    drives = [ContourDrive(**d) for d in request["drives"]]
    channels = assembly_channels(assembly, drives)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.model.add("native-shared-source-assembly")
        occ = gmsh.model.occ
        patch_faces, rigid_roles, chains, apertures = {}, {}, [], []
        plug_faces = []
        for contour, origin, band in assembly.parts:
            cx, cy, cz = origin
            points = {
                p.id: occ.addPoint(cx + p.r_mm, cy, cz + p.z_mm) for p in contour.points
            }
            chain = []
            for identifier, c, i, _, _ in assembly.patches:
                if c is not contour:
                    continue
                s = c.segments[i]
                a, b = points[s.start], points[s.end]
                if s.kind == "line":
                    edge = occ.addLine(a, b)
                else:
                    r, z = c.evaluate(i, 0.5)
                    edge = occ.addCircleArc(
                        a, occ.addPoint(cx + r, cy, cz + z), b, center=False
                    )
                faces = [
                    t
                    for dim, t in occ.revolve(
                        [(1, edge)], cx, cy, cz, 0, 0, 1, 2 * math.pi
                    )
                    if dim == 2
                ]
                if len(faces) != 1:
                    raise ValueError("each canonical patch must revolve into one face")
                patch_faces[identifier] = faces[0]
                chain.append(faces[0])
            if band == "HF":
                role, r, z = "horn-wall", assembly.mouth_radius_mm, assembly.front_z_mm
            else:
                role, r, z = "collar", assembly.aperture_radius_mm, assembly.front_z_mm
            if band == "HF":
                start = points[contour.points[-1].id]
                if assembly.horn_wall is not None:
                    face = assembly.horn_wall.add_occ_face(occ, origin)
                    rigid_roles["horn-wall"] = face
                    chain.append(face)
                for role, (_, b) in wall_edges(assembly).items():
                    end = occ.addPoint(cx + b[0], cy, cz + b[1])
                    edge = occ.addLine(start, end)
                    rigid_roles[role] = next(
                        t
                        for dim, t in occ.revolve(
                            [(1, edge)], cx, cy, cz, 0, 0, 1, 2 * math.pi
                        )
                        if dim == 2
                    )
                    chain.append(rigid_roles[role])
                    start = end
            elif r > contour.points[-1].r_mm:
                end = occ.addPoint(cx + r, cy, z)
                edge = occ.addLine(points[contour.points[-1].id], end)
                rigid_roles[role] = next(
                    t
                    for dim, t in occ.revolve(
                        [(1, edge)], cx, cy, cz, 0, 0, 1, 2 * math.pi
                    )
                    if dim == 2
                )
                chain.append(rigid_roles[role])
            occ.synchronize()
            ring = [
                t
                for dim, t in gmsh.model.getBoundary([(2, chain[-1])], oriented=False)
                if dim == 1
                and abs(occ.getMass(1, t) - 2 * math.pi * r) < 1e-6
                and np.linalg.norm(np.asarray(occ.getCenterOfMass(1, t)) - (cx, cy, z))
                < 1e-6
            ]
            if len(ring) != 1:
                raise ValueError("source must attach to one actual circular boundary")
            apertures.append(ring[0])
            chains.append(chain)
        w, h, z = assembly.width_mm / 2, assembly.height_mm / 2, assembly.front_z_mm
        corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
        front = [occ.addPoint(x, y, z) for x, y in corners]
        back = [occ.addPoint(x, y, z - assembly.depth_mm) for x, y in corners]
        fe = [occ.addLine(front[i], front[(i + 1) % 4]) for i in range(4)]
        be = [occ.addLine(back[i], back[(i + 1) % 4]) for i in range(4)]
        ve = [occ.addLine(front[i], back[i]) for i in range(4)]
        rigid_roles["front"] = occ.addPlaneSurface(
            [occ.addCurveLoop(fe), *[occ.addCurveLoop([-a]) for a in apertures]]
        )
        rigid_roles["back"] = occ.addPlaneSurface([occ.addCurveLoop(be)])
        for i, role in enumerate(["bottom", "right", "top", "left"]):
            rigid_roles[role] = occ.addPlaneSurface(
                [occ.addCurveLoop([fe[i], ve[(i + 1) % 4], -be[i], -ve[i]])]
            )
        cx, cy, cz = assembly.parts[0][1]
        for plug in assembly.phase_plugs:
            body = []
            for role, (a, b) in plug.edges.items():
                edge = occ.addLine(
                    occ.addPoint(cx + a[0], cy, cz + a[1]),
                    occ.addPoint(cx + b[0], cy, cz + b[1]),
                )
                face = next(
                    t
                    for dim, t in occ.revolve(
                        [(1, edge)], cx, cy, cz, 0, 0, 1, 2 * math.pi
                    )
                    if dim == 2
                )
                rigid_roles[role] = face
                body.append(face)
            plug_faces.append(body)
        occ.synchronize()
        faces = list(patch_faces.values()) + list(rigid_roles.values())
        _prune_to(gmsh, _entity_closure(gmsh, [(2, t) for t in faces]))
        original = {
            key: (
                occ.getMass(2, t),
                np.asarray(occ.getCenterOfMass(2, t)),
                np.asarray(occ.getBoundingBox(2, t)),
            )
            for key, t in {
                **{("patch", k): t for k, t in patch_faces.items()},
                **{("rigid", k): t for k, t in rigid_roles.items()},
            }.items()
        }
        plug_face_set = {t for body in plug_faces for t in body}
        bodies = [[t for t in faces if t not in plug_face_set], *plug_faces]
        handles = [
            occ.addVolume([occ.addSurfaceLoop(body, sewing=True)]) for body in bodies
        ]
        occ.synchronize()
        _prune_to(gmsh, _entity_closure(gmsh, [(3, handle) for handle in handles]))
        faces = [t for _, t in gmsh.model.getEntities(2)]
        remapped = {}
        for key, (area, center, bounds) in original.items():
            candidates = [
                t
                for t in faces
                if abs(occ.getMass(2, t) - area) < 1e-7 * max(1, area)
                and np.linalg.norm(np.asarray(occ.getCenterOfMass(2, t)) - center)
                < 1e-7
                and np.max(abs(np.asarray(occ.getBoundingBox(2, t)) - bounds)) < 1e-6
            ]
            if len(candidates) != 1:
                raise ValueError("sewn assembly mapping is ambiguous")
            remapped[key] = candidates[0]
        old_new = {
            **{t: remapped[("patch", k)] for k, t in patch_faces.items()},
            **{t: remapped[("rigid", k)] for k, t in rigid_roles.items()},
        }
        patch_faces = {k: remapped[("patch", k)] for k in patch_faces}
        rigid_roles = {k: remapped[("rigid", k)] for k in rigid_roles}
        joins = []
        for chain in chains:
            chain = [old_new[t] for t in chain] + [rigid_roles["front"]]
            boundaries = [
                {
                    t
                    for dim, t in gmsh.model.getBoundary([(2, f)], oriented=False)
                    if dim == 1
                }
                for f in chain
            ]
            for a, b in pairwise(boundaries):
                if not a & b:
                    raise ValueError(
                        "assembly source/horn/baffle joins must share actual OCC edges"
                    )
                joins.append(len(a & b))
        step = stage / "geometry.step"
        gmsh.write(str(step))
        step_text, count = re.subn(
            r"MANIFOLD_SOLID_BREP\('([^']*)',(#\d+)\)",
            r"SHELL_BASED_SURFACE_MODEL('\1',(\2))",
            step.read_text(),
        )
        if count != len(handles):
            raise ValueError("assembly must export its exact sewn shell inventory")
        step.write_text(
            step_text.replace(
                "ADVANCED_BREP_SHAPE_REPRESENTATION(",
                "MANIFOLD_SURFACE_SHAPE_REPRESENTATION(",
            )
        )
        occ.remove([(3, handle) for handle in handles], recursive=False)
        occ.synchronize()
        order = advanced_face_order_for_surfaces(step, faces)
        selectors = dict(zip(faces, order))
        patches = []
        for index, (identifier, c, i, _, band) in enumerate(assembly.patches):
            s = c.segments[i]
            tag = 1001 + index if s.role == "moving" else 1 + index
            face = patch_faces[identifier]
            patches.append(
                {
                    "id": identifier,
                    "local_patch_id": s.id,
                    "physical_source_id": c.physical_source_id,
                    "role": s.role,
                    "band": band,
                    "advanced_face_indices": [selectors[face]],
                    "area_mm2": occ.getMass(2, face),
                    "mesh_tag": tag,
                }
            )
            gmsh.model.addPhysicalGroup(2, [face], tag)
        gmsh.model.addPhysicalGroup(2, list(rigid_roles.values()), 1 + len(patches))
        gmsh.option.setNumber("Mesh.MeshSizeMin", request["mesh_size_mm"])
        gmsh.option.setNumber("Mesh.MeshSizeMax", request["mesh_size_mm"])
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 64)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        fields, targets = [], [request["mesh_size_mm"]]
        for role, target in surface_targets(
            assembly, request["mesh_size_mm"], request["passage_refinement"]
        ).items():
            targets.append(target)
            constant = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(constant, "F", str(target))
            restricted = gmsh.model.mesh.field.add("Restrict")
            gmsh.model.mesh.field.setNumber(restricted, "InField", constant)
            gmsh.model.mesh.field.setNumbers(
                restricted, "SurfacesList", [rigid_roles[role]]
            )
            fields.append(restricted)
        for identifier, c, i, _, _ in assembly.patches:
            if c.segments[i].kind != "arc":
                continue
            target = min(request["mesh_size_mm"], math.sqrt(8 * 0.02 * c.arc(i)[0]))
            targets.append(target)
            constant = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(constant, "F", str(target))
            restricted = gmsh.model.mesh.field.add("Restrict")
            gmsh.model.mesh.field.setNumber(restricted, "InField", constant)
            gmsh.model.mesh.field.setNumbers(
                restricted, "SurfacesList", [patch_faces[identifier]]
            )
            fields.append(restricted)
        if fields:
            minimum = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
            gmsh.option.setNumber("Mesh.MeshSizeMin", min(targets))
        gmsh.model.mesh.generate(2)
        for face in faces:
            lo, hi = gmsh.model.getParametrizationBounds(2, face)
            uv = [(float(a) + float(b)) / 2 for a, b in zip(lo, hi)]
            normal = np.asarray(gmsh.model.getNormal(face, uv))
            if face in patch_faces.values():
                reverse = normal[2] < 0
            else:
                role = next(k for k, t in rigid_roles.items() if t == face)
                if role == "horn-wall" and assembly.horn_wall is not None:
                    point = np.asarray(gmsh.model.getValue(2, face, uv))
                    radial = point[:2] - np.asarray(assembly.horn_xy_mm)
                    radial /= np.linalg.norm(radial)
                    dr, dz = assembly.horn_wall.spline(uv[1], 1)
                    expected = np.r_[-dz * radial, dr]
                    reverse = normal @ expected < 0
                elif role in rigid_edges(assembly):
                    point = np.asarray(gmsh.model.getValue(2, face, uv))
                    local = point - np.asarray(assembly.parts[0][1])
                    radial = local[:2] / np.linalg.norm(local[:2])
                    a, b = rigid_edges(assembly)[role]
                    dr, dz = b[0] - a[0], b[1] - a[1]
                    sign = -1 if role.startswith("horn-wall") else 1
                    expected = sign * np.array([radial[0] * dz, radial[1] * dz, -dr])
                    reverse = normal @ expected < 0
                else:
                    axis, _, _, _, sign, _ = assembly.box_surfaces[role]
                    reverse = normal[axis] * sign < 0
            if reverse:
                gmsh.model.mesh.reverse([(2, face)])
        triangle_count = sum(len(t) for t in gmsh.model.mesh.getElements(2)[1])
        if triangle_count > request["triangle_limit"]:
            raise ValueError("assembly actual triangle budget exceeded")
        gmsh.write(str(stage / "preview.msh"))
        passage_quality = None
        if assembly.phase_plugs:
            import meshio

            from .passage_mesh import certify_passage_mesh

            mesh = meshio.read(stage / "preview.msh")
            tri = mesh.get_cells_type("triangle")
            tags = mesh.get_cell_data("gmsh:physical", "triangle")
            active = np.isin(
                tags, [p["mesh_tag"] for p in patches if p["role"] == "moving"]
            )
            passage_quality = certify_passage_mesh(assembly, mesh.points, tri, active)
        rigid_faces = [selectors[t] for t in rigid_roles.values()] + [
            p["advanced_face_indices"][0] for p in patches if p["role"] == "rigid"
        ]
        density = {
            "mesh_size_mm": request["mesh_size_mm"],
            "triangle_limit": request["triangle_limit"],
            "triangle_count": triangle_count,
            **(
                {
                    "rigid_role_sizes_mm": surface_targets(
                        assembly, request["mesh_size_mm"], request["passage_refinement"]
                    ),
                    "passage_refinement": request["passage_refinement"],
                }
                if assembly.phase_plugs
                else {}
            ),
        }
        from .general_horn import wall_face_area
        if assembly.horn_wall is not None:
            actual_area = wall_face_area(gmsh, rigid_roles["horn-wall"], assembly.horn_wall)
            if abs(actual_area - assembly.horn_wall.area_mm2) > 1e-7 * max(1, actual_area):
                raise ValueError("general horn CAD area contradicts the canonical axial fit")
        manifest = {
            "version": 1,
            "producer": "native",
            "units": "mm",
            "required_features": [CONTOUR_FEATURE, FEATURE]
            + ([PLUG_FEATURE] if assembly.phase_plugs else [])
            + (["native-general-horn-attachment-v1"]
               if assembly.horn_wall is not None or assembly.woofer is None else []),
            "recipe": assembly.to_dict(),
            "geometry_sha256": assembly.geometry_sha256,
            "channels": channels,
            "excitation_sha256": digest(channels),
            "patches": patches,
            "rigid_faces": [
                {
                    "role": k,
                    "advanced_face_indices": [selectors[t]],
                    "area_mm2": (actual_area if k == "horn-wall" and assembly.horn_wall is not None
                                 else occ.getMass(2, t)),
                }
                for k, t in rigid_roles.items()
            ],
            "rigid_face_indices": sorted(rigid_faces),
            "all_face_indices": sorted(order),
            "shared_join_edge_counts": joins,
            **(
                {
                    "passage_contract": passage_contract(assembly),
                    "passage_quality": passage_quality,
                }
                if assembly.phase_plugs
                else {}
            ),
            "density": density,
            "mesh_density_sha256": digest(density),
            "members": {
                name: "sha256:"
                + hashlib.sha256((stage / name).read_bytes()).hexdigest()
                for name in ("geometry.step", "preview.msh")
            },
        }
        (stage / "source.json").write_bytes(canonical(manifest))
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    _worker(Path(sys.argv[1]))
