"""Atomic native source/straight circular horn export, isolated from caller Gmsh.

The public artifact is native authored evidence, never a managed CAD return.
STEP face selectors are temporary and valid only for the published STEP digest.
"""

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

from .source_contour import FEATURE, ContourDrive, SourceContour, canonical, digest


def export_contour(
    contour,
    drive,
    destination,
    *,
    mouth_radius_mm,
    length_mm,
    mesh_size_mm=2.0,
    triangle_limit=250000,
    housing_radius_mm=None,
    backing_depth_mm=15.0,
):
    """Close a native circular conical horn's throat with the exact meridian.

    Initial attachment scope: aligned +Z, origin at the circular throat, bare
    horn cavity in a circular rigid housing. A later assembly can reuse this meridian.
    """
    drive.validate(contour)
    rim = contour.points[-1].r_mm
    housing_radius_mm = (
        mouth_radius_mm + 1 if housing_radius_mm is None else housing_radius_mm
    )
    if (
        not all(math.isfinite(x) for x in (mouth_radius_mm, length_mm, mesh_size_mm))
        or not mouth_radius_mm > rim
        or length_mm <= contour.axial_bounds_mm[1]
        or mesh_size_mm <= 0
    ):
        raise ValueError("invalid circular horn attachment or mesh dimensions")
    if (
        not all(math.isfinite(x) for x in (housing_radius_mm, backing_depth_mm))
        or housing_radius_mm <= mouth_radius_mm
        or backing_depth_mm <= max(0, -contour.axial_bounds_mm[0])
    ):
        raise ValueError("invalid housing clearance")
    if not isinstance(triangle_limit, int) or not 1 <= triangle_limit <= 250000:
        raise ValueError("triangle budget must be between 1 and 250000")
    # Area/size estimate, including curvature-limited targets. Refuse
    # before OCC or meshing; the exact post-mesh limit is checked as well.
    source_estimate = 0
    for i, (a, b, s) in enumerate(
        zip(contour.points, contour.points[1:], contour.segments)
    ):
        length = (
            math.hypot(b.r_mm - a.r_mm, b.z_mm - a.z_mm)
            if s.kind == "line"
            else contour.arc(i)[0] * abs(contour.arc(i)[2])
        )
        target = (
            mesh_size_mm
            if s.kind == "line"
            else min(mesh_size_mm, math.sqrt(8 * 0.02 * contour.arc(i)[0]))
        )
        source_estimate += 4 * 2 * math.pi * b.r_mm * length / target**2
    bound_area = (
        math.pi * (rim + mouth_radius_mm) * math.hypot(length_mm, mouth_radius_mm - rim)
    )
    bound_area += (
        math.pi * (housing_radius_mm**2 - mouth_radius_mm**2)
        + 2 * math.pi * housing_radius_mm * (length_mm + backing_depth_mm)
        + math.pi * housing_radius_mm**2
    )
    estimate = math.ceil(source_estimate + 4 * bound_area / mesh_size_mm**2)
    if estimate > triangle_limit:
        raise ValueError(
            f"contour estimated triangle budget exceeded: {estimate} > {triangle_limit}"
        )
    mouth_radius_mm, length_mm, mesh_size_mm, housing_radius_mm, backing_depth_mm = map(
        float,
        (mouth_radius_mm, length_mm, mesh_size_mm, housing_radius_mm, backing_depth_mm),
    )
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError("contour publication requires a new destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "contour": contour.to_dict(),
        "drive": asdict(drive),
        "horn": {
            "rim_id": contour.rim_id,
            "radius_mm": rim,
            "mouth_radius_mm": mouth_radius_mm,
            "length_mm": length_mm,
            "housing_radius_mm": housing_radius_mm,
            "backing_depth_mm": backing_depth_mm,
        },
        "mesh_size_mm": mesh_size_mm,
        "triangle_limit": triangle_limit,
        "estimated_triangles": estimate,
    }
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=".contour-"
    ) as temporary:
        stage = Path(temporary)
        (stage / "request.json").write_bytes(canonical(request))
        run = subprocess.run(
            [sys.executable, "-m", "hornlab_mesher.contour_artifact", str(stage)],
            capture_output=True,
            text=True,
            check=False,
        )
        if run.returncode:
            raise ValueError("native contour export failed: " + run.stderr)
        (stage / "request.json").unlink()
        stage.rename(destination)
    return json.loads((destination / "source.json").read_text())


def _worker(stage):
    import gmsh
    import numpy as np

    from .cad import _entity_closure, _prune_to
    from .step_mapping import advanced_face_order_for_surfaces

    request = json.loads((stage / "request.json").read_text())
    contour = SourceContour.from_dict(request["contour"])
    drive = ContourDrive(**request["drive"])
    drive.validate(contour)
    horn = request["horn"]
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.model.add("native-source-contour")
        occ = gmsh.model.occ
        point_tags = {p.id: occ.addPoint(p.r_mm, 0, p.z_mm) for p in contour.points}
        patch_faces = {}
        for segment in contour.segments:
            a, b = point_tags[segment.start], point_tags[segment.end]
            if segment.kind == "line":
                edge = occ.addLine(a, b)
            else:
                # Three points preserve the authored branch even for exactly
                # semicircular surrounds (center-only arcs are ambiguous).
                r, z = contour.evaluate(len(patch_faces), 0.5)
                midpoint = occ.addPoint(r, 0, z)
                edge = occ.addCircleArc(a, midpoint, b, center=False)
            swept = occ.revolve([(1, edge)], 0, 0, 0, 0, 0, 1, 2 * math.pi)
            faces = [tag for dim, tag in swept if dim == 2]
            if len(faces) != 1:
                raise ValueError("a contour segment must revolve into exactly one face")
            patch_faces[segment.id] = faces[0]
        mouth = occ.addPoint(horn["mouth_radius_mm"], 0, horn["length_mm"])
        edge = occ.addLine(point_tags[contour.points[-1].id], mouth)
        swept = occ.revolve([(1, edge)], 0, 0, 0, 0, 0, 1, 2 * math.pi)
        wall = next(tag for dim, tag in swept if dim == 2)
        rigid_faces = [wall]
        previous = mouth
        for r, z in [
            (horn["housing_radius_mm"], horn["length_mm"]),
            (horn["housing_radius_mm"], -horn["backing_depth_mm"]),
            (0, -horn["backing_depth_mm"]),
        ]:
            next_point = occ.addPoint(r, 0, z)
            edge = occ.addLine(previous, next_point)
            swept = occ.revolve([(1, edge)], 0, 0, 0, 0, 0, 1, 2 * math.pi)
            rigid_faces += [tag for dim, tag in swept if dim == 2]
            previous = next_point
        occ.removeAllDuplicates()
        occ.synchronize()
        faces = [tag for dim, tag in gmsh.model.getEntities(2)]
        if set(faces) != {*patch_faces.values(), *rigid_faces}:
            raise ValueError("sewing changed the declared patch coverage")
        # Delete meridian construction debris while retaining every boundary.
        used = set(
            gmsh.model.getBoundary(
                [(2, t) for t in faces], combined=False, oriented=False
            )
        )
        occ.remove(
            [entity for entity in gmsh.model.getEntities(1) if entity not in used],
            recursive=False,
        )
        occ.synchronize()
        rings = [
            {
                abs(tag)
                for dim, tag in gmsh.model.getBoundary([(2, t)], oriented=False)
                if dim == 1
            }
            for t in [*patch_faces.values(), wall]
        ]
        if any(not (a & b) for a, b in pairwise(rings)):
            raise ValueError("source joins and horn rim must share actual OCC edges")
        _prune_to(gmsh, _entity_closure(gmsh, [(2, t) for t in faces]))
        # Gmsh exports free faces as separate one-face shells. Register the
        # sewn shell through its volume handle to retain shared STEP edges,
        # then publish a zero-volume surface representation of that shell.
        shell = occ.addSurfaceLoop(faces, sewing=True)
        original = {
            key: (
                occ.getMass(2, tag),
                np.asarray(occ.getCenterOfMass(2, tag)),
                np.asarray(occ.getBoundingBox(2, tag)),
            )
            for key, tag in {
                **{("patch", key): tag for key, tag in patch_faces.items()},
                **{("rigid", i): t for i, t in enumerate(rigid_faces)},
            }.items()
        }
        handle = occ.addVolume([shell])
        occ.synchronize()
        _prune_to(gmsh, _entity_closure(gmsh, [(3, handle)]))
        faces = [tag for _, tag in gmsh.model.getEntities(2)]
        remapped = {}
        for key, (area, center, bounds) in original.items():
            candidates = [
                tag
                for tag in faces
                if abs(occ.getMass(2, tag) - area) < 1e-7 * max(1, area)
                and np.linalg.norm(np.asarray(occ.getCenterOfMass(2, tag)) - center)
                < 1e-7
                and np.max(abs(np.asarray(occ.getBoundingBox(2, tag)) - bounds)) < 1e-6
            ]
            if len(candidates) != 1:
                raise ValueError("sewn patch mapping is ambiguous")
            remapped[key] = candidates[0]
        rigid_faces = [remapped[("rigid", i)] for i in range(len(rigid_faces))]
        wall = rigid_faces[0]
        patch_faces = {s.id: remapped[("patch", s.id)] for s in contour.segments}
        step = stage / "geometry.step"
        gmsh.write(str(step))
        step_text = step.read_text()
        step_text, n = re.subn(
            r"MANIFOLD_SOLID_BREP\('([^']*)',(#\d+)\)",
            r"SHELL_BASED_SURFACE_MODEL('\1',(\2))",
            step_text,
        )
        if n != 1:
            raise ValueError("expected exactly one registered native shell")
        step_text = step_text.replace(
            "ADVANCED_BREP_SHAPE_REPRESENTATION(",
            "MANIFOLD_SURFACE_SHAPE_REPRESENTATION(",
        )
        step.write_text(step_text)
        occ.remove([(3, handle)], recursive=False)
        occ.synchronize()
        face_order = advanced_face_order_for_surfaces(step, faces)
        identifiers = dict(zip(faces, face_order))
        patches = [
            {
                "id": s.id,
                "role": s.role,
                "advanced_face_indices": [identifiers[patch_faces[s.id]]],
                "area_mm2": occ.getMass(2, patch_faces[s.id]),
                "mesh_tag": 1001 + i if s.role == "moving" else 1 + i,
            }
            for i, s in enumerate(contour.segments)
        ]
        # Geometry and excitation never share an identity with mesh density.
        recipe = {
            "contour": contour.to_dict(),
            "horn": horn,
            "frame": "aligned-z-mm-v1",
        }
        manifest = {
            "version": 1,
            "required_features": [FEATURE],
            "producer": "native",
            "units": "mm",
            "geometry_sha256": digest(recipe),
            "excitation_sha256": drive.excitation_sha256,
            "recipe": recipe,
            "drive": asdict(drive),
            "patches": patches,
            "rigid_face_indices": [identifiers[t] for t in rigid_faces]
            + [
                identifiers[patch_faces[s.id]]
                for s in contour.segments
                if s.role == "rigid"
            ],
            "all_face_indices": sorted(face_order),
        }
        for i, s in enumerate(contour.segments):
            gmsh.model.addPhysicalGroup(
                2, [patch_faces[s.id]], 1001 + i if s.role == "moving" else 1 + i
            )
        gmsh.model.addPhysicalGroup(2, rigid_faces, 1 + len(contour.segments))
        gmsh.option.setNumber("Mesh.MeshSizeMin", request["mesh_size_mm"])
        gmsh.option.setNumber("Mesh.MeshSizeMax", request["mesh_size_mm"])
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 64)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        # OCC's global curvature sizing can miss a small toroidal surround.
        # Bound its meridional chord scale explicitly without refining the
        # entire housing to the source's smallest radius.
        fields = []
        for i, s in enumerate(contour.segments):
            if s.kind == "arc":
                radius = contour.arc(i)[0]
                target = min(request["mesh_size_mm"], math.sqrt(8 * 0.02 * radius))
                constant = gmsh.model.mesh.field.add("MathEval")
                gmsh.model.mesh.field.setString(constant, "F", str(target))
                restricted = gmsh.model.mesh.field.add("Restrict")
                gmsh.model.mesh.field.setNumber(restricted, "InField", constant)
                gmsh.model.mesh.field.setNumbers(
                    restricted, "SurfacesList", [patch_faces[s.id]]
                )
                fields.append(restricted)
        if fields:
            minimum = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
            gmsh.option.setNumber(
                "Mesh.MeshSizeMin",
                min(
                    request["mesh_size_mm"],
                    min(
                        math.sqrt(8 * 0.02 * contour.arc(i)[0])
                        for i, s in enumerate(contour.segments)
                        if s.kind == "arc"
                    ),
                ),
            )
        gmsh.model.mesh.generate(2)
        for face in faces:
            lo, hi = gmsh.model.getParametrizationBounds(2, face)
            uv = [(float(a) + float(b)) / 2 for a, b in zip(lo, hi)]
            n = np.asarray(gmsh.model.getNormal(face, uv))
            xyz = np.asarray(gmsh.model.getValue(2, face, uv))
            if face in patch_faces.values():
                reverse = n[2] < 0
            elif face == wall:
                reverse = float(n[:2] @ xyz[:2]) > 0
            elif face == rigid_faces[1]:
                reverse = n[2] < 0
            elif face == rigid_faces[2]:
                reverse = float(n[:2] @ xyz[:2]) < 0
            else:
                reverse = n[2] > 0
            if reverse:
                gmsh.model.mesh.reverse([(2, face)])
        count = sum(
            len(tags) for dim in [2] for tags in gmsh.model.mesh.getElements(dim)[1]
        )
        if count > request["triangle_limit"]:
            raise ValueError("contour actual triangle budget exceeded")
        gmsh.write(str(stage / "preview.msh"))
        manifest["density"] = {
            "mesh_size_mm": request["mesh_size_mm"],
            "triangle_limit": request["triangle_limit"],
            "triangle_count": count,
        }
        manifest["mesh_density_sha256"] = digest(manifest["density"])
        manifest["members"] = {
            name: "sha256:" + hashlib.sha256((stage / name).read_bytes()).hexdigest()
            for name in ("geometry.step", "preview.msh")
        }
        (stage / "source.json").write_bytes(canonical(manifest))
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    _worker(Path(sys.argv[1]))
