"""STEP export of the authoritative OCC waveguide model.

Every mesh build already authors the waveguide as an OpenCASCADE BRep -- smooth
B-spline wall patches, the wall shell, the rear enclosure and its edge treatment
-- and then throws that model away once Gmsh has tessellated it. This module
writes it out instead, so CAD receives the same geometry the solver saw rather
than a separately derived approximation.

Two differences from the mesh path are deliberate:

* The acoustic level-of-detail pass (``mesher._acoustic_geometry``) is not run.
  It may replace an enclosure fillet with a sharp edge when the fillet is
  smaller than the mesh can resolve, which is the right call for a solve and the
  wrong one for a part someone is going to machine.
* Mesh density is irrelevant, so no sizing fields are configured and no
  elements are generated. Export cost is the geometry build alone.

The source cap is the driver membrane, not material. It has to be present for
the shell to close, so the solid is sewn with it and the cap is then swept
backwards along the axis and cut away, leaving the bore open. Because the cut
follows the cap's own face, a flat disc and a domed cap yield the same solid.
"""

from __future__ import annotations

import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from .geometry import (
    BuiltGeometry,
    HornGeometry,
    OsseHornGeometry,
    PointGridBuildMode,
    PointGridHornGeometry,
)
from .mesher import _GMSH_LOCK, MesherError, _dispatch_builder

CadBody = Literal["solid", "surface"]

# Roles that belong to the acoustic model rather than the part. Interfaces are
# ATH ``Mesh.SubdomainSlices`` partitions floating inside the domain; the mouth
# aperture is an infinite-baffle model's Rayleigh coupling plane.
_NON_MATERIAL_ROLES = ("interface", "mouth_aperture")
# The driver membrane: sewn in to close the shell, then cut away.
_SOURCE_ROLE = "throat_disc"

# Build modes that enclose material. The others are zero-thickness acoustic
# surfaces, and sewing those would enclose the *air* in the bore rather than
# any part, so they export as a surface body instead.
_SOLID_BUILD_MODES = frozenset(
    {PointGridBuildMode.FREESTANDING, PointGridBuildMode.ENCLOSURE}
)

_MIN_SOLID_VOLUME_MM3 = 1.0e-6


@dataclass(frozen=True)
class CadInfo:
    """What was written, and what shape it turned out to be."""

    path: Path
    body: CadBody
    n_faces: int
    volume_mm3: float | None
    bounding_box_mm: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ]
    throat_opened: bool
    units: Literal["mm"] = "mm"


def _axis_direction(source_axis: str) -> tuple[float, float, float]:
    """Unit vector pointing from throat to mouth."""

    axis = str(source_axis)
    sign = -1.0 if axis.startswith("-") else 1.0
    letter = axis[-1]
    if letter not in "xyz":
        raise MesherError(f"unsupported source axis {source_axis!r}")
    vector = [0.0, 0.0, 0.0]
    vector["xyz".index(letter)] = sign
    return (vector[0], vector[1], vector[2])


def _role_tags(built: BuiltGeometry, *roles: str) -> set[int]:
    tags: set[int] = set()
    for role in roles:
        tags.update(int(tag) for tag in built.mesh_surface_groups.get(role, ()))
    return tags


def _solid_capable(geometry: HornGeometry) -> bool:
    if isinstance(geometry, OsseHornGeometry):
        # The OSSE builder emits an inner wall and a throat cap only.
        return False
    if isinstance(geometry, PointGridHornGeometry):
        return geometry.build_mode in _SOLID_BUILD_MODES
    return False


def _prune_to(gmsh, keep: set[tuple[int, int]]) -> None:
    """Delete every entity that is not part of ``keep``.

    Sewing leaves the original faces in the model beside the solid they were
    sewn into, and ``gmsh.write`` emits all of them -- a Fusion import would
    show one solid plus a pile of loose surfaces on top of it. Descending
    dimensions keeps each removal free of dependents.
    """

    for dim in (3, 2, 1, 0):
        stale = [
            (dim, int(tag))
            for entity_dim, tag in gmsh.model.getEntities(dim)
            if (int(entity_dim), int(tag)) not in keep
        ]
        if stale:
            gmsh.model.occ.remove(stale, recursive=False)
    gmsh.model.occ.synchronize()


def _entity_closure(gmsh, dimtags: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """``dimtags`` plus every lower-dimensional entity they are built from."""

    keep = {(int(d), int(t)) for d, t in dimtags}
    frontier = list(keep)
    while frontier:
        boundary = gmsh.model.getBoundary(
            frontier, oriented=False, combined=False, recursive=False
        )
        frontier = [
            (int(d), int(t))
            for d, t in boundary
            if (int(d), int(t)) not in keep
        ]
        keep.update(frontier)
    return keep


def _open_throat(
    gmsh, volume: int, built: BuiltGeometry, cap_tags: list[int]
) -> int:
    """Cut the driver membrane out of the sewn solid, opening the bore."""

    if not cap_tags:
        return volume
    bbox = gmsh.model.getBoundingBox(3, volume)
    extent = max(bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])
    depth = float(extent) + 10.0
    dx, dy, dz = _axis_direction(built.source_axis)
    plugs: list[tuple[int, int]] = []
    for cap in cap_tags:
        swept = gmsh.model.occ.extrude(
            [(2, int(cap))], -dx * depth, -dy * depth, -dz * depth
        )
        plugs.extend((int(d), int(t)) for d, t in swept if int(d) == 3)
    gmsh.model.occ.synchronize()
    if not plugs:
        raise MesherError("could not sweep the source cap to open the throat")
    result, _ = gmsh.model.occ.cut(
        [(3, volume)], plugs, removeObject=True, removeTool=True
    )
    gmsh.model.occ.synchronize()
    volumes = [int(tag) for dim, tag in result if int(dim) == 3]
    if len(volumes) != 1:
        raise MesherError(
            f"opening the throat produced {len(volumes)} solids; expected exactly one"
        )
    return volumes[0]


def _assert_step(text: str, *, body: CadBody) -> None:
    required = ["ISO-10303-21", "END-ISO-10303-21", "ADVANCED_FACE"]
    if body == "solid":
        required.append("MANIFOLD_SOLID_BREP")
    missing = [token for token in required if token not in text]
    if not text.strip() or missing:
        raise MesherError(
            "STEP export did not contain valid geometry (missing "
            + ", ".join(missing or ["content"])
            + ")"
        )
    if "SI_UNIT(.MILLI.,.METRE.)" not in text:
        raise MesherError("STEP export did not declare millimetre length units")


def write_step(
    geometry: HornGeometry,
    output_path: str | Path | None = None,
    *,
    open_throat: bool = True,
) -> tuple[Path, CadInfo]:
    """Write the waveguide's OCC model to a STEP file in millimetres.

    Freestanding (wall-shell) and enclosure builds export as a closed solid.
    Bare and infinite-baffle builds have no wall thickness, so they export as a
    surface body -- there is no material for a solid to enclose.

    ``open_throat`` cuts the driver membrane away so the bore runs through.
    Turning it off keeps the sewn acoustic boundary, throat plug and all.
    """

    import gmsh

    if isinstance(geometry, PointGridHornGeometry) and not geometry.closed:
        raise MesherError(
            "STEP export needs the full model; this geometry is a symmetry-reduced "
            f"sector bounded by {', '.join(geometry.symmetry_planes)}=0. Rebuild it "
            "with quadrants=1234 (closed=True) and export that."
        )

    owns_out_path = output_path is None
    if output_path is None:
        handle = tempfile.NamedTemporaryFile(
            prefix="hornlab-cad-", suffix=".step", delete=False
        )
        out_path = Path(handle.name)
        handle.close()
    else:
        out_path = Path(output_path)
        if out_path.suffix.lower() not in (".step", ".stp"):
            raise MesherError(
                f"STEP output path must end in .step or .stp, got {out_path.name!r}"
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with _GMSH_LOCK:
        initialized_here = False
        wrote = False
        staged_path: Path | None = None
        try:
            if not gmsh.isInitialized():
                gmsh.initialize(interruptible=False)
                initialized_here = True
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
            gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-8)
            gmsh.clear()
            gmsh.model.add("HornLabCad")

            # Deliberately not _acoustic_geometry(): CAD keeps the geometry as
            # designed, including fillets too small for the mesh to resolve.
            built = _dispatch_builder(geometry)
            gmsh.model.occ.synchronize()

            excluded = _role_tags(built, *_NON_MATERIAL_ROLES)
            cap_tags = sorted(_role_tags(built, _SOURCE_ROLE))
            body: CadBody = "solid" if _solid_capable(geometry) else "surface"
            throat_opened = bool(open_throat and cap_tags)
            if body == "surface" and open_throat:
                # A solid needs the cap to close its shell before cutting it
                # away; a surface body has nothing to close, so the membrane is
                # simply left out.
                excluded |= set(cap_tags)
            faces = [
                int(tag)
                for _, tag in gmsh.model.getEntities(2)
                if int(tag) not in excluded
            ]
            if not faces:
                raise MesherError("geometry build produced no surfaces to export")

            volume: int | None = None
            if body == "solid":
                shell = gmsh.model.occ.addSurfaceLoop(faces, sewing=True)
                volume = int(gmsh.model.occ.addVolume([shell]))
                gmsh.model.occ.synchronize()
                mass = float(gmsh.model.occ.getMass(3, volume))
                if mass <= _MIN_SOLID_VOLUME_MM3:
                    raise MesherError(
                        "sewing the waveguide surfaces did not produce a closed solid "
                        f"(enclosed volume {mass:.6g} mm^3); the geometry has a gap"
                    )
                if throat_opened:
                    volume = _open_throat(gmsh, volume, built, cap_tags)
                keep = _entity_closure(gmsh, [(3, volume)])
            else:
                keep = _entity_closure(gmsh, [(2, tag) for tag in faces])
            _prune_to(gmsh, keep)

            offset = float(getattr(geometry, "vertical_offset_mm", 0.0) or 0.0)
            if offset:
                # Mesh.VerticalOffset is a rigid +y placement of the finished
                # model; the mesh applies it the same way, so a CAD assembly and
                # the solved model agree on where the waveguide sits.
                gmsh.model.occ.translate(
                    gmsh.model.getEntities(3) or gmsh.model.getEntities(2),
                    0.0,
                    offset,
                    0.0,
                )
                gmsh.model.occ.synchronize()

            if volume is not None:
                volume_mm3 = float(gmsh.model.occ.getMass(3, volume))
                box = gmsh.model.getBoundingBox(3, volume)
                n_faces = len(
                    gmsh.model.getBoundary(
                        [(3, volume)], oriented=False, combined=False
                    )
                )
            else:
                volume_mm3 = None
                box = gmsh.model.getBoundingBox(-1, -1)
                n_faces = len(gmsh.model.getEntities(2))

            # Stage beside the target and swap only once the file validates, so
            # a rejected export never leaves a corrupt STEP at the caller's path.
            with tempfile.NamedTemporaryFile(
                dir=out_path.parent,
                prefix=f".{out_path.name}.",
                suffix=".step",
                delete=False,
            ) as tmp:
                staged_path = Path(tmp.name)
            gmsh.write(str(staged_path))
            text = staged_path.read_text(encoding="utf-8", errors="replace")
            _assert_step(text, body=body)
            staged_path.replace(out_path)
            staged_path = None
            wrote = True
            return out_path, CadInfo(
                path=out_path,
                body=body,
                n_faces=int(n_faces),
                volume_mm3=volume_mm3,
                bounding_box_mm=(
                    (float(box[0]), float(box[1]), float(box[2])),
                    (float(box[3]), float(box[4]), float(box[5])),
                ),
                throat_opened=throat_opened,
            )
        except MesherError:
            raise
        except Exception as exc:
            raise MesherError(f"STEP export failed: {exc}") from exc
        finally:
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)
            if owns_out_path and not wrote:
                out_path.unlink(missing_ok=True)
            if initialized_here and gmsh.isInitialized():
                gmsh.finalize()


def write_step_from_config(
    config,
    output_path: str | Path | None = None,
    *,
    open_throat: bool = True,
    full_model: bool = True,
) -> tuple[Path, CadInfo]:
    """Write a STEP file from the same config dict that drives a mesh build.

    ``full_model`` rebuilds a symmetry-reduced config over all four quadrants.
    A solve may legitimately run on a quarter model; a part cannot be a quarter
    of itself, so CAD export closes it back up by default.
    """

    from .config_builder import resolve_geometry

    if not isinstance(config, Mapping):
        raise MesherError("config must be a mapping")
    working = deepcopy(dict(config))
    if full_model:
        mesh = working.get("mesh")
        mesh = dict(mesh) if isinstance(mesh, Mapping) else {}
        mesh["quadrants"] = "1234"
        working["mesh"] = mesh
        working.pop("quadrants", None)
    resolved = resolve_geometry(working)
    return write_step(resolved.geometry, output_path, open_throat=open_throat)


__all__ = ["CadBody", "CadInfo", "write_step", "write_step_from_config"]
