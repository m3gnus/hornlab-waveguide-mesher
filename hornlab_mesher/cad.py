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

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from .datums import DEFAULT_PLANE_TOLERANCE_MM, derive_datums
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
SOURCE_INTERFACE_FEATURE = "source-interface-v1"
_SOURCE_INTERFACE_KEYS = frozenset(
    {
        "id",
        "role",
        "required",
        "default_drive_channel_id",
        "patch_policy",
        "expected_connected_components",
        "suggested_resolution_mm",
    }
)
_SOURCE_PATCH_POLICIES = frozenset({"single-connected", "explicit-disconnected"})


@dataclass(frozen=True)
class CadInfo:
    """What was written, and what shape it turned out to be.

    ``throat_opened`` reports the effective OUTCOME, not the request. An
    enclosure build always reports ``False`` even for ``open_throat=True``: the
    waveguide is a blind pocket in a solid block, so there is no bore to open
    and cutting would tunnel out through the back face. Callers that record
    what a part looks like should read this field rather than echo the flag
    they passed in.
    """

    path: Path
    body: CadBody
    n_faces: int
    volume_mm3: float | None
    bounding_box_mm: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ]
    throat_opened: bool
    units: Literal["mm"] = "mm"


@dataclass(frozen=True)
class WgLinkIdentity:
    """Caller-owned identity/provenance sections emitted without augmentation."""

    bundle: Mapping[str, Any] | None = None
    generator: Mapping[str, Any] | None = None
    design: Mapping[str, Any] | None = None
    export: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class WgLinkSourceInterface:
    """WG-authored acoustic source policy carried by ``interface.sources[]``.

    Geometry is deliberately not selected by face index or CAD body name. The
    O4 Onshape adapter materializes the linked throat sheet from the manifest's
    throat datum and diameter parameter, while this record owns the stable
    acoustic identity and meshing policy for that sheet.
    """

    id: str
    role: str
    required: bool
    default_drive_channel_id: str
    patch_policy: Literal["single-connected", "explicit-disconnected"]
    expected_connected_components: int
    suggested_resolution_mm: float


@dataclass(frozen=True)
class WgLinkInfo:
    """All products of a bundle write, including the underlying CAD result."""

    path: Path
    manifest_path: Path
    step_path: Path
    point_grid_path: Path
    manifest: dict[str, Any]
    cad_info: CadInfo


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


def _throat_opens_to_free_space(geometry: HornGeometry) -> bool:
    """Is there actually free space behind the driver membrane?

    Only a freestanding shell has any. In an enclosure build the waveguide is a
    blind pocket carved out of a solid block, so there is material behind the
    throat however the cap is treated -- the membrane is not plugging anything.

    Sweeping the cap backwards there does not open a bore, it drills a tunnel
    through the entire enclosure and out through the back face. Measured on
    Tritonia-V: a 25.4 mm hole 185 mm long, removing 93,706 mm^3 of the back of
    the cabinet. A blind horn-shaped pocket is the correct part.
    """

    mode = getattr(geometry, "build_mode", None)
    return mode is not PointGridBuildMode.ENCLOSURE


def _open_throat(
    gmsh, volume: int, built: BuiltGeometry, cap_tags: list[int]
) -> int:
    """Cut the driver membrane out of the sewn solid, opening the bore.

    The sweep depth spans the whole model deliberately: it must clear whatever
    material sits behind the cap. That is safe only where the caller has
    established there is free space behind it -- see
    ``_throat_opens_to_free_space``.
    """

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


# ISO 10303-21 clause 8 fixes the header order: file_description, then
# file_name, then file_schema. OpenCASCADE emitted them that way through 7.6,
# and since 7.7 it writes file_name first -- gmsh 4.12 and newer inherit the
# bug. Fusion and gmsh itself do not care; strict readers (CATIA among them)
# parse the header positionally and can take the file's product structure
# while dropping every shape in it, which looks like an empty tree rather than
# an error. We write through OCC, so the fix is to reorder the header text.
_HEADER_ORDER = ("FILE_DESCRIPTION", "FILE_NAME", "FILE_SCHEMA")


def _split_step_statements(block: str) -> list[str]:
    """Split a STEP section into statements, ignoring ``;`` inside strings.

    STEP single-quoted strings escape a quote by doubling it, so a quote that
    follows a quote does not end the literal.
    """

    statements: list[str] = []
    start = 0
    in_string = False
    index = 0
    while index < len(block):
        char = block[index]
        if char == "'":
            if in_string and index + 1 < len(block) and block[index + 1] == "'":
                index += 2
                continue
            in_string = not in_string
        elif char == ";" and not in_string:
            statements.append(block[start : index + 1])
            start = index + 1
        index += 1
    tail = block[start:]
    if tail.strip():
        statements.append(tail)
    return statements


def normalise_step_header(text: str) -> str:
    """Return ``text`` with its header entities in ISO 10303-21 order.

    Anything that is not one of the three mandatory entities keeps its relative
    position after them, and a header already in order is returned unchanged.
    """

    head, sep, rest = text.partition("HEADER;")
    if not sep:
        return text
    block, end_sep, tail = rest.partition("ENDSEC;")
    if not end_sep:
        return text

    statements = _split_step_statements(block)
    keyed: dict[str, str] = {}
    others: list[str] = []
    for statement in statements:
        stripped = statement.lstrip()
        for keyword in _HEADER_ORDER:
            if stripped.startswith(keyword) and keyword not in keyed:
                keyed[keyword] = statement.strip()
                break
        else:
            others.append(statement)
    if len(keyed) != len(_HEADER_ORDER):
        # Not the header we know how to reorder; leave it as OCC wrote it.
        return text
    ordered = "\n" + "\n".join(keyed[keyword] for keyword in _HEADER_ORDER) + "\n"
    trailing = "".join(others).strip()
    if trailing:
        ordered += trailing + "\n"
    return f"{head}{sep}{ordered}{end_sep}{tail}"


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
    header = text.partition("HEADER;")[2].partition("ENDSEC;")[0]
    positions = [header.find(keyword) for keyword in _HEADER_ORDER]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise MesherError(
            "STEP header is not in ISO 10303-21 order (file_description, "
            "file_name, file_schema); strict CAD readers reject it"
        )


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
            throat_opened = bool(
                open_throat and cap_tags and _throat_opens_to_free_space(geometry)
            )
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
            text = normalise_step_header(
                staged_path.read_text(encoding="utf-8", errors="replace")
            )
            _assert_step(text, body=body)
            staged_path.write_text(text, encoding="utf-8")
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


def _realized_bundle_geometry(geometry: PointGridHornGeometry) -> BuiltGeometry:
    """Build the non-OCC realized metadata needed by a standalone writer.

    Normal mesh/export pipelines should pass their actual ``BuiltGeometry``.
    This fallback uses the enclosure builder's own pure bounds function, so it
    applies the identical rounding and clamps without constructing OCC twice.
    """

    bounds = None
    if geometry.build_mode is PointGridBuildMode.ENCLOSURE:
        if geometry.enclosure is None:
            raise MesherError("enclosure build has no HornEnclosure metadata")
        from .builders.enclosure import enclosure_box_bounds

        bounds = enclosure_box_bounds(
            np.asarray(geometry.inner_points, dtype=np.float64),
            geometry.enclosure,
            closed=geometry.closed,
            symmetry_planes=geometry.symmetry_planes,
        )
    points = np.asarray(geometry.inner_points, dtype=np.float64)
    return BuiltGeometry(
        surface_groups={},
        axial_bounds_mm=(
            float(np.mean(points[:, 0, 2])),
            float(np.mean(points[:, -1, 2])),
        ),
        source_axis="z",
        enclosure_bounds=bounds,
        symmetry_snap_axes=() if geometry.closed else tuple(geometry.symmetry_planes),
    )


def _validate_realized_bundle_geometry(
    geometry: PointGridHornGeometry, built: BuiltGeometry
) -> None:
    """Reject enclosure metadata that cannot describe this point grid's solid."""

    if geometry.build_mode is not PointGridBuildMode.ENCLOSURE:
        return
    if geometry.enclosure is None or built.enclosure_bounds is None:
        raise MesherError("enclosure export requires realized enclosure_bounds")
    from .builders.enclosure import enclosure_box_bounds

    expected = enclosure_box_bounds(
        np.asarray(geometry.inner_points, dtype=np.float64),
        geometry.enclosure,
        closed=geometry.closed,
        symmetry_planes=geometry.symmetry_planes,
    )
    for name, expected_value in expected.items():
        actual_value = built.enclosure_bounds.get(name)
        if actual_value is None or not np.isclose(
            float(actual_value), float(expected_value), rtol=0.0, atol=1.0e-9
        ):
            raise MesherError(
                "built_geometry enclosure_bounds do not match the realized "
                f"point-grid enclosure: {name} is {actual_value!r}, expected "
                f"{expected_value!r}"
            )


def _points_are_planar(points: np.ndarray) -> bool:
    samples = np.asarray(points, dtype=np.float64)
    origin = np.mean(samples, axis=0)
    _u, _s, vh = np.linalg.svd(samples - origin, full_matrices=False)
    errors = np.abs((samples - origin) @ vh[-1])
    return float(np.max(errors)) <= DEFAULT_PLANE_TOLERANCE_MM


def _point_grid_payload(
    geometry: PointGridHornGeometry,
    *,
    check_points: object | None,
) -> dict[str, Any]:
    inner = np.asarray(geometry.inner_points, dtype=np.float64)
    if inner.ndim != 3 or inner.shape[2] != 3 or inner.shape[0] < 3:
        raise MesherError("wglink requires inner_points shaped (n_phi, n_length, 3)")
    if not np.isfinite(inner).all():
        raise MesherError("wglink point grid contains non-finite inner points")
    outer = None
    if geometry.outer_points is not None:
        outer = np.asarray(geometry.outer_points, dtype=np.float64)
        if outer.shape != inner.shape or not np.isfinite(outer).all():
            raise MesherError("outer_points must be finite and match inner_points")

    ring_z: list[float] = []
    ring_planar: list[bool] = []
    for station in range(inner.shape[1]):
        z = inner[:, station, 2]
        ring_planar.append(_points_are_planar(inner[:, station, :]))
        ring_z.append(0.5 * (float(np.min(z)) + float(np.max(z))))

    checks = np.asarray([] if check_points is None else check_points, dtype=np.float64)
    if checks.size == 0:
        checks = np.empty((0, 3), dtype=np.float64)
    try:
        checks = checks.reshape((-1, 3))
    except ValueError as exc:
        raise MesherError("check_points must be an array of xyz triples") from exc
    if not np.isfinite(checks).all():
        raise MesherError("check_points contains non-finite coordinates")

    # ONE frame per bundle (plan D1). The STEP body and every datum are in the
    # placed link-local frame (the body sits at y = vertical_offset), so the
    # grid ships placed too -- a consumer that lofts these points and mates
    # against the datums must not need to know about the offset at all. The
    # stored geometry keeps its unshifted grid; only the payload is placed.
    offset = float(geometry.vertical_offset_mm)
    if offset:
        shift = np.asarray([0.0, offset, 0.0], dtype=np.float64)
        inner = inner + shift
        if outer is not None:
            outer = outer + shift
        if checks.size:
            checks = checks + shift

    return {
        "units": "mm",
        "build_mode": geometry.build_mode.value,
        "frame": "link-local",
        # Informational: already applied to every coordinate in this payload.
        "vertical_offset_mm": float(geometry.vertical_offset_mm),
        "wall_thickness_mm": float(geometry.wall_thickness_mm),
        "n_phi": int(inner.shape[0]),
        # Kept compatible with wg_profile_points.py: this is the station count,
        # despite the historical field name.
        "n_length": int(inner.shape[1]),
        "closed": bool(geometry.closed),
        "all_rings_planar": all(ring_planar),
        "ring_planar": ring_planar,
        "ring_z_mm": ring_z,
        "inner_points": inner.tolist(),
        "has_outer_points": outer is not None,
        "outer_points": outer.tolist() if outer is not None else None,
        "check_points": checks.tolist(),
    }


def _parameter_table(
    geometry: PointGridHornGeometry,
    built: BuiltGeometry,
    *,
    instance_slug: str,
    informational_parameters: Mapping[str, float] | None,
) -> list[dict[str, Any]]:
    slug = re.sub(r"[^a-z0-9_]+", "_", instance_slug.strip().lower()).strip("_")
    if not slug:
        raise MesherError("instance_slug must contain at least one letter or digit")
    prefix = f"wg_{slug}_"
    inner = np.asarray(geometry.inner_points, dtype=np.float64)
    throat = inner[:, 0, :]
    center = np.mean(throat, axis=0)
    radii = np.linalg.norm(throat[:, :2] - center[:2], axis=1)
    positive = radii[radii > 1.0e-9]
    throat_dia = 2.0 * (float(np.mean(positive)) if len(positive) else 0.0)
    mouth = inner[:, -1, :]
    z_front = (
        float(built.enclosure_bounds["z_front"])
        if built.enclosure_bounds is not None
        else float(np.max(mouth[:, 2]))
    )
    z_throat = float(np.mean(throat[:, 2]))
    values = {
        "throat_dia": throat_dia,
        "mouth_w": float(np.ptp(mouth[:, 0])),
        "mouth_h": float(np.ptp(mouth[:, 1])),
        "depth": z_front - z_throat,
        "wall_t": float(geometry.wall_thickness_mm),
        "vertical_offset": float(geometry.vertical_offset_mm),
    }
    if geometry.build_mode is PointGridBuildMode.ENCLOSURE:
        bounds = built.enclosure_bounds
        if bounds is None:
            raise MesherError("enclosure parameters require realized enclosure_bounds")
        # enclosure_bounds is unplaced; absolute placement shares the datums'
        # link-local frame, with vertical_offset_mm applied on y only.
        values.update(
            {
                "enc_w": float(bounds["bx1"]) - float(bounds["bx0"]),
                "enc_h": float(bounds["by1"]) - float(bounds["by0"]),
                "enc_depth": float(bounds["enc_depth"]),
                "enc_edge": float(bounds["clamped_edge"]),
                "enc_x0": float(bounds["bx0"]),
                "enc_y0": float(bounds["by0"]) + float(geometry.vertical_offset_mm),
                "enc_z_front": float(bounds["z_front"]),
            }
        )
    table = [
        {"name": prefix + name, "value": value, "unit": "mm", "role": "interface"}
        for name, value in values.items()
    ]
    for name, value in (informational_parameters or {}).items():
        table.append(
            {
                "name": prefix + str(name),
                "value": float(value),
                "role": "informational",
            }
        )
    return table


def _identity_sections(
    identity: WgLinkIdentity | Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    if identity is None:
        return {}
    if isinstance(identity, WgLinkIdentity):
        sections = {
            "bundle": identity.bundle,
            "generator": identity.generator,
            "design": identity.design,
            "export": identity.export,
        }
    elif isinstance(identity, Mapping):
        sections = dict(identity)
    else:
        raise MesherError("identity must be WgLinkIdentity, a mapping, or None")
    unknown = sorted(set(sections).difference({"bundle", "generator", "design", "export"}))
    if unknown:
        raise MesherError("unknown wglink identity section(s): " + ", ".join(unknown))
    return {
        key: deepcopy(dict(value))
        for key, value in sections.items()
        if value is not None
    }


def _source_interface_table(
    sources: Sequence[WgLinkSourceInterface | Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Validate and normalize the additive ``source-interface-v1`` table."""

    if sources is None:
        return []
    if isinstance(sources, (str, bytes)):
        raise MesherError("interface_sources must be a sequence of source records")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(sources):
        if isinstance(source, WgLinkSourceInterface):
            record = {
                "id": source.id,
                "role": source.role,
                "required": source.required,
                "default_drive_channel_id": source.default_drive_channel_id,
                "patch_policy": source.patch_policy,
                "expected_connected_components": source.expected_connected_components,
                "suggested_resolution_mm": source.suggested_resolution_mm,
            }
        elif isinstance(source, Mapping):
            record = dict(source)
        else:
            raise MesherError(f"interface.sources[{index}] must be an object")
        unknown = sorted(set(record).difference(_SOURCE_INTERFACE_KEYS))
        missing = sorted(_SOURCE_INTERFACE_KEYS.difference(record))
        if unknown or missing:
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unknown:
                details.append("unknown " + ", ".join(unknown))
            raise MesherError(f"interface.sources[{index}] is invalid: {'; '.join(details)}")
        for key in ("id", "role", "default_drive_channel_id"):
            value = record[key]
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise MesherError(
                    f"interface.sources[{index}].{key} must be a non-empty trimmed string"
                )
        source_id = str(record["id"])
        if source_id in seen:
            raise MesherError(f"interface.sources has duplicate id {source_id!r}")
        seen.add(source_id)
        if not isinstance(record["required"], bool):
            raise MesherError(f"interface.sources[{index}].required must be boolean")
        policy = record["patch_policy"]
        if policy not in _SOURCE_PATCH_POLICIES:
            raise MesherError(
                f"interface.sources[{index}].patch_policy must be single-connected "
                "or explicit-disconnected"
            )
        components = record["expected_connected_components"]
        if isinstance(components, bool) or not isinstance(components, int) or components < 1:
            raise MesherError(
                f"interface.sources[{index}].expected_connected_components "
                "must be an integer >= 1"
            )
        if policy == "single-connected" and components != 1:
            raise MesherError(
                f"interface.sources[{index}].expected_connected_components must be 1 "
                "for single-connected"
            )
        resolution = record["suggested_resolution_mm"]
        if isinstance(resolution, bool) or not isinstance(resolution, (int, float)):
            raise MesherError(
                f"interface.sources[{index}].suggested_resolution_mm must be positive"
            )
        resolution = float(resolution)
        if not math.isfinite(resolution) or resolution <= 0.0:
            raise MesherError(
                f"interface.sources[{index}].suggested_resolution_mm must be positive"
            )
        result.append(
            {
                "id": source_id,
                "role": str(record["role"]),
                "required": bool(record["required"]),
                "default_drive_channel_id": str(record["default_drive_channel_id"]),
                "patch_policy": str(policy),
                "expected_connected_components": int(components),
                "suggested_resolution_mm": resolution,
            }
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _default_generator() -> dict[str, Any]:
    try:
        app_version = version("hornlab-waveguide-mesher")
    except PackageNotFoundError:
        app_version = "unknown"
    return {
        "app": "hornlab-waveguide-mesher",
        "app_version": app_version,
        "datum_schema": 1,
    }


def _fsync_directory(path: Path) -> None:
    """Persist directory entries on POSIX; other platforms lack this primitive."""

    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_staged_bundle(staging: Path) -> None:
    for child in staging.iterdir():
        if child.is_file():
            # Windows' CRT rejects ``_commit`` (the implementation behind
            # ``os.fsync``) for a read-only descriptor with ``EBADF``.  Every
            # staged member was just written by us, so opening it read/write
            # preserves the same durability barrier on every platform.
            with child.open("rb+") as stream:
                os.fsync(stream.fileno())
    _fsync_directory(staging)


_BUNDLE_THREAD_LOCKS: dict[str, threading.Lock] = {}
_BUNDLE_THREAD_LOCKS_GUARD = threading.Lock()
_TRANSACTION_SCHEMA = 1
_PRIVATE_STATE_SCHEMA = 1


def _windows_current_user_sid():
    import ctypes
    from ctypes import wintypes

    class SidAndAttributes(ctypes.Structure):
        _fields_ = [("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD)]

    class TokenUser(ctypes.Structure):
        _fields_ = [("User", SidAndAttributes)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    token_query = 0x0008
    token_user = 1
    process = kernel32.GetCurrentProcess()
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(process, token_query, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        needed = wintypes.DWORD()
        advapi32.GetTokenInformation(token, token_user, None, 0, ctypes.byref(needed))
        buffer = ctypes.create_string_buffer(needed.value)
        if not advapi32.GetTokenInformation(
            token,
            token_user,
            buffer,
            needed.value,
            ctypes.byref(needed),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return ctypes.cast(buffer, ctypes.POINTER(TokenUser)).contents.User.Sid, buffer
    finally:
        kernel32.CloseHandle(token)


def _windows_verify_owner_only_dacl(path: Path, *, directory: bool) -> None:
    import ctypes
    from ctypes import wintypes

    owner_security_information = 0x00000001
    dacl_security_information = 0x00000004
    se_file_object = 1
    access_allowed_ace_type = 0
    file_all_access = 0x001F01FF

    class Acl(ctypes.Structure):
        _fields_ = [
            ("AclRevision", ctypes.c_ubyte),
            ("Sbz1", ctypes.c_ubyte),
            ("AclSize", wintypes.WORD),
            ("AceCount", wintypes.WORD),
            ("Sbz2", wintypes.WORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi32.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
    ]
    advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi32.EqualSid.argtypes = [wintypes.LPVOID, wintypes.LPVOID]
    advapi32.EqualSid.restype = wintypes.BOOL
    advapi32.GetAce.argtypes = [
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
    ]
    advapi32.GetAce.restype = wintypes.BOOL
    advapi32.GetSecurityDescriptorControl.argtypes = [
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.WORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetSecurityDescriptorControl.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL

    owner = wintypes.LPVOID()
    dacl = wintypes.LPVOID()
    descriptor = wintypes.LPVOID()
    result = advapi32.GetNamedSecurityInfoW(
        str(path),
        se_file_object,
        owner_security_information | dacl_security_information,
        ctypes.byref(owner),
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(descriptor),
    )
    if result:
        raise ctypes.WinError(result)
    current_sid, sid_buffer = _windows_current_user_sid()
    try:
        if not advapi32.EqualSid(owner, current_sid):
            raise MesherError(f"wglink private state has a different owner: {path}")
        if not dacl:
            raise MesherError(f"wglink private state has no protected DACL: {path}")
        control = wintypes.WORD()
        revision = wintypes.DWORD()
        if not advapi32.GetSecurityDescriptorControl(
            descriptor, ctypes.byref(control), ctypes.byref(revision)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if not int(control.value) & 0x1000:  # SE_DACL_PROTECTED
            raise MesherError(f"wglink private state has no protected DACL: {path}")
        acl = ctypes.cast(dacl, ctypes.POINTER(Acl)).contents
        if int(acl.AceCount) != 1:
            raise MesherError(f"wglink private state DACL is not owner-only: {path}")
        ace = wintypes.LPVOID()
        if not advapi32.GetAce(dacl, 0, ctypes.byref(ace)):
            raise ctypes.WinError(ctypes.get_last_error())
        ace_address = int(ace.value)
        ace_type = ctypes.c_ubyte.from_address(ace_address).value
        ace_flags = ctypes.c_ubyte.from_address(ace_address + 1).value
        access_mask = wintypes.DWORD.from_address(ace_address + 4).value
        ace_sid = wintypes.LPVOID(ace_address + 8)
        required_flags = 0x03 if directory else 0x00
        if (
            ace_type != access_allowed_ace_type
            or access_mask != file_all_access
            or ace_flags & 0x03 != required_flags
            or not advapi32.EqualSid(ace_sid, current_sid)
        ):
            raise MesherError(f"wglink private state DACL is not owner-only: {path}")
    finally:
        del sid_buffer
        kernel32.LocalFree(descriptor)


def _windows_apply_owner_only_dacl(path: Path, *, directory: bool) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi32.ConvertSidToStringSidW.argtypes = [
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = (
        wintypes.BOOL
    )
    advapi32.GetSecurityDescriptorDacl.argtypes = [
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.BOOL),
    ]
    advapi32.GetSecurityDescriptorDacl.restype = wintypes.BOOL
    advapi32.SetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.LPVOID,
    ]
    advapi32.SetNamedSecurityInfoW.restype = wintypes.DWORD
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL

    dacl_security_information = 0x00000004
    protected_dacl_security_information = 0x80000000
    se_file_object = 1
    sddl_revision_1 = 1
    current_sid, sid_buffer = _windows_current_user_sid()
    sid_string = wintypes.LPWSTR()
    if not advapi32.ConvertSidToStringSidW(current_sid, ctypes.byref(sid_string)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        inheritance = "OICI" if directory else ""
        sddl = f"D:P(A;{inheritance};FA;;;{sid_string.value})"
    finally:
        kernel32.LocalFree(ctypes.cast(sid_string, wintypes.HLOCAL))
        del sid_buffer
    descriptor = wintypes.LPVOID()
    descriptor_size = wintypes.DWORD()
    if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        sddl,
        sddl_revision_1,
        ctypes.byref(descriptor),
        ctypes.byref(descriptor_size),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        present = wintypes.BOOL()
        defaulted = wintypes.BOOL()
        dacl = wintypes.LPVOID()
        if not advapi32.GetSecurityDescriptorDacl(
            descriptor,
            ctypes.byref(present),
            ctypes.byref(dacl),
            ctypes.byref(defaulted),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if not present or not dacl:
            raise MesherError("could not construct owner-only Windows DACL")
        result = advapi32.SetNamedSecurityInfoW(
            str(path),
            se_file_object,
            dacl_security_information | protected_dacl_security_information,
            None,
            None,
            dacl,
            None,
        )
        if result:
            raise ctypes.WinError(result)
    finally:
        kernel32.LocalFree(descriptor)
    _windows_verify_owner_only_dacl(path, directory=directory)


def _validate_private_directory(path: Path) -> None:
    metadata = os.lstat(path)
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    file_attributes = int(getattr(metadata, "st_file_attributes", 0))
    if path.is_symlink() or (reparse_flag and file_attributes & reparse_flag):
        raise MesherError(f"wglink private state is a reparse point: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise MesherError(f"wglink private state is not a directory: {path}")
    if os.name == "posix":
        if int(metadata.st_uid) != int(os.getuid()):
            raise MesherError(f"wglink private state has a different owner: {path}")
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            raise MesherError(f"wglink private state must have mode 0700: {path}")
    elif sys.platform == "win32":
        _windows_verify_owner_only_dacl(path, directory=True)


def _ensure_private_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        if os.name == "posix":
            _validate_private_directory(path)
            return
        metadata = os.lstat(path)
        reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or path.is_symlink()
            or reparse_flag
            and int(getattr(metadata, "st_file_attributes", 0)) & reparse_flag
        ):
            raise MesherError(f"wglink private state is not a plain directory: {path}")
    if os.name == "posix":
        path.chmod(0o700)
    elif sys.platform == "win32":
        _windows_apply_owner_only_dacl(path, directory=True)
    _validate_private_directory(path)


def _private_lock_root() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
        vendor_root = base / "HornLab"
        _ensure_private_directory(vendor_root)
        app_root = vendor_root / "WaveguideMesher"
        _ensure_private_directory(app_root)
        root = app_root / "lock-state-v1"
    else:
        root = Path(tempfile.gettempdir()) / f"hornlab-waveguide-mesher-{os.getuid()}"
    _ensure_private_directory(root)
    return root


def _target_lock_key(target: Path) -> str:
    normalized = os.path.normcase(os.path.realpath(os.path.abspath(target)))
    return hashlib.sha256(os.fsencode(normalized)).hexdigest()


def _publish_lock_path(target: Path) -> Path:
    return _private_lock_root() / f"{_target_lock_key(target)}.lock"


def _private_state_path(target: Path) -> Path:
    return _private_lock_root() / f"{_target_lock_key(target)}.state.json"


def _transaction_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.publish.transaction")


def _transaction_record_path(target: Path) -> Path:
    return _transaction_path(target) / "record.json"


def _private_file_flags() -> int:
    flags = os.O_RDWR
    for name in ("O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= int(getattr(os, name, 0))
    return flags


def _validate_private_file_stat(path: Path, metadata: os.stat_result) -> None:
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    file_attributes = int(getattr(metadata, "st_file_attributes", 0))
    if reparse_flag and file_attributes & reparse_flag:
        raise MesherError(f"wglink coordination file is a reparse point: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise MesherError(f"wglink coordination path is not a regular file: {path}")
    if int(metadata.st_nlink) != 1:
        raise MesherError(
            f"wglink coordination file must have exactly one link: {path}"
        )
    if hasattr(os, "getuid") and int(metadata.st_uid) != int(os.getuid()):
        raise MesherError(f"wglink coordination file has a different owner: {path}")
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600:
        raise MesherError(f"wglink coordination file must have mode 0600: {path}")
    if sys.platform == "win32":
        _windows_verify_owner_only_dacl(path, directory=False)


def _validate_private_file_identity(path: Path, descriptor: int) -> os.stat_result:
    handle_metadata = os.fstat(descriptor)
    _validate_private_file_stat(path, handle_metadata)
    try:
        path_metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise MesherError(
            f"wglink coordination file changed while open: {path}"
        ) from exc
    _validate_private_file_stat(path, path_metadata)
    handle_identity = (int(handle_metadata.st_dev), int(handle_metadata.st_ino))
    path_identity = (int(path_metadata.st_dev), int(path_metadata.st_ino))
    if handle_identity != path_identity or int(handle_metadata.st_nlink) != 1:
        raise MesherError(f"wglink coordination file changed while open: {path}")
    return handle_metadata


def _create_private_file(path: Path) -> int:
    flags = _private_file_flags() | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        raise
    except OSError as exc:
        raise MesherError(
            f"could not create wglink coordination file {path}: {exc}"
        ) from exc
    try:
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        elif sys.platform == "win32":
            _windows_apply_owner_only_dacl(path, directory=False)
        _validate_private_file_identity(path, descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_existing_private_file(path: Path) -> int:
    try:
        path_metadata = os.lstat(path)
    except FileNotFoundError:
        raise
    _validate_private_file_stat(path, path_metadata)
    try:
        descriptor = os.open(path, _private_file_flags())
    except OSError as exc:
        raise MesherError(
            f"could not open wglink coordination file {path}: {exc}"
        ) from exc
    try:
        _validate_private_file_identity(path, descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_or_create_private_file(path: Path) -> int:
    try:
        return _create_private_file(path)
    except FileExistsError:
        return _open_existing_private_file(path)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while persisting wglink coordination state")
        view = view[written:]


def _read_private_payload(path: Path, *, limit: int) -> bytes:
    descriptor = _open_existing_private_file(path)
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        payload = os.read(descriptor, limit + 1)
        if len(payload) > limit:
            raise MesherError(f"wglink coordination file is too large: {path}")
        _validate_private_file_identity(path, descriptor)
        return payload
    finally:
        os.close(descriptor)


def _atomic_write_private_payload(path: Path, payload: bytes, *, replace: bool) -> None:
    token = secrets.token_hex(16)
    temporary = path.with_name(f".{path.name}.{token}.tmp")
    descriptor = _create_private_file(temporary)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        _validate_private_file_identity(temporary, descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    try:
        if replace:
            if path.exists() or path.is_symlink():
                existing = _open_existing_private_file(path)
                os.close(existing)
        elif path.exists() or path.is_symlink():
            raise MesherError(f"wglink coordination file already exists: {path}")
        temporary.replace(path)
        descriptor = _open_existing_private_file(path)
        os.close(descriptor)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _private_state_temps(state_path: Path) -> list[Path]:
    pattern = re.compile(rf"^\.{re.escape(state_path.name)}\.[0-9a-f]{{32}}\.tmp$")
    return [
        child for child in state_path.parent.iterdir() if pattern.fullmatch(child.name)
    ]


def _cleanup_private_state_temps(state_path: Path) -> None:
    for temporary in _private_state_temps(state_path):
        descriptor = _open_existing_private_file(temporary)
        os.close(descriptor)
        temporary.unlink()
    _fsync_directory(state_path.parent)


def _validate_private_state(state_path: Path, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "secret",
        "active_token",
    }:
        raise MesherError(f"wglink private state is invalid: {state_path}")
    secret = value.get("secret")
    active_token = value.get("active_token")
    if (
        value.get("schema") != _PRIVATE_STATE_SCHEMA
        or not isinstance(secret, str)
        or re.fullmatch(r"[0-9a-f]{64}", secret) is None
        or active_token is not None
        and (
            not isinstance(active_token, str)
            or re.fullmatch(r"[0-9a-f]{32}", active_token) is None
        )
    ):
        raise MesherError(f"wglink private state is invalid: {state_path}")
    return value


def _write_private_state(
    target: Path, state: Mapping[str, Any], *, replace: bool
) -> None:
    state_path = _private_state_path(target)
    validated = _validate_private_state(state_path, dict(state))
    payload = json.dumps(validated, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    _atomic_write_private_payload(state_path, payload, replace=replace)


def _load_or_create_private_state(target: Path) -> dict[str, Any]:
    state_path = _private_state_path(target)
    _cleanup_private_state_temps(state_path)
    try:
        payload = _read_private_payload(state_path, limit=4096)
    except FileNotFoundError:
        state = {
            "schema": _PRIVATE_STATE_SCHEMA,
            "secret": secrets.token_hex(32),
            "active_token": None,
        }
        _write_private_state(target, state, replace=False)
        return state
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MesherError(f"wglink private state is invalid: {state_path}") from exc
    return _validate_private_state(state_path, value)


def _set_active_transaction(
    target: Path, state: dict[str, Any], token: str | None
) -> None:
    updated = dict(state)
    updated["active_token"] = token
    _write_private_state(target, updated, replace=True)
    state.clear()
    state.update(updated)


def _lock_file(descriptor: int) -> None:
    if sys.platform == "win32":
        import errno
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                time.sleep(0.05)

    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_EX)


def _unlock_file(descriptor: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextmanager
def _bundle_publish_lock(target: Path):
    """Serialize same-account publishers through stable private state.

    A fixed, owner-checked claim beside the target separately makes publishers
    from other OS accounts fail safely instead of running concurrent renames.
    """

    lock_path = _publish_lock_path(target)
    lock_key = str(lock_path.absolute())
    with _BUNDLE_THREAD_LOCKS_GUARD:
        thread_lock = _BUNDLE_THREAD_LOCKS.setdefault(lock_key, threading.Lock())
    with thread_lock:
        descriptor = _open_or_create_private_file(lock_path)
        try:
            _lock_file(descriptor)
            try:
                metadata = _validate_private_file_identity(lock_path, descriptor)
                os.lseek(descriptor, 0, os.SEEK_SET)
                marker = os.read(descriptor, 2)
                if metadata.st_size == 0:
                    os.ftruncate(descriptor, 0)
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    _write_all(descriptor, b"\0")
                    os.fsync(descriptor)
                    _validate_private_file_identity(lock_path, descriptor)
                elif metadata.st_size != 1 or marker != b"\0":
                    raise MesherError(
                        f"wglink publication lock has invalid contents: {lock_path}"
                    )
                state = _load_or_create_private_state(target)
                try:
                    yield state
                finally:
                    _validate_private_file_identity(lock_path, descriptor)
            finally:
                _unlock_file(descriptor)
        finally:
            os.close(descriptor)


def _remove_transaction_directory(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_dir():
        raise MesherError(f"wglink replacement state is not a directory: {path}")
    shutil.rmtree(path)


def _transaction_mac(lock_secret: bytes, record: Mapping[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(lock_secret, payload, hashlib.sha256).hexdigest()


def _create_transaction_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        raise
    if os.name == "posix":
        path.chmod(0o700)
    elif sys.platform == "win32":
        _windows_apply_owner_only_dacl(path, directory=True)
    _validate_private_directory(path)
    _fsync_directory(path.parent)


def _write_transaction_record(
    target: Path,
    staging: Path,
    backup: Path,
    token: str,
    state: Mapping[str, Any],
) -> None:
    transaction_path = _transaction_path(target)
    record_path = _transaction_record_path(target)
    record = {
        "schema": _TRANSACTION_SCHEMA,
        "target": target.name,
        "token": token,
        "staging": staging.name,
        "backup": backup.name,
    }
    record["mac"] = _transaction_mac(bytes.fromhex(state["secret"]), record)
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    try:
        _create_transaction_directory(transaction_path)
    except FileExistsError as exc:
        raise MesherError(
            f"wglink transaction claim already exists: {transaction_path}"
        ) from exc
    temporary = transaction_path / f".record.{token}.tmp"
    descriptor = _create_private_file(temporary)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        _validate_private_file_identity(temporary, descriptor)
    finally:
        os.close(descriptor)
    temporary.replace(record_path)
    descriptor = _open_existing_private_file(record_path)
    os.close(descriptor)
    _fsync_directory(transaction_path)
    _fsync_directory(target.parent)


def _read_transaction_record(
    target: Path, state: Mapping[str, Any]
) -> tuple[dict[str, Any], Path, Path] | None:
    transaction_path = _transaction_path(target)
    if not transaction_path.exists() and not transaction_path.is_symlink():
        return None
    _validate_private_directory(transaction_path)
    record_path = _transaction_record_path(target)
    if not record_path.exists() and not record_path.is_symlink():
        raise MesherError(f"wglink transaction record is incomplete: {record_path}")
    payload = _read_private_payload(record_path, limit=4096)
    try:
        record = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MesherError(
            f"wglink transaction record is invalid: {record_path}"
        ) from exc
    if not isinstance(record, dict) or set(record) != {
        "schema",
        "target",
        "token",
        "staging",
        "backup",
        "mac",
    }:
        raise MesherError(f"wglink transaction record is invalid: {record_path}")
    token = record.get("token")
    staging_name = record.get("staging")
    provided_mac = record.get("mac")
    signed_record = {key: value for key, value in record.items() if key != "mac"}
    expected_mac = _transaction_mac(bytes.fromhex(state["secret"]), signed_record)
    expected_backup = (
        f".{target.name}.publish.{token}.previous" if isinstance(token, str) else None
    )
    if (
        record.get("schema") != _TRANSACTION_SCHEMA
        or record.get("target") != target.name
        or not isinstance(token, str)
        or re.fullmatch(r"[0-9a-f]{32}", token) is None
        or not isinstance(staging_name, str)
        or Path(staging_name).name != staging_name
        or not staging_name.startswith(f".{target.name}.")
        or record.get("backup") != expected_backup
        or not isinstance(provided_mac, str)
        or not hmac.compare_digest(provided_mac, expected_mac)
    ):
        raise MesherError(
            f"wglink transaction record is invalid or unauthenticated: {record_path}"
        )
    return record, target.parent / staging_name, target.parent / expected_backup


def _remove_transaction_record(target: Path) -> None:
    transaction_path = _transaction_path(target)
    _validate_private_directory(transaction_path)
    record_path = _transaction_record_path(target)
    extras = [child for child in transaction_path.iterdir() if child != record_path]
    if extras:
        raise MesherError(
            "wglink transaction claim contains unexpected state; cleanup was refused"
        )
    descriptor = _open_existing_private_file(record_path)
    try:
        _validate_private_file_identity(record_path, descriptor)
    finally:
        os.close(descriptor)
    record_path.unlink()
    transaction_path.rmdir()
    _fsync_directory(target.parent)


def _unowned_backup_candidates(target: Path) -> list[Path]:
    prefix = f".{target.name}."
    return [
        child
        for child in target.parent.iterdir()
        if child.name.startswith(prefix) and child.name.endswith(".previous")
    ]


def _recover_directory_replacement(target: Path, state: dict[str, Any]) -> None:
    """Recover only the exact replacement named by an owned record."""

    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise MesherError(f"wglink output is not a directory: {target}")
    transaction_path = _transaction_path(target)
    if transaction_path.exists() or transaction_path.is_symlink():
        _validate_private_directory(transaction_path)
        if not _transaction_record_path(target).exists():
            if state["active_token"] is None:
                raise MesherError(
                    "wglink transaction state is incomplete or unowned; "
                    "recovery was refused"
                )
            _remove_transaction_directory(transaction_path)
            _fsync_directory(target.parent)
            _set_active_transaction(target, state, None)
            return
    transaction = _read_transaction_record(target, state)
    if transaction is None:
        if state["active_token"] is not None:
            _set_active_transaction(target, state, None)
        if not target.exists() and _unowned_backup_candidates(target):
            raise MesherError(
                "unowned wglink replacement state exists while the live bundle "
                "is missing; recovery was refused"
            )
        return
    record, staging, backup = transaction
    if not hmac.compare_digest(record["token"], state["active_token"] or ""):
        raise MesherError(
            "wglink transaction record is stale or replayed; recovery was refused"
        )
    staging_exists = staging.exists() or staging.is_symlink()
    backup_exists = backup.exists() or backup.is_symlink()
    if staging_exists and (staging.is_symlink() or not staging.is_dir()):
        raise MesherError(f"wglink replacement staging is not a directory: {staging}")
    if backup_exists and (backup.is_symlink() or not backup.is_dir()):
        raise MesherError(f"wglink replacement backup is not a directory: {backup}")

    if not target.exists():
        if not backup_exists:
            raise MesherError(
                "wglink transaction state is ambiguous; recovery was refused"
            )
        if staging_exists:
            _remove_transaction_directory(staging)
        backup.replace(target)
        _fsync_directory(target.parent)
    elif staging_exists and backup_exists:
        raise MesherError("wglink transaction state is ambiguous; recovery was refused")
    elif staging_exists:
        _remove_transaction_directory(staging)
    elif backup_exists:
        _remove_transaction_directory(backup)

    _remove_transaction_record(target)
    _set_active_transaction(target, state, None)


def _replace_directories_with_rollback(
    staging: Path, target: Path, backup: Path
) -> None:
    """Replace a directory while holding its publication lock and record."""

    if backup.exists() or backup.is_symlink():
        raise MesherError(f"wglink replacement backup already exists: {backup}")
    target.replace(backup)
    try:
        staging.replace(target)
    except BaseException as publish_error:
        try:
            backup.replace(target)
        except BaseException as restore_error:
            raise MesherError(
                "could not publish the staged wglink bundle and could not "
                f"restore the previous generation: {restore_error}"
            ) from publish_error
        raise


def _publish_bundle_without_exchange(staging: Path, target: Path) -> None:
    """Publish on Windows with serialized, crash-recoverable renames."""

    with _bundle_publish_lock(target) as state:
        _recover_directory_replacement(target, state)
        if target.exists():
            token = secrets.token_hex(16)
            backup = target.with_name(f".{target.name}.publish.{token}.previous")
            _set_active_transaction(target, state, token)
            try:
                _write_transaction_record(target, staging, backup, token, state)
                _replace_directories_with_rollback(staging, target, backup)
            except BaseException:
                try:
                    _recover_directory_replacement(target, state)
                finally:
                    if state["active_token"] is not None:
                        _set_active_transaction(target, state, None)
                raise
            _fsync_directory(target.parent)
            _recover_directory_replacement(target, state)
        else:
            # Even a first publication takes the fixed output-side claim.  It
            # closes the otherwise uncoordinated two-user race while keeping
            # all secret/authentication material in private per-user state.
            token = secrets.token_hex(16)
            _set_active_transaction(target, state, token)
            try:
                _create_transaction_directory(_transaction_path(target))
                staging.replace(target)
                _fsync_directory(target.parent)
                _remove_transaction_directory(_transaction_path(target))
                _fsync_directory(target.parent)
                _set_active_transaction(target, state, None)
            except BaseException:
                try:
                    _recover_directory_replacement(target, state)
                finally:
                    if state["active_token"] is not None:
                        _set_active_transaction(target, state, None)
                raise


def _atomic_exchange_directories(left: Path, right: Path) -> None:
    """Atomically exchange two directories on POSIX."""

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    left_bytes = os.fsencode(left)
    right_bytes = os.fsencode(right)
    if hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-2, left_bytes, -2, right_bytes, 2)  # AT_FDCWD, RENAME_SWAP
    elif hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            -100, left_bytes, -100, right_bytes, 2
        )  # AT_FDCWD, RENAME_EXCHANGE
    else:
        raise MesherError(
            "atomic replacement of an existing wglink directory is unavailable; "
            "the live bundle was left unchanged"
        )
    if result != 0:
        error = ctypes.get_errno()
        raise MesherError(
            "could not atomically replace the existing wglink bundle: "
            f"{os.strerror(error)}"
        )


def _publish_bundle(staging: Path, target: Path) -> None:
    """Publish a durable generation, atomically swapping a live generation."""

    _sync_staged_bundle(staging)
    if os.name != "posix":
        _publish_bundle_without_exchange(staging, target)
        return
    if target.exists():
        if target.is_symlink() or not target.is_dir():
            raise MesherError(f"wglink output is not a directory: {target}")
        _atomic_exchange_directories(staging, target)
        _fsync_directory(target.parent)
        shutil.rmtree(staging, ignore_errors=True)
        return
    try:
        staging.replace(target)
    except OSError:
        # Another writer may have published after the existence check. Exchange
        # only when the raced-in endpoint is a real directory.
        if target.is_symlink() or not target.is_dir():
            raise
        _atomic_exchange_directories(staging, target)
        shutil.rmtree(staging, ignore_errors=True)
    _fsync_directory(target.parent)


def write_wglink(
    geometry: PointGridHornGeometry,
    output_path: str | Path,
    *,
    built_geometry: BuiltGeometry | None = None,
    identity: WgLinkIdentity | Mapping[str, Mapping[str, Any]] | None = None,
    instance_slug: str = "waveguide",
    open_throat: bool = True,
    check_points: object | None = None,
    informational_parameters: Mapping[str, float] | None = None,
    interface_sources: Sequence[WgLinkSourceInterface | Mapping[str, Any]]
    | None = None,
) -> WgLinkInfo:
    """Write a checksummed ``.wglink`` directory and return every product.

    Bundle/design/export identity is caller-owned: no identifier, sequence,
    timestamp, or content identity is minted here.  Callers with a mesh build
    should pass its ``BuiltGeometry``; direct tests and adapters may use the
    pure realized-bounds fallback.
    """

    if not isinstance(geometry, PointGridHornGeometry):
        raise MesherError("wglink export requires PointGridHornGeometry")
    mode = geometry.build_mode
    if mode not in _SOLID_BUILD_MODES:
        raise MesherError(
            "wglink export supports only FREESTANDING and ENCLOSURE builds; "
            f"{mode.value.upper()} is not supported"
        )
    if geometry.enclosure is not None and int(geometry.enclosure.plan_type) != 1:
        raise MesherError(
            f"wglink export rejects enclosure plan_type={geometry.enclosure.plan_type}; "
            "only plan_type=1 has a buildable watertight enclosure"
        )
    if not geometry.closed:
        raise MesherError("wglink export requires a full, closed point grid")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    built = built_geometry or _realized_bundle_geometry(geometry)
    _validate_realized_bundle_geometry(geometry, built)
    datums = derive_datums(geometry, built)
    point_grid = _point_grid_payload(geometry, check_points=check_points)
    parameters = _parameter_table(
        geometry,
        built,
        instance_slug=instance_slug,
        informational_parameters=informational_parameters,
    )
    identity_sections = _identity_sections(identity)
    identity_sections.setdefault("generator", _default_generator())
    source_interfaces = _source_interface_table(interface_sources)

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        step_path, cad_info = write_step(
            geometry, staging / "waveguide.step", open_throat=open_throat
        )
        grid_path = staging / "point-grid.json"
        grid_path.write_text(
            json.dumps(point_grid, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        files = {
            "waveguide.step": {
                "sha256": _sha256(step_path),
                "size_bytes": step_path.stat().st_size,
                "media_type": "model/step",
            },
            "point-grid.json": {
                "sha256": _sha256(grid_path),
                "size_bytes": grid_path.stat().st_size,
                "media_type": "application/json",
            },
        }
        required_features = ["checksummed-files-v1", "link-local-frame-v1"]
        if source_interfaces:
            required_features.append(SOURCE_INTERFACE_FEATURE)
        manifest: dict[str, Any] = {
            "wglink_version": "1.1",
            "required_features": required_features,
            **identity_sections,
            "coordinate_system": {
                "length_unit": "mm",
                "handedness": "right",
                "matrix_convention": "row-major-local-to-parent",
                "step_from_design": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            },
            "body": {
                "file": "waveguide.step",
                "kind": cad_info.body,
                "n_faces": cad_info.n_faces,
                "volume_mm3": cad_info.volume_mm3,
                "bbox_mm": [list(v) for v in cad_info.bounding_box_mm],
                "throat_opened": cad_info.throat_opened,
                "semantic_face_names": False,
            },
            "files": files,
            "symmetry": {
                "declared_planes": list(built.symmetry_snap_axes),
                "solver_cut_plane_y_mm": 0.0,
            },
            "datums": datums,
            "interface": {"sources": source_interfaces},
            "parameters": parameters,
            "roles": {
                "scheme": "advanced-face-record-order",
                "assignments": {},
            },
        }
        if geometry.enclosure is not None:
            manifest["enclosure"] = {
                "edge_type": int(geometry.enclosure.edge_type),
                "plan_type": int(geometry.enclosure.plan_type),
            }
        manifest_path = staging / "wglink.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _publish_bundle(staging, target)
        return WgLinkInfo(
            path=target,
            manifest_path=target / "wglink.json",
            step_path=target / "waveguide.step",
            point_grid_path=target / "point-grid.json",
            manifest=manifest,
            cad_info=CadInfo(
                path=target / "waveguide.step",
                body=cad_info.body,
                n_faces=cad_info.n_faces,
                volume_mm3=cad_info.volume_mm3,
                bounding_box_mm=cad_info.bounding_box_mm,
                throat_opened=cad_info.throat_opened,
                units=cad_info.units,
            ),
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_wglink(path: str | Path, *, verify_checksums: bool = True) -> dict[str, Any]:
    """Read a v1 bundle, rejecting unsupported features and corrupt files."""

    bundle = Path(path)
    manifest_path = bundle / "wglink.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MesherError(f"could not read wglink manifest: {exc}") from exc
    version = str(manifest.get("wglink_version", ""))
    if version.split(".", 1)[0] != "1":
        raise MesherError(f"unsupported wglink major version {version!r}")
    supported = {
        "checksummed-files-v1",
        "link-local-frame-v1",
        SOURCE_INTERFACE_FEATURE,
    }
    unknown = sorted(set(manifest.get("required_features", ())).difference(supported))
    if unknown:
        raise MesherError(
            "unsupported required wglink feature(s): " + ", ".join(unknown)
        )
    interface = manifest.get("interface")
    raw_sources = interface.get("sources") if isinstance(interface, Mapping) else None
    has_source_feature = SOURCE_INTERFACE_FEATURE in manifest.get(
        "required_features", ()
    )
    if raw_sources is None:
        raw_sources = []
    if not isinstance(raw_sources, list):
        raise MesherError("wglink.interface.sources must be an array")
    normalized_sources = _source_interface_table(raw_sources)
    if bool(normalized_sources) != has_source_feature:
        raise MesherError(
            "source-interface-v1 is required exactly when wglink.interface.sources is non-empty"
        )
    if not verify_checksums:
        return manifest
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise MesherError("wglink manifest has no files table")
    root = bundle.resolve()
    declared_names = set(files)
    actual_names = {
        child.relative_to(bundle).as_posix()
        for child in bundle.rglob("*")
        if (child.is_file() or child.is_symlink())
        and child.relative_to(bundle).as_posix() != "wglink.json"
    }
    unchecksummed = sorted(actual_names.difference(declared_names))
    if unchecksummed:
        raise MesherError(
            "wglink bundle contains unchecksummed file(s): " + ", ".join(unchecksummed)
        )
    for name, record in files.items():
        if not isinstance(name, str) or not isinstance(record, Mapping):
            raise MesherError("wglink files table is malformed")
        candidate = (bundle / name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise MesherError(
                f"wglink file escapes bundle directory: {name!r}"
            ) from exc
        if not candidate.is_file():
            raise MesherError(f"wglink file is missing: {name}")
        actual_size = candidate.stat().st_size
        expected_size = record.get("size_bytes")
        if actual_size != expected_size:
            raise MesherError(
                f"wglink checksum validation failed for {name}: size is {actual_size}, "
                f"expected {expected_size}"
            )
        actual_digest = _sha256(candidate)
        expected_digest = record.get("sha256")
        if actual_digest != expected_digest:
            raise MesherError(
                f"wglink checksum validation failed for {name}: sha256 mismatch"
            )
    return manifest


__all__ = [
    "CadBody",
    "CadInfo",
    "WgLinkIdentity",
    "WgLinkInfo",
    "WgLinkSourceInterface",
    "SOURCE_INTERFACE_FEATURE",
    "read_wglink",
    "normalise_step_header",
    "write_step",
    "write_step_from_config",
    "write_wglink",
]
