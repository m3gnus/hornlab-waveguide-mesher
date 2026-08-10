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
import json
import re
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from .datums import derive_datums
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
        ring_planar.append(float(np.ptp(z)) < 1.0e-6)
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

    return {
        "units": "mm",
        "build_mode": geometry.build_mode.value,
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
        values.update(
            {
                "enc_w": float(bounds["bx1"]) - float(bounds["bx0"]),
                "enc_h": float(bounds["by1"]) - float(bounds["by0"]),
                "enc_depth": float(bounds["enc_depth"]),
                "enc_edge": float(bounds["clamped_edge"]),
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
    if target.exists():
        raise MesherError(f"wglink output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    built = built_geometry or _realized_bundle_geometry(geometry)
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
        manifest: dict[str, Any] = {
            "wglink_version": "1.0",
            "required_features": ["checksummed-files-v1", "link-local-frame-v1"],
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
            "interface": {},
            "parameters": parameters,
            "roles": {
                "scheme": "advanced-face-record-order",
                "assignments": {},
            },
        }
        manifest_path = staging / "wglink.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging.replace(target)
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
    supported = {"checksummed-files-v1", "link-local-frame-v1"}
    unknown = sorted(set(manifest.get("required_features", ())).difference(supported))
    if unknown:
        raise MesherError("unsupported required wglink feature(s): " + ", ".join(unknown))
    if not verify_checksums:
        return manifest
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise MesherError("wglink manifest has no files table")
    root = bundle.resolve()
    declared_names = set(files)
    actual_names = {
        child.name
        for child in bundle.iterdir()
        if child.is_file() and child.name != "wglink.json"
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
            raise MesherError(f"wglink file escapes bundle directory: {name!r}") from exc
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
    "read_wglink",
    "write_step",
    "write_step_from_config",
    "write_wglink",
]
