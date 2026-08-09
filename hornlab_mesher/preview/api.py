"""Public ``hornlab.preview/1`` geometry API.

Surface orientation is part of the render contract. ``horn.inner`` and
``source_cap`` normals point into the acoustic air domain. ``horn.outer``,
``mouth_rim``, ``wall.rear_cap``, and every ``enclosure.*`` role point toward
the solid exterior (the rim/front roles are front-facing). Every triangle is
counter-clockwise from its shipped normal side: ``cross(b-a, c-a)`` has a
strictly positive dot product with the triangle's average vertex normal.

These rules are checked directly for every emitted triangle with usable area.
Near-degenerate sampling slivers abstain from winding decisions. Signed volume
is deliberately not used because most preview roles are open shells.

``analytic-parametric`` means finite differences of samples evaluated on the
true analytic/canonical surface in its real axial and azimuthal parameter
coordinates.  It never means a derivative of, or normal averaged from, the
emitted triangle mesh.  This definition intentionally includes finite
differences: the method identifies the surface being differentiated, not a
symbolic differentiation implementation.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..builders.enclosure import sample_enclosure_plan
from ..builders.point_grid_sources import _source_cap_height, _source_cap_radius
from ..config_builder import build_geometry_params
from ..freeform import build_freeform_geometry
from ..geometry import PointGridHornGeometry
from ..profile_sampling import (
    ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY,
    FREEFORM_CONTINUOUS_COLLAPSE_KEY,
    _outer_offset_shell,
    build_point_grid_arrays,
)
from ..profile_formulas import (
    osse_coverage_saturation,
    osse_coverage_saturation_probe,
)
from ..profiles import eval_param
from ..viewport import build_viewport_geometry_from_config
from .fidelity import (
    adaptive_grid_indices,
    analytic_grid_curvature,
    analytic_grid_normals,
    estimate_grid_fidelity,
    resample_grid_vectors,
    resample_parametric_grid,
)


_API_VERSION = "hornlab.preview/1"
_METADATA_VERSION = "hornlab.preview/1.3"
_MAX_ARC_INTERVALS = 1024
# Plan corner radius floors. ``sample_rounded_rect`` treats anything at or below
# 1e-3 mm as a sharp box and emits a different vertex count, so both floors stay
# above it; see ``_plan_ring``.
_PLAN_CORNER_FLOOR_MM = 0.1
_FACETED_CORNER_FLOOR_MM = 2.0e-3
# A ruled-band column whose ring chord is below this is a floored-corner
# artifact, not geometry: collapse it to the corner triangle the solver builds.
_DEGENERATE_COLUMN_MM = 1.0e-2
# Dihedral angle across the emitted throat band above which it and the offset
# shell are two faces rather than one curved surface; see _outer_shell_surfaces.
_THROAT_JOG_CREASE_DEG = 15.0
_MAX_ANGULAR_SAMPLES = 4096
_MAX_CANONICAL_VERTICES = 1_000_000
_ORIENTATION_AREA_MEDIAN_FRACTION = 0.125
_ORIENTATION_COSINE_TOLERANCE = 1.0e-10
# A sharp morph corner has no defined offset direction, so the outer shell can
# carry a couple of full-area facets tipped just past perpendicular there. Those
# are singularities, not inverted patches: an inverted patch points decisively
# the wrong way (cosine near -1) or covers real area. Tolerate only the
# combination of "barely past perpendicular" and "negligible share of the
# surface", and count them where they can be seen.
_ORIENTATION_SHALLOW_COSINE = 0.25
_ORIENTATION_SINGULAR_AREA_FRACTION = 0.005
# Internal orientation evidence travels under an identity key so ordinary
# caller metadata cannot accidentally opt out of PreviewSurfaceV1's contract
# check.  The proof itself is also single-use and bound to the exact arrays the
# builder checked; see _OrientationCheckProof.
_ORIENTATION_PROOF_KEY = object()
_ORIENTATION_BY_ROLE = {
    "horn.inner": "air-side",
    "horn.outer": "exterior",
    "wall.throat_band": "exterior",
    "mouth_rim": "exterior",
    "source_cap": "air-side",
    "wall.rear_cap": "exterior",
    # The axial band from the outer throat ring back to the rear rim. The mesh
    # has always built this (it prepends the rear ring to the outer shell); the
    # preview only needs it as its own role because its shell is already
    # emitted by the time the rear plane is known.
    "wall.rear_return": "exterior",
    "enclosure.front": "exterior",
    "enclosure.roundover": "exterior",
    "enclosure.side": "exterior",
    "enclosure.rear": "exterior",
}
_LOD_PRESETS = {
    "coarse": {
        "chord": 0.15,
        "normal": 8.0,
        "silhouette": 64,
        "axial": 12,
        "roundover": 6,
        "cap": 8,
        "master_axial": 48,
    },
    "fine": {
        "chord": 0.05,
        "normal": 3.0,
        "silhouette": 128,
        "axial": 48,
        "roundover": 12,
        "cap": 16,
        "master_axial": 96,
    },
    "inspection": {
        "chord": 0.025,
        "normal": 2.0,
        "silhouette": 256,
        "axial": 96,
        "roundover": 12,
        "cap": 24,
        "master_axial": 192,
    },
}


def _validate_finite_metadata(value: Any, path: str = "metadata") -> None:
    """Reject non-finite numeric metadata before it reaches strict JSON."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_metadata(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_metadata(item, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"{path} must be finite or null")


@dataclass(frozen=True)
class PreviewOptionsV1:
    lod: str = "fine"
    include_inner: bool = True
    include_outer: bool = True
    include_enclosure: bool = True
    include_source_cap: bool = True
    include_rear_cap: bool = True
    include_curvature: bool = True
    max_chord_error_mm: float | None = None
    max_normal_step_deg: float | None = None
    min_silhouette_segments: int | None = None
    max_vertices: int | None = None


@dataclass(frozen=True)
class PreviewSurfaceV1:
    role: str
    positions: NDArray[np.float64]
    indices: NDArray[np.uint32]
    normals: NDArray[np.float64]
    shading: str
    normal_method: str
    closed_phi: bool
    curvature_mean: NDArray[np.float64] | None = None
    curvature_principal: NDArray[np.float64] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positions = np.ascontiguousarray(self.positions, dtype=np.float64)
        normals = np.ascontiguousarray(self.normals, dtype=np.float64)
        indices = np.ascontiguousarray(self.indices, dtype=np.uint32).reshape(-1)
        curvature_mean = (
            None
            if self.curvature_mean is None
            else np.ascontiguousarray(self.curvature_mean, dtype=np.float64).reshape(-1)
        )
        curvature_principal = (
            None
            if self.curvature_principal is None
            else np.ascontiguousarray(
                self.curvature_principal, dtype=np.float64
            ).reshape(-1)
        )
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"{self.role}: positions must have shape (N, 3)")
        if normals.shape != positions.shape:
            raise ValueError(f"{self.role}: normals must be row-aligned with positions")
        if (curvature_mean is None) != (curvature_principal is None):
            raise ValueError(
                f"{self.role}: mean and principal curvature must be provided together"
            )
        if curvature_mean is not None and (
            curvature_mean.shape != (len(positions),)
            or curvature_principal is None
            or curvature_principal.shape != (len(positions),)
        ):
            raise ValueError(f"{self.role}: curvature must be row-aligned with positions")
        if indices.size % 3:
            raise ValueError(f"{self.role}: indices must contain triangles")
        if indices.size and int(indices.max()) >= len(positions):
            raise ValueError(f"{self.role}: index exceeds vertex count")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(normals)):
            raise ValueError(f"{self.role}: positions and normals must be finite")
        if curvature_mean is not None and (
            not np.all(np.isfinite(curvature_mean))
            or curvature_principal is None
            or not np.all(np.isfinite(curvature_principal))
        ):
            raise ValueError(f"{self.role}: curvature must be finite")
        if not np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-3):
            raise ValueError(f"{self.role}: normals must be unit length")
        if self.shading not in {"smooth", "flat"}:
            raise ValueError(f"{self.role}: unsupported shading {self.shading!r}")
        if self.normal_method not in {"analytic-parametric", "exact-planar"}:
            raise ValueError(
                f"{self.role}: unsupported normal method {self.normal_method!r}"
            )
        orientation = _ORIENTATION_BY_ROLE.get(self.role)
        if orientation is None:
            raise ValueError(f"{self.role}: no preview orientation contract")
        metadata = dict(self.metadata)
        proof = metadata.pop(_ORIENTATION_PROOF_KEY, None)
        proven_orientation = (
            proof.consume(self.positions, self.indices, self.normals)
            if isinstance(proof, _OrientationCheckProof)
            else None
        )
        orientation_check = (
            proven_orientation
            if proven_orientation is not None
            else _triangle_orientation_analysis(positions, indices, normals)
        )
        if orientation_check.negative_triangles:
            raise ValueError(
                f"{self.role}: {orientation_check.negative_triangles} non-degenerate "
                "triangle windings disagree with their normals"
            )
        if metadata.get("orientation", orientation) != orientation:
            raise ValueError(
                f"{self.role}: orientation must be {orientation!r}"
            )
        if metadata.get("windingChecked", True) is not True:
            raise ValueError(f"{self.role}: windingChecked must be true")
        metadata["orientation"] = orientation
        metadata["windingChecked"] = True
        metadata["degenerateTriangles"] = orientation_check.degenerate_triangles
        metadata["orientationAbstainingTriangles"] = (
            orientation_check.abstaining_triangles
        )
        metadata["disagreeingTriangles"] = orientation_check.negative_triangles
        metadata["orientationSingularTriangles"] = orientation_check.singular_triangles
        metadata["curvature"] = (
            "absent"
            if curvature_mean is None
            else "planar"
            if self.normal_method == "exact-planar"
            else "analytic"
        )
        _validate_finite_metadata(metadata, f"{self.role}.metadata")
        positions.setflags(write=False)
        normals.setflags(write=False)
        indices.setflags(write=False)
        if curvature_mean is not None:
            curvature_mean.setflags(write=False)
            assert curvature_principal is not None
            curvature_principal.setflags(write=False)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "curvature_mean", curvature_mean)
        object.__setattr__(self, "curvature_principal", curvature_principal)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True)
class PreviewGeometryV1:
    surfaces: list[PreviewSurfaceV1]
    metadata: dict[str, Any] = field(default_factory=dict)


def _lod_config(config: Mapping[str, Any], angular: int, axial: int) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    mesh = result.get("mesh")
    if not isinstance(mesh, Mapping):
        mesh = {}
    else:
        mesh = dict(mesh)
    mesh["angular_segments"] = int(angular)
    mesh["length_segments"] = int(axial)
    # Canonical names win over any imported/camel-case aliases in config_builder.
    mesh.pop("angularSegments", None)
    mesh.pop("lengthSegments", None)
    result["mesh"] = mesh
    return result


def _adaptive_lod_config(
    config: Mapping[str, Any], angular: int, axial: int, *, power: float
) -> dict[str, Any]:
    """Seed the candidate lattice with a nested throat-biased axial map."""

    result = _lod_config(config, angular, axial)
    mesh = dict(result["mesh"])
    sampling = str(
        mesh.get("sampling_mode", mesh.get("samplingMode", "uniform"))
    ).strip().lower()
    formula = str(config.get("formula", "OSSE")).strip().upper()
    if formula != "ICW" and sampling in {"", "uniform", "linear", "canonical", "default"} and not any(
        key in mesh
        for key in ("z_map_points", "zMapPoints", "zmapPoints", "ZMapPoints")
    ):
        mesh["sampling_mode"] = "zmap"
        mesh["z_map_kind"] = "samples"
        parameter = np.linspace(0.0, 1.0, int(axial) + 1, dtype=np.float64)
        # The same analytic map at dyadically related counts makes the default
        # coarse stations exact members of fine/inspection candidate lattices.
        mesh["z_map_points"] = (
            0.5 - 0.5 * np.cos(math.pi * parameter)
        ).tolist()
    result["mesh"] = mesh
    return result


def _surface_grid(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert canonical ``(phi,t,xyz)`` to surface ``(t,phi,xyz)``."""

    return np.ascontiguousarray(np.transpose(points, (1, 0, 2)), dtype=np.float64)


def _grid_indices(n_t: int, n_phi: int, *, closed_phi: bool) -> NDArray[np.uint32]:
    """Two triangles per quad, in the same order the scalar loop emitted."""

    phi_intervals = n_phi if closed_phi else n_phi - 1
    if n_t < 2 or phi_intervals < 1:
        return np.empty(0, dtype=np.uint32)
    row0 = (np.arange(n_t - 1, dtype=np.uint32) * n_phi)[:, None]
    row1 = row0 + n_phi
    ip = np.arange(phi_intervals, dtype=np.uint32)[None, :]
    ip1 = (ip + 1) % n_phi
    triangles = np.empty((n_t - 1, phi_intervals, 6), dtype=np.uint32)
    triangles[:, :, 0] = row0 + ip
    triangles[:, :, 1] = row0 + ip1
    triangles[:, :, 2] = row1 + ip1
    triangles[:, :, 3] = row0 + ip
    triangles[:, :, 4] = row1 + ip1
    triangles[:, :, 5] = row1 + ip
    return triangles.reshape(-1)


@dataclass(frozen=True)
class _OrientationAnalysis:
    positive_triangles: int
    negative_triangles: int
    degenerate_triangles: int
    abstaining_triangles: int
    singular_triangles: int = 0


@dataclass
class _OrientationCheckProof:
    """Single-use evidence for an unchanged, internally checked index buffer."""

    positions: NDArray[np.float64]
    indices: NDArray[np.uint32]
    normals: NDArray[np.float64]
    analysis: _OrientationAnalysis
    digest: bytes
    _used: bool = field(default=False, init=False, repr=False)

    def consume(
        self,
        positions: NDArray[np.float64],
        indices: NDArray[np.uint32],
        normals: NDArray[np.float64],
    ) -> _OrientationAnalysis | None:
        if (
            self._used
            or positions is not self.positions
            or indices is not self.indices
            or normals is not self.normals
            or not _orientation_buffer_is_contiguous(positions, indices, normals)
            or _orientation_buffer_digest(positions, indices, normals) != self.digest
        ):
            return None
        self._used = True
        return self.analysis


def _orientation_buffer_is_contiguous(
    positions: NDArray[np.float64],
    indices: NDArray[np.uint32],
    normals: NDArray[np.float64],
) -> bool:
    return bool(
        positions.dtype == np.dtype(np.float64)
        and indices.dtype == np.dtype(np.uint32)
        and normals.dtype == np.dtype(np.float64)
        and positions.ndim == 2
        and positions.shape[1:] == (3,)
        and normals.shape == positions.shape
        and indices.ndim == 1
        and positions.flags.c_contiguous
        and indices.flags.c_contiguous
        and normals.flags.c_contiguous
    )


def _orientation_buffer_digest(
    positions: NDArray[np.float64],
    indices: NDArray[np.uint32],
    normals: NDArray[np.float64],
) -> bytes:
    """Bind orientation evidence to exact array type, shape, and bytes."""

    digest = hashlib.blake2b(digest_size=32)
    for value in (positions, indices, normals):
        array = np.asarray(value)
        descriptor = repr((array.dtype.str, array.shape)).encode("ascii")
        digest.update(len(descriptor).to_bytes(4, "little"))
        digest.update(descriptor)
        digest.update(memoryview(array).cast("B"))
    return digest.digest()


@dataclass(frozen=True)
class _OrientedIndices:
    indices: NDArray[np.uint32]
    degenerate_triangles: int
    abstaining_triangles: int
    disagreeing_triangles: int
    singular_triangles: int = 0
    proof: _OrientationCheckProof | None = field(
        default=None, compare=False, repr=False
    )


def _triangle_orientation_analysis(
    positions: NDArray[np.float64],
    indices: NDArray[np.uint32],
    normals: NDArray[np.float64],
) -> _OrientationAnalysis:
    """Classify winding using face/normal cosine and relative triangle area.

    A triangle abstains when its doubled area is at most one eighth of the
    surface's median positive doubled area. The median supplies a robust local
    surface scale, while the 12.5% cutoff excludes the measured corner-lattice
    slivers without forgiving a full-area inverted face. Non-degenerate
    face/normal cosines within ``1e-10`` of zero also abstain as numerically
    undecidable.
    """

    triangles = np.asarray(indices, dtype=np.uint32).reshape(-1, 3)
    if not len(triangles):
        return _OrientationAnalysis(0, 0, 0, 0)
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    vectors = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    face_vectors = np.cross(b - a, c - a)
    doubled_area = np.linalg.norm(face_vectors, axis=1)
    positive_area = doubled_area[doubled_area > 0.0]
    median_area = float(np.median(positive_area)) if len(positive_area) else 0.0
    degenerate = doubled_area <= (
        _ORIENTATION_AREA_MEDIAN_FRACTION * median_area
    )
    # Summed rather than np.mean'd: the (n,3,3) gather np.mean needs costs
    # about three times the three (n,3) gathers, for the same three-term
    # sequential sum and the same bits. Internal buffers that already have
    # positive winding reuse this analysis in PreviewSurfaceV1. A flipped
    # buffer is deliberately analysed again because swapping two indices
    # reassociates this floating-point sum.
    average_normal = (
        vectors[triangles[:, 0]] + vectors[triangles[:, 1]] + vectors[triangles[:, 2]]
    ) / 3.0
    normal_length = np.linalg.norm(average_normal, axis=1)
    denominator = doubled_area * normal_length
    cosine = np.zeros(len(triangles), dtype=np.float64)
    usable_denominator = denominator > 0.0
    if usable_denominator.all():
        # No masked copies to make, and no division by zero to avoid.
        cosine = np.einsum("ij,ij->i", face_vectors, average_normal) / denominator
    else:
        cosine[usable_denominator] = np.einsum(
            "ij,ij->i",
            face_vectors[usable_denominator],
            average_normal[usable_denominator],
        ) / denominator[usable_denominator]
    ambiguous = (
        ~np.isfinite(cosine)
        | ~usable_denominator
        | (np.abs(cosine) <= _ORIENTATION_COSINE_TOLERANCE)
    )
    abstaining = degenerate | ambiguous
    positive = (~abstaining) & (cosine > _ORIENTATION_COSINE_TOLERANCE)
    negative = (~abstaining) & (cosine < -_ORIENTATION_COSINE_TOLERANCE)
    # Whichever side is outnumbered defines the winding the surface disagrees
    # with; only that side can be a corner singularity rather than a fault.
    minority = negative if np.count_nonzero(negative) <= np.count_nonzero(positive) else positive
    singular = (
        minority
        & (np.abs(cosine) < _ORIENTATION_SHALLOW_COSINE)
        & (
            doubled_area.sum() <= 0.0
            or doubled_area[minority].sum()
            <= _ORIENTATION_SINGULAR_AREA_FRACTION * doubled_area.sum()
        )
    )
    positive &= ~singular
    negative &= ~singular
    return _OrientationAnalysis(
        positive_triangles=int(np.count_nonzero(positive)),
        negative_triangles=int(np.count_nonzero(negative)),
        degenerate_triangles=int(np.count_nonzero(degenerate)),
        abstaining_triangles=int(np.count_nonzero(abstaining | singular)),
        singular_triangles=int(np.count_nonzero(singular)),
    )


def _orient_indices_to_normals(
    role: str,
    positions: NDArray[np.float64],
    indices: NDArray[np.uint32],
    normals: NDArray[np.float64],
) -> _OrientedIndices:
    """Return one consistently wound index buffer for the shipped normals."""

    triangles = np.asarray(indices, dtype=np.uint32).reshape(-1, 3)
    if not len(triangles):
        return _OrientedIndices(triangles.reshape(-1), 0, 0, 0)
    analysis = _triangle_orientation_analysis(positions, triangles, normals)
    if analysis.positive_triangles and analysis.negative_triangles:
        disagreeing = min(
            analysis.positive_triangles, analysis.negative_triangles
        )
        raise ValueError(
            f"{role}: inconsistent local orientation ({disagreeing}/{len(triangles)} "
            "non-degenerate triangles disagree with their normals)"
        )
    if not analysis.positive_triangles and not analysis.negative_triangles:
        raise ValueError(f"{role}: no non-degenerate triangles establish winding")
    already_oriented = bool(analysis.positive_triangles)
    oriented = (
        triangles.reshape(-1)
        if already_oriented
        else triangles[:, (0, 2, 1)].reshape(-1)
    )
    oriented_indices = np.asarray(oriented, dtype=np.uint32)
    can_prove = already_oriented and _orientation_buffer_is_contiguous(
        positions, oriented_indices, normals
    )
    return _OrientedIndices(
        indices=oriented_indices,
        degenerate_triangles=analysis.degenerate_triangles,
        abstaining_triangles=analysis.abstaining_triangles,
        disagreeing_triangles=0,
        singular_triangles=analysis.singular_triangles,
        proof=(
            _OrientationCheckProof(
                positions=positions,
                indices=oriented_indices,
                normals=normals,
                analysis=analysis,
                digest=_orientation_buffer_digest(
                    positions, oriented_indices, normals
                ),
            )
            if can_prove
            else None
        ),
    )


def _orientation_metadata(result: _OrientedIndices) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "degenerateTriangles": result.degenerate_triangles,
        "orientationAbstainingTriangles": result.abstaining_triangles,
        "disagreeingTriangles": result.disagreeing_triangles,
        "orientationSingularTriangles": result.singular_triangles,
    }
    if result.proof is not None:
        # This internal object key deliberately falls outside the public
        # string-key metadata type. PreviewSurfaceV1 removes it before metadata
        # validation or publication.
        metadata[_ORIENTATION_PROOF_KEY] = result.proof  # type: ignore[index]
    return metadata


def _smooth_grid_surface(
    role: str,
    points: NDArray[np.float64],
    reference: NDArray[np.float64],
    *,
    closed_phi: bool,
    point_t: NDArray[np.float64] | None = None,
    point_phi: NDArray[np.float64] | None = None,
    reference_t: NDArray[np.float64] | None = None,
    reference_phi: NDArray[np.float64] | None = None,
    orientation_hint: NDArray[np.float64] | None = None,
    include_curvature: bool = True,
) -> tuple[PreviewSurfaceV1, dict[str, float]]:
    ref_normals = analytic_grid_normals(
        reference,
        closed_phi=closed_phi,
        t_coordinates=reference_t,
        phi_coordinates=reference_phi,
    )
    resampling = any(
        value is not None
        for value in (point_t, point_phi, reference_t, reference_phi)
    )
    if resampling:
        normals = resample_parametric_grid(
            ref_normals,
            points.shape[:2],
            source_t=reference_t,
            source_phi=reference_phi,
            target_t=point_t,
            target_phi=point_phi,
            normalise=True,
            closed_phi=closed_phi,
        )
    else:
        normals = resample_grid_vectors(
            ref_normals, points.shape[:2], closed_phi=closed_phi
        )
    curvature_mean = curvature_principal = None
    if include_curvature:
        ref_mean, ref_principal = analytic_grid_curvature(
            reference,
            closed_phi=closed_phi,
            t_coordinates=reference_t,
            phi_coordinates=reference_phi,
        )
        if resampling:
            curvature = resample_parametric_grid(
                np.stack((ref_mean, ref_principal), axis=2),
                points.shape[:2],
                source_t=reference_t,
                source_phi=reference_phi,
                target_t=point_t,
                target_phi=point_phi,
                closed_phi=closed_phi,
            )
        else:
            curvature = resample_parametric_grid(
                np.stack((ref_mean, ref_principal), axis=2),
                points.shape[:2],
                closed_phi=closed_phi,
            )
        curvature_mean = curvature[:, :, 0]
        curvature_principal = curvature[:, :, 1]
    if orientation_hint is not None:
        hint = np.broadcast_to(np.asarray(orientation_hint, dtype=np.float64), normals.shape)
        if float(np.median(np.sum(normals * hint, axis=2))) < 0.0:
            normals = -normals
            if curvature_mean is not None:
                curvature_mean = -curvature_mean
                assert curvature_principal is not None
                curvature_principal = -curvature_principal
    positions = points.reshape(-1, 3)
    flat_normals = normals.reshape(-1, 3)
    oriented = _orient_indices_to_normals(
        role,
        positions,
        _grid_indices(*points.shape[:2], closed_phi=closed_phi),
        flat_normals,
    )
    surface = PreviewSurfaceV1(
        role=role,
        positions=positions,
        indices=oriented.indices,
        normals=flat_normals,
        shading="smooth",
        normal_method="analytic-parametric",
        closed_phi=closed_phi,
        curvature_mean=(
            None if curvature_mean is None else curvature_mean.reshape(-1)
        ),
        curvature_principal=(
            None
            if curvature_principal is None
            else curvature_principal.reshape(-1)
        ),
        metadata=_orientation_metadata(oriented),
    )
    fidelity = estimate_grid_fidelity(
        points,
        reference,
        normals,
        closed_phi=closed_phi,
        coarse_t=point_t,
        coarse_phi=point_phi,
        reference_t=reference_t,
        reference_phi=reference_phi,
    )
    return surface, fidelity


def _flat_strip(
    role: str,
    inner: NDArray[np.float64],
    outer: NDArray[np.float64],
    normal: tuple[float, float, float],
    *,
    closed_phi: bool,
    include_curvature: bool = True,
) -> PreviewSurfaceV1:
    points = np.stack((inner, outer), axis=0)
    normals = np.broadcast_to(np.asarray(normal, dtype=np.float64), points.shape).copy()
    positions = points.reshape(-1, 3)
    flat_normals = normals.reshape(-1, 3)
    oriented = _orient_indices_to_normals(
        role,
        positions,
        _grid_indices(2, points.shape[1], closed_phi=closed_phi),
        flat_normals,
    )
    return PreviewSurfaceV1(
        role=role,
        positions=positions,
        indices=oriented.indices,
        normals=flat_normals,
        shading="flat",
        normal_method="exact-planar",
        closed_phi=closed_phi,
        curvature_mean=(
            np.zeros(len(positions), dtype=np.float64) if include_curvature else None
        ),
        curvature_principal=(
            np.zeros(len(positions), dtype=np.float64) if include_curvature else None
        ),
        metadata=_orientation_metadata(oriented),
    )


def _flat_triangle(
    role: str,
    points: NDArray[np.float64],
    normal: tuple[float, float, float],
    *,
    include_curvature: bool = True,
) -> PreviewSurfaceV1:
    positions = np.asarray(points, dtype=np.float64).reshape(3, 3)
    flat_normals = np.broadcast_to(
        np.asarray(normal, dtype=np.float64), positions.shape
    ).copy()
    oriented = _orient_indices_to_normals(
        role,
        positions,
        np.asarray((0, 1, 2), dtype=np.uint32),
        flat_normals,
    )
    return PreviewSurfaceV1(
        role=role,
        positions=positions,
        indices=oriented.indices,
        normals=flat_normals,
        shading="flat",
        normal_method="exact-planar",
        closed_phi=False,
        curvature_mean=(
            np.zeros(len(positions), dtype=np.float64) if include_curvature else None
        ),
        curvature_principal=(
            np.zeros(len(positions), dtype=np.float64) if include_curvature else None
        ),
        metadata=_orientation_metadata(oriented),
    )


def _simplify_planar_ring(
    ring: NDArray[np.float64], tolerance: float
) -> NDArray[np.float64]:
    """Drop ring vertices whose removal moves the boundary less than ``tolerance``.

    A cap fans every boundary chord against a center a couple of hundred mm
    away, so a floored plan corner (a 2 um chamfer chord, a fillet's 0.1 mm
    inner arc walked in thirty samples) turns into fan slivers tens of
    thousands to one. The cap has no ring-correspondence obligation, so it may
    simplify its own boundary; the band it abuts stays within ``tolerance`` of
    the simplified polygon.
    """

    points = [np.asarray(p, dtype=np.float64) for p in ring]
    changed = True
    while changed and len(points) > 3:
        changed = False
        for index in range(len(points)):
            previous = points[index - 1]
            candidate = points[index]
            following = points[(index + 1) % len(points)]
            edge = following - previous
            edge_len = float(np.linalg.norm(edge))
            if edge_len <= 1.0e-12:
                deviation = float(np.linalg.norm(candidate - previous))
            else:
                fraction = float(
                    np.clip(np.dot(candidate - previous, edge) / edge_len**2, 0.0, 1.0)
                )
                deviation = float(
                    np.linalg.norm(candidate - (previous + fraction * edge))
                )
            if deviation < tolerance:
                points.pop(index)
                changed = True
                break
    return np.asarray(points, dtype=np.float64)


def _flat_cap(
    role: str,
    ring: NDArray[np.float64],
    normal: tuple[float, float, float],
    *,
    closed_phi: bool,
    include_curvature: bool = True,
    simplify_tolerance: float | None = None,
) -> PreviewSurfaceV1:
    ring = np.asarray(ring, dtype=np.float64)
    if simplify_tolerance is not None and closed_phi:
        ring = _simplify_planar_ring(ring, simplify_tolerance)
    center = np.mean(ring, axis=0)
    if closed_phi:
        # A corner-refined morph lattice can contain locally out-of-order phi
        # samples even though its boundary is star-shaped. The cap has no row
        # correspondence to preserve, so angular ordering avoids manufacturing
        # inverted fan faces from that sampling artifact.
        angles = np.arctan2(ring[:, 1] - center[1], ring[:, 0] - center[0])
        ring = ring[np.argsort(angles, kind="stable")]
    positions = np.vstack((ring, center))
    center_index = len(ring)
    triangles: list[int] = []
    limit = len(ring) if closed_phi else len(ring) - 1
    for ip in range(limit):
        ip1 = (ip + 1) % len(ring)
        if normal[2] >= 0.0:
            triangles.extend((center_index, ip, ip1))
        else:
            triangles.extend((center_index, ip1, ip))
    normals = np.broadcast_to(np.asarray(normal, dtype=np.float64), positions.shape).copy()
    oriented = _orient_indices_to_normals(
        role,
        positions,
        np.asarray(triangles, dtype=np.uint32),
        normals,
    )
    return PreviewSurfaceV1(
        role=role,
        positions=positions,
        indices=oriented.indices,
        normals=normals,
        shading="flat",
        normal_method="exact-planar",
        closed_phi=closed_phi,
        curvature_mean=(
            np.zeros(len(positions), dtype=np.float64) if include_curvature else None
        ),
        curvature_principal=(
            np.zeros(len(positions), dtype=np.float64) if include_curvature else None
        ),
        metadata=_orientation_metadata(oriented),
    )


def _mouth_exit_direction(inner: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-phi direction the inner profile travels as it leaves the mouth.

    The rim is the end face of the wall, so this is the direction it faces. It
    is +z only while the mouth still opens forward: a rolled-back termination
    (R-OSSE at high ``tmax``, any strong roundover) exits backwards, and a fixed
    +z hint would invert the whole rim rather than orient it.
    """

    if inner.shape[1] < 2:
        return np.tile(np.asarray((0.0, 0.0, 1.0), dtype=np.float64), (len(inner), 1))
    direction = np.asarray(inner[:, -1, :] - inner[:, -2, :], dtype=np.float64)
    lengths = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / np.where(lengths > 0.0, lengths, 1.0)
    direction[lengths[:, 0] <= 1.0e-12] = (0.0, 0.0, 1.0)
    return direction


def _smooth_mouth_rim(
    inner: NDArray[np.float64],
    outer: NDArray[np.float64],
    *,
    closed_phi: bool,
    exit_direction: NDArray[np.float64],
    include_curvature: bool,
) -> PreviewSurfaceV1:
    grid = np.stack((inner, outer), axis=0)
    surface, _fidelity = _smooth_grid_surface(
        "mouth_rim",
        grid,
        grid,
        closed_phi=closed_phi,
        orientation_hint=np.asarray(exit_direction, dtype=np.float64)[None, :, :],
        include_curvature=include_curvature,
    )
    return surface


def _source_geometry(
    params: Mapping[str, Any], formula: str, inner: NDArray[np.float64]
) -> PointGridHornGeometry:
    auto_angle = float(eval_param(params.get("a0"), 0.0, 15.5))
    if formula == "FREEFORM":
        auto_angle = float(
            build_freeform_geometry(params).report()["tangentAnglesDeg"]["H"]["throat"]
        )
    return PointGridHornGeometry(
        inner_points=inner,
        source_shape=int(round(float(eval_param(params.get("sourceShape"), 0.0, 1)))),
        source_radius_mm=float(eval_param(params.get("sourceRadius"), 0.0, -1.0)),
        source_curv=int(round(float(eval_param(params.get("sourceCurv"), 0.0, 0)))),
        source_auto_angle_deg=auto_angle,
    )


def _source_cap(
    inner: NDArray[np.float64],
    params: Mapping[str, Any],
    formula: str,
    radial_intervals: int,
    *,
    closed_phi: bool,
    include_curvature: bool,
) -> tuple[PreviewSurfaceV1, dict[str, float] | None, dict[str, float | None]]:
    ring = np.asarray(inner[:, 0, :], dtype=np.float64)
    center = np.mean(ring, axis=0)
    radial = ring[:, :2] - center[:2]
    radii = np.linalg.norm(radial, axis=1)
    throat_radius = float(np.mean(radii[radii > 1.0e-12]))
    geometry = _source_geometry(params, formula, inner)
    cap_height = _source_cap_height(throat_radius, geometry)
    radius = _source_cap_radius(throat_radius, geometry)
    details: dict[str, float | None] = {
        "source_cap_height_mm": float(cap_height),
        "source_cap_radius_mm": float(radius) if math.isfinite(radius) else None,
    }
    if int(geometry.source_shape) == 0 or cap_height <= 1.0e-12 or not math.isfinite(radius):
        return (
            _flat_cap(
                "source_cap",
                ring,
                (0.0, 0.0, 1.0),
                closed_phi=closed_phi,
                include_curvature=include_curvature,
            ),
            None,
            details,
        )

    radius = max(float(radius), throat_radius * 1.001)
    sign = -1.0 if int(geometry.source_curv) == -1 else 1.0
    sphere_center = center.copy()
    sphere_center[2] += sign * (cap_height - radius)
    rim_angle = math.asin(np.clip(throat_radius / radius, -1.0, 1.0))
    directions = radial / radii[:, None]

    # Pole first, then one ring of ``directions`` per polar level: the same
    # order the per-vertex loop emitted, built as arrays.
    n_phi = len(ring)
    theta = rim_angle * np.arange(1, radial_intervals + 1, dtype=np.float64)
    theta /= radial_intervals
    rho = radius * np.sin(theta)
    level_z = sphere_center[2] + sign * radius * np.cos(theta)
    position_array = np.empty((1 + radial_intervals * n_phi, 3), dtype=np.float64)
    position_array[0] = sphere_center
    position_array[0, 2] += sign * radius
    rings = position_array[1:].reshape(radial_intervals, n_phi, 3)
    rings[:, :, 0] = center[0] + rho[:, None] * directions[None, :, 0]
    rings[:, :, 1] = center[1] + rho[:, None] * directions[None, :, 1]
    rings[:, :, 2] = level_z[:, None]
    normal_array = np.empty_like(position_array)
    normal_array[0] = (0.0, 0.0, sign)
    normal_array[1:] = sign * (position_array[1:] - sphere_center) / radius

    limit = n_phi if closed_phi else n_phi - 1
    ip = np.arange(limit, dtype=np.uint32)
    ip1 = (ip + 1) % n_phi
    fan = np.empty((limit, 3), dtype=np.uint32)
    fan[:, 0] = 0
    fan[:, 1] = 1 + ip
    fan[:, 2] = 1 + ip1
    if radial_intervals > 1:
        row0 = (1 + np.arange(radial_intervals - 1, dtype=np.uint32) * n_phi)[:, None]
        row1 = row0 + n_phi
        bands = np.empty((radial_intervals - 1, limit, 6), dtype=np.uint32)
        bands[:, :, 0] = row0 + ip
        bands[:, :, 1] = row1 + ip
        bands[:, :, 2] = row1 + ip1
        bands[:, :, 3] = row0 + ip
        bands[:, :, 4] = row1 + ip1
        bands[:, :, 5] = row0 + ip1
        indices = np.concatenate((fan.reshape(-1), bands.reshape(-1)))
    else:
        indices = fan.reshape(-1)

    oriented = _orient_indices_to_normals(
        "source_cap",
        position_array,
        indices,
        normal_array,
    )
    surface = PreviewSurfaceV1(
        role="source_cap",
        positions=position_array,
        indices=oriented.indices,
        normals=normal_array,
        shading="smooth",
        normal_method="analytic-parametric",
        closed_phi=closed_phi,
        curvature_mean=(
            -np.sum((position_array - sphere_center) * normal_array, axis=1)
            / (radius * radius)
            if include_curvature
            else None
        ),
        curvature_principal=(
            -np.sum((position_array - sphere_center) * normal_array, axis=1)
            / (radius * radius)
            if include_curvature
            else None
        ),
        metadata=_orientation_metadata(oriented),
    )
    angular_step = 2.0 * math.pi / n_phi if closed_phi else math.pi / max(n_phi - 1, 1)
    polar_step = rim_angle / radial_intervals
    max_angle = max(polar_step, math.sin(rim_angle) * angular_step)
    fidelity = {
        "max_chord_error_mm": max(
            radius * (1.0 - math.cos(polar_step / 2.0)),
            radius * (1.0 - math.cos(max_angle / 2.0)),
        ),
        "max_normal_step_deg": math.degrees(max_angle),
        "reference_density_multiplier": 4,
    }
    return surface, fidelity, details


def _ccw_ring(points: NDArray[np.float64]) -> NDArray[np.float64]:
    ring = np.asarray(points, dtype=np.float64)
    x = ring[:, 0]
    y = ring[:, 1]
    area2 = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return ring if area2 >= 0.0 else ring[::-1].copy()


def _faceted_edge(enclosure: Mapping[str, Any]) -> bool:
    """True when the plan is polygonal at every inset, so its edge is planes.

    ``sample_rounded_rect`` walks ``edge_type == 2`` corners along the straight
    chord between the arc endpoints, so a chamfered rounded rectangle is a
    polygon at every ``radial_t`` and the band it sweeps is a fan of planes.
    Ellipse and superellipse plans stay curved and keep the smooth path.
    """

    return int(enclosure["plan_type"]) == 1 and int(enclosure["edge_type"]) == 2


def _plan_ring(
    enclosure: Mapping[str, Any],
    z: float,
    radial_t: float,
    *,
    corner_intervals: int = 4,
) -> NDArray[np.float64]:
    bounds = enclosure["bounds"]
    edge = float(enclosure["edge_mm"])
    d = edge * (1.0 - radial_t)
    # The inset ring of an edge treatment has a genuinely sharp plan corner, but
    # ``sample_rounded_rect`` switches to its four-corner path below 1e-3 mm and
    # would then emit a different vertex count than the outer ring, so the
    # radius is floored rather than zeroed. A chamfer's band is ruled column by
    # column between the two rings, and 0.1 mm of floor there put the emitted
    # corner 0.05 mm off the plane the solver builds -- right at the finest LOD's
    # whole chord budget. Faceted plans keep the smallest floor that stays on the
    # rounded path.
    radius = max(
        _FACETED_CORNER_FLOOR_MM if _faceted_edge(enclosure) else _PLAN_CORNER_FLOOR_MM,
        edge * radial_t,
    )
    plan_type = int(enclosure["plan_type"])
    if plan_type in {2, 3}:
        count = max(4, 4 * int(corner_intervals))
        cx = 0.5 * (float(bounds["bx0"]) + float(bounds["bx1"]))
        cy = 0.5 * (float(bounds["by0"]) + float(bounds["by1"]))
        a = 0.5 * (float(bounds["bx1"]) - float(bounds["bx0"])) - d
        b = 0.5 * (float(bounds["by1"]) - float(bounds["by0"])) - d
        n = 2.0 if plan_type == 2 else float(enclosure.get("plan_n", 2.0))
        theta = np.arange(count, dtype=np.float64) * math.tau / count
        cosine = np.cos(theta)
        sine = np.sin(theta)
        radial = (
            np.abs(cosine / a) ** n + np.abs(sine / b) ** n
        ) ** (-1.0 / n)
        return np.column_stack(
            (
                cx + radial * cosine,
                cy + radial * sine,
                np.full(count, float(z), dtype=np.float64),
            )
        )
    return _ccw_ring(
        sample_enclosure_plan(
            bx0=float(bounds["bx0"]) + d,
            bx1=float(bounds["bx1"]) - d,
            by0=float(bounds["by0"]) + d,
            by1=float(bounds["by1"]) - d,
            corner_radius=radius,
            edge_type=int(enclosure["edge_type"]),
            z=float(z),
            plan_type=plan_type,
            plan_n=float(enclosure.get("plan_n", 2.0)),
            n_per_edge=1,
            n_per_corner=max(1, int(corner_intervals)),
        )
    )


def _plan_fidelity(
    enclosure: Mapping[str, Any], corner_intervals: int
) -> dict[str, float]:
    """Measure an emitted ellipse/superellipse plan against dense true samples."""

    plan_type = int(enclosure["plan_type"])
    if plan_type == 1:
        if _faceted_edge(enclosure):
            # Straight chords, sampled on the chord: the emitted polygon IS the
            # plan at every subdivision. Modelling it as a 90 degree arc bought
            # dozens of collinear samples per corner and reported a chord error
            # the geometry never had.
            return {
                "max_chord_error_mm": 0.0,
                "max_normal_step_deg": 0.0,
                "reference_density_multiplier": 1,
            }
        radius = float(enclosure.get("edge_mm", 0.1))
        return {
            "max_chord_error_mm": max(
                np.finfo(np.float64).eps,
                radius * (1.0 - math.cos(math.pi / (4.0 * corner_intervals))),
            ),
            "max_normal_step_deg": 90.0 / corner_intervals,
            "reference_density_multiplier": 4,
        }

    bounds = enclosure["bounds"]
    cx = 0.5 * (float(bounds["bx0"]) + float(bounds["bx1"]))
    cy = 0.5 * (float(bounds["by0"]) + float(bounds["by1"]))
    half_width = 0.5 * (float(bounds["bx1"]) - float(bounds["bx0"]))
    half_height = 0.5 * (float(bounds["by1"]) - float(bounds["by0"]))
    n = 2.0 if plan_type == 2 else float(enclosure.get("plan_n", 2.0))
    chord_error = 0.0
    normal_step = 0.0
    # Roundovers traverse the inset family. Sampling several true members makes
    # the published plan bound apply to the emitted transition, not just its
    # largest outer ring.
    for radial_t in np.linspace(0.0, 1.0, 5):
        ring = _plan_ring(
            enclosure, 0.0, float(radial_t), corner_intervals=corner_intervals
        )
        count = len(ring)
        dense_count = 8 * count
        dense = _plan_ring(
            enclosure, 0.0, float(radial_t), corner_intervals=2 * count
        )
        segment = np.arange(dense_count, dtype=np.int64) // 8
        weight = (np.arange(dense_count, dtype=np.float64) % 8) / 8.0
        chord = ring[segment] * (1.0 - weight[:, None]) + ring[
            (segment + 1) % count
        ] * weight[:, None]
        chord_error = max(
            chord_error, float(np.max(np.linalg.norm(dense - chord, axis=1)))
        )

        inset = float(enclosure["edge_mm"]) * (1.0 - float(radial_t))
        a = half_width - inset
        b = half_height - inset
        ux = (ring[:, 0] - cx) / a
        uy = (ring[:, 1] - cy) / b
        gradients = np.column_stack(
            (
                np.sign(ux) * np.abs(ux) ** (n - 1.0) / a,
                np.sign(uy) * np.abs(uy) ** (n - 1.0) / b,
            )
        )
        gradients /= np.linalg.norm(gradients, axis=1, keepdims=True)
        dots = np.sum(gradients * np.roll(gradients, -1, axis=0), axis=1)
        normal_step = max(
            normal_step,
            float(np.degrees(np.arccos(np.clip(np.min(dots), -1.0, 1.0)))),
        )
    return {
        "max_chord_error_mm": max(chord_error, np.finfo(np.float64).eps),
        "max_normal_step_deg": normal_step,
        "reference_density_multiplier": 8,
    }


def _adaptive_plan_intervals(
    enclosure: Mapping[str, Any],
    chord_target: float,
    normal_target: float,
    floor: int,
) -> tuple[int, bool]:
    if int(enclosure["plan_type"]) == 1:
        if _faceted_edge(enclosure):
            return max(1, int(floor)), False
        intervals = _intervals_for_arc(
            float(enclosure.get("edge_mm", 0.1)),
            90.0,
            chord_target,
            normal_target,
            floor,
        )
        return intervals, intervals >= _MAX_ARC_INTERVALS

    low = max(1, int(floor))
    measured = _plan_fidelity(enclosure, low)
    if (
        measured["max_chord_error_mm"] <= chord_target
        and measured["max_normal_step_deg"] <= normal_target
    ):
        return low, False
    high = low
    while high < _MAX_ARC_INTERVALS:
        high = min(_MAX_ARC_INTERVALS, high * 2)
        measured = _plan_fidelity(enclosure, high)
        if (
            measured["max_chord_error_mm"] <= chord_target
            and measured["max_normal_step_deg"] <= normal_target
        ):
            break
    else:
        return _MAX_ARC_INTERVALS, True
    if high == _MAX_ARC_INTERVALS and (
        measured["max_chord_error_mm"] > chord_target
        or measured["max_normal_step_deg"] > normal_target
    ):
        return high, True
    left = low + 1
    right = high
    while left < right:
        middle = (left + right) // 2
        measured = _plan_fidelity(enclosure, middle)
        if (
            measured["max_chord_error_mm"] <= chord_target
            and measured["max_normal_step_deg"] <= normal_target
        ):
            right = middle
        else:
            left = middle + 1
    return left, False


def _ray_cast(
    plan: NDArray[np.float64], center: NDArray[np.float64], direction: NDArray[np.float64]
) -> NDArray[np.float64]:
    best: tuple[float, NDArray[np.float64]] | None = None
    for j, a in enumerate(plan):
        b = plan[(j + 1) % len(plan)]
        segment = b[:2] - a[:2]
        denominator = direction[0] * segment[1] - direction[1] * segment[0]
        if abs(denominator) <= 1.0e-12:
            continue
        offset = a[:2] - center
        ray_t = (offset[0] * segment[1] - offset[1] * segment[0]) / denominator
        seg_t = (offset[0] * direction[1] - offset[1] * direction[0]) / denominator
        if ray_t >= -1.0e-9 and -1.0e-9 <= seg_t <= 1.0 + 1.0e-9:
            hit = a + np.clip(seg_t, 0.0, 1.0) * (b - a)
            if best is None or ray_t < best[0]:
                best = (ray_t, hit)
    if best is None:
        angles = np.arctan2(plan[:, 1] - center[1], plan[:, 0] - center[0])
        target = math.atan2(direction[1], direction[0])
        delta = np.abs(np.angle(np.exp(1j * (angles - target))))
        return plan[int(np.argmin(delta))]
    return best[1]


def _ray_aligned_ring(
    plan: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    center = np.mean(reference[:, :2], axis=0)
    result = np.empty_like(reference)
    for index, point in enumerate(reference):
        direction = point[:2] - center
        direction /= max(float(np.linalg.norm(direction)), 1.0e-14)
        result[index] = _ray_cast(plan, center, direction)
    return result


def _plan_corner_angles(
    plan: NDArray[np.float64], center: NDArray[np.float64]
) -> list[float]:
    """Angles (about ``center``) at which the plan breaks tangency.

    A ray-aligned ring only preserves the plan where a ray happens to sample
    it: between two rays that straddle a convex corner, the aligned polyline
    chord-cuts the corner. These are the directions that must become columns
    of their own. A run of tangent breaks packed inside a floored corner
    radius (the 2 um chamfer chord, a fillet's 0.1 mm inner arc) collapses to
    its centroid -- one column, not thirty needles.
    """

    edges = np.roll(plan[:, :2], -1, axis=0) - plan[:, :2]
    lengths = np.linalg.norm(edges, axis=1)
    directions = edges / np.where(lengths > 1.0e-12, lengths, 1.0)[:, None]
    dots = np.clip(
        np.sum(directions * np.roll(directions, 1, axis=0), axis=1), -1.0, 1.0
    )
    turn_deg = np.degrees(np.arccos(dots))
    flagged = np.flatnonzero((turn_deg > 1.0) & (lengths > 0.0))
    if len(flagged) == 0:
        return []
    # Group cyclically-consecutive flagged vertices.
    groups: list[list[int]] = [[int(flagged[0])]]
    for index in flagged[1:]:
        if int(index) == groups[-1][-1] + 1:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == len(plan) - 1:
        groups[0] = groups.pop() + groups[0]
    angles: list[float] = []
    for group in groups:
        points = plan[group, :2]
        extent = float(
            np.max(np.linalg.norm(points - points.mean(axis=0), axis=1))
        )
        if extent <= 0.5:
            probes = [points.mean(axis=0)]
        else:
            probes = [points[k] for k in range(len(points))]
        for probe in probes:
            angles.append(
                math.atan2(float(probe[1] - center[1]), float(probe[0] - center[0]))
            )
    return angles


def _insert_corner_columns(
    mouth: NDArray[np.float64],
    aligned: NDArray[np.float64],
    plan: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Add a column at every plan corner the mouth stations miss.

    The baffle strip and the aligned ring stay column-matched; the inserted
    mouth points interpolate the (smooth) mouth curve, and the inserted
    aligned points are exact ray hits, which land on the plan's corner. A
    corner that already coincides with a mouth station (the symmetric default
    box puts all eight at power-of-two angles) inserts nothing.
    """

    center = np.mean(mouth[:, :2], axis=0)
    corner_angles = _plan_corner_angles(plan, center)
    if not corner_angles:
        return mouth, aligned
    mouth_theta = np.arctan2(mouth[:, 1] - center[1], mouth[:, 0] - center[0])
    n = len(mouth)
    aligned = np.array(aligned, dtype=np.float64)
    insertions: list[tuple[int, float, NDArray[np.float64], NDArray[np.float64]]] = []
    taken: list[float] = []
    for theta in corner_angles:
        if any(abs(float(np.angle(np.exp(1j * (theta - t))))) < 1.0e-3 for t in taken):
            continue
        offsets = np.abs(np.angle(np.exp(1j * (mouth_theta - theta))))
        nearest = int(np.argmin(offsets))
        if float(offsets[nearest]) < 1.0e-3:
            # A corner that (nearly) coincides with a mouth station gets no
            # column of its own -- that would rule a hair-width baffle quad
            # against the station. Move the station's outer point onto the
            # corner instead: the mouth-side weld is untouched, and the
            # residual chord-cut at fine LOD (0.1 mm for a station 1 mrad off
            # the corner) goes with it.
            direction = np.asarray(
                (math.cos(theta), math.sin(theta)), dtype=np.float64
            )
            aligned[nearest] = _ray_cast(plan, center, direction)
            taken.append(theta)
            continue
        slot = None
        for i in range(n):
            span = float(np.angle(np.exp(1j * (mouth_theta[(i + 1) % n] - mouth_theta[i]))))
            local = float(np.angle(np.exp(1j * (theta - mouth_theta[i]))))
            if abs(span) < 1.0e-12:
                continue
            fraction = local / span
            if 0.0 < fraction < 1.0 and abs(local) <= abs(span):
                slot = (i, fraction)
                break
        if slot is None:
            continue
        i, fraction = slot
        mouth_point = mouth[i] + fraction * (mouth[(i + 1) % n] - mouth[i])
        direction = np.asarray((math.cos(theta), math.sin(theta)), dtype=np.float64)
        aligned_point = _ray_cast(plan, center, direction)
        taken.append(theta)
        insertions.append((i, fraction, mouth_point, aligned_point))
    if not insertions:
        return mouth, aligned
    insertions.sort(key=lambda item: (item[0], item[1]))
    mouth_out = list(map(np.asarray, mouth))
    aligned_out = list(map(np.asarray, aligned))
    for i, _fraction, mouth_point, aligned_point in reversed(insertions):
        mouth_out.insert(i + 1, mouth_point)
        aligned_out.insert(i + 1, aligned_point)
    return (
        np.asarray(mouth_out, dtype=np.float64),
        np.asarray(aligned_out, dtype=np.float64),
    )


def _ring_angle_table(
    ring: NDArray[np.float64], center: NDArray[np.float64], base: float | None = None
) -> tuple[NDArray[np.float64], int]:
    theta = np.arctan2(ring[:, 1] - center[1], ring[:, 0] - center[0])
    if base is None:
        start = 0
        first = float(theta[0])
    else:
        delta = np.abs(np.angle(np.exp(1j * (theta - base))))
        start = int(np.argmin(delta))
        first = base + float(np.angle(np.exp(1j * (theta[start] - base))))
    unwrapped = [first]
    previous = first
    for k in range(1, len(ring) + 1):
        value = float(theta[(start + k) % len(ring)])
        step = (value - previous) % math.tau
        if step > math.tau - 1.0e-9:
            step = 0.0
        unwrapped.append(unwrapped[-1] + step)
        previous = unwrapped[-1]
    return np.asarray(unwrapped, dtype=np.float64), start


def _zipper(
    ring_a: NDArray[np.float64], ring_b: NDArray[np.float64], offset_b: int
) -> list[int]:
    center = np.mean(ring_a[:, :2], axis=0)
    angles_a, start_a = _ring_angle_table(ring_a, center)
    angles_b, start_b = _ring_angle_table(ring_b, center, float(angles_a[0]))
    n_a, n_b = len(ring_a), len(ring_b)

    def index_a(k: int) -> int:
        return (start_a + k) % n_a

    def index_b(k: int) -> int:
        return offset_b + (start_b + k) % n_b

    triangles: list[int] = []
    i = j = 0
    while i < n_a or j < n_b:
        advance_a = j >= n_b or (i < n_a and angles_a[i + 1] <= angles_b[j + 1])
        if advance_a:
            triangles.extend((index_b(j), index_a(i), index_a(i + 1)))
            i += 1
        else:
            triangles.extend((index_b(j), index_a(i), index_b(j + 1)))
            j += 1
    return triangles


def _roundover_piece(
    role: str,
    first_ring: NDArray[np.float64],
    output_grid: NDArray[np.float64],
    reference_grid: NDArray[np.float64],
    center_xy: NDArray[np.float64],
    *,
    include_curvature: bool,
) -> tuple[PreviewSurfaceV1, dict[str, float]]:
    # Ring zero is ray-aligned to the horn; subsequent rings retain the canonical
    # enclosure plan. The first stitch is therefore the unequal-ring zipper used
    # by the browser tessellator, while all following rings are modulo-closed.
    native_normals = analytic_grid_normals(reference_grid, closed_phi=True)
    output_normals = resample_grid_vectors(
        native_normals, output_grid.shape[:2], closed_phi=True
    )
    first_ref_normals = resample_grid_vectors(
        native_normals[:2], (2, len(first_ring)), closed_phi=True
    )[0]
    output_mean = output_principal = first_ref_mean = first_ref_principal = None
    if include_curvature:
        native_mean, native_principal = analytic_grid_curvature(
            reference_grid, closed_phi=True
        )
        output_curvature = resample_parametric_grid(
            np.stack((native_mean, native_principal), axis=2),
            output_grid.shape[:2],
            closed_phi=True,
        )
        first_curvature = resample_parametric_grid(
            np.stack((native_mean[:2], native_principal[:2]), axis=2),
            (2, len(first_ring)),
            closed_phi=True,
        )[0]
        output_mean = output_curvature[:, :, 0]
        output_principal = output_curvature[:, :, 1]
        first_ref_mean = first_curvature[:, 0]
        first_ref_principal = first_curvature[:, 1]
    native_hint = output_grid.copy()
    native_hint[:, :, 0] -= center_xy[0]
    native_hint[:, :, 1] -= center_xy[1]
    native_hint[:, :, 2] = 0.0
    if float(np.median(np.sum(output_normals * native_hint, axis=2))) < 0.0:
        output_normals = -output_normals
        first_ref_normals = -first_ref_normals
        if output_mean is not None:
            output_mean = -output_mean
            output_principal = -output_principal
            first_ref_mean = -first_ref_mean
            first_ref_principal = -first_ref_principal
    positions = [first_ring, *list(output_grid[1:])]
    normals = [first_ref_normals, *list(output_normals[1:])]
    curvature_mean = (
        None
        if first_ref_mean is None or output_mean is None
        else np.concatenate((first_ref_mean, output_mean[1:].reshape(-1)))
    )
    curvature_principal = (
        None
        if first_ref_principal is None or output_principal is None
        else np.concatenate(
            (first_ref_principal, output_principal[1:].reshape(-1))
        )
    )
    offsets = [0]
    for ring in positions[:-1]:
        offsets.append(offsets[-1] + len(ring))
    triangles = _zipper(positions[0], positions[1], offsets[1])
    for level in range(1, len(positions) - 1):
        n_phi = len(positions[level])
        for ip in range(n_phi):
            ip1 = (ip + 1) % n_phi
            a0, a1 = offsets[level] + ip, offsets[level] + ip1
            b0, b1 = offsets[level + 1] + ip, offsets[level + 1] + ip1
            triangles.extend((a0, a1, b1, a0, b1, b0))
    position_array = np.vstack(positions)
    normal_array = np.vstack(normals)
    oriented = _orient_indices_to_normals(
        role,
        position_array,
        np.asarray(triangles, dtype=np.uint32),
        normal_array,
    )
    surface = PreviewSurfaceV1(
        role=role,
        positions=position_array,
        indices=oriented.indices,
        normals=normal_array,
        shading="smooth",
        normal_method="analytic-parametric",
        closed_phi=True,
        curvature_mean=curvature_mean,
        curvature_principal=curvature_principal,
        metadata=_orientation_metadata(oriented),
    )
    fidelity = estimate_grid_fidelity(
        output_grid, reference_grid, output_normals, closed_phi=True
    )
    return surface, fidelity


def _combine_surfaces(role: str, parts: list[PreviewSurfaceV1]) -> PreviewSurfaceV1:
    positions: list[NDArray[np.float64]] = []
    normals: list[NDArray[np.float64]] = []
    indices: list[NDArray[np.uint32]] = []
    curvature_mean: list[NDArray[np.float64]] = []
    curvature_principal: list[NDArray[np.float64]] = []
    offset = 0
    for part in parts:
        positions.append(part.positions)
        normals.append(part.normals)
        indices.append(part.indices + np.uint32(offset))
        if part.curvature_mean is not None and part.curvature_principal is not None:
            curvature_mean.append(part.curvature_mean)
            curvature_principal.append(part.curvature_principal)
        offset += len(part.positions)
    return PreviewSurfaceV1(
        role=role,
        positions=np.vstack(positions),
        indices=np.concatenate(indices),
        normals=np.vstack(normals),
        shading=parts[0].shading,
        normal_method=parts[0].normal_method,
        closed_phi=all(part.closed_phi for part in parts),
        curvature_mean=(
            np.concatenate(curvature_mean) if len(curvature_mean) == len(parts) else None
        ),
        curvature_principal=(
            np.concatenate(curvature_principal)
            if len(curvature_principal) == len(parts)
            else None
        ),
    )


def _hard_side_surface(
    front: NDArray[np.float64],
    back: NDArray[np.float64],
    *,
    include_curvature: bool,
) -> PreviewSurfaceV1:
    """Build one planar quad per plan edge, duplicating every hard seam."""

    parts: list[PreviewSurfaceV1] = []
    for index in range(len(front)):
        next_index = (index + 1) % len(front)
        edge = front[next_index, :2] - front[index, :2]
        outward = np.asarray((edge[1], -edge[0], 0.0), dtype=np.float64)
        outward /= np.linalg.norm(outward)
        parts.append(
            _flat_strip(
                "enclosure.side",
                front[[index, next_index]],
                back[[index, next_index]],
                tuple(float(value) for value in outward),
                closed_phi=False,
                include_curvature=include_curvature,
            )
        )
    return _combine_surfaces("enclosure.side", parts)


def _faceted_band(
    role: str,
    near: NDArray[np.float64],
    far: NDArray[np.float64],
    center: NDArray[np.float64],
    *,
    include_curvature: bool,
) -> PreviewSurfaceV1:
    """Build one planar quad per plan segment across a ruled band.

    A chamfer is not a curved surface being approximated -- it is a finite set
    of planes meeting at real tangent breaks. Ruling it in the plan sampler's
    own parameterisation keeps every column inside a single plane, so each quad
    carries that plane's exact normal and the band hard-shades correctly however
    unevenly the plan itself is sampled. ``center`` is any interior point; it
    only picks the outward sign.
    """

    parts: list[PreviewSurfaceV1] = []
    for index in range(len(near)):
        next_index = (index + 1) % len(near)
        near_edge = near[next_index] - near[index]
        far_edge = far[next_index] - far[index]
        near_len = float(np.linalg.norm(near_edge))
        far_len = float(np.linalg.norm(far_edge))
        # A corner column carries the plan sampler's floored corner radius: a
        # micron-scale chord on one ring against the real corner facet on the
        # other. The solver builds a single corner *triangle* there
        # (``build_sector``'s chamfer is two side parallelograms plus one
        # corner triangle per quadrant), and emitting the trapezoid instead
        # split it into that triangle plus a ~10000:1 sliver.
        if near_len < _DEGENERATE_COLUMN_MM and far_len >= _DEGENERATE_COLUMN_MM:
            apex = 0.5 * (near[index] + near[next_index])
            triangle = np.stack((apex, far[index], far[next_index]))
            normal = np.cross(far_edge, apex - far[index])
        elif far_len < _DEGENERATE_COLUMN_MM and near_len >= _DEGENERATE_COLUMN_MM:
            apex = 0.5 * (far[index] + far[next_index])
            triangle = np.stack((near[index], near[next_index], apex))
            normal = np.cross(near_edge, apex - near[index])
        else:
            triangle = None
            normal = np.cross(near_edge, far[index] - near[index])
        length = float(np.linalg.norm(normal))
        if length <= 1.0e-12:
            # A collapsed column (coincident plan samples, or a ruling parallel
            # to the plan edge) spans no area and cannot orient itself.
            continue
        normal = normal / length
        centroid = 0.25 * (near[index] + near[next_index] + far[index] + far[next_index])
        if float(np.dot(normal, centroid - center)) < 0.0:
            normal = -normal
        if triangle is not None:
            parts.append(
                _flat_triangle(
                    role,
                    triangle,
                    tuple(float(value) for value in normal),
                    include_curvature=include_curvature,
                )
            )
        else:
            parts.append(
                _flat_strip(
                    role,
                    near[[index, next_index]],
                    far[[index, next_index]],
                    tuple(float(value) for value in normal),
                    closed_phi=False,
                    include_curvature=include_curvature,
                )
            )
    if not parts:
        raise ValueError(f"{role}: every faceted column collapsed")
    return _combine_surfaces(role, parts)


def _analytic_fillet(enclosure: Mapping[str, Any]) -> bool:
    """True when the fillet is a rounded box edge with a closed-form model.

    A rounded-rectangle fillet sweeps its plan by ``inset = edge*(1-sin(theta))``
    and ``radius = edge*sin(theta)`` while dropping ``depth*(1-cos(theta))``, so
    every corner arc keeps the *same* centre, ``edge`` in from the box corner.
    That makes each corner exactly one octant of the spheroid with semi-axes
    ``(edge, edge, depth)`` about that centre and each side exactly one quarter
    of an elliptic cylinder -- see :func:`_fillet_pieces`. Ellipse and
    superellipse plans have no such fixed centres and keep the sampled path.
    """

    return int(enclosure["plan_type"]) == 1 and int(enclosure["edge_type"]) == 1


def _fillet_corner_frames(
    enclosure: Mapping[str, Any]
) -> list[tuple[float, float, float]]:
    """The four corner-arc centres and the phi each octant starts at."""

    bounds = enclosure["bounds"]
    edge = float(enclosure["edge_mm"])
    x1 = float(bounds["bx1"]) - edge
    x0 = float(bounds["bx0"]) + edge
    y1 = float(bounds["by1"]) - edge
    y0 = float(bounds["by0"]) + edge
    return [
        (x1, y1, 0.0),
        (x0, y1, 0.5 * math.pi),
        (x0, y0, math.pi),
        (x1, y0, 1.5 * math.pi),
    ]


def _fillet_inset_rectangle(
    enclosure: Mapping[str, Any], z: float
) -> NDArray[np.float64]:
    """The sharp rectangle a rounded-box fillet is tangent to.

    ``_plan_ring`` floors its corner radius at 0.1 mm because
    ``sample_rounded_rect`` changes vertex count below 1e-3 mm, but the fillet
    genuinely runs out to a sharp corner there -- that is where the solver
    starts its roundover, and where :func:`_fillet_pieces` puts each octant's
    pole. The baffle has to end on the same curve, or the 0.041 mm between the
    floored arc and the corner belongs to neither surface.
    """

    return np.asarray(
        [(x, y, float(z)) for x, y, _phi in _fillet_corner_frames(enclosure)],
        dtype=np.float64,
    )


def _fillet_phi_intervals(
    theta: float, edge: float, depth: float, chord: float, normal_deg: float, cap: int
) -> int:
    """How many phi intervals a spheroid octant's row at ``theta`` really needs.

    The octant's unit normal is proportional to ``(sin(theta) cos(phi)/edge,
    sin(theta) sin(phi)/edge, cos(theta)/depth)``: turning ``phi`` rotates only
    the tangential part, so the normal step closes on zero towards the pole
    exactly as the arc radius ``edge*sin(theta)`` does. Spending the equator's
    sample count on every row -- what one fixed ``corner_intervals`` does -- is
    what put thirty-two samples on a 0.1 mm arc and then fanned them onto the
    next ring. Zero intervals means the row is the pole itself.
    """

    radius = edge * math.sin(theta)
    if radius <= 0.0:
        return 0
    by_chord = 1
    if radius > chord:
        half = math.acos(float(np.clip(1.0 - chord / radius, -1.0, 1.0)))
        by_chord = int(math.ceil(0.5 * math.pi / max(2.0 * half, 1.0e-12)))
    tangential = math.sin(theta) / edge
    axial = math.cos(theta) / depth
    by_normal = 1
    scale = tangential * tangential + axial * axial
    if tangential > 0.0:
        cosine = 1.0 - (1.0 - math.cos(math.radians(normal_deg))) * scale / (
            tangential * tangential
        )
        if cosine > -1.0:
            step = math.acos(float(np.clip(cosine, -1.0, 1.0)))
            by_normal = int(math.ceil(0.5 * math.pi / max(step, 1.0e-12)))
    return max(1, min(int(cap), max(by_chord, by_normal)))


def _fillet_patch(
    role: str,
    rows: list[NDArray[np.float64]],
    normals: list[NDArray[np.float64]],
    params: list[NDArray[np.float64]],
    curvature: list[NDArray[np.float64]] | None,
) -> PreviewSurfaceV1:
    """Stitch rows that may hold different sample counts into one patch.

    Rows carry a shared parameter, so the stitch is a two-pointer merge on it:
    a row that collapses to a single sample (the octant's pole) becomes a fan
    rather than a column of needles, and a row that gains samples over its
    neighbour picks them up one triangle at a time.
    """

    offsets = [0]
    for row in rows[:-1]:
        offsets.append(offsets[-1] + len(row))
    triangles: list[int] = []
    for level in range(len(rows) - 1):
        lower, upper = params[level], params[level + 1]
        base_l, base_u = offsets[level], offsets[level + 1]
        i = j = 0
        while i < len(lower) - 1 or j < len(upper) - 1:
            take_lower = j >= len(upper) - 1 or (
                i < len(lower) - 1 and lower[i + 1] <= upper[j + 1]
            )
            if take_lower:
                triangles.extend((base_l + i, base_l + i + 1, base_u + j))
                i += 1
            else:
                triangles.extend((base_l + i, base_u + j + 1, base_u + j))
                j += 1
    positions = np.vstack(rows)
    normal_array = np.vstack(normals)
    oriented = _orient_indices_to_normals(
        role, positions, np.asarray(triangles, dtype=np.uint32), normal_array
    )
    return PreviewSurfaceV1(
        role=role,
        positions=positions,
        indices=oriented.indices,
        normals=normal_array,
        shading="smooth",
        normal_method="analytic-parametric",
        closed_phi=False,
        curvature_mean=(None if curvature is None else np.concatenate(curvature[0])),
        curvature_principal=(
            None if curvature is None else np.concatenate(curvature[1])
        ),
        metadata=_orientation_metadata(oriented),
    )


def _fillet_pieces(
    enclosure: Mapping[str, Any],
    *,
    z_ref: float,
    axial_sign: float,
    rows: int,
    corner_cap: int,
    chord_target: float,
    normal_target: float,
    include_curvature: bool,
) -> tuple[list[PreviewSurfaceV1], dict[str, float]]:
    """Emit a rounded-box fillet as four cylinder strips and four octants.

    ``axial_sign`` is +1 for the front band (which drops away from ``z_ref``
    towards the sides) and -1 for the back one. Positions and normals are the
    closed-form surface, not differences of a sampled grid, so the returned
    fidelity is a measurement of the emitted triangles against that surface
    rather than against a resampling of themselves.
    """

    edge = float(enclosure["edge_mm"])
    depth = float(enclosure["edge_depth"])
    frames = _fillet_corner_frames(enclosure)
    theta = np.linspace(0.0, 0.5 * math.pi, rows + 1)
    # A patch bows away from its chords in both parameters at once and both
    # bows point the same way, so the two budgets add rather than compete. The
    # row count is fixed by the caller and is normal-step bound at every LOD,
    # which leaves the whole chord budget to be split; giving phi half of it
    # keeps the measured total inside what was asked for.
    phi_chord_target = 0.5 * chord_target
    intervals = [
        _fillet_phi_intervals(
            float(value), edge, depth, phi_chord_target, normal_target, corner_cap
        )
        for value in theta
    ]
    intervals[-1] = int(corner_cap)
    for index in range(1, len(intervals)):
        intervals[index] = max(intervals[index], intervals[index - 1])

    def surface_point(cx: float, cy: float, t: float, phi: float) -> NDArray[np.float64]:
        radius = edge * math.sin(t)
        return np.asarray(
            (
                cx + radius * math.cos(phi),
                cy + radius * math.sin(phi),
                z_ref - axial_sign * depth * (1.0 - math.cos(t)),
            ),
            dtype=np.float64,
        )

    def surface_normal(t: float, phi: float) -> NDArray[np.float64]:
        vector = np.asarray(
            (
                math.sin(t) * math.cos(phi) / edge,
                math.sin(t) * math.sin(phi) / edge,
                axial_sign * math.cos(t) / depth,
            ),
            dtype=np.float64,
        )
        return vector / np.linalg.norm(vector)

    def curvatures(t: float, *, ruled: bool) -> tuple[float, float]:
        # Meridian and parallel curvature of the spheroid of revolution with
        # profile ``(edge sin t, depth cos t)``; a side strip is the same
        # meridian swept along a straight ruling, so its parallel curvature is
        # zero. Positive is convex towards the outward normal above.
        root = math.hypot(edge * math.cos(t), depth * math.sin(t))
        meridian = edge * depth / max(root**3, 1.0e-30)
        parallel = 0.0 if ruled else depth / max(edge * root, 1.0e-30)
        mean = 0.5 * (meridian + parallel)
        principal = meridian if abs(meridian) >= abs(parallel) else parallel
        return mean, principal

    pieces: list[PreviewSurfaceV1] = []
    for index, (cx, cy, phi0) in enumerate(frames):
        corner_rows: list[NDArray[np.float64]] = []
        corner_normals: list[NDArray[np.float64]] = []
        corner_params: list[NDArray[np.float64]] = []
        corner_mean: list[NDArray[np.float64]] = []
        corner_principal: list[NDArray[np.float64]] = []
        for level, t in enumerate(theta):
            count = intervals[level]
            fractions = (
                np.zeros(1, dtype=np.float64)
                if count == 0
                else np.linspace(0.0, 1.0, count + 1)
            )
            phis = phi0 + fractions * 0.5 * math.pi
            corner_rows.append(
                np.asarray(
                    [surface_point(cx, cy, float(t), float(p)) for p in phis],
                    dtype=np.float64,
                )
            )
            corner_normals.append(
                np.asarray(
                    [surface_normal(float(t), float(p)) for p in phis],
                    dtype=np.float64,
                )
            )
            corner_params.append(fractions)
            mean, principal = curvatures(float(t), ruled=False)
            corner_mean.append(np.full(len(phis), mean, dtype=np.float64))
            corner_principal.append(np.full(len(phis), principal, dtype=np.float64))
        pieces.append(
            _fillet_patch(
                "enclosure.roundover",
                corner_rows,
                corner_normals,
                corner_params,
                (corner_mean, corner_principal) if include_curvature else None,
            )
        )

        # The side that leaves this corner: a quarter of an elliptic cylinder
        # ruled between this octant's end tangent line and the next octant's
        # start tangent line. Two columns describe it exactly.
        nx, ny, next_phi0 = frames[(index + 1) % len(frames)]
        side_rows: list[NDArray[np.float64]] = []
        side_normals: list[NDArray[np.float64]] = []
        side_params: list[NDArray[np.float64]] = []
        side_mean: list[NDArray[np.float64]] = []
        side_principal: list[NDArray[np.float64]] = []
        for t in theta:
            end_phi = phi0 + 0.5 * math.pi
            side_rows.append(
                np.stack(
                    (
                        surface_point(cx, cy, float(t), end_phi),
                        surface_point(nx, ny, float(t), next_phi0),
                    )
                )
            )
            normal = surface_normal(float(t), end_phi)
            side_normals.append(np.stack((normal, normal)))
            side_params.append(np.asarray((0.0, 1.0), dtype=np.float64))
            mean, principal = curvatures(float(t), ruled=True)
            side_mean.append(np.full(2, mean, dtype=np.float64))
            side_principal.append(np.full(2, principal, dtype=np.float64))
        pieces.append(
            _fillet_patch(
                "enclosure.roundover",
                side_rows,
                side_normals,
                side_params,
                (side_mean, side_principal) if include_curvature else None,
            )
        )

    # Measure the emitted band against the closed-form surface it was built
    # from. All four octants are congruent and the sides share their meridian,
    # so one meridian and one octant bound the whole band; a chord is deepest
    # at its parameter midpoint. This replaces comparing the grid against a
    # reference built at the same interval count, which measured nothing.
    def angle_between(
        first: NDArray[np.float64], second: NDArray[np.float64]
    ) -> float:
        return math.degrees(
            math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0)))
        )

    origin = frames[0]
    meridian_chord = 0.0
    normal_step = 0.0
    for level in range(rows):
        lower, upper = float(theta[level]), float(theta[level + 1])
        emitted = 0.5 * (
            surface_point(origin[0], origin[1], lower, origin[2])
            + surface_point(origin[0], origin[1], upper, origin[2])
        )
        meridian_chord = max(
            meridian_chord,
            float(
                np.linalg.norm(
                    surface_point(
                        origin[0], origin[1], 0.5 * (lower + upper), origin[2]
                    )
                    - emitted
                )
            ),
        )
        normal_step = max(
            normal_step,
            angle_between(surface_normal(lower, 0.0), surface_normal(upper, 0.0)),
        )
    arc_chord = 0.0
    for level, count in enumerate(intervals):
        if count <= 0:
            continue
        step = 0.5 * math.pi / count
        arc_chord = max(
            arc_chord,
            edge * math.sin(float(theta[level])) * (1.0 - math.cos(0.5 * step)),
        )
        normal_step = max(
            normal_step,
            angle_between(
                surface_normal(float(theta[level]), 0.0),
                surface_normal(float(theta[level]), step),
            ),
        )
    fidelity = {
        "max_chord_error_mm": max(
            meridian_chord + arc_chord, float(np.finfo(np.float64).eps)
        ),
        "max_normal_step_deg": normal_step,
        "reference_density_multiplier": 4,
    }
    return pieces, fidelity


def _enclosure_surfaces(
    enclosure: Mapping[str, Any],
    mouth: NDArray[np.float64],
    roundover_intervals: int,
    plan_corner_intervals: int,
    *,
    chord_target: float,
    normal_target: float,
    include_rear: bool,
    include_curvature: bool,
) -> tuple[list[PreviewSurfaceV1], dict[str, dict[str, float]]]:
    bounds = enclosure["bounds"]
    depth = float(enclosure["edge_depth"])
    rounded_edge = int(enclosure["edge_type"]) == 1
    z_front = float(bounds["z_front"])
    z_back = float(bounds["z_back"])
    center_xy = np.asarray((float(bounds["cx"]), float(bounds["cy"])), dtype=np.float64)

    analytic_fillet = depth > 0.0 and _analytic_fillet(enclosure)
    front_native = (
        _fillet_inset_rectangle(enclosure, z_front)
        if analytic_fillet
        else _plan_ring(enclosure, z_front, 0.0, corner_intervals=plan_corner_intervals)
    )
    front_aligned = _ray_aligned_ring(front_native, mouth)
    # The aligned ring only preserves the plan where the mouth stations sample
    # it. On an asymmetric box the plan corners fall between stations, the
    # aligned polyline chord-cuts them, and the flat baffle plane between that
    # chord and the corner is covered by neither the baffle nor the edge band:
    # a real hole (chamfer) or an off-surface ruling (fillet). Give every
    # missed corner its own column in both rings.
    mouth, front_aligned = _insert_corner_columns(mouth, front_aligned, front_native)
    # One annulus, not two. An enclosure horn has no wall end face to name --
    # ``config_builder`` forces wall thickness to zero and ``HornEnclosure``
    # forbids outer points, so the baffle runs unbroken from the mouth to the
    # edge treatment. Splitting it at the midpoint used to emit half of it as
    # ``mouth_rim``, which the renderer paints in the horn material and outlines
    # in edge mode: a hard colour seam and a drawn feature line partway across a
    # surface that is flat and continuous.
    surfaces = [
        _flat_strip(
            "enclosure.front",
            mouth,
            front_aligned,
            (0.0, 0.0, 1.0),
            closed_phi=True,
            include_curvature=include_curvature,
        ),
    ]
    fidelity: dict[str, dict[str, float]] = {}

    if depth > 0.0:
        def front_grid(intervals: int) -> NDArray[np.float64]:
            rings = []
            for fraction in np.linspace(0.0, 1.0, intervals + 1):
                if rounded_edge:
                    theta = float(fraction) * math.pi / 2.0
                    axial_t = 1.0 - math.cos(theta)
                    radial_t = math.sin(theta)
                else:
                    axial_t = radial_t = float(fraction)
                rings.append(
                    _plan_ring(
                        enclosure,
                        z_front - axial_t * depth,
                        radial_t,
                        corner_intervals=plan_corner_intervals,
                    )
                )
            return np.asarray(rings, dtype=np.float64)

        def back_grid(intervals: int) -> NDArray[np.float64]:
            rings = []
            for fraction in np.linspace(0.0, 1.0, intervals + 1):
                if rounded_edge:
                    theta = float(fraction) * math.pi / 2.0
                    axial_t = math.sin(theta)
                    radial_t = math.cos(theta)
                else:
                    axial_t = float(fraction)
                    radial_t = 1.0 - float(fraction)
                rings.append(
                    _plan_ring(
                        enclosure,
                        z_back + (1.0 - axial_t) * depth,
                        radial_t,
                        corner_intervals=plan_corner_intervals,
                    )
                )
            return np.asarray(rings, dtype=np.float64)

        edge_intervals = roundover_intervals if rounded_edge else 1
        front_out = front_grid(edge_intervals)
        back_out = back_grid(edge_intervals)
        if _faceted_edge(enclosure):
            # A rounded-rectangle chamfer is a ruled band between two
            # piecewise-linear plans, so it is exactly a fan of planes. Rule it
            # in the plan's own parameterisation -- the ray-aligned front ring
            # was welded to the baffle but forced an unequal-ring zipper that
            # fanned dozens of mouth stations onto one plan sample, and the
            # resulting slivers were then smooth-shaded across tangent breaks
            # that are real. The inner boundary still lands on the baffle's
            # outer edge, now as coincident points along it rather than shared
            # ones.
            center_xyz = np.asarray(
                (center_xy[0], center_xy[1], 0.5 * (z_front + z_back)), dtype=np.float64
            )
            surfaces.append(
                _combine_surfaces(
                    "enclosure.roundover",
                    [
                        _faceted_band(
                            "enclosure.roundover",
                            front_out[0],
                            front_out[-1],
                            center_xyz,
                            include_curvature=include_curvature,
                        ),
                        _faceted_band(
                            "enclosure.roundover",
                            back_out[-1],
                            back_out[0],
                            center_xyz,
                            include_curvature=include_curvature,
                        ),
                    ],
                )
            )
            # Planes, exactly represented. The 45 degree steps between adjacent
            # facets are the geometry itself, not a sampling error, and no
            # subdivision reduces them.
            fidelity["enclosure.roundover"] = {
                "max_chord_error_mm": 0.0,
                "max_normal_step_deg": 0.0,
                "reference_density_multiplier": 1,
            }
        elif analytic_fillet:
            # A rounded-box fillet has a closed form, so it does not need the
            # ray-aligned first ring and the unequal-ring zipper that welded it
            # to the baffle. That stitch fanned the mouth's evenly spread
            # stations onto a ring carrying one sample per straight side, which
            # is where the band's 7000:1 needles came from; the plan sampler
            # then spent a full corner_intervals on every ring, including the
            # 0.1 mm floored one. The band's inner boundary still lands on the
            # baffle's outer edge -- as coincident points along the same inset
            # rectangle rather than shared ones, exactly as the chamfer does.
            front_pieces, front_fidelity = _fillet_pieces(
                enclosure,
                z_ref=z_front,
                axial_sign=1.0,
                rows=edge_intervals,
                corner_cap=plan_corner_intervals,
                chord_target=chord_target,
                normal_target=normal_target,
                include_curvature=include_curvature,
            )
            back_pieces, back_fidelity = _fillet_pieces(
                enclosure,
                z_ref=z_back,
                axial_sign=-1.0,
                rows=edge_intervals,
                corner_cap=plan_corner_intervals,
                chord_target=chord_target,
                normal_target=normal_target,
                include_curvature=include_curvature,
            )
            surfaces.append(
                _combine_surfaces(
                    "enclosure.roundover", [*front_pieces, *back_pieces]
                )
            )
            fidelity["enclosure.roundover"] = {
                key: max(front_fidelity[key], back_fidelity[key])
                for key in ("max_chord_error_mm", "max_normal_step_deg")
            } | {"reference_density_multiplier": 4}
        else:
            # An ellipse or superellipse plan has no fixed corner centres, so
            # its roundover stays a sampled grid. Its reference must be denser
            # than the emitted grid or the comparison measures nothing.
            front_surface, front_fidelity = _roundover_piece(
                "enclosure.roundover",
                front_aligned,
                front_out,
                front_grid(2 * edge_intervals),
                center_xy,
                include_curvature=include_curvature,
            )
            back_surface, back_fidelity = _roundover_piece(
                "enclosure.roundover",
                back_out[0],
                back_out,
                back_grid(2 * edge_intervals),
                center_xy,
                include_curvature=include_curvature,
            )
            surfaces.append(
                _combine_surfaces("enclosure.roundover", [front_surface, back_surface])
            )
            fidelity["enclosure.roundover"] = {
                key: max(front_fidelity[key], back_fidelity[key])
                for key in ("max_chord_error_mm", "max_normal_step_deg")
            } | {"reference_density_multiplier": 4}
        side_front = front_out[-1]
        side_back = back_out[0]
    else:
        side_front = front_native
        side_back = _plan_ring(
            enclosure, z_back, 0.0, corner_intervals=plan_corner_intervals
        )

    side_grid = np.stack((side_front, side_back), axis=0)
    # Four axial reference intervals measure the exact canonical extrusion.
    side_ref = np.stack(
        [
            side_front + (side_back - side_front) * fraction
            for fraction in np.linspace(0.0, 1.0, 5)
        ],
        axis=0,
    )
    if int(enclosure["edge_type"]) == 2:
        side_surface = _hard_side_surface(
            side_front, side_back, include_curvature=include_curvature
        )
        side_fidelity = _plan_fidelity(enclosure, plan_corner_intervals)
    else:
        side_surface, side_fidelity = _smooth_grid_surface(
            "enclosure.side",
            side_grid,
            side_ref,
            closed_phi=True,
            orientation_hint=np.stack(
                (
                    side_grid[:, :, 0] - center_xy[0],
                    side_grid[:, :, 1] - center_xy[1],
                    np.zeros(side_grid.shape[:2], dtype=np.float64),
                ),
                axis=2,
            ),
            include_curvature=include_curvature,
        )
    surfaces.append(side_surface)
    fidelity["enclosure.side"] = side_fidelity
    if include_rear:
        if analytic_fillet:
            # Same sharp tangent rectangle the back band's poles sit on.
            rear_ring = _fillet_inset_rectangle(enclosure, z_back)
        elif depth > 0.0:
            rear_ring = back_out[-1]
        else:
            rear_ring = side_back
        surfaces.append(
            _flat_cap(
                "enclosure.rear",
                rear_ring,
                (0.0, 0.0, -1.0),
                closed_phi=True,
                include_curvature=include_curvature,
                # The rear ring carries the sampler's floored corners (a 2 um
                # chamfer chord; a fillet's 0.1 mm arc in dozens of samples).
                # Fanned against a center hundreds of mm away they become
                # 50000:1 slivers. The real rear face corner is sharp.
                simplify_tolerance=0.15,
            )
        )
    return surfaces, fidelity


def _even_indices(size: int, count: int, *, closed: bool) -> list[int]:
    if closed:
        count = min(size, max(3, int(count)))
        return sorted({int(index * size // count) for index in range(count)})
    count = min(size, max(2, int(count)))
    return sorted(
        {int(round(index * (size - 1) / (count - 1))) for index in range(count)}
    )


def _band_dihedral_deg(
    rings: list[NDArray[np.float64]], *, closed_phi: bool
) -> float:
    """Median dihedral angle between the two bands three consecutive rings span.

    Measured on the rings that are actually emitted, not on the canonical
    reference: a tangent break the shipped mesh does not resolve cannot shade
    wrong, and one it does resolve must not be smoothed over. This is the same
    criterion the renderer uses to decide a feature edge.
    """

    def band_normals(
        near: NDArray[np.float64], far: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        along = (
            np.roll(near, -1, axis=0) - near
            if closed_phi
            else np.diff(near, axis=0, append=near[-1:])
        )
        normals = np.cross(along, far - near)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / np.where(lengths > 1.0e-12, lengths, 1.0)

    lower = band_normals(rings[0], rings[1])
    upper = band_normals(rings[1], rings[2])
    return float(
        np.median(
            np.degrees(
                np.arccos(np.clip(np.abs(np.sum(lower * upper, axis=1)), -1.0, 1.0))
            )
        )
    )


def _outer_shell_surfaces(
    master: NDArray[np.float64],
    t_indices: NDArray[np.int64],
    phi_indices: NDArray[np.int64],
    *,
    closed_phi: bool,
    t_coordinates: NDArray[np.float64] | None,
    phi_coordinates: NDArray[np.float64] | None,
    include_curvature: bool,
) -> list[PreviewSurfaceV1]:
    """Build the outer wall, splitting the shading at the throat jog.

    ``_outer_offset_shell`` does not put the shell's first station on the
    offset surface: it squares the throat off into a flat rear face, radius
    ``r0 + wall`` at ``z0 - wall``. The band that ring forms with the next
    station therefore meets the offset surface at a real crease, and a station
    on a crease needs two normals -- one per face. Shipping one, taken from a
    central difference straddling the crease, put that ring up to 83 degrees
    off both faces it borders and drew a shading ring around the throat.

    Overriding the ring's normal cannot fix it; whichever face it is made to
    agree with, the other still disagrees. So the jog band becomes its own role
    and carries its own copy of the crease ring, which is where this module
    puts every other hard boundary (``mouth_rim``, ``wall.rear_cap``): the
    normals inside each role stay smooth and the crease falls between them.
    """

    def normals_for(rows: slice) -> NDArray[np.float64]:
        return analytic_grid_normals(
            master[rows],
            closed_phi=closed_phi,
            t_coordinates=None if t_coordinates is None else t_coordinates[rows],
            phi_coordinates=(
                None if phi_coordinates is None else phi_coordinates[rows]
            ),
        )

    def curvature_for(rows: slice) -> tuple[
        NDArray[np.float64] | None, NDArray[np.float64] | None
    ]:
        if not include_curvature:
            return None, None
        return analytic_grid_curvature(
            master[rows],
            closed_phi=closed_phi,
            t_coordinates=None if t_coordinates is None else t_coordinates[rows],
            phi_coordinates=(
                None if phi_coordinates is None else phi_coordinates[rows]
            ),
        )

    def unsplit() -> list[PreviewSurfaceV1]:
        mean, principal = curvature_for(slice(None))
        return [
            _grid_surface_from_selection(
                "horn.outer",
                master,
                normals_for(slice(None)),
                t_indices,
                phi_indices,
                closed_phi=closed_phi,
                curvature_mean=mean,
                curvature_principal=principal,
            )
        ]

    crease = 0.0
    if len(master) >= 4 and len(t_indices) >= 3:
        rings = [master[np.ix_([int(t_indices[k])], phi_indices)][0] for k in range(3)]
        crease = _band_dihedral_deg(rings, closed_phi=closed_phi)
    if crease < _THROAT_JOG_CREASE_DEG:
        return unsplit()

    band_rows = np.asarray(t_indices[:2], dtype=np.int64)
    band_master = master[band_rows]
    band_normals = analytic_grid_normals(
        band_master,
        closed_phi=closed_phi,
        t_coordinates=None if t_coordinates is None else t_coordinates[band_rows],
        phi_coordinates=(
            None if phi_coordinates is None else phi_coordinates[band_rows]
        ),
    )
    band_mean = band_principal = None
    if include_curvature:
        # Two stations carry no second difference in t, so the band's meridian
        # curvature is reported as the straight ruling it is emitted as.
        band_mean = np.zeros(band_master.shape[:2], dtype=np.float64)
        band_principal = np.zeros(band_master.shape[:2], dtype=np.float64)
    shell_mean, shell_principal = curvature_for(slice(1, None))
    shell_normals = np.array(normals_for(slice(None)), dtype=np.float64)
    shell_normals[1:] = normals_for(slice(1, None))
    padded_mean = padded_principal = None
    if shell_mean is not None and shell_principal is not None:
        padded_mean = np.zeros(master.shape[:2], dtype=np.float64)
        padded_principal = np.zeros(master.shape[:2], dtype=np.float64)
        padded_mean[1:] = shell_mean
        padded_principal[1:] = shell_principal
    try:
        return [
            _grid_surface_from_selection(
                "horn.outer",
                master,
                shell_normals,
                np.asarray(t_indices[1:], dtype=np.int64),
                phi_indices,
                closed_phi=closed_phi,
                curvature_mean=padded_mean,
                curvature_principal=padded_principal,
            ),
            _grid_surface_from_selection(
                "wall.throat_band",
                band_master,
                band_normals,
                np.asarray((0, 1), dtype=np.int64),
                phi_indices,
                closed_phi=closed_phi,
                curvature_mean=band_mean,
                curvature_principal=band_principal,
            ),
        ]
    except ValueError:
        # A sharp morph corner has no defined offset direction, so the shell
        # can carry a few facets tipped just past perpendicular there. The
        # whole shell forgives them as the negligible share of its area they
        # are; two rows cut out of it do not have the area to. Splitting is a
        # shading improvement, never a reason to lose the surface, so a band
        # that cannot be wound consistently on its own gives the shading back
        # to the shell it came from.
        return unsplit()


def _nearest_indices(values: NDArray[np.float64], targets: list[float]) -> list[int]:
    return sorted(
        {
            int(np.argmin(np.abs(values - float(target))))
            for target in targets
            if math.isfinite(float(target))
        }
    )


def _semantic_t_stations(
    config: Mapping[str, Any], output: Mapping[str, Any], t_values: NDArray[np.float64]
) -> tuple[list[int], list[str], list[str]]:
    targets = [0.0, 1.0]
    inserted = ["throat", "mouth"]
    unavailable = ["OSSE extension/slot boundaries when expression-valued"]
    params = output["params"]

    morph_start = params.get("morphFixed")
    if isinstance(morph_start, (int, float)) and 0.0 < float(morph_start) < 1.0:
        targets.append(float(morph_start))
        inserted.append("morph start")

    profile = config.get("profile")
    if isinstance(profile, Mapping) and str(output["formula"]).upper() == "FREEFORM":
        for station in profile.get("crossSections", ()):
            if isinstance(station, Mapping) and isinstance(station.get("t"), (int, float)):
                targets.append(float(station["t"]))
        for key in ("profileH", "profileV"):
            descriptor = profile.get(key)
            if not isinstance(descriptor, Mapping):
                continue
            rows = descriptor.get("points")
            if not isinstance(rows, list) or len(rows) < 2:
                continue
            z0 = float(rows[0][0])
            length = float(rows[-1][0]) - z0
            if length > 0.0:
                targets.extend((float(row[0]) - z0) / length for row in rows)
        inserted.extend(["FREEFORM cross-section stations", "FREEFORM H/V anchors"])
        unavailable.remove("OSSE extension/slot boundaries when expression-valued")

    # Rollback extrema are available additively from the canonical candidate
    # grid even though the profile evaluator does not publish named stations.
    master_grid = output["grid"]
    points = master_grid["inner_grid"]
    radius = np.mean(np.linalg.norm(points[:, :, :2], axis=2), axis=0)
    slope = np.diff(radius)
    extrema = np.flatnonzero(slope[:-1] * slope[1:] <= 0.0) + 1
    if len(extrema):
        targets.extend(float(t_values[index]) for index in extrema)
        inserted.append("canonical rollback/radial extrema")

    return _nearest_indices(t_values, targets), inserted, unavailable


def _corner_phi_indices(normals: NDArray[np.float64]) -> list[int]:
    """Return the union of curved-arc rows; planar wall rows remain sparse."""

    mouth = normals[-1]
    dots = np.sum(mouth * np.roll(mouth, -1, axis=0), axis=1)
    changing = np.flatnonzero(1.0 - np.clip(dots, -1.0, 1.0) > 1.0e-10)
    size = normals.shape[1]
    return sorted({int(index) for value in changing for index in (value, (value + 1) % size)})


def _grid_surface_from_selection(
    role: str,
    points: NDArray[np.float64],
    normals: NDArray[np.float64],
    t_indices: NDArray[np.int64],
    phi_indices: NDArray[np.int64],
    *,
    closed_phi: bool,
    normal_sign: float = 1.0,
    curvature_mean: NDArray[np.float64] | None = None,
    curvature_principal: NDArray[np.float64] | None = None,
) -> PreviewSurfaceV1:
    selected_points = points[np.ix_(t_indices, phi_indices)]
    selected_normals = normal_sign * normals[np.ix_(t_indices, phi_indices)]
    positions = selected_points.reshape(-1, 3)
    flat_normals = selected_normals.reshape(-1, 3)
    selected_mean = (
        None
        if curvature_mean is None
        else normal_sign * curvature_mean[np.ix_(t_indices, phi_indices)]
    )
    selected_principal = (
        None
        if curvature_principal is None
        else normal_sign * curvature_principal[np.ix_(t_indices, phi_indices)]
    )
    oriented = _orient_indices_to_normals(
        role,
        positions,
        _grid_indices(*selected_points.shape[:2], closed_phi=closed_phi),
        flat_normals,
    )
    return PreviewSurfaceV1(
        role=role,
        positions=positions,
        indices=oriented.indices,
        normals=flat_normals,
        shading="smooth",
        normal_method="analytic-parametric",
        closed_phi=closed_phi,
        curvature_mean=(None if selected_mean is None else selected_mean.reshape(-1)),
        curvature_principal=(
            None if selected_principal is None else selected_principal.reshape(-1)
        ),
        metadata=_orientation_metadata(oriented),
    )


def _fidelity_record(
    achieved: Mapping[str, Any] | None,
    *,
    chord_target: float,
    normal_target: float,
    silhouette_target: int,
    cap_limited: bool = False,
) -> dict[str, Any]:
    measurement_complete = bool((achieved or {}).get("measurement_complete", True))
    raw_chord = (achieved or {}).get("max_chord_error_mm", np.finfo(np.float64).eps)
    chord = None if not measurement_complete or raw_chord is None else float(raw_chord)
    normal = float((achieved or {}).get("max_normal_step_deg", 0.0))
    unmeasured = int((achieved or {}).get("unmeasured_intervals", 0))
    limited = bool(
        (achieved or {}).get("vertex_cap_limited", False)
        or cap_limited
        or not measurement_complete
    )
    return {
        # Stage-1 aliases remain for consumers already reading them.
        "max_chord_error_mm": chord,
        "max_normal_step_deg": normal,
        "reference_density_multiplier": int(
            (achieved or {}).get("reference_density_multiplier", 4)
        ),
        "max_chord_error_mm_requested": chord_target,
        "max_normal_step_deg_requested": normal_target,
        "min_silhouette_segments_requested": silhouette_target,
        "max_chord_error_mm_achieved": chord,
        "max_normal_step_deg_achieved": normal,
        "vertex_cap_limited": limited,
        "measurement_complete": measurement_complete,
        "unmeasured_intervals": unmeasured,
        "silhouette_segments_achieved": None,
    }


def _silhouette_segments(
    phi: NDArray[np.float64], *, closed_phi: bool
) -> int:
    if closed_phi:
        return int(phi.shape[1])
    equivalents: list[int] = []
    for row in np.asarray(phi, dtype=np.float64):
        span = float(np.unwrap(row)[-1] - np.unwrap(row)[0])
        if span > 0.0:
            equivalents.append(int(round((len(row) - 1) * math.tau / span)))
    return min(equivalents, default=max(0, phi.shape[1] - 1))


def _intervals_for_arc(
    radius: float, span_deg: float, chord: float, normal_deg: float, floor: int
) -> int:
    by_normal = int(math.ceil(span_deg / normal_deg)) + 2
    if radius <= chord:
        by_chord = 1
    else:
        half_angle = math.acos(np.clip(1.0 - chord / radius, -1.0, 1.0))
        by_chord = int(math.ceil(math.radians(span_deg) / max(2.0 * half_angle, 1.0e-12)))
    return min(_MAX_ARC_INTERVALS, max(int(floor), by_normal, by_chord))


def _configuration_has_corners(config: Mapping[str, Any]) -> bool:
    profile = config.get("profile")
    profile = profile if isinstance(profile, Mapping) else {}
    if str(config.get("formula", "OSSE")).strip().upper() == "FREEFORM":
        return any(
            isinstance(station, Mapping)
            and str(station.get("shape", "")).strip().lower() == "rounded_rectangle"
            for station in profile.get("crossSections", ())
        )
    morph = config.get("morph", config.get("MORPH"))
    morph = morph if isinstance(morph, Mapping) else {}
    target = morph.get(
        "morph_target",
        morph.get(
            "morphTarget",
            config.get(
                "morph_target",
                config.get(
                    "morphTarget",
                    profile.get("morph_target", profile.get("morphTarget", 0)),
                ),
            ),
        ),
    )
    return isinstance(target, (int, float)) and int(round(float(target))) == 1


def _replace_grid_with_corner_refinement(
    output: dict[str, Any], corner_intervals: int
) -> None:
    params = dict(output["params"])
    params[ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY] = max(
        1, int(math.ceil(corner_intervals / 3.0))
    )
    if str(params.get("type", "")).strip().upper() == "FREEFORM":
        params[FREEFORM_CONTINUOUS_COLLAPSE_KEY] = True
    grid = build_point_grid_arrays(params)
    vertical_offset = float(grid.get("vertical_offset_mm", 0.0) or 0.0)
    if vertical_offset:
        for key in ("inner_grid", "outer_grid"):
            if grid.get(key) is None:
                continue
            grid[key][:, :, 1] += vertical_offset
    output["params"] = params
    output["grid"] = grid


# Azimuths screened for an unreachable guiding curve. The guiding curve, the
# coverage angle and the termination may all be per-azimuth expressions, so a
# single phi=0 probe would miss a mouth that only goes off-target off-axis.
#
# The step was 15 degrees while each azimuth cost a full coverage inversion,
# and that was demonstrably too coarse: gcurveWidth="1000 - 900*sin(12*p)^2"
# is reachable at every multiple of 15 and saturated at 7.5, so the preview
# said nothing at all. Screening with the bracket probe instead of the full
# inversion (osse_coverage_saturation_probe: 2 radius evaluations, not 26)
# buys the resolution back. Measured, healthy OSSE geometry with a reachable
# type-1 guiding curve (36 us per azimuth by inversion, 5 us by probe):
#
#     15 deg / full inversion  (24 azimuths)   0.85 ms   <- was
#      1 deg / full inversion  (360 azimuths) 12.88 ms
#      1 deg / bracket probe   (360 azimuths)  1.79 ms   <- is
#
# The whole preview build for that config is 96 ms coarse / 554 ms fine, so
# 15x the angular resolution costs +0.94 ms, about 1% of a coarse frame. A
# naive tightening without the probe would have cost 12x that.
#
# STILL BEST-EFFORT. One degree resolves anything up to about a 180th-order
# azimuthal term, which is far past any guiding curve a person writes by hand,
# but a sufficiently spiky expression can still hide between probes and no
# fixed step can rule that out. The absence of a warning is therefore not a
# guarantee that the guiding curve is met; the step is published in the
# preview metadata as ``guiding_curve_probe.step_deg`` so a caller can say how
# much the silence is worth.
_GUIDING_CURVE_PROBE_STEP_DEG = 1.0
_GUIDING_CURVE_PROBE_AZIMUTHS = tuple(
    math.radians(index * _GUIDING_CURVE_PROBE_STEP_DEG)
    for index in range(int(round(360.0 / _GUIDING_CURVE_PROBE_STEP_DEG)))
)


def _guiding_curve_warnings(
    params: Mapping[str, Any], formula: Any
) -> list[str]:
    """Warn when the OSSE coverage solver cannot reach the guiding curve.

    The solver clamps to its bracket instead of failing, which reads to the
    user as "the parameters stopped doing anything" — the mouth is no longer on
    the guiding curve and no further edit to the coverage angle can put it
    back. Reported once with the worst-offending azimuth rather than once per
    probe, so a fully unreachable curve does not emit 360 near-identical lines.

    Screened with the bracket probe rather than the full inversion. The probe
    returns the same saturated result the inversion would, so ranking the
    azimuths on it is exact; only the reported azimuth pays for a full solve,
    and even that one returns from the probe branch without bisecting.
    """

    if str(formula).strip().upper() != "OSSE":
        return []
    worst_phi: float | None = None
    worst_error = -1.0
    saturated = 0
    for phi in _GUIDING_CURVE_PROBE_AZIMUTHS:
        try:
            solved = osse_coverage_saturation_probe(params, phi)
        except (ValueError, ZeroDivisionError, OverflowError):
            # A malformed guiding curve is the config validator's error to
            # raise; a preview warning must not mask it with its own failure.
            return []
        if solved is None or solved.saturated is None:
            continue
        saturated += 1
        error = abs(solved.achieved_radius - solved.target_radius)
        if not math.isfinite(error):
            error = math.inf
        # A rotationally symmetric guiding curve misses every azimuth by the
        # same amount up to rounding, so only a materially worse azimuth may
        # displace the incumbent. Otherwise the reported phi is whichever
        # probe happened to accumulate more floating-point error.
        if error > worst_error * (1.0 + 1.0e-9) + 1.0e-9:
            worst_error = error
            worst_phi = phi
    if worst_phi is None:
        return []
    location = (
        "every probed azimuth"
        if saturated == len(_GUIDING_CURVE_PROBE_AZIMUTHS)
        else None
    )
    reason = osse_coverage_saturation(params, worst_phi, location=location)
    return [reason] if reason is not None else []


def build_preview_geometry(
    config: Mapping[str, Any], options: PreviewOptionsV1 = PreviewOptionsV1()
) -> PreviewGeometryV1:
    """Build complete error-bounded render geometry from a mesher config."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not isinstance(options, PreviewOptionsV1):
        raise TypeError("options must be PreviewOptionsV1")
    lod = str(options.lod).strip().lower()
    if lod not in _LOD_PRESETS:
        raise ValueError("lod must be 'coarse', 'fine', or 'inspection'")
    preset = _LOD_PRESETS[lod]
    chord_target = float(
        preset["chord"]
        if options.max_chord_error_mm is None
        else options.max_chord_error_mm
    )
    normal_target = float(
        preset["normal"]
        if options.max_normal_step_deg is None
        else options.max_normal_step_deg
    )
    try:
        silhouette_target = int(
            preset["silhouette"]
            if options.min_silhouette_segments is None
            else options.min_silhouette_segments
        )
        vertex_cap = None if options.max_vertices is None else int(options.max_vertices)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("silhouette and vertex limits must be finite integers") from exc
    if not math.isfinite(chord_target) or chord_target <= 0.0:
        raise ValueError("max_chord_error_mm must be finite and > 0")
    if not math.isfinite(normal_target) or not 0.0 < normal_target <= 180.0:
        raise ValueError("max_normal_step_deg must be finite and in (0, 180]")
    if silhouette_target < 3:
        raise ValueError("min_silhouette_segments must be >= 3")
    if vertex_cap is not None and vertex_cap < 8:
        raise ValueError(
            "max_vertices must be >= 8 (the minimum robust 2x4 horn topology)"
        )

    warnings: list[str] = []
    preflight_limited = False
    start = time.perf_counter()

    has_corners = _configuration_has_corners(config)
    corner_intervals = _intervals_for_arc(
        200.0,
        90.0,
        chord_target,
        normal_target,
        int(preset["roundover"]),
    )
    if corner_intervals >= _MAX_ARC_INTERVALS:
        preflight_limited = True
        warnings.append(
            f"acoustic corner reference clamped to {_MAX_ARC_INTERVALS} intervals"
        )
    # The canonical corner sampler refines its three stable arc rows in whole
    # multiples. Ordinary/circular grids instead use a 2x candidate lattice.
    if has_corners:
        corner_intervals = 3 * int(math.ceil(corner_intervals / 3.0))
        # Corner grids already add a dense, stable union of analytic arc rows;
        # multiplying the flat-side seed would defeat that adaptive allocation.
        angular_master = max(3, silhouette_target)
    else:
        angular_master = max(
            4 * silhouette_target, 4 * int(math.ceil(360.0 / normal_target))
        )
    if angular_master > _MAX_ANGULAR_SAMPLES:
        angular_master = _MAX_ANGULAR_SAMPLES
        preflight_limited = True
        warnings.append(
            f"canonical azimuth reference clamped to {_MAX_ANGULAR_SAMPLES} samples"
        )
    default_chord = float(preset["chord"])
    chord_ratio = default_chord / chord_target
    axial_scale = math.sqrt(chord_ratio) if math.isfinite(chord_ratio) else math.inf
    scaled_axial = (
        512
        if not math.isfinite(axial_scale)
        or axial_scale >= 512 / int(preset["master_axial"])
        else int(math.ceil(int(preset["master_axial"]) * max(1.0, axial_scale)))
    )
    axial_master = min(512, max(int(preset["master_axial"]), scaled_axial))
    axial_master = min(512, max(axial_master, 4 * int(preset["axial"])))
    formula_name = str(config.get("formula", "OSSE")).strip().upper()
    if formula_name in {"R-OSSE", "ROSSE", "FREEFORM"}:
        axial_master = min(512, axial_master * 2)
    angular_product_cap = max(3, _MAX_CANONICAL_VERTICES // (axial_master + 1))
    if angular_master > angular_product_cap:
        angular_master = angular_product_cap
        preflight_limited = True
        warnings.append(
            f"canonical reference clamped to {_MAX_CANONICAL_VERTICES} points"
        )

    canonical_start = time.perf_counter()
    axial_power = {"coarse": 1.75, "fine": 2.0, "inspection": 2.5}[lod]
    sampling_config = _adaptive_lod_config(
        config, angular_master, axial_master, power=axial_power
    )
    parsed_params, _parsed_formula, _parsed_mode = build_geometry_params(config)
    warnings.extend(_guiding_curve_warnings(parsed_params, _parsed_formula))
    # The wall thickness as configured. ``deferred_wall`` below is only set on
    # the corner-refinement path, where the outer shell is rebuilt here instead
    # of by the sampler, so it is not a reliable thickness on its own.
    wall_mm = float(eval_param(parsed_params.get("wallThickness"), 0.0, 0.0))
    deferred_wall = 0.0
    if has_corners and str(_parsed_mode) == "freestanding":
        deferred_wall = float(eval_param(parsed_params.get("wallThickness"), 0.0, 0.0))
        if deferred_wall > 0.0:
            mesh = dict(sampling_config["mesh"])
            mesh["wall_thickness_mm"] = 0.0
            for alias in ("wallThickness", "wall_thickness", "WallThickness"):
                mesh.pop(alias, None)
            sampling_config["mesh"] = mesh
    # The preview never spells the grid's flat vertex lists: it reads the
    # arrays they would be built from.
    output = build_viewport_geometry_from_config(
        sampling_config, point_lists=False
    )
    if has_corners:
        _replace_grid_with_corner_refinement(output, corner_intervals)
    if deferred_wall > 0.0:
        output["params"]["wallThickness"] = deferred_wall
    canonical_ms = (time.perf_counter() - canonical_start) * 1000.0

    grid_data = output["grid"]
    n_phi = int(grid_data["grid_n_phi"])
    closed_phi = bool(grid_data.get("full_circle", True))
    inner_canonical = grid_data["inner_grid"]
    inner_master = _surface_grid(inner_canonical)
    master_t = np.asarray(grid_data.get("slice_map"), dtype=np.float64)
    master_phi = (
        np.asarray(grid_data["phi_grid"], dtype=np.float64).T
        if grid_data.get("phi_grid") is not None
        else np.broadcast_to(
            np.asarray(grid_data["angle_list"], dtype=np.float64),
            inner_master.shape[:2],
        )
    )
    inner_normals = analytic_grid_normals(
        inner_master,
        closed_phi=closed_phi,
        t_coordinates=master_t,
        phi_coordinates=master_phi,
    )
    inner_curvature_mean = inner_curvature_principal = None
    if options.include_curvature and options.include_inner:
        inner_curvature_mean, inner_curvature_principal = analytic_grid_curvature(
            inner_master,
            closed_phi=closed_phi,
            t_coordinates=master_t,
            phi_coordinates=master_phi,
        )
    semantic_t, semantic_inserted, semantic_unavailable = _semantic_t_stations(
        config, output, master_t
    )
    initial_t = sorted(
        set(
            _even_indices(
                len(master_t), int(preset["axial"]) + 1, closed=False
            )
        ).union(semantic_t)
    )
    initial_phi = _even_indices(n_phi, silhouette_target, closed=closed_phi)
    corner_rows: list[int] = []
    if has_corners:
        corner_rows = _corner_phi_indices(inner_normals)
        initial_phi = sorted(set(initial_phi).union(corner_rows))
    t_indices, phi_indices, horn_achieved = adaptive_grid_indices(
        inner_master,
        inner_normals,
        initial_t,
        initial_phi,
        max_chord_error_mm=chord_target,
        max_normal_step_deg=normal_target,
        max_vertices=vertex_cap,
        closed_phi=closed_phi,
        t_coordinates=master_t,
        phi_coordinates=master_phi,
    )

    assembly_start = time.perf_counter()
    surfaces: list[PreviewSurfaceV1] = []
    fidelity: dict[str, dict[str, Any]] = {}
    if options.include_inner:
        surfaces.append(
            _grid_surface_from_selection(
                "horn.inner",
                inner_master,
                inner_normals,
                t_indices,
                phi_indices,
                closed_phi=closed_phi,
                normal_sign=-1.0,
                curvature_mean=inner_curvature_mean,
                curvature_principal=inner_curvature_principal,
            )
        )
        fidelity["horn.inner"] = _fidelity_record(
            horn_achieved,
            chord_target=chord_target,
            normal_target=normal_target,
            silhouette_target=silhouette_target,
        )

    selected_inner = inner_canonical[np.ix_(phi_indices, t_indices)]
    outer_canonical = None
    selected_outer = None
    if grid_data.get("outer_grid") is not None:
        outer_canonical = grid_data["outer_grid"]
        selected_outer = outer_canonical[np.ix_(phi_indices, t_indices)]
        if options.include_outer:
            outer_master = _surface_grid(outer_canonical)
            for outer_surface in _outer_shell_surfaces(
                outer_master,
                t_indices,
                phi_indices,
                closed_phi=closed_phi,
                t_coordinates=master_t,
                phi_coordinates=master_phi,
                include_curvature=options.include_curvature,
            ):
                surfaces.append(outer_surface)
                fidelity[outer_surface.role] = _fidelity_record(
                    horn_achieved,
                    chord_target=chord_target,
                    normal_target=normal_target,
                    silhouette_target=silhouette_target,
                )
            surfaces.append(
                _smooth_mouth_rim(
                    selected_inner[:, -1, :],
                    selected_outer[:, -1, :],
                    closed_phi=closed_phi,
                    exit_direction=_mouth_exit_direction(selected_inner),
                    include_curvature=options.include_curvature,
                )
            )
            fidelity["mouth_rim"] = _fidelity_record(
                horn_achieved,
                chord_target=chord_target,
                normal_target=normal_target,
                silhouette_target=silhouette_target,
            )
    elif deferred_wall > 0.0:
        selected_outer = _outer_offset_shell(
            selected_inner,
            deferred_wall,
            full_circle=closed_phi,
            t_coordinates=master_t[t_indices],
            phi_coordinates=master_phi[np.ix_(t_indices, phi_indices)],
        )
        outer_canonical = selected_outer
        if options.include_outer:
            selected_outer_master = _surface_grid(selected_outer)
            # The deferred-wall shell is already the selected grid, so its rows
            # are its own stations.
            for outer_surface in _outer_shell_surfaces(
                selected_outer_master,
                np.arange(selected_outer_master.shape[0], dtype=np.int64),
                np.arange(selected_outer_master.shape[1], dtype=np.int64),
                closed_phi=closed_phi,
                t_coordinates=master_t[t_indices],
                phi_coordinates=master_phi[np.ix_(t_indices, phi_indices)],
                include_curvature=options.include_curvature,
            ):
                surfaces.append(outer_surface)
                fidelity[outer_surface.role] = _fidelity_record(
                    horn_achieved,
                    chord_target=chord_target,
                    normal_target=normal_target,
                    silhouette_target=silhouette_target,
                )
            surfaces.append(
                _smooth_mouth_rim(
                    selected_inner[:, -1, :],
                    selected_outer[:, -1, :],
                    closed_phi=closed_phi,
                    exit_direction=_mouth_exit_direction(selected_inner),
                    include_curvature=options.include_curvature,
                )
            )
            fidelity["mouth_rim"] = _fidelity_record(
                horn_achieved,
                chord_target=chord_target,
                normal_target=normal_target,
                silhouette_target=silhouette_target,
            )

    if options.include_source_cap:
        cap_intervals = int(preset["cap"])
        cap_limited = False
        if vertex_cap is not None:
            allowed_radial = max(1, (vertex_cap - 1) // len(phi_indices))
            if cap_intervals > allowed_radial:
                cap_intervals = allowed_radial
                cap_limited = True
        while True:
            cap_surface, cap_fidelity, source_details = _source_cap(
                selected_inner,
                output["params"],
                output["formula"],
                cap_intervals,
                closed_phi=closed_phi,
                include_curvature=options.include_curvature,
            )
            passes = cap_fidelity is None or (
                cap_fidelity["max_chord_error_mm"] <= chord_target
                and cap_fidelity["max_normal_step_deg"] <= normal_target
            )
            if passes:
                break
            proposed = cap_intervals + 1
            if vertex_cap is not None and 1 + proposed * len(phi_indices) > vertex_cap:
                cap_limited = True
                break
            cap_intervals = proposed
        surfaces.append(cap_surface)
        fidelity["source_cap"] = _fidelity_record(
            cap_fidelity,
            chord_target=chord_target,
            normal_target=normal_target,
            silhouette_target=silhouette_target,
            cap_limited=cap_limited,
        )
    else:
        source_details = {}
        cap_intervals = 0

    if output.get("enclosure") is not None:
        # The preview draws every implemented plan, but only the rounded
        # rectangle has a watertight enclosure builder -- ``build_enclosure_box``
        # raises NotImplementedError for the ellipse and superellipse plans in
        # the closed domain, and its open-domain route accepts plan_type=1 only.
        # Say so here rather than let the shape look finished until build time.
        # The toggle above is a display filter, not a config change, so this
        # warning does not depend on it.
        preview_plan_type = int(output["enclosure"].get("plan_type", 1))
        if preview_plan_type in (2, 3):
            warnings.append(
                f"enclosure plan_type={preview_plan_type} is previewed but not "
                "buildable: only the rounded-rectangle plan (plan_type=1) has a "
                "watertight closed-enclosure builder, and the open (reduced) "
                "domain supports plan_type=1 only"
            )

    if output.get("enclosure") is not None and options.include_enclosure:
        enclosure_payload = dict(output["enclosure"])
        enclosure_config = config.get("enclosure")
        if isinstance(enclosure_config, Mapping):
            plan_n = enclosure_config.get(
                "plan_n", enclosure_config.get("planN", enclosure_config.get("encPlanN"))
            )
            if plan_n is not None:
                enclosure_payload["plan_n"] = float(plan_n)
        roundover_intervals = _intervals_for_arc(
            float(enclosure_payload.get("edge_depth", 0.0)),
            90.0,
            chord_target,
            normal_target,
            int(preset["roundover"]),
        )
        plan_corner_intervals, plan_preflight_limited = _adaptive_plan_intervals(
            enclosure_payload,
            chord_target,
            normal_target,
            1,
        )
        if plan_preflight_limited:
            preflight_limited = True
            warnings.append(
                f"enclosure plan reference clamped to {_MAX_ARC_INTERVALS} intervals per quarter"
            )
        enclosure_cap_limited = plan_preflight_limited
        if vertex_cap is not None:
            def enclosure_vertex_estimates(
                radial_intervals: int, plan_intervals: int
            ) -> dict[str, int]:
                plan_vertices = (
                    4 * plan_intervals + 8
                    if int(enclosure_payload["plan_type"]) == 1
                    else 4 * plan_intervals
                )
                estimates = {
                    "enclosure.front": 2 * len(phi_indices),
                    "enclosure.side": (
                        4 * plan_vertices
                        if int(enclosure_payload["edge_type"]) == 2
                        else 2 * plan_vertices
                    ),
                }
                if float(enclosure_payload.get("edge_depth", 0.0)) > 0.0:
                    estimates["enclosure.roundover"] = len(phi_indices) + (
                        2 * radial_intervals + 1
                    ) * plan_vertices
                if options.include_rear_cap:
                    estimates["enclosure.rear"] = plan_vertices + 1
                return estimates

            minimum_estimates = enclosure_vertex_estimates(1, 1)
            topology_minimum = max(minimum_estimates.values())
            if vertex_cap < topology_minimum:
                raise ValueError(
                    f"max_vertices={vertex_cap} is below the enclosure topological "
                    f"minimum {topology_minimum}"
                )
            while max(
                enclosure_vertex_estimates(
                    roundover_intervals, plan_corner_intervals
                ).values()
            ) > vertex_cap and (
                roundover_intervals > 1 or plan_corner_intervals > 1
            ):
                enclosure_cap_limited = True
                if roundover_intervals >= plan_corner_intervals and roundover_intervals > 1:
                    roundover_intervals -= 1
                elif plan_corner_intervals > 1:
                    plan_corner_intervals -= 1
        enclosure_surfaces, enclosure_fidelity = _enclosure_surfaces(
            enclosure_payload,
            selected_inner[:, -1, :],
            roundover_intervals,
            plan_corner_intervals,
            chord_target=chord_target,
            normal_target=normal_target,
            include_rear=options.include_rear_cap,
            include_curvature=options.include_curvature,
        )
        surfaces.extend(enclosure_surfaces)
        plan_measurement = _plan_fidelity(
            enclosure_payload, plan_corner_intervals
        )
        plan_chord = plan_measurement["max_chord_error_mm"]
        plan_normal = plan_measurement["max_normal_step_deg"]
        plan_vertices = len(
            _plan_ring(
                enclosure_payload,
                0.0,
                1.0,
                corner_intervals=plan_corner_intervals,
            )
        )
        # The roundover arc model only describes a fillet. A chamfer sweeps one
        # ruled interval between two polygons and discretises nothing, so
        # charging it a fillet's chord error and normal step reported an error
        # the emitted band does not carry.
        if _faceted_edge(enclosure_payload):
            round_chord = 0.0
            round_normal = 0.0
        else:
            round_chord = float(enclosure_payload.get("edge_depth", 0.0)) * (
                1.0 - math.cos(math.pi / (4.0 * roundover_intervals))
            )
            round_normal = 90.0 / roundover_intervals
        for surface in enclosure_surfaces:
            measured = enclosure_fidelity.get(surface.role, {})
            if surface.role == "enclosure.roundover":
                measured = dict(measured)
                measured["max_chord_error_mm"] = max(
                    float(measured.get("max_chord_error_mm", 0.0)),
                    plan_chord,
                    round_chord,
                )
                measured["max_normal_step_deg"] = max(
                    float(measured.get("max_normal_step_deg", 0.0)),
                    plan_normal,
                    round_normal,
                )
            elif surface.role == "enclosure.side":
                measured = dict(plan_measurement)
            fidelity[surface.role] = _fidelity_record(
                measured,
                chord_target=chord_target,
                normal_target=normal_target,
                silhouette_target=silhouette_target,
                cap_limited=enclosure_cap_limited,
            )
    elif selected_outer is not None and options.include_rear_cap:
        # The rear plate sits on the plane the MESH puts it on, which is not a
        # property of the outer ring at all:
        #     rear_z = mean(inner throat z) - wall
        # See ``point_grid_freestanding.py`` (rear_z) and ``_rear_rim_points``,
        # which keeps x/y and moves only z.
        #
        # Under the old throat clamp the outer throat ring happened to sit on
        # that same plane, so capping straight off it matched the mesh BY
        # ACCIDENT. Now that row 0 lies on the offset surface it does not, and
        # deriving the plane from the ring put the previewed rear face ~4.4 mm
        # forward of the real one and silently dropped the whole rear return.
        rear_z = float(np.mean(selected_inner[:, 0, 2]) - wall_mm)
        rear_ring = np.array(selected_outer[:, 0, :], dtype=np.float64, copy=True)
        rear_ring[:, 2] = rear_z

        # The band between the outer throat ring and the rear rim IS the rear
        # return. The mesh builds it by prepending this ring to the outer shell;
        # the preview has already emitted its shell, so ship the band on its own.
        if options.include_outer:
            # (t, phi, xyz), rear rim first, exactly as the mesh orders it.
            return_master = np.stack(
                (rear_ring, selected_outer[:, 0, :]), axis=0
            )
            # A straight axial extrusion of the throat ring, so its meridian
            # curvature is zero; the hoop term is carried by the ring itself.
            return_curvature = (
                np.zeros(return_master.shape[:2], dtype=np.float64)
                if options.include_curvature
                else None
            )
            surfaces.append(
                _grid_surface_from_selection(
                    "wall.rear_return",
                    return_master,
                    analytic_grid_normals(return_master, closed_phi=closed_phi),
                    np.asarray((0, 1), dtype=np.int64),
                    np.arange(return_master.shape[1], dtype=np.int64),
                    closed_phi=closed_phi,
                    curvature_mean=return_curvature,
                    curvature_principal=return_curvature,
                )
            )

        surfaces.append(
            _flat_cap(
                "wall.rear_cap",
                rear_ring,
                (0.0, 0.0, -1.0),
                closed_phi=closed_phi,
                include_curvature=options.include_curvature,
            )
        )
        fidelity["wall.rear_cap"] = _fidelity_record(
            horn_achieved,
            chord_target=chord_target,
            normal_target=normal_target,
            silhouette_target=silhouette_target,
        )

    # Every rendered role receives target-vs-achieved data, including exact
    # planar faces whose interior error is zero and whose boundary inherits the
    # adjacent adaptive ring.
    for surface in surfaces:
        fidelity.setdefault(
            surface.role,
            _fidelity_record(
                horn_achieved,
                chord_target=chord_target,
                normal_target=normal_target,
                silhouette_target=silhouette_target,
            ),
        )

    selected_phi_coordinates = master_phi[np.ix_(t_indices, phi_indices)]
    horn_silhouette = _silhouette_segments(
        selected_phi_coordinates, closed_phi=closed_phi
    )
    enclosure_plan_roles = {
        "enclosure.roundover",
        "enclosure.side",
        "enclosure.rear",
    }
    for surface in surfaces:
        achieved_silhouette = (
            plan_vertices
            if surface.role in enclosure_plan_roles
            and output.get("enclosure") is not None
            and options.include_enclosure
            else horn_silhouette
        )
        fidelity[surface.role]["silhouette_segments_achieved"] = int(
            achieved_silhouette
        )
        if achieved_silhouette < silhouette_target:
            fidelity[surface.role]["vertex_cap_limited"] = True
        if preflight_limited:
            fidelity[surface.role]["preflight_limited"] = True
        if vertex_cap is not None and len(surface.positions) > vertex_cap:
            raise ValueError(
                f"max_vertices={vertex_cap} cannot represent {surface.role}; "
                f"minimum emitted topology has {len(surface.positions)} vertices"
            )

    assembly_ms = (time.perf_counter() - assembly_start) * 1000.0
    total_ms = (time.perf_counter() - start) * 1000.0
    metadata: dict[str, Any] = {
        "api_version": _API_VERSION,
        "metadata_version": _METADATA_VERSION,
        "units": "mm",
        "coordinate_frame": "mesher-xyz",
        "formula": output["formula"],
        "mode": output["mode"],
        "lod": lod,
        "actual_segment_counts": {
            "horn_phi": len(phi_indices),
            "horn_axial": len(t_indices) - 1,
            "enclosure_roundover_quarter": (
                roundover_intervals
                if output.get("enclosure") is not None
                and float(output["enclosure"].get("edge_depth", 0.0)) > 0.0
                and options.include_enclosure
                and int(output["enclosure"].get("edge_type", 1)) == 1
                else 1
                if output.get("enclosure") is not None
                and float(output["enclosure"].get("edge_depth", 0.0)) > 0.0
                and options.include_enclosure
                else 0
            ),
            "source_cap_radial": (
                cap_intervals
                if options.include_source_cap
                and float(source_details.get("source_cap_height_mm", 0.0)) > 0.0
                else (1 if options.include_source_cap else 0)
            ),
        },
        "timings_ms": {
            "canonical_sampling": canonical_ms,
            "surface_assembly_and_fidelity": assembly_ms,
            "total": total_ms,
        },
        "fidelity": fidelity,
        "warnings": warnings,
        # How finely the guiding-curve saturation screen actually looked. A
        # fixed step can always be out-resolved by a spiky enough expression,
        # so publish it rather than let the absence of a warning read as a
        # proof that the guiding curve is met everywhere.
        "guiding_curve_probe": {
            "step_deg": _GUIDING_CURVE_PROBE_STEP_DEG,
            "azimuths": len(_GUIDING_CURVE_PROBE_AZIMUTHS),
            "best_effort": True,
        },
        "requested_fidelity": {
            "max_chord_error_mm": chord_target,
            "max_normal_step_deg": normal_target,
            "min_silhouette_segments": silhouette_target,
            "max_vertices": vertex_cap,
        },
        "angular_sampling": {
            "strategy": "stable-union-corner-grid" if has_corners else "adaptive-periodic",
            "corner_arc_rows": len(set(phi_indices).intersection(corner_rows)),
            "flat_side_rows": len(phi_indices)
            - len(set(phi_indices).intersection(corner_rows)),
            # The faceted chamfer band rules per plan column and never
            # zippers; only the curved fillet path still stitches unequal
            # rings.
            "uses_zipper_indices": bool(
                output.get("enclosure") is not None
                and not _faceted_edge(output["enclosure"])
            ),
        },
        "vertex_accounting": {
            surface.role: {
                "vertices": len(surface.positions),
                "max_vertices": vertex_cap,
                "vertex_cap_limited": fidelity[surface.role]["vertex_cap_limited"],
            }
            for surface in surfaces
        },
        "semantic_stations": {
            "inserted_first": semantic_inserted
            + ["corner tangencies/cardinals", "enclosure transitions"],
            "unavailable_additively": semantic_unavailable,
        },
        "nesting": {
            "horn_phi": "coarse is a subset of fine for circular grids; corner grids use the stable union row identities",
            "horn_axial": "12 -> 48 -> 96 base stations are nested; semantic stations are shared by value",
            "exceptions": [
                "custom tolerances/caps may choose different refinement paths",
                "canonical ATH z-map modes are not guaranteed to be nested",
            ],
        },
        "normal_convention": (
            "role-oriented analytic/exact normals; every triangle is counter-clockwise "
            "when viewed from its normal side"
        ),
        "normal_method_notes": {
            "analytic-parametric": (
                "finite differences of true analytic/canonical surface samples in "
                "their real axial and azimuthal parameter coordinates; never mesh-derived"
            ),
            "exact-planar": "the exact constant normal of an emitted planar face",
        },
        "surface_metadata": {
            surface.role: dict(surface.metadata) for surface in surfaces
        },
        **source_details,
    }
    _validate_finite_metadata(metadata)
    return PreviewGeometryV1(surfaces=surfaces, metadata=metadata)


__all__ = [
    "PreviewGeometryV1",
    "PreviewOptionsV1",
    "PreviewSurfaceV1",
    "build_preview_geometry",
]
