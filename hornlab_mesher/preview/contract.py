"""The ``hornlab.preview/1`` data contract and its orientation proof.

The surface and option dataclasses, strict-JSON metadata validation, and the
per-triangle winding check every emitted surface passes. The orientation
rules themselves are stated in :mod:`hornlab_mesher.preview.api`.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


_API_VERSION = "hornlab.preview/1"
_METADATA_VERSION = "hornlab.preview/1.3"
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
    folded_triangles: int = 0
    proof: _OrientationCheckProof | None = field(
        default=None, compare=False, repr=False
    )


def _triangle_orientation_analysis(
    positions: NDArray[np.float64],
    indices: NDArray[np.uint32],
    normals: NDArray[np.float64],
) -> _OrientationAnalysis:
    return _classify_triangle_orientation(positions, indices, normals)[0]


def _classify_triangle_orientation(
    positions: NDArray[np.float64],
    indices: NDArray[np.uint32],
    normals: NDArray[np.float64],
) -> tuple[_OrientationAnalysis, NDArray[np.bool_]]:
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
        return _OrientationAnalysis(0, 0, 0, 0), np.zeros(0, dtype=bool)
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
    analysis = _OrientationAnalysis(
        positive_triangles=int(np.count_nonzero(positive)),
        negative_triangles=int(np.count_nonzero(negative)),
        degenerate_triangles=int(np.count_nonzero(degenerate)),
        abstaining_triangles=int(np.count_nonzero(abstaining | singular)),
        singular_triangles=int(np.count_nonzero(singular)),
    )
    return analysis, negative


def _orient_indices_to_normals(
    role: str,
    positions: NDArray[np.float64],
    indices: NDArray[np.uint32],
    normals: NDArray[np.float64],
    *,
    wind_folds_individually: bool = False,
) -> _OrientedIndices:
    """Return one consistently wound index buffer for the shipped normals.

    ``wind_folds_individually`` is for a surface whose own grid folds over
    itself (a constant-distance offset of a tight concavity). Its triangles in
    the fold are geometrically reversed against the smooth normals, so no single
    winding agrees with them all. Rather than refuse the surface, each
    disagreeing triangle is wound to agree with its normals and the count is
    reported in the ``foldedTriangles`` metadata.
    """

    triangles = np.asarray(indices, dtype=np.uint32).reshape(-1, 3)
    if not len(triangles):
        return _OrientedIndices(triangles.reshape(-1), 0, 0, 0)
    analysis = _triangle_orientation_analysis(positions, triangles, normals)
    folded = 0
    if analysis.positive_triangles and analysis.negative_triangles:
        disagreeing = min(
            analysis.positive_triangles, analysis.negative_triangles
        )
        if not wind_folds_individually:
            raise ValueError(
                f"{role}: inconsistent local orientation ({disagreeing}/{len(triangles)} "
                "non-degenerate triangles disagree with their normals)"
            )
        folded = analysis.negative_triangles
        _, negative_mask = _classify_triangle_orientation(
            positions, triangles, normals
        )
        triangles = triangles.copy()
        triangles[negative_mask] = triangles[negative_mask][:, (0, 2, 1)]
        analysis = _triangle_orientation_analysis(positions, triangles, normals)
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
        folded_triangles=folded,
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
    if result.folded_triangles:
        metadata["foldedTriangles"] = result.folded_triangles
    if result.proof is not None:
        # This internal object key deliberately falls outside the public
        # string-key metadata type. PreviewSurfaceV1 removes it before metadata
        # validation or publication.
        metadata[_ORIENTATION_PROOF_KEY] = result.proof  # type: ignore[index]
    return metadata
