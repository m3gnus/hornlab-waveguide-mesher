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
)
from ..profiles import eval_param
from ..profiles import build_point_grid
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
_ORIENTATION_BY_ROLE = {
    "horn.inner": "air-side",
    "horn.outer": "exterior",
    "mouth_rim": "exterior",
    "source_cap": "air-side",
    "wall.rear_cap": "exterior",
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
        orientation_check = _triangle_orientation_analysis(
            positions, indices, normals
        )
        if orientation_check.negative_triangles:
            raise ValueError(
                f"{self.role}: {orientation_check.negative_triangles} non-degenerate "
                "triangle windings disagree with their normals"
            )
        metadata = dict(self.metadata)
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


def _grid(raw: Any, n_phi: int, n_length: int) -> NDArray[np.float64]:
    return np.asarray(raw, dtype=np.float64).reshape(n_phi, n_length + 1, 3)


def _surface_grid(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert canonical ``(phi,t,xyz)`` to surface ``(t,phi,xyz)``."""

    return np.ascontiguousarray(np.transpose(points, (1, 0, 2)), dtype=np.float64)


def _grid_indices(n_t: int, n_phi: int, *, closed_phi: bool) -> NDArray[np.uint32]:
    triangles: list[int] = []
    phi_intervals = n_phi if closed_phi else n_phi - 1
    for jt in range(n_t - 1):
        row0 = jt * n_phi
        row1 = (jt + 1) * n_phi
        for ip in range(phi_intervals):
            ip1 = (ip + 1) % n_phi
            triangles.extend((row0 + ip, row0 + ip1, row1 + ip1))
            triangles.extend((row0 + ip, row1 + ip1, row1 + ip))
    return np.asarray(triangles, dtype=np.uint32)


@dataclass(frozen=True)
class _OrientationAnalysis:
    positive_triangles: int
    negative_triangles: int
    degenerate_triangles: int
    abstaining_triangles: int
    singular_triangles: int = 0


@dataclass(frozen=True)
class _OrientedIndices:
    indices: NDArray[np.uint32]
    degenerate_triangles: int
    abstaining_triangles: int
    disagreeing_triangles: int
    singular_triangles: int = 0


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
    average_normal = np.mean(vectors[triangles], axis=1)
    normal_length = np.linalg.norm(average_normal, axis=1)
    denominator = doubled_area * normal_length
    cosine = np.zeros(len(triangles), dtype=np.float64)
    usable_denominator = denominator > 0.0
    cosine[usable_denominator] = np.einsum(
        "ij,ij->i", face_vectors[usable_denominator], average_normal[usable_denominator]
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
    oriented = (
        triangles.reshape(-1)
        if analysis.positive_triangles
        else triangles[:, (0, 2, 1)].reshape(-1)
    )
    return _OrientedIndices(
        indices=np.asarray(oriented, dtype=np.uint32),
        degenerate_triangles=analysis.degenerate_triangles,
        abstaining_triangles=analysis.abstaining_triangles,
        disagreeing_triangles=0,
        singular_triangles=analysis.singular_triangles,
    )


def _orientation_metadata(result: _OrientedIndices) -> dict[str, int]:
    return {
        "degenerateTriangles": result.degenerate_triangles,
        "orientationAbstainingTriangles": result.abstaining_triangles,
        "disagreeingTriangles": result.disagreeing_triangles,
        "orientationSingularTriangles": result.singular_triangles,
    }


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


def _flat_cap(
    role: str,
    ring: NDArray[np.float64],
    normal: tuple[float, float, float],
    *,
    closed_phi: bool,
    include_curvature: bool = True,
) -> PreviewSurfaceV1:
    ring = np.asarray(ring, dtype=np.float64)
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

    positions: list[NDArray[np.float64]] = []
    normals: list[NDArray[np.float64]] = []
    pole = sphere_center.copy()
    pole[2] += sign * radius
    positions.append(pole)
    normals.append(np.asarray((0.0, 0.0, sign), dtype=np.float64))
    for level in range(1, radial_intervals + 1):
        theta = rim_angle * level / radial_intervals
        rho = radius * math.sin(theta)
        z = sphere_center[2] + sign * radius * math.cos(theta)
        for direction in directions:
            point = np.asarray(
                (center[0] + rho * direction[0], center[1] + rho * direction[1], z),
                dtype=np.float64,
            )
            positions.append(point)
            normals.append(sign * (point - sphere_center) / radius)

    triangles: list[int] = []
    n_phi = len(ring)
    limit = n_phi if closed_phi else n_phi - 1
    first_ring = 1
    for ip in range(limit):
        ip1 = (ip + 1) % n_phi
        triangles.extend((0, first_ring + ip, first_ring + ip1))
    for level in range(1, radial_intervals):
        row0 = 1 + (level - 1) * n_phi
        row1 = 1 + level * n_phi
        for ip in range(limit):
            ip1 = (ip + 1) % n_phi
            triangles.extend((row0 + ip, row1 + ip, row1 + ip1))
            triangles.extend((row0 + ip, row1 + ip1, row0 + ip1))

    normal_array = np.asarray(normals, dtype=np.float64)
    position_array = np.asarray(positions, dtype=np.float64)
    oriented = _orient_indices_to_normals(
        "source_cap",
        position_array,
        np.asarray(triangles, dtype=np.uint32),
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
    radius = max(0.1, edge * radial_t)
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


def _ray_aligned_ring(
    plan: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    center = np.mean(reference[:, :2], axis=0)
    result = np.empty_like(reference)
    for index, point in enumerate(reference):
        direction = point[:2] - center
        direction /= max(float(np.linalg.norm(direction)), 1.0e-14)
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
            result[index] = plan[int(np.argmin(delta))]
        else:
            result[index] = best[1]
    return result


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


def _enclosure_surfaces(
    enclosure: Mapping[str, Any],
    mouth: NDArray[np.float64],
    roundover_intervals: int,
    plan_corner_intervals: int,
    *,
    include_rear: bool,
    include_curvature: bool,
) -> tuple[list[PreviewSurfaceV1], dict[str, dict[str, float]]]:
    bounds = enclosure["bounds"]
    depth = float(enclosure["edge_depth"])
    rounded_edge = int(enclosure["edge_type"]) == 1
    z_front = float(bounds["z_front"])
    z_back = float(bounds["z_back"])
    center_xy = np.asarray((float(bounds["cx"]), float(bounds["cy"])), dtype=np.float64)

    front_native = _plan_ring(
        enclosure, z_front, 0.0, corner_intervals=plan_corner_intervals
    )
    front_aligned = _ray_aligned_ring(front_native, mouth)
    middle = 0.5 * (mouth + front_aligned)
    surfaces = [
        _flat_strip(
            "mouth_rim",
            mouth,
            middle,
            (0.0, 0.0, 1.0),
            closed_phi=True,
            include_curvature=include_curvature,
        ),
        _flat_strip(
            "enclosure.front",
            middle,
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
        front_ref = front_grid(edge_intervals)
        front_surface, front_fidelity = _roundover_piece(
            "enclosure.roundover",
            front_aligned,
            front_out,
            front_ref,
            center_xy,
            include_curvature=include_curvature,
        )

        back_out = back_grid(edge_intervals)
        back_ref = back_grid(edge_intervals)
        back_surface, back_fidelity = _roundover_piece(
            "enclosure.roundover",
            back_out[0],
            back_out,
            back_ref,
            center_xy,
            include_curvature=include_curvature,
        )
        surfaces.append(_combine_surfaces("enclosure.roundover", [front_surface, back_surface]))
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
        rear_ring = back_out[-1] if depth > 0.0 else side_back
        surfaces.append(
            _flat_cap(
                "enclosure.rear",
                rear_ring,
                (0.0, 0.0, -1.0),
                closed_phi=True,
                include_curvature=include_curvature,
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
    n_phi = int(master_grid["grid_n_phi"])
    n_length = int(master_grid["grid_n_length"])
    points = _grid(master_grid["inner_points"], n_phi, n_length)
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
    grid = build_point_grid(params)
    vertical_offset = float(grid.get("vertical_offset_mm", 0.0) or 0.0)
    if vertical_offset:
        n_phi = int(grid["grid_n_phi"])
        n_length = int(grid["grid_n_length"])
        for key in ("inner_points", "outer_points"):
            if grid.get(key) is None:
                continue
            values = _grid(grid[key], n_phi, n_length)
            values[:, :, 1] += vertical_offset
            grid[key] = values.reshape(-1).tolist()
    output["params"] = params
    output["grid"] = grid


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
    deferred_wall = 0.0
    if has_corners and str(_parsed_mode) == "freestanding":
        deferred_wall = float(eval_param(parsed_params.get("wallThickness"), 0.0, 0.0))
        if deferred_wall > 0.0:
            mesh = dict(sampling_config["mesh"])
            mesh["wall_thickness_mm"] = 0.0
            for alias in ("wallThickness", "wall_thickness", "WallThickness"):
                mesh.pop(alias, None)
            sampling_config["mesh"] = mesh
    output = build_viewport_geometry_from_config(
        sampling_config
    )
    if has_corners:
        _replace_grid_with_corner_refinement(output, corner_intervals)
    if deferred_wall > 0.0:
        output["params"]["wallThickness"] = deferred_wall
    canonical_ms = (time.perf_counter() - canonical_start) * 1000.0

    grid_data = output["grid"]
    n_phi = int(grid_data["grid_n_phi"])
    n_length = int(grid_data["grid_n_length"])
    closed_phi = bool(grid_data.get("full_circle", True))
    inner_canonical = _grid(grid_data["inner_points"], n_phi, n_length)
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
    if grid_data.get("outer_points") is not None:
        outer_canonical = _grid(grid_data["outer_points"], n_phi, n_length)
        selected_outer = outer_canonical[np.ix_(phi_indices, t_indices)]
        if options.include_outer:
            outer_master = _surface_grid(outer_canonical)
            outer_normals = analytic_grid_normals(
                outer_master,
                closed_phi=closed_phi,
                t_coordinates=master_t,
                phi_coordinates=master_phi,
            )
            outer_curvature_mean = outer_curvature_principal = None
            if options.include_curvature:
                outer_curvature_mean, outer_curvature_principal = (
                    analytic_grid_curvature(
                        outer_master,
                        closed_phi=closed_phi,
                        t_coordinates=master_t,
                        phi_coordinates=master_phi,
                    )
                )
            surfaces.append(
                _grid_surface_from_selection(
                    "horn.outer",
                    outer_master,
                    outer_normals,
                    t_indices,
                    phi_indices,
                    closed_phi=closed_phi,
                    curvature_mean=outer_curvature_mean,
                    curvature_principal=outer_curvature_principal,
                )
            )
            fidelity["horn.outer"] = _fidelity_record(
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
            selected_inner, deferred_wall, full_circle=closed_phi
        )
        outer_canonical = selected_outer
        if options.include_outer:
            selected_outer_master = _surface_grid(selected_outer)
            selected_outer_normals = analytic_grid_normals(
                selected_outer_master,
                closed_phi=closed_phi,
                t_coordinates=master_t[t_indices],
                phi_coordinates=master_phi[np.ix_(t_indices, phi_indices)],
            )
            selected_outer_mean = selected_outer_principal = None
            if options.include_curvature:
                selected_outer_mean, selected_outer_principal = (
                    analytic_grid_curvature(
                        selected_outer_master,
                        closed_phi=closed_phi,
                        t_coordinates=master_t[t_indices],
                        phi_coordinates=master_phi[np.ix_(t_indices, phi_indices)],
                    )
                )
            outer_positions = selected_outer_master.reshape(-1, 3)
            outer_flat_normals = selected_outer_normals.reshape(-1, 3)
            outer_oriented = _orient_indices_to_normals(
                "horn.outer",
                outer_positions,
                _grid_indices(
                    *selected_outer_master.shape[:2], closed_phi=closed_phi
                ),
                outer_flat_normals,
            )
            surfaces.append(
                PreviewSurfaceV1(
                    role="horn.outer",
                    positions=outer_positions,
                    indices=outer_oriented.indices,
                    normals=outer_flat_normals,
                    shading="smooth",
                    normal_method="analytic-parametric",
                    closed_phi=closed_phi,
                    curvature_mean=(
                        None
                        if selected_outer_mean is None
                        else selected_outer_mean.reshape(-1)
                    ),
                    curvature_principal=(
                        None
                        if selected_outer_principal is None
                        else selected_outer_principal.reshape(-1)
                    ),
                    metadata=_orientation_metadata(outer_oriented),
                )
            )
            fidelity["horn.outer"] = _fidelity_record(
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
                    "mouth_rim": 2 * len(phi_indices),
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
        surfaces.append(
            _flat_cap(
                "wall.rear_cap",
                selected_outer[:, 0, :],
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
            "uses_zipper_indices": bool(output.get("enclosure") is not None),
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
