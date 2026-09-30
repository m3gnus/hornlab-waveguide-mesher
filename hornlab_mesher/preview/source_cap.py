"""The preview source cap: the meshed cap's geometry drawn at render density."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..builders.point_grid_sources import _source_cap_height, _source_cap_radius
from ..config_builder import _source_auto_angle_deg
from ..geometry import PointGridHornGeometry
from ..profiles import eval_param
from .contract import (
    PreviewSurfaceV1,
    _orient_indices_to_normals,
    _orientation_metadata,
)
from .primitives import _flat_cap


def _source_geometry(
    params: Mapping[str, Any], formula: str, inner: NDArray[np.float64]
) -> PointGridHornGeometry:
    # Shared with the solved build so the previewed cap is the meshed cap.
    auto_angle = _source_auto_angle_deg(params, formula)
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
    if not closed_phi:
        # A reduced (quadrant/half) model's throat ring is an arc, so its mean
        # lies off the axis. The solve builds the cap at the origin, then
        # translates the finished model. The preview ring is already placed,
        # so its axis must retain that same rigid Y translation.
        center[0] = 0.0
        center[1] = float(eval_param(params.get("verticalOffset"), 0.0, 0.0))
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
    if closed_phi:
        angular_step = 2.0 * math.pi / n_phi
    else:
        # The ring's real azimuth steps: a quadrant spans a quarter turn, not
        # the half turn ``pi / (n_phi - 1)`` assumed.
        azimuth = np.unwrap(np.arctan2(directions[:, 1], directions[:, 0]))
        angular_step = float(np.max(np.abs(np.diff(azimuth)))) if n_phi > 1 else 0.0
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
