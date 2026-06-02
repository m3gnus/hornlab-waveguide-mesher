from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CrossSection:
    """Superellipse cross-section sampled around each axial station."""

    exponent: float = 2.0
    aspect_ratio: float = 1.0


@dataclass(frozen=True)
class Enclosure:
    """Optional enclosure/cabinet metadata.

    The first mesher release keeps horn enclosures as surface metadata only.
    Standalone cabinet meshes are represented by :class:`CabinetGeometry`.
    """

    depth_mm: float = 0.0
    wall_thickness_mm: float = 0.0


@dataclass(frozen=True)
class AxiHornGeometry:
    """Horn profile defined by ``(z_mm, r_mm)`` control points."""

    profile_points: NDArray[np.float64]
    throat_radius_mm: float
    cross_section: CrossSection = field(default_factory=CrossSection)
    enclosure: Enclosure = field(default_factory=Enclosure)
    n_phi: int = 64


@dataclass(frozen=True)
class RosseHornGeometry:
    """R-OSSE waveguide horn evaluated through the canonical WG JS pipeline.

    ROSSE derives its axial length internally from ``R``, ``r0``, ``k``, ``a``,
    and ``a0`` (see ``calculateROSSE``), so unlike OSSE there is no separate
    ``L`` knob. Defaults for ``m`` / ``r`` / ``b`` use WG canonical values when
    left at ``None``.
    """

    R_mm: float = 150.0
    r0_mm: float = 12.7
    a_deg: float = 60.0
    a0_deg: float = 15.5
    k: float = 1.0
    q: float = 1.0
    m: float | None = None
    r: float | None = None
    b: float | None = None
    cross_section: CrossSection = field(default_factory=CrossSection)
    enclosure: Enclosure = field(default_factory=Enclosure)
    n_phi: int = 64
    n_axial: int = 32


@dataclass(frozen=True)
class LookupHornGeometry:
    """Lookup-table waveguide horn (PCHIP interpolation of (z, r) control points)."""

    lookup_points: NDArray[np.float64]
    cross_section: CrossSection = field(default_factory=CrossSection)
    enclosure: Enclosure = field(default_factory=Enclosure)
    n_phi: int = 64
    n_axial: int = 32


@dataclass(frozen=True)
class OsseHornGeometry:
    """OSSE waveguide horn evaluated through the canonical WG JS pipeline.

    The (z, r) profile is computed via the geometry-cli subprocess so this
    builder shares a single source of truth with the WG browser UI and the
    Optimizer. The mesh build is then handed to the axisymmetric builder.
    """

    L_mm: float = 120.0
    r0_mm: float = 12.7
    a_deg: float = 60.0
    a0_deg: float = 15.5
    k: float = 1.0
    n: float = 4.0
    q: float = 0.995
    s: float = 0.0
    throat_ext_length_mm: float = 0.0
    throat_ext_angle_deg: float = 0.0
    slot_length_mm: float = 0.0
    rot_deg: float = 0.0
    cross_section: CrossSection = field(default_factory=CrossSection)
    enclosure: Enclosure = field(default_factory=Enclosure)
    n_phi: int = 64
    n_axial: int = 32


@dataclass(frozen=True)
class RectHornGeometry:
    """Rectangular conical horn with per-axis expansion."""

    primary_h_deg: float
    primary_v_deg: float
    mouth_width_mm: float
    mouth_height_mm: float
    throat_diameter_mm: float = 25.4
    flare2_h_deg: float = 0.0
    flare2_v_deg: float = 0.0
    flare2_ratio: float = 0.7
    throat_type: Literal["osse", "quadratic", "none"] = "osse"
    throat_driver_deg: float = 15.5
    throat_length_mm: float = 0.0
    body_fillet_mm: float = 0.0
    enclosure: Enclosure = field(default_factory=Enclosure)
    n_phi: int = 80
    n_length: int = 18


@dataclass(frozen=True)
class SlotConfig:
    """Cabinet slot/aperture patch."""

    opening_w_mm: float
    depth_mm: float
    height_mm: float
    apex_width_mm: float = 0.0
    apex_depth_mm: float = 0.0
    topology: Literal["front", "side"] = "front"
    exit_x_offset_mm: float = 0.0


@dataclass(frozen=True)
class DriverConfig:
    """One circular velocity-source patch on a cabinet surface."""

    diameter_mm: float
    tag: int
    position: Literal["slot_wall", "rear_panel"] = "rear_panel"
    offset_mm: float = 0.0


@dataclass(frozen=True)
class CabinetGeometry:
    """Surface-shell cabinet with front apertures and driver source patches."""

    width_mm: float
    depth_mm: float
    height_mm: float
    slots: list[SlotConfig] = field(default_factory=list)
    drivers: list[DriverConfig] = field(default_factory=list)
    aperture_width_mm: float = 0.0
    aperture_height_mm: float = 0.0


@dataclass(frozen=True)
class HornEnclosure:
    """Cabinet enclosure around a point-grid horn (WG ``enc_*`` payload family).

    Defaults mirror ``WaveguideParamsRequest`` so a caller can construct one
    with only ``depth_mm`` set and get the WG-default cabinet shape.

    ``plan_type``: 1=rounded rectangle, 2=ellipse, 3=superellipse.
    ``edge_type``: 1=rounded fillet, 2=chamfer.
    """

    depth_mm: float
    space_l_mm: float = 25.0
    space_t_mm: float = 25.0
    space_r_mm: float = 25.0
    space_b_mm: float = 25.0
    edge_mm: float = 18.0
    edge_type: Literal[1, 2] = 1
    plan_type: Literal[1, 2, 3] = 1
    plan_n: float = 2.0
    depth_margin_mm: float = 1.0


@dataclass(frozen=True)
class PointGridHornGeometry:
    """WG-compatible horn surface from an already-evaluated point grid.

    ``inner_points`` uses the Waveguide-Generator OCC payload shape:
    ``(n_phi, n_length + 1, 3)`` in millimetres. This keeps WG in charge of
    formulas/profile evaluation while the mesher owns Gmsh authoring.

    Three top-level cases, gated by ``enclosure`` and ``outer_points``:

    * ``enclosure is None`` and ``outer_points is None`` — inner-only horn
      (case A).
    * ``enclosure is None`` and ``outer_points is not None`` — freestanding
      wall-shell horn (case B). ``wall_thickness_mm`` is used by the legacy
      rear-disc fallback; the active path reuses outer wall boundary curves.
    * ``enclosure is not None`` — horn inside a cabinet enclosure (case C).
    """

    inner_points: NDArray[np.float64]
    preserve_grid: bool = False
    closed: bool = True
    outer_points: NDArray[np.float64] | None = None
    wall_thickness_mm: float = 6.0
    source_shape: int = 1
    source_radius_mm: float = -1.0
    source_curv: int = 0
    source_auto_angle_deg: float | None = None
    ath_parity_topology: bool = False
    enclosure: HornEnclosure | None = None


HornGeometry = (
    AxiHornGeometry
    | OsseHornGeometry
    | LookupHornGeometry
    | RectHornGeometry
    | CabinetGeometry
    | PointGridHornGeometry
)
# Note: RosseHornGeometry is intentionally NOT in the buildable union — the
# ROSSE curve is non-monotonic in z for typical parameter ranges, so
# build_axisymmetric cannot consume it. Use compute_rosse_profile_points
# (in builders.rosse_waveguide) for the curve, or hand the result to a
# free-form sweep builder once one exists.


@dataclass(frozen=True)
class MeshDensity:
    """Mesh sizing parameters in millimetres."""

    throat_res_mm: float = 4.0
    mouth_res_mm: float = 26.0
    rear_res_mm: float = 25.0
    enc_front_res_mm: float | str | None = None
    enc_back_res_mm: float | str | None = None
    min_size_mm: float | None = None
    max_size_mm: float | None = None


@dataclass
class BuiltGeometry:
    """OCC surface groups returned by geometry builders."""

    surface_groups: dict[int, list[int]]
    axial_bounds_mm: tuple[float, float]
    source_axis: Literal["x", "y", "z"] = "z"
    mesh_surface_groups: dict[str, list[int]] = field(default_factory=dict)
    enclosure_bounds: dict[str, float] | None = None
    symmetry_snap_axes: tuple[Literal["x", "y", "z"], ...] = ()
    symmetry_snap_tol_mm: float = 1.0e-6
    mesh_algorithm: int | None = None


@dataclass(frozen=True)
class MeshInfo:
    path: Path
    n_vertices: int
    n_triangles: int
    physical_groups: dict[int, str]
    bounding_box: tuple[NDArray[np.float64], NDArray[np.float64]]
    units: Literal["m", "mm"]
