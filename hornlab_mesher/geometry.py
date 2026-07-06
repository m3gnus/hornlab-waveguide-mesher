from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CrossSection:
    """Superellipse cross-section sampled around each axial station."""

    exponent: float = 2.0
    aspect_ratio: float = 1.0


@dataclass(frozen=True)
class Enclosure:
    """Optional waveguide rear-enclosure metadata."""

    depth_mm: float = 0.0
    wall_thickness_mm: float = 0.0


@dataclass(frozen=True)
class _AxiHornGeometry:
    """Internal axial loft input used by the OSSE builder."""

    profile_points: NDArray[np.float64]
    throat_radius_mm: float
    cross_section: CrossSection = field(default_factory=CrossSection)
    enclosure: Enclosure = field(default_factory=Enclosure)
    n_phi: int = 64


@dataclass(frozen=True)
class RosseHornGeometry:
    """R-OSSE waveguide profile parameters.

    ROSSE derives its axial length internally from ``R``, ``r0``, ``k``, ``a``,
    and ``a0`` (see ``calculateROSSE``), so unlike OSSE there is no separate
    ``L`` knob. This dataclass is intentionally limited to the 2D profile
    helper inputs; use config-driven point-grid builds for full R-OSSE meshes.
    Defaults for ``m`` / ``r`` / ``b`` use WG canonical values when left at
    ``None``.
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
    throat_ext_length_mm: float = 0.0
    throat_ext_angle_deg: float = 0.0
    slot_length_mm: float = 0.0
    n_axial: int = 32


@dataclass(frozen=True)
class OsseHornGeometry:
    """OSSE waveguide profile parameters.

    The mesh build is handed to an internal axial loft helper.
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
class HornEnclosure:
    """Rear enclosure around a point-grid waveguide (WG ``enc_*`` payload family).

    Defaults mirror ``WaveguideParamsRequest`` so a caller can construct one
    with only ``depth_mm`` set and get the WG-default enclosure shape.

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
    front_mesh_size_mm: float | None = None
    back_mesh_size_mm: float | None = None


@dataclass(frozen=True)
class HornInterface:
    """Offset interface surface tied to a point-grid axial slice.

    ``slice_index`` is the zero-based point-grid ring index. The legacy
    single ``interface_offset_mm`` path maps to the final mouth slice.
    """

    slice_index: int
    offset_mm: float


class PointGridBuildMode(str, Enum):
    """Topology mode implied by point-grid geometry options."""

    BARE = "bare"
    FREESTANDING = "freestanding"
    ENCLOSURE = "enclosure"
    INFINITE_BAFFLE = "infinite-baffle"


@dataclass(frozen=True)
class PointGridHornGeometry:
    """WG-compatible horn surface from an already-evaluated point grid.

    ``inner_points`` has shape ``(n_phi, n_length + 1, 3)`` in millimetres.
    Python owns both profile evaluation and Gmsh authoring.

    Three top-level cases, gated by ``enclosure`` and ``outer_points``:

    * ``enclosure is None`` and ``outer_points is None`` - inner-only horn
      (case A).
    * ``enclosure is None`` and ``outer_points is not None`` - freestanding
      wall-shell horn (case B). ``wall_thickness_mm`` is used by the legacy
      rear-disc fallback; the active path reuses outer wall boundary curves.
    * ``enclosure is not None`` - waveguide inside a rear enclosure (case C).
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
    interface_offset_mm: float = 0.0
    interfaces: tuple[HornInterface, ...] = ()
    wg_topology: bool = True
    enclosure: HornEnclosure | None = None
    infinite_baffle: bool = False
    # Symmetry planes bounding an open (``closed=False``) grid, as snap axes:
    # ``"x"`` is the x=0 (yz) plane, ``"y"`` is the y=0 (xz) plane. The default
    # ``("x", "y")`` is the quarter model (mirrored about both planes). Half
    # models set a single plane: ``("y",)`` for quadrants 12 (xz mirror),
    # ``("x",)`` for quadrants 14 (yz mirror). Unused when ``closed`` is True.
    symmetry_planes: tuple[str, ...] = ("x", "y")
    # Mesh.VerticalOffset, in millimetres. The point grid is built at the origin
    # (cut planes on the coordinate axes); this offset is applied as a single
    # rigid +y translation of the finished mesh, after all reduced-domain
    # cut-plane logic has run at y=0. The declared symmetry plane stays at y=0,
    # so a y-cut (quadrants 1/12) reconstructs about y=0 -- matching ATH.
    vertical_offset_mm: float = 0.0

    @property
    def build_mode(self) -> PointGridBuildMode:
        if self.enclosure is not None:
            return PointGridBuildMode.ENCLOSURE
        if self.infinite_baffle:
            return PointGridBuildMode.INFINITE_BAFFLE
        if self.outer_points is not None:
            return PointGridBuildMode.FREESTANDING
        return PointGridBuildMode.BARE


HornGeometry = (
    OsseHornGeometry
    | PointGridHornGeometry
)
# Note: RosseHornGeometry is intentionally NOT in the buildable union - the
# ROSSE curve is non-monotonic in z for typical parameter ranges, so
# the internal axial loft helper cannot consume it. Use compute_rosse_profile_points
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
    interface_res_mm: float | None = None
    min_size_mm: float | None = None
    max_size_mm: float | None = None
    # Frequency-aware sizing: when max_frequency_hz is set, each resolution
    # is clamped to c / (epw_role * f) so the mesh stays valid for the
    # requested band; the mm knobs above still apply where finer. The
    # per-role targets grade the mesh by acoustic importance: the throat
    # carries the strongest, most detailed field, the inner wall interpolates
    # toward the mouth, and shadowed outer/rear surfaces contribute little to
    # the radiated field so they tolerate fewer elements per wavelength.
    max_frequency_hz: float | None = None
    elements_per_wavelength: float = 6.0
    throat_epw: float = 8.0
    mouth_epw: float = 6.0
    rear_epw: float = 2.5
    interface_epw: float = 6.0
    speed_of_sound_m_s: float = 343.0
    # Gmsh Mesh.MeshSizeFromCurvature segments per 2*pi (0 disables).
    curvature_segments: int = 0

    def _ceiling_mm(self, epw: float) -> float | None:
        if not self.max_frequency_hz or self.max_frequency_hz <= 0.0:
            return None
        epw = max(float(epw), 1.0)
        return (float(self.speed_of_sound_m_s) * 1000.0) / (epw * float(self.max_frequency_hz))

    def frequency_ceiling_mm(self) -> float | None:
        """Global ceiling at the generic elements-per-wavelength target."""

        return self._ceiling_mm(self.elements_per_wavelength)

    def role_ceiling_mm(self, role: str) -> float | None:
        epw = {
            "throat": self.throat_epw,
            "mouth": self.mouth_epw,
            "rear": self.rear_epw,
            "interface": self.interface_epw,
        }.get(role, self.elements_per_wavelength)
        return self._ceiling_mm(epw)


@dataclass
class BuiltGeometry:
    """OCC surface groups returned by geometry builders."""

    surface_groups: dict[int, list[int]]
    axial_bounds_mm: tuple[float, float]
    source_axis: Literal["x", "y", "z", "-x", "-y", "-z"] = "z"
    mesh_surface_groups: dict[str, list[int]] = field(default_factory=dict)
    enclosure_bounds: dict[str, float] | None = None
    symmetry_snap_axes: tuple[Literal["x", "y", "z"], ...] = ()
    symmetry_snap_tol_mm: float = 1.0e-6
    mesh_algorithm: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MeshInfo:
    path: Path
    n_vertices: int
    n_triangles: int
    physical_groups: dict[int, str]
    bounding_box: tuple[NDArray[np.float64], NDArray[np.float64]]
    units: Literal["m", "mm"]
    # Per physical tag edge-length statistics in millimetres:
    # {tag: {"median_edge_mm": ..., "p95_edge_mm": ..., "max_edge_mm": ...}}.
    edge_stats_mm: dict[int, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
