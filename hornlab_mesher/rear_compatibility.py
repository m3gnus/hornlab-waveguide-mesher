"""Rear-return interpretation for the bounded ATH text import contract."""
from dataclasses import dataclass
import numpy as np
from .geometry import (PointGridHornGeometry, _StretchedPointGridHornGeometry,
                       _MouthFittedPointGridHornGeometry, _MouthFittedStretchedPointGridHornGeometry)


class _TextImportedRear:
    # Class-level behavior leaves native dataclass fields/state untouched.
    ath_text_import_rear = True


@dataclass(frozen=True)
class _TextImportedPointGrid(_TextImportedRear, PointGridHornGeometry):
    pass


@dataclass(frozen=True)
class _TextImportedStretchedPointGrid(_TextImportedRear, _StretchedPointGridHornGeometry):
    pass


@dataclass(frozen=True)
class _TextImportedMouthFittedPointGrid(_TextImportedRear, _MouthFittedPointGridHornGeometry):
    pass


@dataclass(frozen=True)
class _TextImportedMouthFittedStretchedPointGrid(_TextImportedRear, _MouthFittedStretchedPointGridHornGeometry):
    pass


def text_import_geometry_class(native_class, *, warped=False):
    if warped:
        return _WARPED_CLASSES[native_class]
    return {
        PointGridHornGeometry: _TextImportedPointGrid,
        _StretchedPointGridHornGeometry: _TextImportedStretchedPointGrid,
        _MouthFittedPointGridHornGeometry: _TextImportedMouthFittedPointGrid,
        _MouthFittedStretchedPointGridHornGeometry: _TextImportedMouthFittedStretchedPointGrid,
    }[native_class]


# Explicit maximum displacement of an imported rear boundary in its planar
# representation: 0.1 micrometre, and at most 0.01% of wall thickness.
_REAR_PLANAR_DISPLACEMENT_MM = 1.0e-4
_REAR_PLANAR_WALL_FRACTION = 1.0e-4


def freestanding_rear_ring(inner_points, outer_points, wall_mm, *, text_import=False, allow_warped=True):
    """Project the outer throat onto the rear plane without moving the bore.

    Native geometry anchors the rear to the mean inner throat. ATH anchors
    each rear point to its outer throat minus the unscaled wall thickness.
    Imported near-planar rings keep the minimax horizontal plane within
    the displacement budget; v2 warped rings use the measured RearShape 1
    rule. The resolver disables warped construction for saved v1 imports.
    The bore and outer shell are never projected or translated.
    """
    inner = np.asarray(inner_points, dtype=np.float64)
    outer = np.asarray(outer_points, dtype=np.float64)
    rear = np.array(outer[:, 0, :], dtype=np.float64, copy=True)
    if text_import:
        wall = float(wall_mm)
        if (not np.all(np.isfinite(rear)) or not np.isfinite(wall)
                or wall <= 0):
            raise ValueError("ATH text import rear ring and positive wall must be finite")
        rear[:, 2] -= wall
        if not np.all(np.isfinite(rear)):
            raise ValueError("ATH text import rear ring must be finite")
        low, high = float(np.min(rear[:, 2])), float(np.max(rear[:, 2]))
        # The midrange minimizes the maximum vertical boundary displacement.
        # Certify the actual float result, rather than assuming half the span:
        # subtraction/rounding can matter when coordinates are very large.
        plane_z = low + (high - low) / 2.0
        displacement = float(np.max(np.abs(rear[:, 2] - plane_z)))
        budget = min(_REAR_PLANAR_DISPLACEMENT_MM,
                     _REAR_PLANAR_WALL_FRACTION * wall)
        if not np.isfinite(displacement) or displacement > budget:
            if not allow_warped:
                raise ValueError(
                    "ATH text import has a nonplanar rear ring; warped rear caps are unsupported"
                )
            # Measured ATH V2026-08c RearShape 1 throat construction.
            # The resolver supplies authored-grid normals to all consumers;
            # this fallback also permits direct grid callers to build a fan.
            from .ath_import_surfaces import meridional_normals
            return inner[:, 0] + wall * meridional_normals(inner)[:, 0] - np.array([0., 0., 5.])
        rear[:, 2] = plane_z
    else:
        rear[:, 2] = float(np.mean(inner[:, 0, 2]) - float(wall_mm))
    return rear


@dataclass(frozen=True)
class _WarpedRear:
    ath_warped_rear_points: np.ndarray | None = None
    ath_warped_rear_apex: np.ndarray | None = None


@dataclass(frozen=True)
class _WarpedTextImportedPointGrid(_WarpedRear, _TextImportedPointGrid):
    pass


@dataclass(frozen=True)
class _WarpedTextImportedStretchedPointGrid(_WarpedRear, _TextImportedStretchedPointGrid):
    pass


@dataclass(frozen=True)
class _WarpedTextImportedMouthFittedPointGrid(_WarpedRear, _TextImportedMouthFittedPointGrid):
    pass


@dataclass(frozen=True)
class _WarpedTextImportedMouthFittedStretchedPointGrid(_WarpedRear, _TextImportedMouthFittedStretchedPointGrid):
    pass


_WARPED_CLASSES = {
    _TextImportedPointGrid: _WarpedTextImportedPointGrid,
    _TextImportedStretchedPointGrid: _WarpedTextImportedStretchedPointGrid,
    _TextImportedMouthFittedPointGrid: _WarpedTextImportedMouthFittedPointGrid,
    _TextImportedMouthFittedStretchedPointGrid: _WarpedTextImportedMouthFittedStretchedPointGrid,
}


def geometry_rear_ring(geometry, inner, outer):
    rear = getattr(geometry, "ath_warped_rear_points", None)
    if rear is not None:
        return rear
    return freestanding_rear_ring(inner, outer, geometry.wall_thickness_mm,
                                  text_import=getattr(geometry, "ath_text_import_rear", False))


def resample_rear_ring(ring, target_ring, *, closed):
    """Interpolate rear radius and height in the source angular interval.

    Polar interpolation preserves circular rings and reduced domains that
    cross zero azimuth. It also keeps the authored rear knots exact.
    """
    from scipy.interpolate import CubicSpline
    ring = np.asarray(ring, dtype=float)
    target = np.asarray(target_ring, dtype=float)
    phi = np.unwrap(np.arctan2(ring[:, 1], ring[:, 0]))
    target_phi = np.arctan2(target[:, 1], target[:, 0])
    target_phi += 2*np.pi * np.round((.5*(phi[0]+phi[-1])-target_phi)/(2*np.pi))
    values = np.column_stack((np.linalg.norm(ring[:, :2], axis=1), ring[:, 2]))
    if closed:
        target_phi = phi[0] + np.mod(target_phi-phi[0], 2*np.pi)
        phi = np.append(phi, phi[0]+2*np.pi)
        values = np.vstack((values, values[0]))
    else:
        target_phi = np.clip(target_phi, phi[0], phi[-1])
    radius, height = CubicSpline(phi, values, bc_type="periodic" if closed else "not-a-knot")(target_phi).T
    return np.column_stack((radius*np.cos(target_phi), radius*np.sin(target_phi), height))
