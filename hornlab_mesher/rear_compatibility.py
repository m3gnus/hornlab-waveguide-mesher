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


def text_import_geometry_class(native_class):
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


def freestanding_rear_ring(inner_points, outer_points, wall_mm, *, text_import=False):
    """Project the outer throat onto the rear plane without moving the bore.

    Native geometry anchors the rear to the mean inner throat. ATH anchors
    each rear point to its outer throat minus the unscaled wall thickness.
    Imported near-planar rings use the minimax horizontal plane only within
    the explicit boundary-displacement budget; materially warped rings fail.
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
            raise ValueError(
                "ATH text import has a nonplanar rear ring; warped rear caps are unsupported"
            )
        rear[:, 2] = plane_z
    else:
        rear[:, 2] = float(np.mean(inner[:, 0, 2]) - float(wall_mm))
    return rear
