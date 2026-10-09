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


def freestanding_rear_ring(inner_points, outer_points, wall_mm, *, text_import=False):
    """Project the outer throat onto the rear plane without moving the bore.

    Native geometry anchors the rear to the mean inner throat. ATH anchors
    each rear point to its outer throat minus the unscaled wall thickness.
    Current planar rear caps cannot represent an imported warped rear ring.
    """
    inner = np.asarray(inner_points, dtype=np.float64)
    outer = np.asarray(outer_points, dtype=np.float64)
    rear = np.array(outer[:, 0, :], dtype=np.float64, copy=True)
    if text_import:
        rear[:, 2] -= float(wall_mm)
        if float(np.ptp(rear[:, 2])) > 1e-7:
            raise ValueError(
                "ATH text import has a nonplanar rear ring; warped rear caps are unsupported"
            )
    else:
        rear[:, 2] = float(np.mean(inner[:, 0, 2]) - float(wall_mm))
    return rear
