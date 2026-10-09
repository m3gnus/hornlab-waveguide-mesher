"""A circular native source on a finite rectangular, closed enclosure."""

import math
from dataclasses import asdict, dataclass

FEATURE = "native-front-baffle-woofer-v1"


@dataclass(frozen=True)
class FrontBaffle:
    width_mm: float
    height_mm: float
    depth_mm: float
    aperture_radius_mm: float
    center_mm: tuple[float, float, float] = (0, 0, 0)
    aperture_id: str = "baffle.aperture"

    def __post_init__(self):
        for name in ("width_mm", "height_mm", "depth_mm", "aperture_radius_mm"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("baffle dimensions must be finite and positive")
            object.__setattr__(self, name, float(value))
        if len(self.center_mm) != 3 or any(
            type(x) not in (int, float) or not math.isfinite(x) for x in self.center_mm
        ):
            raise ValueError("source center must be a finite XYZ vector")
        object.__setattr__(
            self, "center_mm", tuple(float(x) if x else 0.0 for x in self.center_mm)
        )
        if not isinstance(self.aperture_id, str) or not self.aperture_id.strip():
            raise ValueError("aperture identity must be nonempty")

    def validate(self, contour):
        if self.aperture_radius_mm < contour.points[-1].r_mm:
            raise ValueError("aperture cannot be smaller than the contour footprint")
        gap = self.aperture_radius_mm - contour.points[-1].r_mm
        if 0 < gap <= 0.1:
            raise ValueError("a mounting collar must be wider than 0.1 mm")
        x, y, _ = self.center_mm
        clearance = min(self.width_mm / 2 - abs(x), self.height_mm / 2 - abs(y))
        if clearance - self.aperture_radius_mm <= 0.1:
            raise ValueError(
                "aperture must clear every baffle edge by more than 0.1 mm"
            )
        if self.depth_mm + min(0, contour.axial_bounds_mm[0]) <= 0.1:
            raise ValueError("source must clear the enclosure back by more than 0.1 mm")

    def to_dict(self):
        return {"version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        version = value.pop("version", None)
        if type(version) is not int or version != 1:
            raise ValueError("unsupported front baffle version")
        return cls(**value)

    def preview(self, contour, **sizes):
        """Translate the same canonical source used by the exporter."""
        import numpy as np

        self.validate(contour)
        return {
            key: xyz + np.asarray(self.center_mm)
            for key, xyz in contour.preview(**sizes).items()
        }

    def surfaces(self, contour):
        """Finite planar rigid faces; each has a stable structural role."""
        self.validate(contour)
        w, h, d = self.width_mm / 2, self.height_mm / 2, self.depth_mm
        cx, cy, z = self.center_mm
        r = self.aperture_radius_mm
        specs = {
            "front": (
                2,
                z,
                (-w, w),
                (-h, h),
                1,
                self.width_mm * self.height_mm - math.pi * r**2,
            ),
            "back": (2, z - d, (-w, w), (-h, h), -1, self.width_mm * self.height_mm),
            "left": (0, -w, (-h, h), (z - d, z), -1, self.height_mm * d),
            "right": (0, w, (-h, h), (z - d, z), 1, self.height_mm * d),
            "bottom": (1, -h, (-w, w), (z - d, z), -1, self.width_mm * d),
            "top": (1, h, (-w, w), (z - d, z), 1, self.width_mm * d),
        }
        if r > contour.points[-1].r_mm:
            specs["collar"] = (
                2,
                z,
                (cx - r, cx + r),
                (cy - r, cy + r),
                1,
                math.pi * (r**2 - contour.points[-1].r_mm ** 2),
            )
        return specs

    def distance(self, contour, role, xyz):
        """Exact distance to a finite rigid face, including circular trims."""
        import numpy as np

        axis, plane, a, b, _, _ = self.surfaces(contour)[role]
        other = [i for i in range(3) if i != axis]
        projection = np.stack(
            [np.clip(xyz[..., other[0]], *a), np.clip(xyz[..., other[1]], *b)], axis=-1
        )
        planar = np.linalg.norm(xyz[..., other] - projection, axis=-1)
        if role in ("front", "collar"):
            radius = np.linalg.norm(xyz[..., :2] - self.center_mm[:2], axis=-1)
            if role == "front":
                planar = np.where(
                    radius < self.aperture_radius_mm,
                    self.aperture_radius_mm - radius,
                    planar,
                )
            else:
                planar = np.maximum(
                    np.maximum(
                        contour.points[-1].r_mm - radius,
                        radius - self.aperture_radius_mm,
                    ),
                    0,
                )
        return np.hypot(xyz[..., axis] - plane, planar)
