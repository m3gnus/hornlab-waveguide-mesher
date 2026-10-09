"""Two aligned physical sources on one finite enclosure and observation frame."""

import math
from dataclasses import dataclass
from urllib.parse import quote

from .front_baffle import FrontBaffle
from .source_contour import SourceContour, digest

FEATURE = "native-shared-horn-woofer-v1"


def patch_id(contour, segment):
    """Injective source-qualified identity, independent of CAD and mesh tags."""
    return quote(contour.physical_source_id, safe="") + "/" + quote(segment.id, safe="")


@dataclass(frozen=True)
class SourceAssembly:
    horn: SourceContour
    woofer: SourceContour
    width_mm: float
    height_mm: float
    depth_mm: float
    front_z_mm: float
    horn_xy_mm: tuple[float, float]
    horn_length_mm: float
    mouth_radius_mm: float
    woofer_xy_mm: tuple[float, float]
    aperture_radius_mm: float

    def __post_init__(self):
        for name in (
            "width_mm",
            "height_mm",
            "depth_mm",
            "front_z_mm",
            "horn_length_mm",
            "mouth_radius_mm",
            "aperture_radius_mm",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("assembly dimensions must be finite numbers")
            if name != "front_z_mm" and value <= 0:
                raise ValueError("assembly dimensions must be positive")
            object.__setattr__(self, name, float(value) if value else 0.0)
        for name in ("horn_xy_mm", "woofer_xy_mm"):
            value = getattr(self, name)
            if len(value) != 2 or any(
                type(x) not in (int, float) or not math.isfinite(x) for x in value
            ):
                raise ValueError("assembly placement must be a finite XY pair")
            object.__setattr__(self, name, tuple(float(x) if x else 0.0 for x in value))
        if not isinstance(self.horn, SourceContour) or not isinstance(
            self.woofer, SourceContour
        ):
            raise TypeError("assembly sources must use canonical contours")
        if (
            self.horn.physical_source_id == self.woofer.physical_source_id
            or self.horn.rim_id == self.woofer.rim_id
        ):
            raise ValueError(
                "assembly physical source and rim identities must be distinct"
            )
        self.baffle.validate(self.woofer)
        rim = self.horn.points[-1].r_mm
        if self.mouth_radius_mm - rim <= 0.1:
            raise ValueError("horn mouth must exceed throat radius by more than 0.1 mm")
        if self.horn_length_mm - max(0, self.horn.axial_bounds_mm[1]) <= 0.1:
            raise ValueError("horn source must clear mouth by more than 0.1 mm")
        if (
            self.depth_mm - self.horn_length_mm + min(0, self.horn.axial_bounds_mm[0])
            <= 0.1
        ):
            raise ValueError(
                "horn source must clear enclosure back by more than 0.1 mm"
            )
        if (
            min(
                self.width_mm / 2 - abs(self.horn_xy_mm[0]),
                self.height_mm / 2 - abs(self.horn_xy_mm[1]),
            )
            - self.mouth_radius_mm
            <= 0.1
        ):
            raise ValueError(
                "horn aperture must clear every baffle edge by more than 0.1 mm"
            )
        if (
            math.dist(self.horn_xy_mm, self.woofer_xy_mm)
            - self.mouth_radius_mm
            - self.aperture_radius_mm
            <= 0.1
        ):
            raise ValueError(
                "horn and woofer apertures must be separated by more than 0.1 mm"
            )

    @property
    def baffle(self):
        return FrontBaffle(
            self.width_mm,
            self.height_mm,
            self.depth_mm,
            self.aperture_radius_mm,
            (*self.woofer_xy_mm, self.front_z_mm),
            self.woofer.rim_id,
        )

    @property
    def parts(self):
        return [
            (
                self.horn,
                (*self.horn_xy_mm, self.front_z_mm - self.horn_length_mm),
                "HF",
            ),
            (self.woofer, (*self.woofer_xy_mm, self.front_z_mm), "LF"),
        ]

    @property
    def patches(self):
        return [
            (patch_id(c, s), c, i, origin, band)
            for c, origin, band in self.parts
            for i, s in enumerate(c.segments)
        ]

    def to_dict(self):
        return {
            "version": 1,
            "frame": "aligned-z-mm-v1",
            "horn": self.horn.to_dict(),
            "woofer": self.woofer.to_dict(),
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in ("horn", "woofer")
            },
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        version, frame = value.pop("version", None), value.pop("frame", None)
        if type(version) is not int or version != 1 or frame != "aligned-z-mm-v1":
            raise ValueError("unsupported native assembly version or frame")
        return cls(
            **{
                **value,
                "horn": SourceContour.from_dict(value["horn"]),
                "woofer": SourceContour.from_dict(value["woofer"]),
            }
        )

    @property
    def geometry_sha256(self):
        return digest(self.to_dict())

    def preview(self, **sizes):
        import numpy as np

        return {
            patch_id(c, s): np.asarray(c.preview(**sizes)[s.id]) + origin
            for c, origin, _ in self.parts
            for s in c.segments
        }

    def rigid_areas(self):
        result = {
            role: spec[-1] for role, spec in self.baffle.surfaces(self.woofer).items()
        }
        result["front"] -= math.pi * self.mouth_radius_mm**2
        r = self.horn.points[-1].r_mm
        result["horn-wall"] = (
            math.pi
            * (r + self.mouth_radius_mm)
            * math.hypot(self.horn_length_mm, self.mouth_radius_mm - r)
        )
        return result

    def rigid_distance(self, role, xyz):
        import numpy as np

        if role == "horn-wall":
            local = xyz - np.asarray(self.parts[0][1])
            r, z = np.linalg.norm(local[..., :2], axis=-1), local[..., 2]
            a = self.horn.points[-1].r_mm
            dr, dz = self.mouth_radius_mm - a, self.horn_length_mm
            t = np.clip(((r - a) * dr + z * dz) / (dr * dr + dz * dz), 0, 1)
            return np.hypot(r - a - t * dr, z - t * dz)
        result = self.baffle.distance(self.woofer, role, xyz)
        if role == "front":
            r = np.linalg.norm(xyz[..., :2] - self.horn_xy_mm, axis=-1)
            result = np.where(
                r < self.mouth_radius_mm,
                np.hypot(xyz[..., 2] - self.front_z_mm, self.mouth_radius_mm - r),
                result,
            )
        return result

    def patch_distance(self, identifier, xyz):
        """Exact distance to a finite revolved segment (including arc ends)."""
        import numpy as np

        _, contour, i, origin, _ = next(p for p in self.patches if p[0] == identifier)
        local = xyz - np.asarray(origin)
        r, z = np.linalg.norm(local[..., :2], axis=-1), local[..., 2]
        a, b = contour.points[i : i + 2]
        s = contour.segments[i]
        if s.kind == "line":
            dr, dz = b.r_mm - a.r_mm, b.z_mm - a.z_mm
            t = np.clip(
                ((r - a.r_mm) * dr + (z - a.z_mm) * dz) / (dr * dr + dz * dz), 0, 1
            )
            return np.hypot(r - a.r_mm - t * dr, z - a.z_mm - t * dz)
        radius, start, sweep = contour.arc(i)
        cr, cz = s.center_mm
        angle = np.arctan2(z - cz, r - cr)
        progress = np.mod((angle - start) * math.copysign(1, sweep), 2 * math.pi)
        progress = np.where(abs(progress - 2 * math.pi) < 1e-10, 0, progress)
        ends = np.minimum(
            np.hypot(r - a.r_mm, z - a.z_mm), np.hypot(r - b.r_mm, z - b.z_mm)
        )
        return np.where(
            progress <= abs(sweep) + 1e-10, abs(np.hypot(r - cr, z - cz) - radius), ends
        )


def assembly_channels(assembly, drives):
    if len(drives) != 2 or drives[0].channel_id == drives[1].channel_id:
        raise ValueError("assembly requires two distinct drive channels")
    channels = []
    for (c, _, _), drive in zip(assembly.parts, drives):
        drive.validate(c)
        weights = dict(drive.weights)
        channels.append(
            {
                "id": drive.channel_id,
                "physical_source_id": c.physical_source_id,
                "source_ids": [
                    patch_id(c, s) for s in c.segments if s.role == "moving"
                ],
                "patch_weights": {
                    patch_id(c, s): weights[s.id]
                    for s in c.segments
                    if s.role == "moving"
                },
                "motion": drive.motion,
            }
        )
    return channels
