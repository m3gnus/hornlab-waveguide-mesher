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
    woofer: SourceContour | None
    width_mm: float
    height_mm: float
    depth_mm: float
    front_z_mm: float
    horn_xy_mm: tuple[float, float]
    horn_length_mm: float
    mouth_radius_mm: float
    woofer_xy_mm: tuple[float, float]
    aperture_radius_mm: float
    phase_plugs: tuple = ()
    horn_wall: object | None = None

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
            if name == "aperture_radius_mm" and self.woofer is None:
                if value != 0:
                    raise ValueError("horn-only assembly requires zero unused woofer aperture")
            elif name != "front_z_mm" and value <= 0:
                raise ValueError("assembly dimensions must be positive")
            object.__setattr__(self, name, float(value) if value else 0.0)
        for name in ("horn_xy_mm", "woofer_xy_mm"):
            value = getattr(self, name)
            if len(value) != 2 or any(
                type(x) not in (int, float) or not math.isfinite(x) for x in value
            ):
                raise ValueError("assembly placement must be a finite XY pair")
            object.__setattr__(self, name, tuple(float(x) if x else 0.0 for x in value))
        if not isinstance(self.horn, SourceContour) or (self.woofer is not None and not isinstance(
            self.woofer, SourceContour
        )):
            raise TypeError("assembly sources must use canonical contours")
        if self.woofer is not None and (
            self.horn.physical_source_id == self.woofer.physical_source_id
            or self.horn.rim_id == self.woofer.rim_id
        ):
            raise ValueError(
                "assembly physical source and rim identities must be distinct"
            )
        self.baffle.validate(self.woofer or self.horn)
        if self.woofer is None and self.woofer_xy_mm != (0, 0):
            raise ValueError("horn-only assembly requires zero unused woofer placement")
        if self.horn_wall is not None:
            from .general_horn import GeneralHornWall
            if not isinstance(self.horn_wall, GeneralHornWall):
                raise TypeError("horn wall must use canonical general horn authority")
            if (abs(self.horn_wall.throat_radius_mm - self.horn.points[-1].r_mm) > 1e-8
                    or self.horn_wall.length_mm != self.horn_length_mm
                    or self.horn_wall.mouth_radius_mm != self.mouth_radius_mm):
                raise ValueError("source rim or assembly datums contradict resolved general horn wall")
            # Positive convex-hull margins certify the complete wall envelope,
            # including R-OSSE rollback outside the front aperture plane.
            radial_poles = [p[0] for p in self.horn_wall.poles_mm]
            if min(radial_poles) < self.horn.points[-1].r_mm - 1e-8:
                raise ValueError("general horn wall cannot constrict inside the source footprint")
            radial_envelope = max(radial_poles)
            if (min(self.width_mm / 2 - abs(self.horn_xy_mm[0]),
                    self.height_mm / 2 - abs(self.horn_xy_mm[1])) - radial_envelope <= .1):
                raise ValueError("general horn wall must clear every enclosure side by more than 0.1 mm")
            if max(self.horn_wall.radii_at_z(self.horn_length_mm)) > self.mouth_radius_mm + 1e-8:
                raise ValueError("general horn wall intersects the front baffle outside its aperture")
            if min(p[1] for p in self.horn_wall.poles_mm) + self.depth_mm - self.horn_length_mm <= .1:
                raise ValueError("general horn wall must clear the enclosure back by more than 0.1 mm")
            if self.woofer is not None and math.dist(self.horn_xy_mm, self.woofer_xy_mm) - radial_envelope - self.aperture_radius_mm <= .1:
                raise ValueError("general horn wall must clear the woofer aperture by more than 0.1 mm")
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
        if self.woofer is not None and (
            math.dist(self.horn_xy_mm, self.woofer_xy_mm)
            - self.mouth_radius_mm
            - self.aperture_radius_mm
            <= 0.1
        ):
            raise ValueError(
                "horn and woofer apertures must be separated by more than 0.1 mm"
            )
        from .phase_plug import validate_passages

        validate_passages(self)

    @property
    def baffle(self):
        return FrontBaffle(
            self.width_mm,
            self.height_mm,
            self.depth_mm,
            self.aperture_radius_mm if self.woofer else self.mouth_radius_mm,
            (*(self.woofer_xy_mm if self.woofer else self.horn_xy_mm), self.front_z_mm),
            (self.woofer or self.horn).rim_id,
        )

    @property
    def parts(self):
        return [
            (
                self.horn,
                (*self.horn_xy_mm, self.front_z_mm - self.horn_length_mm),
                "HF",
            ),
        ] + ([(self.woofer, (*self.woofer_xy_mm, self.front_z_mm), "LF")] if self.woofer else [])

    @property
    def patches(self):
        return [
            (patch_id(c, s), c, i, origin, band)
            for c, origin, band in self.parts
            for i, s in enumerate(c.segments)
        ]

    def to_dict(self):
        return {
            "version": 2 if self.horn_wall is not None or self.woofer is None else 1,
            "frame": "aligned-z-mm-v1",
            "horn": self.horn.to_dict(),
            "woofer": self.woofer.to_dict() if self.woofer else None,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name not in ("horn", "woofer", "phase_plugs", "horn_wall")
            },
            **({"horn_wall": self.horn_wall.to_dict()} if self.horn_wall is not None else {}),
            **(
                {"phase_plugs": [p.to_dict() for p in self.phase_plugs]}
                if self.phase_plugs
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        version, frame = value.pop("version", None), value.pop("frame", None)
        if type(version) is not int or version not in (1, 2) or frame != "aligned-z-mm-v1":
            raise ValueError("unsupported native assembly version or frame")
        from .phase_plug import PhasePlug
        if version == 1 and ("horn_wall" in value or value.get("woofer") is None):
            raise ValueError("general horn or horn-only assembly requires version 2")
        if "horn_wall" in value:
            from .general_horn import GeneralHornWall
            value["horn_wall"] = GeneralHornWall.from_dict(value["horn_wall"])

        if "phase_plugs" in value:
            value["phase_plugs"] = tuple(PhasePlug(**p) for p in value["phase_plugs"])
        return cls(
            **{
                **value,
                "horn": SourceContour.from_dict(value["horn"]),
                "woofer": SourceContour.from_dict(value["woofer"]) if value["woofer"] is not None else None,
            }
        )

    @property
    def geometry_sha256(self):
        return digest(self.to_dict())

    def preview(self, **sizes):
        import numpy as np

        result = {
            patch_id(c, s): np.asarray(c.preview(**sizes)[s.id]) + origin
            for c, origin, _ in self.parts
            for s in c.segments
        }
        from .phase_plug import rigid_edges

        angles = np.linspace(0, 2 * math.pi, sizes.get("azimuth_steps", 64) + 1)
        for key, (a, b) in rigid_edges(self).items() if self.phase_plugs else []:
            rz = np.linspace(a, b, sizes.get("radial_steps", 32) + 1)
            result[key] = (
                np.stack(
                    np.broadcast_arrays(
                        rz[:, 0, None] * np.cos(angles),
                        rz[:, 0, None] * np.sin(angles),
                        rz[:, 1, None],
                    ),
                    axis=-1,
                )
                + self.parts[0][1]
            )
        if self.horn_wall is not None:
            rz = self.horn_wall.evaluate(np.linspace(0, 1, sizes.get("radial_steps", 32) + 1))
            result["horn-wall"] = np.stack(np.broadcast_arrays(
                rz[:, 0, None] * np.cos(angles), rz[:, 0, None] * np.sin(angles),
                rz[:, 1, None]), axis=-1) + self.parts[0][1]
        return result

    @classmethod
    def attach(cls, horn, horn_config, *, width_mm, height_mm, depth_mm,
               front_z_mm=0, horn_xy_mm=(0, 0), woofer=None,
               woofer_xy_mm=(0, 0), aperture_radius_mm=0, phase_plugs=()):
        from .general_horn import GeneralHornWall
        wall = GeneralHornWall.from_config(horn_config)
        return cls(horn, woofer, width_mm, height_mm, depth_mm, front_z_mm,
                   horn_xy_mm, wall.length_mm, wall.mouth_radius_mm,
                   woofer_xy_mm, aperture_radius_mm, phase_plugs, wall)

    @property
    def box_surfaces(self):
        return {k: v for k, v in self.baffle.surfaces(self.woofer or self.horn).items()
                if self.woofer is not None or k != "collar"}

    def rigid_areas(self):
        result = {
            role: spec[-1] for role, spec in self.box_surfaces.items()
        }
        if self.woofer is not None:
            result["front"] -= math.pi * self.mouth_radius_mm**2
        if self.horn_wall is not None:
            result["horn-wall"] = self.horn_wall.area_mm2
        from .phase_plug import line_area, rigid_edges

        result.update(
            {key: line_area(a, b) for key, (a, b) in rigid_edges(self).items()}
        )
        return result

    def rigid_distance(self, role, xyz):
        import numpy as np

        from .phase_plug import line_distance, rigid_edges

        edges = rigid_edges(self)
        if role == "horn-wall" and self.horn_wall is not None:
            return self.horn_wall.distance(xyz, self.parts[0][1])
        if role in edges:
            return line_distance(*edges[role], xyz, self.parts[0][1])
        result = self.baffle.distance(self.woofer or self.horn, role, xyz)
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
    if len(drives) != len(assembly.parts) or len({d.channel_id for d in drives}) != len(drives):
        raise ValueError("assembly requires one distinct drive channel per physical source")
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
