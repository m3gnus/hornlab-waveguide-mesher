"""Canonical full-circle prescribed source meridians, in finished millimetres.

No curve fitting: one segment is one physical patch, including deliberate corners.
Excitation and tessellation are separate from the immutable geometry recipe.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

FEATURE = "native-source-contour-v1"


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def _id(value):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("identities must be nonempty trimmed strings")


@dataclass(frozen=True)
class ContourPoint:
    id: str
    r_mm: float
    z_mm: float

    def __post_init__(self):
        _id(self.id)
        for key in ("r_mm", "z_mm"):
            value = float(getattr(self, key))
            object.__setattr__(self, key, 0.0 if value == 0 else value)
        if not all(math.isfinite(x) for x in (self.r_mm, self.z_mm)) or self.r_mm < 0:
            raise ValueError("point coordinates must be finite with nonnegative radius")


@dataclass(frozen=True)
class ContourSegment:
    id: str
    start: str
    end: str
    role: str = "moving"
    kind: str = "line"
    center_mm: tuple[float, float] | None = None
    direction: str = "ccw"

    def __post_init__(self):
        for value in (self.id, self.start, self.end):
            _id(value)
        if self.role not in {"moving", "rigid"} or self.kind not in {"line", "arc"}:
            raise ValueError("invalid patch role or segment kind")
        if self.direction not in {"cw", "ccw"}:
            raise ValueError("arc direction must be cw or ccw")
        if self.kind == "line" and self.center_mm is not None:
            raise ValueError("lines cannot carry an arc center")
        if self.kind == "line":
            object.__setattr__(self, "direction", "ccw")
        if self.kind == "arc":
            if self.center_mm is None or len(self.center_mm) != 2:
                raise ValueError("arcs require a radial/axial center")
            object.__setattr__(
                self,
                "center_mm",
                tuple(0.0 if float(x) == 0 else float(x) for x in self.center_mm),
            )
            if not all(math.isfinite(x) for x in self.center_mm):
                raise ValueError("arc center must be finite")


@dataclass(frozen=True)
class SourceContour:
    physical_source_id: str
    rim_id: str
    points: tuple[ContourPoint, ...]
    segments: tuple[ContourSegment, ...]

    def __post_init__(self):
        _id(self.physical_source_id)
        _id(self.rim_id)
        object.__setattr__(self, "points", tuple(self.points))
        object.__setattr__(self, "segments", tuple(self.segments))
        if len(self.points) != len(self.segments) + 1 or not self.segments:
            raise ValueError("contour must be one complete ordered meridian")
        if len(self.segments) > 256:
            raise ValueError("source contour exceeds the 256-patch bound")
        if len({p.id for p in self.points}) != len(self.points) or len(
            {s.id for s in self.segments}
        ) != len(self.segments):
            raise ValueError("point and patch IDs must be unique")
        if self.points[0].r_mm != 0 or self.points[-1].z_mm != 0:
            raise ValueError("contour requires a pole and an exact z=0 rim attachment")
        if not any(s.role == "moving" for s in self.segments):
            raise ValueError("a source requires at least one moving patch")
        for i, segment in enumerate(self.segments):
            a, b = self.points[i : i + 2]
            if (segment.start, segment.end) != (a.id, b.id) or b.r_mm <= a.r_mm:
                raise ValueError(
                    "segments must join in strictly increasing radius (single-valued z(r))"
                )
            if segment.kind == "arc":
                radius, start, sweep = self.arc(i)
                # dr/dt must stay positive throughout the open arc. Check every
                # stationary radius analytically, rather than using a sampled vote.
                end = start + sweep
                low, high = sorted((start, end))
                critical = [
                    k * math.pi
                    for k in range(-4, 5)
                    if low + 1e-12 < k * math.pi < high - 1e-12
                ]
                if (
                    abs(sweep) > math.pi + 1e-12
                    or critical
                    or -radius * math.sin(start + sweep / 2) * sweep <= 0
                ):
                    raise ValueError("arc folds in radius or exceeds a semicircle")

    def arc(self, index):
        segment = self.segments[index]
        a, b = self.points[index : index + 2]
        cr, cz = segment.center_mm
        ra, rb = (
            math.hypot(a.r_mm - cr, a.z_mm - cz),
            math.hypot(b.r_mm - cr, b.z_mm - cz),
        )
        if ra <= 1e-9 or abs(ra - rb) > 1e-9 * max(1, ra):
            raise ValueError("arc endpoints must lie on the same nonzero circle")
        start = math.atan2(a.z_mm - cz, a.r_mm - cr)
        end = math.atan2(b.z_mm - cz, b.r_mm - cr)
        sweep = (end - start) % (2 * math.pi)
        if segment.direction == "cw":
            sweep -= 2 * math.pi
        return ra, start, sweep

    def evaluate(self, index, t):
        if not 0 <= t <= 1:
            raise ValueError("segment parameter must lie in [0,1]")
        a, b = self.points[index : index + 2]
        if t == 0:
            return a.r_mm, a.z_mm
        if t == 1:
            return b.r_mm, b.z_mm
        segment = self.segments[index]
        if segment.kind == "line":
            return a.r_mm + t * (b.r_mm - a.r_mm), a.z_mm + t * (b.z_mm - a.z_mm)
        radius, start, sweep = self.arc(index)
        cr, cz = segment.center_mm
        return cr + radius * math.cos(start + t * sweep), cz + radius * math.sin(
            start + t * sweep
        )

    def to_dict(self):
        return {"version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        if (
            set(value)
            != {"version", "physical_source_id", "rim_id", "points", "segments"}
            or value["version"] != 1
        ):
            raise ValueError("unsupported source contour fields or version")
        return cls(
            value["physical_source_id"],
            value["rim_id"],
            tuple(ContourPoint(**p) for p in value["points"]),
            tuple(ContourSegment(**s) for s in value["segments"]),
        )

    @property
    def geometry_sha256(self):
        return digest(self.to_dict())

    @property
    def axial_bounds_mm(self):
        values = [p.z_mm for p in self.points]
        for i, s in enumerate(self.segments):
            if s.kind == "arc":
                radius, start, sweep = self.arc(i)
                low, high = sorted((start, start + sweep))
                values.extend(
                    s.center_mm[1] + radius * math.sin(k * math.pi / 2)
                    for k in range(-8, 9)
                    if low <= k * math.pi / 2 <= high
                )
        return min(values), max(values)

    def preview(self, radial_steps=32, azimuth_steps=64):
        if not 2 <= radial_steps <= 2048 or not 8 <= azimuth_steps <= 2048:
            raise ValueError("preview sampling exceeds bounds")
        return {
            s.id: [
                [r * math.cos(phi), r * math.sin(phi), z]
                for j in range(radial_steps + 1)
                for r, z in [self.evaluate(i, j / radial_steps)]
                for phi in [
                    2 * math.pi * k / azimuth_steps for k in range(azimuth_steps)
                ]
            ]
            for i, s in enumerate(self.segments)
        }


@dataclass(frozen=True)
class ContourDrive:
    channel_id: str
    weights: tuple[tuple[str, float], ...]
    motion: str = "normal"

    def __post_init__(self):
        _id(self.channel_id)
        object.__setattr__(
            self, "weights", tuple((k, float(v)) for k, v in self.weights)
        )
        if self.motion not in {"normal", "axial"}:
            raise ValueError("motion must be normal or axial")
        if len(dict(self.weights)) != len(self.weights) or not self.weights:
            raise ValueError("weights must have unique patch IDs")
        for key, value in self.weights:
            _id(key)
            if not math.isfinite(value):
                raise ValueError("patch weights must be finite")

    def validate(self, contour):
        if set(dict(self.weights)) != {
            s.id for s in contour.segments if s.role == "moving"
        }:
            raise ValueError(
                "drive must cover exactly the moving patches; rigid roles are independent of weight"
            )

    @property
    def excitation_sha256(self):
        return digest(
            {
                "channel_id": self.channel_id,
                "weights": dict(self.weights),
                "motion": self.motion,
            }
        )


def dome(
    radius_mm,
    height_mm,
    *,
    surround_width_mm=0,
    surround_depth_mm=0,
    land_width_mm=0,
    physical_source_id="diaphragm",
    rim_id="horn.throat",
):
    """Expand independently dimensioned spherical dome/surround/land patches."""
    if not all(
        math.isfinite(x)
        for x in (
            radius_mm,
            height_mm,
            surround_width_mm,
            surround_depth_mm,
            land_width_mm,
        )
    ):
        raise ValueError("preset dimensions must be finite")
    if (
        radius_mm <= 0
        or not 0 < height_mm <= radius_mm
        or min(surround_width_mm, land_width_mm) < 0
    ):
        raise ValueError("invalid dome dimensions")
    center_z = (height_mm**2 - radius_mm**2) / (2 * height_mm)
    points = [
        ContourPoint("pole", 0, height_mm),
        ContourPoint("dome.rim", radius_mm, 0),
    ]
    segments = [
        ContourSegment(
            "dome",
            "pole",
            "dome.rim",
            kind="arc",
            center_mm=(0, center_z),
            direction="cw",
        )
    ]
    _finish(points, segments, surround_width_mm, surround_depth_mm, land_width_mm)
    return SourceContour(physical_source_id, rim_id, points, segments)


def cone(
    radius_mm,
    depth_mm,
    *,
    cap_radius_mm,
    cap_height_mm,
    surround_width_mm=0,
    surround_depth_mm=0,
    land_width_mm=0,
    physical_source_id="diaphragm",
    rim_id="horn.throat",
):
    """Expand a spherical dust cap, straight cone, surround and rigid land."""
    if (
        not all(
            math.isfinite(x)
            for x in (radius_mm, depth_mm, cap_radius_mm, cap_height_mm)
        )
        or not 0 < cap_radius_mm < radius_mm
        or depth_mm < 0
    ):
        raise ValueError("invalid independent cone dimensions")
    cap = dome(cap_radius_mm, cap_height_mm)
    points = [
        ContourPoint("pole", 0, cap_height_mm - depth_mm),
        ContourPoint("cap.rim", cap_radius_mm, -depth_mm),
        ContourPoint("cone.rim", radius_mm, 0),
    ]
    segments = [
        ContourSegment(
            "dust-cap",
            "pole",
            "cap.rim",
            kind="arc",
            center_mm=(0, cap.segments[0].center_mm[1] - depth_mm),
            direction="cw",
        ),
        ContourSegment("cone", "cap.rim", "cone.rim"),
    ]
    _finish(points, segments, surround_width_mm, surround_depth_mm, land_width_mm)
    return SourceContour(physical_source_id, rim_id, points, segments)


def _finish(points, segments, width, depth, land):
    if (
        not all(math.isfinite(x) for x in (width, depth, land))
        or min(width, land) < 0
        or (width == 0 and depth != 0)
        or abs(depth) > width / 2
    ):
        raise ValueError("invalid surround or land dimensions")
    if width:
        a = points[-1]
        points.append(ContourPoint("surround.rim", a.r_mm + width, 0))
        kwargs = (
            {}
            if depth == 0
            else {
                "kind": "arc",
                "center_mm": (
                    a.r_mm + width / 2,
                    (depth**2 - (width / 2) ** 2) / (2 * depth),
                ),
                "direction": "cw" if depth > 0 else "ccw",
            }
        )
        segments.append(ContourSegment("surround", a.id, points[-1].id, **kwargs))
    if land:
        a = points[-1]
        points.append(ContourPoint("rim", a.r_mm + land, 0))
        segments.append(ContourSegment("land", a.id, "rim", role="rigid"))


def flat(radius_mm, *, physical_source_id="diaphragm", rim_id="horn.throat"):
    return SourceContour(
        physical_source_id,
        rim_id,
        (ContourPoint("pole", 0, 0), ContourPoint("rim", radius_mm, 0)),
        (ContourSegment("piston", "pole", "rim"),),
    )
