"""Bounded passive annular vanes with explicit passage and shell topology."""

import math
from dataclasses import asdict, dataclass
from urllib.parse import quote

FEATURE = "native-phase-plug-passages-v1"


@dataclass(frozen=True)
class PhasePlug:
    """A closed linearly tapered central body or annular vane, in horn mm."""

    id: str
    z0_mm: float
    z1_mm: float
    inner0_mm: float
    outer0_mm: float
    inner1_mm: float
    outer1_mm: float

    def __post_init__(self):
        if (
            not isinstance(self.id, str)
            or not self.id.strip()
            or self.id != self.id.strip()
            or len(self.id) > 128
        ):
            raise ValueError("passive body ID must be a nonempty trimmed string")
        for name in self.__dataclass_fields__:
            if name == "id":
                continue
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("passive dimensions must be finite numbers")
            object.__setattr__(self, name, float(value) if value else 0.0)
        if self.z1_mm - self.z0_mm <= 0.1:
            raise ValueError("passive body length must exceed 0.1 mm")
        if min(self.inner0_mm, self.inner1_mm) < 0:
            raise ValueError("passive inner radii cannot be negative")
        if (self.inner0_mm == 0) != (self.inner1_mm == 0):
            raise ValueError("central and annular topology cannot change along a body")
        if min(self.outer0_mm - self.inner0_mm, self.outer1_mm - self.inner1_mm) <= 0.1:
            raise ValueError("passive thickness must exceed 0.1 mm")
        if self.inner0_mm and min(self.inner0_mm, self.inner1_mm) <= 0.1:
            raise ValueError("annular inner passage radius must exceed 0.1 mm")

    def to_dict(self):
        return asdict(self)

    @property
    def edges(self):
        vertices = [
            (self.inner0_mm, self.z0_mm),
            (self.outer0_mm, self.z0_mm),
            (self.outer1_mm, self.z1_mm),
            (self.inner1_mm, self.z1_mm),
        ]
        return {
            "plug/" + quote(self.id, safe="") + "/" + name: (a, b)
            for name, a, b in zip(
                ("inlet", "outer", "outlet", "inner"),
                vertices,
                vertices[1:] + vertices[:1],
            )
            if a[0] or b[0]
        }

    def radius(self, z, *, inner=False):
        a, b = (
            (self.inner0_mm, self.inner1_mm)
            if inner
            else (self.outer0_mm, self.outer1_mm)
        )
        return a + (b - a) * (z - self.z0_mm) / (self.z1_mm - self.z0_mm)


def line_distance(a, b, xyz, origin):
    """Exact point distance to a finite revolved meridian line."""
    import numpy as np

    local = np.asarray(xyz) - np.asarray(origin)
    r, z = np.linalg.norm(local[..., :2], axis=-1), local[..., 2]
    dr, dz = b[0] - a[0], b[1] - a[1]
    t = np.clip(((r - a[0]) * dr + (z - a[1]) * dz) / (dr * dr + dz * dz), 0, 1)
    return np.hypot(r - a[0] - t * dr, z - a[1] - t * dz)


def line_area(a, b):
    return math.pi * (a[0] + b[0]) * math.dist(a, b)


def validate_passages(assembly):
    """Conservative analytical separation of nested, coextensive finite bodies.

    Radial gap divided by maximum meridian slope bounds Euclidean separation.
    This bound includes all end faces; it is not a mesh or sample estimate.
    """
    plugs = assembly.phase_plugs
    if (
        not isinstance(plugs, tuple)
        or len(plugs) > 8
        or any(not isinstance(p, PhasePlug) for p in plugs)
    ):
        raise ValueError(
            "phase plugs must be a tuple of at most eight canonical bodies"
        )
    if len({p.id for p in plugs}) != len(plugs):
        raise ValueError("passive body IDs must be distinct")
    if not plugs:
        return {}
    z0, z1 = plugs[0].z0_mm, plugs[0].z1_mm
    if any((p.z0_mm, p.z1_mm) != (z0, z1) for p in plugs):
        raise ValueError("passage vanes must share inlet and outlet planes")
    source = max(0, assembly.horn.axial_bounds_mm[1])
    clearances = {"source": z0 - source, "mouth": assembly.horn_length_mm - z1}
    if min(clearances.values()) <= 0.1:
        raise ValueError(
            "passive bodies must clear source and mouth by more than 0.1 mm"
        )
    throat = assembly.horn.points[-1].r_mm
    slope = (assembly.mouth_radius_mm - throat) / assembly.horn_length_mm
    previous = None
    for p in plugs:
        if previous is not None:
            slopes = [
                (previous.outer1_mm - previous.outer0_mm) / (z1 - z0),
                (p.inner1_mm - p.inner0_mm) / (z1 - z0),
            ]
            gap = min(
                p.inner0_mm - previous.outer0_mm, p.inner1_mm - previous.outer1_mm
            )
            clearances["between/" + p.id] = gap / math.sqrt(
                1 + max(abs(s) for s in slopes) ** 2
            )
        elif p.inner0_mm:
            clearances["central-passage"] = min(p.inner0_mm, p.inner1_mm)
        previous = p
    outer = plugs[-1]
    gap = min(
        throat + slope * z0 - outer.outer0_mm, throat + slope * z1 - outer.outer1_mm
    )
    # Both slopes count: a tapered outer body can reach an adjacent horn point.
    outer_slope = (outer.outer1_mm - outer.outer0_mm) / (z1 - z0)
    clearances["horn-wall"] = gap / math.sqrt(
        1 + max(abs(slope), abs(outer_slope)) ** 2
    )
    if min(clearances.values()) <= 0.1:
        raise ValueError("every phase-plug passage must clear by more than 0.1 mm")
    return clearances


def wall_edges(assembly):
    r = assembly.horn.points[-1].r_mm
    slope = (assembly.mouth_radius_mm - r) / assembly.horn_length_mm
    if not assembly.phase_plugs:
        return {
            "horn-wall": ((r, 0), (assembly.mouth_radius_mm, assembly.horn_length_mm))
        }
    z0, z1 = assembly.phase_plugs[0].z0_mm, assembly.phase_plugs[0].z1_mm
    zs = [0, z0, z1, assembly.horn_length_mm]
    return {
        "horn-wall/" + role: ((r + slope * a, a), (r + slope * b, b))
        for role, a, b in zip(("inlet", "passage", "outlet"), zs, zs[1:])
    }


def rigid_edges(assembly):
    return {
        **wall_edges(assembly),
        **{key: edge for p in assembly.phase_plugs for key, edge in p.edges.items()},
    }


def surface_targets(assembly, mesh_size_mm, refinement=1):
    """Local refinement of facing walls; leave remote enclosure faces coarse."""
    gap = min(validate_passages(assembly).values()) if assembly.phase_plugs else None
    tolerance = min(0.15, gap / 10) if gap is not None else 0.15
    targets = {
        role: min(
            mesh_size_mm,
            math.sqrt(8 * (tolerance / 4 if gap else 0.1) * min(a[0], b[0])),
        )
        for role, (a, b) in wall_edges(assembly).items()
    }
    if not assembly.phase_plugs:
        return targets
    if type(refinement) is not int or refinement not in (1, 2, 4):
        raise ValueError("passage refinement must be 1, 2 or 4")
    if "collar" in assembly.rigid_areas():
        targets["collar"] = min(
            mesh_size_mm, math.sqrt(8 * tolerance / 4 * assembly.woofer.points[-1].r_mm)
        )
    targets["horn-wall/passage"] = min(
        targets["horn-wall/passage"],
        validate_passages(assembly)["horn-wall"] / 3,
        math.sqrt(8 * tolerance / 4 * assembly.horn.points[-1].r_mm),
    )
    for p in assembly.phase_plugs:
        thickness = min(p.outer0_mm - p.inner0_mm, p.outer1_mm - p.inner1_mm)
        for role, (a, b) in p.edges.items():
            positive = [v for v in (a[0], b[0]) if v > 0]
            targets[role] = min(
                mesh_size_mm,
                gap / 3,
                thickness / 2,
                math.sqrt(8 * tolerance / 4 * min(positive)),
            )
    return {
        role: target / refinement
        if role.startswith("plug/") or role == "horn-wall/passage"
        else target
        for role, target in targets.items()
    }


def passage_contract(assembly):
    clearances = validate_passages(assembly)
    if not clearances:
        return None
    plugs = assembly.phase_plugs
    z0, z1 = plugs[0].z0_mm, plugs[0].z1_mm
    throat = assembly.horn.points[-1].r_mm
    slope = (assembly.mouth_radius_mm - throat) / assembly.horn_length_mm
    passages = []
    previous = None
    for plug in plugs:
        if plug.inner0_mm:
            inner = (previous.outer0_mm, previous.outer1_mm) if previous else (0, 0)
            passages.append(
                {
                    "id": "passage/"
                    + (quote(previous.id, safe="") if previous else "axis")
                    + "/"
                    + quote(plug.id, safe=""),
                    "inlet_radii_mm": [inner[0], plug.inner0_mm],
                    "outlet_radii_mm": [inner[1], plug.inner1_mm],
                }
            )
        previous = plug
    passages.append(
        {
            "id": "passage/" + quote(plugs[-1].id, safe="") + "/horn-wall",
            "inlet_radii_mm": [plugs[-1].outer0_mm, throat + slope * z0],
            "outlet_radii_mm": [plugs[-1].outer1_mm, throat + slope * z1],
        }
    )
    return {
        "version": 1,
        "clearances_mm": clearances,
        "surface_tolerance_mm": min(0.15, min(clearances.values()) / 10),
        "shell_count": 1 + len(assembly.phase_plugs),
        "open_passage_count": len(passages),
        "passages": passages,
        "bodies": [
            {"id": p.id, "genus": 1 if p.inner0_mm else 0, "rigid_roles": list(p.edges)}
            for p in assembly.phase_plugs
        ],
    }
