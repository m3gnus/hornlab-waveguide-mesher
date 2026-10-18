from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from typing import Any


class PhysicalGroup(IntEnum):
    """Waveguide mesh physical-group contract."""

    RIGID_WALL = 1
    PRIMARY_SOURCE = 2
    ENCLOSURE_WALL = 3
    INTERFACE = 4
    MID_CHAMBER = 8
    PORT_INTERIOR = 9
    MID_PORT_EXIT_LEFT = 10
    MID_PORT_EXIT_RIGHT = 11
    MOUTH_APERTURE = 12


def printable_surface_tags(config: Mapping[str, Any]) -> tuple[int, ...]:
    """Return physical tags retained by a printable STL of a validated config.

    Horns retain the rigid surface only. A standalone SOURCE-DISK also retains
    its moving disc, which is part of the printable body. This selection does
    not describe solver boundaries or the imported-geometry tag namespace.
    An omitted formula uses the ordinary horn default; SOURCE-DISK configs
    require their formula (or type alias) at the root.
    """
    formula = str(config.get("formula", config.get("type", "OSSE"))).strip().upper()
    rigid = int(PhysicalGroup.RIGID_WALL)
    if formula == "SOURCE-DISK":
        return (rigid, int(PhysicalGroup.PRIMARY_SOURCE))
    return (rigid,)


PHYSICAL_NAMES: dict[int, str] = {
    PhysicalGroup.RIGID_WALL: "SD1G0",
    PhysicalGroup.PRIMARY_SOURCE: "SD1D1001",
    PhysicalGroup.ENCLOSURE_WALL: "SD2G0",
    PhysicalGroup.INTERFACE: "I1-2",
    PhysicalGroup.MID_CHAMBER: "mid_chamber",
    PhysicalGroup.PORT_INTERIOR: "mid_port_interior",
    PhysicalGroup.MID_PORT_EXIT_LEFT: "mid_port_exit_left",
    PhysicalGroup.MID_PORT_EXIT_RIGHT: "mid_port_exit_right",
    PhysicalGroup.MOUTH_APERTURE: "mouth_aperture",
}


SOURCE_TAGS = {
    PhysicalGroup.PRIMARY_SOURCE,
    PhysicalGroup.MID_PORT_EXIT_LEFT,
    PhysicalGroup.MID_PORT_EXIT_RIGHT,
}
