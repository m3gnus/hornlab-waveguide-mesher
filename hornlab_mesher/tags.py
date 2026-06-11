from __future__ import annotations

from enum import IntEnum


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


PHYSICAL_NAMES: dict[int, str] = {
    PhysicalGroup.RIGID_WALL: "SD1G0",
    PhysicalGroup.PRIMARY_SOURCE: "SD1D1001",
    PhysicalGroup.ENCLOSURE_WALL: "SD2G0",
    PhysicalGroup.INTERFACE: "I1-2",
    PhysicalGroup.MID_CHAMBER: "mid_chamber",
    PhysicalGroup.PORT_INTERIOR: "mid_port_interior",
    PhysicalGroup.MID_PORT_EXIT_LEFT: "mid_port_exit_left",
    PhysicalGroup.MID_PORT_EXIT_RIGHT: "mid_port_exit_right",
}


SOURCE_TAGS = {
    PhysicalGroup.PRIMARY_SOURCE,
    PhysicalGroup.MID_PORT_EXIT_LEFT,
    PhysicalGroup.MID_PORT_EXIT_RIGHT,
}
