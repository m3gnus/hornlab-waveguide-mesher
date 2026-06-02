from __future__ import annotations

from enum import IntEnum


class PhysicalGroup(IntEnum):
    """Waveguide mesh physical-group contract."""

    RIGID_WALL = 1
    PRIMARY_SOURCE = 2


PHYSICAL_NAMES: dict[int, str] = {
    PhysicalGroup.RIGID_WALL: "SD1G0",
    PhysicalGroup.PRIMARY_SOURCE: "SD1D1001",
}


SOURCE_TAGS = {
    PhysicalGroup.PRIMARY_SOURCE,
}
