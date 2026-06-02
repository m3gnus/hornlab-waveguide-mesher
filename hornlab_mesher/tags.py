from __future__ import annotations

from enum import IntEnum


class PhysicalGroup(IntEnum):
    """Shared mesher/solver physical-group contract.

    Tag 1 is always rigid wall. Tags >= 2 are velocity sources, except
    for ``MID_CHAMBER`` and ``PORT_INTERIOR`` which are passive surface
    groups (rigid by default; opt-in carriers for Robin/impedance BCs).
    """

    RIGID_WALL = 1
    PRIMARY_SOURCE = 2
    APERTURE = 3
    HF_THROAT = 4
    MID_DRIVER_LEFT = 6
    MID_DRIVER_RIGHT = 7
    # Interior cavity surfaces between drivers and ports (e.g. BIGMEH
    # mid-port Helmholtz chambers). Rigid by default; carriers for
    # ``SolveConfig.impedance_sources`` Robin BCs when wall damping is
    # required to suppress fictitious interior-Dirichlet resonances.
    MID_CHAMBER = 8
    # Port-tube interior wall, distinct from the port aperture (which is
    # a velocity source). Same default+opt-in semantics as
    # ``MID_CHAMBER``; emitted by BIGMEH as a separate group so port and
    # chamber walls can be damped independently.
    PORT_INTERIOR = 9
    # Horn-side mid-port source caps for LEM/TMM -> BEM coupling. These are
    # not physical driver diaphragms; they represent the port exits after the
    # chamber/tube have been reduced to a lumped or transfer-matrix model.
    MID_PORT_EXIT_LEFT = 10
    MID_PORT_EXIT_RIGHT = 11


PHYSICAL_NAMES: dict[int, str] = {
    PhysicalGroup.RIGID_WALL: "SD1G0",
    PhysicalGroup.PRIMARY_SOURCE: "SD1D1001",
    PhysicalGroup.APERTURE: "SD1D1002",
    PhysicalGroup.HF_THROAT: "SD1D1003",
    PhysicalGroup.MID_DRIVER_LEFT: "SD1D1005",
    PhysicalGroup.MID_DRIVER_RIGHT: "SD1D1006",
    PhysicalGroup.MID_CHAMBER: "mid_chamber",
    PhysicalGroup.PORT_INTERIOR: "mid_port_interior",
    PhysicalGroup.MID_PORT_EXIT_LEFT: "mid_port_exit_left",
    PhysicalGroup.MID_PORT_EXIT_RIGHT: "mid_port_exit_right",
}


SOURCE_TAGS = {
    PhysicalGroup.PRIMARY_SOURCE,
    PhysicalGroup.APERTURE,
    PhysicalGroup.HF_THROAT,
    PhysicalGroup.MID_DRIVER_LEFT,
    PhysicalGroup.MID_DRIVER_RIGHT,
    PhysicalGroup.MID_PORT_EXIT_LEFT,
    PhysicalGroup.MID_PORT_EXIT_RIGHT,
}
