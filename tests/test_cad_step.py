from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from hornlab_mesher import MesherError
from hornlab_mesher.cad import write_step
from hornlab_mesher.geometry import HornEnclosure, PointGridHornGeometry


def _inner_grid(
    *,
    n_phi: int = 40,
    n_length: int = 14,
    length: float = 140.0,
    r0: float = 12.7,
    r1: float = 100.0,
) -> np.ndarray:
    points = np.empty((n_phi, n_length + 1, 3), dtype=np.float64)
    for i in range(n_phi):
        phi = math.tau * i / n_phi
        for j in range(n_length + 1):
            t = j / n_length
            radius = r0 + (r1 - r0) * t
            points[i, j] = (
                radius * math.cos(phi),
                radius * math.sin(phi),
                length * t,
            )
    return points


def _outer_grid(inner_points: np.ndarray, *, wall_thickness: float = 6.0) -> np.ndarray:
    outer = np.array(inner_points, dtype=np.float64, copy=True)
    radial = np.linalg.norm(outer[:, :, :2], axis=2)
    scale = (radial + float(wall_thickness)) / np.maximum(radial, 1.0e-12)
    outer[:, :, 0] *= scale
    outer[:, :, 1] *= scale
    return outer


def _freestanding(**overrides) -> PointGridHornGeometry:
    inner = _inner_grid()
    return PointGridHornGeometry(
        inner_points=inner,
        outer_points=_outer_grid(inner),
        wall_thickness_mm=6.0,
        **overrides,
    )


def _reimported_solids(path: Path) -> list[tuple[int, float]]:
    """Re-read a written STEP and return its (tag, volume) solids."""

    import gmsh

    initialized_here = False
    try:
        if not gmsh.isInitialized():
            gmsh.initialize(interruptible=False)
            initialized_here = True
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.clear()
        gmsh.model.add("reimport")
        gmsh.model.occ.importShapes(str(path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        return [
            (int(tag), float(gmsh.model.occ.getMass(3, int(tag))))
            for dim, tag in gmsh.model.getEntities(3)
        ]
    finally:
        if initialized_here and gmsh.isInitialized():
            gmsh.finalize()


def _points_inside(path: Path, probes: list[tuple[float, float, float]]) -> list[bool]:
    """Which probe points fall inside the STEP's solid?"""

    import gmsh

    initialized_here = False
    try:
        if not gmsh.isInitialized():
            gmsh.initialize(interruptible=False)
            initialized_here = True
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.clear()
        gmsh.model.add("probe")
        gmsh.model.occ.importShapes(str(path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        volumes = [int(tag) for _, tag in gmsh.model.getEntities(3)]
        assert volumes, "STEP contains no solid to probe"
        return [
            any(
                bool(gmsh.model.isInside(3, tag, list(probe)))
                for tag in volumes
            )
            for probe in probes
        ]
    finally:
        if initialized_here and gmsh.isInitialized():
            gmsh.finalize()


def test_freestanding_exports_a_closed_solid_in_millimetres(tmp_path):
    path, info = write_step(_freestanding(), tmp_path / "horn.step")

    assert info.body == "solid"
    assert info.units == "mm"
    assert info.volume_mm3 is not None and info.volume_mm3 > 0.0
    text = path.read_text(encoding="utf-8", errors="replace")
    assert "MANIFOLD_SOLID_BREP" in text
    assert "SI_UNIT(.MILLI.,.METRE.)" in text
    # Smooth patches, not a tessellation and not degree-1 ruled strips.
    assert "B_SPLINE_SURFACE" in text


def test_step_reimports_as_exactly_one_solid(tmp_path):
    path, info = write_step(_freestanding(), tmp_path / "horn.step")

    solids = _reimported_solids(path)
    assert len(solids) == 1
    _, volume = solids[0]
    assert volume == pytest.approx(info.volume_mm3, rel=1.0e-9)


def test_step_carries_no_loose_surfaces_beside_the_solid(tmp_path):
    """Sewing leaves the source faces behind; they must not reach the file.

    A Fusion import of an unpruned model shows the solid with every face that
    built it stacked on top as separate surface bodies.
    """

    path, _ = write_step(_freestanding(), tmp_path / "horn.step")

    text = path.read_text(encoding="utf-8", errors="replace")
    assert "SHELL_BASED_SURFACE_MODEL" not in text
    assert text.count("MANIFOLD_SOLID_BREP") == 1


def test_throat_is_an_open_bore_not_a_plug(tmp_path):
    """The driver membrane is not material; the exported bore runs through.

    The sewn acoustic boundary encloses the wall *and* a plug of material
    between the rear face and the source cap. Exporting that as-is hands CAD a
    horn with a blocked throat.
    """

    inner = _inner_grid()
    rear_z = float(np.mean(inner[:, 0, 2])) - 6.0
    on_axis_behind_throat = (0.0, 0.0, 0.5 * rear_z)

    opened, _ = write_step(_freestanding(), tmp_path / "open.step")
    plugged, _ = write_step(
        _freestanding(), tmp_path / "plugged.step", open_throat=False
    )

    assert _points_inside(opened, [on_axis_behind_throat]) == [False]
    assert _points_inside(plugged, [on_axis_behind_throat]) == [True]


def test_open_bore_keeps_the_surrounding_wall(tmp_path):
    """Opening the throat must remove the plug and nothing else."""

    path, _ = write_step(_freestanding(), tmp_path / "horn.step")

    inside = _points_inside(
        path,
        [
            (0.0, 0.0, -3.0),  # bore axis behind the throat: open
            (15.7, 0.0, -3.0),  # wall material beside it: solid
            (0.0, 0.0, 70.0),  # bore at mid-horn: open
            (58.0, 0.0, 70.0),  # wall at mid-horn: solid
        ],
    )
    assert inside == [False, True, False, True]


def test_solid_is_independent_of_the_acoustic_source_cap_shape(tmp_path):
    """A flat disc and a domed cap model the same driver and the same part.

    The cut follows the cap's own face, so whichever membrane the solve used
    leaves the same material behind.
    """

    flat, flat_info = write_step(
        _freestanding(source_shape=0), tmp_path / "flat.step"
    )
    domed, domed_info = write_step(
        _freestanding(source_shape=1, source_curv=1), tmp_path / "domed.step"
    )

    assert flat_info.volume_mm3 == pytest.approx(domed_info.volume_mm3, rel=1.0e-9)
    assert flat_info.n_faces == domed_info.n_faces


def test_enclosure_exports_a_solid_with_the_horn_bored_into_it(tmp_path):
    geometry = PointGridHornGeometry(
        inner_points=_inner_grid(),
        enclosure=HornEnclosure(depth_mm=200.0, edge_mm=18.0),
    )

    path, info = write_step(geometry, tmp_path / "enclosure.step")

    assert info.body == "solid"
    assert info.volume_mm3 is not None and info.volume_mm3 > 0.0
    assert len(_reimported_solids(path)) == 1
    # The bore is open through the front; the box body around it is material.
    assert _points_inside(path, [(0.0, 0.0, 70.0), (120.0, 0.0, 70.0)]) == [
        False,
        True,
    ]


def test_bare_horn_exports_a_surface_without_the_driver_membrane(tmp_path):
    """No wall thickness means no material, so there is no solid to make."""

    path, info = write_step(
        PointGridHornGeometry(inner_points=_inner_grid()), tmp_path / "bare.step"
    )

    assert info.body == "surface"
    assert info.volume_mm3 is None
    assert info.n_faces == 1
    text = path.read_text(encoding="utf-8", errors="replace")
    assert "MANIFOLD_SOLID_BREP" not in text
    assert "ADVANCED_FACE" in text
    # A rounded membrane domes millimetres past the throat plane; the exported
    # bore stops there, give or take the wall patch's own spline overshoot.
    assert info.bounding_box_mm[0][2] == pytest.approx(0.0, abs=0.1)


def test_symmetry_reduced_geometry_is_refused_with_the_fix(tmp_path):
    inner = _inner_grid(n_phi=11)
    geometry = PointGridHornGeometry(
        inner_points=inner,
        outer_points=_outer_grid(inner),
        wall_thickness_mm=6.0,
        closed=False,
        symmetry_planes=("x", "y"),
    )

    with pytest.raises(MesherError, match="quadrants=1234"):
        write_step(geometry, tmp_path / "quarter.step")


def test_vertical_offset_places_the_exported_part(tmp_path):
    """Mesh.VerticalOffset positions the mesh; CAD has to agree with it."""

    _, centred = write_step(_freestanding(), tmp_path / "centred.step")
    _, offset = write_step(
        _freestanding(vertical_offset_mm=30.0), tmp_path / "offset.step"
    )

    assert offset.bounding_box_mm[0][1] == pytest.approx(
        centred.bounding_box_mm[0][1] + 30.0, abs=1.0e-6
    )
    assert offset.bounding_box_mm[0][0] == pytest.approx(
        centred.bounding_box_mm[0][0], abs=1.0e-6
    )
    assert offset.volume_mm3 == pytest.approx(centred.volume_mm3, rel=1.0e-9)


def test_export_keeps_an_enclosure_fillet_the_mesh_would_suppress(tmp_path):
    """CAD gets the geometry as designed, not the acoustic level of detail.

    ``mesher._acoustic_geometry`` squares off a fillet smaller than the mesh can
    resolve. That is right for a solve and wrong for a part, so the CAD path
    never runs it -- a 2 mm fillet still rounds the box here.
    """

    inner = _inner_grid()
    _, filleted = write_step(
        PointGridHornGeometry(
            inner_points=inner,
            enclosure=HornEnclosure(depth_mm=200.0, edge_mm=2.0, edge_type=1),
        ),
        tmp_path / "filleted.step",
    )
    _, sharp = write_step(
        PointGridHornGeometry(
            inner_points=inner,
            enclosure=HornEnclosure(depth_mm=200.0, edge_mm=0.0, edge_type=1),
        ),
        tmp_path / "sharp.step",
    )

    # Rounding the box edges takes material off the corners.
    assert filleted.volume_mm3 is not None and sharp.volume_mm3 is not None
    assert filleted.volume_mm3 < sharp.volume_mm3


def _osse_config(**mesh_overrides) -> dict:
    return {
        "formula": "OSSE",
        "mode": "freestanding",
        "profile": {
            "L_mm": 120.0,
            "r0_mm": 12.7,
            "a_deg": 60.0,
            "a0_deg": 15.5,
            "k": 1.0,
            "n": 4.0,
            "q": 0.995,
            "s": 0.0,
        },
        "mesh": {
            "angular_segments": 32,
            "length_segments": 16,
            "wall_thickness_mm": 6.0,
            "throat_res_mm": 8.0,
            "mouth_res_mm": 26.0,
            "rear_res_mm": 25.0,
            **mesh_overrides,
        },
    }


def test_config_export_builds_the_geometry_a_solve_would_use(tmp_path):
    """Going through the config must not re-derive the shape differently."""

    from hornlab_mesher.cad import write_step_from_config
    from hornlab_mesher.config_builder import resolve_geometry

    config = _osse_config()
    _, from_config = write_step_from_config(config, tmp_path / "config.step")
    _, direct = write_step(
        resolve_geometry(config).geometry, tmp_path / "direct.step"
    )

    assert from_config.body == "solid"
    assert from_config.volume_mm3 == pytest.approx(direct.volume_mm3, rel=1.0e-12)


def test_config_export_closes_a_symmetry_reduced_design(tmp_path):
    """A quarter-model solve still exports a whole part."""

    from hornlab_mesher.cad import write_step_from_config

    quarter = _osse_config(quadrants="1")
    full = _osse_config(quadrants="1234")

    _, from_quarter = write_step_from_config(quarter, tmp_path / "quarter.step")
    _, from_full = write_step_from_config(full, tmp_path / "full.step")

    assert from_quarter.body == "solid"
    assert from_quarter.volume_mm3 == pytest.approx(
        from_full.volume_mm3, rel=1.0e-9
    )


def test_output_path_must_name_a_step_file(tmp_path):
    with pytest.raises(MesherError, match=r"\.step or \.stp"):
        write_step(_freestanding(), tmp_path / "horn.iges")


def test_a_rejected_export_does_not_overwrite_the_target(tmp_path):
    """Staging means a failure leaves the previous good file in place."""

    target = tmp_path / "horn.step"
    write_step(_freestanding(), target)
    good = target.read_bytes()

    monkey = _freestanding()
    object.__setattr__(monkey, "inner_points", monkey.inner_points[:, :1, :])
    with pytest.raises(MesherError):
        write_step(monkey, target)

    assert target.read_bytes() == good
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".horn.step")]


def test_failed_export_leaves_no_temporary_file_behind(tmp_path):
    inner = _inner_grid(n_phi=11)
    geometry = PointGridHornGeometry(
        inner_points=inner,
        outer_points=_outer_grid(inner),
        wall_thickness_mm=6.0,
        closed=False,
        symmetry_planes=("x", "y"),
    )

    before = set(Path(tmp_path).iterdir())
    with pytest.raises(MesherError):
        write_step(geometry)
    assert set(Path(tmp_path).iterdir()) == before
