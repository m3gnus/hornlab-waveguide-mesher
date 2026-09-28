"""Regression tests for the build-pipeline review fixes.

Covers the typed triangle-budget refusal and the message text WG matches on, the
``auto`` surface-fit fallback, the clearance-aware pre-mesh estimate, per-quadrant
enclosure parity between reduced and full builds, quality-report units, and the
gmsh option state a build inherits from a caller-owned session.
"""

from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")
meshio = pytest.importorskip("meshio")

from hornlab_mesher import MesherError, TriangleBudgetExceeded  # noqa: E402
from hornlab_mesher.config_builder import build_from_config  # noqa: E402
from hornlab_mesher.geometry import MeshDensity  # noqa: E402

OSSE = {"L_mm": 120.0, "r0_mm": 12.7, "a_deg": 45.0, "a0_deg": 15.5}
ROSSE = {
    "R_mm": 120.0, "r0_mm": 12.7, "a_deg": 25.0, "a0_deg": 15.5,
    "tmax": 1.0, "m": 0.85, "r": 0.4, "b": 0.2, "k": 2.0, "q": 3.4,
}


def _config(formula="OSSE", mode="freestanding", *, wall=5.0, **mesh):
    base = {
        "angular_segments": 32, "length_segments": 12, "throat_res_mm": 6.0,
        "mouth_res_mm": 20.0, "rear_res_mm": 25.0, "wall_thickness_mm": wall,
        "quadrants": 1234,
    }
    base.update(mesh)
    return {
        "formula": formula, "mode": mode,
        "profile": OSSE if formula == "OSSE" else ROSSE,
        "mesh": base, "source": {"source_shape": 1},
    }


# --------------------------------------------------------------------------
# Typed budget refusal and the text WG matches
# --------------------------------------------------------------------------


def _wg_matches(detail: str) -> bool:
    """The exact test WG's mesh builder applies to a refusal."""

    text = detail.lower()
    return "triangle" in text and (
        "effective limit" in text or "pre-mesh safety margin" in text
    )


def test_post_mesh_budget_refusal_is_typed_and_keeps_its_text(tmp_path):
    config = _config(wall=0.0, max_triangles=1000, mode="bare")
    with pytest.raises(TriangleBudgetExceeded) as caught:
        build_from_config(config, tmp_path / "over.msh")
    assert isinstance(caught.value, MesherError)
    text = str(caught.value)
    assert text.startswith("mesh build failed: generated mesh contains ")
    assert " triangles, exceeding the effective limit 1,000; " in text
    assert (
        "increase the relevant mm resolution, raise max_triangles, or set "
        "allow_large_mesh=true explicitly"
    ) in text
    assert _wg_matches(text)


def test_pre_mesh_budget_refusal_is_typed_and_keeps_its_text(tmp_path):
    config = _config(wall=0.0, max_triangles=20, mode="bare")
    with pytest.raises(TriangleBudgetExceeded) as caught:
        build_from_config(config, tmp_path / "over.msh")
    text = str(caught.value)
    assert "estimated mesh size " in text
    assert " triangles exceeds the effective limit 20 by more than the " in text
    assert "pre-mesh safety margin; the largest estimated contribution is " in text
    assert "Increase that mm resolution, raise max_triangles" in text
    assert _wg_matches(text)


def test_other_build_failures_are_not_budget_refusals(tmp_path, monkeypatch):
    import hornlab_mesher.mesher as mesher

    def boom(_geometry):
        raise RuntimeError("boom")

    monkeypatch.setattr(mesher, "_dispatch_builder", boom)
    with pytest.raises(MesherError) as caught:
        build_from_config(_config(mode="bare", wall=0.0), tmp_path / "bad.msh")
    assert not isinstance(caught.value, TriangleBudgetExceeded)


# --------------------------------------------------------------------------
# ``auto`` surface fit
# --------------------------------------------------------------------------


def test_auto_fit_never_retries_a_budget_refusal(tmp_path, monkeypatch):
    import hornlab_mesher.config_builder as builder

    original = builder.build_mesh_with_info
    fits = []

    def counting(geometry, *args, **kwargs):
        fits.append(geometry.surface_fit)
        return original(geometry, *args, **kwargs)

    monkeypatch.setattr(builder, "build_mesh_with_info", counting)
    config = _config(mode="bare", wall=0.0, max_triangles=1000)
    with pytest.raises(TriangleBudgetExceeded):
        builder.build_from_config(config, tmp_path / "over.msh")
    assert fits == ["interpolate"]


def test_metadata_records_the_fit_that_shipped(tmp_path, monkeypatch):
    import hornlab_mesher.config_builder as builder

    result = builder.build_from_config(
        _config(mode="bare", wall=0.0), tmp_path / "auto.msh"
    )
    assert result.metadata["surfaceFit"] == "interpolate"

    explicit = _config(mode="bare", wall=0.0, surface_fit="approximate")
    result = builder.build_from_config(explicit, tmp_path / "approx.msh")
    assert result.metadata["surfaceFit"] == "approximate"

    original = builder.build_mesh_with_info

    def fail_interpolation(geometry, *args, **kwargs):
        if geometry.surface_fit == "interpolate":
            raise MesherError("injected fit failure")
        return original(geometry, *args, **kwargs)

    monkeypatch.setattr(builder, "build_mesh_with_info", fail_interpolation)
    result = builder.build_from_config(
        _config(mode="bare", wall=0.0), tmp_path / "fallback.msh"
    )
    assert result.metadata["surfaceFit"] == "approximate"


# --------------------------------------------------------------------------
# Clearance-aware estimate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rear,mouth", [(25.0, 26.0), (60.0, 60.0), (100.0, 100.0)])
def test_estimate_sees_the_clearance_caps(tmp_path, rear, mouth):
    config = _config(
        "R-OSSE", wall=3.0, throat_res_mm=4.0, mouth_res_mm=mouth,
        rear_res_mm=rear, max_triangles=100000,
    )
    result = build_from_config(config, tmp_path / "capped.msh")
    metadata = result.metadata
    assert metadata["outerWallClearance"]["capActive"] is True
    estimate = metadata["meshTriangleEstimate"]
    # It read 0.2x-0.8x low before the caps were included.
    assert 0.8 * result.n_triangles <= estimate <= 1.25 * result.n_triangles


def test_budget_advice_admits_the_caps_when_they_are_active(tmp_path):
    config = _config(
        "R-OSSE", wall=3.0, throat_res_mm=4.0, mouth_res_mm=26.0, rear_res_mm=25.0,
        max_triangles=1500,
    )
    with pytest.raises(TriangleBudgetExceeded) as caught:
        build_from_config(config, tmp_path / "over.msh")
    text = str(caught.value)
    assert "effective limit" in text
    assert "wall-clearance guard" in text
    assert "increase the relevant mm resolution" not in text.lower()
    assert "Increase that mm resolution" not in text


# --------------------------------------------------------------------------
# Enclosure per-quadrant parity
# --------------------------------------------------------------------------


def _enclosure_config(quadrants, front):
    config = _config(mode="enclosure", wall=0.0, quadrants=quadrants,
                     enc_front_res_mm=front, enc_back_res_mm=front,
                     max_triangles=100000)
    config["profile"] = OSSE
    config["enclosure"] = {
        "depth_mm": 180.0, "space_l_mm": 30, "space_r_mm": 30,
        "space_t_mm": 30, "space_b_mm": 30, "edge_mm": 10,
    }
    return config


def _first_quadrant_count(path):
    mesh = meshio.read(str(path))
    tri = np.vstack([b.data for b in mesh.cells if b.type == "triangle"])
    centre = mesh.points[tri].mean(axis=1)
    return int(np.count_nonzero((centre[:, 0] > 1e-9) & (centre[:, 1] > 1e-9)))


def test_reduced_enclosure_meshes_its_quadrant_as_the_full_model_does(tmp_path):
    front = "10,30,30,30"
    full = build_from_config(_enclosure_config(1234, front), tmp_path / "full.msh")
    quarter = build_from_config(_enclosure_config(1, front), tmp_path / "q1.msh")
    from_full = _first_quadrant_count(full.mesh_path)
    from_quarter = _first_quadrant_count(quarter.mesh_path)
    # 1,467 against 1,088 before the bilinear bounds were mirror-completed.
    assert abs(from_quarter - from_full) <= 0.03 * from_full


def test_uniform_quadrant_values_leave_reduced_output_alone(tmp_path):
    a = build_from_config(_enclosure_config(1, "30"), tmp_path / "a.msh")
    b = build_from_config(_enclosure_config(1, "30,30,30,30"), tmp_path / "b.msh")
    assert a.n_triangles == b.n_triangles


# --------------------------------------------------------------------------
# quality.py
# --------------------------------------------------------------------------


def test_interior_edges_excludes_non_manifold_edges():
    from hornlab_mesher.quality import _interior_edges

    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
    edges, _left, _right = _interior_edges(faces)
    assert [0, 1] not in edges.tolist()
    two = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int64)
    edges, left, right = _interior_edges(two)
    assert edges.tolist() == [[0, 1]]
    assert sorted((int(left[0]), int(right[0]))) == [0, 1]


def test_quality_gate_messages_are_in_millimetres_for_any_units():
    from hornlab_mesher.quality import evaluate_quality_gate

    def report(units, z, r):
        return {
            "vertex_units": units,
            "element_shape": {
                "p1_angle_deg": 5.0, "sliver_count": 3, "sliver_fraction": 0.1,
                "sliver_angle_deg": 15.0, "worst": [{"z": z, "radius": r}],
            },
        }

    metres = evaluate_quality_gate(report("m", 0.0348, 0.1472)).warnings[0]
    millimetres = evaluate_quality_gate(report("mm", 34.8, 147.2)).warnings[0]
    assert "z=34.8 mm, r=147.2 mm" in metres
    assert "z=34.8 mm, r=147.2 mm" in millimetres


# --------------------------------------------------------------------------
# gmsh option state
# --------------------------------------------------------------------------


def test_a_callers_mesh_options_neither_leak_in_nor_get_clobbered(tmp_path):
    config = _config(mode="bare", wall=0.0)
    counts = []
    for options in ({}, {"Mesh.MeshSizeFactor": 0.5, "Mesh.ElementOrder": 2.0,
                         "Mesh.SaveAll": 1.0, "Mesh.MeshSizeFromCurvature": 20.0}):
        gmsh.initialize(interruptible=False)
        try:
            for name, value in options.items():
                gmsh.option.setNumber(name, value)
            result = build_from_config(config, tmp_path / "s.msh")
            counts.append(result.n_triangles)
            for name, value in options.items():
                assert gmsh.option.getNumber(name) == value
            assert gmsh.option.getNumber("Mesh.MeshSizeMax") > 1.0e6
        finally:
            gmsh.finalize()
    assert counts[0] == counts[1]


def test_direct_api_rear_resolution_matches_the_config_default():
    assert MeshDensity().rear_res_mm == 15.0


# --------------------------------------------------------------------------
# Non-axisymmetric freestanding grids keep their throat weld under every fit
# --------------------------------------------------------------------------


def _nonaxisymmetric(kind):
    config = _config(wall=5.0, throat_res_mm=6.0, mouth_res_mm=20.0, rear_res_mm=25.0)
    config["profile"] = dict(OSSE)
    if kind == "morph":
        config["morph"] = {
            "morph_target": 1, "morph_width_mm": 400, "morph_height_mm": 300,
            "morph_corner_mm": 35,
        }
    else:
        config["cross_section"] = {"aspect_ratio": 1.4}
    return config


@pytest.mark.parametrize("quadrants", [1, 12, 14, 1234])
@pytest.mark.parametrize("fit", ["interpolate", "approximate", "auto"])
@pytest.mark.parametrize("kind", ["morph", "aspect"])
def test_nonaxisymmetric_freestanding_welds_under_every_fit(
    tmp_path, kind, fit, quadrants
):
    config = _nonaxisymmetric(kind)
    config["mesh"]["quadrants"] = quadrants
    config["mesh"]["surface_fit"] = fit
    result = build_from_config(config, tmp_path / "m.msh")
    assert result.n_triangles > 0
    # ``auto`` resolves to interpolate here and must not fall back.
    expected = "approximate" if fit == "approximate" else "interpolate"
    assert result.metadata["surfaceFit"] == expected
