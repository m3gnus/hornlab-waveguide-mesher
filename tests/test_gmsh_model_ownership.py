"""Public exports preserve all models and views in a caller-owned session."""

import gmsh
import pytest

from hornlab_mesher import MeshDensity, MesherError, OsseHornGeometry, build_mesh
from hornlab_mesher.cad import write_step


@pytest.mark.parametrize("operation", ["mesh", "step"])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("current", ["", "HornLabMesher", "HornLabCad"])
def test_exports_preserve_caller_models_and_views(
    tmp_path, monkeypatch, operation, fail, current
):
    from hornlab_mesher import cad, mesher

    geometry = OsseHornGeometry(L_mm=40, a_deg=30, n_axial=8, n_phi=16)
    density = MeshDensity(throat_res_mm=8, mouth_res_mm=16)
    standalone_bytes = None
    if operation == "mesh" and not fail:
        standalone = build_mesh(geometry, density, tmp_path / "standalone.msh")
        standalone_bytes = standalone.read_bytes()
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        retained = {}
        for name in ("", "HornLabMesher", "HornLabCad"):
            if name:
                gmsh.model.add(name)
            tag = gmsh.model.occ.addPoint(2, 3, 4)
            gmsh.model.occ.synchronize()
            retained[name] = tag
        view = gmsh.view.add("retained")
        gmsh.view.addListData(view, "SP", 1, [2, 3, 4, 42])
        gmsh.model.setCurrent(current)
        models = list(gmsh.model.list())
        view_data = gmsh.view.getListData(view)
        gmsh.option.setNumber("Geometry.Tolerance", 1e-6)

        if fail:

            def boom(_geometry):
                gmsh.model.occ.addPoint(9, 8, 7)
                raise RuntimeError("injected builder failure")

            monkeypatch.setattr(
                cad if operation == "step" else mesher, "_dispatch_builder", boom
            )

        def export():
            if operation == "mesh":
                return build_mesh(
                    geometry,
                    density,
                    tmp_path / "horn.msh",
                )
            return write_step(geometry, tmp_path / "horn.step")

        if fail:
            with pytest.raises(MesherError, match="injected builder failure"):
                export()
        else:
            export()
            if standalone_bytes is not None:
                assert (tmp_path / "horn.msh").read_bytes() == standalone_bytes
        assert gmsh.isInitialized()
        assert list(gmsh.model.list()) == models
        assert gmsh.model.getCurrent() == current
        assert gmsh.option.getNumber("Geometry.Tolerance") == 1e-6
        assert list(gmsh.view.getTags()) == [view]
        actual_view = gmsh.view.getListData(view)
        assert actual_view[:2] == view_data[:2]
        assert actual_view[2][0].tolist() == view_data[2][0].tolist()
        for name, tag in retained.items():
            gmsh.model.setCurrent(name)
            assert gmsh.model.getEntities() == [(0, tag)]
            assert gmsh.model.getValue(0, tag, []).tolist() == [2, 3, 4]
    finally:
        gmsh.finalize()
