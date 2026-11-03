"""Exports and preview preserve all models and views in a caller-owned session."""

import gmsh
import pytest

from hornlab_mesher import MeshDensity, MesherError, OsseHornGeometry, build_mesh
from hornlab_mesher.cad import write_step


@pytest.mark.parametrize("operation", ["mesh", "step", "preview"])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("current", ["", "HornLabMesher", "HornLabCad", "HornLabPreview"])
def test_exports_preserve_caller_models_and_views(
    tmp_path, monkeypatch, operation, fail, current
):
    from hornlab_mesher import cad, mesher
    from hornlab_mesher.preview.ath_import import _model

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
        for name in ("", "HornLabMesher", "HornLabCad", "HornLabPreview"):
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
        preview_options = {
            **mesher._PINNED_MESH_OPTIONS,
            "General.Terminal": 0,
            "General.NumThreads": 3,
            "Mesh.MeshSizeMin": 0.25,
            "Mesh.MeshSizeMax": 37,
            "Mesh.MeshSizeFromPoints": 1,
            "Mesh.MeshSizeFromCurvature": 5,
            "Mesh.MeshSizeExtendFromBoundary": 1,
            "Mesh.Algorithm": 8,
        }
        if operation == "preview":
            for name, value in preview_options.items():
                gmsh.option.setNumber(name, value)

        if fail and operation != "preview":

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
            if operation == "step":
                return write_step(geometry, tmp_path / "horn.step")
            with _model() as preview_gmsh:
                assert preview_gmsh is gmsh
                gmsh.model.occ.addPoint(9, 8, 7)
                if fail:
                    raise RuntimeError("injected builder failure")

        if fail:
            error = RuntimeError if operation == "preview" else MesherError
            with pytest.raises(error, match="injected builder failure"):
                export()
        else:
            export()
            if standalone_bytes is not None:
                assert (tmp_path / "horn.msh").read_bytes() == standalone_bytes
        assert gmsh.isInitialized()
        assert list(gmsh.model.list()) == models
        assert gmsh.model.getCurrent() == current
        assert gmsh.option.getNumber("Geometry.Tolerance") == 1e-6
        if operation == "preview":
            for name, value in preview_options.items():
                assert gmsh.option.getNumber(name) == value
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


@pytest.mark.parametrize("removed", ["preview", "caller", "session"])
def test_preview_cleanup_preserves_body_error_after_model_removal(removed):
    from hornlab_mesher.preview.ath_import import _model

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("caller")
        original_size = gmsh.option.getNumber("Mesh.MeshSizeMax")
        with pytest.raises(RuntimeError, match="injected preview failure"), _model():
            if removed == "session":
                gmsh.finalize()
            else:
                if removed == "caller":
                    gmsh.model.setCurrent("caller")
                gmsh.model.remove()
            raise RuntimeError("injected preview failure")
        if removed == "session":
            assert not gmsh.isInitialized()
        else:
            assert gmsh.model.list() == ([""] if removed == "caller" else ["", "caller"])
            assert gmsh.model.getCurrent() == ("" if removed == "caller" else "caller")
            assert gmsh.option.getNumber("Mesh.MeshSizeMax") == original_size
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
