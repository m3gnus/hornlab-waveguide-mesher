"""Temporary-face removal preserves surviving OCC topology."""

import gmsh
import pytest

import hornlab_mesher.step_prepare as sp
from test_step_prepare import _gmsh_session


def _cleanup_snapshot(*, batch, failed_shell=False):
    with _gmsh_session():
        occ = gmsh.model.occ
        vertices = [occ.addPoint(*p) for p in
                    ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))]
        lines = [occ.addLine(vertices[a], vertices[b]) for a, b in
                 ((0, 1), (1, 2), (2, 0), (2, 3), (3, 0))]
        first = occ.addPlaneSurface([occ.addCurveLoop(lines[:3])])
        occ.addPlaneSurface([occ.addCurveLoop([-lines[2], *lines[3:]])])
        detached = occ.addRectangle(3, 0, 0, 1, 1)
        box = occ.addBox(5, 0, 0, 1, 1, 1)
        occ.addBox(7, 0, 0, 1, 1, 1)  # A volume that must survive.
        occ.synchronize()
        if batch:
            if failed_shell:
                original = occ.addSurfaceLoop
                occ.addSurfaceLoop = lambda *_a, **_k: (_ for _ in ()).throw(
                    RuntimeError("injected shell failure")
                )
            try:
                sp._remove_temporary_geometry([first, detached], box)
            finally:
                if failed_shell:
                    occ.addSurfaceLoop = original
        else:
            occ.remove([(2, first), (2, detached), (3, box)], recursive=True)
            occ.synchronize()
        entities = gmsh.model.getEntities()
        return [
            (dim, tag, gmsh.model.getType(dim, tag),
             tuple(gmsh.model.getBoundingBox(dim, tag)),
             tuple(gmsh.model.getBoundary([(dim, tag)], oriented=True)) if dim else ())
            for dim, tag in entities
        ]


@pytest.mark.parametrize("failed_shell", [False, True])
def test_batch_cleanup_matches_recursive_removal_with_shared_boundaries(failed_shell):
    assert _cleanup_snapshot(batch=True, failed_shell=failed_shell) == _cleanup_snapshot(batch=False)


def test_batch_cleanup_does_not_leave_a_container_when_removal_fails(monkeypatch):
    with _gmsh_session():
        occ = gmsh.model.occ
        faces = [occ.addRectangle(3 * i, 0, 0, 1, 1) for i in range(2)]
        occ.synchronize()
        original = occ.remove
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("injected removal failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(occ, "remove", fail_once)
        with pytest.raises(RuntimeError, match="injected removal failure"):
            sp._remove_temporary_geometry(faces, None)
        occ.synchronize()
        assert not occ.getEntities(2)
        assert not occ.getEntities(3)


def test_batch_cleanup_recovers_when_occ_silently_keeps_the_container(monkeypatch):
    with _gmsh_session():
        occ = gmsh.model.occ
        faces = [occ.addRectangle(3 * i, 0, 0, 1, 1) for i in range(2)]
        occ.synchronize()
        original = occ.remove
        calls = 0

        def keep_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                return original(*args, **kwargs)

        monkeypatch.setattr(occ, "remove", keep_once)
        sp._remove_temporary_geometry(faces, None)
        assert not occ.getEntities(2)
        assert not occ.getEntities(3)


def test_batch_cleanup_releases_a_volume_created_before_an_error(monkeypatch):
    with _gmsh_session():
        occ = gmsh.model.occ
        faces = [occ.addRectangle(3 * i, 0, 0, 1, 1) for i in range(2)]
        occ.synchronize()
        original = occ.addVolume

        def fail_after_creation(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected volume failure")

        monkeypatch.setattr(occ, "addVolume", fail_after_creation)
        sp._remove_temporary_geometry(faces, None)
        assert not occ.getEntities(2)
        assert not occ.getEntities(3)


def test_failed_removal_that_already_removed_part_still_releases_the_rest(monkeypatch):
    with _gmsh_session():
        occ = gmsh.model.occ
        # A closed set of faces, so the temporary shell volume can be built.
        volume = occ.addBox(0, 0, 0, 1, 1, 1)
        occ.synchronize()
        faces = [tag for _d, tag in gmsh.model.getBoundary([(3, volume)], oriented=False)]
        occ.remove([(3, volume)], recursive=False)
        occ.synchronize()
        original = occ.remove
        calls = 0

        def partial_then_fail(dimtags, recursive=False):
            nonlocal calls
            calls += 1
            if calls == 1:
                # Remove the container and one face, then fail: the request
                # is now partly gone.
                original([dimtags[0]], recursive=False)
                original([(2, faces[0])], recursive=True)
                raise RuntimeError("injected removal failure")
            # A stale entity in the request makes OCC fail, as in practice.
            present = set(occ.getEntities())
            if any(item not in present for item in dimtags):
                raise RuntimeError("stale entity in removal request")
            return original(dimtags, recursive=recursive)

        monkeypatch.setattr(occ, "remove", partial_then_fail)
        with pytest.raises(RuntimeError, match="injected removal failure"):
            sp._remove_temporary_geometry(faces, None)
        monkeypatch.undo()
        assert not occ.getEntities(2)
        assert not occ.getEntities(3)
