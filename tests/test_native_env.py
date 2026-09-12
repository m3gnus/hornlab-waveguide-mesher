"""The native-PATH guard around the mesher's own Gmsh session.

On Windows, ``gmsh.initialize()`` has been measured truncating the process's
native PATH (1486 characters down to 316, cut mid-entry) while ``os.environ``
kept the old value, so the damage is invisible from Python and only shows
through ``GetEnvironmentVariableW``. Anything afterwards that resolves an
executable or lazily loads a DLL through PATH then fails for no visible reason.

The mesher opens a session in exactly two places, each only when no session is
active: ``build_mesh_with_info`` and ``write_step``. Both must open and close
that session inside ``native_env.preserve_native_windows_path``.

The wiring tests run on every platform with a fake ``gmsh`` module, so they do
not depend on whether some other test left a real session open. The
Windows-only tests inject the truncation through ``SetEnvironmentVariableW``;
hosted Windows runners do not reproduce the real truncation, so these tests
measure the guard against injected damage, not against Gmsh itself.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest

import hornlab_mesher.cad as cad_module
import hornlab_mesher.mesher as mesher_module
from hornlab_mesher import MesherError
from hornlab_mesher.geometry import OsseHornGeometry


windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="the guard is a no-op off Windows"
)


class _StopAfterOpen(Exception):
    """Ends a build right after the session opens; no real geometry is needed."""


class _FakeGmsh:
    """Just enough of ``gmsh`` to reach the session open and close.

    The first option call after ``initialize`` raises, which sends both entry
    points down their failure path and into the ``finally`` that closes the
    session they opened.
    """

    def __init__(
        self,
        depth: Callable[[], int],
        *,
        active: bool = False,
        on_session_call: Callable[[], None] = lambda: None,
    ) -> None:
        self._depth = depth
        self._active = active
        self._on_session_call = on_session_call
        self.calls: list[tuple[str, int]] = []
        self.option = types.SimpleNamespace(setNumber=self._stop)

    def isInitialized(self) -> bool:
        return self._active

    def initialize(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("initialize", self._depth()))
        self._active = True
        self._on_session_call()

    def finalize(self) -> None:
        self.calls.append(("finalize", self._depth()))
        self._active = False
        self._on_session_call()

    @staticmethod
    def _stop(*args: object) -> None:
        raise _StopAfterOpen("stop after the session opened")


class _GuardRecorder:
    """Stands in for the guard and reports how deeply a call is nested in it."""

    def __init__(self) -> None:
        self.depth = 0
        self.entries = 0

    @contextmanager
    def guard(self) -> Iterator[None]:
        self.depth += 1
        self.entries += 1
        try:
            yield
        finally:
            self.depth -= 1


def _build_mesh(tmp_path) -> None:
    mesher_module.build_mesh_with_info(
        OsseHornGeometry(), output_path=tmp_path / "out.msh"
    )


def _write_step(tmp_path) -> None:
    cad_module.write_step(OsseHornGeometry(), tmp_path / "out.step")


ENTRY_POINTS = [
    pytest.param(_build_mesh, id="build_mesh_with_info"),
    pytest.param(_write_step, id="write_step"),
]


def _install_recorder(monkeypatch: pytest.MonkeyPatch) -> _GuardRecorder:
    recorder = _GuardRecorder()
    # raising=False so that, on code without the guard, the test still runs
    # and fails on the assertion that explains what is missing.
    for module in (mesher_module, cad_module):
        monkeypatch.setattr(
            module, "preserve_native_windows_path", recorder.guard, raising=False
        )
    return recorder


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_the_session_this_call_opens_is_opened_and_closed_inside_the_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path, entry
) -> None:
    recorder = _install_recorder(monkeypatch)
    fake = _FakeGmsh(lambda: recorder.depth)
    monkeypatch.setitem(sys.modules, "gmsh", fake)

    with pytest.raises(MesherError):
        entry(tmp_path)

    assert fake.calls == [("initialize", 1), ("finalize", 1)]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_session_someone_else_opened_is_left_to_its_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path, entry
) -> None:
    """A caller that already holds a session owns it, and owns its guard."""

    recorder = _install_recorder(monkeypatch)
    fake = _FakeGmsh(lambda: recorder.depth, active=True)
    monkeypatch.setitem(sys.modules, "gmsh", fake)

    with pytest.raises(MesherError):
        entry(tmp_path)

    assert fake.calls == []
    assert recorder.entries == 0


@pytest.mark.skipif(sys.platform == "win32", reason="checks the non-Windows half")
def test_the_guard_is_a_no_op_off_windows() -> None:
    from hornlab_mesher import native_env

    assert not hasattr(native_env, "_kernel32")
    with pytest.raises(RuntimeError, match="geometry failed"):
        with native_env.preserve_native_windows_path():
            raise RuntimeError("geometry failed")


# --- Windows: injected truncation -------------------------------------------


@contextmanager
def _native_path_restored_afterwards() -> Iterator[str]:
    """Snapshot the native PATH and put it back even if a test fails."""

    from hornlab_mesher import native_env

    before = native_env._read_native_windows_path()
    assert before is not None
    try:
        yield before
    finally:
        native_env._kernel32.SetEnvironmentVariableW("PATH", before)


def _truncate_native_path() -> None:
    from hornlab_mesher import native_env

    native_env._kernel32.SetEnvironmentVariableW(
        "PATH", "C:\\Program Files (x86)\\Common F"
    )


@windows_only
@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_injected_truncation_by_the_session_calls_does_not_outlive_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path, entry
) -> None:
    """Both the open and the close truncate PATH; neither survives the call."""

    from hornlab_mesher import native_env

    fake = _FakeGmsh(lambda: 0, on_session_call=_truncate_native_path)
    monkeypatch.setitem(sys.modules, "gmsh", fake)

    with _native_path_restored_afterwards() as before:
        with pytest.raises(MesherError):
            entry(tmp_path)
        assert [name for name, _ in fake.calls] == ["initialize", "finalize"]
        assert native_env._read_native_windows_path() == before


def _fail_the_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make only the PATH write fail, leaving the read side real.

    The real handle is captured before the patch: delegating through the
    module attribute would resolve back to the double and recurse.
    """

    from hornlab_mesher import native_env

    real = native_env._kernel32

    class _FailingRestore:
        def __getattr__(self, name: str) -> object:
            return getattr(real, name)

        @staticmethod
        def SetEnvironmentVariableW(name: str, value: str | None) -> int:
            return 0

    monkeypatch.setattr(native_env, "_kernel32", _FailingRestore())


@windows_only
def test_restores_a_path_the_body_truncated() -> None:
    from hornlab_mesher import native_env

    with _native_path_restored_afterwards() as before:
        with native_env.preserve_native_windows_path():
            _truncate_native_path()
            assert (
                native_env._read_native_windows_path()
                == "C:\\Program Files (x86)\\Common F"
            )
        assert native_env._read_native_windows_path() == before


@windows_only
def test_a_failed_restore_does_not_mask_the_body_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The body's exception is the one that explains a failed mesh or export."""

    from hornlab_mesher import native_env

    _fail_the_restore(monkeypatch)

    with pytest.raises(RuntimeError, match="geometry failed"):
        with native_env.preserve_native_windows_path():
            raise RuntimeError("geometry failed")


@windows_only
def test_a_failed_restore_still_raises_when_nothing_is_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hornlab_mesher import native_env

    _fail_the_restore(monkeypatch)

    with pytest.raises(OSError):
        with native_env.preserve_native_windows_path():
            pass
