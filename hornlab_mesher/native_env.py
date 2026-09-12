"""Keep the Windows native PATH intact across the mesher's own Gmsh session.

On Windows, ``gmsh.initialize()`` (Gmsh 4.15.2) has been measured truncating
the process's native PATH -- 1486 characters down to 316, cut in the middle of
an entry -- while Python's cached ``os.environ`` keeps the old value. The
damage is invisible from Python and only shows through
``GetEnvironmentVariableW``: afterwards, anything that resolves an executable
by bare name or lazily loads a DLL through PATH fails for no visible reason.
The truncation is host-specific; hosted Windows runners have not reproduced it,
so the regression tests inject it instead.

``preserve_native_windows_path`` snapshots the native PATH through kernel32 and
writes it back when the block exits. Off Windows it is a no-op that touches no
environment at all.

The mesher opens a Gmsh session in exactly two places, each only when no
session is already active: ``mesher.build_mesh_with_info`` and
``cad.write_step``. Both open and close that session inside this guard. A caller
that opens its own session before calling the mesher owns that session and must
guard it itself; the mesher then never initializes. Waveguide Generator's
counterpart is ``_preserve_native_windows_path`` in
``server/mesh/gmsh_worker.py``, which guards the application's worker-thread
session with the same semantics. Change the two together.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.GetEnvironmentVariableW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        wintypes.DWORD,
    )
    _kernel32.GetEnvironmentVariableW.restype = wintypes.DWORD
    _kernel32.SetEnvironmentVariableW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
    _kernel32.SetEnvironmentVariableW.restype = wintypes.BOOL


logger = logging.getLogger(__name__)

_ERROR_ENVVAR_NOT_FOUND = 203


def _read_native_windows_path() -> str | None:
    """Read PATH from Win32, whose value can diverge from ``os.environ``."""

    ctypes.set_last_error(0)
    size = _kernel32.GetEnvironmentVariableW("PATH", None, 0)
    if size == 0:
        error = ctypes.get_last_error()
        if error in (0, _ERROR_ENVVAR_NOT_FOUND):
            return None
        raise ctypes.WinError(error)

    while True:
        buffer = ctypes.create_unicode_buffer(size)
        ctypes.set_last_error(0)
        length = _kernel32.GetEnvironmentVariableW("PATH", buffer, size)
        if length == 0:
            error = ctypes.get_last_error()
            if error in (0, _ERROR_ENVVAR_NOT_FOUND):
                return None
            raise ctypes.WinError(error)
        if length < size:
            return buffer.value
        size = length


@contextmanager
def preserve_native_windows_path() -> Iterator[None]:
    """Undo native PATH mutations made by a Windows Gmsh API call."""

    if sys.platform != "win32":
        yield
        return

    path = _read_native_windows_path()
    try:
        yield
    finally:
        if not _kernel32.SetEnvironmentVariableW("PATH", path):
            failure = ctypes.WinError(ctypes.get_last_error())
            # Raising from here would replace whatever the body was already
            # raising, and the body's exception is the one that explains a
            # failed mesh or export. Report a damaged PATH on its own only
            # when the body succeeded and there is nothing to mask.
            if sys.exc_info()[1] is None:
                raise failure
            logger.error("Could not restore the native PATH after a Gmsh call: %s", failure)


__all__ = ["preserve_native_windows_path"]
