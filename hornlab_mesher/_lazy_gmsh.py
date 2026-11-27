"""One lazy import of gmsh, shared by the modules that drive OCC."""

from __future__ import annotations

from typing import Any


class LazyGmsh:
    """Import gmsh on first use, never at module import.

    Every gmsh import in this package is lazy by contract: importing
    hornlab_mesher must stay numpy-only. Eagerly importing gmsh here broke two
    things at once when ``step_prepare`` joined the package's eager exports — grid
    resolution stopped being importable without gmsh, and gmsh's C++ runtime
    installs its own signal handlers on load, which on Linux rearms SIGPIPE's
    default action and turned a closed-socket write in a WG v2 test into a
    process kill (pytest exit 141 on ubuntu CI). The first attribute access
    imports the real module and replaces this shim in the module globals, so
    the indirection costs one lookup, once.

    Each module creates its own, with its globals: ``gmsh = LazyGmsh(globals())``.
    """

    def __init__(self, namespace: dict[str, Any]) -> None:
        self._namespace = namespace

    def __getattr__(self, name: str) -> Any:
        import gmsh as _gmsh_module

        self._namespace["gmsh"] = _gmsh_module
        return getattr(_gmsh_module, name)
