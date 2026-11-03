"""Model ownership shared by the mesh, preview and CAD export pipelines."""

from contextlib import contextmanager
from uuid import uuid4


@contextmanager
def operation_model(gmsh, *, initialized_here: bool, name: str):
    """Isolate an export without clearing a caller's models or views.

    A session opened by the export keeps the historical model setup and is
    finalized by its caller. In an existing session, only our unique model
    is removed, including when construction or export raises.
    """
    if initialized_here:
        gmsh.clear()
        gmsh.model.add(name)
        yield
        return

    previous = gmsh.model.getCurrent()
    owned = f"{name}-{uuid4().hex}"
    gmsh.model.add(owned)
    try:
        yield
    finally:
        if gmsh.isInitialized():
            if owned in gmsh.model.list():
                gmsh.model.setCurrent(owned)
                gmsh.model.remove()
            if previous in gmsh.model.list():
                gmsh.model.setCurrent(previous)
