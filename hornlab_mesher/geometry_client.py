"""Python client for the @hornlab/geometry NDJSON CLI.

Spawns a long-lived Node subprocess running
``hornlab-geometry/bin/geometry-cli.js`` and exchanges NDJSON requests
over stdin/stdout. The CLI hosts the canonical JS geometry evaluators
(``calculateOSSE``, ``calculateROSSE``) so Python callers
share the same single source of truth as the WG browser UI.

The ``hornlab-geometry`` sibling package must be co-located at the repository
root (standalone layout) or at ``<HornLab>/hornlab-geometry/`` (monorepo
layout).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import atexit
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HORNLAB_ROOT = Path(__file__).resolve().parents[2]
_GEOMETRY_ROOT = (
    _REPO_ROOT / "hornlab-geometry"
    if (_REPO_ROOT / "hornlab-geometry").is_dir()
    else _HORNLAB_ROOT / "hornlab-geometry"
)
_CLI_PATH = _GEOMETRY_ROOT / "bin" / "geometry-cli.js"


class GeometryClientError(RuntimeError):
    """Raised when the CLI returns an error response or the subprocess dies."""


class GeometryClient:
    """Long-lived NDJSON REPL client over the geometry CLI subprocess.

    Thread-safe via an internal lock. One subprocess per client instance.
    """

    def __init__(self, cli_path: Path | str | None = None, node_bin: str = "node") -> None:
        self._cli_path = Path(cli_path) if cli_path else _CLI_PATH
        if not self._cli_path.is_file():
            raise FileNotFoundError(
                f"Geometry CLI not found at {self._cli_path}. "
                "The hornlab-geometry package must be bundled at the repository root "
                "or co-located at the HornLab workspace root."
            )
        self._node_bin = node_bin
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._next_id = 0
        atexit.register(self.close)

    def _ensure_proc(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        env = os.environ.copy()
        self._proc = subprocess.Popen(
            [self._node_bin, str(self._cli_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(_GEOMETRY_ROOT),
            text=True,
            env=env,
            bufsize=1,
        )
        return self._proc

    def _call(self, op: str, params: dict[str, Any]) -> Any:
        with self._lock:
            proc = self._ensure_proc()
            self._next_id += 1
            req_id = str(self._next_id)
            request = json.dumps({"id": req_id, "op": op, "params": params})
            assert proc.stdin is not None and proc.stdout is not None
            try:
                proc.stdin.write(request + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as err:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise GeometryClientError(
                    f"Geometry CLI subprocess died: {err}; stderr: {stderr}"
                ) from err
            line = proc.stdout.readline()
            if not line:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise GeometryClientError(
                    f"Geometry CLI returned no response; stderr: {stderr}"
                )
            response = json.loads(line)
            if response.get("id") != req_id:
                raise GeometryClientError(
                    f"Geometry CLI response id mismatch: expected {req_id}, got {response.get('id')}"
                )
            if "error" in response:
                raise GeometryClientError(f"{op}: {response['error']}")
            return response["result"]

    def close(self) -> None:
        with self._lock:
            if self._proc is None:
                return
            proc = self._proc
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass
            finally:
                for stream in (proc.stdout, proc.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass
                self._proc = None

    def __enter__(self) -> "GeometryClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- High-level operations ----

    def health(self) -> dict[str, Any]:
        return self._call("health", {})

    def compute_osse_profile(
        self,
        t_values: Sequence[float] | np.ndarray,
        phi: float,
        **params: Any,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        t_list = list(np.asarray(t_values, dtype=float).tolist())
        result = self._call(
            "compute_osse_profile",
            {"phi": float(phi), "t_values": t_list, "params": params},
        )
        x = np.asarray(result["x"], dtype=float)
        y = np.asarray(result["y"], dtype=float)
        return x, y, float(result["total_length"])

    def compute_rosse_profile(
        self,
        t_values: Sequence[float] | np.ndarray,
        phi: float,
        **params: Any,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        t_list = list(np.asarray(t_values, dtype=float).tolist())
        result = self._call(
            "compute_rosse_profile",
            {"phi": float(phi), "t_values": t_list, "params": params},
        )
        x = np.asarray(result["x"], dtype=float)
        y = np.asarray(result["y"], dtype=float)
        return x, y, float(result["total_length"])

    def build_inner_points(self, params: dict[str, Any]) -> dict[str, Any]:
        """Build the 3D `inner_points` grid for a full WG params dict.

        ``params`` must include ``type`` (``OSSE``/``R-OSSE``/``ROSSE``) and
        all WG geometry knobs the chosen profile expects, plus ``angularSegments``
        and ``lengthSegments``. Returns a dict with ``inner_points`` (flat list
        of length ``grid_n_phi * (grid_n_length + 1) * 3``), ``grid_n_phi``,
        ``grid_n_length``, ``full_circle``, ``angle_list``, and optional
        ``slice_map``.
        """
        return self._call("build_inner_points", {"params": params})

    def build_point_grid(self, params: dict[str, Any]) -> dict[str, Any]:
        """Build WG point-grid fields for a full geometry params dict.

        Returns ``inner_points`` and, for freestanding thickened horns
        (``encDepth <= 0`` and ``wallThickness > 0``), ``outer_points``. When
        an enclosure is requested the JS pass deliberately returns only the
        horn grid; the Python mesher owns the enclosure construction.
        """
        return self._call("build_point_grid", {"params": params})


_default_client: GeometryClient | None = None


def get_default_client() -> GeometryClient:
    """Return the module-level singleton client, creating it lazily."""
    global _default_client
    if _default_client is None:
        _default_client = GeometryClient()
    return _default_client
