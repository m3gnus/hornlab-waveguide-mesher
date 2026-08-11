from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

from hornlab_mesher import mesh_sizing


MODULE_PATH = Path(mesh_sizing.__file__).resolve()


def test_module_imports_bare_without_site_packages_or_numpy():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(MODULE_PATH.parent)!r}); "
                "import mesh_sizing; "
                "print(mesh_sizing.__file__)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == MODULE_PATH


def test_module_level_imports_are_stdlib_only():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_roots.add((node.module or "").split(".", 1)[0])

    assert imported_roots == {"__future__", "dataclasses", "math", "typing"}


def test_triangle_count_accepts_regions_and_bare_pairs_equally():
    pairs = [(50_000.0, 3.0), (120_000.0, 6.0), (9_000.0, 1.5)]
    regions = [mesh_sizing.Region(area, size) for area, size in pairs]

    assert mesh_sizing.estimate_triangle_count(pairs) == 29_644
    assert mesh_sizing.estimate_triangle_count(regions) == 29_644


@pytest.mark.parametrize(
    ("n_triangles", "per_freq", "cubic_upper"),
    [
        (8_000, 1.0000, 0.4806),
        (28_178, 21.0000, 21.0000),
        (120_000, 697.9207, 1621.9316),
    ],
)
def test_calibrated_solve_cost_numbers_are_unchanged(
    n_triangles: int,
    per_freq: float,
    cubic_upper: float,
):
    assert round(mesh_sizing.solve_seconds_per_freq(n_triangles), 4) == per_freq
    assert round(mesh_sizing.solve_seconds_cubic_upper(n_triangles), 4) == cubic_upper


def test_cost_module_keeps_its_previous_public_surface():
    from hornlab_mesher import cost

    previous_names = {
        "TRIANGLES_PER_AREA_OVER_H2",
        "COMPLEX128_BYTES",
        "SOLVE_CALIBRATION_SEC_PER_FREQ",
        "_CUBIC_ANCHOR",
        "RAM_CAUTION_GB",
        "RAM_WARN_GB",
        "RAM_INFEASIBLE_GB",
        "SolveCostEstimate",
        "estimate_solve_cost",
        "estimate_triangle_count",
        "matrix_ram_bytes",
        "solve_seconds_per_freq",
        "solve_seconds_cubic_upper",
        "feasibility_from_ram_gb",
    }

    assert all(hasattr(cost, name) for name in previous_names)
