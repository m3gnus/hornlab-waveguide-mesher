"""Pre-mesh size and BEM solve-cost prediction for canonical meshes.

This is the canonical home for the triangle-count / dense-BEM-cost math so
every consumer of ``build_from_config`` gets the same forecast. It is the twin of the Fusion STEP pipeline's
``HornLab/scripts/wg_mesh_sizing.py``; the constants and formulas are kept
identical on purpose (the two mesh generators are separate codebases but must
agree). Pure stdlib so any consumer can import it.

* ``N_triangles ~= 2.3 * sum_region(A_region / h_region^2)`` (validated
  constant 2.33 +/- 0.15, ~4% mean error across the 260612 mesh-sizing study).
* Dense complex128 BEM matrix RAM ``N^2 * 16`` bytes; solve time per frequency
  calibrated from the study, with a conservative ``O(N^3)`` upper bound.
"""

from __future__ import annotations

from .mesh_sizing import (
    COMPLEX128_BYTES,
    RAM_CAUTION_GB,
    RAM_INFEASIBLE_GB,
    RAM_WARN_GB,
    SOLVE_CALIBRATION_SEC_PER_FREQ,
    TRIANGLES_PER_AREA_OVER_H2,
    SolveCostEstimate,
    _CUBIC_ANCHOR,
    estimate_solve_cost,
    estimate_triangle_count,
    feasibility_from_ram_gb,
    matrix_ram_bytes,
    solve_seconds_cubic_upper,
    solve_seconds_per_freq,
)

__all__ = [
    "COMPLEX128_BYTES",
    "RAM_CAUTION_GB",
    "RAM_INFEASIBLE_GB",
    "RAM_WARN_GB",
    "SOLVE_CALIBRATION_SEC_PER_FREQ",
    "TRIANGLES_PER_AREA_OVER_H2",
    "SolveCostEstimate",
    "_CUBIC_ANCHOR",
    "estimate_solve_cost",
    "estimate_triangle_count",
    "feasibility_from_ram_gb",
    "matrix_ram_bytes",
    "solve_seconds_cubic_upper",
    "solve_seconds_per_freq",
]
