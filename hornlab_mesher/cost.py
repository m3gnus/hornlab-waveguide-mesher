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

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

# Triangle-count constant: a near-equilateral triangle of edge h covers
# ~0.433 h^2, so area A holds ~A / 0.433 h^2 = 2.31 A / h^2 triangles.
TRIANGLES_PER_AREA_OVER_H2 = 2.3
COMPLEX128_BYTES = 16

# Solve-time calibration from the 260612 mesh-sizing study (one frequency, the
# m2-clone quarter model, hornlab-metal-bem native yz+xz symmetry). The matrix
# dimension is the quarter triangle count. The 37665-triangle point measured
# 93 s but at 23 GB is RAM-bound, so the power-law fit uses the two lower,
# compute-bound points and the cubic bound is anchored at the clean mid point.
SOLVE_CALIBRATION_SEC_PER_FREQ = ((8000.0, 1.0), (28178.0, 21.0))
_CUBIC_ANCHOR = (28178.0, 21.0)

# Dense-matrix RAM feasibility bands (gigabytes): measured 8k tris -> 1.1 GB,
# 28k -> 12.7 GB, 38k -> 23 GB.
RAM_CAUTION_GB = 8.0
RAM_WARN_GB = 24.0
RAM_INFEASIBLE_GB = 40.0


def estimate_triangle_count(regions: Iterable[Sequence[float]]) -> int:
    """``N ~= 2.3 * sum(A / h^2)`` over ``(area_mm2, size_mm)`` regions.

    The sum is per region by design: a graded mesh evaluated at a single
    global ``h`` underpredicts by up to ~25 %.
    """
    total = 0.0
    for area_mm2, size_mm in regions:
        area_mm2 = float(area_mm2)
        size_mm = float(size_mm)
        if size_mm > 0.0 and area_mm2 > 0.0:
            total += TRIANGLES_PER_AREA_OVER_H2 * area_mm2 / (size_mm * size_mm)
    return int(round(total))


def matrix_ram_bytes(n_triangles: int) -> int:
    """Dense complex128 BEM matrix RAM in bytes: ``N^2 * 16``."""
    if n_triangles <= 0:
        return 0
    return int(n_triangles) * int(n_triangles) * COMPLEX128_BYTES


def _solve_power_law() -> tuple[float, float]:
    (n0, t0), (n1, t1) = SOLVE_CALIBRATION_SEC_PER_FREQ
    p = math.log(t1 / t0) / math.log(n1 / n0)
    return t0 / (n0**p), p


def solve_seconds_per_freq(n_triangles: int) -> float:
    """Calibrated dense-solve wall time per frequency for ``n_triangles``."""
    if n_triangles <= 0:
        return 0.0
    c, p = _solve_power_law()
    return float(c * (n_triangles**p))


def solve_seconds_cubic_upper(n_triangles: int) -> float:
    """Conservative ``O(N^3)`` per-frequency upper bound (dense LU scaling)."""
    if n_triangles <= 0:
        return 0.0
    n_anchor, t_anchor = _CUBIC_ANCHOR
    return float((t_anchor / (n_anchor**3)) * (n_triangles**3))


def feasibility_from_ram_gb(ram_gb: float) -> str:
    """Severity label for a dense-matrix RAM footprint."""
    if ram_gb >= RAM_INFEASIBLE_GB:
        return "infeasible"
    if ram_gb >= RAM_WARN_GB:
        return "warn"
    if ram_gb >= RAM_CAUTION_GB:
        return "caution"
    return "ok"


@dataclass(frozen=True)
class SolveCostEstimate:
    n_triangles: int
    ram_bytes: int
    ram_gb: float
    solve_seconds_per_freq: float
    solve_seconds_total: float
    solve_seconds_cubic_upper_per_freq: float
    freq_count: int
    feasibility: str

    def to_dict(self) -> dict[str, object]:
        return {
            "n_triangles": int(self.n_triangles),
            "ram_bytes": int(self.ram_bytes),
            "ram_gb": round(float(self.ram_gb), 3),
            "solve_seconds_per_freq": round(float(self.solve_seconds_per_freq), 3),
            "solve_seconds_total": round(float(self.solve_seconds_total), 1),
            "solve_seconds_cubic_upper_per_freq": round(
                float(self.solve_seconds_cubic_upper_per_freq), 3
            ),
            "freq_count": int(self.freq_count),
            "feasibility": self.feasibility,
        }


def estimate_solve_cost(n_triangles: int, *, freq_count: int = 1) -> SolveCostEstimate:
    """Dense-BEM RAM, solve time and feasibility for a triangle count.

    ``n_triangles`` is the matrix dimension: the symmetry-reduced (e.g.
    quarter) mesh the solver assembles, which is what ``BuildResult`` reports.
    """
    n = int(n_triangles)
    ram_bytes = matrix_ram_bytes(n)
    ram_gb = ram_bytes / 1.0e9
    per_freq = solve_seconds_per_freq(n)
    return SolveCostEstimate(
        n_triangles=n,
        ram_bytes=ram_bytes,
        ram_gb=ram_gb,
        solve_seconds_per_freq=per_freq,
        solve_seconds_total=per_freq * max(int(freq_count), 1),
        solve_seconds_cubic_upper_per_freq=solve_seconds_cubic_upper(n),
        freq_count=max(int(freq_count), 1),
        feasibility=feasibility_from_ram_gb(ram_gb),
    )
