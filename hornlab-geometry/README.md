# @hornlab/geometry

Bundled JavaScript geometry evaluator for `hornlab-waveguide-mesher`.

This package is scoped to OSSE and R-OSSE waveguide profile evaluation, point
grid generation, and hidden ATH parity sampling used by the Python mesher.
The public CLI rejects non-OSSE profile families.

## Consumers

- **hornlab-mesher**: spawns the geometry CLI subprocess to evaluate OSSE or
  R-OSSE point grids before lofting Gmsh surfaces.
- **Applications and tools**: can use the CLI/API to generate OSSE or R-OSSE
  point grids before meshing or solving.
