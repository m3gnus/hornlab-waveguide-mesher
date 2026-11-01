# Tessellation report

`scripts/report_tessellation.py` measures the fine preview and the generated solve
mesh for mesher configurations. Pass JSON, TOML, or ATH text files; a JSON record
containing a `config` object is accepted too. For example, from a development
checkout with the mesher dependencies installed:

```sh
python scripts/report_tessellation.py examples/osse-freestanding.toml \
  tests/fixtures/tessellation_report/IC01-default.json --output report.json
```

The command processes every named input, writes strict JSON, and returns nonzero
if any build failed. `--preview-only` omits solve meshes. `--no-meridian` skips
reference-surface sampling. A saved-design corpus can be passed as a list of
translated mesher configuration files; native examples and ATH config files use
the same input path. Representative inputs are in
`tests/fixtures/tessellation_report/`. Cases carry their input-list index so
multiple files named `config.txt` remain distinct.
If a preview succeeds but the solve is refused, the case is marked `partial`:
the preview measurements and the solve error are both retained, and the command
still returns a failing status.

Each preview role and each solve physical group reports triangle count, element
shape (including the fraction with minimum angle below 15 degrees), and the
existing dihedral-based chord estimate. Preview fidelity metadata records the
requested and achieved chord and normal targets, including incomplete or
vertex-limited measurements. Build times are wall-clock observations, not
performance guarantees.

For axisymmetric point-grid horns, a separately sampled canonical meridian
provides centroid-to-surface distances and shape by ten equal arc-length
regions. Its 4,097 stations are an approximation to the analytic surface; the
result is not an exact maximum over the whole triangle. For other point-grid
horns the report compares centroids against a 256-by-256 canonical sampled
surface. The nearest-facet search examines 16 candidates, so that value is an
approximation, not a certified closest distance; region boundaries are ten
equal z bands rather than arc-length deciles. Native bodies outside the point
sampler may use the inspection preview as their reference and explicitly say so.
These approximate references cannot certify a missing preview fidelity bound.

To identify the solve acoustic wall when physical groups also include the
baffle or shell, the report selects triangles whose three nodes are within
0.05 mm of the sampled wall. It records both selected and unselected triangle
counts. A selected triangle need not represent every facet of a high-error or
poorly fitted wall, and adjoining faces can pass the proximity test, so inspect
that selection before treating the sampled deviations as a wall-wide result.
All inputs are left unchanged.

## Baseline data

`docs/data/tessellation-baseline.json` records report runs against mesher base
`746c6b1ca6b8782d018d5b86b0f7c5e5cadca2bd`: 34 translated saved designs,
23 ATH archive examples, two native fixtures, and three repository TOML
examples. Two saved designs were excluded because the existing solver refuses
them: `FF21-cfg-mm-points`
(outward wall curvature guard) and `IC02-coverage` (invalid coverage angle).
Each row identifies its input by corpus and case ID and contains the inner
preview, preview fidelity, total preview triangle count, and the selected solve
wall and its selection counts. The triangle-centroid deviation and regional
shape figures use the approximations described above. A missing fidelity field
is recorded as null; it is not a passing certificate. Times are single-run wall
clock observations on the measuring machine and should not be used as a
benchmark without repeated, controlled measurements.
