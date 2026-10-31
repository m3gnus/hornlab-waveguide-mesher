# ATH V2026-09a import blocker witnesses

These exact inputs and full-ring profile exports were captured by running the
existing ATH V2026-09a executable natively on Windows, in a separate writable
reference directory. The original installation and captures were preserved.
No Wine, replacement version, Gmsh or solver was used to generate these grids.

Executable banner: `Build number: V2026-09a`.
Executable SHA-256:
`4795b08e9b03f83489ec390cbba8abe0fa900aff2d87862f8fca5f014d003d8d`.
Per-case invocation: `ath.exe <case>.cfg`; raw stdout/stderr and all emitted
files are retained privately. `manifest.json` records each committed input and
CSV hash. CSVs retain ATH's six-decimal x/y/z millimetres and profile blocks;
only CRLF-to-LF and the final blank-line normalization were applied.
The complete licensed archive, executables, machine paths and raw process
logs are excluded from public commits.

`r-base`, `r-slot` and `r-ext` use R=150, r0=12.7, a0=10, a=50, k=1,
r=0.4, b=0.2, m=0.85, q=3.4; 32 angular and 20 body segments. Slot=12
translates the complete body. Extension=10, angle=5 adds ten pre-body
segments. Further cases cover slot+extension, truncation, a negative slot,
negative extension angle, and 4/15 mm extensions whose junction arithmetic
can round below zero. `r-ext-run21` independently covers the larger 600 mm
R-OSSE and its 12 mm extension.

The morph matrix covers OSSE with a(p)=50-15*sin(p)^2 and R-OSSE, target
width/height 240/360 (x smaller), 360/240 (y smaller), 240/220 (both smaller),
and 420/400 (neither smaller), with AllowShrinkage=0 and 1. Budgets are
32/20, 48/10 and 64/20 angular/body segments. The same fixed-part/rate and
20 mm corners isolate the boundary and angular-sampling rule.

Comparison methods:

- Full grids: identical profile/row counts and x/y/z at existing 2e-4 mm
  absolute tolerance, zero relative tolerance; throat directions separately
  witness the rebuilt angular construction.
- Physical slot: recover each exported point's continuous body parameter
  independently from its radius and compare axial position on both sides of
  the rollback; separately check scalar/vector evaluation away from rows.
- Refinement: preserve authored dimensions/direction correspondence and
  compare the same physical stations after adding intermediate azimuths.
- Consumers: compare preview mouth geometry across LODs and physical closest
  points on reimported STEP surfaces, independently of vertex numbering.

The no-shrink rule and consumer handoff are documented in
[text import geometry](../../../../../docs/text-import-geometry.md).
These small profile witnesses do not replace the private full archive gate.
