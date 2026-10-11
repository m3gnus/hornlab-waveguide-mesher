# ATH import fixtures, ATH build V2026-08c

Profile grids exported by ATH (`GridExport`, profiles only, `x;y;z` in mm, one
blank-line-separated block per profile) and the configs that produced them.

| Pair | What it establishes |
| --- | --- |
| `osse-block-with-top-level-length`, `osse-block-without-top-level-length` | ATH ignores a top-level `Length` beside an `OSSE` block: the block's `L = 160` sets the horn length, and the two exports are byte-identical. |
| `sf1`, `sf2` | `sf(m,n1,n2,n3,a,b,p)` is the Gielis superformula, including odd `m` and non-unit `a`, `b`. |
| `sf3` | `log` is natural logarithm, `round(2.5)` is 3, and `exp`/`floor` compose with azimuth-dependent expressions. |
| `f0_plain`, `f0_plain_nac` | CORE OSSE with p-dependent `a`, `n`, `s`; no morph/stretch. Without NAC, 100 angular plus 4 corner segments gives 104; with NAC, 100. |
| `f0_morph`, `f0_morph_nac` | Rectangular morph, `Rate=3`, `FixedPart=0`: blending starts at the first positive axial station, with zero blend there. |
| `f0_stretch`, `f0_stretch_nac` | `s1=1.14`, `s2=0.025` transform z after sampling; x/y match the plain exports byte for byte. |
| `f0_both`, `f0_both_nac` | Combined morph/stretch. `f0_both_nac` is CORE at its original mesh settings with Roundover/RearShape omitted. Morph progress is measured before stretch. |

The `.csv` files are ATH's output with line endings normalised to LF and the trailing blank line removed; they are
otherwise unedited.

All `f0_*` configurations are:
Derived from ATH CORE (C) 2026 Marcel Batík, CC-BY-4.0, at-horns.eu/core.html
The same attribution is included as a comment at the top of each cfg.

`GridExport` exports all four quadrants despite `Mesh.Quadrants=1`; the
tests compare both the full grid and the first quadrant including its two
cardinal endpoints. Every imported probe grid is compared at 2e-4 mm.

The CORE inner grid depends on two ATH sampling details. Plain, morphed and stretched exports
recover the same pre-stretch default axial map: Bézier controls `(0.5,0.1)`
and `(0.5,0.92)`, selecting the first 0.001 parameter step whose abscissa
reaches the requested station. The prior dense approximation with second
ordinate 0.95 missed CORE z by 3.374063 mm before stretch and 3.589088 mm
after stretch. At the corrected stations, starting morph at the throat
instead of the first positive station left a 0.408976 mm radial error at
station 12, p=0. Reserving that first segment closes the remaining error.
The combined fixture's x/y coordinates equal the morph-only fixture's
exactly, establishing that progress precedes stretch.
