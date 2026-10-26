# ATH import fixtures, ATH build V2026-09a

Profile grids exported by ATH (`GridExport`, profiles only, `x;y;z` in mm, one
blank-line-separated block per profile) and the configs that produced them.

| Pair | What it establishes |
| --- | --- |
| `m3_rosse_morph`, `m3_rosse_nomorph` | R-OSSE rectangular morph changes radial profiles: unmodified mouth radius 150 mm; morphed radii 160–217.766142 mm. Each meridian blends additively in radius with `((u - u0)/(1 - u0))^Rate`, `u = z/z_mouth`; above 1 along the rollback. |
| `rm_fp0`, `rm_fp05` | R-OSSE `Morph.FixedPart` is compared with `z/z_max` and starts at the first station at or past it (never the throat); `u0` is that station's `z/z_mouth`. |
| `rm_fp09` | A start beyond the mouth's z (`u0 > 1`) gives negative factors; a radius crossing the axis stays positive. CSV rounding amplifies to ~1.5e-3 mm. With a fractional Rate ATH writes NaN. |
| `rm_fp1`, `ao_fp1` | FixedPart above 0.99 is used as 0.99 ("set too big"). On OSSE a start on the mouth station moves the mouth alone onto the target. |
| `rm_rate1`, `ao_rate1` | `Morph.Rate` below 1.01 is used as 1.01 ("set too small"), OSSE included. |
| `rm_ls10`, `rm_tmax095`, `rm_wall` | The R-OSSE morph rule holds for 10 segments, `tmax = 0.95` and a free-standing wall. |
| `rm_circ_var` | A Circle target on R-OSSE ignores FixedPart (blend from the first station); on OSSE it does not. |
| `rm_implicit`, `rm_cap` | An implicit target half-size is `floor(mouth) + 1` (151 for R = 150); TargetWidth/Height above 1000 mm are used as 1000. |
| `rm_wh`, `rm_cs8`, `rm_na100`, `ao_cs4` | Angular sampling: AngularSegments rounds up to a multiple of 8 and CornerSegments does not add to it (100 -> 104, 64 + CornerSegments 4 -> 64); the corner arc has `max(3, CornerSegments - 1)` intervals equal in arc angle, five when the key is absent; a .5 wall share rounds up. |
| `rm_var_small` | ATH still morphs an R-OSSE whose length varies a little with azimuth, per meridian. `rm_var_large` (config only) varies more: ATH builds it unmorphed ("attempt to stretch a variable-length profile!"), and WG applies the morph. |
| `m4_enc` | Enclosure Spacing is ordered +x,+y,-x,-y; 25,30,35,40 gives x bounds -196/186 mm on this horn. `m4_enc-enclosure.csv` contains the outer enclosure boundary witnesses. |
| `m5_osse_omit_a_s`, `m5_rosse_omit_all` | Omitted import coefficients use ATH V2026-09a defaults: OSSE a=45, s=0.5; R-OSSE R=200, a=40, a0=0, r=0.25, m=0.9, b=0.4, q=3.5. |
| `m6_var_wall` | Azimuth-varying OSSE with a wall builds a RearShape 1 throat shifted -5 mm and offset by wall thickness along its meridional chord normal; rear z span 0.258 mm. `m6_var_wall-rear.csv` contains the throat-ring witnesses, including duplicated ATH endpoints. |
| `m7_rot_none`, `m7_rot_in`, `m7_rot_top`, `m7_rot_both` | OSSE-block Rot is ignored; top-level Rot is used even if both are written. |
| `m10_ext_rot` | Extension remains unrotated; the body rotates about the extension endpoint. Ten extension segments precede the twenty body segments. |
| `sf1`, `sf2` | `sf(m,n1,n2,n3,a,b,p)` is the Gielis superformula, including odd `m` and non-unit `a`, `b`. |
| `sf3` | `log` is natural logarithm, `round(2.5)` is 3, and `exp`/`floor` compose with azimuth-dependent expressions. |
| `f0_plain`, `f0_plain_nac` | CORE OSSE with p-dependent `a`, `n`, `s`; no morph/stretch. Without NAC, 100 angular plus 4 corner segments gives 104; with NAC, 100. |
| `f0_morph`, `f0_morph_nac` | Rectangular morph, `Rate=3`, `FixedPart=0`: blending starts at the first positive axial station, with zero blend there. |
| `f0_stretch`, `f0_stretch_nac` | `s1=1.14`, `s2=0.025` transform z after sampling; x/y match the plain exports byte for byte. |
| `p1` | Plain OSSE lip, six rows from the original mouth. |
| `segdef` | Omitted Segments gives five lip rows. |
| `of20` | Opus with lip, Step=50 and FrontOffset=20; front is shifted without a normal offset. |
| `onr` | Opus without a lip, Step=25 and FrontOffset=40. |
| `odef` | Omitted Opus Step/FrontOffset give 25/40. |
| `ow0` | Opus authored zero wall is clamped to 0.1 mm. |
| `ofull` | Full-ring Opus with lip; duplicated ATH ring endpoints retained in the extracted point list. |
| `rs1` | RearShape 1 with lip: -5 mm throat shift and selected concentric outer-lip rows. |
| `coreabec` | Full CORE configuration, morph/expressions/stretch/lip, warped Opus throat and selected rows [4,7,9,11,12]. |
| `f0_both`, `f0_both_nac` | Combined morph/stretch. `f0_both_nac` is CORE at its original mesh settings with Roundover/RearShape omitted. Morph progress is measured before stretch. |

The `.csv` files are ATH's output with line endings normalised to LF and the trailing blank line removed; they are
otherwise unedited.

All `f0_*` configurations and `coreabec` are:
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

For `p1` and `segdef`, `ABEC.SimType=2` is added to declare the supported
free-standing topology; these profile-only probes wrote no mesh. This changes
no GridExport coordinates. All other configs preserve their ATH controls.
The CORE GridExport includes morph, expressions and stretch before the lip.

`*-rear.csv` extracts only points with the ATH rear/front sizing values
(`RearResolution`, `1.2*MouthResolution`) from `ABEC_FreeStanding/bem_mesh.geo`.
The body, the auxiliary mouth-disc center and source sphere controls are
excluded. CSV columns are x/y/z in mm with ATH's original three decimals.
Duplicated full-ring endpoints may occur; comparisons are vertex-set comparisons
in both directions, not numbering/topology comparisons. The files include the
shifted/offset throat, every selected middle ring, the Opus front ring (where
present) and the fan apex. The whole .geo files are intentionally omitted.
