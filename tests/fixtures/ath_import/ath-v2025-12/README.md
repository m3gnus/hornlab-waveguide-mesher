# ATH import fixtures, ATH build V2025-12

Profile grids exported by ATH (`GridExport`, profiles only, `x;y;z` in mm, one
blank-line-separated block per profile) and the configs that produced them.
Build V2026-09a gives byte-identical exports for `osse-block-without-s` and
`osse-block-s-0.5`.

| Set | What it establishes |
| --- | --- |
| `m7_rot_none`, `m7_rot_in`, `m7_rot_top`, `m7_rot_both` | Block-only OSSE Rot leaves the grid byte-identical to no Rot; top-level Rot controls the grid even when the block also supplies it. |
| `osse-block-without-s`, `osse-block-s-0.5`, `osse-block-s-0.7` | ATH's default for an omitted OSSE `s` (`Term.s`) is 0.5, not the 0.7 the 4.8.2 User Guide states: the export without `s` is byte-identical to the one with `s = 0.5`, and differs from `s = 0.7` (mouth width 144.74 mm against 164.83 mm). |

The `.csv` files are ATH's output with line endings normalised to LF and the trailing blank line removed; they are
otherwise unedited.
