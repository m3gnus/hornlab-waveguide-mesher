# ATH import fixtures, ATH build V2026-08c

Profile grids exported by ATH (`GridExport`, profiles only, `x;y;z` in mm, one
blank-line-separated block per profile) and the configs that produced them.

| Pair | What it establishes |
| --- | --- |
| `osse-block-with-top-level-length`, `osse-block-without-top-level-length` | ATH ignores a top-level `Length` beside an `OSSE` block: the block's `L = 160` sets the horn length, and the two exports are byte-identical. |

The `.csv` files are ATH's output with line endings normalised to LF and the trailing blank line removed; they are
otherwise unedited.
