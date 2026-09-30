# ATH V2025-12 throat adapter reference exports

Provenance: ATH V2025-12, exported on 2026-09-30 in the C5 verification run
`20260930T084815Z` (broker job `260930-102659-compute-6bb1`). Selection follows
[the throat adapter contract](../../../../docs/throat-adapter-contract.md), the
verification run's `PROBES.md`, `RESULTS.md`, `metrics.json`, and exit statuses.
No ATH executable or executable hash is distributed here.

Use **0.001 mm Euclidean point tolerance**. These are integrity fixtures only:
no mesher adapter code exists yet, and the tests do not compare mesher geometry
to these exports. A passing integrity test does not establish adapter support,
body-formula parity, source construction, mesh topology, or CAD agreement.

ATH V2026-08c reproduces the block-form numbers on a 32-case sample to all six
exported decimals, but rejects the flat (top-level) OSSE form with
`Parsing NULL expression!`. These top-level cases cannot be regenerated on
newer builds; use V2025-12 for them. Reference fixtures acquired on newer builds
use block form and must be marked with their actual build. This directory
contains only V2025-12 exports.

## Files and coordinates

- `index.json` lists every case, filenames, columns, delimiter, retained
  meridian angles, station counts, total point counts, and CSV SHA-256 digests.
- `<case>.csv` is the original profiles GridExport: no header; semicolon
  separated `x;y;z` in mm; one meridian per block, separated by a blank line.
  Every original six-decimal coordinate is preserved, including all stations.
  CRLF is normalized to LF. No slices CSV is needed: the verification run
  proved slices are the transpose of profiles after removing the closing
  duplicate. Files containing N512 retain 0, 45 and 90 degrees; all other
  cases retain all 16 meridians (0 through 337.5 degrees, step 22.5).
- `<case>.json` records the case id, ATH version, contract interpretation,
  profile syntax, import expectation, role, and config keys and original value
  strings. Block nesting is retained, including GridExport settings; only
  `Output.DestDir` and the generated config comment are removed. The values
  therefore retain their generating precision and key scope.

For Ctrl cases, `baseline_case_id` points to the otherwise identical config
with Ctrl omitted. `adapter.control_points_join_local_mm` stores the three
supplied `(z,r)` pairs without refitting, followed by `(0,rJ)`. The automatic
fourth radius is measured at the exported 0-degree join and divided by export
scale; it is resolved evidence, not an independently editable input. The test
uses independent de Casteljau interpolation, then the declared driver-datum
translation and scale, to reproduce every exported cubic station at every
retained meridian. GridExport is centered: `Mesh.VerticalOffset` affects ABEC
artifacts, not these points. Source/wall artifacts stay outside this archive.

`first_station` skips a preceding cone's driver point when present. ATH uses
10 total extension segments by default; that cone occupies one segment,
leaving 9 cubic segments. Without a cone there are 10 cubic segments, or 8
with `Mesh.ThroatExtSegments=8`. Cubic samples use uniform u including both
endpoints; no body stations are removed. For R1 with a slot, ATH replaces the
u=1 cubic station with the far slot endpoint. `endpoint_axial_offset_mm`
records that slot length: the test checks the cubic's exported interior and
the slot endpoint translated from B(1) separately, without inventing a missing
join station. O1's slot belongs to the body and leaves this offset zero.

## Included cases

Each row below includes both the listed Ctrl case and its `<case>-noctrl`
baseline (40 files of each type in total). The two original flat OSSE bases
`base-osse-top-05` and `base-osse-top-06` bring the total to **42 cases**.

| Ctrl case | Purpose |
|---|---|
| `a1-b0-block-ttrunc` | Explicit R1 split |
| `a1-b0-top-fortyone` | Ignored top-level R1 trunc scope |
| `a2-t1-block-two-roots` | Verified first root |
| `a2-t0p8-block-two-roots` | Required refusal: ATH failed root search and discontinuous body |
| `a3-ext10-angle5-diam40` | Cone before unchanged cubic |
| `a3-v3-plus2` | Preserved supplied corner |
| `a4-osse-top-top-L120-slot0` | Flat O1, Length120, slot0 |
| `a4-osse-top-top-L120-slot10` | Flat O1, Length120, slot10 |
| `a4-osse-top-top-L120-slot20` | Flat O1, Length120, slot20 |
| `a4-osse-top-top-L140-slot0` | Flat O1, Length140, slot0 |
| `a4-osse-top-top-L140-slot10` | Flat O1, Length140, slot10 |
| `a4-osse-top-top-L140-slot20` | Flat O1, Length140, slot20 |
| `a4-rosse-Labsent-slot10` | R1 cylindrical slot after cubic |
| `a5-scale2-offset7-source2-wall0` | GridExport scale and driver datum |
| `a7-rot5` | R1 retained-body join-pivot rotation |
| `a7-osse-guiding` | Qualified O1 guiding subtype |
| `a8-rosse-s1-0p45-s2-0p2-trunc41-none` | Stretched R1 split |
| `a8-osse-s1-0p45-s2-0p2-trunc41-slot` | O1 slot/stretch order |
| `a9-rosse-N512-mesh-throatextsegments` | Eight cubic segments, fine R1 body |
| `a9-osse-N512-omitted` | Default cubic segments, fine O1 body |

The 15 geometry cases named by the contract's parity selection are all here,
plus every clean flat OSSE case within the supported O1 matrix and their
partners. `expected_import=refuse` on the A2 tmax=0.8 pair prevents promoting
that output into qualified R1 behavior. Its cubic is intact and testable; its
retained body is not a supported import. Baseline interpretation labels describe
the paired construction's evidence, not an active adapter.

The contract separately names `a6-arity5` and `a9-rosse-rollback` as diagnostic
fixtures, not parity geometry: the former falls back to the trimmed body and
the latter exits 255 without a CSV. Neither is included in the geometry index.
Hidden circular arc, rollback, expression controls, morph, arcterm, positive
walls and unverified combinations also remain outside the supported selection.
