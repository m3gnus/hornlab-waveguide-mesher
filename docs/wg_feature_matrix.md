# WG → hornlab-mesher feature parity matrix

Historical source: deleted `Waveguide-Generator/server/solver/waveguide_builder.py`
(1302 lines) plus `waveguide_enclosure.py` (~1170 lines). Survey of which
payload fields the WG frontend (`Waveguide-Generator/src/`) and the Optimizer
(`Optimizer-Dashboard/bem_optimizer/`) actually exercise. Compiled 2026-05-17;
updated after legacy builder deletion on 2026-05-24.

## TL;DR

`waveguide_builder.py` looks big but its geometric vocabulary is small. Every
real WG/optimizer payload falls into one of three top-level cases below, all
gated by two payload fields: `enc_depth` and `outer_points`.

```
case A: inner-only horn         enc_depth == 0  AND  outer_points is None     ← hornlab-mesher
case B: freestanding wall shell enc_depth == 0  AND  outer_points provided   ← hornlab-mesher
case C: horn in cabinet box     enc_depth >  0                                ← hornlab-mesher
```

The horn formula (`formula_type`, R-OSSE / OSSE / Classical / LOOKUP /
TRACTRIX) is invisible to the mesher — the JS frontend evaluates the profile
and ships `inner_points` (and optionally `outer_points`). `formula_type` is
validated but never consumed inside `build_waveguide_mesh`. The mesher's job is
OCC surface construction + physical group assignment + Gmsh sizing.

## Case-by-case breakdown

### Case A — inner-only horn (done)

**Trigger:** `outer_points is None` and `enc_depth == 0`.

**Surfaces built:**
- Inner wall, from `inner_points`. BSpline path or faceted path (see "cross
  section" below).
- Throat source disc, reusing the inner-wall boundary curves at z_min.

**Physical groups:** all wall surfaces → tag 1 (SD1G0), throat disc → tag 2
(SD1D1001).

**Hornlab-mesher status:** implemented in
`hornlab_mesher/builders/point_grid.py`. Pre-deletion parity confirmed
node/triangle/tag-count compatibility against the WG private helpers; the
post-deletion contract guard is `tests/test_point_grid_contract.py`.

### Case B — freestanding wall shell

**Trigger:** `enc_depth == 0` AND `outer_points` provided. UI ships
`outer_points` only when the user enabled freestanding-shell mode (sets
`wall_thickness > 0` with no cabinet). Optimizer rarely uses this case.

**Surfaces built** (waveguide_builder.py lines 1096-1142):
- Inner wall (same as case A).
- Outer wall from `outer_points`.
- Mouth rim: ruled annular surface connecting inner+outer wall boundary curves
  at the mouth (z_max). Built with
  `_build_mouth_rim_from_boundaries(inner_dimtags, outer_dimtags)`. Falls back
  to control-point wire if boundary extraction fails.
- Rear disc: flat planar surface from the outer-wall throat boundary curves at
  z_throat - wall_thickness. Built with `_build_rear_disc_assembly`.
- Throat disc (same as case A — explicitly NOT connected to outer wall, so the
  throat cavity stays hollow and the source remains attached only to the
  inner wall).

**Group keys for density:** `inner`, `outer`, `mouth`, `rear`, `rear_cap`,
`throat_disc`. `density.py` already handles all of these (free-standing rear
constant resolution mode is wired).

**Physical groups:** outer + rear + mouth all join tag 1 (SD1G0). Throat disc
tag 2.

**Payload fields:** `outer_points`, `wall_thickness` (only used in the legacy
rear-disc fallback; the preferred path reuses outer wall boundary curves).

### Case C — horn in cabinet box

**Trigger:** `enc_depth > 0`. UI ships this whenever the user is doing a
standard cabinet simulation. Optimizer default is `enc_depth = 320 mm`.

**Surfaces built** (`waveguide_enclosure.py::_build_enclosure_box`):
- Inner wall (same as case A).
- **Front baffle**: planar surface with two loops — outer rounded-rect/ellipse/
  superellipse plan minus the inner horn mouth opening. Hole-cut.
- **Front roundover/chamfer**: 0–2 ruled surfaces from the front-baffle
  inset ring sweeping out + back through one quarter turn of the edge.
- **Side walls**: ruled surface from front-edge outer ring at
  `z_front - edge_depth` down to back-edge outer ring at
  `z_back + edge_depth`.
- **Back roundover/chamfer**: mirror image of the front.
- **Back cap**: planar surface closing the box at `z_back`.

**Variants (UI exercises all of these)**:
- `enc_plan_type ∈ {1, 2, 3}` — rounded rectangle / ellipse / superellipse.
  Three different XY-plane samplers (`_sample_rounded_rect`,
  `_sample_ellipse`, `_sample_superellipse`).
- `enc_edge_type ∈ {1, 2}` — rounded fillet (3 profiles, ruled thru-sections)
  vs. chamfer (2 profiles, single ruled surface).
- `enc_plan_n` — superellipse exponent, only used when plan_type=3. Typical
  2–4.
- `enc_space_l/t/r/b` — per-side wall clearances. Independent.
- `enc_edge` — corner radius (auto-clamped to half min(w, h) - 0.1).
- `enc_depth_margin` — auto-clamps `enc_depth` so the back wall never cuts
  through the horn. Optimizer leans on this by setting `enc_depth = 0` and
  `enc_depth_margin = 20`.

**Group keys for density:** `enclosure` (sides), `enclosure_edges_front`,
`enclosure_edges_back`, `enclosure_front`, `enclosure_back`. Plus
`enclosure_bounds` dict carrying `bx0, bx1, by0, by1, z_front, z_back, cx, cy`.

`density.py` already implements the bilinear front/back panel formula and the
z-interpolated side-wall formula — only the geometry builder is missing.

**Physical groups:** enclosure surfaces all join tag 1 (SD1G0).

**Partial-domain (open) case:** bare and freestanding point-grid payloads now
carry `quadrants=14`, `12`, or `1` through the WG bridge with
`grid_closed=false`. The throat source is filled as an axis-centered open
sector, so tag 2 keeps the half/quarter physical area expected by symmetry
solves. Enclosure payloads still require `quadrants=1234`; partial-domain
enclosure seams are rejected before mesh build until the enclosure stitcher is
validated for open rings.

## Cross-section variants (orthogonal to A/B/C)

- **BSpline path** (default, `_build_surface_from_points`): single
  `addBSplineSurface` patch fitted to the (n_phi, n_length+1) grid with column
  0 duplicated to close. Smooth.
- **Faceted path** (`_build_faceted_surface_from_points`): one ruled surface
  per cell of the grid. Triggered by `grid_preserve_rings == True` or
  (legacy) `cross_section.aspect_ratio_mode == "natural"` AND
  `cross_section.exponent >= 20`. UI emits `grid_preserve_rings` directly.

**Hornlab-mesher status:** both paths already implemented in
`builders/_occ.py` (`build_surface_from_points` /
`build_faceted_surface_from_points`) and wired through
`PointGridHornGeometry.preserve_grid`. The `cross_section` dict from WG is not
yet read by the mesher dataclass — but the JS frontend resolves it down to
`grid_preserve_rings` before sending, so we don't need to handle the dict
itself.

## Dead / legacy code paths (do NOT port)

- `symmetry_cut="yz"` — kwarg exists on `build_waveguide_mesh` but nobody in
  routes/services/optimizer calls it with a non-None value. Skip for now.
- `closed=False` enclosure path — bare/freestanding partial-domain meshes are
  active, but enclosure partial-domain seams are still unsupported.
- `TRACTRIX` formula type — auto-migrated to `Classical shape=6` in frontend
  `state.js:30-33`. Validation list still accepts it for old saved presets.
  No mesher action needed (formula doesn't enter the mesher anyway).
- `_build_throat_disc` (legacy flat disc, no boundary attach) — present as a
  helper but not called by `build_waveguide_mesh`. Skip.
- `_build_rear_wall` (annular ruled between inner+outer at throat) — present
  but the active wall-shell path explicitly avoids it to keep the throat
  cavity hollow.

## Density / mesh sizing parity

Already implemented in `hornlab_mesher/density.py`:
- Throat→mouth axial-z linear interpolation (inner, mouth groups).
- Throat-disc constant size.
- Free-standing-wall outer/rear constant size at `rear_res_mm`.
- Per-quadrant front/back enclosure resolution string parsing ("25,25,25,25"
  or scalar broadcast).
- Bilinear panel formula + z-interpolated side wall formula.
- Front/back roundover surfaces follow nearest panel formula.
- MeshSizeMin/Max derived from collected resolution values × {0.5, 1.5}.

`build_point_grid` populates `BuiltGeometry.mesh_surface_groups` and
`BuiltGeometry.enclosure_bounds` so `density.py` activates the enclosure path.

## Frontend vs. optimizer payload overlap

| Field family | Frontend | Optimizer |
|---|---|---|
| Profile params (R, r, b, m, q, L, s, n, a, k, formula_type, …) | yes | yes |
| Enclosure (enc_depth, enc_space_*, enc_plan_type, enc_edge_type, enc_edge) | yes | yes, default enc_depth=320 |
| Guiding curve (gcurve_*, morph_*) | yes | yes |
| Cross-section (classical_cross_section, gcurve, grid_preserve_rings) | yes | yes |
| `outer_points` freestanding shell | yes (only when user toggles) | not in practice |
| LOOKUP profile | yes | yes (CMA can emit) |
| `quadrants` override | yes for bare/freestanding `14`, `12`, `1`; enclosure remains full-domain | no |
| `symmetry_cut` kwarg | no | no |

Frontend and optimizer overlap >95%. The only asymmetry: freestanding shells
(case B) are a frontend-only flow in current usage.

## Historical extension scope

In priority order:

1. **Case C, closed-domain, plan_type=1, edge_type=1** (rounded-rect enclosure
   with rounded fillets). Covers the largest fraction of real WG/optimizer
   traffic.
2. **Case C, plan_type ∈ {2, 3}** (ellipse + superellipse plans).
3. **Case C, edge_type=2** (chamfered edges).
4. **Case B** (freestanding wall shell). Frontend-only, smaller impact.
5. Skip: `closed=False` enclosure path, `symmetry_cut`, legacy throat-disc
   helpers.

The live post-deletion smoke coverage is in `tests/test_point_grid_contract.py`
and `tests/test_density_contract.py`; historical exact parity depended on the
now-deleted WG helper modules.
