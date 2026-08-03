# Batch H — preview API stage 2: error-bounded adaptive sampling (branch `preview-api`)

Stage 1 (commit aea5f07) ships complete surfaces + analytic normals with fixed sampling. Stage 2 makes tessellation **error-bounded**: density driven by geometric tolerances, not segment labels. The normative sources are `../wg-rebuild-reviews/tessellation-review-260803.md` (P0.2, P1.1, P1.2, P1.4, and the "Mesher preview API" section) and `../waveguide-generator-v2/docs/FRAME-SPEC.md` §fidelity. Read both first.

**Path discipline: you may create/modify ONLY files under `hornlab_mesher/preview/` and `tests/test_preview_api.py` (+ new test files `tests/test_preview_*.py`). All other existing modules remain untouched — same additive discipline as stage 1, now scoped to the preview package you own.**

Runtime: `../Waveguide Generator/.venv/bin/python -m pytest` from this repo root.

## Scope

1. **Honor the reserved options** in `PreviewOptionsV1`: `max_chord_error_mm`, `max_normal_step_deg`, plus new `min_silhouette_segments` and `max_vertices` (per-call cap). LOD presets become tolerance sets (review's suggested targets): coarse = 64-segment silhouette floor, ≥12 axial intervals, ≤8–10° normal step, ≥6 intervals/quarter-roundover; fine = chord ≤0.03–0.05 mm and normal step ≤3°, ≥12/quarter-roundover; inspection = ≤0.02–0.03 mm and ≤2°.
2. **Adaptive subdivision in both parametric directions**: subdivide until midpoint-chord deviation AND endpoint normal change pass tolerance. **Semantic stations always inserted first** (throat boundary, morph start, rollback extrema, FREEFORM anchors/stations, corner tangencies, mouth, enclosure transitions — whatever is available from the importable canonical info; document which you could and couldn't obtain additively).
3. **Corner-concentrated angular sampling** (review P1.2): subdivide each corner arc independently to tolerance; keep flat sides sparse; preserve stable ring identity (FREEFORM row-correspondence) — where per-ring counts must differ, emit zipper indices rather than forcing uniform counts.
4. **Per-body budgets** (review P1.1): horn, outer wall, enclosure plan, roundovers, caps each get their own vertex accounting under `max_vertices`; when the cap binds, report `vertex_cap_limited: true` per surface and degrade gracefully (largest-error-first refinement so the budget goes where it matters).
5. **Anti-pop nesting** (review P1.4 / API-7): coarse stations are a subset of fine stations where practicable; document where not.
6. **Fidelity metadata becomes target-vs-achieved**: requested + achieved chord/normal-step per surface, cap-limited flag — matching FRAME-SPEC field names.
7. **Perf guard test**: fine-LOD OSSE eval (the reference config from stage 1 tests) asserted < 150 ms on this machine (measured headroom allows ~3× density; the guard catches accidental explosions, not micro-regressions — use a generous bound and mark it as machine-local).

## Tests

Extend/add `tests/test_preview_*.py`: achieved ≤ requested wherever not cap-limited (all four families, freestanding + enclosure); silhouette floor respected at coarse; roundover interval minimums; corner arcs meet tolerance while flat sides stay sparse; coarse⊂fine nesting where claimed; cap behavior (tiny `max_vertices` → cap-limited flags set, no crash, valid mesh); determinism; **full repo suite green, zero regressions** (baseline: 575 passed, 23 skipped).

## Rules
- Keep stage-1 public signatures backward-compatible (v2 server batch F is being written against them concurrently — additive fields only).
- Self-verify: full suite + new tests; iterate to green.
- Final message: algorithm summary (how subdivision decides), per-family before/after vertex counts and achieved errors at each LOD, perf numbers, nesting/semantic-station limitations, test counts baseline vs final.
