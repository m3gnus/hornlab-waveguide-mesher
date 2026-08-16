# Batch D — mesher preview API, stage 1 (STRICTLY ADDITIVE)

You are adding a versioned preview-geometry API to `hornlab-waveguide-mesher` (this repo, branch `preview-api`). This implements the "Mesher preview API" contract from two documents you must read first:
1. `../waveguide-generator/docs/reference/FRAME-SPEC.md` — especially the v1.1 surfaces/normals/fidelity section.
2. `../wg-rebuild-reviews/tessellation-review-260803.md` — the findings driving this work (P0.1 analytic normals, P0.3 complete surfaces, P1.3 explicit shading semantics).

**HARD CONSTRAINT — additive only:** create NEW files only (`hornlab_mesher/preview/` package + `tests/test_preview_api.py` + optional new test helpers). You may NOT modify any existing module, test, or config. This repo is a pushed public science package; the whole existing suite must stay green untouched. If stage 1 seems to need an existing-module change, work around it via the public/importable functions and record the limitation in your final report instead.

Stage 1 scope (stage 2 — error-bounded adaptive sampling — comes later; do NOT attempt it):

## API

`hornlab_mesher/preview/api.py`:

```python
@dataclass(frozen=True)
class PreviewOptionsV1:
    lod: str = "fine"            # "coarse" | "fine" | "inspection" — presets mapping to existing density knobs
    include_outer: bool = True
    include_enclosure: bool = True
    include_source_cap: bool = True
    include_rear_cap: bool = True
    # stage-2 fields reserved (accept + ignore w/ warning field in result): max_chord_error_mm, max_normal_step_deg

def build_preview_geometry(config: Mapping[str, Any], options: PreviewOptionsV1 = PreviewOptionsV1()) -> PreviewGeometryV1
```

`PreviewGeometryV1`: list of surfaces, each `{role, positions: f64 (N,3), indices: u32, normals: f64 (N,3), shading: "smooth"|"flat", normal_method: "analytic-parametric"|"exact-planar", closed_phi: bool}` + `metadata` (units mm, actual segment counts, per-stage timings, achieved-fidelity estimates, `api_version: "hornlab.preview/1"`).

Surface roles for stage 1 (complete model — the spike rendered only two grids; that gap is review finding P0.3):
- `horn.inner` (always), `horn.outer` (freestanding wall configs), `mouth_rim` (annulus/strip joining inner→outer or inner→enclosure front),
- `source_cap` (flat = exact planar disk; "rounded"/spherical source = properly tessellated spherical cap with analytic radial normals — NOT a cone fan; reuse existing cap math where importable),
- `enclosure.front`, `enclosure.roundover` (front + rear as present), `enclosure.side`, `enclosure.rear`,
- `wall.rear_cap` (freestanding rear closure).
Build these from the EXISTING canonical grid/ring functions (`build_viewport_geometry_from_config`, the ring builders in `hornlab_mesher/viewport.py`, enclosure plan builders) — port the assembly/stitching semantics that today live in the v1 browser tessellator (`../Waveguide Generator/src/geometry/viewportTessellator.js`: ray-aligned first enclosure ring, zipper stitching between unequal rings, modulo phi closure) into this package.

## Normals (the headline)

- Smooth surfaces: analytic-parametric normals `normalize(∂P/∂φ × ∂P/∂t)`. Where closed-form derivatives aren't importable without modifying existing modules, compute derivatives by central finite differences of the ANALYTIC sampler at parametric offsets (this is still "analytic-parametric" — it samples the true surface, never the triangle mesh). Unit length within 1e-3, finite, outward-consistent orientation (document the convention).
- Flat surfaces: exact planar normals.
- Hard boundaries (mouth rim edges, throat seam, intended enclosure edges): duplicated vertices between surfaces — one vertex never carries incompatible normals. No dihedral-angle inference anywhere.

## Fidelity metadata (measured, stage 1)

For each curved surface, estimate achieved `max_chord_error_mm` and `max_normal_step_deg` by comparing against a 4× denser reference sampling of the same analytic surface (helper in `hornlab_mesher/preview/fidelity.py`). Report per-surface in metadata. (Stage 2 will make these into targets; stage 1 just measures honestly.)

## Tests — `tests/test_preview_api.py`

Cover at least: OSSE freestanding, R-OSSE with enclosure + roundovers, ICW flat_baffle, FREEFORM (use configs adapted from existing test fixtures in this repo). Invariants per config:
- every requested role present; positions/normals finite; normals unit within 1e-3 and row-aligned; indices in range; phi closure on closed surfaces (no duplicated wrap seam);
- flat surfaces exactly planar with exact normals; spherical source cap: max radial deviation from the analytic sphere below the fine-LOD chord estimate; roundover surfaces present with ≥ the existing interval counts;
- fidelity metadata present and plausible (achieved chord error > 0, < 5 mm);
- determinism: two identical calls give byte-identical arrays.
Then run the ENTIRE existing repo suite and confirm zero regressions (it must be untouched).

Runtime: `../Waveguide Generator/.venv/bin/python -m pytest` from this repo root (editable install resolves to this tree). Some suites skip without ATH_REFERENCE_ROOT — skips are fine, failures are not; run the suite BEFORE your changes to record the baseline pass/skip counts and compare after.

Final message: files created, surface roles implemented per family, normals method used per surface, measured fidelity numbers for one example config per family, baseline vs after test counts, and every limitation you hit under the additive-only constraint.
