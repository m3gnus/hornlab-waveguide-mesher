# Translated half-model preview regressions

`reproductions.json` contains eight configurations for the source-cap orientation
failure: OSSE and R-OSSE in bare, freestanding, enclosure and infinite-baffle
modes, with quadrants 14, a 7 mm vertical placement, and global scale 0.48.
Their rounded morphs and throat prefixes also exercise placement independently
of the wall construction. Only parameters supported by the baseline are used.

The preview tests exercise both levels and quadrant 1/12 controls. Analytic cap
tests cover either half plane, both translation signs, flat/convex/concave
sources, rim attachment, normals and curvature.

Native parity tests use a denser control lattice to limit CAD fitting error and
keep the automatic 5 degree cap. OSSE and R-OSSE cases use the reproductions'
global scale 0.48; FREEFORM and ICW cases run at scale 1. FREEFORM retains its
supported approximate fit. STEP exports the full model with the source membrane
retained. Infinite-baffle points are registered to the established mouth-origin
frame before comparison.

Every sampled distance (solve nodes to STEP, preview vertices to STEP, preview
vertices to solve triangles) is bounded by one 1 mm placement bound. It was
loosened from an earlier draft's 0.1/0.15/0.25 mm, which mixed placement with
the base's own cap-fitting differences (up to about 0.19 mm at scale 1 on
FREEFORM and ICW rounded caps). The bound is set from the defect, not from the
residuals: without the fix the cap sits several millimetres off (reverting the
fix fails the rounded half-model cases at 2.7 to 2.9 mm), while measured
preview-to-STEP cap differences are about 0.014 mm. It is a placement
regression bound, not a CAD accuracy promise. Exact placement is pinned
separately, to 1e-12 mm absolute, by the analytic translation tests.
