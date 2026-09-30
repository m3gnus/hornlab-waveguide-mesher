# Translated half-model preview regressions

`reproductions.json` contains eight configurations for the source-cap orientation
failure: OSSE and R-OSSE in bare, freestanding, enclosure and infinite-baffle
modes, with quadrants 14, a 7 mm vertical placement, and global scale 0.48.
Their rounded morphs and throat prefixes also exercise placement independently
of the wall construction. Only parameters supported by the baseline are used.

The preview tests exercise both levels and quadrant 1/12 controls. Analytic cap
tests cover either half plane, both translation signs, flat/convex/concave
sources, rim attachment, normals and curvature.

Native parity tests use a denser control lattice to limit CAD fitting error,
keep the automatic 5 degree cap, and also cover scaled FREEFORM and ICW cases.
FREEFORM retains its supported approximate fit. STEP exports the full model
with the source membrane retained. Infinite-baffle points are registered to
the established mouth-origin frame before comparison. Sampled distances are
bounded by 0.1 mm for solve nodes to STEP, 0.15 mm for preview vertices to STEP,
and 0.25 mm for preview vertices to solve triangles. These are placement
regressions; they do not promise these bounds for every native cap fit or a
coarse CAD lattice.
