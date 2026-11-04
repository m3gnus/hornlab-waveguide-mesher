# ATH lip and Opus composition witnesses

ATH V2026-09a, generated 4 November 2026. Each family (`lip`, `opus`,
`lip_opus`) has an unchanged `base` and seven variants changing one control:
Scale=.7, Scale.Z=1.4, extension length=8, extension length=8 with angle=12,
Rot=12, VerticalOffset=17 and Source.Shape=2. Radius=130 and Curv=1 are
explicit in both source-shape variants.

Each compressed NumPy fixture contains the first quadrant (eleven meridians)
of the original profile CSV as `inner`, and every rear-ring point plus apex
extracted from bem_mesh.geo as `rear`. Bore points, the auxiliary mouth-disc
centre and source controls are excluded by their construction order. ATH's
three-decimal rear coordinates are preserved. The offset variant's rear Y
coordinates have 17 subtracted, matching the mesher's intrinsic-frame storage;
its placement is checked separately in viewport, preview and mesh tests. ATH
exports profile CSVs in the intrinsic frame even when VerticalOffset is active.

The fixture assertions compare the entire rear set in both directions. They
establish the shared sampled construction, not ATH surface numbering or identical
triangulation. Scale.Z leaves the body rear normals unchanged; lip angles use
the scaled body. Other dimensions (lip radius, wall, throat shift and Opus
FrontOffset/Step) are not multiplied by either scale.
