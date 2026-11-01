# ATH V2026-09a control witnesses

52 input/export pairs cover independent axial scale (including global-scale
override), positive/negative/azimuth-dependent R-OSSE rotation, straight
extension counts, slot/rotation/extension compositions, rectangular morphing,
and six R-OSSE default body budgets. CSV values are native GridExport output
in millimetres, with LF endings and one terminal newline for Git transport.
The test tolerance is .0002 mm; measured fresh residuals are
smaller than .00006 mm.

`source-caps.json` contains eleven native source measurements. Nine contain
pole/centre coordinates from ATH's generated geometry script, rounded there
to .001 mm. Two extension-source cases mark an inconsistent reference pole:
their `radius_mm` is the separately logged matched wavefront radius. Tests
verify that radius and explicitly retain the pole discrepancy; they do not
declare the malformed reference sphere equivalent to a consistent cap.

`manifest.json` records hashes of every input, export and measurement file.
Licensed executables, analysis projects and raw process captures are not part
of these fixtures. See `docs/ath-import-controls.md` for the supported domain.
