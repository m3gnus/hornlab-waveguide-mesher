# ATH V2025-12 throat-stretch exports

These are verbatim subsets of GridExport profile points from 36 configs
executed by ATH V2025-12. Each generated `config.txt` identifies that version.
`cases.json` contains the generating text (output destination removed),
normalized parameters, and full-export comparison results. No executable or
private reference archive is required for these tests.

Each CSV keeps four meridians (0, 45, 90, 135 degrees) and selected ordinal
stations, including throat/extension joins, interior stations, foldback and
mouth. `ath_x_mm`, `ath_y_mm`, `ath_z_mm` are unmodified exported coordinates;
`zero_*` are the matching ordinal points in the paired s1=0, s2>0 export.
`phi_rad` is the meridian azimuth. `station` is the unscaled native evaluator
parameter: OSSE obtains it from the paired zero axial coordinate (undoing
Rot), and R-OSSE obtains main t by inverting the independently transcribed
published radial equation in ordinal order, then converts it to the composite
parameter where prefixed. This never searches nearest axial coordinates.
For the refused OSSE slot cases, station is a diagnostic radius match to the
existing native slot model, not a claim to reproduce ATH's unknown transition.

The full comparison covered all 60,096 exported points, 46,528 in supported
cases. Both scalar and vector results use maximum axial/radial component
deviation in mm after Scale. The target is 2e-4 mm; supported cases max at
2.634e-5 mm. Paired ATH transverse coordinates are identical, and the degree
map fits the paired axial values within 2.628e-6 mm even for refused slot
cases. Sparse fixtures test these facts and refusal, independently of the
optional external ATH archive gate.

The initial implementation dropped top-level Rot beside an OSSE block and
stretched before rotation when Rot was passed directly. Both are fixed.
OSSE block slots fail even without stretch; they are refused rather than
silently meshed as the native explicit tube. Native zero-stretch evaluation
is unchanged. See `docs/geometry-contract.md` for the measured contract and
unmeasured combinations which require additional ATH probes.

| Case | Full points | 899c2fa import max (mm) | Final scalar/vector max (mm) | Import |
| --- | ---: | ---: | ---: | --- |
| osse-desmos-combined-stretch | 1776 | 6.11474193 | 6.11474193 | refused; diagnostic only |
| osse-desmos-combined-zero | 1776 | 7.69985741 | 7.69985741 | refused; diagnostic only |
| osse-desmos-extension-stretch | 1776 | 1.99539907e-06 | 1.99539907e-06 | supported |
| osse-desmos-extension-zero | 1776 | 1.78694e-06 | 1.78694e-06 | supported |
| osse-desmos-gcurve-stretch | 1616 | 5.64384879e-06 | 5.64384879e-06 | supported |
| osse-desmos-gcurve-zero | 1616 | 5.64384879e-06 | 5.64384879e-06 | supported |
| osse-desmos-length-stretch | 1616 | 1.99539906e-06 | 1.99539906e-06 | supported |
| osse-desmos-length-zero | 1616 | 1.78694e-06 | 1.78694e-06 | supported |
| osse-desmos-plain-stretch | 1616 | 1.99539906e-06 | 1.99539906e-06 | supported |
| osse-desmos-plain-zero | 1616 | 1.78694e-06 | 1.78694e-06 | supported |
| osse-desmos-rotated-stretch | 1616 | 36.7176344 | 7.30462759e-06 | supported |
| osse-desmos-rotated-zero | 1616 | 36.4538227 | 7.30462759e-06 | supported |
| osse-desmos-slot-stretch | 1616 | 6.11474193 | 6.11474193 | refused; diagnostic only |
| osse-desmos-slot-zero | 1616 | 7.69985741 | 7.69985741 | refused; diagnostic only |
| osse-tritonia-s-combined-stretch | 1776 | 3.64793276 | 3.64793276 | refused; diagnostic only |
| osse-tritonia-s-combined-zero | 1776 | 3.78855643 | 3.78855643 | refused; diagnostic only |
| osse-tritonia-s-extension-stretch | 1776 | 1.96745003e-06 | 1.96745003e-06 | supported |
| osse-tritonia-s-extension-zero | 1776 | 1.96745003e-06 | 1.96745003e-06 | supported |
| osse-tritonia-s-gcurve-stretch | 1616 | 8.75078731e-06 | 8.75078731e-06 | supported |
| osse-tritonia-s-gcurve-zero | 1616 | 8.75078731e-06 | 8.75078731e-06 | supported |
| osse-tritonia-s-length-stretch | 1616 | 1.96744996e-06 | 1.96744996e-06 | supported |
| osse-tritonia-s-length-zero | 1616 | 1.96744996e-06 | 1.96744996e-06 | supported |
| osse-tritonia-s-plain-stretch | 1616 | 1.96744996e-06 | 1.96744996e-06 | supported |
| osse-tritonia-s-plain-zero | 1616 | 1.96744996e-06 | 1.96744996e-06 | supported |
| osse-tritonia-s-rotated-stretch | 1616 | 18.5667339 | 3.48859076e-06 | supported |
| osse-tritonia-s-rotated-zero | 1616 | 17.5770861 | 3.48859076e-06 | supported |
| osse-tritonia-s-slot-stretch | 1616 | 3.64793276 | 3.64793276 | refused; diagnostic only |
| osse-tritonia-s-slot-zero | 1616 | 3.78855643 | 3.78855643 | refused; diagnostic only |
| rosse-published-combined-stretch | 1776 | 2.6337474e-05 | 2.6337474e-05 | supported |
| rosse-published-combined-zero | 1776 | 6.43484028e-06 | 6.43484028e-06 | supported |
| rosse-published-extension-stretch | 1776 | 2.6337474e-05 | 2.6337474e-05 | supported |
| rosse-published-extension-zero | 1776 | 6.43484027e-06 | 6.43484027e-06 | supported |
| rosse-published-plain-stretch | 1616 | 2.6337474e-05 | 2.6337474e-05 | supported |
| rosse-published-plain-zero | 1616 | 6.43484028e-06 | 6.43484028e-06 | supported |
| rosse-published-slot-stretch | 1616 | 2.6337474e-05 | 2.6337474e-05 | supported |
| rosse-published-slot-zero | 1616 | 6.43484027e-06 | 6.43484027e-06 | supported |
