# Design-enclosure reference extracts

`ath_reference.json` contains only aperture centres/radii, enclosure envelopes,
LF characteristic lengths and the q14 meridian needed by these tests. Source:
ATH V2026-09a LF reference capture dated 2026-10-21, cases named in `case_id`.
Each extracted `expected.json` has its original SHA-256 recorded. The GEO hash
and the q14 construction points are retained separately. At radius 50 mm the
GEO rounds the join to (r=17.500,z=100.000), sphere centre to z=80.434 and apex
to z=106.684, with the aperture at z=120.000. Native Gmsh 4.15.0 tessellated
ATH's original GEO; the tests compare to the exact meridian and its X reflection,
rather than reproducing ATH's missing negative-X diaphragm in full output.

The ten config-driven cases resolve the measured enclosure envelopes into the
existing margin contract (including the four-sided Dim case); no ATH parser is
added. Their apertures use the cardinal mouth sample and measured edge gap.
The q14 surface comparisons reconstruct full geometry and refine LF resolution
to 3 mm to certify the complete triangle interiors within 0.1 mm. Other cases
keep their measured LF resolution, including 6 mm for `b-resolution`.

`horn_only.json` stores six representative configs and mesh hashes captured on
base `e35f9da`, Python 3.13 / Gmsh 4.15.0. Tests additionally export that base and
the current revision in fresh processes using the same runtime, compare bytes,
and compare absent versus empty `woofers`. Historical hashes are provenance;
the live comparison remains valid across supported native runtimes.
