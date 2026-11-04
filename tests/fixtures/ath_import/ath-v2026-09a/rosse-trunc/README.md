ATH V2026-09a witnesses for R-OSSE throat cuts (November 2026).

The five CSVs are unmodified `profiles_g` exports, eight azimuths and 33
stations per meridian, with coordinates in millimetres. The matching CFGs
contain the original inputs. `trunc-tmax08.cfg` is a refusal witness: ATH
fails the first meridian's root search and produces inconsistent subsequent
meridians. It must not become an accepted zero-cut fallback.

The successful witnesses establish axial-mm removal after throat stretch,
the first forward root when two exist, translation by the requested cut,
a flat enlarged source rim, and default retained-body station mapping even
when a linear or nonlinear Z-map was supplied. CSV precision is six decimals;
ATH's root search stops at approximately 0.001 mm axial precision. An exact
root matches every exported coordinate to 0.001 mm.
