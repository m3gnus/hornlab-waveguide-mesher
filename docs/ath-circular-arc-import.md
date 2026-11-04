# ATH circular-arc import

Fresh, unstamped imports and imports identifying ATH V2026-09a with a populated
`OSSE` block build ordinary OSSE even when they state `Throat.Profile = 3`.
ATH ignores that selector, `CircArc.Radius` and `CircArc.TermAngle` both beside
and inside this block. Each ignored value is reported. Unknown `CircArc.*`
members still refuse, as do empty selectors, R-OSSE and unmeasured versions.
Saved geometry interpretations keep their existing fields and refusals.

The ten retained `arc-00` through `arc-09` block probes have byte-identical
profile and slice exports, including the profile-1 baseline. All 1040 profile
points match the ordinary OSSE owner to six-decimal export precision. The
independent `osse-base` / `osse-circarc` probes also have identical mesh bytes
after replacing the embedded probe name. The config echoes identify V2026-09a.

Flat syntax activates a different construction. At Length 120, throat radius
18 and coverage angle 40 degrees, the existing circular-arc profile evaluator
matches explicit radius 200, radius 400, angle 45 degrees, and radius 200 with
angle 60 degrees within 0.000002 mm. The last case confirms radius precedence.
The default and angle-60-only probes differ by up to 75.908888 and 570.550506 mm.
The default log reports radius 103.781 with centre (16.235, 120.503); the
angle-60 log reports centre (345.526, -271.931) and a radius-too-small warning.
The flat profile-1 run returns success but logs `Parsing NULL expression!`
and exports no grid. Exit status alone does not qualify a construction.

Active flat imports remain refused through preview, solve meshes and export.
There is also a saved-design boundary: `build_geometry_params` currently drops
the ordinary WG `throatProfile`, `circArcRadius` and `circArcTermAngle` fields.
Passing these fields through globally would change existing designs. That
requires an owner decision or a new explicit, versioned construction; this
change does neither and leaves the existing profile formula untouched.
