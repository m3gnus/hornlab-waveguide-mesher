# Attach canonical sources to a resolved horn

`SourceAssembly.attach(horn, horn_config, ...)` resolves the existing horn design
through `resolve_geometry` and transports its axial spline into the canonical
assembly. OSSE, R-OSSE, ICW and rotational FREEFORM walls use their existing
axial fitting policy. The route supports an optional woofer and passive central
or annular bodies. A horn-only assembly has one physical source and channel.

The source rim must match the actual first resolved ring, including any throat
extension. Attaching never rescales an authored contour. Horn length and mouth
radius derive from the frozen wall. Source IDs, signed or zero weights and
normal/axial motion retain their authored values.

The wall is an exact circular revolution of that nonlinear axial spline. This
removes the existing tensor surface's approximate angular circle fit. Transport
records a conservative `circle_correction_bound_mm`, with a fixed 0.15 mm
admission limit. A design exceeding this limit must refine its resolved angular
sampling before attachment. For an acoustic fit, use finer `Mesh.ThroatResolution`
and `Mesh.MouthResolution` values in millimetres: the fit allocates its actual
angular stations from those resolutions and can override `Mesh.AngularSegments`.
Check the resulting `circle_correction_bound_mm` against the unchanged limit.
Axial spline fitting is unchanged; the wall is not
refitted to a cone. Representative original tensor surface differences were
0.000088 mm (OSSE), 0.000056 mm (R-OSSE), 0.000085 mm (ICW), and 0.055935 mm
(approximating FREEFORM). These are measured examples, not universal bounds.

Ordinary R-OSSE axial rollback is supported. A monotone projection of all exact
Bezier tangent hulls proves meridian simplicity. Interior knot multiplicity
must preserve continuity, and exact knot insertion constructs span certificates
at C0 joins. Arbitrary folded meridians lacking this proof are refused.

The canonical spline, preview, actual mesh and STEP share one authority. Finite
curve chord certificates use a fixed 1e-7 mm bound. Global wall/body, enclosure
side/back and woofer margins remain enforced. Existing whole-facet deviation,
source selectors, separate passive shells and passage clearance certificates
remain required. Reopened spline surface areas use converged CAD-derivative
quadrature at the existing area admission tolerance.

This route requires a full rotational wall with circular coaxial rims. Actual
noncircular or azimuth-varying walls, reduced domains, existing external shells,
interfaces, enclosure, placement, infinite baffles and separately authored
terminal geometry are outside this composition contract and are refused.
Rotational circle morphs are admitted by resolved geometry rather than by their
configuration key. This capability does not establish complete ATH parity.
