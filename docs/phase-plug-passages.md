# Native phase-plug passages

`SourceAssembly.phase_plugs` adds up to eight passive revolved bodies inside the
shared horn. `PhasePlug(id, z0_mm, z1_mm, inner0_mm, outer0_mm, inner1_mm,
outer1_mm)` describes a closed linear meridian. Zero inner radii at both ends
produce a central body; positive inner radii produce an annular vane. The stack
shares inlet/outlet Z planes, with independently tapered radial dimensions.
Dimensions are in millimetres from the horn source rim; axes remain parallel +Z.

Bodies must be ordered from the axis outwards. Length, radial thickness, source
and mouth clearance, adjacent body separation and horn-wall clearance must
exceed 0.1 mm. Contacts, crossing radii, topology changes, duplicate IDs and
nonfinite dimensions are rejected. The sloping-surface separation uses a
conservative Euclidean bound, rather than treating a radial difference as a
normal gap. Annular vanes create multiple full circular passages: N vanes
without a core produce N+1 passages. A core removes the central passage.

The additive required feature is `native-phase-plug-passages-v1`, alongside
`native-source-contour-v1` and `native-shared-horn-woofer-v1`. `source.json`
records each passive body's stable percent-encoded `plug/<id>/<surface>` rigid
roles, inlet/outlet passage radii, analytical clearances and expected shell
topology. Passive bodies never become moving patches or drive channels. Geometry,
excitation and density identities remain separate; omitting bodies preserves the
existing assembly recipe and geometry identity.

STEP contains one enclosure shell and one closed shell per body, with actual
shared OCC edges within each shell and zero volumes. The preview mesh must have
opposite paired edge incidence, positive signed volume on every component and
Euler characteristic 2 for the enclosure/core or 0 for an annular vane.

The complete mesh must lie within `min(0.15 mm, minimum clearance / 10)` of the
finite canonical surfaces. Whole-facet bounds include interior samples and a
Lipschitz covering radius; planar disks/annuli use analytical triangle bounds.
The certified residual gap subtracts two maximum surface bounds from the
analytical clearance and must remain positive. Targets resolve gaps with at
least three local target lengths and body thickness with at least two.
`export_assembly(..., passage_refinement=1|2|4)` refines bodies and the adjacent
horn-wall section locally. It retains the native 250,000-triangle ceiling;
unsatisfied tolerance or resources refuse atomic publication.

Waveguide Generator's counterpart is documented in
`docs/architecture/NATIVE-SOURCE-CONTOURS.md` and its assembly editor. It reopens
the STEP, validates mappings/body counts and certifies the persisted imported
mesh again before publishing an immutable ingestion ID. Its 22,000-triangle and
memory admission ceilings remain in force.

This contract supports authored linear tapers, finite tips and open annular
passages. Automatic equal-path channel design, curved/meandering vanes,
azimuthal supports, arbitrary axes and manufacturing solids require additional
contracts. Geometry refinement and prescribed-velocity dispatch checks establish
no acoustic accuracy or acoustic convergence.
