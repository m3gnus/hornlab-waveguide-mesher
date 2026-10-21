"""Opt-in feasibility experiment: pistons in the existing design enclosure.

Not a released config contract. Coordinates and recess are pre-placement mm.
The hook runs after the ordinary formula builder, before density and tagging.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .builders._occ import require_gmsh


@dataclass(frozen=True)
class SpikeWoofer:
    x_mm: float
    y_mm: float
    radius_mm: float
    recess_mm: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x_mm, self.y_mm, self.radius_mm, self.recess_mm)):
            raise ValueError('woofer dimensions must be finite')
        if self.radius_mm <= 0 or self.recess_mm < 0:
            raise ValueError('woofer radius must be positive and recess nonnegative')
        # The shared postprocessor welds vertices within 0.005 mm. Smaller
        # recesses would move the piston rim back onto the baffle.
        if 0 < self.recess_mm < .01:
            raise ValueError('woofer recess must be zero or at least 0.01 mm (vertex weld tolerance)')


def cut_woofers(built, geometry):
    """Fragment the actual enclosure front faces; preserve the horn builder."""
    woofers = geometry.enclosure.spike_woofers
    if not woofers:
        return
    if not geometry.closed:
        raise ValueError('woofer spike requires full domain (quadrants=1234)')
    if set(built.surface_groups) - {1, 2, 3}:
        raise ValueError('woofer spike does not support subdomain interfaces')
    gmsh = require_gmsh()
    occ = gmsh.model.occ
    occ.synchronize()
    z = built.enclosure_bounds['z_front']
    front = []
    for tag in built.mesh_surface_groups['enclosure']:
        # OCC bounding boxes include face tolerances (up to 0.05 mm here).
        # Identify the actual plane by its type and centre instead.
        if gmsh.model.getType(2, tag) == 'Plane' and abs(occ.getCenterOfMass(2, tag)[2] - z) < 1e-6:
            front.append((2, tag))
    if not front:
        raise ValueError('no planar front baffle found')
    for i, w in enumerate(woofers):
        if w.recess_mm >= z - built.enclosure_bounds['z_back']:
            raise ValueError('woofer recess reaches enclosure back')
        for other in woofers[:i]:
            if math.hypot(w.x_mm-other.x_mm, w.y_mm-other.y_mm) <= w.radius_mm+other.radius_mm+0.1:
                raise ValueError('woofer apertures overlap or lack 0.1 mm clearance')
    discs = [(2, occ.addDisk(w.x_mm, w.y_mm, z, w.radius_mm, w.radius_mm)) for w in woofers]
    _, maps = occ.fragment(front, discs)
    front_result = {tag for mapping in maps[:len(front)] for dim, tag in mapping if dim == 2}
    patches = [{tag for dim, tag in mapping if dim == 2} for mapping in maps[len(front):]]
    # A tool outside the baffle still exists after fragment. Require every disc
    # fragment to belong to an actual original baffle face, and its full area.
    for w, patch in zip(woofers, patches):
        area = sum(occ.getMass(2, tag) for tag in patch & front_result)
        if patch - front_result or not math.isclose(area, math.pi*w.radius_mm**2, rel_tol=1e-7):
            raise ValueError('woofer aperture is not wholly inside the flat front baffle (mouth/edge collision)')
    source_top = set().union(*patches)
    replacements = front_result - source_top
    old_front = {tag for _, tag in front}
    for groups in (built.surface_groups, built.mesh_surface_groups):
        for key, tags in groups.items():
            groups[key] = sorted((set(tags) - old_front) | (replacements if set(tags) & old_front else set()))
    # Keep the emitted circle chords below the spike's 0.1 mm aperture budget.
    max_radius = max(w.radius_mm for w in woofers)
    circle_points = max(16, math.ceil(math.pi / math.acos(1-min(0.05/max_radius, 1.0))))
    gmsh.option.setNumber('Mesh.MinimumCirclePoints', circle_points)
    sources = {'horn-throat': 2}
    tag_map = {'1': {'source_id': None, 'instance_id': None, 'role': 'rigid'},
               '2': {'source_id': 'horn-throat', 'instance_id': None, 'role': 'primary'}}
    for i, (w, patch) in enumerate(zip(woofers, patches)):
        if w.recess_mm:
            # Extrude the split disc faces. Retain only the external cylinder
            # walls: faces on internal sector seams are not physical boundaries.
            occ.synchronize()
            boundary = {abs(tag) for dim, tag in gmsh.model.getBoundary(
                [(2, tag) for tag in patch], combined=True, oriented=True) if dim == 1}
            lower = set()
            collar = set()
            for tag in patch:
                rim = {abs(t) for dim, t in gmsh.model.getBoundary([(2, tag)], oriented=True) if dim == 1}
                out = occ.extrude([(2, tag)], 0, 0, -w.recess_mm)
                lower.add(out[0][1])
                # OCC returns bottom, volume, then one lateral per rim edge.
                occ.synchronize()
                for dim, side in out[2:]:
                    if dim != 2:
                        continue
                    curves = {abs(t) for d, t in gmsh.model.getBoundary([(2, side)], oriented=False) if d == 1}
                    if curves & (rim & boundary):
                        collar.add(side)
                occ.remove([dt for dt in out if dt[0] == 3], recursive=False)
                internal = [(d, t) for d, t in out[2:] if d == 2 and t not in collar]
                if internal:
                    occ.remove(internal, recursive=False)
            occ.remove([(2, tag) for tag in patch], recursive=False)
            patch = lower
            built.mesh_surface_groups['enclosure'].extend(sorted(collar))
            built.surface_groups.setdefault(3, []).extend(sorted(collar))
        source_id, physical = f'woofer-{i+1}', 101+i
        sources[source_id] = physical
        tag_map[str(physical)] = {'source_id': source_id, 'instance_id': None, 'role': 'secondary'}
        built.surface_groups[physical] = sorted(patch)
        # Use front-baffle density for the pistons in this spike.
        built.mesh_surface_groups['enclosure'].extend(sorted(patch))
    # Imported-style single rigid tag; retain enclosure roles for density.
    built.surface_groups[1] = sorted(set(built.surface_groups[1] + built.surface_groups.pop(3, [])))
    built.metadata.update(source_tags=sources, tag_map=tag_map,
                          tag_namespace='spike-design-assembly-v1', rigid_tag=1,
                          spikeBaffleZMm=z, spikeWooferCount=len(woofers),
                          spikePhysicalNames={tag: name for name, tag in sources.items() if tag != 2})
    occ.synchronize()
