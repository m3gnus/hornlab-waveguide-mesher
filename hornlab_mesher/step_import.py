"""Caller-neutral STEP import, face mapping, and surface-mesh validation.

This module is the stable import path. The implementation lives in three
modules, re-exported here name for name so existing imports keep working:

* :mod:`step_mapping` -- STEP faces onto gmsh surfaces, surface re-anchoring
  after healing, and the OCC healing ladder;
* :mod:`mesh_repair` -- weld, degenerate removal, winding and orientation
  (:func:`postprocess_mesh`);
* :mod:`mesh_validation` -- topology, symmetry-rim detection and frequency
  limits.

The STEP text parsers live in :mod:`step_text` (standard library only) and are
re-exported here too. Callers that monkeypatch a name must patch the module
that defines it, except :func:`postprocess_mesh`, which consumers look up here
at call time.

Units: a live gmsh model is in millimetres (OCC's default target unit), and
the ``*_MM`` constants are millimetres. Functions that take mesh coordinates
in "step units" say so and take ``unit_scale_to_m`` (metres per coordinate
unit, 1e-3 for millimetres); every millimetre quantity they apply is
converted through it, never compared with raw coordinates.
"""

from __future__ import annotations

from .step_prepare import snap_symmetry_plane_vertices
# Re-exported, not merely used: callers have always imported the STEP text
# parsers from this module, and several names below have no other use here.
from .step_text import (  # noqa: F401
    _STEP_CONTROL_RE,
    _STEP_RECORD_RE,
    BODY_ENTITIES,
    SOLID_BODY_ENTITIES,
    SURFACE_BODY_ENTITIES,
    VOIDED_SOLID_BODY_ENTITIES,
    StepBody,
    advanced_face_order,
    advanced_face_body_positions_from_text,
    advanced_face_order_from_text,
    advanced_face_placements_from_text,
    advanced_face_surface_kinds_from_text,
    advanced_face_topology_from_text,
    advanced_face_vertices,
    advanced_face_vertices_from_text,
    blank_step_strings,
    count_step_bodies,
    decode_step_string,
    first_step_string,
    occ_make_solids_is_safe,
    parse_named_shell_faces,
    parse_named_shell_faces_from_text,
    parse_solid_brep_faces,
    parse_styled_face_groups,
    parse_styled_face_groups_from_text,
    read_step_text,
    step_length_unit_mm_from_text,
    record_entity_types,
    step_body_inventory,
    step_records,
    step_refs,
    strip_step_comments,
)

from .step_mapping import (  # noqa: F401
    advanced_face_order_for_surfaces,
    ANCHOR_MAX_AREA_REL_DIFF,
    ANCHOR_MAX_CENTROID_DISTANCE_MM,
    _anchor_pair,
    anchor_surface_order,
    _AnchorPair,
    _coerce_surface_geometry,
    _face_identity_classes,
    FACE_MATCH_TOLERANCE,
    gmsh,
    _gmsh_model_mm_per_step_unit,
    _gmsh_surface_boundaries,
    gmsh_surface_geometries,
    _GMSH_SURFACE_KINDS,
    gmsh_surface_tags,
    _greedy_anchor_matches,
    _greedy_over,
    _neighbours_from_shared,
    OCC_HEALING_FALLBACKS,
    _OCC_TARGET_UNIT_MM,
    RIGID_TAG,
    run_occ_healing_fallbacks,
    StepFaceGroup,
    StepFaceOrderError,
    StepLabelSelector,
    SurfaceGeometry,
)
from .mesh_validation import (  # noqa: F401
    detect_symmetry_planes,
    _edge_direction_stats,
    _edge_frequency_stats,
    _edge_on_expected_plane,
    FREQUENCY_ELEMENTS_PER_WAVELENGTH,
    mesh_frequency_validation,
    _signed_volume,
    _signed_volume_noise_floor,
    SIGNED_VOLUME_NOISE_REL,
    _source_wall_stats,
    SPEED_OF_SOUND_M_S,
    _topology_stats,
    _triangle_area2,
    _triangle_edge_lengths,
)
from .mesh_repair import (  # noqa: F401
    _BORE_ALIGNMENT_MARGIN,
    _bore_alignment_verdict,
    _compact_unused_vertices,
    DEGENERATE_MIN_QUALITY,
    _free_edges_on_expected_planes,
    _mesh_triangle_data,
    _normalize_to_positive_side,
    postprocess_mesh,
    REDUCED_ORIENTATION_MIRRORED_PARENT,
    REDUCED_ORIENTATION_SOURCE_ANCHOR,
    REDUCED_ORIENTATIONS,
    _remove_degenerate_triangles,
    _repair_triangle_winding,
    _source_anchored_verdict,
    _source_normal_projections,
    _SOURCE_RULE_BORE_ALIGNMENT,
    _SOURCE_RULE_NO_SOURCE,
    _SOURCE_RULE_PROJECTION,
    _SOURCE_RULE_UNRESOLVED,
    _SOURCE_RULE_VOLUME_FALLBACK,
    _SYMMETRY_PROJECTION_DEGENERATE,
    _SYMMETRY_PROJECTION_EMPTY,
    _SYMMETRY_PROJECTION_NO_OPEN_AXIS,
    _SYMMETRY_PROJECTION_NO_SOURCE,
    _SYMMETRY_PROJECTION_OK,
    _symmetry_source_projection_detail,
    _weld_near_duplicate_vertices,
    WELD_TOLERANCE_MM,
)


# The Part 21 text layer lives in ``step_text`` because it must be importable
# where numpy and meshio are not -- an embedded CAD Python runs the same body
# rule. These aliases keep the private spellings this module and its tests have
# always used.
_step_records = step_records
_step_refs = step_refs
_decode_step_string = decode_step_string
_first_step_string = first_step_string
_parse_named_shell_faces = parse_named_shell_faces
_parse_solid_brep_faces = parse_solid_brep_faces
_parse_styled_face_groups = parse_styled_face_groups
_advanced_face_order = advanced_face_order


# The detector is public because it is the only way to re-read a cut from a
# finished mesh with no knowledge of what was cut, which is what makes an
# auto-cut self-checking: a plane that was cut but does not come back as an
# open rim was capped, and a capped plane meshes as a rigid baffle rather than
# as a symmetry plane.
_detect_symmetry_planes = detect_symmetry_planes

_snap_symmetry_plane_vertices = snap_symmetry_plane_vertices

# Private spellings of names that are public now; kept for existing imports.
_gmsh_surface_tags = gmsh_surface_tags
_gmsh_surface_geometries = gmsh_surface_geometries
_anchor_surface_order = anchor_surface_order
