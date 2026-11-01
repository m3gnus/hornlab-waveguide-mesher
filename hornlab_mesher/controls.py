"""Control names generated from WG's parameterRegistry.ts; aliases contain no wording."""
from __future__ import annotations

import json
from pathlib import Path

CONTROL_CATALOG = json.loads(Path(__file__).with_name("control_labels.json").read_text(encoding="utf-8"))
# Mesher spellings of the same controls. Values are registry identifiers.
ALIASES = {
    "a": "osse.a", "a0": "osse.a0", "L": "osse.L", "L_mm": "osse.L",
    "r0": "osse.r0", "r0_mm": "osse.r0", "R": "rosse.R", "R_mm": "rosse.R",
    "wall_thickness_mm": "mesh.wall_thickness", "wall thickness": "mesh.wall_thickness",
    "throat_res_mm": "mesh.throat_resolution", "mouth_res_mm": "mesh.mouth_resolution",
    "rear_res_mm": "mesh.rear_resolution", "angular_segments": "mesh.angular_segments",
    "length_segments": "mesh.length_segments", "corner_segments": "mesh.corner_segments",
    "driver_exit_diameter_mm": "native.adapter.driver_exit_diameter_mm",
    "exit_half_angle_deg": "native.adapter.exit_half_angle_deg", "length_mm": "native.adapter.length_mm",
    "join_t": "native.adapter.join_t", "driver_handle_mm": "native.adapter.driver_handle_mm",
    "body_handle_mm": "native.adapter.body_handle_mm", "radius_mm": "native.source.radius_mm",
    "outer_radius_mm": "native.source.outer_radius_mm", "depth_mm": "native.source.depth_mm",
    "vertical_offset_mm": "mesh.vertical_offset", "aperture_res_scale": "mesh.aperture_resolution_scale",
    "aspect_ratio": "cross_section.aspect_ratio", "exponent": "cross_section.exponent",
    "depth_margin_mm": "enclosure.depth_margin_mm", "edge_mm": "enclosure.edge_radius",
    "edge_type": "enclosure.edge_type", "space_b_mm": "enclosure.space_b",
    "space_l_mm": "enclosure.space_l", "space_r_mm": "enclosure.space_r", "space_t_mm": "enclosure.space_t",
    "max_triangles": "mesh.max_triangles",
    "rot": "osse.rotation", "gcurveType": "guide.curve_type", "gcurveWidth": "guide.width",
    "gcurveAspectRatio": "guide.aspect_ratio", "gcurveDist": "guide.distance", "gcurveRot": "guide.rotation",
    "gcurveSF": "guide.superformula", "gcurveSf": "guide.superformula", "gcurveSeN": "guide.superellipse_n",
}
for key in ("a", "b", "m1", "m2", "n1", "n2", "n3"):
    ALIASES["gcurveSf" + key.upper()] = "guide.sf_" + key


def control_label(key: str) -> str:
    """Fail closed on invented control names rather than supplying a phantom label."""
    identifier = ALIASES.get(key, key)
    if identifier in CONTROL_CATALOG:
        return CONTROL_CATALOG[identifier]["label"]
    for control in reversed(list(CONTROL_CATALOG.values())):
        if identifier in (control.get("path"), control["legacyKey"]):
            return control["label"]
    raise KeyError(f"Unknown registered control: {key}")
