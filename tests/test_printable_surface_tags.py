"""Printable selection is distinct from solver and imported-geometry tags."""

from copy import deepcopy

import pytest

import hornlab_mesher
from hornlab_mesher import printable_surface_tags
from hornlab_mesher.tags import printable_surface_tags as tag_selection


@pytest.mark.parametrize("formula", [
    "OSSE", "R-OSSE", "FREEFORM", "LOOKUP", "ICW",
    "OSSE-ADAPTER", "OSSE-ADAPTER-CONTROLS", "OSSE-ADAPTER-AXIAL",
    "OSSE-ROUNDOVER", "OSSE-AXIAL", "OSSE-ARC", "SOURCE-DISK",
])
def test_printable_tags_for_every_formula(formula):
    config = {"formula": formula, "mesh": {"wall_thickness_mm": 5}}
    original = deepcopy(config)
    tags = printable_surface_tags(config)
    assert tags == ((1, 2) if formula == "SOURCE-DISK" else (1,))
    assert isinstance(tags, tuple)
    assert all(type(tag) is int for tag in tags)
    assert config == original


@pytest.mark.parametrize("config, expected", [
    ({}, (1,)),
    ({"type": "SOURCE-DISK"}, (1, 2)),
    ({"formula": " source-disk "}, (1, 2)),
    ({"profile": {"type": "R-OSSE"}}, (1,)),
])
def test_config_formula_defaults_and_aliases(config, expected):
    assert printable_surface_tags(config) == expected


def test_selection_is_exported_from_public_api():
    assert printable_surface_tags is tag_selection
    assert "printable_surface_tags" in hornlab_mesher.__all__
