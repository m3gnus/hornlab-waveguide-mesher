"""User refusals retain diagnostics without exposing implementation details."""
import pickle

import pytest

from hornlab_mesher.config_parser import ConfigError
from hornlab_mesher.config_builder import build_geometry_params
from hornlab_mesher.source_contour import dome


def test_refusal_is_a_compatible_picklable_config_error():
    from hornlab_mesher.errors import UserRefusal
    refusal = UserRefusal("Wall thickness", "The wall folds over itself.", "Reduce Wall thickness.", diagnostic="offset grid flip")
    restored = pickle.loads(pickle.dumps(refusal))
    assert isinstance(restored, ConfigError)
    assert (restored.field, restored.user_message, restored.next_step) == ("Wall thickness", "The wall folds over itself.", "Reduce Wall thickness.")
    assert str(restored) == "offset grid flip"


def test_coverage_angle_names_its_control_and_remedy():
    from hornlab_mesher.errors import UserRefusal
    with pytest.raises(UserRefusal) as caught:
        build_geometry_params({"formula": "OSSE", "L": 120, "a": 95})
    assert caught.value.field == "Mouth coverage angle"
    assert "90" in caught.value.user_message
    assert "Mouth coverage angle" in caught.value.next_step


def test_dome_dimensions_name_controls_and_remedy():
    from hornlab_mesher.errors import UserRefusal
    with pytest.raises(UserRefusal) as caught:
        dome(radius_mm=12, height_mm=0)
    assert "Dome" in caught.value.user_message
    assert "Dome height" in caught.value.next_step


@pytest.mark.parametrize("family", ["icw", "lookup", "slot", "zmap", "baffle"])
def test_reachable_families_carry_named_remedies(family):
    from hornlab_mesher.errors import UserRefusal
    from hornlab_mesher.icw.solver import ICWTargets
    from hornlab_mesher.profile_formulas import lookup_profile_array
    from hornlab_mesher.profile_sampling import _custom_zmap
    from hornlab_mesher.builders.enclosure import EnclosureBaffleError
    from hornlab_mesher.config_builder import resolve_geometry
    calls = {
        "icw": (lambda: ICWTargets(mode="flat_baffle", r0=12.7, theta0_deg=15, coverage_angle_deg=90), "Coverage angle"),
        "lookup": (lambda: lookup_profile_array({"lookupProfile": [[1, 2], [0, 3]]}), "Lookup profile points"),
        "slot": (lambda: resolve_geometry({"formula": "OSSE", "L": 120, "a": 45, "slotLength": 120, "lengthMode": "total"}), "Straight slot length"),
        "zmap": (lambda: _custom_zmap(4, [0, 1, 1, 0]), "Z-map points"),
        "baffle": (lambda: (_ for _ in ()).throw(EnclosureBaffleError("A flush enclosure has no front baffle for a woofer.")), "Enclosure margins"),
    }
    call, label = calls[family]
    with pytest.raises(UserRefusal) as caught:
        call()
    assert caught.value.field == label
    assert caught.value.user_message
    assert caught.value.next_step
    restored = pickle.loads(pickle.dumps(caught.value))
    assert (restored.field, restored.user_message, restored.next_step) == (caught.value.field, caught.value.user_message, caught.value.next_step)


@pytest.mark.parametrize("bad_radius", [-20.0, 0.0])
def test_lookup_refuses_a_radius_at_or_through_the_axis(bad_radius):
    """A LOOKUP row with radius <= 0 is refused at the shared validator.

    The radius is a distance from the axis, so it must be positive: a sign
    error such as [[0, 10], [100, -20]] used to be accepted and the sampled
    wall crossed the axis near z = 33.3 mm (10 - 30 z / 100 = 0), reversing
    azimuth past it. The config resolver uses the same validator, so both entry
    points refuse.
    """
    from hornlab_mesher.config_builder import resolve_geometry
    from hornlab_mesher.errors import UserRefusal
    from hornlab_mesher.profile_formulas import lookup_profile_array

    table = [[0.0, 10.0], [50.0, 15.0], [100.0, bad_radius]]
    with pytest.raises(UserRefusal) as caught:
        lookup_profile_array({"lookupProfile": table})
    assert caught.value.field == "Lookup profile points"
    assert "row 3" in str(caught.value)
    with pytest.raises(UserRefusal):
        resolve_geometry({
            "formula": "LOOKUP", "profile": {"lookupProfile": table},
            "mesh": {"topology_mode": "legacy", "angular_segments": 8, "length_segments": 4},
        })
