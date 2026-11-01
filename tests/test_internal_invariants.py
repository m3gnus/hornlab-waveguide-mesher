"""Private invariants retain retry/catch categories and original diagnostics."""
import pickle

import numpy as np
import pytest

from hornlab_mesher.errors import ConfigError, InternalInvariantError, UserRefusal
from hornlab_mesher.freeform import validate_outer_offset_grid
from hornlab_mesher.profile_formulas import _verify_meridian_self_contact


@pytest.mark.parametrize("kind", ["shape", "finite", "meridian"])
def test_generated_invariants_keep_their_private_type(kind):
    with pytest.raises(InternalInvariantError) as caught:
        if kind == "shape":
            validate_outer_offset_grid(np.zeros((8, 2, 3)), np.zeros((8, 3, 3)), full_circle=True)
        elif kind == "finite":
            validate_outer_offset_grid(np.zeros((8, 2, 3)), np.full((8, 2, 3), np.nan), full_circle=True)
        else:
            _verify_meridian_self_contact(np.array([[0, 0], [1, np.nan]]))
    assert isinstance(caught.value, ConfigError)
    assert isinstance(caught.value, ValueError)
    restored = pickle.loads(pickle.dumps(caught.value))
    assert type(restored) is InternalInvariantError
    assert str(restored) == str(caught.value)


def test_authored_self_contact_remains_a_public_refusal():
    with pytest.raises(UserRefusal):
        _verify_meridian_self_contact(np.array([[0, 2], [2, 2], [2, 3], [1, 2]], dtype=float))
