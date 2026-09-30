"""Shared stretch bounds and canonical identity of an inactive axial map."""

from typing import Any, Mapping


# At most 900,000 mm of displacement, far below half an ulp at the largest
# finite float. Thus adding the displacement to any finite x stays finite.
# The atan argument may saturate to infinity; atan still has a finite limit.
STRETCH_COEFFICIENT_MAX = 10_000.0


def stretch_is_inactive(params: Mapping[str, Any]) -> bool:
    """A literal zero (including a numeric string) disables the whole map.

    Expressions remain potentially active; never evaluate them at a single
    angle to decide the identity of an entire horn.
    """
    for key in ("s1", "s2"):
        try:
            if float(params.get(key, 0.0)) == 0.0:
                return True
        except (TypeError, ValueError):
            pass
    return False


def canonical_stretch_params(params: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(params)
    if stretch_is_inactive(result):
        result.pop("s1", None)
        result.pop("s2", None)
    return result
