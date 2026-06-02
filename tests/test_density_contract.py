from __future__ import annotations

from hornlab_mesher.density import (
    _enclosure_resolution_formula,
    _parse_quadrant_resolutions,
)


def test_quadrant_resolution_parsing_matches_wg_contract():
    assert _parse_quadrant_resolutions("5", 9.0) == [5.0, 5.0, 5.0, 5.0]
    assert _parse_quadrant_resolutions("6,7,8,9", 9.0) == [6.0, 7.0, 8.0, 9.0]
    assert _parse_quadrant_resolutions("6,7", 9.0) == [6.0, 7.0, 9.0, 9.0]
    assert _parse_quadrant_resolutions("", 9.0) == [9.0, 9.0, 9.0, 9.0]


def test_enclosure_resolution_formula_hits_wg_corner_targets():
    formula = _enclosure_resolution_formula(
        [2.0, 3.0, 4.0, 5.0],
        [12.0, 13.0, 14.0, 15.0],
        bx0=-10.0,
        bx1=10.0,
        by0=-20.0,
        by1=20.0,
        z_front=100.0,
        z_back=40.0,
    )

    def eval_formula(x, y, z):
        return float(eval(formula, {"__builtins__": {}}, {"x": x, "y": y, "z": z}))

    assert eval_formula(10.0, 20.0, 100.0) == 2.0
    assert eval_formula(-10.0, 20.0, 100.0) == 3.0
    assert eval_formula(-10.0, -20.0, 100.0) == 4.0
    assert eval_formula(10.0, -20.0, 100.0) == 5.0
    assert eval_formula(10.0, 20.0, 40.0) == 12.0
    assert eval_formula(-10.0, 20.0, 40.0) == 13.0
    assert eval_formula(-10.0, -20.0, 40.0) == 14.0
    assert eval_formula(10.0, -20.0, 40.0) == 15.0
