import pytest

import automatix.logic.strel as strel
from automatix.logic.ltl import Identifier, TimeInterval

CASES = [
    (
        "(G ! obstacle) & ((somewhere[0,2] groundstation) U goal)",
        strel.AndOp(
            strel.GloballyOp(None, strel.NotOp(strel.Identifier("obstacle"))),
            strel.UntilOp(
                strel.SomewhereOp(strel.DistanceInterval(0, 2), strel.Identifier("groundstation")),
                None,
                strel.Identifier("goal"),
            ),
        ),
    ),
    (
        "G( (somewhere[1,2] drone) | (F[0, 100] somewhere[1,2] (drone | groundstation)) )",
        strel.GloballyOp(
            None,
            strel.OrOp(
                strel.SomewhereOp(strel.DistanceInterval(1, 2), strel.Identifier("drone")),
                strel.EventuallyOp(
                    TimeInterval(0, 100),
                    strel.SomewhereOp(
                        strel.DistanceInterval(1, 2),
                        strel.OrOp(Identifier("drone"), Identifier("groundstation")),
                    ),
                ),
            ),
        ),
    ),
]


@pytest.mark.parametrize("expr,expected_ast", CASES)
def test_strel_parsing(expr: str, expected_ast: strel.Expr) -> None:
    parsed = strel.parse(expr)
    assert parsed == expected_ast
