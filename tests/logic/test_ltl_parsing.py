import pytest

import automatix.logic.ltl as ltl

CASES = [
    (
        "X(Gp2 U Fp2)",
        ltl.NextOp(
            None,
            ltl.UntilOp(
                ltl.GloballyOp(None, ltl.Identifier("p2")),
                None,
                ltl.EventuallyOp(None, ltl.Identifier("p2")),
            ),
        ),
    ),
    ("!Fp2", ltl.NotOp(ltl.EventuallyOp(None, ltl.Identifier("p2")))),
]


@pytest.mark.parametrize("expr,expected_ast", CASES)
def test_ltl_parsing(expr: str, expected_ast: ltl.Expr) -> None:
    parsed = ltl.parse(expr)
    assert parsed == expected_ast
