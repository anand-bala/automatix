"""Unit tests for polynomial operator implementation."""

import typing

import algebraic
import logic_asts as logic
import numpy as np
import pytest
from algebraic.polynomials import RankDecomposition
from algebraic.semirings import boolean_algebra

from automatix.operators.polynomial import boolexpr_to_polynomial

K = typing.TypeVar("K", bound=algebraic.BoundedDistributiveLattice)


def extract_scalar(poly_result: RankDecomposition, algebra: algebraic.BoundedDistributiveLattice) -> np.generic:
    """Extract scalar value from polynomial evaluation result.

    RankDecomposition.evaluate() returns a constant RankDecomposition.
    Extract the actual scalar value from factors[0, 0, 0].
    """
    assert poly_result.algebra == algebra
    val: np.generic = np.asarray(poly_result.factors.data[0, 0, 0]).flat[0]
    return val


class TestBoolExprConversion:
    """Test boolean expression to polynomial conversion."""

    @pytest.fixture
    def algebra(self) -> algebraic.BooleanAlgebra:
        """Boolean algebra for tests."""
        return boolean_algebra()

    def test_literal_true(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Literal(True) -> one polynomial."""
        expr = logic.Literal(True)
        poly = boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra, backend="numpy")

        point = np.array([algebra.zero, algebra.zero, algebra.zero])
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert np.allclose(scalar, np.asarray(algebra.one))

    def test_literal_false(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Literal(False) -> zero polynomial."""
        expr = logic.Literal(False)
        poly = boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra, backend="numpy")

        point = np.array([algebra.one, algebra.one, algebra.one])
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert np.allclose(scalar, np.asarray(algebra.zero))

    def test_variable(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Variable(q) -> x_q polynomial."""
        expr = logic.Variable(2)
        poly = boolexpr_to_polynomial(expr, num_vars=5, algebra=algebra, backend="numpy")

        point = np.array([algebra.zero, algebra.zero, algebra.one, algebra.zero, algebra.zero])
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert np.allclose(scalar, np.asarray(algebra.one))

        point2 = np.array([algebra.zero, algebra.zero, algebra.zero, algebra.zero, algebra.zero])
        result = poly.evaluate(point2)
        scalar = extract_scalar(result, algebra)
        assert np.allclose(scalar, np.asarray(algebra.zero))

    def test_and_operation(self, algebra: algebraic.BooleanAlgebra) -> None:
        """And(x, y) -> x * y."""
        expr: logic.BoolExpr[int] = logic.And(tuple(logic.Variable(i) for i in [0, 1]))
        poly = boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra, backend="numpy")

        test_cases = [
            ({0: False, 1: False}, False),
            ({0: False, 1: True}, False),
            ({0: True, 1: False}, False),
            ({0: True, 1: True}, True),
        ]

        for point_dict, expected in test_cases:
            point = np.array([algebra.one if point_dict.get(i, False) else algebra.zero for i in range(3)])
            result = poly.evaluate(point)
            scalar = extract_scalar(result, algebra)
            expected_val = algebra.one if expected else algebra.zero
            assert np.allclose(scalar, np.asarray(expected_val)), f"Failed for {point_dict}"

    def test_or_operation(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Or(x, y) -> x + y."""
        expr: logic.BoolExpr[int] = logic.Or(tuple(logic.Variable(i) for i in [0, 1]))
        poly = boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra, backend="numpy")

        test_cases = [
            ({0: False, 1: False}, False),
            ({0: False, 1: True}, True),
            ({0: True, 1: False}, True),
            ({0: True, 1: True}, True),
        ]

        for point_dict, expected in test_cases:
            point = np.array([algebra.one if point_dict.get(i, False) else algebra.zero for i in range(3)])
            result = poly.evaluate(point)
            scalar = extract_scalar(result, algebra)
            expected_val = algebra.one if expected else algebra.zero
            assert np.allclose(scalar, np.asarray(expected_val)), f"Failed for {point_dict}"

    def test_not_operator_raises_assertion(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Not operator should raise ValueError."""
        expr: logic.BoolExpr[int] = logic.Not(logic.Variable(0))

        with pytest.raises(ValueError, match="Not operator encountered"):
            boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra, backend="numpy")

    def test_complex_expression(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Test complex expression: (x_0 AND x_1) OR x_2."""
        x0 = logic.Variable(0)
        x1 = logic.Variable(1)
        x2 = logic.Variable(2)
        expr = typing.cast(logic.BoolExpr[int], (x0 & x1) | x2)
        poly = boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra, backend="numpy")

        point = np.array([algebra.zero, algebra.zero, algebra.zero])
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert np.allclose(scalar, np.asarray(algebra.zero))

        point = np.array([algebra.one, algebra.one, algebra.zero])
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert np.allclose(scalar, np.asarray(algebra.one))

        point = np.array([algebra.zero, algebra.zero, algebra.one])
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert np.allclose(scalar, np.asarray(algebra.one))

    def test_invalid_state_index(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Invalid state index should raise ValueError."""
        expr = logic.Variable(10)

        with pytest.raises(ValueError, match="Invalid state variable"):
            boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra, backend="numpy")
