"""Unit tests for polynomial operator implementation."""

import jax.numpy as jnp
import logic_asts as logic
import pytest
from algebraic.semirings import boolean_algebra

from automatix.operators.polynomial import boolexpr_to_polynomial


def extract_scalar(poly_result, algebra):
    """Extract scalar value from polynomial evaluation result.

    RankDecomposition.evaluate() returns a constant RankDecomposition.
    Extract the actual scalar value from factors[0, 0, 0].
    """
    # The result is a constant polynomial with factors shape (rank, degree, num_vars+1)
    # For a constant, we look at factors[0, 0, 0] which is the constant term
    return poly_result.factors.data[0, 0, 0]


class TestBoolExprConversion:
    """Test boolean expression to polynomial conversion."""

    @pytest.fixture
    def algebra(self):
        """Boolean algebra for tests."""
        return boolean_algebra()

    def test_literal_true(self, algebra) -> None:
        """Literal(True) -> one polynomial."""
        expr = logic.Literal(True)
        poly = boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra)

        # Evaluate at any point: should be 1
        point = {0: algebra.zero, 1: algebra.zero, 2: algebra.zero}
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert jnp.allclose(scalar, jnp.asarray(algebra.one))

    def test_literal_false(self, algebra) -> None:
        """Literal(False) -> zero polynomial."""
        expr = logic.Literal(False)
        poly = boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra)

        # Evaluate at any point: should be 0
        point = {0: algebra.one, 1: algebra.one, 2: algebra.one}
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert jnp.allclose(scalar, jnp.asarray(algebra.zero))

    def test_variable(self, algebra) -> None:
        """Variable(q) -> x_q polynomial."""
        expr = logic.Variable(2)
        poly = boolexpr_to_polynomial(expr, num_vars=5, algebra=algebra)

        # Evaluate at x_2 = 1, others = 0: should be 1
        point = {i: algebra.zero for i in range(5)}
        point[2] = algebra.one
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert jnp.allclose(scalar, jnp.asarray(algebra.one))

        # Evaluate at x_2 = 0: should be 0
        point[2] = algebra.zero
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert jnp.allclose(scalar, jnp.asarray(algebra.zero))

    def test_and_operation(self, algebra) -> None:
        """And(x, y) -> x * y."""
        # Use parse_expr to construct the expression
        expr = logic.parse_expr("x_0 & x_1")
        poly = boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra)

        # Test truth table
        test_cases = [
            ({0: False, 1: False}, False),
            ({0: False, 1: True}, False),
            ({0: True, 1: False}, False),
            ({0: True, 1: True}, True),
        ]

        for point_dict, expected in test_cases:
            point = {i: algebra.one if point_dict.get(i, False) else algebra.zero for i in range(3)}
            result = poly.evaluate(point)
            scalar = extract_scalar(result, algebra)
            expected_val = algebra.one if expected else algebra.zero
            assert jnp.allclose(scalar, jnp.asarray(expected_val)), f"Failed for {point_dict}"

    def test_or_operation(self, algebra) -> None:
        """Or(x, y) -> x + y."""
        # Use parse_expr to construct the expression
        expr = logic.parse_expr("x_0 | x_1")
        poly = boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra)

        # Test truth table
        test_cases = [
            ({0: False, 1: False}, False),
            ({0: False, 1: True}, True),
            ({0: True, 1: False}, True),
            ({0: True, 1: True}, True),
        ]

        for point_dict, expected in test_cases:
            point = {i: algebra.one if point_dict.get(i, False) else algebra.zero for i in range(3)}
            result = poly.evaluate(point)
            scalar = extract_scalar(result, algebra)
            expected_val = algebra.one if expected else algebra.zero
            assert jnp.allclose(scalar, jnp.asarray(expected_val)), f"Failed for {point_dict}"

    def test_not_operator_raises_assertion(self, algebra) -> None:
        """Not operator should raise AssertionError."""
        expr = logic.parse_expr("! x_0")

        with pytest.raises(AssertionError, match="Not operator encountered"):
            boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra)

    def test_complex_expression(self, algebra) -> None:
        """Test complex expression: (x_0 AND x_1) OR x_2."""
        expr = logic.parse_expr("(x_0 & x_1) | x_2")
        poly = boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra)

        # Test a few points
        # (0,0,0) -> False
        point = {0: algebra.zero, 1: algebra.zero, 2: algebra.zero}
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert jnp.allclose(scalar, jnp.asarray(algebra.zero))

        # (1,1,0) -> True (x_0 AND x_1)
        point = {0: algebra.one, 1: algebra.one, 2: algebra.zero}
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert jnp.allclose(scalar, jnp.asarray(algebra.one))

        # (0,0,1) -> True (x_2)
        point = {0: algebra.zero, 1: algebra.zero, 2: algebra.one}
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert jnp.allclose(scalar, jnp.asarray(algebra.one))

    def test_invalid_state_index(self, algebra) -> None:
        """Invalid state index should raise ValueError."""
        expr = logic.Variable(10)  # Out of range

        with pytest.raises(ValueError, match="Invalid state variable"):
            boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra)
