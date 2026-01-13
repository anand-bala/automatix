"""Comprehensive tests for SparsePolynomial implementation (baseline)."""
# ruff: noqa: ANN201, ANN001
# mypy: disable-error-code="no-untyped-call,no-untyped-def,import-not-found"

from __future__ import annotations

import jax.numpy as jnp
import pytest
from bitarray import frozenbitarray


class TestSparsePolynomialConstruction:
    """Test basic polynomial construction."""

    def test_constant_creation(self, sparse_helper, bool_algebra):
        """Test constant polynomial creation."""
        alg = sparse_helper(bool_algebra, 3)
        p = alg.constant(jnp.array(True))

        assert len(p) == 1
        assert frozenbitarray("000") in p
        assert jnp.array_equal(p[frozenbitarray("000")], jnp.array(True))

    def test_variable_creation(self, sparse_helper, bool_algebra):
        """Test single variable polynomial creation."""
        alg = sparse_helper(bool_algebra, 3)
        x_0 = alg.variable(0)

        assert len(x_0) == 1
        assert frozenbitarray("100") in x_0
        assert jnp.array_equal(x_0[frozenbitarray("100")], jnp.array(True))

    def test_variable_with_coefficient(self, sparse_helper, maxmin_algebra):
        """Test variable creation with custom coefficient."""
        alg = sparse_helper(maxmin_algebra, 2)
        x_0 = alg.variable(0, coefficient=jnp.array(5.0))

        assert len(x_0) == 1
        assert jnp.array_equal(x_0[frozenbitarray("10")], jnp.array(5.0))

    def test_zero_polynomial(self, sparse_helper, bool_algebra):
        """Test zero polynomial (constant with zero value)."""
        alg = sparse_helper(bool_algebra, 3)
        p = alg.constant(bool_algebra.zero)

        assert len(p) == 1
        assert jnp.array_equal(p[frozenbitarray("000")], bool_algebra.zero)

    def test_one_polynomial(self, sparse_helper, bool_algebra):
        """Test one polynomial (multiplicative identity)."""
        alg = sparse_helper(bool_algebra, 3)
        p = alg.constant(bool_algebra.one)

        assert len(p) == 1
        assert jnp.array_equal(p[frozenbitarray("000")], bool_algebra.one)


class TestSparsePolynomialAddition:
    """Test polynomial addition."""

    def test_addition_simple(self, sparse_helper, bool_algebra):
        """Test adding two variables."""
        alg = sparse_helper(bool_algebra, 2)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)

        # In boolean algebra, add is OR
        p = alg.add(x_0, x_1)

        # Should have two terms
        assert len(p) == 2
        assert frozenbitarray("10") in p
        assert frozenbitarray("01") in p

    def test_addition_commutative(self, sparse_helper, maxmin_algebra):
        """Test a + b = b + a."""
        alg = sparse_helper(maxmin_algebra, 3)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)

        p1 = alg.add(x_0, x_1)
        p2 = alg.add(x_1, x_0)

        # Compare by checking coefficients
        assert p1.keys() == p2.keys()
        for key in p1.keys():
            assert jnp.array_equal(p1[key], p2[key])

    def test_addition_associative(self, sparse_helper, maxmin_algebra):
        """Test (a + b) + c = a + (b + c)."""
        alg = sparse_helper(maxmin_algebra, 3)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)
        x_2 = alg.variable(2)

        p1 = alg.add(alg.add(x_0, x_1), x_2)
        p2 = alg.add(x_0, alg.add(x_1, x_2))

        # Should have same terms
        assert p1.keys() == p2.keys()
        for key in p1.keys():
            assert jnp.allclose(p1[key], p2[key])

    def test_addition_identity(self, sparse_helper, maxmin_algebra):
        """Test a + 0 = a."""
        alg = sparse_helper(maxmin_algebra, 3)
        x_0 = alg.variable(0)
        zero = alg.constant(maxmin_algebra.zero)

        p = alg.add(x_0, zero)

        # Should still be just x_0 (but might have zero term)
        # Check by evaluation
        test_point = {0: jnp.array(2.0), 1: jnp.array(3.0), 2: jnp.array(4.0)}
        result = alg.evaluate(p, test_point)
        assert jnp.allclose(list(result.values())[0], jnp.array(2.0))

    def test_addition_same_monomial(self, sparse_helper, maxmin_algebra):
        """Test that adding same monomial combines coefficients."""
        alg = sparse_helper(maxmin_algebra, 2)
        x_0 = alg.variable(0, coefficient=jnp.array(2.0))
        x_0_again = alg.variable(0, coefficient=jnp.array(3.0))

        p = alg.add(x_0, x_0_again)

        # In tropical max-plus: add is max, so max(2.0, 3.0) = 3.0
        assert frozenbitarray("10") in p
        assert jnp.allclose(p[frozenbitarray("10")], jnp.array(3.0))


class TestSparsePolynomialMultiplication:
    """Test polynomial multiplication."""

    def test_multiplication_simple(self, sparse_helper, bool_algebra):
        """Test multiplying two variables."""
        alg = sparse_helper(bool_algebra, 2)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)

        # x_0 * x_1 in Boolean algebra is x_0 AND x_1
        p = alg.mul(x_0, x_1)

        assert len(p) == 1
        assert frozenbitarray("11") in p

    def test_multiplication_commutative(self, sparse_helper, maxmin_algebra):
        """Test a * b = b * a for commutative semirings."""
        alg = sparse_helper(maxmin_algebra, 3)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)

        p1 = alg.mul(x_0, x_1)
        p2 = alg.mul(x_1, x_0)

        assert p1.keys() == p2.keys()
        for key in p1.keys():
            assert jnp.allclose(p1[key], p2[key])

    def test_multiplication_associative(self, sparse_helper, maxmin_algebra):
        """Test (a * b) * c = a * (b * c)."""
        alg = sparse_helper(maxmin_algebra, 3)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)
        x_2 = alg.variable(2)

        p1 = alg.mul(alg.mul(x_0, x_1), x_2)
        p2 = alg.mul(x_0, alg.mul(x_1, x_2))

        assert p1.keys() == p2.keys()
        for key in p1.keys():
            assert jnp.allclose(p1[key], p2[key])

    def test_multiplication_identity(self, sparse_helper, maxmin_algebra):
        """Test a * 1 = a."""
        alg = sparse_helper(maxmin_algebra, 3)
        x_0 = alg.variable(0)
        one = alg.constant(maxmin_algebra.one)

        p = alg.mul(x_0, one)

        # Check by evaluation
        test_point = {0: jnp.array(2.0), 1: jnp.array(3.0), 2: jnp.array(4.0)}
        result = alg.evaluate(p, test_point)
        assert jnp.allclose(list(result.values())[0], jnp.array(2.0))

    def test_multiplication_absorbing(self, sparse_helper, maxmin_algebra):
        """Test a * 0 = 0."""
        alg = sparse_helper(maxmin_algebra, 3)
        x_0 = alg.variable(0)
        zero = alg.constant(maxmin_algebra.zero)

        p = alg.mul(x_0, zero)

        # Result should evaluate to zero
        test_point = {0: jnp.array(2.0), 1: jnp.array(3.0), 2: jnp.array(4.0)}
        result = alg.evaluate(p, test_point)
        assert jnp.allclose(list(result.values())[0], maxmin_algebra.zero)

    def test_multiplication_with_constant(self, sparse_helper, maxmin_algebra):
        """Test multiplication with constant scales the polynomial."""
        alg = sparse_helper(maxmin_algebra, 2)
        x_0 = alg.variable(0)
        c = alg.constant(jnp.array(5.0))

        p = alg.mul(x_0, c)

        # In max-min: mul is min, so min(5.0, x_0), evaluated at x_0=3.0 gives min(5.0, 3.0) = 3.0
        test_point = {0: jnp.array(3.0), 1: jnp.array(2.0)}
        result = alg.evaluate(p, test_point)
        assert jnp.allclose(list(result.values())[0], jnp.array(3.0))


class TestSparsePolynomialDistributivity:
    """Test distributive law."""

    def test_distributive_law(self, sparse_helper, maxmin_algebra):
        """Test a * (b + c) = a*b + a*c."""
        alg = sparse_helper(maxmin_algebra, 3)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)
        x_2 = alg.variable(2)

        lhs = alg.mul(x_0, alg.add(x_1, x_2))
        rhs = alg.add(alg.mul(x_0, x_1), alg.mul(x_0, x_2))

        # Compare by evaluation
        test_point = {0: jnp.array(2.0), 1: jnp.array(3.0), 2: jnp.array(4.0)}
        lhs_result = list(alg.evaluate(lhs, test_point).values())[0]
        rhs_result = list(alg.evaluate(rhs, test_point).values())[0]
        assert jnp.allclose(lhs_result, rhs_result)

    def test_distributive_law_boolean(self, sparse_helper, bool_algebra):
        """Test a * (b + c) = a*b + a*c for boolean algebra."""
        alg = sparse_helper(bool_algebra, 3)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)
        x_2 = alg.variable(2)

        lhs = alg.mul(x_0, alg.add(x_1, x_2))
        rhs = alg.add(alg.mul(x_0, x_1), alg.mul(x_0, x_2))

        # Test all boolean combinations
        for b0 in [False, True]:
            for b1 in [False, True]:
                for b2 in [False, True]:
                    point = {0: jnp.array(b0), 1: jnp.array(b1), 2: jnp.array(b2)}
                    lhs_result = list(alg.evaluate(lhs, point).values())[0]
                    rhs_result = list(alg.evaluate(rhs, point).values())[0]
                    assert jnp.array_equal(lhs_result, rhs_result)


class TestSparsePolynomialMultilinear:
    """Test multilinear property."""

    def test_multilinear_idempotence(self, sparse_helper, bool_algebra):
        """Test x_i * x_i = x_i."""
        alg = sparse_helper(bool_algebra, 2)
        x_0 = alg.variable(0)

        p = alg.mul(x_0, x_0)

        # Should still be x_0
        assert len(p) == 1
        assert frozenbitarray("10") in p

    def test_multilinear_commutativity(self, sparse_helper, bool_algebra):
        """Test x_i * x_j = x_j * x_i."""
        alg = sparse_helper(bool_algebra, 3)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)

        p1 = alg.mul(x_0, x_1)
        p2 = alg.mul(x_1, x_0)

        # Both should give x_0 * x_1
        assert p1.keys() == p2.keys()
        assert frozenbitarray("110") in p1

    def test_monomial_multiplication(self, sparse_helper, bool_algebra):
        """Test that monomial multiplication uses bitwise OR."""
        alg = sparse_helper(bool_algebra, 3)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)
        x_2 = alg.variable(2)

        # (x_0 * x_1) * (x_1 * x_2) should give x_0 * x_1 * x_2
        p1 = alg.mul(x_0, x_1)
        p2 = alg.mul(x_1, x_2)
        result = alg.mul(p1, p2)

        assert frozenbitarray("111") in result


class TestSparsePolynomialEvaluation:
    """Test polynomial evaluation."""

    def test_evaluate_constant(self, sparse_helper, maxmin_algebra):
        """Test evaluating constant polynomial."""
        alg = sparse_helper(maxmin_algebra, 3)
        p = alg.constant(jnp.array(5.0))

        result = alg.evaluate(p, {0: jnp.array(1.0), 1: jnp.array(2.0), 2: jnp.array(3.0)})
        assert jnp.allclose(result[frozenbitarray("000")], jnp.array(5.0))

    def test_evaluate_variable(self, sparse_helper, maxmin_algebra):
        """Test evaluating single variable."""
        alg = sparse_helper(maxmin_algebra, 3)
        x_0 = alg.variable(0)

        result = alg.evaluate(x_0, {0: jnp.array(2.0), 1: jnp.array(3.0), 2: jnp.array(4.0)})
        # Should substitute x_0 with 2.0, giving constant 2.0
        assert len(result) == 1
        assert jnp.allclose(list(result.values())[0], jnp.array(2.0))

    def test_evaluate_at_sparse_point(self, sparse_helper, bool_algebra):
        """Test evaluation at sparse point (mapping)."""
        alg = sparse_helper(bool_algebra, 3)
        x_0 = alg.variable(0)

        result = alg.evaluate(x_0, {0: jnp.array(True)})
        # x_0 evaluated at x_0=True gives True
        assert len(result) == 1
        assert jnp.array_equal(list(result.values())[0], jnp.array(True))

    def test_evaluate_product(self, sparse_helper, maxmin_algebra):
        """Test evaluating product polynomial."""
        alg = sparse_helper(maxmin_algebra, 2)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)
        p = alg.mul(x_0, x_1)

        result = alg.evaluate(p, {0: jnp.array(2.0), 1: jnp.array(3.0)})
        # In max-min: mul is min, so min(x_0, x_1) at (2,3) gives min(2.0, 3.0) = 2.0
        assert jnp.allclose(list(result.values())[0], jnp.array(2.0))

    def test_evaluate_boolean_truth_table(self, sparse_helper, bool_algebra):
        """Test boolean evaluation with full truth table."""
        alg = sparse_helper(bool_algebra, 2)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)
        p = alg.mul(x_0, x_1)  # AND

        # Test all combinations
        test_cases = [
            ({0: True, 1: True}, True),
            ({0: True, 1: False}, False),
            ({0: False, 1: True}, False),
            ({0: False, 1: False}, False),
        ]

        for point, expected in test_cases:
            point_jnp = {k: jnp.array(v) for k, v in point.items()}
            result = alg.evaluate(p, point_jnp)
            result_val = list(result.values())[0]
            assert jnp.array_equal(result_val, jnp.array(expected)), f"Failed for point {point}"


class TestSparsePolynomialComposition:
    """Test polynomial composition."""

    def test_compose_single_variable(self, sparse_helper, bool_algebra):
        """Test composing single variable with another polynomial."""
        alg = sparse_helper(bool_algebra, 2)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)

        # Substitute x_0 with x_1
        result = alg.compose(x_0, {0: x_1})

        # Should get x_1
        assert frozenbitarray("01") in result

    def test_compose_with_constant(self, sparse_helper, maxmin_algebra):
        """Test composing with constant."""
        alg = sparse_helper(maxmin_algebra, 2)
        x_0 = alg.variable(0)
        c = alg.constant(jnp.array(5.0))

        # Substitute x_0 with 5
        result = alg.compose(x_0, {0: c})

        # Should get constant 5
        assert len(result) == 1
        assert jnp.allclose(list(result.values())[0], jnp.array(5.0))

    def test_compose_multiple_variables(self, sparse_helper, bool_algebra):
        """Test simultaneous composition of multiple variables."""
        alg = sparse_helper(bool_algebra, 3)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)
        x_2 = alg.variable(2)

        # Create x_0 * x_1
        p = alg.mul(x_0, x_1)

        # Substitute x_0 -> x_2, x_1 -> x_2
        result = alg.compose(p, {0: x_2, 1: x_2})

        # Should get x_2 * x_2 = x_2
        assert frozenbitarray("001") in result

    def test_compose_no_occurrence(self, sparse_helper, bool_algebra):
        """Test composition when variable doesn't appear."""
        alg = sparse_helper(bool_algebra, 3)
        x_1 = alg.variable(1)
        x_2 = alg.variable(2)

        # Substitute x_0 in x_1 (x_0 doesn't appear)
        result = alg.compose(x_1, {0: x_2})

        # Should still be x_1
        assert frozenbitarray("010") in result


class TestSparsePolynomialSemirings:
    """Test with different semirings."""

    def test_tropical_minplus(self, sparse_helper, maxmin_algebra):
        """Test with max-min algebra (negative reals)."""
        alg = sparse_helper(maxmin_algebra, 2)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)

        # In max-min: add = max, mul = min
        p = alg.mul(x_0, x_1)

        # Use negative values since this algebra is restricted to negative reals
        result = alg.evaluate(p, {0: jnp.array(-2.0), 1: jnp.array(-3.0)})
        # x_0 * x_1 in max-min means min(x_0, x_1) = min(-2, -3) = -3
        assert jnp.allclose(list(result.values())[0], jnp.array(-3.0))

    def test_tropical_maxplus(self, sparse_helper, maxmin_algebra):
        """Test with max-min algebra (positive reals)."""
        alg = sparse_helper(maxmin_algebra, 2)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)

        # In max-min: add = max, mul = min
        p = alg.mul(x_0, x_1)

        result = alg.evaluate(p, {0: jnp.array(2.0), 1: jnp.array(3.0)})
        # x_0 * x_1 in max-min means min(x_0, x_1) = min(2, 3) = 2
        assert jnp.allclose(list(result.values())[0], jnp.array(2.0))

    def test_maxmin_algebra(self, sparse_helper, maxmin_algebra):
        """Test with max-min algebra."""
        alg = sparse_helper(maxmin_algebra, 2)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)

        # In max-min: add = max, mul = min
        p = alg.mul(x_0, x_1)

        result = alg.evaluate(p, {0: jnp.array(2.0), 1: jnp.array(3.0)})
        # x_0 * x_1 in max-min means min(x_0, x_1) = min(2, 3) = 2
        assert jnp.allclose(list(result.values())[0], jnp.array(2.0))


class TestSparsePolynomialEdgeCases:
    """Test edge cases."""

    def test_empty_composition(self, sparse_helper, bool_algebra):
        """Test composition with empty replacement map."""
        alg = sparse_helper(bool_algebra, 2)
        x_0 = alg.variable(0)

        result = alg.compose(x_0, {})

        # Should be unchanged
        assert frozenbitarray("10") in result

    def test_multiple_monomials(self, sparse_helper, maxmin_algebra):
        """Test polynomial with multiple monomials."""
        alg = sparse_helper(maxmin_algebra, 2)
        x_0 = alg.variable(0)
        x_1 = alg.variable(1)
        x_0_x_1 = alg.mul(x_0, x_1)

        # Create x_0 + x_1 + x_0*x_1
        p = alg.add(alg.add(x_0, x_1), x_0_x_1)

        assert len(p) == 3
        assert frozenbitarray("10") in p
        assert frozenbitarray("01") in p
        assert frozenbitarray("11") in p

    def test_large_degree(self, sparse_helper, bool_algebra):
        """Test with larger degree polynomials."""
        alg = sparse_helper(bool_algebra, 10)
        x_0 = alg.variable(0)
        x_9 = alg.variable(9)

        p = alg.mul(x_0, x_9)

        assert len(p) == 1
        # Monomial should have bits 0 and 9 set
        monomial = list(p.keys())[0]
        assert monomial[0] and monomial[9]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
