"""Comprehensive tests for PolyDict implementation (baseline)."""

from __future__ import annotations

import pytest
from algebraic.polynomials.dok import PolyDict
from algebraic.spec import BooleanAlgebra, DeMorganAlgebra
from algebraic.utils.testing import assert_close, assert_equal
from bitarray import frozenbitarray


class TestPolyDictConstruction:
    """Test basic polynomial construction."""

    def test_constant_creation(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test constant polynomial creation."""
        num_vars = 3
        p = PolyDict.constant((True), num_vars, algebra=bool_algebra, backend=backend)

        assert len(p) == 1
        assert frozenbitarray("000") in p
        assert_equal(p[frozenbitarray("000")], (True))

    def test_variable_creation(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test single variable polynomial creation."""
        num_vars = 3
        x_0 = PolyDict.variable(0, num_vars, algebra=bool_algebra, backend=backend)

        assert len(x_0) == 1
        assert frozenbitarray("100") in x_0
        assert_equal(x_0[frozenbitarray("100")], (True))

    def test_variable_with_coefficient(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test variable creation with custom coefficient."""
        num_vars = 2
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend) * PolyDict.constant(
            (5.0), num_vars, algebra=maxmin_algebra, backend=backend
        )

        assert len(x_0) == 1
        assert_equal(x_0[frozenbitarray("10")], (5.0))

    def test_zero_polynomial(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test zero polynomial (constant with zero value)."""
        num_vars = 3
        p = PolyDict.constant(bool_algebra.zero, num_vars, algebra=bool_algebra, backend=backend)

        assert len(p) == 1
        assert_equal(p[frozenbitarray("000")], bool_algebra.zero)

    def test_one_polynomial(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test one polynomial (multiplicative identity)."""
        num_vars = 3
        p = PolyDict.constant(bool_algebra.one, num_vars, algebra=bool_algebra, backend=backend)

        assert len(p) == 1
        assert_equal(p[frozenbitarray("000")], bool_algebra.one)


class TestPolyDictAddition:
    """Test polynomial addition."""

    def test_addition_simple(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test adding two variables."""
        num_vars = 2
        x_0 = PolyDict.variable(0, num_vars, algebra=bool_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=bool_algebra, backend=backend)

        # In boolean algebra, add is OR
        p = x_0 + x_1

        # Should have two terms
        assert len(p) == 2
        assert frozenbitarray("10") in p
        assert frozenbitarray("01") in p

    def test_addition_commutative(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test a + b = b + a."""
        num_vars = 3
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=maxmin_algebra, backend=backend)

        p1 = x_0 + x_1
        p2 = x_1 + x_0

        # Compare by checking coefficients
        assert p1.keys() == p2.keys()
        for key in p1.keys():
            assert_equal(p1[key], p2[key])

    def test_addition_associative(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test (a + b) + c = a + (b + c)."""
        num_vars = 3
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=maxmin_algebra, backend=backend)
        x_2 = PolyDict.variable(2, num_vars, algebra=maxmin_algebra, backend=backend)

        p1 = (x_0 + x_1) + x_2
        p2 = x_0 + (x_1 + x_2)

        # Should have same terms
        assert p1.keys() == p2.keys()
        for key in p1.keys():
            assert_close(p1[key], p2[key])

    def test_addition_identity(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test a + 0 = a."""
        num_vars = 3
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        zero = PolyDict.constant(maxmin_algebra.zero, num_vars, algebra=maxmin_algebra, backend=backend)

        p = x_0 + zero

        # Should still be just x_0 (but might have zero term)
        # Check by evaluation
        test_point = {0: (2.0), 1: (3.0), 2: (4.0)}
        result = p.evaluate(test_point)
        assert_close(list(result.values())[0], (2.0))

    def test_addition_same_monomial(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test that adding same monomial combines coefficients."""
        num_vars = 2
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend) * PolyDict.constant(
            (2.0), num_vars, algebra=maxmin_algebra, backend=backend
        )
        x_0_again = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend) * PolyDict.constant(
            (3.0), num_vars, algebra=maxmin_algebra, backend=backend
        )

        p = x_0 + x_0_again

        # In tropical max-plus: add is max, so max(2.0, 3.0) = 3.0
        assert frozenbitarray("10") in p
        assert_close(p[frozenbitarray("10")], (3.0))


class TestPolyDictMultiplication:
    """Test polynomial multiplication."""

    def test_multiplication_simple(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test multiplying two variables."""
        num_vars = 2
        x_0 = PolyDict.variable(0, num_vars, algebra=bool_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=bool_algebra, backend=backend)

        # x_0 * x_1 in Boolean algebra is x_0 AND x_1
        p = x_0 * x_1

        assert len(p) == 1
        assert frozenbitarray("11") in p

    def test_multiplication_commutative(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test a * b = b * a for commutative semirings."""
        num_vars = 3
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=maxmin_algebra, backend=backend)

        p1 = x_0 * x_1
        p2 = x_1 * x_0

        assert p1.keys() == p2.keys()
        for key in p1.keys():
            assert_close(p1[key], p2[key])

    def test_multiplication_associative(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test (a * b) * c = a * (b * c)."""
        num_vars = 3
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=maxmin_algebra, backend=backend)
        x_2 = PolyDict.variable(2, num_vars, algebra=maxmin_algebra, backend=backend)

        p1 = (x_0 * x_1) * x_2
        p2 = x_0 * (x_1 * x_2)

        assert p1.keys() == p2.keys()
        for key in p1.keys():
            assert_close(p1[key], p2[key])

    def test_multiplication_identity(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test a * 1 = a."""
        num_vars = 3
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        one = PolyDict.constant(maxmin_algebra.one, num_vars, algebra=maxmin_algebra, backend=backend)

        p = x_0 * one

        # Check by evaluation
        test_point = {0: (2.0), 1: (3.0), 2: (4.0)}
        result = p.evaluate(test_point)
        assert_close(list(result.values())[0], (2.0))

    def test_multiplication_absorbing(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test a * 0 = 0."""
        num_vars = 3
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        zero = PolyDict.constant(maxmin_algebra.zero, num_vars, algebra=maxmin_algebra, backend=backend)

        p = x_0 * zero

        # Result should evaluate to zero
        test_point = {0: (2.0), 1: (3.0), 2: (4.0)}
        result = p.evaluate(test_point)
        assert_close(list(result.values())[0], maxmin_algebra.zero)

    def test_multiplication_with_constant(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test multiplication with constant scales the polynomial."""
        num_vars = 2
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        c = PolyDict.constant((5.0), num_vars, algebra=maxmin_algebra, backend=backend)

        p = x_0 * c

        # In max-min: mul is min, so min(5.0, x_0), evaluated at x_0=3.0 gives min(5.0, 3.0) = 3.0
        test_point = {0: (3.0), 1: (2.0)}
        result = p.evaluate(test_point)
        assert_close(list(result.values())[0], (3.0))


class TestPolyDictDistributivity:
    """Test distributive law."""

    def test_distributive_law(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test a * (b + c) = a*b + a*c."""
        num_vars = 3
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=maxmin_algebra, backend=backend)
        x_2 = PolyDict.variable(2, num_vars, algebra=maxmin_algebra, backend=backend)

        lhs = x_0 * (x_1 + x_2)
        rhs = (x_0 * x_1) + (x_0 * x_2)

        # Compare by evaluation
        test_point = {0: (2.0), 1: (3.0), 2: (4.0)}
        lhs_result = list(lhs.evaluate(test_point).values())[0]
        rhs_result = list(rhs.evaluate(test_point).values())[0]
        assert_close(lhs_result, rhs_result)

    def test_distributive_law_boolean(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test a * (b + c) = a*b + a*c for boolean algebra."""
        num_vars = 3
        x_0 = PolyDict.variable(0, num_vars, algebra=bool_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=bool_algebra, backend=backend)
        x_2 = PolyDict.variable(2, num_vars, algebra=bool_algebra, backend=backend)

        lhs = x_0 * (x_1 + x_2)
        rhs = (x_0 * x_1) + (x_0 * x_2)

        # Test all boolean combinations
        for b0 in [False, True]:
            for b1 in [False, True]:
                for b2 in [False, True]:
                    point = {0: (b0), 1: (b1), 2: (b2)}
                    lhs_result = list(lhs.evaluate(point).values())[0]
                    rhs_result = list(rhs.evaluate(point).values())[0]
                    assert_equal(lhs_result, rhs_result)


class TestPolyDictMultilinear:
    """Test multilinear property."""

    def test_multilinear_idempotence(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test x_i * x_i = x_i."""
        num_vars = 2
        x_0 = PolyDict.variable(0, num_vars, algebra=bool_algebra, backend=backend)

        p = x_0 * x_0

        # Should still be x_0
        assert len(p) == 1
        assert frozenbitarray("10") in p

    def test_multilinear_commutativity(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test x_i * x_j = x_j * x_i."""
        num_vars = 3
        x_0 = PolyDict.variable(0, num_vars, algebra=bool_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=bool_algebra, backend=backend)

        p1 = x_0 * x_1
        p2 = x_1 * x_0

        # Both should give x_0 * x_1
        assert p1.keys() == p2.keys()
        assert frozenbitarray("110") in p1

    def test_monomial_multiplication(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test that monomial multiplication uses bitwise OR."""
        num_vars = 3
        x_0 = PolyDict.variable(0, num_vars, algebra=bool_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=bool_algebra, backend=backend)
        x_2 = PolyDict.variable(2, num_vars, algebra=bool_algebra, backend=backend)

        # (x_0 * x_1) * (x_1 * x_2) should give x_0 * x_1 * x_2
        p1 = x_0 * x_1
        p2 = x_1 * x_2
        result = p1 * p2

        assert frozenbitarray("111") in result


class TestPolyDictEvaluation:
    """Test polynomial evaluation."""

    def test_evaluate_constant(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test evaluating constant polynomial."""
        num_vars = 3
        p = PolyDict.constant((5.0), num_vars, algebra=maxmin_algebra, backend=backend)

        result = p.evaluate({0: (1.0), 1: (2.0), 2: (3.0)})
        assert_close(result[frozenbitarray("000")], (5.0))

    def test_evaluate_variable(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test evaluating single variable."""
        num_vars = 3
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)

        result = x_0.evaluate({0: (2.0), 1: (3.0), 2: (4.0)})
        # Should substitute x_0 with 2.0, giving constant 2.0
        assert len(result) == 1
        assert_close(list(result.values())[0], (2.0))

    def test_evaluate_at_sparse_point(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test evaluation at sparse point (mapping)."""
        num_vars = 3
        x_0 = PolyDict.variable(0, num_vars, algebra=bool_algebra, backend=backend)

        result = x_0.evaluate({0: (True)})
        # x_0 evaluated at x_0=True gives True
        assert len(result) == 1
        assert_equal(list(result.values())[0], (True))

    def test_evaluate_product(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test evaluating product polynomial."""
        num_vars = 2
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=maxmin_algebra, backend=backend)
        p = x_0 * x_1

        result = p.evaluate({0: (2.0), 1: (3.0)})
        # In max-min: mul is min, so min(x_0, x_1) at (2,3) gives min(2.0, 3.0) = 2.0
        assert_close(list(result.values())[0], (2.0))

    def test_evaluate_boolean_truth_table(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test boolean evaluation with full truth table."""
        num_vars = 2
        x_0 = PolyDict.variable(0, num_vars, algebra=bool_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=bool_algebra, backend=backend)
        p = x_0 * x_1  # AND

        # Test all combinations
        test_cases = [
            ({0: True, 1: True}, True),
            ({0: True, 1: False}, False),
            ({0: False, 1: True}, False),
            ({0: False, 1: False}, False),
        ]

        for point, expected in test_cases:
            point_jnp = {k: (v) for k, v in point.items()}
            result = p.evaluate(point_jnp)
            result_val = list(result.values())[0]
            assert_equal(result_val, (expected)), f"Failed for point {point}"


class TestPolyDictComposition:
    """Test polynomial composition."""

    def test_compose_single_variable(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test composing single variable with another polynomial."""
        num_vars = 2
        x_0 = PolyDict.variable(0, num_vars, algebra=bool_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=bool_algebra, backend=backend)

        # Substitute x_0 with x_1
        result = x_0.compose({0: x_1})

        # Should get x_1
        assert frozenbitarray("01") in result

    def test_compose_with_constant(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test composing with constant."""
        num_vars = 2
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        c = PolyDict.constant((5.0), num_vars, algebra=maxmin_algebra, backend=backend)

        # Substitute x_0 with 5
        result = x_0.compose({0: c})

        # Should get constant 5
        assert len(result) == 1
        assert_close(list(result.values())[0], (5.0))

    def test_compose_multiple_variables(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test simultaneous composition of multiple variables."""
        num_vars = 3
        x_0 = PolyDict.variable(0, num_vars, algebra=bool_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=bool_algebra, backend=backend)
        x_2 = PolyDict.variable(2, num_vars, algebra=bool_algebra, backend=backend)

        # Create x_0 * x_1
        p = x_0 * x_1

        # Substitute x_0 -> x_2, x_1 -> x_2
        result = p.compose({0: x_2, 1: x_2})

        # Should get x_2 * x_2 = x_2
        assert frozenbitarray("001") in result

    def test_compose_no_occurrence(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test composition when variable doesn't appear."""
        num_vars = 3
        x_1 = PolyDict.variable(1, num_vars, algebra=bool_algebra, backend=backend)
        x_2 = PolyDict.variable(2, num_vars, algebra=bool_algebra, backend=backend)

        # Substitute x_0 in x_1 (x_0 doesn't appear)
        result = x_1.compose({0: x_2})

        # Should still be x_1
        assert frozenbitarray("010") in result


class TestPolyDictSemirings:
    """Test with different semirings."""

    def test_tropical_minplus(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test with max-min algebra (negative reals)."""
        num_vars = 2
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=maxmin_algebra, backend=backend)

        # In max-min: add = max, mul = min
        p = x_0 * x_1

        # Use negative values since this algebra is restricted to negative reals
        result = p.evaluate({0: (-2.0), 1: (-3.0)})
        # x_0 * x_1 in max-min means min(x_0, x_1) = min(-2, -3) = -3
        assert_close(list(result.values())[0], (-3.0))

    def test_tropical_maxplus(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test with max-min algebra (positive reals)."""
        num_vars = 2
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=maxmin_algebra, backend=backend)

        # In max-min: add = max, mul = min
        p = x_0 * x_1

        result = p.evaluate({0: (2.0), 1: (3.0)})
        # x_0 * x_1 in max-min means min(x_0, x_1) = min(2, 3) = 2
        assert_close(list(result.values())[0], (2.0))

    def test_maxmin_algebra(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test with max-min algebra."""
        num_vars = 2
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=maxmin_algebra, backend=backend)

        # In max-min: add = max, mul = min
        p = x_0 * x_1

        result = p.evaluate({0: (2.0), 1: (3.0)})
        # x_0 * x_1 in max-min means min(x_0, x_1) = min(2, 3) = 2
        assert_close(list(result.values())[0], (2.0))


class TestPolyDictEdgeCases:
    """Test edge cases."""

    def test_empty_composition(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test composition with empty replacement map."""
        num_vars = 2
        x_0 = PolyDict.variable(0, num_vars, algebra=bool_algebra, backend=backend)

        result = x_0.compose({})

        # Should be unchanged
        assert frozenbitarray("10") in result

    def test_multiple_monomials(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test polynomial with multiple monomials."""
        num_vars = 2
        x_0 = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=maxmin_algebra, backend=backend)
        x_0_x_1 = x_0 * x_1

        # Create x_0 + x_1 + x_0*x_1
        p = (x_0 + x_1) + x_0_x_1

        assert len(p) == 3
        assert frozenbitarray("10") in p
        assert frozenbitarray("01") in p
        assert frozenbitarray("11") in p

    def test_large_degree(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test with larger degree polynomials."""
        num_vars = 10
        x_0 = PolyDict.variable(0, num_vars, algebra=bool_algebra, backend=backend)
        x_9 = PolyDict.variable(9, num_vars, algebra=bool_algebra, backend=backend)

        p = x_0 * x_9

        assert len(p) == 1
        # Monomial should have bits 0 and 9 set
        monomial = list(p.keys())[0]
        assert monomial[0] and monomial[9]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
