"""Tests for MonomialBasis representation against PolyDict baseline."""

# ruff: noqa: ANN001
# mypy: disable-error-code="no-untyped-call,no-untyped-def,import-not-found"

from __future__ import annotations

import algebraic
import hypothesis
import numpy as np
import pytest
from algebraic import AlgebraicArray, BooleanAlgebra, DeMorganAlgebra
from algebraic.polynomials.dok import PolyDict
from algebraic.polynomials.monomial_basis import MonomialBasis
from algebraic.utils.testing import assert_allclose, assert_close, assert_equal, make_array
from bitarray import frozenbitarray
from hypothesis import given, settings
from hypothesis import strategies as st


class TestMonomialBasisConversion:
    """Test conversion between sparse and monomial basis representations."""

    def test_from_sparse_constant(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test converting constant from sparse to monomial basis."""
        # Create constant in sparse form
        p_sparse = PolyDict.constant(bool_algebra.one, 3, algebra=bool_algebra, backend=backend)

        # Convert to monomial basis
        p_monomial = MonomialBasis.from_sparse(p_sparse, backend=backend)

        # Check shape
        assert p_monomial.shape == (2, 2, 2)

        # Check that constant term is correct (use quaxified array_equal)
        assert_equal(p_monomial.coeffs[0, 0, 0].data, bool_algebra.one)

    def test_from_sparse_variable(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test converting variable from sparse to monomial basis."""
        num_vars = 3

        # Create variable in sparse form
        x_0 = PolyDict.variable(0, num_vars, algebra=bool_algebra, backend=backend)

        # Convert to monomial basis
        x_0_monomial = MonomialBasis.from_sparse(x_0, backend=backend)

        # Check that variable term is correct
        assert_equal(x_0_monomial.coeffs[1, 0, 0], bool_algebra.one)

    def test_to_sparse_from_monomial(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test converting monomial basis back to sparse."""
        num_vars = 2

        # Create variable x_0
        x_0 = MonomialBasis.variable(0, num_vars, bool_algebra, backend=backend)
        # Convert to sparse
        x_0_sparse = MonomialBasis.to_sparse(x_0)

        # Check that we have the right monomial
        assert len(x_0_sparse) == 1
        assert frozenbitarray("10") in x_0_sparse

    def test_round_trip_conversion(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test sparse -> monomial -> sparse preserves polynomial."""
        num_vars = 2

        # Create polynomial x_0 + x_1
        x_0 = PolyDict.variable(0, num_vars, algebra=bool_algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=bool_algebra, backend=backend)
        p_sparse = x_0 + x_1

        # Convert to monomial and back
        p_monomial = MonomialBasis.from_sparse(p_sparse, backend=backend)
        p_back = MonomialBasis.to_sparse(p_monomial)

        # Check we have the same monomials
        assert p_sparse.keys() == p_back.keys()
        for key in p_sparse.keys():
            assert_equal(p_sparse[key], p_back[key])


class TestMonomialBasisAddition:
    """Test that addition matches sparse representation."""

    def test_add_variables(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test adding two variables."""
        num_vars = 3
        algebra = maxmin_algebra

        # Create in both representations
        x_0_sparse = PolyDict.variable(0, num_vars, algebra=algebra, backend=backend)
        x_1_sparse = PolyDict.variable(1, num_vars, algebra=algebra, backend=backend)
        x_0_monomial = MonomialBasis.from_sparse(x_0_sparse, backend=backend)
        x_1_monomial = MonomialBasis.from_sparse(x_1_sparse, backend=backend)

        # Add in both
        sum_sparse = x_0_sparse + x_1_sparse
        sum_monomial = x_0_monomial + x_1_monomial

        # Convert monomial result back to sparse
        sum_monomial_as_sparse = MonomialBasis.to_sparse(sum_monomial)

        # Compare
        assert sum_sparse.keys() == sum_monomial_as_sparse.keys()
        for key in sum_sparse.keys():
            assert_allclose(sum_sparse[key], sum_monomial_as_sparse[key])

    # @pytest.mark.skip(reason="Too slow - hypothesis test disabled temporarily")
    @given(degree=st.integers(2, 3))
    @settings(max_examples=2, deadline=None, suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture])
    def test_add_with_hypothesis(self, degree: int, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test addition with random polynomials using hypothesis."""

        num_vars = degree
        algebra = maxmin_algebra

        # Create two random simple polynomials
        x_0 = PolyDict.variable(0, num_vars, algebra=algebra, backend=backend)
        x_1 = (
            PolyDict.variable(1, num_vars, algebra=algebra, backend=backend)
            if degree > 1
            else PolyDict.variable(0, num_vars, algebra=algebra, backend=backend)
        )

        p1_sparse = x_0 + PolyDict.constant(2.0, num_vars, algebra=algebra, backend=backend)
        p2_sparse = x_1 + PolyDict.constant(3.0, num_vars, algebra=algebra, backend=backend)

        # Convert to monomial
        p1_monomial = MonomialBasis.from_sparse(p1_sparse, backend=backend)
        p2_monomial = MonomialBasis.from_sparse(p2_sparse, backend=backend)

        # Add in both representations
        sum_sparse = p1_sparse + p2_sparse
        sum_monomial = p1_monomial + p2_monomial

        # Test by evaluation at a random point
        rng = np.random.default_rng(42)
        point_np = rng.uniform(-5.0, 5.0, size=(degree,))
        point_dict = {i: float(point_np[i]) for i in range(degree)}
        point = make_array(point_np, backend)

        # Evaluate sparse
        result_sparse = sum_sparse.evaluate(point_dict)
        assert isinstance(result_sparse, PolyDict)

        # Evaluate monomial
        result_monomial = MonomialBasis.evaluate(sum_monomial, point)
        assert isinstance(result_monomial, MonomialBasis)
        # Convert to sparse repr and check equality
        sparse_repr = MonomialBasis.to_sparse(result_monomial)

        assert set(result_sparse.keys()) == set(sparse_repr.keys())
        for monom in result_sparse.keys():
            assert_allclose(result_sparse[monom], sparse_repr[monom])


class TestMonomialBasisMultiplication:
    """Test that multiplication matches sparse representation."""

    def test_multiply_variables(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test multiplying two variables."""
        num_vars = 2
        algebra = bool_algebra

        # Create in both representations
        x_0_sparse = PolyDict.variable(0, num_vars, algebra=algebra, backend=backend)
        x_1_sparse = PolyDict.variable(1, num_vars, algebra=algebra, backend=backend)
        x_0_monomial = MonomialBasis.from_sparse(x_0_sparse, backend=backend)
        x_1_monomial = MonomialBasis.from_sparse(x_1_sparse, backend=backend)

        # Multiply in both
        prod_sparse = x_0_sparse * x_1_sparse
        prod_monomial = x_0_monomial * x_1_monomial

        # Convert monomial result back to sparse
        prod_monomial_as_sparse = MonomialBasis.to_sparse(prod_monomial)

        # Compare
        assert prod_sparse.keys() == prod_monomial_as_sparse.keys()
        for key in prod_sparse.keys():
            assert_equal(prod_sparse[key], prod_monomial_as_sparse[key])

    def test_multiply_with_constant(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test multiplication with constant."""
        num_vars = 2
        algebra = maxmin_algebra

        # Create variable and constant
        x_0_sparse = PolyDict.variable(0, num_vars, algebra=algebra, backend=backend)
        c_sparse = PolyDict.constant(5.0, num_vars, algebra=algebra, backend=backend)

        x_0_monomial = MonomialBasis.from_sparse(x_0_sparse, backend=backend)
        c_monomial = MonomialBasis.from_sparse(c_sparse, backend=backend)

        # Multiply
        prod_sparse = x_0_sparse * c_sparse
        prod_monomial = x_0_monomial * c_monomial

        # Compare by evaluation
        test_point = {0: 3.0, 1: 2.0}

        result_sparse, *_ = prod_sparse.evaluate(test_point).values()
        _result_monomial = prod_monomial.coeffs[tuple([0] * 2)]  # Constant term after evaluation

        # Evaluate monomial at the test point
        point_array = make_array([3.0, 2.0], backend)
        result_monomial_eval = MonomialBasis.evaluate(prod_monomial, point_array)
        result_monomial_val = result_monomial_eval.coeffs[0, 0].data

        assert_close(result_sparse, result_monomial_val)

    # @pytest.mark.skip(reason="Too slow - hypothesis test disabled temporarily")
    @given(degree=st.integers(2, 3))
    @settings(max_examples=2, deadline=None, suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture])
    def test_multiply_with_hypothesis(self, degree: int, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test multiplication with random polynomials."""

        num_vars = degree
        algebra = maxmin_algebra

        # Create two simple polynomials
        x_0 = PolyDict.variable(0, num_vars, algebra=algebra, backend=backend)
        x_1 = (
            PolyDict.variable(1, num_vars, algebra=algebra, backend=backend)
            if degree > 1
            else PolyDict.variable(0, num_vars, algebra=algebra, backend=backend)
        )

        # Convert to monomial
        x_0_monomial = MonomialBasis.from_sparse(x_0, backend=backend)
        x_1_monomial = MonomialBasis.from_sparse(x_1, backend=backend)

        # Multiply
        prod_sparse = x_0 * x_1
        prod_monomial = x_0_monomial * x_1_monomial

        # Test by evaluation
        rng = np.random.default_rng(123)
        point_np = rng.uniform(-5.0, 5.0, size=(degree,))
        point_dict = {i: float(point_np[i]) for i in range(degree)}
        point = make_array(point_np, backend)

        result_sparse = PolyDict.evaluate(prod_sparse, point_dict)
        assert isinstance(result_sparse, PolyDict)
        assert len(result_sparse) == 1
        assert list(result_sparse.keys())[0] == frozenbitarray(degree)
        sparse_value = result_sparse[frozenbitarray(degree)]

        result_monomial = MonomialBasis.evaluate(prod_monomial, point)
        assert isinstance(result_monomial, MonomialBasis)
        monomial_value = result_monomial.coeffs[tuple([0] * degree)]
        assert isinstance(monomial_value, AlgebraicArray)

        assert_allclose(sparse_value, monomial_value, rtol=1e-5)


class TestMonomialBasisEvaluation:
    """Test that evaluation matches sparse representation."""

    def test_evaluate_variable(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test evaluating a variable."""
        num_vars = 3
        algebra = maxmin_algebra

        # Create variable
        x_0_sparse = PolyDict.variable(0, num_vars, algebra=algebra, backend=backend)
        x_0_monomial = MonomialBasis.from_sparse(x_0_sparse, backend=backend)

        # Evaluate at a point
        point_dict = {0: 2.0, 1: 3.0, 2: 4.0}
        point_array = make_array([2.0, 3.0, 4.0], backend)

        result_sparse = list(PolyDict.evaluate(x_0_sparse, point_dict).values())[0]
        result_monomial = MonomialBasis.evaluate(x_0_monomial, point_array)

        # After evaluating all variables, we should have a constant
        monomial_value = result_monomial.coeffs[0, 0, 0].data

        assert_close(result_sparse, monomial_value)

    def test_evaluate_product(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test evaluating a product."""
        num_vars = 2
        algebra = maxmin_algebra

        # Create x_0 * x_1
        x_0 = PolyDict.variable(0, num_vars, algebra=algebra, backend=backend)
        x_1 = PolyDict.variable(1, num_vars, algebra=algebra, backend=backend)
        p_sparse = x_0 * x_1
        p_monomial = MonomialBasis.from_sparse(p_sparse, backend=backend)

        # Evaluate
        point_dict = {0: 2.0, 1: 3.0}
        point_array = make_array([2.0, 3.0], backend)

        result_sparse = list(PolyDict.evaluate(p_sparse, point_dict).values())[0]
        result_monomial = MonomialBasis.evaluate(p_monomial, point_array)
        monomial_value = result_monomial.coeffs[0, 0].data

        assert_close(result_sparse, monomial_value)


class TestMonomialBasisCompose:
    """Test that composition matches sparse representation."""

    def test_compose_simple(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test simple composition."""
        num_vars = 2
        algebra = bool_algebra

        # Create x_0 and x_1
        x_0_sparse = PolyDict.variable(0, num_vars, algebra=algebra, backend=backend)
        x_1_sparse = PolyDict.variable(1, num_vars, algebra=algebra, backend=backend)

        x_0_monomial = MonomialBasis.from_sparse(x_0_sparse, backend=backend)
        x_1_monomial = MonomialBasis.from_sparse(x_1_sparse, backend=backend)

        # Compose x_0 with x_1 (replace x_0 with x_1)
        result_sparse = PolyDict.compose(x_0_sparse, {0: x_1_sparse})
        result_monomial = MonomialBasis.compose(x_0_monomial, {0: x_1_monomial})

        # Convert monomial result to sparse
        result_monomial_as_sparse = MonomialBasis.to_sparse(result_monomial)

        # Compare - filter out zero coefficients for comparison
        # (sparse representation may include explicit zeros)

        # Get non-zero keys from result_sparse
        nonzero_keys_sparse = {k for k, v in result_sparse.items() if not algebraic.equal(v, bool_algebra.zero)}
        nonzero_keys_monomial = set(result_monomial_as_sparse.keys())

        assert nonzero_keys_sparse == nonzero_keys_monomial
        for key in nonzero_keys_sparse:
            assert_equal(result_sparse[key], result_monomial_as_sparse[key])


class TestMonomialBasisEdgeCases:
    """Test edge cases for monomial basis."""

    def test_zero_polynomial(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test zero polynomial."""
        algebra = maxmin_algebra
        num_vars = 2

        zero = MonomialBasis.zero(num_vars, algebra=algebra, backend=backend)

        # In tropical max-plus, zero = -inf
        expected = make_array(np.full((2, 2), maxmin_algebra.zero), backend)
        assert_close(zero.coeffs, expected)

    def test_large_degree(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test with larger degree (but still manageable)."""
        num_vars = 6
        algebra = bool_algebra
        # 2^6 = 64 entries, should be fine

        x_0 = PolyDict.variable(0, num_vars, algebra=algebra, backend=backend)
        x_0_monomial = MonomialBasis.from_sparse(x_0, backend=backend)

        assert x_0_monomial.shape == (2,) * 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
