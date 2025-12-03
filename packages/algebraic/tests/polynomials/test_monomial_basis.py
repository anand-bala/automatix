"""Tests for MonomialBasis representation against SparsePolynomial baseline."""

# ruff: noqa: ANN201, ANN001

from __future__ import annotations

import jax.numpy as jnp
import pytest
from algebraic.polynomials.jax import MonomialBasis, MonomialBasisAlgebra
from algebraic.polynomials.sparse import SparsePolynomial, SparsePolynomialAlgebra
from algebraic.tensor_algebra.jax import max_min_algebra
from bitarray import frozenbitarray
from hypothesis import given, settings
from hypothesis import strategies as st


class TestMonomialBasisConversion:
    """Test conversion between sparse and monomial basis representations."""

    def test_from_sparse_constant(self, sparse_alg_factory, monomial_alg_factory, bool_algebra, bool_module):
        """Test converting constant from sparse to monomial basis."""
        sparse_alg = sparse_alg_factory(bool_algebra, 3)
        monomial_alg = monomial_alg_factory(bool_module, 3)

        # Create constant in sparse form
        p_sparse = sparse_alg.constant(jnp.array(True))

        # Convert to monomial basis
        p_monomial = monomial_alg.from_sparse(p_sparse)

        # Check shape
        assert p_monomial.shape == (2, 2, 2)

        # Check that constant term is correct
        assert jnp.array_equal(p_monomial.coeffs[0, 0, 0], jnp.array(True))

    def test_from_sparse_variable(self, sparse_alg_factory, monomial_alg_factory, bool_algebra, bool_module):
        """Test converting variable from sparse to monomial basis."""
        sparse_alg = sparse_alg_factory(bool_algebra, 3)
        monomial_alg = monomial_alg_factory(bool_module, 3)

        # Create variable in sparse form
        x_0 = sparse_alg.variable(0)

        # Convert to monomial basis
        x_0_monomial = monomial_alg.from_sparse(x_0)

        # Check that variable term is correct
        assert jnp.array_equal(x_0_monomial.coeffs[1, 0, 0], jnp.array(True))

    def test_to_sparse_from_monomial(self, monomial_alg_factory, bool_module):
        """Test converting monomial basis back to sparse."""
        monomial_alg = monomial_alg_factory(bool_module, 2)

        # Create variable x_0
        x_0 = monomial_alg.variable(0)

        # Convert to sparse
        x_0_sparse = monomial_alg.to_sparse(x_0)

        # Check that we have the right monomial
        assert len(x_0_sparse) == 1
        assert frozenbitarray("10") in x_0_sparse

    def test_round_trip_conversion(self, sparse_alg_factory, monomial_alg_factory, bool_algebra, bool_module):
        """Test sparse -> monomial -> sparse preserves polynomial."""
        sparse_alg = sparse_alg_factory(bool_algebra, 2)
        monomial_alg = monomial_alg_factory(bool_module, 2)

        # Create polynomial x_0 + x_1
        x_0 = sparse_alg.variable(0)
        x_1 = sparse_alg.variable(1)
        p_sparse = sparse_alg.add(x_0, x_1)

        # Convert to monomial and back
        p_monomial = monomial_alg.from_sparse(p_sparse)
        p_back = monomial_alg.to_sparse(p_monomial)

        # Check we have the same monomials
        assert p_sparse.keys() == p_back.keys()
        for key in p_sparse.keys():
            assert jnp.array_equal(p_sparse[key], p_back[key])


class TestMonomialBasisAddition:
    """Test that addition matches sparse representation."""

    def test_add_variables(self, sparse_alg_factory, monomial_alg_factory, tropical_maxplus_algebra, tropical_maxplus_module):
        """Test adding two variables."""
        sparse_alg = sparse_alg_factory(tropical_maxplus_algebra, 3)
        monomial_alg = monomial_alg_factory(tropical_maxplus_module, 3)

        # Create in both representations
        x_0_sparse = sparse_alg.variable(0)
        x_1_sparse = sparse_alg.variable(1)
        x_0_monomial = monomial_alg.from_sparse(x_0_sparse)
        x_1_monomial = monomial_alg.from_sparse(x_1_sparse)

        # Add in both
        sum_sparse = sparse_alg.add(x_0_sparse, x_1_sparse)
        sum_monomial = monomial_alg.add(x_0_monomial, x_1_monomial)

        # Convert monomial result back to sparse
        sum_monomial_as_sparse = monomial_alg.to_sparse(sum_monomial)

        # Compare
        assert sum_sparse.keys() == sum_monomial_as_sparse.keys()
        for key in sum_sparse.keys():
            assert jnp.allclose(sum_sparse[key], sum_monomial_as_sparse[key])

    @pytest.mark.skip(reason="Too slow - hypothesis test disabled temporarily")
    @given(st.integers(2, 3))
    @settings(max_examples=2, deadline=500)
    def test_add_with_hypothesis(self, degree):
        """Test addition with random polynomials using hypothesis."""

        module = max_min_algebra()
        algebra = module.algebra
        sparse_alg = SparsePolynomialAlgebra(algebra=algebra, degree=degree)
        monomial_alg = MonomialBasisAlgebra(num_vars=degree, module=module)

        # Create two random simple polynomials
        x_0 = sparse_alg.variable(0)
        x_1 = sparse_alg.variable(1) if degree > 1 else sparse_alg.variable(0)

        p1_sparse = sparse_alg.add(x_0, sparse_alg.constant(jnp.array(2.0)))
        p2_sparse = sparse_alg.add(x_1, sparse_alg.constant(jnp.array(3.0)))

        # Convert to monomial
        p1_monomial = monomial_alg.from_sparse(p1_sparse)
        p2_monomial = monomial_alg.from_sparse(p2_sparse)

        # Add in both representations
        sum_sparse = sparse_alg.add(p1_sparse, p2_sparse)
        sum_monomial = monomial_alg.add(p1_monomial, p2_monomial)

        # Test by evaluation at a random point
        import jax.random as jrandom

        key = jrandom.PRNGKey(42)
        point = jrandom.uniform(key, shape=(degree,), minval=-5.0, maxval=5.0)
        point_dict = {i: point[i] for i in range(degree)}

        # Evaluate sparse
        result_sparse = sparse_alg.evaluate(sum_sparse, point_dict)
        assert isinstance(result_sparse, SparsePolynomial)

        # Evaluate monomial
        result_monomial = monomial_alg.evaluate(sum_monomial, point)
        assert isinstance(result_monomial, MonomialBasis)
        # Convert to sparse repr and check equality
        sparse_repr = monomial_alg.to_sparse(result_monomial)

        assert set(result_sparse.keys()) == set(sparse_repr.keys())
        for monom in result_sparse.keys():
            if not jnp.allclose(result_sparse[monom], sparse_repr[monom]):
                raise AssertionError(f"{result_sparse[monom]=} != {sparse_repr[monom]=} at {monom}")


class TestMonomialBasisMultiplication:
    """Test that multiplication matches sparse representation."""

    def test_multiply_variables(self, sparse_alg_factory, monomial_alg_factory, bool_algebra, bool_module):
        """Test multiplying two variables."""
        sparse_alg = sparse_alg_factory(bool_algebra, 2)
        monomial_alg = monomial_alg_factory(bool_module, 2)

        # Create in both representations
        x_0_sparse = sparse_alg.variable(0)
        x_1_sparse = sparse_alg.variable(1)
        x_0_monomial = monomial_alg.from_sparse(x_0_sparse)
        x_1_monomial = monomial_alg.from_sparse(x_1_sparse)

        # Multiply in both
        prod_sparse = sparse_alg.mul(x_0_sparse, x_1_sparse)
        prod_monomial = monomial_alg.mul(x_0_monomial, x_1_monomial)

        # Convert monomial result back to sparse
        prod_monomial_as_sparse = monomial_alg.to_sparse(prod_monomial)

        # Compare
        assert prod_sparse.keys() == prod_monomial_as_sparse.keys()
        for key in prod_sparse.keys():
            assert jnp.array_equal(prod_sparse[key], prod_monomial_as_sparse[key])

    def test_multiply_with_constant(self, sparse_alg_factory, monomial_alg_factory, tropical_maxplus_algebra, tropical_maxplus_module):
        """Test multiplication with constant."""
        sparse_alg = sparse_alg_factory(tropical_maxplus_algebra, 2)
        monomial_alg = monomial_alg_factory(tropical_maxplus_module, 2)

        # Create variable and constant
        x_0_sparse = sparse_alg.variable(0)
        c_sparse = sparse_alg.constant(jnp.array(5.0))

        x_0_monomial = monomial_alg.from_sparse(x_0_sparse)
        c_monomial = monomial_alg.from_sparse(c_sparse)

        # Multiply
        prod_sparse = sparse_alg.mul(x_0_sparse, c_sparse)
        prod_monomial = monomial_alg.mul(x_0_monomial, c_monomial)

        # Compare by evaluation
        test_point = {0: jnp.array(3.0), 1: jnp.array(2.0)}

        result_sparse = list(sparse_alg.evaluate(prod_sparse, test_point).values())[0]
        result_monomial = prod_monomial.coeffs[tuple([0] * 2)]  # Constant term after evaluation

        # Evaluate monomial at the test point
        point_array = jnp.array([3.0, 2.0])
        result_monomial_eval = monomial_alg.evaluate(prod_monomial, point_array)
        result_monomial_val = result_monomial_eval.coeffs[0, 0]

        assert jnp.allclose(result_sparse, result_monomial_val)

    @pytest.mark.skip(reason="Too slow - hypothesis test disabled temporarily")
    @given(st.integers(2, 3))
    @settings(max_examples=2, deadline=1000)
    def test_multiply_with_hypothesis(self, degree):
        """Test multiplication with random polynomials."""
        from algebraic.polynomials.jax import MonomialBasisAlgebra
        from algebraic.polynomials.sparse import SparsePolynomialAlgebra
        from algebraic.tensor_algebra.jax import max_min_algebra

        module = max_min_algebra()
        algebra = module.algebra
        sparse_alg = SparsePolynomialAlgebra(algebra=algebra, degree=degree)
        monomial_alg = MonomialBasisAlgebra(num_vars=degree, module=module)

        # Create two simple polynomials
        x_0 = sparse_alg.variable(0)
        x_1 = sparse_alg.variable(1) if degree > 1 else sparse_alg.variable(0)

        # Convert to monomial
        x_0_monomial = monomial_alg.from_sparse(x_0)
        x_1_monomial = monomial_alg.from_sparse(x_1)

        # Multiply
        prod_sparse = sparse_alg.mul(x_0, x_1)
        prod_monomial = monomial_alg.mul(x_0_monomial, x_1_monomial)

        # Test by evaluation
        import jax.random as jrandom

        key = jrandom.PRNGKey(123)
        point = jrandom.uniform(key, shape=(degree,), minval=-5.0, maxval=5.0)
        point_dict = {i: point[i] for i in range(degree)}

        result_sparse = list(sparse_alg.evaluate(prod_sparse, point_dict).values())[0]
        result_monomial = monomial_alg.evaluate(prod_monomial, point)
        monomial_value = result_monomial.coeffs[tuple([0] * degree)]

        assert jnp.allclose(result_sparse, monomial_value, rtol=1e-5)


class TestMonomialBasisEvaluation:
    """Test that evaluation matches sparse representation."""

    def test_evaluate_variable(self, sparse_alg_factory, monomial_alg_factory, tropical_maxplus_algebra, tropical_maxplus_module):
        """Test evaluating a variable."""
        sparse_alg = sparse_alg_factory(tropical_maxplus_algebra, 3)
        monomial_alg = monomial_alg_factory(tropical_maxplus_module, 3)

        # Create variable
        x_0_sparse = sparse_alg.variable(0)
        x_0_monomial = monomial_alg.from_sparse(x_0_sparse)

        # Evaluate at a point
        point_dict = {0: jnp.array(2.0), 1: jnp.array(3.0), 2: jnp.array(4.0)}
        point_array = jnp.array([2.0, 3.0, 4.0])

        result_sparse = list(sparse_alg.evaluate(x_0_sparse, point_dict).values())[0]
        result_monomial = monomial_alg.evaluate(x_0_monomial, point_array)

        # After evaluating all variables, we should have a constant
        monomial_value = result_monomial.coeffs[0, 0, 0]

        assert jnp.allclose(result_sparse, monomial_value)

    def test_evaluate_product(self, sparse_alg_factory, monomial_alg_factory, tropical_maxplus_algebra, tropical_maxplus_module):
        """Test evaluating a product."""
        sparse_alg = sparse_alg_factory(tropical_maxplus_algebra, 2)
        monomial_alg = monomial_alg_factory(tropical_maxplus_module, 2)

        # Create x_0 * x_1
        x_0 = sparse_alg.variable(0)
        x_1 = sparse_alg.variable(1)
        p_sparse = sparse_alg.mul(x_0, x_1)
        p_monomial = monomial_alg.from_sparse(p_sparse)

        # Evaluate
        point_dict = {0: jnp.array(2.0), 1: jnp.array(3.0)}
        point_array = jnp.array([2.0, 3.0])

        result_sparse = list(sparse_alg.evaluate(p_sparse, point_dict).values())[0]
        result_monomial = monomial_alg.evaluate(p_monomial, point_array)
        monomial_value = result_monomial.coeffs[0, 0]

        assert jnp.allclose(result_sparse, monomial_value)


class TestMonomialBasisCompose:
    """Test that composition matches sparse representation."""

    def test_compose_simple(self, sparse_alg_factory, monomial_alg_factory, bool_algebra, bool_module):
        """Test simple composition."""
        sparse_alg = sparse_alg_factory(bool_algebra, 2)
        monomial_alg = monomial_alg_factory(bool_module, 2)

        # Create x_0 and x_1
        x_0_sparse = sparse_alg.variable(0)
        x_1_sparse = sparse_alg.variable(1)

        x_0_monomial = monomial_alg.from_sparse(x_0_sparse)
        x_1_monomial = monomial_alg.from_sparse(x_1_sparse)

        # Compose x_0 with x_1 (replace x_0 with x_1)
        result_sparse = sparse_alg.compose(x_0_sparse, {0: x_1_sparse})
        result_monomial = monomial_alg.compose(x_0_monomial, {0: x_1_monomial})

        # Convert monomial result to sparse
        result_monomial_as_sparse = monomial_alg.to_sparse(result_monomial)

        # Compare
        assert result_sparse.keys() == result_monomial_as_sparse.keys()
        for key in result_sparse.keys():
            assert jnp.array_equal(result_sparse[key], result_monomial_as_sparse[key])


class TestMonomialBasisEdgeCases:
    """Test edge cases for monomial basis."""

    def test_zero_polynomial(self, monomial_alg_factory, tropical_maxplus_module):
        """Test zero polynomial."""
        monomial_alg = monomial_alg_factory(tropical_maxplus_module, 2)

        zero = monomial_alg.constant(tropical_maxplus_module.algebra.zero)

        # In tropical max-plus, zero = -inf
        expected = jnp.full((2, 2), tropical_maxplus_module.algebra.zero)
        assert jnp.allclose(zero.coeffs, expected)

    def test_large_degree(self, sparse_alg_factory, monomial_alg_factory, bool_algebra, bool_module):
        """Test with larger degree (but still manageable)."""
        # 2^6 = 64 entries, should be fine
        sparse_alg = sparse_alg_factory(bool_algebra, 6)
        monomial_alg = monomial_alg_factory(bool_module, 6)

        x_0 = sparse_alg.variable(0)
        x_0_monomial = monomial_alg.from_sparse(x_0)

        assert x_0_monomial.shape == (2,) * 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
