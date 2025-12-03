"""Tests for RankDecomposition polynomial implementation against SparsePolynomial baseline."""
# ruff: noqa: ANN201, ANN001

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from algebraic.polynomials.jax import MonomialBasisAlgebra, RankDecompositionAlgebra
from algebraic.polynomials.sparse import SparsePolynomial, SparsePolynomialAlgebra


class TestRankDecompositionConversion:
    """Test conversion between sparse and rank decomposition representations."""

    def test_from_sparse_constant(self, sparse_alg_factory, rank_alg_factory, bool_algebra, bool_module):
        """Test converting constant from sparse to rank decomposition."""
        sparse_alg = sparse_alg_factory(bool_algebra, 3)
        rank_alg = rank_alg_factory(bool_module, 3)

        # Create constant in sparse form
        p_sparse = sparse_alg.constant(jnp.array(True))

        # Convert to rank decomposition via monomial basis
        monomial_alg = MonomialBasisAlgebra(num_vars=3, module=bool_module)
        p_monomial = monomial_alg.from_sparse(p_sparse)

        # For now, test basic properties
        # TODO: Implement from_sparse for RankDecomposition
        # p_rank = rank_alg.from_sparse(p_sparse)

    def test_variable_creation(self, rank_alg_factory, bool_module):
        """Test creating a single variable."""
        rank_alg = rank_alg_factory(bool_module, 3)

        x_0 = rank_alg.variable(0)

        # Evaluate at a point where x_0 = True
        result = rank_alg.evaluate(x_0, jnp.array([True, False, False]))
        # Result is a constant polynomial, extract the scalar
        assert jnp.array_equal(result.factors[0, 0, 0], jnp.array(True))

        # Evaluate at a point where x_0 = False
        result = rank_alg.evaluate(x_0, jnp.array([False, True, True]))
        assert jnp.array_equal(result.factors[0, 0, 0], jnp.array(False))

    def test_constant_creation(self, rank_alg_factory, bool_module):
        """Test creating a constant."""
        rank_alg = rank_alg_factory(bool_module, 3)

        c = rank_alg.constant(jnp.array(True))

        # Should evaluate to True at any point
        result = rank_alg.evaluate(c, jnp.array([True, True, True]))
        assert jnp.array_equal(result.factors[0, 0, 0], jnp.array(True))

        result = rank_alg.evaluate(c, jnp.array([False, False, False]))
        assert jnp.array_equal(result.factors[0, 0, 0], jnp.array(True))


class TestRankDecompositionAddition:
    """Test that addition matches sparse representation."""

    def test_add_variables(self, sparse_alg_factory, rank_alg_factory, tropical_maxplus_algebra, tropical_maxplus_module):
        """Test adding two variables."""
        sparse_alg = sparse_alg_factory(tropical_maxplus_algebra, 3)
        rank_alg = rank_alg_factory(tropical_maxplus_module, 3)

        # Create in both representations
        x_0_sparse = sparse_alg.variable(0)
        x_1_sparse = sparse_alg.variable(1)
        x_0_rank = rank_alg.variable(0)
        x_1_rank = rank_alg.variable(1)

        # Add in both
        sum_sparse = sparse_alg.add(x_0_sparse, x_1_sparse)
        sum_rank = rank_alg.add(x_0_rank, x_1_rank)

        # Test by evaluation at a point
        point = jnp.array([2.0, 3.0, 4.0])

        result_sparse = list(sparse_alg.evaluate(sum_sparse, {i: point[i] for i in range(3)}).values())[0]
        result_rank = rank_alg.evaluate(sum_rank, point)
        rank_value = result_rank.factors[0, 0, 0]

        assert jnp.allclose(result_sparse, rank_value)


class TestRankDecompositionMultiplication:
    """Test that multiplication matches sparse representation."""

    def test_multiply_variables_bool(self, sparse_alg_factory, rank_alg_factory, bool_algebra, bool_module):
        """Test multiplying two variables in Boolean algebra."""
        sparse_alg = sparse_alg_factory(bool_algebra, 2)
        rank_alg = rank_alg_factory(bool_module, 2)

        # Create in both representations
        x_0_sparse = sparse_alg.variable(0)
        x_1_sparse = sparse_alg.variable(1)
        x_0_rank = rank_alg.variable(0)
        x_1_rank = rank_alg.variable(1)

        # Multiply in both (Boolean: mul = AND)
        prod_sparse = sparse_alg.mul(x_0_sparse, x_1_sparse)
        prod_rank = rank_alg.mul(x_0_rank, x_1_rank)

        # Test truth table
        test_points = [
            jnp.array([True, True]),
            jnp.array([True, False]),
            jnp.array([False, True]),
            jnp.array([False, False]),
        ]

        for point in test_points:
            result_sparse = list(sparse_alg.evaluate(prod_sparse, {i: point[i] for i in range(2)}).values())[0]
            result_rank = rank_alg.evaluate(prod_rank, point)
            rank_value = result_rank.factors[0, 0, 0]

            assert jnp.array_equal(result_sparse, rank_value), f"Mismatch at {point}"

    def test_multiply_with_constant(self, sparse_alg_factory, rank_alg_factory, tropical_maxplus_algebra, tropical_maxplus_module):
        """Test multiplication with constant."""
        sparse_alg = sparse_alg_factory(tropical_maxplus_algebra, 2)
        rank_alg = rank_alg_factory(tropical_maxplus_module, 2)

        # Create variable and constant
        x_0_sparse = sparse_alg.variable(0)
        c_sparse = sparse_alg.constant(jnp.array(5.0))

        x_0_rank = rank_alg.variable(0)
        c_rank = rank_alg.constant(jnp.array(5.0))

        # Multiply
        prod_sparse = sparse_alg.mul(x_0_sparse, c_sparse)
        prod_rank = rank_alg.mul(x_0_rank, c_rank)

        # Test by evaluation
        test_point = jnp.array([3.0, 2.0])

        result_sparse = list(sparse_alg.evaluate(prod_sparse, {0: test_point[0], 1: test_point[1]}).values())[0]
        result_rank = rank_alg.evaluate(prod_rank, test_point)
        rank_value = result_rank.factors[0, 0, 0]

        # In tropical max-plus: mul is +, so 5.0 + 3.0 = 8.0
        assert jnp.allclose(result_sparse, rank_value)


class TestRankDecompositionEvaluation:
    """Test that evaluation matches sparse representation."""

    def test_evaluate_variable(self, sparse_alg_factory, rank_alg_factory, tropical_maxplus_algebra, tropical_maxplus_module):
        """Test evaluating a variable."""
        sparse_alg = sparse_alg_factory(tropical_maxplus_algebra, 3)
        rank_alg = rank_alg_factory(tropical_maxplus_module, 3)

        # Create variable
        x_0_sparse = sparse_alg.variable(0)
        x_0_rank = rank_alg.variable(0)

        # Evaluate at a point
        point_dict = {0: jnp.array(2.0), 1: jnp.array(3.0), 2: jnp.array(4.0)}
        point_array = jnp.array([2.0, 3.0, 4.0])

        result_sparse = list(sparse_alg.evaluate(x_0_sparse, point_dict).values())[0]
        result_rank = rank_alg.evaluate(x_0_rank, point_array)
        rank_value = result_rank.factors[0, 0, 0]

        assert jnp.allclose(result_sparse, rank_value)

    def test_evaluate_product(self, sparse_alg_factory, rank_alg_factory, tropical_maxplus_algebra, tropical_maxplus_module):
        """Test evaluating a product."""
        sparse_alg = sparse_alg_factory(tropical_maxplus_algebra, 2)
        rank_alg = rank_alg_factory(tropical_maxplus_module, 2)

        # Create x_0 * x_1
        x_0_sparse = sparse_alg.variable(0)
        x_1_sparse = sparse_alg.variable(1)
        p_sparse = sparse_alg.mul(x_0_sparse, x_1_sparse)

        x_0_rank = rank_alg.variable(0)
        x_1_rank = rank_alg.variable(1)
        p_rank = rank_alg.mul(x_0_rank, x_1_rank)

        # Evaluate
        point_dict = {0: jnp.array(2.0), 1: jnp.array(3.0)}
        point_array = jnp.array([2.0, 3.0])

        result_sparse = list(sparse_alg.evaluate(p_sparse, point_dict).values())[0]
        result_rank = rank_alg.evaluate(p_rank, point_array)
        rank_value = result_rank.factors[0, 0, 0]

        assert jnp.allclose(result_sparse, rank_value)


class TestRankDecompositionCompose:
    """Test that composition matches sparse representation."""

    def test_compose_simple(self, sparse_alg_factory, rank_alg_factory, bool_algebra, bool_module):
        """Test simple composition."""
        sparse_alg = sparse_alg_factory(bool_algebra, 2)
        rank_alg = rank_alg_factory(bool_module, 2)

        # Create x_0 and x_1
        x_0_sparse = sparse_alg.variable(0)
        x_1_sparse = sparse_alg.variable(1)

        x_0_rank = rank_alg.variable(0)
        x_1_rank = rank_alg.variable(1)

        # Compose x_0 with x_1 (replace x_0 with x_1)
        result_sparse = sparse_alg.compose(x_0_sparse, {0: x_1_sparse})
        result_rank = rank_alg.compose(x_0_rank, {0: x_1_rank})

        # Test by evaluation
        test_points = [
            jnp.array([True, True]),
            jnp.array([True, False]),
            jnp.array([False, True]),
        ]

        for point in test_points:
            val_sparse = list(sparse_alg.evaluate(result_sparse, {i: point[i] for i in range(2)}).values())[0]
            val_rank = rank_alg.evaluate(result_rank, point)
            rank_value = val_rank.factors[0, 0, 0]

            assert jnp.array_equal(val_sparse, rank_value), f"Mismatch at {point}"


class TestRankDecompositionEdgeCases:
    """Test edge cases for rank decomposition."""

    def test_zero_polynomial(self, rank_alg_factory, tropical_maxplus_module):
        """Test zero polynomial."""
        rank_alg = rank_alg_factory(tropical_maxplus_module, 2)

        zero = rank_alg.zero

        # Should evaluate to zero (which is -inf in tropical max-plus) at any point
        result = rank_alg.evaluate(zero, jnp.array([1.0, 2.0]))
        expected = tropical_maxplus_module.algebra.zero
        assert jnp.allclose(result.factors[0, 0, 0], expected)

    def test_multilinear_idempotence(self, rank_alg_factory, bool_module):
        """Test that x_i * x_i = x_i (multilinear property)."""
        rank_alg = rank_alg_factory(bool_module, 2)

        x_0 = rank_alg.variable(0)

        # x_0 * x_0 should behave like x_0
        p = rank_alg.mul(x_0, x_0)

        # Test at True
        result = rank_alg.evaluate(p, jnp.array([True, False]))
        assert jnp.array_equal(result.factors[0, 0, 0], jnp.array(True))

        # Test at False
        result = rank_alg.evaluate(p, jnp.array([False, False]))
        assert jnp.array_equal(result.factors[0, 0, 0], jnp.array(False))


class TestRankDecompositionJAXTransformations:
    """Test JAX transformations on rank decomposition polynomials."""

    def test_jit_compilation(self, rank_alg_factory, tropical_maxplus_module):
        """Test that evaluation can be JIT-compiled."""
        rank_alg = rank_alg_factory(tropical_maxplus_module, 2)
        x_0 = rank_alg.variable(0)
        x_1 = rank_alg.variable(1)
        p = rank_alg.mul(x_0, x_1)

        # Create JIT-compiled evaluation function
        @jax.jit
        def eval_fn(point):
            return rank_alg.evaluate(p, point)

        result = eval_fn(jnp.array([2.0, 3.0]))
        # In tropical max-plus: mul is +, so 2.0 + 3.0 = 5.0
        assert jnp.allclose(result.factors[0, 0, 0], jnp.array(5.0))

    def test_vmap_evaluation(self, rank_alg_factory, tropical_maxplus_module):
        """Test batched evaluation using vmap."""
        rank_alg = rank_alg_factory(tropical_maxplus_module, 2)
        x_0 = rank_alg.variable(0)

        # Batch of points to evaluate
        points = jnp.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

        # Use vmap to evaluate at all points
        results = jax.vmap(lambda pt: rank_alg.evaluate(x_0, pt).factors[0, 0, 0])(points)

        expected = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(results, expected)

    def test_grad_evaluation(self, rank_alg_factory, tropical_maxplus_module):
        """Test gradient computation through polynomial evaluation."""
        rank_alg = rank_alg_factory(tropical_maxplus_module, 2)
        x_0 = rank_alg.variable(0)
        x_1 = rank_alg.variable(1)
        p = rank_alg.mul(x_0, x_1)  # In tropical max-plus: x_0 + x_1

        # Compute gradient
        grad_fn = jax.grad(lambda x: rank_alg.evaluate(p, x).factors[0, 0, 0])

        point = jnp.array([2.0, 3.0])
        gradient = grad_fn(point)

        # d/dx_0 (x_0 + x_1) = 1.0
        # d/dx_1 (x_0 + x_1) = 1.0
        expected = jnp.array([1.0, 1.0])
        assert jnp.allclose(gradient, expected)


class TestRankDecompositionTropical:
    """Test rank decomposition with tropical algebras."""

    def test_tropical_min_plus(self, rank_alg_factory):
        """Test tropical min-plus algebra."""
        from algebraic.tensor_algebra.jax import tropical_semiring

        module = tropical_semiring(minplus=True)
        rank_alg = rank_alg_factory(module, 2)

        x_0 = rank_alg.variable(0)
        x_1 = rank_alg.variable(1)

        # In min-plus: add = min, mul = +
        # x_0 * x_1 means x_0 + x_1
        p = rank_alg.mul(x_0, x_1)

        result = rank_alg.evaluate(p, jnp.array([2.0, 3.0]))
        assert jnp.allclose(result.factors[0, 0, 0], jnp.array(5.0))  # 2 + 3

    def test_tropical_max_plus(self, rank_alg_factory, tropical_maxplus_module):
        """Test tropical max-plus algebra."""
        rank_alg = rank_alg_factory(tropical_maxplus_module, 2)

        x_0 = rank_alg.variable(0)
        x_1 = rank_alg.variable(1)

        # In max-plus: add = max, mul = +
        # x_0 * x_1 means x_0 + x_1
        p = rank_alg.mul(x_0, x_1)

        result = rank_alg.evaluate(p, jnp.array([2.0, 3.0]))
        assert jnp.allclose(result.factors[0, 0, 0], jnp.array(5.0))  # 2 + 3


class TestRankDecompositionMemoryEfficiency:
    """Test memory efficiency of rank decomposition."""

    def test_large_num_vars_feasible(self, rank_alg_factory, tropical_maxplus_module):
        """Test that rank decomposition can handle large num_vars."""
        # n=15 would require 2^15 = 32768 entries in monomial basis form
        rank_alg = rank_alg_factory(tropical_maxplus_module, 15)

        x_0 = rank_alg.variable(0)
        x_1 = rank_alg.variable(1)
        x_2 = rank_alg.variable(2)

        p = rank_alg.mul(rank_alg.mul(x_0, x_1), x_2)

        # Should be able to evaluate efficiently
        point = jnp.ones(15) * 2.0
        result = rank_alg.evaluate(p, point)

        # In tropical max-plus: 2 + 2 + 2 = 6
        assert jnp.allclose(result.factors[0, 0, 0], jnp.array(6.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
