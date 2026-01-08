"""Tests for RankDecomposition polynomial implementation against SparsePolynomial baseline."""
# ruff: noqa: ANN201, ANN001

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
import quax
from algebraic.polynomials.rank_decomp import RankDecomposition


class TestRankDecompositionConversion:
    """Test conversion between sparse and rank decomposition representations."""

    def test_from_sparse_constant(self, sparse_helper, rank_helper, bool_algebra):
        """Test converting constant from sparse to rank decomposition."""
        sparse_alg = sparse_helper(bool_algebra, 3)
        rank_alg = rank_helper(bool_algebra, 3)

        # Create constant in sparse form
        p_sparse = sparse_alg.constant(jnp.array(True))

        # For now, test basic properties
        p_rank = RankDecomposition.from_sparse(p_sparse)

    def test_variable_creation(self, rank_helper, bool_algebra):
        """Test creating a single variable."""
        rank_alg = rank_helper(bool_algebra, 3)

        x_0 = rank_alg.variable(0)

        # Evaluate at a point where x_0 = True
        result = rank_alg.evaluate(x_0, jnp.array([True, False, False]))
        # Result is a constant polynomial, extract the scalar
        array_equal = quax.quaxify(jnp.array_equal)
        assert array_equal(result.factors[0, 0, 0].data, jnp.array(True))

        # Evaluate at a point where x_0 = False
        result = rank_alg.evaluate(x_0, jnp.array([False, True, True]))
        array_equal = quax.quaxify(jnp.array_equal)
        assert array_equal(result.factors[0, 0, 0].data, jnp.array(False))

    def test_constant_creation(self, rank_helper, bool_algebra):
        """Test creating a constant."""
        rank_alg = rank_helper(bool_algebra, 3)

        c = rank_alg.constant(jnp.array(True))

        # Should evaluate to True at any point
        result = rank_alg.evaluate(c, jnp.array([True, True, True]))
        array_equal = quax.quaxify(jnp.array_equal)
        assert array_equal(result.factors[0, 0, 0].data, jnp.array(True))

        result = rank_alg.evaluate(c, jnp.array([False, False, False]))
        array_equal = quax.quaxify(jnp.array_equal)
        assert array_equal(result.factors[0, 0, 0].data, jnp.array(True))


class TestRankDecompositionAddition:
    """Test that addition matches sparse representation."""

    def test_add_variables(self, sparse_helper, rank_helper, maxmin_algebra):
        """Test adding two variables."""
        sparse_alg = sparse_helper(maxmin_algebra, 3)
        rank_alg = rank_helper(maxmin_algebra, 3)

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
        rank_value = result_rank.factors[0, 0, 0].data

        assert quax.quaxify(jnp.allclose)(result_sparse, rank_value)


class TestRankDecompositionMultiplication:
    """Test that multiplication matches sparse representation."""

    def test_multiply_variables_bool(self, sparse_helper, rank_helper, bool_algebra):
        """Test multiplying two variables in Boolean algebra."""
        sparse_alg = sparse_helper(bool_algebra, 2)
        rank_alg = rank_helper(bool_algebra, 2)

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
            rank_value = result_rank.factors[0, 0, 0].data

            assert quax.quaxify(jnp.array_equal)(result_sparse, rank_value), f"Mismatch at {point}"

    def test_multiply_with_constant(self, sparse_helper, rank_helper, maxmin_algebra):
        """Test multiplication with constant."""
        sparse_alg = sparse_helper(maxmin_algebra, 2)
        rank_alg = rank_helper(maxmin_algebra, 2)

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
        rank_value = result_rank.factors[0, 0, 0].data

        assert quax.quaxify(jnp.allclose)(result_sparse, rank_value)


class TestRankDecompositionEvaluation:
    """Test that evaluation matches sparse representation."""

    def test_evaluate_variable(self, sparse_helper, rank_helper, maxmin_algebra):
        """Test evaluating a variable."""
        sparse_alg = sparse_helper(maxmin_algebra, 3)
        rank_alg = rank_helper(maxmin_algebra, 3)

        # Create variable
        x_0_sparse = sparse_alg.variable(0)
        x_0_rank = rank_alg.variable(0)

        # Evaluate at a point
        point_dict = {0: jnp.array(2.0), 1: jnp.array(3.0), 2: jnp.array(4.0)}
        point_array = jnp.array([2.0, 3.0, 4.0])

        result_sparse = list(sparse_alg.evaluate(x_0_sparse, point_dict).values())[0]
        result_rank = rank_alg.evaluate(x_0_rank, point_array)
        rank_value = result_rank.factors[0, 0, 0].data

        assert quax.quaxify(jnp.allclose)(result_sparse, rank_value)

    def test_evaluate_product(self, sparse_helper, rank_helper, maxmin_algebra):
        """Test evaluating a product."""
        sparse_alg = sparse_helper(maxmin_algebra, 2)
        rank_alg = rank_helper(maxmin_algebra, 2)

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
        rank_value = result_rank.factors[0, 0, 0].data

        assert quax.quaxify(jnp.allclose)(result_sparse, rank_value)


class TestRankDecompositionCompose:
    """Test that composition matches sparse representation."""

    def test_compose_simple(self, sparse_helper, rank_helper, bool_algebra):
        """Test simple composition."""
        sparse_alg = sparse_helper(bool_algebra, 2)
        rank_alg = rank_helper(bool_algebra, 2)

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
            rank_value = val_rank.factors[0, 0, 0].data

            assert quax.quaxify(jnp.array_equal)(val_sparse, rank_value), f"Mismatch at {point}"


class TestRankDecompositionEdgeCases:
    """Test edge cases for rank decomposition."""

    def test_zero_polynomial(self, rank_helper, maxmin_algebra):
        """Test zero polynomial."""
        rank_alg = rank_helper(maxmin_algebra, 2)

        zero = rank_alg.zero

        # Should evaluate to zero (which is -inf in tropical max-plus) at any point
        result = rank_alg.evaluate(zero, jnp.array([1.0, 2.0]))
        expected = maxmin_algebra.zero
        allclose = quax.quaxify(jnp.allclose)
        assert allclose(result.factors[0, 0, 0], expected)

    def test_multilinear_idempotence(self, rank_helper, bool_algebra):
        """Test that x_i * x_i = x_i (multilinear property)."""
        rank_alg = rank_helper(bool_algebra, 2)

        x_0 = rank_alg.variable(0)

        # x_0 * x_0 should behave like x_0
        p = rank_alg.mul(x_0, x_0)

        # Test at True
        result = rank_alg.evaluate(p, jnp.array([True, False]))
        array_equal = quax.quaxify(jnp.array_equal)
        assert array_equal(result.factors[0, 0, 0].data, jnp.array(True))

        # Test at False
        result = rank_alg.evaluate(p, jnp.array([False, False]))
        array_equal = quax.quaxify(jnp.array_equal)
        assert array_equal(result.factors[0, 0, 0].data, jnp.array(False))


class TestRankDecompositionJAXTransformations:
    """Test JAX transformations on rank decomposition polynomials."""

    def test_jit_compilation(self, rank_helper, maxmin_algebra):
        """Test that evaluation can be JIT-compiled."""
        rank_alg = rank_helper(maxmin_algebra, 2)
        x_0 = rank_alg.variable(0)
        x_1 = rank_alg.variable(1)
        p = rank_alg.mul(x_0, x_1)

        # Create JIT-compiled evaluation function
        @jax.jit
        def eval_fn(point):
            return rank_alg.evaluate(p, point)

        result = eval_fn(jnp.array([2.0, 3.0]))
        # In max-min: mul is min, so min(2.0, 3.0) = 2.0
        allclose = quax.quaxify(jnp.allclose)
        assert allclose(result.factors[0, 0, 0].data, jnp.array(2.0))

    def test_vmap_evaluation(self, rank_helper, maxmin_algebra):
        """Test batched evaluation using vmap."""
        rank_alg = rank_helper(maxmin_algebra, 2)
        x_0 = rank_alg.variable(0)

        # Batch of points to evaluate
        points = jnp.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

        # Use vmap to evaluate at all points
        results = jax.vmap(lambda pt: rank_alg.evaluate(x_0, pt).factors[0, 0, 0])(points)

        expected = jnp.array([1.0, 2.0, 3.0])
        assert quax.quaxify(jnp.allclose)(results, expected)

    def test_grad_evaluation(self, rank_helper, maxmin_algebra):
        """Test gradient computation through polynomial evaluation."""
        rank_alg = rank_helper(maxmin_algebra, 2)
        x_0 = rank_alg.variable(0)
        x_1 = rank_alg.variable(1)
        p = rank_alg.mul(x_0, x_1)  # In tropical max-plus: x_0 + x_1

        def fn(x):
            return rank_alg.evaluate(p, x).factors[0, 0, 0].data

        # Compute gradient
        grad_fn = eqx.filter_grad(fn)

        point = jnp.array([2.0, 3.0])
        gradient = grad_fn(point)

        # d/dx_0 min(x_0, x_1) at (2.0, 3.0) = 1.0 (x_0 is active minimum)
        # d/dx_1 min(x_0, x_1) at (2.0, 3.0) = 0.0 (x_1 is not the minimum)
        expected = jnp.array([1.0, 0.0])
        assert quax.quaxify(jnp.allclose)(gradient, expected)


class TestRankDecompositionTropical:
    """Test rank decomposition with tropical algebras."""

    def test_tropical_min_plus(self, rank_helper, maxmin_algebra):
        """Test max-min algebra (negative reals) - similar to tropical min-plus."""
        rank_alg = rank_helper(maxmin_algebra, 2)

        x_0 = rank_alg.variable(0)
        x_1 = rank_alg.variable(1)

        # In max-min: add = max, mul = min
        # x_0 * x_1 means min(x_0, x_1)
        p = rank_alg.mul(x_0, x_1)

        # Use negative values for negative reals algebra
        result = rank_alg.evaluate(p, jnp.array([-2.0, -3.0]))
        # min(-2, -3) = -3
        allclose = quax.quaxify(jnp.allclose)
        assert allclose(result.factors[0, 0, 0].data, jnp.array(-3.0))

    def test_tropical_max_plus(self, rank_helper, maxmin_algebra):
        """Test max-min algebra (positive reals) - similar to tropical max-plus."""
        rank_alg = rank_helper(maxmin_algebra, 2)

        x_0 = rank_alg.variable(0)
        x_1 = rank_alg.variable(1)

        # In max-min: add = max, mul = min
        # x_0 * x_1 means min(x_0, x_1)
        p = rank_alg.mul(x_0, x_1)

        result = rank_alg.evaluate(p, jnp.array([2.0, 3.0]))
        # min(2, 3) = 2
        allclose = quax.quaxify(jnp.allclose)
        assert allclose(result.factors[0, 0, 0].data, jnp.array(2.0))


class TestRankDecompositionMemoryEfficiency:
    """Test memory efficiency of rank decomposition."""

    def test_large_num_vars_feasible(self, rank_helper, maxmin_algebra):
        """Test that rank decomposition can handle large num_vars."""
        # n=15 would require 2^15 = 32768 entries in monomial basis form
        rank_alg = rank_helper(maxmin_algebra, 15)

        x_0 = rank_alg.variable(0)
        x_1 = rank_alg.variable(1)
        x_2 = rank_alg.variable(2)

        p = rank_alg.mul(rank_alg.mul(x_0, x_1), x_2)

        # Should be able to evaluate efficiently
        point = jnp.ones(15) * 2.0
        result = rank_alg.evaluate(p, point)

        # In max-min: min(min(2, 2), 2) = 2
        allclose = quax.quaxify(jnp.allclose)
        assert allclose(result.factors[0, 0, 0].data, jnp.array(2.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
