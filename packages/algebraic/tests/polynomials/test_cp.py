"""Tests for RankDecomposition polynomial implementation against SparsePolynomial baseline."""
# ruff: noqa: ANN201, ANN001
# mypy: disable-error-code="no-untyped-call,no-untyped-def,import-not-found"

from __future__ import annotations

import equinox as eqx
import hypothesis
import numpy as np
import pytest
from algebraic import AlgebraicArray, BooleanAlgebra, DeMorganAlgebra
from algebraic.polynomials import PolyDict as SparsePolynomial
from algebraic.polynomials import RankDecomposition
from algebraic.polynomials.dok import PolyDict
from algebraic.types import Array, Backend, is_torch_array
from algebraic.utils.testing import assert_close, assert_equal, make_array
from bitarray import frozenbitarray
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Shaped


class TestRankDecompositionConversion:
    """Test conversion between sparse and rank decomposition representations."""

    def test_from_sparse_constant(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test converting constant from sparse to rank decomposition."""
        num_vars = 3

        # Create constant in sparse form
        p_sparse = PolyDict.constant(bool_algebra.one, num_vars, algebra=bool_algebra, backend=backend)

        # For now, test basic properties
        _p_rank = RankDecomposition.from_sparse(p_sparse)

    def test_variable_creation(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test creating a single variable."""
        num_vars = 3

        xp = Backend(backend).get_array_namespace()

        x_0 = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)

        # Evaluate at a point where x_0 = bool_algebra.one
        result = x_0.evaluate((xp.asarray([bool_algebra.one, bool_algebra.zero, bool_algebra.zero])))
        assert_equal(result.factors[0, 0, 0].data, (bool_algebra.one))

        # Evaluate at a point where x_0 = bool_algebra.zero
        result = x_0.evaluate(xp.asarray([bool_algebra.zero, bool_algebra.one, bool_algebra.one]))
        assert_equal(result.factors[0, 0, 0].data, (bool_algebra.zero))

    def test_constant_creation(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test creating a constant."""
        num_vars = 3

        xp = Backend(backend).get_array_namespace()

        c = RankDecomposition.constant((bool_algebra.one), num_vars, bool_algebra, backend=backend)

        # Should evaluate to bool_algebra.one at any point
        result = c.evaluate(xp.asarray([bool_algebra.one, bool_algebra.one, bool_algebra.one]))
        assert_equal(result.factors[0, 0, 0].data, (bool_algebra.one))

        result = c.evaluate(xp.asarray([bool_algebra.zero, bool_algebra.zero, bool_algebra.zero]))
        assert_equal(result.factors[0, 0, 0].data, (bool_algebra.one))


class TestRankDecompositionAddition:
    """Test that addition matches sparse representation."""

    def test_add_variables(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test adding two variables."""
        num_vars = 3
        xp = Backend(backend).get_array_namespace()

        # Create in both representations
        x_0_sparse = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        x_1_sparse = PolyDict.variable(1, num_vars, algebra=maxmin_algebra, backend=backend)
        x_0_rank = RankDecomposition.variable(0, num_vars, maxmin_algebra, backend=backend)
        x_1_rank = RankDecomposition.variable(1, num_vars, maxmin_algebra, backend=backend)

        # Add in both
        sum_sparse = x_0_sparse + x_1_sparse
        sum_rank = x_0_rank + x_1_rank

        # Test by evaluation at a point
        point = xp.asarray([2.0, 3.0, 4.0])

        result_sparse = list(sum_sparse.evaluate(point).values())[0]
        result_rank = sum_rank.evaluate(point)
        rank_value = result_rank.factors[0, 0, 0].data

        assert_close(result_sparse, rank_value)

    @given(degree=st.integers(2, 7))
    @settings(max_examples=10, deadline=None, suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture])
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

        p1_sparse = x_0 + PolyDict.constant((2.0), num_vars, algebra=algebra, backend=backend)
        p2_sparse = x_1 + PolyDict.constant((3.0), num_vars, algebra=algebra, backend=backend)

        # Convert to rank decomposition
        p1_rank = RankDecomposition.from_sparse(p1_sparse)
        p2_rank = RankDecomposition.from_sparse(p2_sparse)

        # Add in both representations
        sum_sparse = p1_sparse + p2_sparse
        sum_rank = p1_rank + p2_rank

        # Test by evaluation at a random point (use numpy arrays to match backend)
        rng = np.random.default_rng(42)
        point = rng.uniform(-5.0, 5.0, size=(degree,))
        point_dict = {i: point[i] for i in range(degree)}

        # Evaluate sparse
        result_sparse = sum_sparse.evaluate(point_dict)
        assert isinstance(result_sparse, SparsePolynomial)

        # Evaluate rank decomposition
        result_rank = sum_rank.evaluate(make_array(point, backend))
        assert isinstance(result_rank, RankDecomposition)
        # Convert to sparse repr and check equality
        sparse_repr = RankDecomposition.to_sparse(result_rank)

        assert set(result_sparse.keys()) == set(sparse_repr.keys())
        for monom in result_sparse.keys():
            assert_close(result_sparse[monom], sparse_repr[monom])


class TestRankDecompositionMultiplication:
    """Test that multiplication matches sparse representation."""

    def test_multiply_variables_bool(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test multiplying two variables in Boolean algebra."""
        num_vars = 2
        xp = Backend(backend).get_array_namespace()

        # Create in both representations
        x_0_sparse = PolyDict.variable(0, num_vars, algebra=bool_algebra, backend=backend)
        x_1_sparse = PolyDict.variable(1, num_vars, algebra=bool_algebra, backend=backend)
        x_0_rank = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)
        x_1_rank = RankDecomposition.variable(1, num_vars, bool_algebra, backend=backend)

        # Multiply in both (Boolean: mul = AND)
        prod_sparse = x_0_sparse * x_1_sparse
        prod_rank = x_0_rank * x_1_rank

        # Test truth table
        test_points = [
            xp.asarray([bool_algebra.one, bool_algebra.one]),
            xp.asarray([bool_algebra.one, bool_algebra.zero]),
            xp.asarray([bool_algebra.zero, bool_algebra.one]),
            xp.asarray([bool_algebra.zero, bool_algebra.zero]),
        ]

        for point in test_points:
            result_sparse = list(prod_sparse.evaluate(point).values())[0]
            result_rank = prod_rank.evaluate(point)
            rank_value = result_rank.factors[0, 0, 0].data

            assert_equal(result_sparse, rank_value)

    def test_multiply_with_constant(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test multiplication with constant."""
        num_vars = 2
        xp = Backend(backend).get_array_namespace()

        # Create variable and constant
        x_0_sparse = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        c_sparse = PolyDict.constant((5.0), num_vars, algebra=maxmin_algebra, backend=backend)

        x_0_rank = RankDecomposition.variable(0, num_vars, maxmin_algebra, backend=backend)
        c_rank = RankDecomposition.constant((5.0), num_vars, maxmin_algebra, backend=backend)

        # Multiply
        prod_sparse = x_0_sparse * c_sparse
        prod_rank = x_0_rank * c_rank

        # Test by evaluation
        test_point = xp.asarray([3.0, 2.0])

        result_sparse = list(prod_sparse.evaluate(test_point).values())[0]
        result_rank = prod_rank.evaluate(test_point)
        rank_value = result_rank.factors[0, 0, 0].data

        assert_close(result_sparse, rank_value)

    def test_multiply_with_zero_constant(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test multiplication with zero constant (regression test).

        This tests the fix for a bug where multiplying any polynomial by zero
        did not properly simplify to a zero polynomial. The issue was that
        rank-1 components with all-zero factors were not being removed.
        """
        num_vars = 3
        xp = Backend(backend).get_array_namespace()

        # Create const(1) and const(0)
        const_one = RankDecomposition.constant((bool_algebra.one), num_vars, bool_algebra, backend=backend)
        const_zero = RankDecomposition.zero(num_vars, bool_algebra, backend=backend)

        # Multiply them
        product = const_one * const_zero

        # Result should be zero polynomial
        # Check by evaluating at any point
        result = product.evaluate(xp.asarray([bool_algebra.one, bool_algebra.one, bool_algebra.zero]))
        assert_equal(result.factors[0, 0, 0].data, (bool_algebra.zero))

        # Also test that factors are properly zero
        assert_close(result.factors[0, 0, 0], bool_algebra.zero)

    def test_multiply_zero_with_variable(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test multiplying zero with a variable (regression test)."""
        num_vars = 3
        xp = Backend(backend).get_array_namespace()

        zero = RankDecomposition.zero(num_vars, bool_algebra, backend=backend)
        x_0 = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)

        # Zero * variable should be zero
        result = zero * x_0

        # Should evaluate to bool_algebra.zero at any point
        test_point = xp.asarray([bool_algebra.one, bool_algebra.one, bool_algebra.zero])
        eval_result = result.evaluate(test_point)

        assert_equal(eval_result.factors[0, 0, 0].data, (bool_algebra.zero))

    @given(degree=st.integers(2, 7))
    @settings(max_examples=10, deadline=None, suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture])
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

        # Convert to rank decomposition
        x_0_rank = RankDecomposition.from_sparse(x_0)
        x_1_rank = RankDecomposition.from_sparse(x_1)

        # Multiply
        prod_sparse = x_0 * x_1
        prod_rank = x_0_rank * x_1_rank

        # Test by evaluation (use numpy arrays to match backend)
        rng = np.random.default_rng(123)
        point = rng.uniform(-5.0, 5.0, size=(degree,))
        point_dict = {i: point[i] for i in range(degree)}

        result_sparse = prod_sparse.evaluate(point_dict)
        assert isinstance(result_sparse, SparsePolynomial)
        assert len(result_sparse) == 1
        assert list(result_sparse.keys())[0] == frozenbitarray(degree)
        sparse_value = result_sparse[frozenbitarray(degree)]
        assert isinstance(sparse_value, AlgebraicArray)

        result_rank = prod_rank.evaluate(make_array(point, backend))
        assert isinstance(result_rank, RankDecomposition)
        rank_value = result_rank.factors[0, 0, 0]
        assert isinstance(rank_value, AlgebraicArray)

        assert_close(sparse_value, rank_value, rtol=1e-5)


class TestRankDecompositionEvaluation:
    """Test that evaluation matches sparse representation."""

    def test_evaluate_variable(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test evaluating a variable."""
        num_vars = 3

        # Create variable
        x_0_sparse = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        x_0_rank = RankDecomposition.variable(0, num_vars, maxmin_algebra, backend=backend)

        # Evaluate at a point
        point_dict = {0: (2.0), 1: (3.0), 2: (4.0)}
        point_array = make_array([2.0, 3.0, 4.0], backend)

        result_sparse = list(x_0_sparse.evaluate(point_dict).values())[0]
        result_rank = x_0_rank.evaluate(point_array)
        rank_value = result_rank.factors[0, 0, 0].data

        assert_close(result_sparse, rank_value)

    def test_evaluate_product(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test evaluating a product."""
        num_vars = 2

        # Create x_0 * x_1
        x_0_sparse = PolyDict.variable(0, num_vars, algebra=maxmin_algebra, backend=backend)
        x_1_sparse = PolyDict.variable(1, num_vars, algebra=maxmin_algebra, backend=backend)
        p_sparse = x_0_sparse * x_1_sparse

        x_0_rank = RankDecomposition.variable(0, num_vars, maxmin_algebra, backend=backend)
        x_1_rank = RankDecomposition.variable(1, num_vars, maxmin_algebra, backend=backend)
        p_rank = x_0_rank * x_1_rank

        # Evaluate
        point_dict = {0: (2.0), 1: (3.0)}
        point_array = make_array([2.0, 3.0], backend)

        result_sparse = list(p_sparse.evaluate(point_dict).values())[0]
        result_rank = p_rank.evaluate(point_array)
        rank_value = result_rank.factors[0, 0, 0].data

        assert_close(result_sparse, rank_value)


class TestRankDecompositionCompose:
    """Test that composition matches sparse representation."""

    def test_compose_simple(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test simple composition."""
        num_vars = 2

        # Create x_0 and x_1
        x_0_sparse = PolyDict.variable(0, num_vars, algebra=bool_algebra, backend=backend)
        x_1_sparse = PolyDict.variable(1, num_vars, algebra=bool_algebra, backend=backend)

        x_0_rank = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)
        x_1_rank = RankDecomposition.variable(1, num_vars, bool_algebra, backend=backend)

        # Compose x_0 with x_1 (replace x_0 with x_1)
        result_sparse = x_0_sparse.compose({0: x_1_sparse})
        result_rank = x_0_rank.compose([x_1_rank, x_1_rank])

        # Test by evaluation
        test_points = [
            ([bool_algebra.one, bool_algebra.one]),
            ([bool_algebra.one, bool_algebra.zero]),
            ([bool_algebra.zero, bool_algebra.one]),
        ]

        for point in test_points:
            val_sparse = list(result_sparse.evaluate({i: point[i] for i in range(2)}).values())[0]
            val_rank = result_rank.evaluate(make_array(point, backend))
            rank_value = val_rank.factors[0, 0, 0].data

            assert_equal(val_sparse, rank_value)

    def test_compose_zero_polynomial(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test composing zero polynomial with constants (regression test).

        This tests the fix for a bug where composing a zero polynomial with
        constant replacements would incorrectly return a non-zero polynomial.
        The issue was in how compose() handled all-zero factors.
        """
        num_vars = 3

        # Create zero polynomial
        zero = RankDecomposition.zero(num_vars, bool_algebra, backend=backend)

        # Create constant replacements
        const_one = RankDecomposition.constant((bool_algebra.one), num_vars, bool_algebra, backend=backend)
        replacements = [
            const_one,
            const_one,
            const_one,
        ]

        # Compose zero polynomial with constants
        result = zero.compose(replacements)

        # Result should still be zero
        eval_result = result.evaluate(make_array([bool_algebra.one, bool_algebra.one, bool_algebra.one], backend))
        assert_equal(eval_result.factors[0, 0, 0].data, (bool_algebra.zero))

    def test_compose_product_with_zero_substitution(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test composing product where one variable is replaced with zero (regression test)."""
        num_vars = 3

        # Create x_0 * x_1
        x_0 = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)
        x_1 = RankDecomposition.variable(1, num_vars, bool_algebra, backend=backend)
        x_2 = RankDecomposition.variable(2, num_vars, bool_algebra, backend=backend)
        product = x_0 * x_1

        # Compose: replace x_1 with zero, keep x_0 and x_2 as-is
        zero = RankDecomposition.zero(num_vars, bool_algebra, backend=backend)
        result = product.compose([x_0, zero, x_2])

        # Result should be zero (since x_0 * 0 = 0)
        eval_result = result.evaluate(make_array([bool_algebra.one, bool_algebra.one, bool_algebra.zero], backend))
        assert_equal(eval_result.factors[0, 0, 0].data, (bool_algebra.zero))


class TestRankDecompositionEdgeCases:
    """Test edge cases for rank decomposition."""

    def test_zero_polynomial(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test zero polynomial."""
        num_vars = 2

        zero = RankDecomposition.zero(num_vars, maxmin_algebra, backend=backend)

        # Should evaluate to zero (which is -inf in tropical max-plus) at any point
        result = zero.evaluate(make_array([1.0, 2.0], backend))
        expected = maxmin_algebra.zero
        assert_close(result.factors[0, 0, 0], expected)

    def test_multilinear_idempotence(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Test that x_i * x_i = x_i (multilinear property)."""
        num_vars = 2

        x_0 = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)

        # x_0 * x_0 should behave like x_0
        p = x_0 * x_0

        # Test at bool_algebra.one
        result = p.evaluate(make_array([bool_algebra.one, bool_algebra.zero], backend))
        assert_equal(result.factors[0, 0, 0].data, (bool_algebra.one))

        # Test at bool_algebra.zero
        result = p.evaluate(make_array([bool_algebra.zero, bool_algebra.zero], backend))
        assert_equal(result.factors[0, 0, 0].data, (bool_algebra.zero))


class TestRankDecompositionJAXTransformations:
    """Test JAX transformations on rank decomposition polynomials."""

    def test_jit_compilation(self, maxmin_algebra: DeMorganAlgebra, jax_backend: str) -> None:
        """Test that evaluation can be JIT-compiled."""
        import jax

        num_vars = 2
        x_0 = RankDecomposition.variable(0, num_vars, maxmin_algebra, backend=jax_backend)
        x_1 = RankDecomposition.variable(1, num_vars, maxmin_algebra, backend=jax_backend)
        p = x_0 * x_1

        # Create JIT-compiled evaluation function
        @jax.jit
        def eval_fn(point: Shaped[Array, "2"]) -> Array:
            return RankDecomposition.evaluate(p, point).factors[0, 0, 0].data

        result = eval_fn(make_array([2.0, 3.0], jax_backend))
        # In max-min: mul is min, so min(2.0, 3.0) = 2.0
        assert_close(result, (2.0))

    def test_vmap_evaluation(self, maxmin_algebra: DeMorganAlgebra, jax_backend: str) -> None:
        """Test batched evaluation using vmap."""
        import jax

        num_vars = 2
        x_0 = RankDecomposition.variable(0, num_vars, maxmin_algebra, backend=jax_backend)

        # Batch of points to evaluate
        points = make_array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], jax_backend)

        # Use vmap to evaluate at all points
        results = jax.vmap(lambda pt: x_0.evaluate(pt).factors[0, 0, 0].data)(points)

        expected = make_array([1.0, 2.0, 3.0], jax_backend)
        assert_close(results, expected)

    def test_grad_evaluation(self, maxmin_algebra: DeMorganAlgebra, jax_backend: str) -> None:
        """Test gradient computation through polynomial evaluation."""
        num_vars = 2
        x_0 = RankDecomposition.variable(0, num_vars, maxmin_algebra, backend=jax_backend)
        x_1 = RankDecomposition.variable(1, num_vars, maxmin_algebra, backend=jax_backend)
        p = x_0 * x_1  # In tropical max-plus: x_0 + x_1

        # Create JIT-compiled evaluation function
        def fn(x: Shaped[Array, "2"]) -> Array:
            return RankDecomposition.evaluate(p, x).factors[0, 0, 0].data

        # Compute gradient
        grad_fn = eqx.filter_grad(fn)

        point = make_array([2.0, 3.0], jax_backend)
        gradient = grad_fn(point)

        # d/dx_0 min(x_0, x_1) at (2.0, 3.0) = 1.0 (x_0 is active minimum)
        # d/dx_1 min(x_0, x_1) at (2.0, 3.0) = 0.0 (x_1 is not the minimum)
        expected = make_array([1.0, 0.0], jax_backend)
        assert_close(gradient, expected)


class TestRankDecompositionTropical:
    """Test rank decomposition with tropical algebras."""

    def test_tropical_min_plus(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test max-min algebra (negative reals) - similar to tropical min-plus."""
        num_vars = 2

        x_0 = RankDecomposition.variable(0, num_vars, maxmin_algebra, backend=backend)
        x_1 = RankDecomposition.variable(1, num_vars, maxmin_algebra, backend=backend)

        # In max-min: add = max, mul = min
        # x_0 * x_1 means min(x_0, x_1)
        p = x_0 * x_1

        # Use negative values for negative reals algebra
        result = p.evaluate(make_array([-2.0, -3.0], backend))
        # min(-2, -3) = -3
        assert_close(result.factors[0, 0, 0].data, (-3.0))

    def test_tropical_max_plus(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test max-min algebra (positive reals) - similar to tropical max-plus."""
        num_vars = 2

        x_0 = RankDecomposition.variable(0, num_vars, maxmin_algebra, backend=backend)
        x_1 = RankDecomposition.variable(1, num_vars, maxmin_algebra, backend=backend)

        # In max-min: add = max, mul = min
        # x_0 * x_1 means min(x_0, x_1)
        p = x_0 * x_1

        result = p.evaluate(make_array([2.0, 3.0], backend))
        # min(2, 3) = 2
        assert_close(result.factors[0, 0, 0].data, (2.0))

    def test_multiply_with_zero_maxmin(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test multiplication with zero in max-min algebra (regression test).

        This verifies that the algebra-aware zero detection works correctly
        for non-Boolean semirings like max-min where zero = -inf.
        """
        num_vars = 3

        # Create const(one) and const(zero) for max-min algebra
        # In max-min: zero = -inf, one = 0 (identity for min)
        const_one = RankDecomposition.constant((0.0), num_vars, maxmin_algebra, backend=backend)
        const_zero = RankDecomposition.zero(num_vars, maxmin_algebra, backend=backend)

        # Multiply them: 0 * (-inf) should give -inf
        product = const_one * const_zero

        # Result should be zero polynomial (zero = -inf in max-min)
        result = product.evaluate(make_array([1.0, 2.0, 3.0], backend))

        # Extract the scalar value
        result_value = result.factors[0, 0, 0].data.item()

        # Should be -inf (the zero element)
        assert np.isinf(result_value) and result_value < 0


class TestRankDecompositionMemoryEfficiency:
    """Test memory efficiency of rank decomposition."""

    def test_large_num_vars_feasible(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """Test that rank decomposition can handle large num_vars."""
        # n=15 would require 2^15 = 32768 entries in monomial basis form
        num_vars = 15

        x_0 = RankDecomposition.variable(0, num_vars, maxmin_algebra, backend=backend)
        x_1 = RankDecomposition.variable(1, num_vars, maxmin_algebra, backend=backend)
        x_2 = RankDecomposition.variable(2, num_vars, maxmin_algebra, backend=backend)

        p = (x_0 * x_1) * x_2

        # Should be able to evaluate efficiently
        point = make_array(np.ones(15) * 2.0, backend)
        result = p.evaluate(point)

        # In max-min: min(min(2, 2), 2) = 2
        assert_close(result.factors[0, 0, 0].data, (2.0))


class TestTorchBackendRegressions:
    """Regression tests for torch backend bugs in pad_upto and compose."""

    def test_addition_different_degree_torch(self) -> None:
        """(x0 * x1) + x2 must not crash on torch backend (pad_upto shape mismatch)."""
        import algebraic as alg

        torch = pytest.importorskip("torch")
        bool_alg = alg.semirings.boolean_algebra(mode="soft")

        x0 = RankDecomposition.variable(0, num_vars=3, algebra=bool_alg, backend="torch")
        x1 = RankDecomposition.variable(1, num_vars=3, algebra=bool_alg, backend="torch")
        x2 = RankDecomposition.variable(2, num_vars=3, algebra=bool_alg, backend="torch")

        p = (x0 * x1) + x2

        assert isinstance(p, RankDecomposition)
        assert isinstance(p.factors.data, torch.Tensor)

    def test_compose_different_degree_torch(self) -> None:
        """compose() with substitution polynomials of different degree must not crash."""
        import algebraic as alg

        torch = pytest.importorskip("torch")
        bool_alg = alg.semirings.boolean_algebra(mode="soft")

        x0 = RankDecomposition.variable(0, num_vars=3, algebra=bool_alg, backend="torch")
        x1 = RankDecomposition.variable(1, num_vars=3, algebra=bool_alg, backend="torch")
        x2 = RankDecomposition.variable(2, num_vars=3, algebra=bool_alg, backend="torch")

        sub_data = torch.randn(1, 2, 4)
        sub_arr = alg.array(sub_data, semiring=bool_alg, backend="torch")
        sub_rd = RankDecomposition(
            factors=sub_arr,
            max_rank=1,
            max_degree=2,
            max_replacement_degree=3,
            backend="torch",
        )

        result = x0.compose([sub_rd, x1, x2])

        assert isinstance(result, RankDecomposition)
        assert isinstance(result.factors.data, torch.Tensor)

    def test_grad_flows_through_pad_upto_torch(self) -> None:
        """Gradients must flow through pad_upto (the Bug 2 fix) on the torch backend."""
        import algebraic as alg
        from algebraic.polynomials.rank_decomp import pad_upto

        torch = pytest.importorskip("torch")
        bool_alg = alg.semirings.boolean_algebra(mode="soft")

        # param has requires_grad=True; wrap it as a degree-1 RankDecomposition
        param = torch.randn(1, 1, 4, requires_grad=True)
        factors = alg.array(param, semiring=bool_alg, backend="torch")

        # Pad from degree-1 to degree-2; this exercises _set_at_index / index_put
        padded = pad_upto(factors, max_rank=1, max_degree=2, algebra=bool_alg)
        assert is_torch_array(padded.data)

        loss = padded.data.sum()
        loss.backward()

        assert param.grad is not None, "gradient must flow back through pad_upto"

    def test_pad_upto_torch_incompatible_rank_degree_n_plus_1(self) -> None:
        """pad_upto must not crash when rank, degree, and n_plus_1 are all different sizes.

        Regression for the _set_at_index torch broadcast bug: when padding a factor of
        shape (1, 2, 3) up to (2, 3, 3), the index [:, :2, :] produces torchy aranges of
        shapes (2,), (2,), (3,) — previously torch.broadcast_shapes raised RuntimeError
        because 2 != 3 and neither is 1 in the degree/n_plus_1 positions.
        """
        import algebraic as alg
        from algebraic.polynomials.rank_decomp import pad_upto

        torch = pytest.importorskip("torch")
        bool_alg = alg.semirings.boolean_algebra(mode="soft")

        # shape (1, 2, 3): rank=1, degree=2, n_plus_1=3
        data = torch.randn(1, 2, 3)
        factors = alg.array(data, semiring=bool_alg, backend="torch")

        # Padding rank 1→2, degree 2→3 forces [:, :2, :] assignment on a (2, 3, 3) base,
        # producing torchy shapes (2,), (2,), (3,) — incompatible under the old code.
        padded = pad_upto(factors, max_rank=2, max_degree=3, algebra=bool_alg)

        assert padded.shape == (2, 3, 3)
        assert is_torch_array(padded.data)
        # The original factors must appear in the first rank slot, first two degree slots.
        import numpy as np

        np.testing.assert_allclose(
            padded.data[:1, :2, :].detach().cpu().numpy(),
            data.numpy(),
        )

    def test_compose_degree_stays_bounded(self) -> None:
        """compose() must not let degree grow beyond max_degree across repeated calls.

        Regression for the degree-explosion bug: without normalize(), each compose()
        call multiplied degrees so that d_1=5, d_2=11, d_3=23, ...
        """
        import algebraic as alg

        pytest.importorskip("torch")
        bool_alg = alg.semirings.boolean_algebra(mode="logic")

        # Two-variable polynomial: p(x0, x1) = x0
        num_vars = 2
        x0 = RankDecomposition.variable(0, num_vars=num_vars, algebra=bool_alg, backend="torch")
        x1 = RankDecomposition.variable(1, num_vars=num_vars, algebra=bool_alg, backend="torch")

        # Replacements have degree > 1 so composition inflates degree if not normalized.
        # sub0 = x0 * x1 (degree 2), sub1 = x1 (degree 1)
        sub0 = x0 * x1  # degree 2
        sub1 = x1  # degree 1

        p = x0
        for _ in range(5):
            p = p.compose([sub0, sub1])

        assert p.degree <= p.max_degree, f"degree {p.degree} exceeded max_degree {p.max_degree} after repeated compose()"

    def test_compose_byte_canonical_after_repeated_steps(self) -> None:
        """Semantically identical residuals must produce identical factor bytes.

        Stepping a fixed-point polynomial (one that maps to itself) with the same symbol
        repeatedly must yield the same factor bytes on every step — demonstrating that
        normalize() produces a canonical representation.
        """
        import algebraic as alg

        pytest.importorskip("torch")
        bool_alg = alg.semirings.boolean_algebra(mode="logic")

        # p(x0) = x0: stepping with identity substitution [x0] should be a fixed point.
        num_vars = 1
        x0 = RankDecomposition.variable(0, num_vars=num_vars, algebra=bool_alg, backend="torch")

        p = x0
        results = []
        for _ in range(4):
            p = p.compose([x0])
            assert is_torch_array(p.factors.data)
            results.append(p.factors.data.detach().cpu().numpy().tobytes())

        assert len(set(results)) == 1, "compose() of fixed-point polynomial must produce identical bytes on every call"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
