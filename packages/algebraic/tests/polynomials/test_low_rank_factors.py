"""Tests for LowRankFactors polynomial implementation against RankDecomposition baseline."""
# ruff: noqa: ANN201, ANN001
# mypy: disable-error-code="no-untyped-call,no-untyped-def,import-not-found"

from __future__ import annotations

import numpy as np
import pytest
from algebraic import BooleanAlgebra, DeMorganAlgebra
from algebraic.polynomials import LowRankFactors, RankDecomposition
from algebraic.polynomials.dok import PolyDict
from algebraic.types import Backend, is_torch_array
from algebraic.utils.testing import assert_close, assert_equal, make_array
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


class TestLowRankFactorsCreation:
    """Test factory methods and shape invariants."""

    def test_variable_shapes(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        num_vars = 3
        x = LowRankFactors.variable(0, num_vars, bool_algebra, backend=backend)
        assert x.weights.shape == (1, 1, num_vars)
        assert x.bias.shape == (1, 1)
        assert x.rank == 1
        assert x.degree == 1
        assert x.num_vars == num_vars

    def test_constant_shapes(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        num_vars = 3
        c = LowRankFactors.constant(bool_algebra.one, num_vars, bool_algebra, backend=backend)
        assert c.weights.shape == (1, 1, num_vars)
        assert c.bias.shape == (1, 1)

    def test_variable_creation(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        num_vars = 3
        xp = Backend(backend).get_array_namespace()

        x_0 = LowRankFactors.variable(0, num_vars, bool_algebra, backend=backend)

        result = x_0.evaluate(xp.asarray([bool_algebra.one, bool_algebra.zero, bool_algebra.zero]))
        assert_equal(result.bias[0, 0].data, bool_algebra.one)

        result = x_0.evaluate(xp.asarray([bool_algebra.zero, bool_algebra.one, bool_algebra.one]))
        assert_equal(result.bias[0, 0].data, bool_algebra.zero)

    def test_constant_creation(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        num_vars = 3
        xp = Backend(backend).get_array_namespace()

        c = LowRankFactors.constant(bool_algebra.one, num_vars, bool_algebra, backend=backend)

        result = c.evaluate(xp.asarray([bool_algebra.one, bool_algebra.one, bool_algebra.one]))
        assert_equal(result.bias[0, 0].data, bool_algebra.one)

        result = c.evaluate(xp.asarray([bool_algebra.zero, bool_algebra.zero, bool_algebra.zero]))
        assert_equal(result.bias[0, 0].data, bool_algebra.one)

    def test_zero_one(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        num_vars = 2
        xp = Backend(backend).get_array_namespace()

        z = LowRankFactors.zero(num_vars, bool_algebra, backend=backend)
        o = LowRankFactors.one(num_vars, bool_algebra, backend=backend)
        point = xp.asarray([bool_algebra.one, bool_algebra.one])

        assert_equal(z.evaluate(point).bias[0, 0].data, bool_algebra.zero)
        assert_equal(o.evaluate(point).bias[0, 0].data, bool_algebra.one)


class TestLowRankFactorsAddition:
    """Test addition matches RankDecomposition."""

    def test_add_variables(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        num_vars = 3
        xp = Backend(backend).get_array_namespace()

        x_0_rd = RankDecomposition.variable(0, num_vars, maxmin_algebra, backend=backend)
        x_1_rd = RankDecomposition.variable(1, num_vars, maxmin_algebra, backend=backend)

        x_0_lr = LowRankFactors.variable(0, num_vars, maxmin_algebra, backend=backend)
        x_1_lr = LowRankFactors.variable(1, num_vars, maxmin_algebra, backend=backend)

        sum_rd = x_0_rd + x_1_rd
        sum_lr = x_0_lr + x_1_lr

        point = xp.asarray([2.0, 3.0, 4.0])

        rd_value = sum_rd.evaluate(point).factors[0, 0, 0].data
        lr_value = sum_lr.evaluate(point).bias[0, 0].data

        assert_close(rd_value, lr_value)

    @given(degree=st.integers(2, 7))
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_add_with_hypothesis(self, degree: int, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        num_vars = degree
        algebra = maxmin_algebra

        x_0 = PolyDict.variable(0, num_vars, algebra=algebra, backend=backend)
        x_1 = (
            PolyDict.variable(1, num_vars, algebra=algebra, backend=backend)
            if degree > 1
            else PolyDict.variable(0, num_vars, algebra=algebra, backend=backend)
        )

        p1_sparse = x_0 + PolyDict.constant(2.0, num_vars, algebra=algebra, backend=backend)
        p2_sparse = x_1 + PolyDict.constant(3.0, num_vars, algebra=algebra, backend=backend)

        p1_rd = RankDecomposition.from_sparse(p1_sparse)
        p2_rd = RankDecomposition.from_sparse(p2_sparse)
        p1_lr = LowRankFactors.from_rank_decomposition(p1_rd)
        p2_lr = LowRankFactors.from_rank_decomposition(p2_rd)

        sum_rd = p1_rd + p2_rd
        sum_lr = p1_lr + p2_lr

        rng = np.random.default_rng(42)
        point = rng.uniform(-5.0, 5.0, size=(degree,))

        result_rd = sum_rd.evaluate(make_array(point, backend))
        result_lr = sum_lr.evaluate(make_array(point, backend))

        rd_sparse = result_rd.to_sparse()
        lr_to_rd = result_lr.to_rank_decomposition()
        lr_sparse = lr_to_rd.to_sparse()

        assert set(rd_sparse.keys()) == set(lr_sparse.keys())
        for monom in rd_sparse.keys():
            assert_close(rd_sparse[monom], lr_sparse[monom])


class TestLowRankFactorsMultiplication:
    """Test multiplication matches RankDecomposition."""

    def test_multiply_variables_bool(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        num_vars = 2
        xp = Backend(backend).get_array_namespace()

        x_0_rd = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)
        x_1_rd = RankDecomposition.variable(1, num_vars, bool_algebra, backend=backend)
        x_0_lr = LowRankFactors.variable(0, num_vars, bool_algebra, backend=backend)
        x_1_lr = LowRankFactors.variable(1, num_vars, bool_algebra, backend=backend)

        prod_rd = x_0_rd * x_1_rd
        prod_lr = x_0_lr * x_1_lr

        test_points = [
            xp.asarray([bool_algebra.one, bool_algebra.one]),
            xp.asarray([bool_algebra.one, bool_algebra.zero]),
            xp.asarray([bool_algebra.zero, bool_algebra.one]),
            xp.asarray([bool_algebra.zero, bool_algebra.zero]),
        ]

        for point in test_points:
            rd_value = prod_rd.evaluate(point).factors[0, 0, 0].data
            lr_value = prod_lr.evaluate(point).bias[0, 0].data
            assert_equal(rd_value, lr_value)

    def test_multiply_with_constant(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        num_vars = 2
        xp = Backend(backend).get_array_namespace()

        x_0_lr = LowRankFactors.variable(0, num_vars, maxmin_algebra, backend=backend)
        c_lr = LowRankFactors.constant(5.0, num_vars, maxmin_algebra, backend=backend)

        x_0_rd = RankDecomposition.variable(0, num_vars, maxmin_algebra, backend=backend)
        c_rd = RankDecomposition.constant(5.0, num_vars, maxmin_algebra, backend=backend)

        prod_lr = x_0_lr * c_lr
        prod_rd = x_0_rd * c_rd

        test_point = xp.asarray([3.0, 2.0])

        rd_value = prod_rd.evaluate(test_point).factors[0, 0, 0].data
        lr_value = prod_lr.evaluate(test_point).bias[0, 0].data

        assert_close(rd_value, lr_value)

    def test_multiply_with_zero(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        num_vars = 3
        xp = Backend(backend).get_array_namespace()

        const_one = LowRankFactors.constant(bool_algebra.one, num_vars, bool_algebra, backend=backend)
        const_zero = LowRankFactors.zero(num_vars, bool_algebra, backend=backend)

        product = const_one * const_zero

        result = product.evaluate(xp.asarray([bool_algebra.one, bool_algebra.one, bool_algebra.zero]))
        assert_equal(result.bias[0, 0].data, bool_algebra.zero)

    def test_multiply_zero_with_variable(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        num_vars = 3
        xp = Backend(backend).get_array_namespace()

        zero = LowRankFactors.zero(num_vars, bool_algebra, backend=backend)
        x_0 = LowRankFactors.variable(0, num_vars, bool_algebra, backend=backend)

        result = zero * x_0

        test_point = xp.asarray([bool_algebra.one, bool_algebra.one, bool_algebra.zero])
        eval_result = result.evaluate(test_point)
        assert_equal(eval_result.bias[0, 0].data, bool_algebra.zero)


class TestLowRankFactorsEvaluation:
    """Test evaluation matches RankDecomposition."""

    def test_evaluate_variable(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        num_vars = 3

        x_0_rd = RankDecomposition.variable(0, num_vars, maxmin_algebra, backend=backend)
        x_0_lr = LowRankFactors.variable(0, num_vars, maxmin_algebra, backend=backend)

        point = make_array([2.0, 3.0, 4.0], backend)

        rd_value = x_0_rd.evaluate(point).factors[0, 0, 0].data
        lr_value = x_0_lr.evaluate(point).bias[0, 0].data

        assert_close(rd_value, lr_value)

    def test_evaluate_product(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        num_vars = 2

        x_0_rd = RankDecomposition.variable(0, num_vars, maxmin_algebra, backend=backend)
        x_1_rd = RankDecomposition.variable(1, num_vars, maxmin_algebra, backend=backend)
        p_rd = x_0_rd * x_1_rd

        x_0_lr = LowRankFactors.variable(0, num_vars, maxmin_algebra, backend=backend)
        x_1_lr = LowRankFactors.variable(1, num_vars, maxmin_algebra, backend=backend)
        p_lr = x_0_lr * x_1_lr

        point = make_array([2.0, 3.0], backend)

        rd_value = p_rd.evaluate(point).factors[0, 0, 0].data
        lr_value = p_lr.evaluate(point).bias[0, 0].data

        assert_close(rd_value, lr_value)


class TestLowRankFactorsCompose:
    """Test composition matches RankDecomposition."""

    def test_compose_simple(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        num_vars = 2

        x_0_rd = RankDecomposition.variable(0, num_vars, bool_algebra, backend=backend)
        x_1_rd = RankDecomposition.variable(1, num_vars, bool_algebra, backend=backend)

        x_0_lr = LowRankFactors.variable(0, num_vars, bool_algebra, backend=backend)
        x_1_lr = LowRankFactors.variable(1, num_vars, bool_algebra, backend=backend)

        result_rd = x_0_rd.compose([x_1_rd, x_1_rd])
        result_lr = x_0_lr.compose([x_1_lr, x_1_lr])

        test_points = [
            [bool_algebra.one, bool_algebra.one],
            [bool_algebra.one, bool_algebra.zero],
            [bool_algebra.zero, bool_algebra.one],
        ]

        for point in test_points:
            rd_val = result_rd.evaluate(make_array(point, backend)).factors[0, 0, 0].data
            lr_val = result_lr.evaluate(make_array(point, backend)).bias[0, 0].data
            assert_equal(rd_val, lr_val)

    def test_compose_zero_polynomial(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        num_vars = 3

        zero = LowRankFactors.zero(num_vars, bool_algebra, backend=backend)
        const_one = LowRankFactors.constant(bool_algebra.one, num_vars, bool_algebra, backend=backend)

        result = zero.compose([const_one, const_one, const_one])

        eval_result = result.evaluate(make_array([bool_algebra.one, bool_algebra.one, bool_algebra.one], backend))
        assert_equal(eval_result.bias[0, 0].data, bool_algebra.zero)

    def test_compose_product_with_zero_substitution(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        num_vars = 3

        x_0 = LowRankFactors.variable(0, num_vars, bool_algebra, backend=backend)
        x_1 = LowRankFactors.variable(1, num_vars, bool_algebra, backend=backend)
        x_2 = LowRankFactors.variable(2, num_vars, bool_algebra, backend=backend)
        product = x_0 * x_1

        zero = LowRankFactors.zero(num_vars, bool_algebra, backend=backend)
        result = product.compose([x_0, zero, x_2])

        eval_result = result.evaluate(make_array([bool_algebra.one, bool_algebra.one, bool_algebra.zero], backend))
        assert_equal(eval_result.bias[0, 0].data, bool_algebra.zero)


class TestLowRankFactorsConversion:
    """Test round-trip conversions."""

    def test_round_trip_from_rd(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """LowRankFactors -> RankDecomposition -> LowRankFactors preserves evaluation."""
        num_vars = 2

        x_0 = RankDecomposition.variable(0, num_vars, maxmin_algebra, backend=backend)
        x_1 = RankDecomposition.variable(1, num_vars, maxmin_algebra, backend=backend)
        p_rd = x_0 + x_1

        p_lr = LowRankFactors.from_rank_decomposition(p_rd)
        p_rd_back = p_lr.to_rank_decomposition()

        point = make_array([2.0, 3.0], backend)
        rd_value = p_rd.evaluate(point).factors[0, 0, 0].data
        rd_back_value = p_rd_back.evaluate(point).factors[0, 0, 0].data

        assert_close(rd_value, rd_back_value)

    def test_round_trip_from_lr(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """RankDecomposition -> LowRankFactors -> RankDecomposition preserves evaluation."""
        num_vars = 2
        xp = Backend(backend).get_array_namespace()

        x_0 = LowRankFactors.variable(0, num_vars, bool_algebra, backend=backend)
        x_1 = LowRankFactors.variable(1, num_vars, bool_algebra, backend=backend)
        p_lr = x_0 * x_1

        p_rd = p_lr.to_rank_decomposition()
        p_lr_back = LowRankFactors.from_rank_decomposition(p_rd)

        test_points = [
            xp.asarray([bool_algebra.one, bool_algebra.one]),
            xp.asarray([bool_algebra.one, bool_algebra.zero]),
            xp.asarray([bool_algebra.zero, bool_algebra.one]),
        ]

        for point in test_points:
            lr_value = p_lr.evaluate(point).bias[0, 0].data
            lr_back_value = p_lr_back.evaluate(point).bias[0, 0].data
            assert_equal(lr_value, lr_back_value)

    def test_merged_split_round_trip(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        """to_merged -> from_merged preserves factors."""
        num_vars = 3
        x = LowRankFactors.variable(1, num_vars, maxmin_algebra, backend=backend)
        merged = x.to_merged()
        x_back = LowRankFactors.from_merged(merged, x.max_rank, x.max_degree, x.max_replacement_degree, backend=backend)

        point = make_array([1.0, 2.0, 3.0], backend)
        assert_close(
            x.evaluate(point).bias[0, 0].data,
            x_back.evaluate(point).bias[0, 0].data,
        )


class TestLowRankFactorsEdgeCases:
    """Test edge cases."""

    def test_zero_polynomial(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        num_vars = 2
        zero = LowRankFactors.zero(num_vars, maxmin_algebra, backend=backend)

        result = zero.evaluate(make_array([1.0, 2.0], backend))
        assert_close(result.bias[0, 0], maxmin_algebra.zero)

    def test_multilinear_idempotence(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        num_vars = 2
        x_0 = LowRankFactors.variable(0, num_vars, bool_algebra, backend=backend)
        p = x_0 * x_0

        result = p.evaluate(make_array([bool_algebra.one, bool_algebra.zero], backend))
        assert_equal(result.bias[0, 0].data, bool_algebra.one)

        result = p.evaluate(make_array([bool_algebra.zero, bool_algebra.zero], backend))
        assert_equal(result.bias[0, 0].data, bool_algebra.zero)

    def test_large_num_vars(self, maxmin_algebra: DeMorganAlgebra, backend: str) -> None:
        num_vars = 15
        x_0 = LowRankFactors.variable(0, num_vars, maxmin_algebra, backend=backend)
        x_1 = LowRankFactors.variable(1, num_vars, maxmin_algebra, backend=backend)
        x_2 = LowRankFactors.variable(2, num_vars, maxmin_algebra, backend=backend)

        p = (x_0 * x_1) * x_2

        point = make_array(np.ones(15) * 2.0, backend)
        result = p.evaluate(point)
        assert_close(result.bias[0, 0].data, 2.0)


class TestLowRankFactorsJAXTransformations:
    """Test JAX transformations (pytree support)."""

    def test_jit_compilation(self, maxmin_algebra: DeMorganAlgebra, jax_backend: str) -> None:
        import jax

        num_vars = 2
        x_0 = LowRankFactors.variable(0, num_vars, maxmin_algebra, backend=jax_backend)
        x_1 = LowRankFactors.variable(1, num_vars, maxmin_algebra, backend=jax_backend)
        p = x_0 * x_1

        @jax.jit
        def eval_fn(point):
            return LowRankFactors.evaluate(p, point).bias[0, 0].data

        result = eval_fn(make_array([2.0, 3.0], jax_backend))
        assert_close(result, 2.0)

    def test_vmap_evaluation(self, maxmin_algebra: DeMorganAlgebra, jax_backend: str) -> None:
        import jax

        num_vars = 2
        x_0 = LowRankFactors.variable(0, num_vars, maxmin_algebra, backend=jax_backend)

        points = make_array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], jax_backend)

        results = jax.vmap(lambda pt: x_0.evaluate(pt).bias[0, 0].data)(points)

        expected = make_array([1.0, 2.0, 3.0], jax_backend)
        assert_close(results, expected)

    def test_grad_evaluation(self, maxmin_algebra: DeMorganAlgebra, jax_backend: str) -> None:
        import equinox as eqx

        num_vars = 2
        x_0 = LowRankFactors.variable(0, num_vars, maxmin_algebra, backend=jax_backend)
        x_1 = LowRankFactors.variable(1, num_vars, maxmin_algebra, backend=jax_backend)
        p = x_0 * x_1

        def fn(x):
            return LowRankFactors.evaluate(p, x).bias[0, 0].data

        grad_fn = eqx.filter_grad(fn)
        point = make_array([2.0, 3.0], jax_backend)
        gradient = grad_fn(point)

        expected = make_array([1.0, 0.0], jax_backend)
        assert_close(gradient, expected)


class TestLowRankFactorsTorchGradients:
    """Test gradient flow on torch backend."""

    def test_grad_flows_through_weights_and_bias(self) -> None:
        """Gradients must flow independently to weights and bias."""
        import algebraic as alg

        torch = pytest.importorskip("torch")
        bool_alg = alg.semirings.boolean_algebra(mode="soft")

        num_vars = 3
        w = torch.randn(1, 1, num_vars, requires_grad=True)
        b = torch.randn(1, 1, requires_grad=True)

        weights = alg.array(w, semiring=bool_alg, backend="torch")
        bias = alg.array(b, semiring=bool_alg, backend="torch")
        lr = LowRankFactors(weights, bias, backend="torch")

        merged = lr.to_merged()
        assert is_torch_array(merged.data)
        loss = merged.data.sum()
        loss.backward()

        assert w.grad is not None, "gradient must flow to weights"
        assert b.grad is not None, "gradient must flow to bias"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
