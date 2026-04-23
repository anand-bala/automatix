"""Tests for batched operations on RankDecomposition and LowRankFactors."""
# ruff: noqa: ANN201, ANN001
# mypy: disable-error-code="no-untyped-call,no-untyped-def,import-not-found"

from __future__ import annotations

import numpy as np
import pytest
from algebraic import BooleanAlgebra
from algebraic.polynomials import LowRankFactors, RankDecomposition
from algebraic.types import Backend
from algebraic.utils.testing import assert_equal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rd_variable(i, num_vars, algebra, backend, batch_shape=()):
    return RankDecomposition.variable(i, num_vars=num_vars, algebra=algebra, backend=backend, batch_shape=batch_shape)


def _lrf_variable(i, num_vars, algebra, backend, batch_shape=()):
    return LowRankFactors.variable(i, num_vars=num_vars, algebra=algebra, backend=backend, batch_shape=batch_shape)


# ---------------------------------------------------------------------------
# RankDecomposition -- shape invariants
# ---------------------------------------------------------------------------


class TestRankDecompositionBatchShape:
    """batch_shape property and factory shapes."""

    def test_unbatched_batch_shape(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        x = _rd_variable(0, num_vars=3, algebra=bool_algebra, backend=backend)
        assert x.batch_shape == ()
        assert x.factors.shape == (1, 1, 4)

    def test_batched_batch_shape(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B = 5
        x = _rd_variable(0, num_vars=3, algebra=bool_algebra, backend=backend, batch_shape=(B,))
        assert x.batch_shape == (B,)
        assert x.factors.shape == (B, 1, 1, 4)

    def test_batched_rank_degree_num_vars(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 3, 4
        x = _rd_variable(0, num_vars=num_vars, algebra=bool_algebra, backend=backend, batch_shape=(B,))
        assert x.rank == 1
        assert x.degree == 1
        assert x.num_vars == num_vars

    def test_constant_batch_shape(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 4, 3
        c = RankDecomposition.constant(bool_algebra.one, num_vars, bool_algebra, backend=backend, batch_shape=(B,))
        assert c.batch_shape == (B,)
        assert c.factors.shape == (B, 1, 1, num_vars + 1)

    def test_zero_one_batch_shape(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 2, 3
        z = RankDecomposition.zero(num_vars, bool_algebra, backend=backend, batch_shape=(B,))
        o = RankDecomposition.one(num_vars, bool_algebra, backend=backend, batch_shape=(B,))
        assert z.batch_shape == (B,)
        assert o.batch_shape == (B,)


# ---------------------------------------------------------------------------
# RankDecomposition -- addition
# ---------------------------------------------------------------------------


class TestRankDecompositionBatchedAdd:
    """Batched __add__ agrees element-wise with the unbatched path."""

    def test_add_shape(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 4, 3
        x0 = _rd_variable(0, num_vars, bool_algebra, backend, batch_shape=(B,))
        x1 = _rd_variable(1, num_vars, bool_algebra, backend, batch_shape=(B,))
        s = x0 + x1
        assert s.batch_shape == (B,)
        assert s.factors.shape[0] == B
        assert s.num_vars == num_vars

    def test_add_evaluate_matches_unbatched(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Batched (x0 + x1) evaluated at each row matches unbatched eval."""
        B, num_vars = 4, 3
        xp = Backend(backend).get_array_namespace()

        x0_batch = _rd_variable(0, num_vars, bool_algebra, backend, batch_shape=(B,))
        x1_batch = _rd_variable(1, num_vars, bool_algebra, backend, batch_shape=(B,))
        s_batch = x0_batch + x1_batch

        x0 = _rd_variable(0, num_vars, bool_algebra, backend)
        x1 = _rd_variable(1, num_vars, bool_algebra, backend)
        s_ref = x0 + x1

        pts_row = np.array([True, False, True], dtype=bool)
        pts_batch = np.tile(pts_row, (B, 1))

        batched_result = s_batch.evaluate(pts_batch)
        ref_result = s_ref.evaluate(xp.asarray(pts_row))

        assert batched_result.shape == (B,)
        for b in range(B):
            assert_equal(batched_result[b], ref_result)

    def test_add_respects_algebra(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """x0 OR x0 == x0 (idempotent add in boolean algebra)."""
        B, num_vars = 3, 2

        x0 = _rd_variable(0, num_vars, bool_algebra, backend, batch_shape=(B,))
        s = x0 + x0

        pts = np.array([[True, False]] * B, dtype=bool)
        result = s.evaluate(pts)
        # x0 OR x0 at (True, False) should be True
        for b in range(B):
            assert_equal(result[b], bool(True))


# ---------------------------------------------------------------------------
# RankDecomposition -- multiplication
# ---------------------------------------------------------------------------


class TestRankDecompositionBatchedMul:
    """Batched __mul__ agrees element-wise with the unbatched path."""

    def test_mul_shape(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 4, 3
        x0 = _rd_variable(0, num_vars, bool_algebra, backend, batch_shape=(B,))
        x1 = _rd_variable(1, num_vars, bool_algebra, backend, batch_shape=(B,))
        p = x0 * x1
        assert p.batch_shape == (B,)
        assert p.factors.shape[0] == B
        assert p.num_vars == num_vars

    def test_mul_evaluate_matches_unbatched(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 4, 3
        xp = Backend(backend).get_array_namespace()

        x0_batch = _rd_variable(0, num_vars, bool_algebra, backend, batch_shape=(B,))
        x1_batch = _rd_variable(1, num_vars, bool_algebra, backend, batch_shape=(B,))
        p_batch = x0_batch * x1_batch

        x0 = _rd_variable(0, num_vars, bool_algebra, backend)
        x1 = _rd_variable(1, num_vars, bool_algebra, backend)
        p_ref = x0 * x1

        pts_row = np.array([True, True, False], dtype=bool)
        pts_batch = np.tile(pts_row, (B, 1))

        batched_result = p_batch.evaluate(pts_batch)
        ref_result = p_ref.evaluate(xp.asarray(pts_row))

        assert batched_result.shape == (B,)
        for b in range(B):
            assert_equal(batched_result[b], ref_result)


# ---------------------------------------------------------------------------
# RankDecomposition -- evaluate
# ---------------------------------------------------------------------------


class TestRankDecompositionBatchedEvaluate:
    """batched evaluate returns Array of shape (B,)."""

    def test_evaluate_shape(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 6, 4
        x = _rd_variable(0, num_vars, bool_algebra, backend, batch_shape=(B,))
        pts = np.zeros((B, num_vars), dtype=bool)
        result = x.evaluate(pts)
        assert result.shape == (B,)

    def test_evaluate_correct_values(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        """Each row of pts is evaluated independently."""
        B, num_vars = 3, 2
        xp = Backend(backend).get_array_namespace()

        x0 = _rd_variable(0, num_vars, bool_algebra, backend, batch_shape=(B,))

        # Row 0: x0=True -> result True
        # Row 1: x0=False -> result False
        # Row 2: x0=True -> result True
        pts = np.array([[True, False], [False, True], [True, True]], dtype=bool)
        result = x0.evaluate(pts)

        assert result.shape == (B,)

        x0_ref = _rd_variable(0, num_vars, bool_algebra, backend)
        for b, row in enumerate(pts):
            ref = x0_ref.evaluate(xp.asarray(row))
            assert_equal(result[b], ref)

    def test_evaluate_all_true(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 4, 3
        x0 = _rd_variable(0, num_vars, bool_algebra, backend, batch_shape=(B,))
        pts = np.ones((B, num_vars), dtype=bool)
        result = x0.evaluate(pts)
        for b in range(B):
            assert_equal(result[b], bool(True))


# ---------------------------------------------------------------------------
# RankDecomposition -- compose
# ---------------------------------------------------------------------------


class TestRankDecompositionBatchedCompose:
    """Batched compose agrees element-wise with the unbatched path."""

    def test_compose_shape(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 3, 2
        x0 = _rd_variable(0, num_vars, bool_algebra, backend, batch_shape=(B,))
        # Replace both vars with unbatched constants
        rep = [
            RankDecomposition.one(num_vars, bool_algebra, backend=backend),
            RankDecomposition.zero(num_vars, bool_algebra, backend=backend),
        ]
        composed = x0.compose(rep)
        assert composed.batch_shape == (B,)

    def test_compose_matches_unbatched(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 4, 3
        xp = Backend(backend).get_array_namespace()

        # poly = (x0 + x1) over a batch
        x0b = _rd_variable(0, num_vars, bool_algebra, backend, batch_shape=(B,))
        x1b = _rd_variable(1, num_vars, bool_algebra, backend, batch_shape=(B,))
        poly_batch = x0b + x1b

        # Replacements: substitute x_i -> x_i (identity compose) using unbatched replacements
        reps = [_rd_variable(i, num_vars, bool_algebra, backend) for i in range(num_vars)]
        composed = poly_batch.compose(reps)
        assert composed.batch_shape == (B,)

        # Result should agree with unbatched (x0 + x1).compose(same reps)
        x0 = _rd_variable(0, num_vars, bool_algebra, backend)
        x1 = _rd_variable(1, num_vars, bool_algebra, backend)
        poly_ref = x0 + x1
        composed_ref = poly_ref.compose(reps)

        pts_row = np.array([True, False, True], dtype=bool)
        pts_batch = np.tile(pts_row, (B, 1))

        batch_result = composed.evaluate(pts_batch)
        ref_result = composed_ref.evaluate(xp.asarray(pts_row))
        for b in range(B):
            assert_equal(batch_result[b], ref_result)


# ---------------------------------------------------------------------------
# RankDecomposition -- guarded operations
# ---------------------------------------------------------------------------


class TestRankDecompositionBatchedGuards:
    """to_sparse and normalize raise ValueError for batched polynomials."""

    def test_to_sparse_raises(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        x = _rd_variable(0, num_vars=3, algebra=bool_algebra, backend=backend, batch_shape=(2,))
        with pytest.raises(ValueError, match="batch"):
            x.to_sparse()

    def test_normalize_raises(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        x = _rd_variable(0, num_vars=3, algebra=bool_algebra, backend=backend, batch_shape=(2,))
        # Force degree > max_degree to trigger normalize path
        x_high = RankDecomposition(
            x.factors,
            max_rank=x.max_rank,
            max_degree=0,  # lower than actual degree so normalize would fire
            backend=backend,
        )
        with pytest.raises(ValueError, match="batch"):
            x_high.normalize()


# ---------------------------------------------------------------------------
# LowRankFactors -- batch_shape property and factory shapes
# ---------------------------------------------------------------------------


class TestLowRankFactorsBatchShape:
    def test_unbatched_batch_shape(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        x = _lrf_variable(0, num_vars=3, algebra=bool_algebra, backend=backend)
        assert x.batch_shape == ()
        assert x.weights.shape == (1, 1, 3)
        assert x.bias.shape == (1, 1)

    def test_batched_batch_shape(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 5, 3
        x = _lrf_variable(0, num_vars, bool_algebra, backend, batch_shape=(B,))
        assert x.batch_shape == (B,)
        assert x.weights.shape == (B, 1, 1, num_vars)
        assert x.bias.shape == (B, 1, 1)

    def test_constant_batch_shape(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 3, 4
        c = LowRankFactors.constant(bool_algebra.one, num_vars, bool_algebra, backend=backend, batch_shape=(B,))
        assert c.batch_shape == (B,)
        assert c.weights.shape == (B, 1, 1, num_vars)
        assert c.bias.shape == (B, 1, 1)


# ---------------------------------------------------------------------------
# LowRankFactors -- merge/split round-trip for batched shapes
# ---------------------------------------------------------------------------


class TestLowRankFactorsMergeSplit:
    """to_merged / from_merged work correctly for batched LowRankFactors."""

    def test_merge_split_round_trip(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 4, 3
        x = _lrf_variable(0, num_vars, bool_algebra, backend, batch_shape=(B,))
        merged = x.to_merged()
        assert merged.shape == (B, 1, 1, num_vars + 1)

        recovered = LowRankFactors.from_merged(merged)
        assert recovered.weights.shape == (B, 1, 1, num_vars)
        assert recovered.bias.shape == (B, 1, 1)

    def test_merge_split_values_preserved(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 3, 2
        x = _lrf_variable(1, num_vars, bool_algebra, backend, batch_shape=(B,))
        merged = x.to_merged()
        recovered = LowRankFactors.from_merged(merged)

        assert_equal(recovered.weights.data, x.weights.data)
        assert_equal(recovered.bias.data, x.bias.data)


# ---------------------------------------------------------------------------
# LowRankFactors -- add / mul / evaluate
# ---------------------------------------------------------------------------


class TestLowRankFactorsBatchedOps:
    """Batched ops on LowRankFactors agree element-wise with unbatched path."""

    def test_add_shape(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 4, 3
        x0 = _lrf_variable(0, num_vars, bool_algebra, backend, batch_shape=(B,))
        x1 = _lrf_variable(1, num_vars, bool_algebra, backend, batch_shape=(B,))
        s = x0 + x1
        assert s.batch_shape == (B,)
        assert s.weights.shape[0] == B

    def test_add_evaluate_matches_unbatched(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 3, 2
        xp = Backend(backend).get_array_namespace()

        x0b = _lrf_variable(0, num_vars, bool_algebra, backend, batch_shape=(B,))
        x1b = _lrf_variable(1, num_vars, bool_algebra, backend, batch_shape=(B,))
        s_batch = x0b + x1b

        x0 = _lrf_variable(0, num_vars, bool_algebra, backend)
        x1 = _lrf_variable(1, num_vars, bool_algebra, backend)
        s_ref = x0 + x1

        pts_row = np.array([True, False], dtype=bool)
        pts_batch = np.tile(pts_row, (B, 1))

        batch_result = s_batch.evaluate(pts_batch)
        ref_result = s_ref.evaluate(xp.asarray(pts_row))

        assert batch_result.shape == (B,)
        for b in range(B):
            assert_equal(batch_result[b], ref_result)

    def test_mul_shape(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 4, 3
        x0 = _lrf_variable(0, num_vars, bool_algebra, backend, batch_shape=(B,))
        x1 = _lrf_variable(1, num_vars, bool_algebra, backend, batch_shape=(B,))
        p = x0 * x1
        assert p.batch_shape == (B,)

    def test_mul_evaluate_matches_unbatched(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 3, 2
        xp = Backend(backend).get_array_namespace()

        x0b = _lrf_variable(0, num_vars, bool_algebra, backend, batch_shape=(B,))
        x1b = _lrf_variable(1, num_vars, bool_algebra, backend, batch_shape=(B,))
        p_batch = x0b * x1b

        x0 = _lrf_variable(0, num_vars, bool_algebra, backend)
        x1 = _lrf_variable(1, num_vars, bool_algebra, backend)
        p_ref = x0 * x1

        pts_row = np.array([True, True], dtype=bool)
        pts_batch = np.tile(pts_row, (B, 1))

        batch_result = p_batch.evaluate(pts_batch)
        ref_result = p_ref.evaluate(xp.asarray(pts_row))

        assert batch_result.shape == (B,)
        for b in range(B):
            assert_equal(batch_result[b], ref_result)

    def test_evaluate_shape(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        B, num_vars = 5, 3
        x = _lrf_variable(0, num_vars, bool_algebra, backend, batch_shape=(B,))
        pts = np.zeros((B, num_vars), dtype=bool)
        result = x.evaluate(pts)
        assert result.shape == (B,)

    def test_normalize_raises_for_batched(self, bool_algebra: BooleanAlgebra, backend: str) -> None:
        x = _lrf_variable(0, num_vars=3, algebra=bool_algebra, backend=backend, batch_shape=(2,))
        lrf_high = LowRankFactors(x.weights, x.bias, max_rank=x.max_rank, max_degree=0, backend=backend)
        with pytest.raises(ValueError, match="batch"):
            lrf_high.normalize()
