"""Tests for algebraic.einsum -- semiring-aware Einstein summation."""

# mypy: disable-error-code="no-untyped-def"

from __future__ import annotations

import algebraic
import numpy as np
import pytest
from algebraic import AlgebraicArray
from algebraic.semirings import (
    boolean_algebra,
    counting_semiring,
    max_min_algebra,
    tropical_semiring,
)
from algebraic.utils.testing import assert_close, make_array


class TestEinsumBinaryContraction:
    """Two-operand contractions with counting semiring (matches numpy.einsum)."""

    def test_matmul(self, backend: str) -> None:
        sr = counting_semiring()
        a = algebraic.array(make_array([[1.0, 2.0], [3.0, 4.0]], backend), semiring=sr)
        b = algebraic.array(make_array([[5.0, 6.0], [7.0, 8.0]], backend), semiring=sr)
        result = algebraic.einsum("ij,jk->ik", a, b)
        expected = np.einsum("ij,jk->ik", [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]])
        assert isinstance(result, AlgebraicArray)
        assert result.shape == (2, 2)
        assert_close(result, expected)

    def test_outer_product(self, backend: str) -> None:
        sr = counting_semiring()
        x = algebraic.array(make_array([1.0, 2.0, 3.0], backend), semiring=sr)
        y = algebraic.array(make_array([4.0, 5.0], backend), semiring=sr)
        result = algebraic.einsum("i,j->ij", x, y)
        expected = np.einsum("i,j->ij", [1.0, 2.0, 3.0], [4.0, 5.0])
        assert result.shape == (3, 2)
        assert_close(result, expected)

    def test_batched_matmul(self, backend: str) -> None:
        sr = counting_semiring()
        a_np = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
        b_np = np.arange(2 * 4 * 5, dtype=float).reshape(2, 4, 5)
        a = algebraic.array(make_array(a_np, backend), semiring=sr)
        b = algebraic.array(make_array(b_np, backend), semiring=sr)
        result = algebraic.einsum("bij,bjk->bik", a, b)
        expected = np.einsum("bij,bjk->bik", a_np, b_np)
        assert result.shape == (2, 3, 5)
        assert_close(result, expected)

    def test_elementwise(self, backend: str) -> None:
        sr = counting_semiring()
        a = algebraic.array(make_array([[1.0, 2.0], [3.0, 4.0]], backend), semiring=sr)
        b = algebraic.array(make_array([[5.0, 6.0], [7.0, 8.0]], backend), semiring=sr)
        result = algebraic.einsum("ab,ab->ab", a, b)
        expected = np.einsum("ab,ab->ab", [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]])
        assert_close(result, expected)

    def test_full_reduction(self, backend: str) -> None:
        sr = counting_semiring()
        a = algebraic.array(make_array([[1.0, 2.0], [3.0, 4.0]], backend), semiring=sr)
        b = algebraic.array(make_array([[5.0, 6.0], [7.0, 8.0]], backend), semiring=sr)
        result = algebraic.einsum("ab,ab->", a, b)
        expected = np.einsum("ab,ab->", [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]])
        assert result.shape == ()
        assert_close(result, expected)

    def test_vecdot(self, backend: str) -> None:
        sr = counting_semiring()
        a = algebraic.array(make_array([1.0, 2.0, 3.0], backend), semiring=sr)
        b = algebraic.array(make_array([4.0, 5.0, 6.0], backend), semiring=sr)
        result = algebraic.einsum("i,i->", a, b)
        expected = np.einsum("i,i->", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert result.shape == ()
        assert_close(result, expected)


class TestEinsumUnary:
    """Single-operand einsum (trace, diagonal, transpose, reduction)."""

    def test_trace(self, backend: str) -> None:
        sr = counting_semiring()
        a = algebraic.array(make_array([[1.0, 2.0], [3.0, 4.0]], backend), semiring=sr)
        result = algebraic.einsum("ii->", a)
        expected = np.einsum("ii->", [[1.0, 2.0], [3.0, 4.0]])
        assert result.shape == ()
        assert_close(result, expected)

    def test_diagonal(self, backend: str) -> None:
        sr = counting_semiring()
        a = algebraic.array(make_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], backend), semiring=sr)
        result = algebraic.einsum("ii->i", a)
        expected = np.einsum("ii->i", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        assert result.shape == (3,)
        assert_close(result, expected)

    def test_transpose(self, backend: str) -> None:
        sr = counting_semiring()
        a_np = np.arange(6, dtype=float).reshape(2, 3)
        a = algebraic.array(make_array(a_np, backend), semiring=sr)
        result = algebraic.einsum("ij->ji", a)
        expected = np.einsum("ij->ji", a_np)
        assert result.shape == (3, 2)
        assert_close(result, expected)

    def test_sum_all_axes(self, backend: str) -> None:
        sr = counting_semiring()
        a = algebraic.array(make_array([[1.0, 2.0], [3.0, 4.0]], backend), semiring=sr)
        result = algebraic.einsum("ij->", a)
        expected = np.einsum("ij->", [[1.0, 2.0], [3.0, 4.0]])
        assert result.shape == ()
        assert_close(result, expected)

    def test_sum_one_axis(self, backend: str) -> None:
        sr = counting_semiring()
        a_np = np.arange(6, dtype=float).reshape(2, 3)
        a = algebraic.array(make_array(a_np, backend), semiring=sr)
        result = algebraic.einsum("ij->i", a)
        expected = np.einsum("ij->i", a_np)
        assert result.shape == (2,)
        assert_close(result, expected)

    def test_repeated_index_with_reduction(self, backend: str) -> None:
        sr = counting_semiring()
        a_np = np.arange(27, dtype=float).reshape(3, 3, 3)
        a = algebraic.array(make_array(a_np, backend), semiring=sr)
        result = algebraic.einsum("iji->j", a)
        expected = np.einsum("iji->j", a_np)
        assert result.shape == (3,)
        assert_close(result, expected)


class TestEinsumMultiOperand:
    """Multi-operand (3+) contraction chains."""

    def test_three_operand_chain(self, backend: str) -> None:
        sr = counting_semiring()
        np.random.seed(42)
        a_np, b_np, c_np = np.random.randn(3, 4), np.random.randn(4, 5), np.random.randn(5, 2)
        a = algebraic.array(make_array(a_np, backend), semiring=sr)
        b = algebraic.array(make_array(b_np, backend), semiring=sr)
        c = algebraic.array(make_array(c_np, backend), semiring=sr)
        result = algebraic.einsum("ij,jk,kl->il", a, b, c)
        expected = np.einsum("ij,jk,kl->il", a_np, b_np, c_np)
        assert result.shape == (3, 2)
        assert_close(result, expected)

    def test_four_operand(self, backend: str) -> None:
        sr = counting_semiring()
        np.random.seed(0)
        a_np = np.random.randn(2, 3)
        b_np = np.random.randn(3, 4)
        c_np = np.random.randn(4, 5)
        d_np = np.random.randn(5, 2)
        a = algebraic.array(make_array(a_np, backend), semiring=sr)
        b = algebraic.array(make_array(b_np, backend), semiring=sr)
        c = algebraic.array(make_array(c_np, backend), semiring=sr)
        d = algebraic.array(make_array(d_np, backend), semiring=sr)
        result = algebraic.einsum("ij,jk,kl,lm->im", a, b, c, d)
        expected = np.einsum("ij,jk,kl,lm->im", a_np, b_np, c_np, d_np)
        assert result.shape == (2, 2)
        assert_close(result, expected)


class TestEinsumImplicitOutput:
    """Implicit output subscripts (no ``->`` in the equation)."""

    def test_implicit_matmul(self, backend: str) -> None:
        sr = counting_semiring()
        a = algebraic.array(make_array([[1.0, 2.0], [3.0, 4.0]], backend), semiring=sr)
        b = algebraic.array(make_array([[5.0, 6.0], [7.0, 8.0]], backend), semiring=sr)
        result = algebraic.einsum("ij,jk", a, b)
        expected = np.einsum("ij,jk", [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]])
        assert_close(result, expected)

    def test_implicit_trace(self, backend: str) -> None:
        sr = counting_semiring()
        a = algebraic.array(make_array([[1.0, 2.0], [3.0, 4.0]], backend), semiring=sr)
        result = algebraic.einsum("ii", a)
        expected = np.einsum("ii", [[1.0, 2.0], [3.0, 4.0]])
        assert result.shape == ()
        assert_close(result, expected)


class TestEinsumTropicalSemiring:
    """Einsum with tropical (min-plus / max-plus) semirings."""

    def test_tropical_minplus_matmul(self, backend: str) -> None:
        sr = tropical_semiring(minplus=True)
        a = algebraic.array(make_array([[1.0, 2.0], [3.0, 4.0]], backend), semiring=sr)
        b = algebraic.array(make_array([[5.0, 6.0], [7.0, 8.0]], backend), semiring=sr)
        result = algebraic.einsum("ij,jk->ik", a, b)
        # min-plus: result[i,k] = min_j(a[i,j] + b[j,k])
        expected = np.array([[6.0, 7.0], [8.0, 9.0]])
        assert_close(result, expected)

    def test_tropical_maxplus_matmul(self, backend: str) -> None:
        sr = tropical_semiring(minplus=False)
        a = algebraic.array(make_array([[1.0, 2.0], [3.0, 4.0]], backend), semiring=sr)
        b = algebraic.array(make_array([[5.0, 6.0], [7.0, 8.0]], backend), semiring=sr)
        result = algebraic.einsum("ij,jk->ik", a, b)
        # max-plus: result[i,k] = max_j(a[i,j] + b[j,k])
        expected = np.array([[9.0, 10.0], [11.0, 12.0]])
        assert_close(result, expected)

    def test_tropical_three_operand(self, backend: str) -> None:
        sr = tropical_semiring(minplus=True)
        a = algebraic.array(make_array([[1.0, 2.0], [3.0, 4.0]], backend), semiring=sr)
        b = algebraic.array(make_array([[5.0, 6.0], [7.0, 8.0]], backend), semiring=sr)
        c = algebraic.array(make_array([[1.0], [2.0]], backend), semiring=sr)
        result = algebraic.einsum("ij,jk,kl->il", a, b, c)
        # Step 1: A @ B (min-plus) = [[6,7],[8,9]]
        # Step 2: [[6,7],[8,9]] @ [[1],[2]] (min-plus) = [[min(7,9)],[min(9,11)]] = [[7],[9]]
        expected = np.array([[7.0], [9.0]])
        assert_close(result, expected)


class TestEinsumMaxMinAlgebra:
    """Einsum with max-min algebra (robustness semantics)."""

    def test_maxmin_matmul(self, backend: str) -> None:
        sr = max_min_algebra(smooth=False)
        a = algebraic.array(make_array([[0.2, 0.8], [0.5, 0.3]], backend), semiring=sr)
        b = algebraic.array(make_array([[0.9, 0.1], [0.4, 0.7]], backend), semiring=sr)
        result = algebraic.einsum("ij,jk->ik", a, b)
        # add=max, mul=min: result[i,k] = max_j(min(a[i,j], b[j,k]))
        # [0,0]: max(min(0.2,0.9), min(0.8,0.4)) = max(0.2, 0.4) = 0.4
        # [0,1]: max(min(0.2,0.1), min(0.8,0.7)) = max(0.1, 0.7) = 0.7
        # [1,0]: max(min(0.5,0.9), min(0.3,0.4)) = max(0.5, 0.3) = 0.5
        # [1,1]: max(min(0.5,0.1), min(0.3,0.7)) = max(0.1, 0.3) = 0.3
        expected = np.array([[0.4, 0.7], [0.5, 0.3]])
        assert_close(result, expected, atol=1e-5)


class TestEinsumBooleanAlgebra:
    """Einsum with Boolean algebra (logic mode for exact results)."""

    def test_boolean_matmul(self, backend: str) -> None:
        sr = boolean_algebra(mode="logic")
        a = algebraic.array(make_array([[1.0, 0.0], [1.0, 1.0]], backend), semiring=sr)
        b = algebraic.array(make_array([[0.0, 1.0], [1.0, 0.0]], backend), semiring=sr)
        result = algebraic.einsum("ij,jk->ik", a, b)
        # add=max(OR), mul=min(AND): result[i,k] = OR_j(a[i,j] AND b[j,k])
        # [0,0]: (1 AND 0) OR (0 AND 1) = 0
        # [0,1]: (1 AND 1) OR (0 AND 0) = 1
        # [1,0]: (1 AND 0) OR (1 AND 1) = 1
        # [1,1]: (1 AND 1) OR (1 AND 0) = 1
        expected = np.array([[0.0, 1.0], [1.0, 1.0]])
        assert_close(result, expected)


class TestEinsumValidation:
    """Edge cases and error handling."""

    def test_no_operands_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one operand"):
            algebraic.einsum("ij->ij")

    def test_semiring_mismatch_raises(self, backend: str) -> None:
        sr1 = counting_semiring()
        sr2 = tropical_semiring(minplus=True)
        a = algebraic.array(make_array([[1.0, 2.0]], backend), semiring=sr1)
        b = algebraic.array(make_array([[3.0], [4.0]], backend), semiring=sr2)
        with pytest.raises(ValueError, match="same semiring"):
            algebraic.einsum("ij,jk->ik", a, b)

    def test_preserves_semiring(self, backend: str) -> None:
        sr = tropical_semiring(minplus=True)
        a = algebraic.array(make_array([[1.0, 2.0], [3.0, 4.0]], backend), semiring=sr)
        b = algebraic.array(make_array([[5.0, 6.0], [7.0, 8.0]], backend), semiring=sr)
        result = algebraic.einsum("ij,jk->ik", a, b)
        assert result.semiring is sr

    def test_single_operand_identity(self, backend: str) -> None:
        sr = counting_semiring()
        a_np = np.arange(6, dtype=float).reshape(2, 3)
        a = algebraic.array(make_array(a_np, backend), semiring=sr)
        result = algebraic.einsum("ij->ij", a)
        assert_close(result, a_np)
