"""Tests for AlgebraicArray core functionality, especially dot_general_p overload."""

# mypy: disable-error-code="no-untyped-def"

from __future__ import annotations

import algebraic
import numpy as np
import pytest
from algebraic import AlgebraicArray, Ring
from algebraic.semirings import boolean_algebra, counting_semiring, tropical_semiring
from algebraic.utils.testing import assert_close, assert_equal, make_array


def ring_spec() -> Ring:
    """Create a standard ring (integers) for testing."""
    return Ring(
        add=lambda x, y: x + y,
        mul=lambda x, y: x * y,
        zero=0,
        one=1,
        additive_inverse=lambda x: -x,
    )


class TestDotGeneralWithJNP:
    """Test that AlgebraicArray works with algebraic operations via dot_general_p."""

    def test_vdot_counting_semiring(self, backend: str) -> None:
        """Test algebraic.vecdot with AlgebraicArray using counting semiring."""
        semiring = counting_semiring()

        a_data = make_array([1.0, 2.0, 3.0], backend)
        b_data = make_array([4.0, 5.0, 6.0], backend)

        a = algebraic.array(a_data, semiring=semiring, backend=backend)
        b = algebraic.array(b_data, semiring=semiring, backend=backend)

        result = algebraic.vecdot(a, b)

        # Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        expected = make_array(32.0, backend)

        assert isinstance(result, AlgebraicArray)
        assert_close(result, expected)

    def test_vdot_tropical_semiring(self, backend: str) -> None:
        """Test algebraic.vecdot with AlgebraicArray using tropical (max-plus) semiring."""
        semiring = tropical_semiring(minplus=False)

        a_data = make_array([1.0, 2.0, 3.0], backend)
        b_data = make_array([4.0, 5.0, 6.0], backend)

        a = algebraic.array(a_data, semiring=semiring, backend=backend)
        b = algebraic.array(b_data, semiring=semiring, backend=backend)

        result = algebraic.vecdot(a, b)

        # Expected: max(1+4, 2+5, 3+6) = max(5, 7, 9) = 9
        expected = make_array(9.0, backend)

        assert isinstance(result, AlgebraicArray)
        assert_close(result, expected)

    def test_matmul_counting_semiring(self, backend: str) -> None:
        """Test algebraic.matmul with AlgebraicArray using counting semiring."""
        semiring = counting_semiring()

        a_data = make_array([[1.0, 2.0], [3.0, 4.0]], backend)
        b_data = make_array([[5.0, 6.0], [7.0, 8.0]], backend)

        a = algebraic.array(a_data, semiring=semiring, backend=backend)
        b = algebraic.array(b_data, semiring=semiring, backend=backend)

        result = algebraic.matmul(a, b)

        # Expected standard matrix multiplication
        expected = np.matmul(
            np.asarray(a_data) if not isinstance(a_data, np.ndarray) else a_data,
            np.asarray(b_data) if not isinstance(b_data, np.ndarray) else b_data,
        )

        assert isinstance(result, AlgebraicArray)
        assert_close(result, expected)

    def test_matmul_tropical_semiring(self, backend: str) -> None:
        """Test algebraic.matmul with AlgebraicArray using tropical semiring."""
        semiring = tropical_semiring(minplus=False)

        a_data = make_array([[1.0, 2.0], [3.0, 4.0]], backend)
        b_data = make_array([[5.0, 6.0], [7.0, 8.0]], backend)

        a = algebraic.array(a_data, semiring=semiring, backend=backend)
        b = algebraic.array(b_data, semiring=semiring, backend=backend)

        result = algebraic.matmul(a, b)

        # Expected: result[i,j] = max(a[i,k] + b[k,j] for all k)
        # result[0,0] = max(1+5, 2+7) = max(6, 9) = 9
        # result[0,1] = max(1+6, 2+8) = max(7, 10) = 10
        # result[1,0] = max(3+5, 4+7) = max(8, 11) = 11
        # result[1,1] = max(3+6, 4+8) = max(9, 12) = 12
        expected = make_array([[9.0, 10.0], [11.0, 12.0]], backend)

        assert isinstance(result, AlgebraicArray)
        assert_close(result, expected)

    def test_matmul_boolean_algebra(self, backend: str) -> None:
        """Test @ operator with boolean algebra (regression test for dtype issue).

        This test ensures that matmul works with boolean dtypes, which previously
        failed due to dtype mismatch between semiring.zero (float32) and data (bool).
        """
        bool_alg = boolean_algebra(mode="logic")

        a_data = make_array([[True, False], [False, True]], backend)
        b_data = make_array([[True, True], [False, True]], backend)

        a = algebraic.array(a_data, semiring=bool_alg, backend=backend)
        b = algebraic.array(b_data, semiring=bool_alg, backend=backend)

        result = a @ b

        # Expected: (A @ B)[i,j] = OR_k(A[i,k] AND B[k,j])
        # result[0,0] = (T AND T) OR (F AND F) = T
        # result[0,1] = (T AND T) OR (F AND T) = T
        # result[1,0] = (F AND T) OR (T AND F) = F
        # result[1,1] = (F AND T) OR (T AND T) = T
        expected = make_array([[True, True], [False, True]], backend)

        assert isinstance(result, AlgebraicArray)
        assert_equal(result, expected)

    def test_tensordot_counting_semiring(self, backend: str) -> None:
        """Test algebraic.tensordot with AlgebraicArray using counting semiring."""
        semiring = counting_semiring()

        a_np = np.arange(12.0).reshape(3, 4)
        b_np = np.arange(20.0).reshape(4, 5)

        a_data = make_array(a_np, backend)
        b_data = make_array(b_np, backend)

        a = algebraic.array(a_data, semiring=semiring, backend=backend)
        b = algebraic.array(b_data, semiring=semiring, backend=backend)

        result = algebraic.tensordot(a, b, axes=1)

        expected = np.tensordot(a_np, b_np, axes=1)

        assert isinstance(result, AlgebraicArray)
        assert result.data.shape == expected.shape
        assert_close(result, expected)

    def test_tensordot_tropical_semiring(self, backend: str) -> None:
        """Test algebraic.tensordot with AlgebraicArray using tropical semiring."""
        semiring = tropical_semiring(minplus=False)

        a_data = make_array([[1.0, 2.0], [3.0, 4.0]], backend)
        b_data = make_array([[5.0, 6.0], [7.0, 8.0]], backend)

        a = algebraic.array(a_data, semiring=semiring, backend=backend)
        b = algebraic.array(b_data, semiring=semiring, backend=backend)

        result = algebraic.tensordot(a, b, axes=1)

        # Expected: same as matmul for 2D case with axes=1
        # result[i,j] = max(a[i,k] + b[k,j] for all k)
        expected = make_array([[9.0, 10.0], [11.0, 12.0]], backend)

        assert isinstance(result, AlgebraicArray)
        assert_close(result, expected)

    def test_tensordot_with_axes_specification(self, backend: str) -> None:
        """Test algebraic.tensordot with explicit axes specification."""
        semiring = counting_semiring()

        a_np = np.arange(24.0).reshape(2, 3, 4)
        b_np = np.arange(60.0).reshape(4, 3, 5)

        a_data = make_array(a_np, backend)
        b_data = make_array(b_np, backend)

        a = algebraic.array(a_data, semiring=semiring, backend=backend)
        b = algebraic.array(b_data, semiring=semiring, backend=backend)

        # Contract axes [2, 1] of a with axes [0, 1] of b
        result = algebraic.tensordot(a, b, axes=([2, 1], [0, 1]))

        expected = np.tensordot(a_np, b_np, axes=([2, 1], [0, 1]))

        assert isinstance(result, AlgebraicArray)
        assert result.data.shape == expected.shape
        assert_close(result, expected)

    def test_different_semiring_error(self, backend: str) -> None:
        """Test that operations on arrays with different semirings raise an error."""
        semiring1 = counting_semiring()
        semiring2 = tropical_semiring(minplus=False)

        a = algebraic.array(make_array([1.0, 2.0], backend), semiring=semiring1, backend=backend)
        b = algebraic.array(make_array([3.0, 4.0], backend), semiring=semiring2, backend=backend)

        with pytest.raises(ValueError, match="different semirings|same semiring"):
            algebraic.vecdot(a, b)


class TestArithmeticPrimitives:
    """Test arithmetic primitive overloads."""

    def test_add_counting_semiring(self, backend: str) -> None:
        """Test element-wise addition with counting semiring."""
        semiring = counting_semiring()
        a = algebraic.array(make_array([1.0, 2.0, 3.0], backend), semiring=semiring, backend=backend)
        b = algebraic.array(make_array([4.0, 5.0, 6.0], backend), semiring=semiring, backend=backend)

        result = a + b

        assert isinstance(result, AlgebraicArray)
        assert_close(result, make_array([5.0, 7.0, 9.0], backend))

    def test_add_tropical_semiring(self, backend: str) -> None:
        """Test element-wise addition with tropical semiring (max)."""
        semiring = tropical_semiring(minplus=False)
        a = algebraic.array(make_array([1.0, 5.0, 3.0], backend), semiring=semiring, backend=backend)
        b = algebraic.array(make_array([4.0, 2.0, 6.0], backend), semiring=semiring, backend=backend)

        result = a + b

        # Tropical addition is max
        assert isinstance(result, AlgebraicArray)
        assert_close(result, make_array([4.0, 5.0, 6.0], backend))

    def test_mul_counting_semiring(self, backend: str) -> None:
        """Test element-wise multiplication with counting semiring."""
        semiring = counting_semiring()
        a = algebraic.array(make_array([2.0, 3.0, 4.0], backend), semiring=semiring, backend=backend)
        b = algebraic.array(make_array([5.0, 6.0, 7.0], backend), semiring=semiring, backend=backend)

        result = a * b

        assert isinstance(result, AlgebraicArray)
        assert_close(result, make_array([10.0, 18.0, 28.0], backend))

    def test_mul_tropical_semiring(self, backend: str) -> None:
        """Test element-wise multiplication with tropical semiring (addition)."""
        semiring = tropical_semiring(minplus=False)
        a = algebraic.array(make_array([1.0, 2.0, 3.0], backend), semiring=semiring, backend=backend)
        b = algebraic.array(make_array([4.0, 5.0, 6.0], backend), semiring=semiring, backend=backend)

        result = a * b

        # Tropical multiplication is addition
        assert isinstance(result, AlgebraicArray)
        assert_close(result, make_array([5.0, 7.0, 9.0], backend))

    def test_sub_ring(self, backend: str) -> None:
        """Test subtraction with Ring (has additive inverse)."""
        semiring = ring_spec()
        a = algebraic.array(make_array([5.0, 7.0, 9.0], backend), semiring=semiring, backend=backend)
        b = algebraic.array(make_array([2.0, 3.0, 4.0], backend), semiring=semiring, backend=backend)

        result = a - b

        assert isinstance(result, AlgebraicArray)
        assert_close(result, make_array([3.0, 4.0, 5.0], backend))

    def test_sub_semiring_fails(self, backend: str) -> None:
        """Test that subtraction fails on plain Semiring."""
        semiring = counting_semiring()
        a = algebraic.array(make_array([5.0, 7.0], backend), semiring=semiring, backend=backend)
        b = algebraic.array(make_array([2.0, 3.0], backend), semiring=semiring, backend=backend)

        with pytest.raises(NotImplementedError):
            a - b

    def test_neg_ring(self, backend: str) -> None:
        """Test negation with Ring (additive inverse)."""
        semiring = ring_spec()
        a = algebraic.array(make_array([1.0, -2.0, 3.0], backend), semiring=semiring, backend=backend)

        result = -a

        assert isinstance(result, AlgebraicArray)
        assert_close(result, make_array([-1.0, 2.0, -3.0], backend))

    def test_neg_boolean_algebra(self, backend: str) -> None:
        """Test negation with Boolean algebra (complement)."""
        semiring = boolean_algebra(mode="ste")
        a = algebraic.array(make_array([1.0, 0.0, 1.0], backend), semiring=semiring, backend=backend)

        result = -a

        # Complement: 1 - x
        assert isinstance(result, AlgebraicArray)
        assert_close(result, make_array([0.0, 1.0, 0.0], backend))

    def test_neg_semiring_fails(self, backend: str) -> None:
        """Test that negation fails on plain Semiring."""
        semiring = tropical_semiring(minplus=False)
        a = algebraic.array(make_array([1.0, 2.0], backend), semiring=semiring, backend=backend)

        with pytest.raises(NotImplementedError):
            _ = -a

    def test_ops_different_semiring_error(self, backend: str) -> None:
        """Test that algebraic.add/multiply fail with mismatched semirings."""
        semiring1 = counting_semiring()
        semiring2 = tropical_semiring(minplus=False)
        a = algebraic.array(make_array([1.0, 2.0], backend), semiring=semiring1, backend=backend)
        b = algebraic.array(make_array([3.0, 4.0], backend), semiring=semiring2, backend=backend)

        with pytest.raises(ValueError):
            algebraic.add(a, b)

        with pytest.raises(ValueError):
            algebraic.multiply(a, b)


class TestCumulativeOperations:
    """Test cumulative operations (cumsum, cumprod)."""

    def test_cumsum_counting_semiring(self, backend: str) -> None:
        """Test cumsum with counting semiring."""
        semiring = counting_semiring()
        a = algebraic.array(make_array([1.0, 2.0, 3.0, 4.0], backend), semiring=semiring, backend=backend)

        result = algebraic.cumulative_sum(a)

        # Expected: [1, 1+2, 1+2+3, 1+2+3+4] = [1, 3, 6, 10]
        assert isinstance(result, AlgebraicArray)
        assert_close(result, make_array([1.0, 3.0, 6.0, 10.0], backend))

    def test_cumsum_tropical_semiring(self, backend: str) -> None:
        """Test cumsum with tropical semiring (cumulative max)."""
        semiring = tropical_semiring(minplus=False)
        a = algebraic.array(make_array([1.0, 5.0, 3.0, 7.0], backend), semiring=semiring, backend=backend)

        result = algebraic.cumulative_sum(a)

        # Tropical addition is max, so cumsum becomes cumulative max
        # Expected: [1, max(1,5), max(1,5,3), max(1,5,3,7)] = [1, 5, 5, 7]
        assert isinstance(result, AlgebraicArray)
        assert_close(result, make_array([1.0, 5.0, 5.0, 7.0], backend))

    def test_cumprod_counting_semiring(self, backend: str) -> None:
        """Test cumprod with counting semiring."""
        semiring = counting_semiring()
        a = algebraic.array(make_array([1.0, 2.0, 3.0, 4.0], backend), semiring=semiring, backend=backend)

        result = algebraic.cumulative_prod(a)

        # Expected: [1, 1*2, 1*2*3, 1*2*3*4] = [1, 2, 6, 24]
        assert isinstance(result, AlgebraicArray)
        assert_close(result, make_array([1.0, 2.0, 6.0, 24.0], backend))

    def test_cumprod_tropical_semiring(self, backend: str) -> None:
        """Test cumprod with tropical semiring (cumulative sum)."""
        semiring = tropical_semiring(minplus=False)
        a = algebraic.array(make_array([1.0, 5.0, 3.0, 7.0], backend), semiring=semiring, backend=backend)

        result = algebraic.cumulative_prod(a)

        # Tropical multiplication is addition, so cumprod becomes cumulative sum
        # Expected: [1, 1+5, 1+5+3, 1+5+3+7] = [1, 6, 9, 16]
        assert isinstance(result, AlgebraicArray)
        assert_close(result, make_array([1.0, 6.0, 9.0, 16.0], backend))


class TestAlgebraicArrayBasics:
    """Test basic AlgebraicArray functionality."""

    def test_zeros(self, backend: str) -> None:
        """Test zeros creation."""
        semiring = counting_semiring()
        z = algebraic.zeros((3, 4), semiring=semiring, backend=backend)

        assert isinstance(z, AlgebraicArray)
        assert z.data.shape == (3, 4)
        assert_close(z, np.zeros((3, 4)))

    def test_ones(self, backend: str) -> None:
        """Test ones creation."""
        semiring = counting_semiring()
        o = algebraic.ones((3, 4), semiring=semiring, backend=backend)

        assert isinstance(o, AlgebraicArray)
        assert o.data.shape == (3, 4)
        assert_close(o, np.ones((3, 4)))
