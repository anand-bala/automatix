"""Tests for AlgebraicArray core functionality, especially dot_general_p overload."""

import jax.numpy as jnp
import pytest
import quax
from algebraic.array.core import AlgebraicArray, ones, zeros
from algebraic.semirings import boolean_algebra, counting_semiring, tropical_semiring
from algebraic.spec import Ring


def ring_spec() -> Ring:
    """Create a standard ring (integers) for testing."""
    return Ring(
        add=lambda x, y: x + y,
        mul=lambda x, y: x * y,
        zero=jnp.asarray(0.0),
        one=jnp.asarray(1.0),
        additive_inverse=lambda x: -x,
    )


class TestDotGeneralWithJNP:
    """Test that AlgebraicArray works with jax.numpy operations via dot_general_p."""

    def test_vdot_counting_semiring(self) -> None:
        """Test jnp.vdot with AlgebraicArray using counting semiring."""
        semiring = counting_semiring()

        # Create test vectors
        a_data = jnp.array([1.0, 2.0, 3.0])
        b_data = jnp.array([4.0, 5.0, 6.0])

        a = AlgebraicArray(a_data, semiring)
        b = AlgebraicArray(b_data, semiring)

        # Compute using jnp.vdot (should use our dot_general_p overload)
        result = quax.quaxify(jnp.vdot)(a, b)  # type: ignore[arg-type]

        # Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        expected = jnp.asarray(32.0)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, expected)

    def test_vdot_tropical_semiring(self) -> None:
        """Test jnp.vdot with AlgebraicArray using tropical (max-plus) semiring."""
        semiring = tropical_semiring(minplus=False)

        # Create test vectors
        a_data = jnp.array([1.0, 2.0, 3.0])
        b_data = jnp.array([4.0, 5.0, 6.0])

        a = AlgebraicArray(a_data, semiring)
        b = AlgebraicArray(b_data, semiring)

        # Compute using jnp.vdot
        result = quax.quaxify(jnp.vdot)(a, b)  # type: ignore[arg-type]

        # Expected: max(1+4, 2+5, 3+6) = max(5, 7, 9) = 9
        expected = jnp.asarray(9.0)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, expected)

    def test_matmul_counting_semiring(self) -> None:
        """Test jnp.matmul with AlgebraicArray using counting semiring."""
        semiring = counting_semiring()

        # Create test matrices
        a_data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        a = AlgebraicArray(a_data, semiring)
        b = AlgebraicArray(b_data, semiring)

        # Compute using jnp.matmul
        result = quax.quaxify(jnp.matmul)(a, b)  # type: ignore[arg-type]

        # Expected standard matrix multiplication
        expected = jnp.matmul(a_data, b_data)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, expected)

    def test_matmul_tropical_semiring(self) -> None:
        """Test jnp.matmul with AlgebraicArray using tropical semiring."""
        semiring = tropical_semiring(minplus=False)

        # Create test matrices
        a_data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        a = AlgebraicArray(a_data, semiring)
        b = AlgebraicArray(b_data, semiring)

        # Compute using jnp.matmul
        result = quax.quaxify(jnp.matmul)(a, b)  # type: ignore[arg-type]

        # Expected: result[i,j] = max(a[i,k] + b[k,j] for all k)
        # result[0,0] = max(1+5, 2+7) = max(6, 9) = 9
        # result[0,1] = max(1+6, 2+8) = max(7, 10) = 10
        # result[1,0] = max(3+5, 4+7) = max(8, 11) = 11
        # result[1,1] = max(3+6, 4+8) = max(9, 12) = 12
        expected = jnp.array([[9.0, 10.0], [11.0, 12.0]])

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, expected)

    def test_matmul_boolean_algebra(self) -> None:
        """Test @ operator with boolean algebra (regression test for dtype issue).

        This test ensures that matmul works with boolean dtypes, which previously
        failed due to dtype mismatch between semiring.zero (float32) and data (bool).
        """
        bool_alg = boolean_algebra(mode="logic")

        # Create boolean matrices
        a_data = jnp.array([[True, False], [False, True]])
        b_data = jnp.array([[True, True], [False, True]])

        a = AlgebraicArray(a_data, bool_alg)
        b = AlgebraicArray(b_data, bool_alg)

        # Compute using @ operator
        result = a @ b

        # Expected: (A @ B)[i,j] = OR_k(A[i,k] AND B[k,j])
        # result[0,0] = (T AND T) OR (F AND F) = T
        # result[0,1] = (T AND T) OR (F AND T) = T
        # result[1,0] = (F AND T) OR (T AND F) = F
        # result[1,1] = (F AND T) OR (T AND T) = T
        expected = jnp.array([[True, True], [False, True]])

        assert isinstance(result, AlgebraicArray)
        assert jnp.array_equal(result.data, expected)

    def test_tensordot_counting_semiring(self) -> None:
        """Test jnp.tensordot with AlgebraicArray using counting semiring."""
        semiring = counting_semiring()

        # Create test tensors
        a_data = jnp.arange(12.0).reshape(3, 4)
        b_data = jnp.arange(20.0).reshape(4, 5)

        a = AlgebraicArray(a_data, semiring)
        b = AlgebraicArray(b_data, semiring)

        # Compute using jnp.tensordot (contract last axis of a with first axis of b)
        result = quax.quaxify(jnp.tensordot)(a, b, axes=1)  # type: ignore[arg-type]

        # Expected standard tensordot
        expected = jnp.tensordot(a_data, b_data, axes=1)

        assert isinstance(result, AlgebraicArray)
        assert result.data.shape == expected.shape
        assert jnp.allclose(result.data, expected)

    def test_tensordot_tropical_semiring(self) -> None:
        """Test jnp.tensordot with AlgebraicArray using tropical semiring."""
        semiring = tropical_semiring(minplus=False)

        # Create simple test tensors
        a_data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        a = AlgebraicArray(a_data, semiring)
        b = AlgebraicArray(b_data, semiring)

        # Compute using jnp.tensordot
        result = quax.quaxify(jnp.tensordot)(a, b, axes=1)  # type: ignore[arg-type]

        # Expected: same as matmul for 2D case with axes=1
        # result[i,j] = max(a[i,k] + b[k,j] for all k)
        expected = jnp.array([[9.0, 10.0], [11.0, 12.0]])

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, expected)

    def test_tensordot_with_axes_specification(self) -> None:
        """Test jnp.tensordot with explicit axes specification."""
        semiring = counting_semiring()

        # Create 3D tensors
        a_data = jnp.arange(24.0).reshape(2, 3, 4)
        b_data = jnp.arange(60.0).reshape(4, 3, 5)

        a = AlgebraicArray(a_data, semiring)
        b = AlgebraicArray(b_data, semiring)

        # Contract axes [2, 1] of a with axes [0, 1] of b
        result = quax.quaxify(jnp.tensordot)(a, b, axes=([2, 1], [0, 1]))  # type: ignore[arg-type]

        # Expected
        expected = jnp.tensordot(a_data, b_data, axes=([2, 1], [0, 1]))

        assert isinstance(result, AlgebraicArray)
        assert result.data.shape == expected.shape
        assert jnp.allclose(result.data, expected)

    def test_einsum_matrix_multiply(self) -> None:
        """Test jnp.einsum matrix multiplication with AlgebraicArray."""
        semiring = counting_semiring()

        a_data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        a = AlgebraicArray(a_data, semiring)
        b = AlgebraicArray(b_data, semiring)

        # Matrix multiplication using einsum
        result = quax.quaxify(jnp.einsum)("ij,jk->ik", a, b)

        expected = jnp.matmul(a_data, b_data)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, expected)

    def test_einsum_trace(self) -> None:
        """Test jnp.einsum for trace calculation."""
        semiring = counting_semiring()

        a_data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        a = AlgebraicArray(a_data, semiring)

        # Trace: sum of diagonal elements
        result = quax.quaxify(jnp.einsum)("ii->", a)

        # Expected: 1 + 5 + 9 = 15
        expected = jnp.asarray(15.0)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, expected)

    def test_einsum_outer_product(self) -> None:
        """Test jnp.einsum for outer product."""
        semiring = counting_semiring()

        a_data = jnp.array([1.0, 2.0, 3.0])
        b_data = jnp.array([4.0, 5.0])

        a = AlgebraicArray(a_data, semiring)
        b = AlgebraicArray(b_data, semiring)

        # Outer product
        result = quax.quaxify(jnp.einsum)("i,j->ij", a, b)

        expected = jnp.outer(a_data, b_data)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, expected)

    def test_einsum_tropical_semiring(self) -> None:
        """Test jnp.einsum with tropical semiring."""
        semiring = tropical_semiring(minplus=False)

        a_data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b_data = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        a = AlgebraicArray(a_data, semiring)
        b = AlgebraicArray(b_data, semiring)

        # Matrix multiplication using einsum
        result = quax.quaxify(jnp.einsum)("ij,jk->ik", a, b)

        # Expected: max-plus matrix multiplication
        expected = jnp.array([[9.0, 10.0], [11.0, 12.0]])

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, expected)

    def test_different_semiring_error(self) -> None:
        """Test that operations on arrays with different semirings raise an error."""
        semiring1 = counting_semiring()
        semiring2 = tropical_semiring(minplus=False)

        a = AlgebraicArray(jnp.array([1.0, 2.0]), semiring1)
        b = AlgebraicArray(jnp.array([3.0, 4.0]), semiring2)

        # Should raise ValueError
        with pytest.raises(ValueError, match="different semirings"):
            quax.quaxify(jnp.vdot)(a, b)  # type: ignore[arg-type]

    def test_batched_matmul(self) -> None:
        """Test batched matrix multiplication."""
        semiring = counting_semiring()

        # Create batched matrices (batch_size=2, 3x4 and 4x5)
        a_data = jnp.arange(24.0).reshape(2, 3, 4)
        b_data = jnp.arange(40.0).reshape(2, 4, 5)

        a = AlgebraicArray(a_data, semiring)
        b = AlgebraicArray(b_data, semiring)

        # Batched matmul
        result = quax.quaxify(jnp.einsum)("bij,bjk->bik", a, b)

        # Expected
        expected = jnp.einsum("bij,bjk->bik", a_data, b_data)

        assert isinstance(result, AlgebraicArray)
        assert result.data.shape == expected.shape
        assert jnp.allclose(result.data, expected)


class TestArithmeticPrimitives:
    """Test arithmetic primitive overloads."""

    def test_add_counting_semiring(self) -> None:
        """Test element-wise addition with counting semiring."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0]), semiring)
        b = AlgebraicArray(jnp.array([4.0, 5.0, 6.0]), semiring)

        result = quax.quaxify(lambda x, y: x + y)(a, b)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([5.0, 7.0, 9.0]))

    def test_add_tropical_semiring(self) -> None:
        """Test element-wise addition with tropical semiring (max)."""
        semiring = tropical_semiring(minplus=False)
        a = AlgebraicArray(jnp.array([1.0, 5.0, 3.0]), semiring)
        b = AlgebraicArray(jnp.array([4.0, 2.0, 6.0]), semiring)

        result = quax.quaxify(lambda x, y: x + y)(a, b)

        # Tropical addition is max
        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([4.0, 5.0, 6.0]))

    def test_mul_counting_semiring(self) -> None:
        """Test element-wise multiplication with counting semiring."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([2.0, 3.0, 4.0]), semiring)
        b = AlgebraicArray(jnp.array([5.0, 6.0, 7.0]), semiring)

        result = quax.quaxify(lambda x, y: x * y)(a, b)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([10.0, 18.0, 28.0]))

    def test_mul_tropical_semiring(self) -> None:
        """Test element-wise multiplication with tropical semiring (addition)."""
        semiring = tropical_semiring(minplus=False)
        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0]), semiring)
        b = AlgebraicArray(jnp.array([4.0, 5.0, 6.0]), semiring)

        result = quax.quaxify(lambda x, y: x * y)(a, b)

        # Tropical multiplication is addition
        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([5.0, 7.0, 9.0]))

    def test_sub_ring(self) -> None:
        """Test subtraction with Ring (has additive inverse)."""
        semiring = ring_spec()
        a = AlgebraicArray(jnp.array([5.0, 7.0, 9.0]), semiring)
        b = AlgebraicArray(jnp.array([2.0, 3.0, 4.0]), semiring)

        result = quax.quaxify(lambda x, y: x - y)(a, b)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([3.0, 4.0, 5.0]))

    def test_sub_semiring_fails(self) -> None:
        """Test that subtraction fails on plain Semiring."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([5.0, 7.0]), semiring)
        b = AlgebraicArray(jnp.array([2.0, 3.0]), semiring)

        with pytest.raises(TypeError, match="Subtraction requires a Ring"):
            quax.quaxify(lambda x, y: x - y)(a, b)

    def test_neg_ring(self) -> None:
        """Test negation with Ring (additive inverse)."""
        semiring = ring_spec()
        a = AlgebraicArray(jnp.array([1.0, -2.0, 3.0]), semiring)

        result = quax.quaxify(lambda x: -x)(a)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([-1.0, 2.0, -3.0]))

    def test_neg_boolean_algebra(self) -> None:
        """Test negation with Boolean algebra (complement)."""
        semiring = boolean_algebra(mode="ste")
        a = AlgebraicArray(jnp.array([1.0, 0.0, 1.0]), semiring)

        result = quax.quaxify(lambda x: -x)(a)

        # Complement: 1 - x
        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([0.0, 1.0, 0.0]))

    def test_neg_semiring_fails(self) -> None:
        """Test that negation fails on plain Semiring."""
        semiring = tropical_semiring(minplus=False)
        a = AlgebraicArray(jnp.array([1.0, 2.0]), semiring)

        with pytest.raises(TypeError, match="Negation requires either"):
            quax.quaxify(lambda x: -x)(a)

    def test_arithmetic_different_semiring_error(self) -> None:
        """Test that arithmetic ops fail with mismatched semirings."""
        semiring1 = counting_semiring()
        semiring2 = tropical_semiring(minplus=False)
        a = AlgebraicArray(jnp.array([1.0, 2.0]), semiring1)
        b = AlgebraicArray(jnp.array([3.0, 4.0]), semiring2)

        with pytest.raises(ValueError, match="different semirings"):
            quax.quaxify(lambda x, y: x + y)(a, b)

        with pytest.raises(ValueError, match="different semirings"):
            quax.quaxify(lambda x, y: x * y)(a, b)


class TestCumulativeOperations:
    """Test cumulative operations (cumsum, cumprod)."""

    def test_cumsum_counting_semiring(self) -> None:
        """Test cumsum with counting semiring."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)

        result = quax.quaxify(jnp.cumsum)(a)

        # Expected: [1, 1+2, 1+2+3, 1+2+3+4] = [1, 3, 6, 10]
        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([1.0, 3.0, 6.0, 10.0]))

    def test_cumsum_tropical_semiring(self) -> None:
        """Test cumsum with tropical semiring (cumulative max)."""
        semiring = tropical_semiring(minplus=False)
        a = AlgebraicArray(jnp.array([1.0, 5.0, 3.0, 7.0]), semiring)

        result = quax.quaxify(jnp.cumsum)(a)

        # Tropical addition is max, so cumsum becomes cumulative max
        # Expected: [1, max(1,5), max(1,5,3), max(1,5,3,7)] = [1, 5, 5, 7]
        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([1.0, 5.0, 5.0, 7.0]))

    def test_cumprod_counting_semiring(self) -> None:
        """Test cumprod with counting semiring."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)

        result = quax.quaxify(jnp.cumprod)(a)

        # Expected: [1, 1*2, 1*2*3, 1*2*3*4] = [1, 2, 6, 24]
        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([1.0, 2.0, 6.0, 24.0]))

    def test_cumprod_tropical_semiring(self) -> None:
        """Test cumprod with tropical semiring (cumulative sum)."""
        semiring = tropical_semiring(minplus=False)
        a = AlgebraicArray(jnp.array([1.0, 5.0, 3.0, 7.0]), semiring)

        result = quax.quaxify(jnp.cumprod)(a)

        # Tropical multiplication is addition, so cumprod becomes cumulative sum
        # Expected: [1, 1+5, 1+5+3, 1+5+3+7] = [1, 6, 9, 16]
        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([1.0, 6.0, 9.0, 16.0]))


class TestAlgebraicArrayBasics:
    """Test basic AlgebraicArray functionality."""

    def test_zeros(self) -> None:
        """Test zeros creation."""
        semiring = counting_semiring()
        z = zeros((3, 4), semiring)

        assert isinstance(z, AlgebraicArray)
        assert z.data.shape == (3, 4)
        assert jnp.allclose(z.data, 0.0)

    def test_ones(self) -> None:
        """Test ones creation."""
        semiring = counting_semiring()
        o = ones((3, 4), semiring)

        assert isinstance(o, AlgebraicArray)
        assert o.data.shape == (3, 4)
        assert jnp.allclose(o.data, 1.0)

    def test_aval(self) -> None:
        """Test aval returns correct shaped array."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([[1.0, 2.0], [3.0, 4.0]]), semiring)

        aval = a.aval()
        assert aval.shape == (2, 2)
        assert aval.dtype == jnp.float32 or aval.dtype == jnp.float64

