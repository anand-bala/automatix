"""Tests for algebraic.numpy module."""

import algebraic.numpy as alge
import pytest
from algebraic import AlgebraicArray
from algebraic.semirings import counting_semiring, tropical_semiring


class TestAlgebraicNumpyBasics:
    """Test basic functionality of algebraic.numpy module."""

    def test_add_with_algebraic_arrays(self) -> None:
        """Test that add works with AlgebraicArray without explicit quaxify."""
        semiring = counting_semiring()
        a = alge.zeros((3, 3), semiring)
        b = alge.zeros((3, 3), semiring)

        result = alge.add(a, b)

        assert isinstance(result, AlgebraicArray)
        assert result.semiring is semiring

    def test_matmul_with_algebraic_arrays(self) -> None:
        """Test that matmul works with AlgebraicArray without explicit quaxify."""
        semiring = counting_semiring()
        a = alge.zeros((3, 3), semiring)
        b = alge.zeros((3, 3), semiring)

        result = alge.matmul(a, b)

        assert isinstance(result, AlgebraicArray)
        assert result.semiring is semiring

    def test_sum_with_algebraic_array(self) -> None:
        """Test that sum works with AlgebraicArray without explicit quaxify."""
        semiring = tropical_semiring(minplus=False)
        a = alge.zeros((3, 3), semiring)

        result = alge.sum(a)

        assert isinstance(result, AlgebraicArray)
        assert result.semiring is semiring

    def test_transpose_with_algebraic_array(self) -> None:
        """Test that transpose works with AlgebraicArray without explicit quaxify."""
        semiring = counting_semiring()
        a = alge.zeros((3, 4), semiring)

        result = alge.transpose(a)

        assert isinstance(result, AlgebraicArray)
        assert result.shape == (4, 3)
        assert result.semiring is semiring

    def test_reshape_with_algebraic_array(self) -> None:
        """Test that reshape works with AlgebraicArray without explicit quaxify."""
        semiring = counting_semiring()
        a = alge.zeros((3, 4), semiring)

        result = alge.reshape(a, (12,))

        assert isinstance(result, AlgebraicArray)
        assert result.shape == (12,)
        assert result.semiring is semiring

    def test_multiply_with_algebraic_arrays(self) -> None:
        """Test that multiply works with AlgebraicArray without explicit quaxify."""
        semiring = counting_semiring()
        a = alge.zeros((3, 3), semiring)
        b = alge.zeros((3, 3), semiring)

        result = alge.multiply(a, b)

        assert isinstance(result, AlgebraicArray)
        assert result.semiring is semiring

    def test_cumsum_with_algebraic_array(self) -> None:
        """Test that cumsum works with AlgebraicArray without explicit quaxify."""
        semiring = counting_semiring()
        a = alge.zeros((5,), semiring)

        result = alge.cumsum(a)

        assert isinstance(result, AlgebraicArray)
        assert result.shape == (5,)
        assert result.semiring is semiring


class TestAlgebraicNumpyExclusions:
    def test_eye_raises_attribute_error(self) -> None:
        """Test that eye raises TypeError with helpful message."""
        with pytest.raises(TypeError) as exc_info:
            alge.eye(3)  # type: ignore[call-arg]

        assert "missing" in str(exc_info.value)
        assert "dtype" in str(exc_info.value)

    def test_array_raises_attribute_error(self) -> None:
        """Test that array raises TypeError with helpful message."""
        with pytest.raises(TypeError) as exc_info:
            alge.array([1, 2, 3])  # type: ignore[attr-defined]

        assert "missing" in str(exc_info.value)
        assert "dtype" in str(exc_info.value)


class TestAlgebraicNumpyInvalidAttributes:
    """Test that invalid attributes raise appropriate errors."""

    def test_nonexistent_function_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent functions raises AttributeError."""
        with pytest.raises(AttributeError) as exc_info:
            alge.this_function_does_not_exist()  # type: ignore[attr-defined]

        assert "not available" in str(exc_info.value)
        assert "this_function_does_not_exist" in str(exc_info.value)
