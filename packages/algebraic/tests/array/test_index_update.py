"""Tests for AlgebraicArray index update functionality (at[] API)."""
# mypy: disable-error-code="type-arg,operator,list-item"

from __future__ import annotations

import algebraic
import numpy as np
import pytest
from algebraic import AlgebraicArray, Ring
from algebraic.semirings import counting_semiring, tropical_semiring


def ring_spec() -> Ring:
    """Create a standard ring (integers) for testing."""
    return Ring(
        add=lambda x, y: x + y,
        mul=lambda x, y: x * y,
        zero=0,
        one=1,
        additive_inverse=lambda x: -x,
    )


class TestIndexUpdateSet:
    """Test the set() method of index updates."""

    def test_set_scalar(self, backend: str) -> None:
        """Test setting a single element."""
        semiring = counting_semiring()
        a = algebraic.array([1.0, 2.0, 3.0, 4.0], semiring=semiring, backend=backend)

        result = a.at[1].set(10.0)

        assert isinstance(result, AlgebraicArray)
        np.testing.assert_allclose(np.asarray(result.data), [1.0, 10.0, 3.0, 4.0])

    def test_set_slice(self, backend: str) -> None:
        """Test setting a slice."""
        semiring = counting_semiring()
        a = algebraic.array([1.0, 2.0, 3.0, 4.0], semiring=semiring, backend=backend)

        result = a.at[1:3].set(algebraic.array([10.0, 20.0], semiring=semiring, backend=backend))

        assert isinstance(result, AlgebraicArray)
        np.testing.assert_allclose(np.asarray(result.data), [1.0, 10.0, 20.0, 4.0])

    def test_set_2d(self, backend: str) -> None:
        """Test setting in 2D array."""
        semiring = counting_semiring()
        a = algebraic.array([[1.0, 2.0], [3.0, 4.0]], semiring=semiring, backend=backend)

        result = a.at[0, 1].set(10.0)

        assert isinstance(result, AlgebraicArray)
        np.testing.assert_allclose(np.asarray(result.data), [[1.0, 10.0], [3.0, 4.0]])

    def test_set_with_algebraic_array(self, backend: str) -> None:
        """Test setting with an AlgebraicArray value."""
        semiring = counting_semiring()
        a = algebraic.array([1.0, 2.0, 3.0, 4.0], semiring=semiring, backend=backend)
        values = algebraic.array([10.0, 20.0], semiring=semiring, backend=backend)

        result = a.at[1:3].set(values)

        assert isinstance(result, AlgebraicArray)
        np.testing.assert_allclose(np.asarray(result.data), [1.0, 10.0, 20.0, 4.0])


class TestIndexUpdateAdd:
    """Test the add() method of index updates."""

    def test_add_counting_semiring(self, backend: str) -> None:
        """Test adding to indexed elements with counting semiring."""
        semiring = counting_semiring()
        a = algebraic.array([1.0, 2.0, 3.0, 4.0], semiring=semiring, backend=backend)

        result = a.at[1].add(10.0)

        assert isinstance(result, AlgebraicArray)
        # Semiring addition: 2.0 + 10.0 = 12.0
        np.testing.assert_allclose(np.asarray(result.data), [1.0, 12.0, 3.0, 4.0])

    def test_add_tropical_semiring(self, backend: str) -> None:
        """Test adding with tropical semiring (max operation)."""
        semiring = tropical_semiring(minplus=False)
        a = algebraic.array([1.0, 2.0, 3.0, 4.0], semiring=semiring, backend=backend)

        result = a.at[1].add(5.0)

        assert isinstance(result, AlgebraicArray)
        # Tropical addition is max: max(2.0, 5.0) = 5.0
        np.testing.assert_allclose(np.asarray(result.data), [1.0, 5.0, 3.0, 4.0])

    def test_add_slice(self, backend: str) -> None:
        """Test adding to a slice."""
        semiring = counting_semiring()
        a = algebraic.array([1.0, 2.0, 3.0, 4.0], semiring=semiring, backend=backend)

        result = a.at[1:3].add(algebraic.array([10.0, 20.0], semiring=semiring, backend=backend))

        assert isinstance(result, AlgebraicArray)
        np.testing.assert_allclose(np.asarray(result.data), [1.0, 12.0, 23.0, 4.0])


class TestIndexUpdateMultiply:
    """Test the multiply() method of index updates."""

    def test_multiply_counting_semiring(self, backend: str) -> None:
        """Test multiplying indexed elements with counting semiring."""
        semiring = counting_semiring()
        a = algebraic.array([1.0, 2.0, 3.0, 4.0], semiring=semiring, backend=backend)

        result = a.at[1].multiply(5.0)

        assert isinstance(result, AlgebraicArray)
        # Semiring multiplication: 2.0 * 5.0 = 10.0
        np.testing.assert_allclose(np.asarray(result.data), [1.0, 10.0, 3.0, 4.0])

    def test_multiply_tropical_semiring(self, backend: str) -> None:
        """Test multiplying with tropical semiring (addition)."""
        semiring = tropical_semiring(minplus=False)
        a = algebraic.array([1.0, 2.0, 3.0, 4.0], semiring=semiring, backend=backend)

        result = a.at[1].multiply(5.0)

        assert isinstance(result, AlgebraicArray)
        # Tropical multiplication is addition: 2.0 + 5.0 = 7.0
        np.testing.assert_allclose(np.asarray(result.data), [1.0, 7.0, 3.0, 4.0])

    def test_multiply_slice(self, backend: str) -> None:
        """Test multiplying a slice."""
        semiring = counting_semiring()
        a = algebraic.array([1.0, 2.0, 3.0, 4.0], semiring=semiring, backend=backend)

        result = a.at[1:3].multiply(algebraic.array([5.0, 10.0], semiring=semiring, backend=backend))

        assert isinstance(result, AlgebraicArray)
        np.testing.assert_allclose(np.asarray(result.data), [1.0, 10.0, 30.0, 4.0])


class TestIndexUpdateSubtract:
    """Test the subtract() method of index updates."""

    def test_subtract_ring(self, backend: str) -> None:
        """Test subtracting from indexed elements with a Ring."""
        semiring = ring_spec()
        a = algebraic.array([1.0, 10.0, 3.0, 4.0], semiring=semiring, backend=backend)

        result = a.at[1].subtract(3.0)

        assert isinstance(result, AlgebraicArray)
        # Ring subtraction: 10.0 - 3.0 = 7.0
        np.testing.assert_allclose(np.asarray(result.data), [1.0, 7.0, 3.0, 4.0])

    def test_subtract_semiring_fails(self, backend: str) -> None:
        """Test that subtraction fails on plain Semiring."""
        semiring = counting_semiring()
        a = algebraic.array([1.0, 10.0, 3.0, 4.0], semiring=semiring, backend=backend)

        with pytest.raises(TypeError):
            a.at[1].subtract(3.0)

    def test_subtract_slice(self, backend: str) -> None:
        """Test subtracting from a slice."""
        semiring = ring_spec()
        a = algebraic.array([10.0, 20.0, 30.0, 40.0], semiring=semiring, backend=backend)

        result = a.at[1:3].subtract(algebraic.array([5.0, 10.0], semiring=semiring, backend=backend))

        assert isinstance(result, AlgebraicArray)
        np.testing.assert_allclose(np.asarray(result.data), [10.0, 15.0, 20.0, 40.0])


class TestIndexUpdateGet:
    """Test the get() method of index updates."""

    def test_get_single(self, backend: str) -> None:
        """Test getting a single element."""
        semiring = counting_semiring()
        a = algebraic.array([1.0, 2.0, 3.0, 4.0], semiring=semiring, backend=backend)

        result = a.at[1].get()

        assert isinstance(result, AlgebraicArray)
        np.testing.assert_allclose(np.asarray(result.data), 2.0)

    def test_get_slice(self, backend: str) -> None:
        """Test getting a slice."""
        semiring = counting_semiring()
        a = algebraic.array([1.0, 2.0, 3.0, 4.0], semiring=semiring, backend=backend)

        result = a.at[1:3].get()

        assert isinstance(result, AlgebraicArray)
        np.testing.assert_allclose(np.asarray(result.data), [2.0, 3.0])


class TestIndexUpdateApply:
    """Test the apply() method of index updates."""

    def test_apply_function(self, backend: str) -> None:
        """Test applying a function to indexed elements."""
        semiring = counting_semiring()
        a = algebraic.array([1.0, 2.0, 3.0, 4.0], semiring=semiring, backend=backend)

        # Square the element at index 1
        result = a.at[1].apply(lambda x: x**2)

        assert isinstance(result, AlgebraicArray)
        np.testing.assert_allclose(np.asarray(result.data), [1.0, 4.0, 3.0, 4.0])

    def test_apply_to_slice(self, backend: str) -> None:
        """Test applying a function to a slice."""
        semiring = counting_semiring()
        a = algebraic.array([1.0, 2.0, 3.0, 4.0], semiring=semiring, backend=backend)

        # Double the elements at indices 1:3
        result = a.at[1:3].apply(lambda x: x * 2)

        assert isinstance(result, AlgebraicArray)
        np.testing.assert_allclose(np.asarray(result.data), [1.0, 4.0, 6.0, 4.0])
