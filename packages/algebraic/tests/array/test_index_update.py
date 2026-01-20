"""Tests for AlgebraicArray index update functionality (at[] API)."""

import equinox as eqx
import jax.numpy as jnp
import pytest
import quax
from algebraic import AlgebraicArray
from algebraic.semirings import counting_semiring, tropical_semiring
from algebraic.spec import Ring, Shape
from jaxtyping import Array, Shaped


def ring_spec() -> Ring:
    """Create a standard ring (integers) for testing."""

    def zeros(shape: Shape) -> Shaped[Array, " {shape}"]:
        return jnp.zeros(shape, dtype=jnp.int32)

    def ones(shape: Shape) -> Shaped[Array, " {shape}"]:
        return jnp.ones(shape, dtype=jnp.int32)

    return Ring(
        add=lambda x, y: x + y,
        mul=lambda x, y: x * y,
        zeros=zeros,
        ones=ones,
        additive_inverse=lambda x: -x,
    )


class TestIndexUpdateSet:
    """Test the set() method of index updates."""

    def test_set_scalar(self) -> None:
        """Test setting a single element."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)

        result = a.at[1].set(10.0)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([1.0, 10.0, 3.0, 4.0]))

    def test_set_slice(self) -> None:
        """Test setting a slice."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)

        result = a.at[1:3].set(jnp.array([10.0, 20.0]))

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([1.0, 10.0, 20.0, 4.0]))

    def test_set_2d(self) -> None:
        """Test setting in 2D array."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([[1.0, 2.0], [3.0, 4.0]]), semiring)

        result = a.at[0, 1].set(10.0)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([[1.0, 10.0], [3.0, 4.0]]))

    def test_set_with_algebraic_array(self) -> None:
        """Test setting with an AlgebraicArray value."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)
        values = AlgebraicArray(jnp.array([10.0, 20.0]), semiring)

        result = a.at[1:3].set(values)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([1.0, 10.0, 20.0, 4.0]))


class TestIndexUpdateAdd:
    """Test the add() method of index updates."""

    def test_add_counting_semiring(self) -> None:
        """Test adding to indexed elements with counting semiring."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)

        result = a.at[1].add(10.0)

        assert isinstance(result, AlgebraicArray)
        # Semiring addition: 2.0 + 10.0 = 12.0
        assert jnp.allclose(result.data, jnp.array([1.0, 12.0, 3.0, 4.0]))

    def test_add_tropical_semiring(self) -> None:
        """Test adding with tropical semiring (max operation)."""
        semiring = tropical_semiring(minplus=False)
        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)

        result = a.at[1].add(5.0)

        assert isinstance(result, AlgebraicArray)
        # Tropical addition is max: max(2.0, 5.0) = 5.0
        assert jnp.allclose(result.data, jnp.array([1.0, 5.0, 3.0, 4.0]))

    def test_add_slice(self) -> None:
        """Test adding to a slice."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)

        result = a.at[1:3].add(jnp.array([10.0, 20.0]))

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([1.0, 12.0, 23.0, 4.0]))


class TestIndexUpdateMultiply:
    """Test the multiply() method of index updates."""

    def test_multiply_counting_semiring(self) -> None:
        """Test multiplying indexed elements with counting semiring."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)

        result = a.at[1].multiply(5.0)

        assert isinstance(result, AlgebraicArray)
        # Semiring multiplication: 2.0 * 5.0 = 10.0
        assert jnp.allclose(result.data, jnp.array([1.0, 10.0, 3.0, 4.0]))

    def test_multiply_tropical_semiring(self) -> None:
        """Test multiplying with tropical semiring (addition)."""
        semiring = tropical_semiring(minplus=False)
        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)

        result = a.at[1].multiply(5.0)

        assert isinstance(result, AlgebraicArray)
        # Tropical multiplication is addition: 2.0 + 5.0 = 7.0
        assert jnp.allclose(result.data, jnp.array([1.0, 7.0, 3.0, 4.0]))

    def test_multiply_slice(self) -> None:
        """Test multiplying a slice."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)

        result = a.at[1:3].multiply(jnp.array([5.0, 10.0]))

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([1.0, 10.0, 30.0, 4.0]))


class TestIndexUpdateSubtract:
    """Test the subtract() method of index updates."""

    def test_subtract_ring(self) -> None:
        """Test subtracting from indexed elements with a Ring."""
        semiring = ring_spec()
        a = AlgebraicArray(jnp.array([1.0, 10.0, 3.0, 4.0]), semiring)

        result = a.at[1].subtract(3.0)

        assert isinstance(result, AlgebraicArray)
        # Ring subtraction: 10.0 - 3.0 = 7.0
        assert jnp.allclose(result.data, jnp.array([1.0, 7.0, 3.0, 4.0]))

    def test_subtract_semiring_fails(self) -> None:
        """Test that subtraction fails on plain Semiring."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([1.0, 10.0, 3.0, 4.0]), semiring)

        with pytest.raises(TypeError, match="Subtraction requires a Ring"):
            a.at[1].subtract(3.0)

    def test_subtract_slice(self) -> None:
        """Test subtracting from a slice."""
        semiring = ring_spec()
        a = AlgebraicArray(jnp.array([10.0, 20.0, 30.0, 40.0]), semiring)

        result = a.at[1:3].subtract(jnp.array([5.0, 10.0]))

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([10.0, 15.0, 20.0, 40.0]))


class TestIndexUpdateGet:
    """Test the get() method of index updates."""

    def test_get_single(self) -> None:
        """Test getting a single element."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)

        result = a.at[1].get()

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, 2.0)

    def test_get_slice(self) -> None:
        """Test getting a slice."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)

        result = a.at[1:3].get()

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([2.0, 3.0]))


class TestIndexUpdateApply:
    """Test the apply() method of index updates."""

    def test_apply_function(self) -> None:
        """Test applying a function to indexed elements."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)

        # Square the element at index 1
        result = a.at[1].apply(lambda x: x**2)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([1.0, 4.0, 3.0, 4.0]))

    def test_apply_to_slice(self) -> None:
        """Test applying a function to a slice."""
        semiring = counting_semiring()
        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)

        # Double the elements at indices 1:3
        result = a.at[1:3].apply(lambda x: x * 2)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([1.0, 4.0, 6.0, 4.0]))


class TestIndexUpdateJIT:
    """Test that index updates work with JAX jit."""

    def test_jit_set(self) -> None:
        """Test that set() works under jit."""
        semiring = counting_semiring()

        @eqx.filter_jit
        @quax.quaxify
        def update_fn(a: AlgebraicArray, value: float) -> AlgebraicArray:
            return a.at[1].set(value)

        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)
        result = update_fn(a, 10.0)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([1.0, 10.0, 3.0, 4.0]))

    def test_jit_add(self) -> None:
        """Test that add() works under jit."""
        semiring = counting_semiring()

        @eqx.filter_jit
        @quax.quaxify
        def update_fn(a: AlgebraicArray, value: float) -> AlgebraicArray:
            return a.at[1].add(value)

        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)
        result = update_fn(a, 10.0)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([1.0, 12.0, 3.0, 4.0]))

    def test_jit_multiply(self) -> None:
        """Test that multiply() works under jit."""
        semiring = counting_semiring()

        @eqx.filter_jit
        @quax.quaxify
        def update_fn(a: AlgebraicArray, value: float) -> AlgebraicArray:
            return a.at[1].multiply(value)

        a = AlgebraicArray(jnp.array([1.0, 2.0, 3.0, 4.0]), semiring)
        result = update_fn(a, 5.0)

        assert isinstance(result, AlgebraicArray)
        assert jnp.allclose(result.data, jnp.array([1.0, 10.0, 3.0, 4.0]))
