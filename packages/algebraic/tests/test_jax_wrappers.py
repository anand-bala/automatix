"""Tests for algebraic._jax_wrappers module."""

from algebraic import AlgebraicArray, Semiring
from algebraic._jax_wrappers import jit, vmap
from algebraic.numpy import ones, zeros
from algebraic.semirings import counting_semiring


class TestJit:
    """Test jit wrapper with AlgebraicArray."""

    def test_jit_with_algebraic_array(self) -> None:
        """Test that jit works with AlgebraicArray without explicit quaxify."""
        semiring = counting_semiring()

        @jit
        def add_arrays(x: AlgebraicArray[Semiring], y: AlgebraicArray[Semiring]) -> AlgebraicArray[Semiring]:
            return x + y

        a = zeros((3, 3), semiring)
        b = ones((3, 3), semiring)

        result = add_arrays(a, b)

        assert isinstance(result, AlgebraicArray)
        assert result.semiring is semiring

    def test_jit_with_arguments(self) -> None:
        """Test that jit works with static_argnames and other arguments."""
        semiring = counting_semiring()

        @jit(static_argnames=["multiplier"])
        def scale_array(x: AlgebraicArray[Semiring], multiplier: int) -> AlgebraicArray[Semiring]:
            result = x
            for _ in range(multiplier):
                result = result + x
            return result

        a = ones((3, 3), semiring)
        result = scale_array(a, multiplier=2)

        assert isinstance(result, AlgebraicArray)
        assert result.semiring is semiring


class TestVmap:
    """Test vmap wrapper with AlgebraicArray."""

    def test_vmap_with_algebraic_array(self) -> None:
        """Test that vmap works with AlgebraicArray without explicit quaxify."""
        semiring = counting_semiring()
        one_array = ones((3, 3), semiring)

        @vmap(in_axes=(0, None))
        def batch_add(x: AlgebraicArray[Semiring], one: AlgebraicArray[Semiring]) -> AlgebraicArray[Semiring]:
            return x + one

        # Create a batch of arrays
        batch = zeros((5, 3, 3), semiring)

        result = batch_add(batch, one_array)

        assert isinstance(result, AlgebraicArray)
        assert result.shape == (5, 3, 3)
        assert result.semiring is semiring

    def test_vmap_with_in_axes(self) -> None:
        """Test that vmap works with custom in_axes."""
        semiring = counting_semiring()

        @vmap(in_axes=0)
        def add_arrays(x: AlgebraicArray[Semiring], y: AlgebraicArray[Semiring]) -> AlgebraicArray[Semiring]:
            return x + y

        a = zeros((5, 3, 3), semiring)
        b = ones((5, 3, 3), semiring)

        result = add_arrays(a, b)

        assert isinstance(result, AlgebraicArray)
        assert result.shape == (5, 3, 3)
        assert result.semiring is semiring


class TestCombinedTransformations:
    """Test combining multiple transformations."""

    def test_jit_vmap_composition(self) -> None:
        """Test that jit and vmap can be composed."""
        semiring = counting_semiring()
        one_array = ones((3, 3), semiring)

        @jit
        @vmap(in_axes=(0, None))
        def batch_add(x: AlgebraicArray[Semiring], one: AlgebraicArray[Semiring]) -> AlgebraicArray[Semiring]:
            return x + one

        batch = zeros((5, 3, 3), semiring)
        result = batch_add(batch, one_array)

        assert isinstance(result, AlgebraicArray)
        assert result.shape == (5, 3, 3)
        assert result.semiring is semiring

    def test_vmap_jit_composition(self) -> None:
        """Test that vmap and jit can be composed in reverse order."""
        semiring = counting_semiring()
        one_array = ones((3, 3), semiring)

        @vmap(in_axes=(0, None))
        @jit
        def batch_add(x: AlgebraicArray[Semiring], one: AlgebraicArray[Semiring]) -> AlgebraicArray[Semiring]:
            return x + one

        batch = zeros((5, 3, 3), semiring)
        result = batch_add(batch, one_array)

        assert isinstance(result, AlgebraicArray)
        assert result.shape == (5, 3, 3)
        assert result.semiring is semiring
