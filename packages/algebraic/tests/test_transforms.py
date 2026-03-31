"""Tests for backend-agnostic jit and vmap transformations."""
# mypy: disable-error-code="no-untyped-def"

from __future__ import annotations

import algebraic
import algebraic.utils.jax
import numpy as np
import pytest
from algebraic import AlgebraicArray
from algebraic._jax_wrappers import jit
from algebraic.semirings import counting_semiring
from algebraic.transforms import vmap


class TestJit:
    """Test jit wrapper."""

    def test_jit_jax(self, jax_backend: str) -> None:
        semiring = counting_semiring()

        @jit(backend=jax_backend)
        def add_arrays(x: AlgebraicArray, y: AlgebraicArray) -> AlgebraicArray:
            return x + y

        a = algebraic.zeros((3, 3), semiring=semiring, backend=jax_backend)
        b = algebraic.ones((3, 3), semiring=semiring, backend=jax_backend)
        result = add_arrays(a, b)
        assert isinstance(result, AlgebraicArray)
        assert result.semiring is semiring

    def test_jit_numpy_passthrough(self) -> None:
        """Numpy jit is a no-op passthrough."""
        semiring = counting_semiring()

        @jit(backend="numpy")
        def add_arrays(x: AlgebraicArray, y: AlgebraicArray) -> AlgebraicArray:
            return x + y

        a = algebraic.zeros((3, 3), semiring=semiring, backend="numpy")
        b = algebraic.ones((3, 3), semiring=semiring, backend="numpy")
        result = add_arrays(a, b)
        assert isinstance(result, AlgebraicArray)
        assert result.semiring is semiring

    def test_jit_with_arguments(self, jax_backend: str) -> None:
        """Test that jit works with static_argnames (JAX-only)."""
        semiring = counting_semiring()

        @jit(backend=jax_backend)
        def scale_array(x: AlgebraicArray, multiplier: int) -> AlgebraicArray:
            result = x
            for _ in range(multiplier):
                result = result + x
            return result

        a = algebraic.ones((3, 3), semiring=semiring, backend=jax_backend)
        result = scale_array(a, multiplier=2)
        assert isinstance(result, AlgebraicArray)
        assert result.semiring is semiring


class TestVmap:
    """Test vmap wrapper (JAX only -- torch.vmap does not support non-Tensor args)."""

    def test_vmap_basic(self, jax_backend: str) -> None:
        semiring = counting_semiring()
        one_arr = algebraic.ones((3, 3), semiring=semiring, backend=jax_backend)

        @vmap(backend=jax_backend, in_axes=(0, None))
        def batch_add(x: AlgebraicArray, one: AlgebraicArray) -> AlgebraicArray:
            return x + one

        batch = algebraic.zeros((5, 3, 3), semiring=semiring, backend=jax_backend)
        result = batch_add(batch, one_arr)
        assert isinstance(result, AlgebraicArray)
        assert result.shape == (5, 3, 3)
        assert result.semiring is semiring

    def test_vmap_in_axes(self, jax_backend: str) -> None:
        semiring = counting_semiring()

        @vmap(backend=jax_backend, in_axes=0)
        def add_arrays(x: AlgebraicArray, y: AlgebraicArray) -> AlgebraicArray:
            return x + y

        a = algebraic.zeros((5, 3, 3), semiring=semiring, backend=jax_backend)
        b = algebraic.ones((5, 3, 3), semiring=semiring, backend=jax_backend)
        result = add_arrays(a, b)
        assert isinstance(result, AlgebraicArray)
        assert result.shape == (5, 3, 3)

    def test_vmap_numpy_raises(self) -> None:
        with pytest.raises(NotImplementedError):

            @vmap(backend="numpy")
            def f(x: AlgebraicArray) -> AlgebraicArray:
                return x + x


class TestCombinedTransformations:
    """Test combining jit and vmap (JAX only)."""

    def test_jit_vmap(self, jax_backend: str) -> None:
        semiring = counting_semiring()
        one_arr = algebraic.ones((3, 3), semiring=semiring, backend=jax_backend)

        @jit(backend=jax_backend)
        @vmap(backend=jax_backend, in_axes=(0, None))
        def batch_add(x: AlgebraicArray, one: AlgebraicArray) -> AlgebraicArray:
            return x + one

        batch = algebraic.zeros((5, 3, 3), semiring=semiring, backend=jax_backend)
        result = batch_add(batch, one_arr)
        assert isinstance(result, AlgebraicArray)
        assert result.shape == (5, 3, 3)

    def test_vmap_jit(self, jax_backend: str) -> None:
        semiring = counting_semiring()
        one_arr = algebraic.ones((3, 3), semiring=semiring, backend=jax_backend)

        @vmap(backend=jax_backend, in_axes=(0, None))
        @jit(backend=jax_backend)
        def batch_add(x: AlgebraicArray, one: AlgebraicArray) -> AlgebraicArray:
            return x + one

        batch = algebraic.zeros((5, 3, 3), semiring=semiring, backend=jax_backend)
        result = batch_add(batch, one_arr)
        assert isinstance(result, AlgebraicArray)
        assert result.shape == (5, 3, 3)


class TestJitIndexUpdates:
    """Test index updates under JAX jit."""

    def test_jit_set(self, jax_backend: str) -> None:
        semiring = counting_semiring()

        @jit(backend=jax_backend)
        def update_fn(a: AlgebraicArray) -> AlgebraicArray:
            return a.at[1].set(10.0)

        a = algebraic.array([1.0, 2.0, 3.0, 4.0], semiring=semiring, backend=jax_backend)
        result = update_fn(a)
        assert isinstance(result, AlgebraicArray)
        np.testing.assert_allclose(np.asarray(result.data), [1.0, 10.0, 3.0, 4.0])

    def test_jit_add(self, jax_backend: str) -> None:
        semiring = counting_semiring()

        @jit(backend=jax_backend)
        def update_fn(a: AlgebraicArray) -> AlgebraicArray:
            return a.at[1].add(10.0)

        a = algebraic.array([1.0, 2.0, 3.0, 4.0], semiring=semiring, backend=jax_backend)
        result = update_fn(a)
        np.testing.assert_allclose(np.asarray(result.data), [1.0, 12.0, 3.0, 4.0])

    def test_jit_multiply(self, jax_backend: str) -> None:
        semiring = counting_semiring()

        @jit(backend=jax_backend)
        def update_fn(a: AlgebraicArray) -> AlgebraicArray:
            return a.at[1].multiply(5.0)

        a = algebraic.array([1.0, 2.0, 3.0, 4.0], semiring=semiring, backend=jax_backend)
        result = update_fn(a)
        np.testing.assert_allclose(np.asarray(result.data), [1.0, 10.0, 3.0, 4.0])
