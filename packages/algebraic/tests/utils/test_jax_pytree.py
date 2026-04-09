"""Tests for JAX pytree registration of algebraic polynomial types."""
# ruff: noqa: ANN201, ANN001
# mypy: disable-error-code="no-untyped-call,no-untyped-def,import-not-found"

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from algebraic.polynomials import MonomialBasis, PolyDict, RankDecomposition
from algebraic.semirings import boolean_algebra, max_min_algebra
from algebraic.utils.testing import assert_close


@pytest.fixture(autouse=True)
def activate_jax_pytrees() -> None:
    """Importing this module activates JAX pytree registration for all algebraic types."""
    import algebraic.utils.jax  # noqa: F401


class TestRankDecompositionJAX:
    """Test JAX jit and vmap through RankDecomposition."""

    def test_jit_evaluation(self, jax_backend: str) -> None:
        """jit-compiled evaluation should produce correct results."""
        alg = max_min_algebra()
        x0 = RankDecomposition.variable(0, 2, alg, backend=jax_backend)

        result = jax.jit(lambda pt: x0.evaluate(pt).factors[0, 0, 0].data)(jnp.array([2.0, 3.0]))
        assert_close(result, 2.0)

    def test_vmap_evaluation(self, jax_backend: str) -> None:
        """vmap should batch evaluation over a leading axis of points."""
        alg = max_min_algebra()
        x0 = RankDecomposition.variable(0, 2, alg, backend=jax_backend)

        pts = jnp.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        results = jax.vmap(lambda pt: x0.evaluate(pt).factors[0, 0, 0].data)(pts)

        assert_close(results, jnp.array([1.0, 2.0, 3.0]))

    def test_tree_flatten_unflatten_roundtrip(self, jax_backend: str) -> None:
        """Flattening and unflattening should reconstruct an equivalent polynomial."""
        alg = max_min_algebra()
        x0 = RankDecomposition.variable(0, 2, alg, backend=jax_backend)
        p = x0 * x0

        leaves, treedef = jax.tree_util.tree_flatten(p)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        pt = jnp.array([3.0, 1.0])
        assert_close(
            p.evaluate(pt).factors[0, 0, 0].data,
            reconstructed.evaluate(pt).factors[0, 0, 0].data,
        )


class TestPolyDictJAX:
    """Test JAX pytree support for PolyDict."""

    def test_tree_flatten_unflatten_roundtrip(self, jax_backend: str) -> None:
        """Flattening and unflattening should reconstruct an equivalent PolyDict."""
        alg = max_min_algebra()
        x0 = PolyDict.variable(0, 2, algebra=alg, backend=jax_backend)
        x1 = PolyDict.variable(1, 2, algebra=alg, backend=jax_backend)
        p = x0 + x1

        leaves, treedef = jax.tree_util.tree_flatten(p)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        pt = jnp.array([2.0, 5.0])
        original_val = list(p.evaluate(pt).values())[0].data
        reconstructed_val = list(reconstructed.evaluate(pt).values())[0].data
        assert_close(original_val, reconstructed_val)


class TestMonomialBasisJAX:
    """Test JAX pytree support for MonomialBasis."""

    def test_tree_flatten_unflatten_roundtrip(self, jax_backend: str) -> None:
        """Flattening and unflattening should reconstruct an equivalent MonomialBasis."""
        alg = boolean_algebra(mode="logic")
        x0 = MonomialBasis.variable(0, 2, algebra=alg, backend=jax_backend)
        x1 = MonomialBasis.variable(1, 2, algebra=alg, backend=jax_backend)
        p = x0 * x1

        leaves, treedef = jax.tree_util.tree_flatten(p)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        pt = jnp.array([True, True])
        original_val = p.evaluate(pt).coeffs.data
        reconstructed_val = reconstructed.evaluate(pt).coeffs.data
        assert_close(original_val, reconstructed_val)
