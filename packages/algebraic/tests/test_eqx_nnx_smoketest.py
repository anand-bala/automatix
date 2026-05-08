"""Pytree-registration round-trips for ``AlgebraicArray`` /
``LowRankFactors`` under ``jax.jit``, ``flax.nnx.jit``, and
``eqx.filter_jit``.

The algebraic library registers
:class:`~algebraic.array.AlgebraicArray` /
:class:`~algebraic.polynomials.LowRankFactors` as JAX pytrees via the
side-effect import ``algebraic.utils.jax``. The ``tree_unflatten``
classmethods bypass ``__init__`` so they can round-trip arbitrary leaf
values, including:

* concrete arrays (``jax.jit``),
* ``jax.Tracer`` objects (``nnx.jit``'s
  ``flax.nnx.extract.clear_non_graph_nodes`` preprocessing,
  ``eqx.filter_jit``'s output reconstruction),
* Python ``bool`` filter-mask leaves (``eqx.filter_jit``'s
  partition pass).

Without the ``__new__``-based bypass, ``__init__``/``__post_init__``
validation (``is_array(data)``, ``weights.shape[-1]``, etc.) rejected
those leaves and broke the jit boundary in either direction. These
tests pin the round-trip behaviour for all three jit flavours.
"""

from __future__ import annotations

# Side-effect import: registers algebraic dataclasses with
# ``jax.tree_util``. Without this, ``LowRankFactors`` is a leaf
# (treedef ``PyTreeDef(*)``), which is a *separate* failure mode
# from what is exercised below; covered by
# ``test_registration_makes_lowrank_a_pytree``.
import algebraic.utils.jax  # noqa: F401
import equinox as eqx
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest
from algebraic.array import AlgebraicArray
from algebraic.polynomials import LowRankFactors
from algebraic.semirings import boolean_algebra

ALGEBRA = boolean_algebra(mode="soft")


def _make_lowrank(weights: jax.Array, bias: jax.Array) -> LowRankFactors:
    """Wrap raw arrays into a JAX-backed :class:`LowRankFactors`."""
    return LowRankFactors(
        weights=AlgebraicArray(data=weights, semiring=ALGEBRA),
        bias=AlgebraicArray(data=bias, semiring=ALGEBRA),
        max_rank=int(weights.shape[-3]),
        max_degree=int(weights.shape[-2]),
        backend="jax",
    )


# ---------------------------------------------------------------------------
# Sanity: registration is in effect
# ---------------------------------------------------------------------------


class TestRegistration:
    """``algebraic.utils.jax`` registers the dataclasses with ``jax.tree_util``."""

    def test_registration_makes_lowrank_a_pytree(self) -> None:
        """``LowRankFactors`` flattens to its two array leaves."""
        lr = _make_lowrank(
            jnp.zeros((2, 2, 4), dtype=jnp.float32),
            jnp.zeros((2, 2), dtype=jnp.float32),
        )
        leaves = jax.tree_util.tree_leaves(lr)
        assert len(leaves) == 2, "expected weights.data + bias.data leaves"
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)


# ---------------------------------------------------------------------------
# Output side: ``LowRankFactors`` returned from a jit
# ---------------------------------------------------------------------------


class TestOutputSide:
    """Returning a ``LowRankFactors`` from a jit is fine for all three flavours."""

    def test_jax_jit_returning_lowrank(self) -> None:
        """``jax.jit`` builds and returns a ``LowRankFactors`` cleanly."""

        @jax.jit
        def build(w: jax.Array, b: jax.Array) -> LowRankFactors:
            return _make_lowrank(w, b)

        w = jnp.ones((2, 2, 4), dtype=jnp.float32) * 0.3
        b = jnp.ones((2, 2), dtype=jnp.float32) * 0.7
        out = build(w, b)
        assert isinstance(out, LowRankFactors)
        assert out.weights.data.shape == (2, 2, 4)
        assert out.bias.data.shape == (2, 2)

    def test_nnx_jit_returning_lowrank(self) -> None:
        """``nnx.jit`` builds and returns a ``LowRankFactors`` cleanly."""

        @nnx.jit
        def build(w: jax.Array, b: jax.Array) -> LowRankFactors:
            return _make_lowrank(w, b)

        w = jnp.ones((2, 2, 4), dtype=jnp.float32) * 0.3
        b = jnp.ones((2, 2), dtype=jnp.float32) * 0.7
        out = build(w, b)
        assert isinstance(out, LowRankFactors)
        assert out.weights.data.shape == (2, 2, 4)
        assert out.bias.data.shape == (2, 2)

    def test_eqx_filter_jit_returning_lowrank(self) -> None:
        """``eqx.filter_jit`` builds and returns a ``LowRankFactors`` cleanly.

        Exercises the partition path that calls ``tree_unflatten`` with
        ``jax.Tracer`` leaves on the output side.
        """

        @eqx.filter_jit
        def build(w: jax.Array, b: jax.Array) -> LowRankFactors:
            return _make_lowrank(w, b)

        w = jnp.ones((2, 2, 4), dtype=jnp.float32) * 0.3
        b = jnp.ones((2, 2), dtype=jnp.float32) * 0.7
        out = build(w, b)
        assert isinstance(out, LowRankFactors)
        assert out.weights.data.shape == (2, 2, 4)


# ---------------------------------------------------------------------------
# Input side: ``LowRankFactors`` passed into a jit
# ---------------------------------------------------------------------------


class TestInputSide:
    """Passing a ``LowRankFactors`` into a jit works for all three flavours.

    ``nnx.jit``'s ``flax.nnx.extract.clear_non_graph_nodes`` and
    ``eqx.filter_jit``'s partition pass both invoke ``tree_unflatten``
    with non-array leaves (``jax.Tracer`` and Python ``bool`` filter
    masks respectively); raw ``jax.jit`` threads ``Tracer`` objects
    through directly without re-unflattening.
    """

    def test_jax_jit_identity_lowrank(self) -> None:
        """``jax.jit`` round-trips a ``LowRankFactors`` argument cleanly."""

        @jax.jit
        def identity(lr: LowRankFactors) -> LowRankFactors:
            return lr

        w = jnp.ones((2, 2, 4), dtype=jnp.float32) * 0.3
        b = jnp.ones((2, 2), dtype=jnp.float32) * 0.7
        lr = _make_lowrank(w, b)
        out = identity(lr)
        assert out.weights.data.shape == (2, 2, 4)

    def test_jax_jit_reading_lowrank_data(self) -> None:
        """``jax.jit`` lets the body read ``.data`` off a ``LowRankFactors`` arg."""

        @jax.jit
        def sum_data(lr: LowRankFactors) -> jax.Array:
            return jnp.asarray(lr.weights.data).sum() + jnp.asarray(lr.bias.data).sum()

        w = jnp.ones((2, 2, 4), dtype=jnp.float32) * 0.3
        b = jnp.ones((2, 2), dtype=jnp.float32) * 0.7
        lr = _make_lowrank(w, b)
        out = sum_data(lr)
        # 2*2*4*0.3 + 2*2*0.7 = 4.8 + 2.8 = 7.6
        assert float(out) == pytest.approx(7.6, abs=1e-5)

    def test_nnx_jit_identity_lowrank(self) -> None:
        """``nnx.jit`` round-trips a ``LowRankFactors`` argument cleanly.

        Exercises ``clear_non_graph_nodes``'s ``tree_unflatten`` call
        with ``jax.Tracer`` leaves.
        """

        @nnx.jit
        def identity(lr: LowRankFactors) -> LowRankFactors:
            return lr

        w = jnp.ones((2, 2, 4), dtype=jnp.float32) * 0.3
        b = jnp.ones((2, 2), dtype=jnp.float32) * 0.7
        lr = _make_lowrank(w, b)
        out = identity(lr)
        assert out.weights.data.shape == (2, 2, 4)

    def test_nnx_jit_reading_lowrank_data(self) -> None:
        """``nnx.jit`` body reads ``.data`` off a ``LowRankFactors`` arg cleanly."""

        @nnx.jit
        def sum_data(lr: LowRankFactors) -> jax.Array:
            return jnp.asarray(lr.weights.data).sum() + jnp.asarray(lr.bias.data).sum()

        w = jnp.ones((2, 2, 4), dtype=jnp.float32) * 0.3
        b = jnp.ones((2, 2), dtype=jnp.float32) * 0.7
        lr = _make_lowrank(w, b)
        out = sum_data(lr)
        assert float(out) == pytest.approx(7.6, abs=1e-5)

    def test_eqx_filter_jit_identity_lowrank(self) -> None:
        """``eqx.filter_jit`` round-trips a ``LowRankFactors`` argument cleanly.

        Exercises the partition pass that invokes ``tree_unflatten``
        with Python ``bool`` filter-mask leaves on the input side.
        """

        @eqx.filter_jit
        def identity(lr: LowRankFactors) -> LowRankFactors:
            return lr

        w = jnp.ones((2, 2, 4), dtype=jnp.float32) * 0.3
        b = jnp.ones((2, 2), dtype=jnp.float32) * 0.7
        lr = _make_lowrank(w, b)
        out = identity(lr)
        assert out.weights.data.shape == (2, 2, 4)

    def test_eqx_filter_jit_reading_lowrank_data(self) -> None:
        """``eqx.filter_jit`` body reads ``.data`` off a ``LowRankFactors`` arg cleanly."""

        @eqx.filter_jit
        def sum_data(lr: LowRankFactors) -> jax.Array:
            return jnp.asarray(lr.weights.data).sum() + jnp.asarray(lr.bias.data).sum()

        w = jnp.ones((2, 2, 4), dtype=jnp.float32) * 0.3
        b = jnp.ones((2, 2), dtype=jnp.float32) * 0.7
        lr = _make_lowrank(w, b)
        out = sum_data(lr)
        assert float(out) == pytest.approx(7.6, abs=1e-5)


# ---------------------------------------------------------------------------
# Raw-array boundary: still works
# ---------------------------------------------------------------------------


class TestRawArrayBoundary:
    """Passing raw arrays across the jit boundary works.

    Inputs are ``jax.Array``; the ``LowRankFactors`` is constructed
    *inside* the jit. Kept as a regression check for the
    construct-in-trace path.
    """

    def test_raw_array_inputs_lowrank_output(self) -> None:
        """Construct ``LowRankFactors`` inside the jit; inputs are arrays."""

        @jax.jit
        def embed_and_sum(w: jax.Array, b: jax.Array) -> jax.Array:
            lr = _make_lowrank(w, b)
            return jnp.asarray(lr.weights.data).sum() + jnp.asarray(lr.bias.data).sum()

        w = jnp.ones((2, 2, 4), dtype=jnp.float32) * 0.3
        b = jnp.ones((2, 2), dtype=jnp.float32) * 0.7
        out = embed_and_sum(w, b)
        # 2*2*4*0.3 + 2*2*0.7 = 4.8 + 2.8 = 7.6
        assert float(out) == pytest.approx(7.6, abs=1e-5)
