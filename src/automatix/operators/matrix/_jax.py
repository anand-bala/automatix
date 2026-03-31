from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx
from algebraic import AlgebraicArray, Semiring
from algebraic.types import Backend
from jaxtyping import Shaped

from automatix.spec import Guard

from ._base import MatrixOperator


class JaxMatrixOperator(MatrixOperator, eqx.Module):
    """JAX-backend weighted NFA operator.

    An :class:`equinox.Module` and therefore a JAX PyTree.

    Learnable fields (dynamic PyTree leaves)
    ----------------------------------------
    * ``initial_weights`` — e.g. soft initial-state distribution
    * ``final_weights`` — e.g. soft final-state distribution
    * ``weight_function`` — if it is itself an :class:`equinox.Module`,
        its parameters are reachable by :func:`equinox.filter_grad` /
        :func:`equinox.filter_jit`

    Static fields (not traced)
    --------------------------
    * ``semiring`` — algebraic structure, constant after construction
    * ``_transition_graph`` — NFA edge list, constant after construction

    .. note::
        ``_transition_graph`` is marked ``static``, so JAX retraces when
        the automaton structure changes.  In practice this never happens at
        runtime.  For very large automata the static hash comparison at JIT
        cache lookup may be slow; this is a known limitation.

    .. note::
        Because ``JaxMatrixOperator`` is an :class:`equinox.Module`, use
        :func:`equinox.filter_jit` (not :func:`jax.jit`) when compiling
        :py:meth:`~MatrixOperator.cost_transitions` as a method::

            @eqx.filter_jit
            def run(op, x):
                return op.cost_transitions(x)

        ``jax.jit(op.cost_transitions)`` raises ``TypeError`` because
        ``jax.jit`` tries to hash the bound ``self``, which fails for
        modules that hold JAX arrays.
    """

    initial_weights: Shaped[AlgebraicArray, " q"]
    final_weights: Shaped[AlgebraicArray, " q"]
    weight_function: Callable[..., Any]
    semiring: Semiring = eqx.field(static=True)
    _transition_graph: tuple[tuple[int, int, Guard[Any]], ...] = eqx.field(static=True)
    backend: ClassVar[Backend] = Backend.JAX

    @classmethod
    def _make(
        cls,
        initial_weights: AlgebraicArray,
        final_weights: AlgebraicArray,
        weight_function: Callable[..., Any],
        semiring: Semiring,
        transition_graph: tuple[tuple[int, int, Guard[Any]], ...],
    ) -> MatrixOperator:
        return JaxMatrixOperator(  # type: ignore[return-value]
            initial_weights=initial_weights,
            final_weights=final_weights,
            weight_function=weight_function,
            semiring=semiring,
            _transition_graph=transition_graph,
        )
