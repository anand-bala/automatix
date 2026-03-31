"""Matrix-based weighted automaton operator.

Provides :class:`MatrixOperator` for constructing weighted finite-word
automaton operators from an NFA and a weight function.

The operator implements :class:`~algebraic.types.AlgebraicPyTree`.  Use
``algebraic.utils.jax.jaxify()`` or ``algebraic.utils.torch.torchify()``
for backend-specific integration.

Usage
-----
Construct via :py:meth:`MatrixOperator.make`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import algebraic
from algebraic import AlgebraicArray, Semiring
from algebraic.types import AnyPyTree, Backend
from jaxtyping import Shaped
from morphata.examples.nfa import NFA
from typing_extensions import Self

from automatix._backend import _StaticAux, resolve_backend
from automatix.spec import Guard, WeightFunction


@dataclass
class MatrixOperator:
    """Weighted NFA operator implementing :class:`~algebraic.types.AlgebraicPyTree`.

    Construct via :py:meth:`make`; do not instantiate directly.

    Fields
    ------
    initial_weights : AlgebraicArray, shape ``(q,)``
        Semiring weight for each initial state.
    final_weights : AlgebraicArray, shape ``(q,)``
        Semiring weight for each final state.
    weight_function : Callable
        Maps ``(input_symbol, guard) -> semiring_value``.
    semiring : Semiring
        The semiring used to construct and fill the transition matrix.
    _transition_graph : tuple
        Frozen tuple of ``(src, dst, guard)`` triples from the NFA.
    """

    initial_weights: Shaped[AlgebraicArray, " q"]
    final_weights: Shaped[AlgebraicArray, " q"]
    weight_function: WeightFunction
    semiring: Semiring
    _transition_graph: tuple[tuple[int, int, Guard[Any]], ...]

    def cost_transitions(self, x: object) -> AlgebraicArray:
        """Compute the ``q × q`` transition matrix for input *x*.

        Parameters
        ----------
        x :
            Input symbol passed through to the weight function.

        Returns
        -------
        AlgebraicArray, shape ``(q, q)``
            ``matrix[src, dst]`` is the semiring weight assigned by the weight
            function to the ``(src, dst)`` transition under input *x*.
        """
        n_q: int = self.initial_weights.shape[0]
        backend = str(Backend.from_array(self.initial_weights.data))
        matrix = algebraic.zeros(
            (n_q, n_q),
            semiring=self.semiring,
            backend=backend,
        )
        for src, dst, guard in self._transition_graph:
            weight = self.weight_function(x, guard)
            matrix = matrix.at[src, dst].set(weight)
        return matrix

    # ------------------------------------------------------------------
    # AlgebraicPyTree
    # ------------------------------------------------------------------

    def tree_flatten(self) -> tuple[list[AlgebraicArray], tuple[Any, ...]]:
        return [self.initial_weights, self.final_weights], (
            _StaticAux(self.weight_function),
            self.semiring,
            self._transition_graph,
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[Any, ...],
        children: Sequence[AnyPyTree],
    ) -> Self:
        wf_wrapped, semiring, transition_graph = aux_data
        initial_weights, final_weights = children
        assert isinstance(initial_weights, AlgebraicArray)
        assert isinstance(final_weights, AlgebraicArray)
        return cls(
            initial_weights=initial_weights,
            final_weights=final_weights,
            weight_function=wf_wrapped.value,
            semiring=semiring,
            _transition_graph=transition_graph,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def make(
        aut: NFA[Any],
        semiring: Semiring,
        *,
        weight_function: WeightFunction | Callable[..., Any],
        backend: str | Backend | None = None,
        initial_weights: AlgebraicArray | None = None,
        final_weights: AlgebraicArray | None = None,
    ) -> MatrixOperator:
        """Construct a :class:`MatrixOperator` from an NFA and weight function.

        Parameters
        ----------
        aut :
            The NFA defining guards and transitions.
        semiring :
            The semiring for output values (e.g. Boolean, Tropical, MaxMin).
        weight_function :
            Maps ``(input_symbol, guard) -> semiring_value``.
        backend :
            Which backend to use: ``'numpy'``, ``'jax'``, or ``'torch'``.
            If *None*, inferred from *initial_weights* or *final_weights*.
        initial_weights :
            Pre-built initial-state weight vector.  Defaults to ``semiring.one``
            at initial locations and ``semiring.zero`` elsewhere.
        final_weights :
            Pre-built final-state weight vector.  Defaults to ``semiring.one``
            at final locations and ``semiring.zero`` elsewhere.

        Returns
        -------
        MatrixOperator
            The constructed operator.  Use ``jaxify()`` or ``torchify()``
            from ``algebraic.utils`` for backend-specific integration.

        Raises
        ------
        ValueError
            If *backend* is *None* and cannot be inferred from the provided
            weight arrays, or if the automaton has no locations.
        """
        resolved = resolve_backend(backend, initial_weights, final_weights)
        n_q = aut.num_locations
        backend_str = str(resolved)

        if initial_weights is None:
            initial_weights = algebraic.zeros((n_q,), semiring=semiring, backend=backend_str)
            for q in aut.initial_state:
                initial_weights = initial_weights.at[q].set(semiring.one)

        if final_weights is None:
            final_weights = algebraic.zeros((n_q,), semiring=semiring, backend=backend_str)
            for q in aut.final_locations:
                final_weights = final_weights.at[q].set(semiring.one)

        transition_graph: tuple[tuple[int, int, Guard[Any]], ...] = tuple(
            (src, dst, guard) for src, dst, guard in aut.transitions
        )

        return MatrixOperator(
            initial_weights=initial_weights,
            final_weights=final_weights,
            weight_function=weight_function,
            semiring=semiring,
            _transition_graph=transition_graph,
        )
