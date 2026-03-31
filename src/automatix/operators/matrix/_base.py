from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import algebraic
from algebraic import AlgebraicArray, Semiring
from algebraic.types import Backend
from jaxtyping import Shaped
from morphata.examples.nfa import NFA

from automatix._backend import resolve_backend
from automatix.spec import Guard, WeightFunction


@dataclass
class MatrixOperator:
    """Backend-agnostic weighted NFA operator.

    This class provides the shared :py:meth:`cost_transitions` method and the
    :py:meth:`make` factory. Do not instantiate directly — use
    :py:meth:`make` instead.

    Fields
    ------

    initial_weights : AlgebraicArray, shape ``(q,)``
        Semiring weight for each initial state.
    final_weights : AlgebraicArray, shape ``(q,)``
        Semiring weight for each final state.
    weight_function : Callable
        Maps ``(input_symbol, guard) -> semiring_value``.  May be an
        :class:`equinox.Module` or :class:`torch.nn.Module` for learnable
        operators.
    semiring : Semiring
        The semiring used to construct and fill the transition matrix.
    _transition_graph : tuple
        Frozen tuple of ``(src, dst, guard)`` triples from the NFA.
    backend : ClassVar[Backend]
        Which backend this subclass targets.
    """

    initial_weights: Shaped[AlgebraicArray, " q"]
    final_weights: Shaped[AlgebraicArray, " q"]
    weight_function: WeightFunction
    semiring: Semiring
    _transition_graph: tuple[tuple[int, int, Guard[Any]], ...]

    backend: ClassVar[Backend] = Backend.NUMPY

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
        n_q: int = self.initial_weights.shape[0]  # type: ignore[attr-defined]
        matrix = algebraic.zeros(
            (n_q, n_q),
            semiring=self.semiring,  # type: ignore[attr-defined]
            backend=str(self.backend),
        )
        for src, dst, guard in self._transition_graph:  # type: ignore[attr-defined]
            weight = self.weight_function(x, guard)  # type: ignore[attr-defined]
            matrix = matrix.at[src, dst].set(weight)
        return matrix

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
            The appropriate backend-specific subclass instance.

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

        if resolved == Backend.JAX:
            from ._jax import JaxMatrixOperator

            return JaxMatrixOperator._make(initial_weights, final_weights, weight_function, semiring, transition_graph)
        if resolved == Backend.TORCH:
            from ._torch import TorchMatrixOperator

            return TorchMatrixOperator._make(initial_weights, final_weights, weight_function, semiring, transition_graph)
        return MatrixOperator(
            initial_weights=initial_weights,
            final_weights=final_weights,
            weight_function=weight_function,
            semiring=semiring,
            _transition_graph=transition_graph,
        )
