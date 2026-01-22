"""Matrix-based weighted automaton operator.

Provides MatrixOperator for constructing weighted finite-word automaton operators
from NFA and weight functions.
"""

from __future__ import annotations

import functools
import typing
from collections.abc import Callable
from typing import TypeVar

import algebraic.numpy as alge
import equinox as eqx
import jax.numpy as jnp
from algebraic import AlgebraicArray, Semiring
from jaxtyping import Array, Num, Shaped
from morphata.examples.nfa import NFA

from automatix.spec import WeightFunction

S = TypeVar("S", bound=Semiring)
In = TypeVar("In")


class MatrixOperator(eqx.Module, typing.Generic[S, In]):
    """JAX module representing a weighted finite-word automaton operator.

    This operator computes weighted transitions based on input symbols and guard
    evaluations using a semiring. It encodes:
    - Initial state weights
    - Final state weights
    - A function computing transition matrices for each input
    """

    initial_weights: Shaped[AlgebraicArray[S], " q"]
    final_weights: Shaped[AlgebraicArray[S], " q"]
    cost_transitions: Callable[[Num[Array, "..."]], Shaped[AlgebraicArray[S], "q q"]]

    @classmethod
    def make(
        cls,
        aut: NFA[In],
        semiring: S,
        *,
        weight_function: WeightFunction[Num[Array, "..."], In],
        initial_weights: None | Shaped[AlgebraicArray[S], " q"] = None,
        final_weights: None | Shaped[AlgebraicArray[S], " q"] = None,
    ) -> MatrixOperator[S, In]:
        """Create an automaton operator from an NFA and weight function.

        The operator computes weighted paths through the automaton by:
        1. Starting with initial state weights
        2. For each input, computing weighted transitions via the weight function
        3. Accumulating weights through algebraic operations
        4. Accepting at final states weighted by final_weights

        Parameters
        ----------

        aut : NFA
            The nondeterministic finite automaton defining guards and transitions.
        semiring : Semiring
            The semiring for output values (e.g., Boolean, Tropical, MaxMin).
        weight_function : WeightFunction
            A function mapping (input_symbol, guard) to semiring values.
            Implements lambda(x, Delta) from weighted automata theory.
        initial_weights : Optional[Array], optional
            Initial state weights. If None, set to 1 at initial locations.
        final_weights : Optional[Array], optional
            Final state weights. If None, set to 1 at final locations.

        Returns
        -------
        AutomatonOperator
            An operator that computes weighted transitions for inputs.

        Notes
        -----
        1. The number of states in the automaton must be known up front, otherwise, the matrix operator cannot be formed.
        2. The matrix operator only makes sense for finite acceptance conditions or, if handled correctly, Büchi/co-Büchi acceptance conditions.
        """
        n_q = aut.num_locations

        if initial_weights is None:
            initial_weights = (
                alge.zeros(aut.num_locations, semiring)
                .at[jnp.array(list(aut.initial_state))]
                .set(alge.ones(1, semiring).data.item())
            )
        if final_weights is None:
            final_weights = (
                alge.zeros(aut.num_locations, semiring)
                .at[jnp.array(list(aut.final_locations))]
                .set(alge.ones(1, semiring).data.item())
            )

        assert initial_weights.shape == (n_q,)
        assert final_weights.shape == (n_q,)

        transitions = {(src, dst): functools.partial(weight_function, guard=guard) for src, dst, guard in aut.transitions}

        def cost_transitions(x: Num[Array, "..."]) -> Shaped[AlgebraicArray[S], "q q"]:
            """Compute transition matrix for input x using weight_function.

            Parameters
            ----------
            x : Array
                Input symbol (vector in state space).

            Returns
            -------
            AlgebraicArray[S]
                q x q weighted transition matrix where element [i,j] is
                weight_function(x, guard_{i,j}).
            """

            matrix = alge.zeros((n_q, n_q), semiring)
            for (src, dst), guard in transitions.items():
                # Apply weight function: lambda(x, guard)
                weight = guard(x)
                matrix = matrix.at[src, dst].set(weight)

            return matrix

        return MatrixOperator(initial_weights=initial_weights, final_weights=final_weights, cost_transitions=cost_transitions)
