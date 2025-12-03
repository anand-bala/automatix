"""Weighted automaton operators using semirings.

Provides MatrixOperator for constructing weighted finite-word automaton operators
from NFA and weight functions.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from algebraic import Semiring
from algebraic.tensor_algebra.jax import JaxBiModule
from jaxtyping import Array, Num

from automatix.automata.nfa import NFA
from automatix.spec import WeightFunction


class MatrixOperator[S: Semiring, In](eqx.Module):
    """JAX module representing a weighted finite-word automaton operator.

    This operator computes weighted transitions based on input symbols and guard
    evaluations using a semiring. It encodes:
    - Initial state weights
    - Final state weights
    - A function computing transition matrices for each input
    """

    initial_weights: Num[Array, " q"]
    final_weights: Num[Array, " q"]
    cost_transitions: Callable[[Num[Array, "..."]], Num[Array, "q q"]]

    @classmethod
    def make(
        cls,
        aut: NFA[In],
        algebra: JaxBiModule[S],
        *,
        weight_function: WeightFunction[Num[Array, "..."], In],
        initial_weights: None | Num[Array, " {len(aut)}"] = None,
        final_weights: None | Num[Array, " {len(aut)}"] = None,
    ) -> MatrixOperator:
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
        algebra : Semiring
            The algebra for output values (e.g., Boolean, Tropical, MaxMin).
        weight_function : WeightFunction
            A function mapping (input_symbol, guard) to algebra values.
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
                algebra.zeros(aut.num_locations).at[jnp.array(list(aut.initial_state))].set(algebra.ones(1).item())
            )
        if final_weights is None:
            final_weights = (
                algebra.zeros(aut.num_locations).at[jnp.array(list(aut.final_locations))].set(algebra.ones(1).item())
            )

        assert initial_weights.shape == (n_q,)
        assert final_weights.shape == (n_q,)

        transitions = {(src, dst): functools.partial(weight_function, guard=guard) for src, dst, guard in aut.transitions}

        # # Build list of transitions for use in cost_transitions
        # idx: Sequence[tuple[int, int]]
        # guards: Sequence[Guard[In]]

        # idx, guards = tuple(zip(*(((src, dst), guard) for src, dst, guard in aut.transitions)))

        def cost_transitions(x: Num[Array, "..."]) -> Num[Array, "q q"]:
            """Compute transition matrix for input x using weight_function.

            Parameters
            ----------
            x : Array
                Input symbol (vector in state space).

            Returns
            -------
            Array
                q x q weighted transition matrix where element [i,j] is
                weight_function(x, guard_{i,j}).
            """

            matrix = algebra.zeros((n_q, n_q))
            for (src, dst), guard in transitions.items():
                # Apply weight function: lambda(x, guard)
                weight = guard(x)  # type: ignore[no-untyped-call]
                matrix = matrix.at[src, dst].set(weight)

            return matrix

        return MatrixOperator(initial_weights=initial_weights, final_weights=final_weights, cost_transitions=cost_transitions)
