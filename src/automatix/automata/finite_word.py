"""Finite-word automata and operators.

This module provides nondeterministic finite automata (NFA) for finite-word recognition,
along with weighted automaton operators for semiring-based evaluation.
"""

from __future__ import annotations

import typing
from collections.abc import Iterable
from typing import Callable, Optional, Type

import equinox as eqx
import jax.numpy as jnp
import logic_asts
import networkx as nx
from jaxtyping import Array, Num
from lark.exceptions import LarkError
from logic_asts.base import Expr
from typing_extensions import overload

from automatix.algebra.spec import AbstractSemiring
from automatix.weights import WeightFunction


class NFA:
    """Nondeterministic Finite Automaton for finite-word recognition.

    An NFA is defined by a set of locations, transitions labeled with guards,
    and initial/final locations. It can be used standalone or wrapped in an
    AutomatonOperator for weighted evaluation over semirings.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph[int] = nx.DiGraph()
        self._initial_location: set[int] = set()
        self._final_locations: set[int] = set()

    def add_location(self, location: int, initial: bool = False, final: bool = False) -> None:
        """Add a location to the automaton.

        Parameters
        ----------
        location : int
            The location index (must be unique)
        initial : bool, optional
            Whether this is an initial location
        final : bool, optional
            Whether this is a final/accepting location
        """
        if location in self._graph.nodes:
            raise ValueError(f"Location {location} already exists in automaton")
        if initial:
            self._initial_location.add(location)
        if final:
            self._final_locations.add(location)
        self._graph.add_node(location, initial=initial, final=final)

    def add_transition(self, src: int, dst: int, guard: str | Expr) -> None:
        """Add a transition between two locations.

        Parameters
        ----------
        src : int
            Source location
        dst : int
            Destination location
        guard : str or Expr
            Guard expression (string or parsed logic_asts.Expr)
        """
        if (src, dst) in self._graph.edges:
            raise ValueError(f"Transition from {src} to {dst} already exists. Did you want to update the guard?")
        if isinstance(guard, str):
            try:
                guard = logic_asts.parse_expr(guard)
            except LarkError as e:
                raise ValueError("Unable to parse guard as a boolean expression") from e
        if guard.horizon() != 0:
            raise ValueError("Given guard has temporal operators")
        self._graph.add_edge(src, dst, guard=guard)

    @property
    def num_locations(self) -> int:
        """Get the number of locations in this automaton."""
        return len(self._graph)

    def __len__(self) -> int:
        return self.num_locations

    @property
    def initial_locations(self) -> set[int]:
        """Get the set of initial locations."""
        return self._initial_location

    @property
    def final_locations(self) -> set[int]:
        """Get the set of final/accepting locations."""
        return self._final_locations

    @overload
    def guards(self, src: int, dst: int) -> Expr: ...

    @overload
    def guards(self, src: int, dst: None = None) -> dict[int, Expr]: ...

    def guards(self, src: int, dst: int | None = None) -> Expr | dict[int, Expr]:
        """Get a transition guard or the set of transition guards for each successor state.

        Parameters
        ----------
        src : int
            Source location
        dst : int or None
            Destination location (if None, returns all outgoing guards)

        Returns
        -------
        Expr or dict[int, Expr]
            Single guard if dst is specified, else dict of {destination: guard}
        """
        if dst is None:
            return {succ: guard for _, succ, guard in self._graph.edges(src, "guard")}  # type: ignore[var-annotated]
        return typing.cast(Expr, self._graph.edges[src, dst]["guard"])

    @property
    def transitions(self) -> Iterable[tuple[int, int, Expr]]:
        """Get an iterable of (src, dst, guard) tuples for all transitions."""
        return self._graph.edges.data("guard")


class AutomatonOperator(eqx.Module):
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


def make_automaton_operator(
    aut: NFA,
    semiring: Type[AbstractSemiring],
    *,
    weight_function: WeightFunction,
    initial_weights: Optional[Num[Array, " {len(aut)}"]] = None,
    final_weights: Optional[Num[Array, " {len(aut)}"]] = None,
) -> AutomatonOperator:
    """Create an automaton operator from an NFA and weight function.

    The operator computes weighted paths through the automaton by:
    1. Starting with initial state weights
    2. For each input, computing weighted transitions via the weight function
    3. Accumulating weights through semiring operations
    4. Accepting at final states weighted by final_weights

    Parameters
    ----------
    aut : NFA
        The nondeterministic finite automaton defining guards and transitions.
    semiring : Type[AbstractSemiring]
        The semiring for output values (e.g., Boolean, Tropical, MaxMin).
    weight_function : WeightFunction
        A function mapping (input_symbol, guard) to semiring values.
        Implements lambda(x, Delta) from weighted automata theory.
    initial_weights : Optional[Array], optional
        Initial state weights. If None, set to semiring.one at initial locations.
    final_weights : Optional[Array], optional
        Final state weights. If None, set to semiring.one at final locations.

    Returns
    -------
    AutomatonOperator
        An operator that computes weighted transitions for inputs.
    """
    n_q = aut.num_locations

    if initial_weights is None:
        initial_weights = (
            semiring.zeros(aut.num_locations).at[jnp.array(list(aut.initial_locations))].set(semiring.ones(1).item())
        )
    if final_weights is None:
        final_weights = semiring.zeros(aut.num_locations).at[jnp.array(list(aut.final_locations))].set(semiring.ones(1).item())

    assert initial_weights.shape == (n_q,)
    assert final_weights.shape == (n_q,)

    # Build list of transitions for use in cost_transitions
    transitions_list = list(aut.transitions)

    def cost_transitions(x: Num[Array, "..."]) -> Num[Array, " {n_q} {n_q}"]:
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
        matrix = semiring.zeros((n_q, n_q))
        for src, dst, guard in transitions_list:
            # Apply weight function: lambda(x, guard)
            weight = weight_function(x, guard)
            matrix = matrix.at[src, dst].set(weight)

        return matrix

    return AutomatonOperator(initial_weights=initial_weights, final_weights=final_weights, cost_transitions=cost_transitions)
