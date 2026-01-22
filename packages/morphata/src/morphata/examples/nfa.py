"""Nondeterministic finite automaton (NFA) implementation.

This is an example implementation of the Automaton interface for finite-word
recognition. It provides a graph-based builder API and converts to the
structural Automaton representation.
"""

from __future__ import annotations

import typing
from collections.abc import Hashable, Iterable, Iterator
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

import networkx as nx
from logic_asts.base import BaseExpr as BoolExpr
from logic_asts.base import simple_eval
from typing_extensions import overload, override

from morphata import AcceptanceCondition
from morphata.acceptance import Finite
from morphata.spec import Automaton, Domain, NonDeterministicTransitions

type Guard[AtomicPredicate] = BoolExpr[AtomicPredicate]
"""Guard expression over atomic predicates (NFA implementation detail)."""

type NFAState = frozenset[int]

AP = typing.TypeVar("AP", bound=Hashable)


@dataclass
class NFADomain(Domain[int, AbstractSet[AP]]):
    """Domain for NFA with integer states and set-based symbols."""

    _graph: nx.DiGraph[int]

    @property
    @override
    def states(self) -> Iterable[int] | None:
        yield from self._graph.nodes

    @property
    @override
    def symbols(self) -> Iterable[AbstractSet[AP]] | None:
        # Symbolic - cannot enumerate all possible input sets
        return None


class NFATransition[AP](NonDeterministicTransitions[int, AbstractSet[AP]]):
    """Transition relation for NFA that evaluates guards."""

    def __init__(self, graph: nx.DiGraph[int]) -> None:
        self._graph = graph

    @override
    def __call__(self, state: int, symbol: AbstractSet[AP]) -> Iterable[int]:
        """Evaluate guards and return successor states."""
        symbol_set = set(symbol)
        successors: list[int] = []
        for _, succ, guard_data in self._graph.edges(state, data="guard"):
            guard: Guard[AP] = typing.cast(Guard[AP], guard_data)
            if simple_eval(guard, symbol_set):
                successors.append(succ)
        return successors


class NFA(typing.Generic[AP]):
    """Nondeterministic Finite Automaton for finite-word recognition.

    An NFA is defined by a set of locations, transitions labeled with guards,
    and initial/final locations. This class provides a builder API for
    constructing NFAs and can convert to the structural Automaton representation.

    Here, the underlying NFA is defined by a graph with integer nodes.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph[int] = nx.DiGraph()
        self._initial_locations: set[int] = set()
        self._final_locations: set[int] = set()

    def __len__(self) -> int:
        return self.num_locations

    def __iter__(self) -> Iterator[int]:
        return iter(self._graph.nodes)

    def __call__(self, input_symbol: AbstractSet[AP], state: NFAState) -> tuple[bool, NFAState]:
        """Transition function for runtime stepping.

        Parameters
        ----------
        input_symbol : AbstractSet[AP]
            Input symbol (set of atomic predicates)
        state : NFAState
            Current state (set of locations)

        Returns
        -------
        tuple[bool, NFAState]
            (accepting, next_state) where accepting indicates if any
            successor is in a final location
        """
        input_symbol_set = set(input_symbol)
        successors: set[int] = set()
        for src in state:
            successors |= {succ for succ, guard in self.guards(src).items() if simple_eval(guard, input_symbol_set)}
        accepting = not successors.isdisjoint(self._final_locations)
        return accepting, frozenset(successors)

    @property
    def initial_state(self) -> NFAState:
        """Get the set of initial locations."""
        return frozenset(self._initial_locations)

    @property
    def final_locations(self) -> NFAState:
        """Get the set of final/accepting locations."""
        return frozenset(self._final_locations)

    @property
    def acceptance_condition(self) -> Finite[int]:
        """Acceptance condition based on final locations."""
        return Finite[int](frozenset(self._final_locations))

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
            self._initial_locations.add(location)
        if final:
            self._final_locations.add(location)
        self._graph.add_node(location, initial=initial, final=final)

    def add_transition(self, src: int, dst: int, guard: Guard[AP]) -> None:
        """Add a transition between two locations.

        Parameters
        ----------
        src : int
            Source location
        dst : int
            Destination location
        guard : Guard[AP]
            Guard expression
        """
        if (src, dst) in self._graph.edges:
            raise ValueError(f"Transition from {src} to {dst} already exists. Did you want to update the guard?")
        if guard.horizon() != 0:
            raise ValueError("Given guard has temporal operators")
        self._graph.add_edge(src, dst, guard=guard)

    @property
    def num_locations(self) -> int:
        """Get the number of locations in this automaton."""
        return len(self._graph)

    @overload
    def guards(self, src: int, dst: int) -> Guard[AP]: ...

    @overload
    def guards(self, src: int, dst: None = None) -> dict[int, Guard[AP]]: ...

    def guards(self, src: int, dst: int | None = None) -> Guard[AP] | dict[int, Guard[AP]]:
        """Get a transition guard or the set of transition guards for each successor state.

        Parameters
        ----------
        src : int
            Source location
        dst : int or None
            Destination location (if None, returns all outgoing guards)

        Returns
        -------
        Guard[AP] or dict[int, Guard[AP]]
            Single guard if dst is specified, else dict of {destination: guard}
        """
        if dst is None:
            return {succ: guard for _, succ, guard in self._graph.edges(src, "guard")}
        return typing.cast(Guard[AP], self._graph.edges[src, dst]["guard"])

    @property
    def transitions(self) -> Iterable[tuple[int, int, Guard[AP]]]:
        """Get an iterable of (src, dst, guard) tuples for all transitions."""
        return self._graph.edges.data("guard")

    def to_automaton(self) -> Automaton[int, AbstractSet[AP]]:
        """Convert to structural Automaton representation.

        Returns
        -------
        Automaton[int, AbstractSet[AP]]
            Structural automaton with integer states and set-based symbols
        """
        domain: Domain[int, AbstractSet[AP]] = NFADomain[AP](self._graph)
        delta: NonDeterministicTransitions[int, AbstractSet[AP]] = NFATransition[AP](self._graph)
        initial: AbstractSet[int] = self.initial_state
        acceptance: AcceptanceCondition[int] = self.acceptance_condition

        return Automaton(
            domain=domain,  # ty:ignore[invalid-argument-type]
            initial=initial,
            delta=delta,  # ty:ignore[invalid-argument-type]
            acceptance=acceptance,  # ty:ignore[invalid-argument-type]
        )
