"""Nondeterministic finite automaton (NFA) implementation."""
from __future__ import annotations

import typing
from collections.abc import Iterable, Iterator
from collections.abc import Set as AbstractSet

import networkx as nx
from logic_asts.base import simple_eval
from typing_extensions import overload, override

from automatix.acc import Finite
from automatix.spec import Guard, SizedAutomaton

type NFAState = frozenset[int]


class NFA[In](SizedAutomaton[AbstractSet[In], bool, NFAState, int]):
    """Nondeterministic Finite Automaton for finite-word recognition.

    An NFA is defined by a set of locations, transitions labeled with guards,
    and initial/final locations. It can be used standalone or wrapped in an
    AutomatonOperator for weighted evaluation over semirings.

    Here, the underlying NFA is defined by a graph with integer nodes.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph[int] = nx.DiGraph()
        self._initial_location: set[int] = set()
        self._final_locations: set[int] = set()

    @override
    def __len__(self) -> int:
        return self.num_locations

    @override
    def __iter__(self) -> Iterator[int]:
        return iter(self._graph.nodes)

    @override
    def __call__(self, input_symbol: AbstractSet[In], state: NFAState) -> tuple[bool, NFAState]:
        input_symbol = set(input_symbol)
        successors: set[int] = set()
        for src in state:
            successors |= {succ for succ, guard in self.guards(src).items() if simple_eval(guard, input_symbol)}
        good = not successors.isdisjoint(self._final_locations)
        return good, frozenset(successors)

    @property
    @override
    def initial_state(self) -> NFAState:
        """Get the set of initial locations."""
        return frozenset(self._initial_location)

    @property
    @override
    def acceptance_condition(self) -> Finite[int]:
        """Acceptance condition based on final locations."""
        return Finite(self.final_locations)

    @property
    def final_locations(self) -> NFAState:
        """Get the set of final/accepting locations."""
        return frozenset(self._final_locations)

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

    def add_transition(self, src: int, dst: int, guard: Guard[In]) -> None:
        """Add a transition between two locations.

        Parameters
        ----------
        src : int
            Source location
        dst : int
            Destination location
        guard : Expr
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
    def guards(self, src: int, dst: int) -> Guard[In]: ...

    @overload
    def guards(self, src: int, dst: None = None) -> dict[int, Guard[In]]: ...

    def guards(self, src: int, dst: int | None = None) -> Guard[In] | dict[int, Guard[In]]:
        """Get a transition guard or the set of transition guards for each successor state.

        Parameters
        ----------
        src : int
            Source location
        dst : int or None
            Destination location (if None, returns all outgoing guards)

        Returns
        -------
        Guard[In] or dict[int, Guard[In]]
            Single guard if dst is specified, else dict of {destination: guard}
        """
        if dst is None:
            return {succ: guard for _, succ, guard in self._graph.edges(src, "guard")}  # type: ignore[var-annotated]
        return typing.cast(Guard[In], self._graph.edges[src, dst]["guard"])

    @property
    def transitions(self) -> Iterable[tuple[int, int, Guard[In]]]:
        """Get an iterable of (src, dst, guard) tuples for all transitions."""
        return self._graph.edges.data("guard")
