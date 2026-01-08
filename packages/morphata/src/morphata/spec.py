"""Base interfaces for automata structures.

This module provides pure structural interfaces for automata without
any weight function or semiring concepts. These interfaces are extended
by automatix for weighted automata semantics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

import logic_asts.base as exprs

type Guard[AtomicPredicate] = exprs.BaseExpr[AtomicPredicate]
"""Guard expression over atomic predicates.

Guards are boolean expressions used to label transitions in automata.
They are evaluated against input symbols to determine which transitions
are enabled.
"""


class AbstractAutomaton[In, Out, StateRep, Q: Hashable](ABC):
    """Base interface for automaton-like transition systems.

    This provides the minimal interface for any automaton-like system:
    an initial state and a transition function. Specific automaton types
    can add their own acceptance_condition property as needed.

    Type Parameters
    ---------------
    In : type
        Input alphabet type
    Out : type
        Output type (e.g., bool for acceptance)
    StateRep : type
        Runtime state representation (e.g., frozenset[int] for NFA)
    Q : Hashable
        Underlying location/state space type
    """

    @property
    @abstractmethod
    def initial_state(self) -> StateRep:
        """The initial state of the automaton."""

    @abstractmethod
    def __call__(self, input_symbol: In, state: StateRep) -> tuple[Out, StateRep]:
        """Transition function.

        Takes an input symbol and current state, returns output and next state.

        Parameters
        ----------
        input_symbol : In
            The input symbol to process
        state : StateRep
            The current state

        Returns
        -------
        tuple[Out, StateRep]
            Output value and successor state
        """


class SizedAutomaton[In, Out, StateRep, Q: Hashable](AbstractAutomaton[In, Out, StateRep, Q]):
    """Automaton with a finite, a priori known state space.

    Sized automata have a fixed number of states known at construction time.
    This allows for efficient matrix-based representations and operations.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Number of states in the automaton."""

    @abstractmethod
    def __iter__(self) -> Iterator[Q]:
        """Iterate over all states in the automaton."""


@dataclass(frozen=True)
class FiniteAcceptance[Q: Hashable]:
    """Finite-word acceptance condition with concrete accepting states.

    This is a minimal acceptance condition for graph-based automata.
    A run is accepting if it ends in a state that is in the accepting set.

    This differs from morphata.acceptance.Finite which is an expression-based
    representation for HOA parsing. This class is for runtime checking.

    Parameters
    ----------
    accepting : AbstractSet[Q]
        Set of accepting states
    """

    accepting: AbstractSet[Q]
    """Set of states that are accepting (final states)"""
