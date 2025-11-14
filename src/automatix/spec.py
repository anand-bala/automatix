"""Pure abstract interfaces for various automata and related concepts"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator
from typing import Protocol, runtime_checkable

import logic_asts.base as exprs
from jaxtyping import Array, ScalarLike

import automatix.acc as acc

type AcceptanceCondition[Q: Hashable] = (
    acc.Finite[Q]
    | acc.Buchi[Q]
    | acc.CoBuchi[Q]
    | acc.GeneralizedBuchi[Q]
    | acc.GeneralizedCoBuchi[Q]
    | acc.Rabin[Q]
    | acc.Streett[Q]
    | acc.Muller[Q]
)

type Guard[AtomicPredicate] = exprs.BaseExpr[AtomicPredicate]


class AbstractAutomaton[In, Out, StateRep, Q: Hashable](ABC):
    r"""
    The `Automaton` class defines a general interface for all automata-like transition
    systems, and can be used by other components in `automatix` to define their own
    semantics.
    """

    @property
    @abstractmethod
    def initial_state(self) -> StateRep: ...

    @property
    @abstractmethod
    def acceptance_condition(self) -> AcceptanceCondition[Q]: ...

    @abstractmethod
    def __call__(self, input_symbol: In, state: StateRep) -> tuple[Out, StateRep]:
        """
        The transition function that reads the input symbol `input_symbol` at an
        automaton state `state` and outputs an output symbol in `Out` and a successor
        `StateRep`.
        """


class SizedAutomaton[In, Out, StateRep, Q: Hashable](AbstractAutomaton[In, Out, StateRep, Q]):
    """A sized automaton has a finite number, a priori known, number of states in it."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __iter__(self) -> Iterator[Q]: ...


@runtime_checkable
class WeightFunction[In](Protocol):
    """Weight function mapping (input, guard) to semiring value.

    A weight function implements lambda(x, Delta) from weighted automata theory:
    - Takes an input symbol x and guard expression Delta
    - Returns a weight in the target semiring
    - Used to compute transition weights in automaton operators

    Examples
    --------
    Simple constant weight function:

    >>> def constant(x, guard):
    ...     return 1.0

    Distance-based weight function:

    >>> def distance_weight(x, guard):
    ...     # Distance from x to satisfying guard
    ...     return compute_distance(x, guard)

    Predicate-based weight function:

    >>> def predicate_weight(x, guard):
    ...     # Evaluate guard with input x
    ...     return evaluate_guard(x, guard)
    """

    def __call__(self, x: In, guard: Guard) -> Array | ScalarLike: ...
