"""Base automata interfaces and definitions"""

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable
from typing import Generic, TypeVar

Alph = TypeVar("Alph", bound=Hashable)
Q = TypeVar("Q")


class WordAutomaton(Generic[Alph, Q], ABC):
    r"""Generic word automaton

    An automaton is a a tuple \(\mathcal{A} = \left( \Sigma, Q, q_0, \Delta, F \right)\),
    where \(\Sigma\) is a nonempty alphabet, \(Q\) is a finite set of states with initial
    state \(q_0 \in Q\), \(F \subseteq Q\) is a set of accepting states, and \(\Delta\) is
    a transition relation function.

    The `Automaton` class defines a general interface for all automata-like transition
    systems, and can be used by other components in `automatix` to define their own
    semantics.
    """

    def __len__(self) -> int:
        return self.num_locations()

    @abstractmethod
    def num_locations(self) -> int:
        r"""Get the number of locations in the automaton, i.e., the size of the set \(Q\)"""

    @abstractmethod
    def transition(self, location: Q, symbol: Alph) -> Iterable[Iterable[Q] | Q]:
        """A transition function outputs the "sum of products" form of the successor
        locations.

        If the iterable contains another iterable, the states within the nested set are
        part of a universal transition, while the elements of the outer iterable are
        part of a extential non-deterministic transition.
        """

    @abstractmethod
    def contains(self, location: Q) -> bool:
        """Check if the given `location` is in the automaton"""

    def __contains__(self, location: Q) -> bool:
        return self.contains(location)

    @property
    @abstractmethod
    def is_deterministic(self) -> bool:
        """Check if the automaton is deterministic"""

    @property
    @abstractmethod
    def is_alternating(self) -> bool:
        """Check if the automaton has alternating transitions"""

    @abstractmethod
    def is_initial(self, state: Q) -> bool:
        """Check if the given automaton state is an initial state"""

    @abstractmethod
    def is_accepting(self, state: Q) -> bool:
        """Check if the given automaton state is an accepting state"""
