"""Base automata interfaces and definitions"""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Collection, Hashable, Iterable
from typing import Generic, TypeVar

from typing_extensions import final, override

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


@final
class NFW(WordAutomaton[Alph, Q]):
    """Non-deterministic Finite Word automaton

    A non-deterministic automaton that recognizes finite words
    """

    def __init__(self, alphabet: Collection[Alph]) -> None:
        super().__init__()

        self._alph = alphabet
        self._adj: dict[Q, dict[Alph, set[Q]]] = defaultdict(lambda: defaultdict(set))
        self._initial_states: set[Q] = set()
        self._final_states: set[Q] = set()
        self._is_deterministic: bool | None = None

    def add_location(
        self, location: Q, initial: bool = False, final: bool = False
    ) -> None:
        """Add a location to the automaton."""
        if location in self._adj:
            raise ValueError(f"Location {location} already exists in automaton")
        if initial:
            self._initial_states.add(location)
        if final:
            self._initial_states.add(location)
        assert len(self._adj[location]) == 0

    def add_transition(self, src: Q, guard: Alph, dst: Iterable[Q]) -> None:
        dst = set(dst)
        self._is_deterministic = len(dst) == 1
        self._adj[src][guard].update(dst)

    @override
    def contains(self, location: Q) -> bool:
        return location in self._adj

    @override
    def num_locations(self) -> int:
        return len(self._adj.keys())

    @override
    def transition(self, location: Q, symbol: Alph) -> set[Q]:
        return self._adj[location][symbol]

    @property
    @override
    def is_deterministic(self) -> bool:
        return self._is_deterministic if self._is_deterministic is not None else True

    @property
    @override
    def is_alternating(self) -> bool:
        return False

    @override
    def is_initial(self, state: Q) -> bool:
        return state in self._initial_states

    @override
    def is_accepting(self, state: Q) -> bool:
        return state in self._final_states


@final
class AFW(WordAutomaton[Alph, Q]):
    """Alternating Finite Word automaton"""

    def __init__(self, alphabet: Collection[Alph]) -> None:
        super().__init__()

        self._alph = alphabet
        self._adj: dict[Q, dict[Alph, set[Q | tuple[Q, ...]]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._initial_states: set[Q] = set()
        self._final_states: set[Q] = set()
        self._is_deterministic: bool | None = None
        self._is_alternating: bool | None = None

    def add_location(
        self, location: Q, initial: bool = False, final: bool = False
    ) -> None:
        """Add a location to the automaton."""
        if location in self._adj:
            raise ValueError(f"Location {location} already exists in automaton")
        if initial:
            self._initial_states.add(location)
        if final:
            self._initial_states.add(location)
        assert len(self._adj[location]) == 0

    def add_transition(
        self, src: Q, guard: Alph, dst: Iterable[Q | tuple[Q, ...]]
    ) -> None:
        dst = set(dst)
        self._is_deterministic = len(dst) == 1
        self._is_alternating = any(
            len(out) > 1 for out in dst if isinstance(out, tuple)
        )
        self._adj[src][guard].update(dst)

    @override
    def contains(self, location: Q) -> bool:
        return location in self._adj

    @override
    def num_locations(self) -> int:
        return len(self._adj.keys())

    @override
    def transition(self, location: Q, symbol: Alph) -> set[Q | tuple[Q, ...]]:
        return self._adj[location][symbol]

    @property
    @override
    def is_deterministic(self) -> bool:
        return self._is_deterministic if self._is_deterministic is not None else True

    @property
    @override
    def is_alternating(self) -> bool:
        return self._is_alternating if self._is_alternating is not None else False

    @override
    def is_initial(self, state: Q) -> bool:
        return state in self._initial_states

    @override
    def is_accepting(self, state: Q) -> bool:
        return state in self._final_states
