from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable, Iterable
from typing import Generic, TypeVar

from typing_extensions import override

Alph = TypeVar("Alph", bound=Hashable)


class WordAutomaton(Generic[Alph], ABC):
    r"""Base word automaton

    An automaton is a a tuple \(\mathcal{A} = \left( \Sigma, Q, q_0, \Delta, F \right)\),
    where \(\Sigma\) is a nonempty alphabet, \(Q\) is a finite set of states with initial
    state \(q_0 \in Q\), \(F \subseteq Q\) is a set of accepting states, and \(\Delta\) is
    a transition relation function.

    The `Automaton` class defines a general interface for all automata-like transition
    systems, and can be used by other components in `automatix` to define their own
    semantics.

    Moreover, the locations in the automaton are always `int`.
    """

    def __len__(self) -> int:
        return self.num_locations()

    @abstractmethod
    def num_locations(self) -> int:
        r"""Get the number of locations in the automaton, i.e., the size of the set \(Q\)"""

    @abstractmethod
    def transition(self, location: int, symbol: Alph) -> Iterable[Iterable[int] | int]:
        """A transition function outputs the "sum of products" form of the successor
        locations.

        If the iterable contains another iterable, the states within the nested set are
        part of a universal transition, while the elements of the outer iterable are
        part of a extential non-deterministic transition.
        """

    @abstractmethod
    def contains(self, location: int) -> bool:
        """Check if the given `location` is in the automaton"""

    def __contains__(self, location: int) -> bool:
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
    def is_initial(self, state: int) -> bool:
        """Check if the given automaton state is an initial state"""

    @abstractmethod
    def is_accepting(self, state: int) -> bool:
        """Check if the given automaton state is an accepting state"""


class AlternatingAutomaton(WordAutomaton[Alph]):
    """Alternating Finite Word automaton"""

    def __init__(self, alphabet: Collection[Alph]) -> None:
        super().__init__()
        self._alph = alphabet
        self._adj: dict[int, dict[Alph, set[int | tuple[int, ...]]]] = dict()
        self._initial_states: set[int] = set()
        self._final_states: set[int] = set()
        self._is_deterministic: bool | None = None
        self._is_alternating: bool | None = None

    def add_location(self, location: int, initial: bool = False, final: bool = False) -> None:
        """Add a location to the automaton."""
        if location in self._adj:
            raise ValueError(f"Location {location} already exists in automaton")
        self._adj[location] = dict()
        if initial:
            self._initial_states.add(location)
        if final:
            self._initial_states.add(location)
        assert len(self._adj[location]) == 0

    def add_transition(self, src: int, guard: Alph, dst: Iterable[int | tuple[int, ...]]) -> None:
        dst = set(dst)
        self._is_deterministic = len(dst) == 1
        self._is_alternating = any(len(out) > 1 for out in dst if isinstance(out, tuple))
        self._adj.setdefault(src, dict()).setdefault(guard, set()).update(dst)

    @override
    def contains(self, location: int) -> bool:
        return location in self._adj

    @override
    def num_locations(self) -> int:
        return len(self._adj.keys())

    @override
    def transition(self, location: int, symbol: Alph) -> set[int | tuple[int, ...]]:
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
    def is_initial(self, state: int) -> bool:
        return state in self._initial_states

    @override
    def is_accepting(self, state: int) -> bool:
        return state in self._final_states


class NondeterministicAutomaton(AlternatingAutomaton[Alph]):
    """Non-deterministic Finite Word automaton

    A non-deterministic automaton that recognizes finite words
    """

    def __init__(self, alphabet: Collection[Alph]) -> None:
        self._alph = alphabet
        self._adj: dict[int, dict[Alph, set[int]]] = dict()  # type: ignore[assignment]
        self._initial_states: set[int] = set()
        self._final_states: set[int] = set()
        self._is_deterministic: bool | None = None

    def add_location(self, location: int, initial: bool = False, final: bool = False) -> None:
        """Add a location to the automaton."""
        if location in self._adj:
            raise ValueError(f"Location {location} already exists in automaton")
        if initial:
            self._initial_states.add(location)
        if final:
            self._initial_states.add(location)
        assert len(self._adj[location]) == 0

    @override
    def add_transition(self, src: int, guard: Alph, dst: Iterable[int | tuple[int, ...]]) -> None:
        dst = list(dst)
        for check_dst in dst:
            # Check that all the incoming destinations are not tuples, i.e., universal transitions
            if isinstance(check_dst, int):
                self._adj.setdefault(src, dict()).setdefault(guard, set()).add(check_dst)
            else:
                raise TypeError(f"Unsupported destination for nondeterministic automaton: {check_dst}")
        self._is_deterministic = len(dst) == 1

    @override
    def contains(self, location: int) -> bool:
        return location in self._adj

    @override
    def num_locations(self) -> int:
        return len(self._adj.keys())

    @override
    def transition(self, location: int, symbol: Alph) -> set[int | tuple[int, ...]]:
        return self._adj[location][symbol]  # type: ignore[return-value]

    @property
    @override
    def is_deterministic(self) -> bool:
        return self._is_deterministic if self._is_deterministic is not None else True

    @property
    @override
    def is_alternating(self) -> bool:
        return False

    @override
    def is_initial(self, state: int) -> bool:
        return state in self._initial_states

    @override
    def is_accepting(self, state: int) -> bool:
        return state in self._final_states
