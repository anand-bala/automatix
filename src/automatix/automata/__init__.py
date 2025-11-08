import typing
from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import Generic, TypeAlias, TypeVar

import logic_asts.base as exprs
import networkx as nx
from typing_extensions import final, overload, override

import automatix.automata.acc as acc

In = TypeVar("In", bound=Hashable)
Out = TypeVar("Out")

AcceptanceCondition: TypeAlias = (
    acc.Finite | acc.Buchi | acc.CoBuchi | acc.GeneralizedBuchi | acc.GeneralizedCoBuchi | acc.Rabin | acc.Streett | acc.Muller
)

State: TypeAlias = int | frozenset[int] | tuple[frozenset[int]]

Guard: TypeAlias = str | exprs.BaseExpr[str]


class AbstractAutomaton(ABC, Generic[In, Out]):
    r"""
    The `Automaton` class defines a general interface for all automata-like transition
    systems, and can be used by other components in `automatix` to define their own
    semantics.

    The locations in the automaton are always indexed by an `int`, but the actual
    **state** of the automaton (`State`) can be an `int` (deterministic), a set of `int`
    (non-deterministic) or a list of set of `int` (alternating).
    In the alternating case, the user should read the `State` as a conjunction over
    disjunction of states, while in the non-deterministic case, the user should just
    read it as a disjunction over states.
    """

    @property
    @abstractmethod
    def num_states(self) -> int: ...

    @property
    @abstractmethod
    def initial_states(self) -> frozenset[int]: ...

    @property
    @abstractmethod
    def acceptance_condition(self) -> AcceptanceCondition: ...

    @abstractmethod
    def __call__(self, input_symbol: In, state: State) -> tuple[Out, State]: ...

    @property
    @abstractmethod
    def is_deterministic(self) -> bool:
        """Check if the automaton is deterministic"""

    @property
    @abstractmethod
    def is_alternating(self) -> bool:
        """Check if the automaton has alternating transitions"""


@final
class ExplicitNWA(AbstractAutomaton[frozenset[str], None]):
    """An explicit representation of a non-deterministic word automaton.

    The reason this representation is deemed to be "explicit" is that we use networkx DiGraphs to represent the entire automaton.
    In general, this may be inefficient if the successor states can be computed lazily.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph[int] = nx.DiGraph()
        self._initial: set[int] = set()
        self._acc_condition: AcceptanceCondition = acc.Finite(frozenset())
        self._deterministic = True

    def add_location(self, location: int, initial: bool = False) -> None:
        if location in self._graph.nodes:
            raise ValueError(f"Location {location} already exists in automaton")
        if initial:
            self._initial.add(location)
        self._graph.add_node(location, initial=initial)

    def add_transition(self, src: int, dst: int, guard: str | exprs.BaseExpr[str]) -> None:
        if (src, dst) in self._graph.edges:
            raise ValueError(f"Transition from {src} to {dst} already exists. Did you want to update the guard?")
        if isinstance(guard, str):
            import logic_asts
            from lark import LarkError

            try:
                expr = logic_asts.parse_expr(guard)
            except LarkError as e:
                raise ValueError("Unable to parse guard as a boolean expression") from e
            assert isinstance(
                expr, (exprs.Literal, exprs.Variable, exprs.Not, exprs.And, exprs.Or, exprs.Implies, exprs.Equiv, exprs.Xor)
            )
        else:
            expr = guard

        self._graph.add_edge(src, dst, guard=guard)

    @property
    @override
    def num_states(self) -> int:
        return len(self._graph)

    @property
    @override
    def initial_states(self) -> frozenset[int]:
        return frozenset(self._initial)

    @property
    @override
    def acceptance_condition(self) -> AcceptanceCondition:
        return self._acc_condition

    @acceptance_condition.setter
    def acceptance_condition(self, acc: AcceptanceCondition) -> None:
        self._acc_condition = acc

    @overload
    def guards(self, src: int, dst: int) -> exprs.BaseExpr[str]: ...

    @overload
    def guards(self, src: int, dst: None = None) -> dict[int, exprs.BaseExpr[str]]: ...

    def guards(self, src: int, dst: int | None = None) -> exprs.BaseExpr[str] | dict[int, exprs.BaseExpr[str]]:
        """Get a transition guard or the set of transition guards for each successor state"""
        if dst is None:
            return {succ: guard for _, succ, guard in self._graph.edges(src, "guard")}  # type: ignore[var-annotated]
        return typing.cast(exprs.BaseExpr[str], self._graph.edges[src, dst]["guard"])

    @override
    def __call__(self, input_symbol: frozenset[str], state: State) -> tuple[None, int | frozenset[int]]:
        if isinstance(state, int):
            ret = set(dst for dst, guard in self.guards(state).items() if exprs.simple_eval(guard, input_symbol))  # type: ignore[arg-type]
        elif isinstance(state, frozenset):
            ret = set()
            for dst in (self.__call__(input_symbol, s)[1] for s in state):
                if isinstance(dst, int):
                    ret.add(dst)
                else:
                    ret.update(dst)
        elif isinstance(state, tuple):
            raise TypeError("Nondeterminisitc Automata cannot have alternating states")
        if len(ret) == 1:
            return None, ret.pop()
        else:
            return None, frozenset(ret)

    @property
    @override
    def is_deterministic(self) -> bool:
        """Check if the automaton is deterministic"""
        return False

    @property
    @override
    def is_alternating(self) -> bool:
        return False


# Re-export finite_word module components for convenience
from automatix.automata.finite_word import NFA, AutomatonOperator, make_automaton_operator  # noqa: E402

__all__ = [
    "AbstractAutomaton",
    "ExplicitNWA",
    "NFA",
    "AutomatonOperator",
    "make_automaton_operator",
    "State",
    "Guard",
    "AcceptanceCondition",
]
