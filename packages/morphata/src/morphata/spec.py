"""Base interfaces for automata structures.

This module provides pure structural interfaces for automata without
any weight function or semiring concepts. These interfaces are extended
by automatix for weighted automata semantics.
"""

from __future__ import annotations

import functools
import typing as ty
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable
from collections.abc import Set as AbstractSet

import logic_asts.base as logic

type BoolExpr[Var] = logic.BaseExpr[Var]

State = ty.TypeVar("State", bound=Hashable)
Symbol = ty.TypeVar("Symbol", contravariant=True)


class AcceptanceCondition(ABC, ty.Generic[State]):
    """Base class for finite and omega-regular acceptance conditions"""

    @classmethod
    @abstractmethod
    def is_omega_regular(cls) -> bool:
        """Determines if the concrete `AcceptanceCondition` is ω-regular or not."""


class Domain(ABC, ty.Generic[State, Symbol]):
    """Capability-based domain.

    Returning `None` means that the property of the domain is not enumerable or is
    symbolic.
    """

    @property
    @abstractmethod
    def states(self) -> Iterable[State] | None:
        return None

    @property
    @abstractmethod
    def symbols(self) -> Iterable[Symbol] | None:
        return None


class DeterministicTransitions(ABC, ty.Generic[State, Symbol]):
    @abstractmethod
    def __call__(self, state: State, symbol: Symbol) -> State: ...


class NonDeterministicTransitions(ABC, ty.Generic[State, Symbol]):
    @abstractmethod
    def __call__(self, state: State, symbol: Symbol) -> Iterable[State]: ...


class UniversalTransitions(ABC, ty.Generic[State, Symbol]):
    @abstractmethod
    def __call__(self, state: State, symbol: Symbol) -> Iterable[State]: ...


class AlternatingTransitions(ABC, ty.Generic[State, Symbol]):
    @abstractmethod
    def __call__(self, state: State, symbol: Symbol) -> BoolExpr[State]: ...

    def step_run(self, run_state: BoolExpr[State], symbol: Symbol) -> BoolExpr[State]:
        """Given the formula representing the current state of the run tree, compute the successor states in the run tree given the input symbol."""
        cache: dict[BoolExpr[State], BoolExpr[State]] = dict()
        for expr in run_state.iter_subtree():
            match expr:
                case logic.Literal():
                    cache[expr] = expr
                case logic.Variable(q):
                    cache[expr] = self.__call__(ty.cast(State, q), symbol)
                case logic.Or(args):
                    cache[expr] = functools.reduce(lambda a, b: a | b, (cache[ty.cast(BoolExpr[State], arg)] for arg in args))
                case logic.And(args):
                    cache[expr] = functools.reduce(lambda a, b: a & b, (cache[ty.cast(BoolExpr[State], arg)] for arg in args))
                case _:
                    raise TypeError(f"run_state expr can only be positive boolean expressions, got {type(expr)}")

        return cache[run_state]


type InitialState[State] = State | AbstractSet[State] | BoolExpr[State]
type TransitionRelation[State, Symbol] = (
    DeterministicTransitions[State, Symbol]
    | NonDeterministicTransitions[State, Symbol]
    # | UniversalTransitions[State, Symbol]
    | AlternatingTransitions[State, Symbol]
)


class Automaton(ty.Generic[State, Symbol]):
    domain: Domain[State, Symbol]
    initial: InitialState[State]
    delta: TransitionRelation[State, Symbol]
    acceptance: AcceptanceCondition[State]

    def __init__(
        self,
        domain: Domain[State, Symbol],
        initial: State | Iterable[State] | BoolExpr[State],
        delta: TransitionRelation[State, Symbol],
        acceptance: AcceptanceCondition[State],
    ) -> None:
        self.domain = domain
        self.acceptance = acceptance
        if isinstance(initial, Iterable):
            initial = frozenset(initial)
        elif logic.is_bool_expr(initial) and not isinstance(delta, AlternatingTransitions):
            raise TypeError(
                "Cannot instantiate 'Automaton' with Boolean expression initial configuration and {type(delta).__name__} transitions"
            )

        self.initial = initial
        self.delta = delta
