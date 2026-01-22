"""Concrete classes for acceptance conditions

Provides abstract and concrete acceptance condition classes for omega-automata
and finite-word automata.
"""

from __future__ import annotations

import typing as ty
from collections.abc import Iterable

from attrs import frozen
from typing_extensions import overload

from morphata.spec import AcceptanceCondition, State


@overload
def acc_from_name(name: ty.Literal["Finite"], arg: Iterable[State], /) -> Finite[State]: ...


@overload
def acc_from_name(name: ty.Literal["Buchi"], arg: Iterable[State], /) -> Buchi[State]: ...


@overload
def acc_from_name(name: ty.Literal["co-Buchi"], arg: Iterable[State], /) -> CoBuchi[State]: ...


@overload
def acc_from_name(name: ty.Literal["generalized-Buchi"], /, *args: Iterable[State]) -> GeneralizedBuchi[State]: ...


@overload
def acc_from_name(name: ty.Literal["generalized-co-Buchi"], /, *args: Iterable[State]) -> GeneralizedCoBuchi[State]: ...


@overload
def acc_from_name(name: ty.Literal["Muller"], /, *args: Iterable[State]) -> Muller[State]: ...


@overload
def acc_from_name(name: ty.Literal["Streett"], /, *args: AccPair[State]) -> Streett[State]: ...


@overload
def acc_from_name(name: ty.Literal["Rabin"], /, *args: AccPair[State]) -> Rabin[State]: ...


def acc_from_name(name: str, /, *args) -> AcceptanceCondition[State]:
    match name:
        case "Finite":
            assert len(args) == 1, "Finite acceptance condition requires 1 accepting set"
            arg = args[0]
            assert isinstance(arg, Iterable)
            return Finite(frozenset(arg))
        case "Buchi":
            assert len(args) == 1, "Buchi acceptance condition requires 1 accepting set"
            arg = args[0]
            assert isinstance(arg, Iterable)
            return Buchi(frozenset(arg))
        case "generalized-Buchi":
            assert len(args) >= 0 and all(isinstance(arg, Iterable) for arg in args), (
                "Generalized Buchi condition needs a list of accepting sets"
            )
            return GeneralizedBuchi(tuple(frozenset(arg) for arg in args))
        case "co-Buchi":
            assert len(args) == 1, "CoBuchi acceptance condition requires 1 non-accepting set"
            arg = args[0]
            assert isinstance(arg, Iterable)
            return CoBuchi(frozenset(arg))
        case "generalized-co-Buchi":
            assert len(args) >= 0 and all(isinstance(arg, Iterable) for arg in args), (
                "Generalized CoBuchi condition needs a list of non-accepting sets"
            )
            return GeneralizedCoBuchi(tuple(frozenset(arg) for arg in args))
        case "Streett":
            assert len(args) >= 0 and all(isinstance(arg, tuple) and len(arg) == 2 for arg in args), (
                "Streett condition needs a list of 2-tuples of rejecting (Fin) and accepting (Inf) sets"
            )
            return Streett(tuple(AccPair(*(frozenset(s) for s in arg)) for arg in args))
        case "Rabin":
            assert len(args) >= 0 and all(isinstance(arg, tuple) and len(arg) == 2 for arg in args), (
                "Rabin condition needs a list of 2-tuples of rejecting (Fin) and accepting (Inf) sets"
            )
            return Rabin(tuple(AccPair(*(frozenset(s) for s in arg)) for arg in args))

        case _:
            raise ValueError(f"Unknown/unsupported named acceptance condition: {name} {args=}")


# @frozen
# class GenericCondition(AcceptanceCondition[State]):
#     acceptance_sets: tuple[frozenset[State], ...]
#     expr: AccExpr

#     @override
#     def __len__(self) -> int:
#         return self.num_sets

#     @override
#     def to_expr(self) -> AccExpr:
#         return self.expr


@frozen
class Finite(AcceptanceCondition[State]):
    """Finite-word acceptance condition.

    For finite automata, acceptance means reaching a final state. This uses
    a single acceptance set  to mark final states.
    """

    accepting: frozenset[State]

    @classmethod
    def is_omega_regular(cls) -> bool:
        """Finite-word acceptance is not omega-regular."""
        return False


@frozen
class Buchi(AcceptanceCondition[State]):
    """Büchi condition: a run, r, is accepting iff inf(r) intersects with `accepting`"""

    accepting: frozenset[State]

    @classmethod
    def is_omega_regular(cls) -> bool:
        """Büchi acceptance is omega-regular."""
        return True


@frozen
class GeneralizedBuchi(AcceptanceCondition[State]):
    """Generalized Büchi condition: a run, r, is accepting iff inf(r) intersects with `accepting[i]` for some i"""

    accepting: tuple[frozenset[State], ...]

    @classmethod
    def is_omega_regular(cls) -> bool:
        """Generalized Büchi acceptance is omega-regular."""
        return True


@frozen
class CoBuchi(AcceptanceCondition[State]):
    """co-Büchi condition: a run, r, is accepting iff inf(r) does not intersect with `rejecting`"""

    rejecting: frozenset[State]

    @classmethod
    def is_omega_regular(cls) -> bool:
        """co-Büchi acceptance is omega-regular."""
        return True


@frozen
class GeneralizedCoBuchi(AcceptanceCondition[State]):
    """Generalized co-Büchi condition: a run, r, is accepting iff inf(r) does not intersect with `rejecting[i]` for some i"""

    rejecting: tuple[frozenset[State], ...]

    @classmethod
    def is_omega_regular(cls) -> bool:
        """Generalized co-Büchi acceptance is omega-regular."""
        return True


class AccPair(ty.NamedTuple, ty.Generic[State]):
    """Pair of accepting and rejecting state sets for Rabin/Streett conditions."""

    rejecting: frozenset[State]
    """States that must not appear infinitely often"""
    accepting: frozenset[State]
    """States that must appear infinitely often"""


@frozen
class Streett(AcceptanceCondition[State]):
    """Streett condition: a run, r, is accpting iff _for all_ `i`, we have that inf(r) does not intersect with `pairs[i].rejecting` and does intersect with `pairs[i].accepting`"""

    pairs: tuple[AccPair[State], ...]

    @property
    def index(self) -> int:
        """Number of pairs in this condition."""
        return len(self.pairs)

    @classmethod
    def is_omega_regular(cls) -> bool:
        """Streett acceptance is omega-regular."""
        return True


@frozen
class Rabin(AcceptanceCondition[State]):
    """Rabin condition: a run, r, is accpting iff _for some_ `i`, we have that inf(r) does not intersect with `pairs[i].rejecting` and does intersect with `pairs[i].accepting`"""

    pairs: tuple[AccPair[State], ...]

    @property
    def index(self) -> int:
        """Number of pairs in this condition."""
        return len(self.pairs)

    @classmethod
    def is_omega_regular(cls) -> bool:
        """Rabin acceptance is omega-regular."""
        return True


@frozen
class Muller(AcceptanceCondition[State]):
    """Muller condition: a run, r, is accepting iff for some `i`, we have that inf(r) is exactly `sets[i]`"""

    sets: tuple[frozenset[State], ...]

    @classmethod
    def is_omega_regular(cls) -> bool:
        """Muller acceptance is omega-regular."""
        return True
