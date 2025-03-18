from abc import ABC, abstractmethod
from dataclasses import dataclass


class AcceptanceFormula(ABC):
    """
    Acceptance formulas are positive Boolean formula over atoms of the form
    `t`, `f`, `Inf(n)`, or `Fin(n)`, where `n` is a non-negative integer
    denoting an acceptance set.

    - `t` denotes the true acceptance condition: any run is accepting
    - `f` denotes the false acceptance condition: no run is accepting
    - `Inf(n)` means that a run is accepting if it visits infinitely often
      the acceptance set `n`
    - `Fin(n)` means that a run is accepting if it visits finitely often the
      acceptance set `n`

    The above atoms can be combined using only the operator `&` and `|`
    (with obvious semantics), and parentheses for grouping. Note that there
    is no negation, but an acceptance condition can be negated swapping `t`
    and `f`, `&` and `|`, and `Fin(n)` and `Inf(n)`.

    For instance the formula `Inf(0)&Inf(1)` specifies that accepting runs
    should visit infinitely often the acceptance 0, and infinitely often the
    acceptance set 1. This corresponds the generalized Büchi acceptance with
    two sets.

    The opposite acceptance condition `Fin(0)|Fin(1)` is known as
    *generalized co-Büchi acceptance* (with two sets). Accepting runs have
    to visit finitely often set 0 *or* finitely often set 1.

    A *Rabin acceptance condition* with 3 pairs corresponds to the following
    formula: `(Fin(0)&Inf(1)) | (Fin(2)&Inf(3)) |
    (Fin(4)&Inf(5))`
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None: ...  # noqa: ANN002, ANN003

    def __and__(self, other: "AcceptanceFormula") -> "AcceptanceFormula":
        match (self, other):
            case (And(lhs), And(rhs)):
                return And(lhs + rhs)
            case (And(args), expr) | (expr, And(args)):
                return And(args + [expr])
            case (lhs, rhs):
                return And([lhs, rhs])

    def __or__(self, other: "AcceptanceFormula") -> "AcceptanceFormula":
        match (self, other):
            case (Or(lhs), Or(rhs)):
                return Or(lhs + rhs)
            case (Or(args), expr) | (expr, Or(args)):
                return Or(args + [expr])
            case (lhs, rhs):
                return Or([lhs, rhs])


@dataclass(frozen=True, slots=True, eq=True)
class And(AcceptanceFormula):
    args: list[AcceptanceFormula]


@dataclass(frozen=True, slots=True, eq=True)
class Or(AcceptanceFormula):
    args: list[AcceptanceFormula]


@dataclass(frozen=True, slots=True, eq=True)
class Fin(AcceptanceFormula):
    invert: bool
    arg: int


@dataclass(frozen=True, slots=True, eq=True)
class Inf(AcceptanceFormula):
    invert: bool
    arg: int


@dataclass(frozen=True, slots=True, eq=True)
class Literal(AcceptanceFormula):
    value: bool
