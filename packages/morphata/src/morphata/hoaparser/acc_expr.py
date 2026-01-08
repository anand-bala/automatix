"""Acceptance condition expressions for automata

This module extends the standard HOA v1 format with
a Final(n) operator for expressing finite-word acceptance conditions.

Extensions to HOA v1:
    - Final(n): For finite-word automata (not part of standard HOA specification)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import override


class AccExpr(ABC):
    """Generalized omega-regular and regular acceptance conditions

    Acceptance formulas are positive Boolean formula over atoms of the form
    `t`, `f`, `Inf(n)`, `Fin(n)`, or `Final(n)`, where `n` is a non-negative
    integer denoting an acceptance set.

    Standard HOA v1 operators (for omega-automata):
    - `t` denotes the true acceptance condition: any run is accepting
    - `f` denotes the false acceptance condition: no run is accepting
    - `Inf(n)` means that a run is accepting if it visits infinitely often
        the acceptance set `n`
    - `Fin(n)` means that a run is accepting if it visits finitely often the
        acceptance set `n`

    Extended operators (morphata-specific):
    - `Final(n)` means that a finite run is accepting if it ends in a state
        marked with acceptance set `n` (for finite-word automata)

    The above atoms can be combined using only the operator `&` and `|`
    (with obvious semantics), and parentheses for grouping. Note that there
    is no negation, but an acceptance condition can be negated swapping `t`
    and `f`, `&` and `|`, and `Fin(n)` and `Inf(n)`.

    Examples:
    - `Inf(0)&Inf(1)`: Generalized Büchi acceptance - accepting runs visit
        both acceptance set 0 and set 1 infinitely often
    - `Fin(0)|Fin(1)`: Generalized co-Büchi acceptance - accepting runs visit
        set 0 or set 1 finitely often
    - `(Fin(0)&Inf(1)) | (Fin(2)&Inf(3)) | (Fin(4)&Inf(5))`: Rabin acceptance
        with 3 pairs
    - `Final(0)`: Finite-word acceptance - accepting runs end in states marked
        with set 0
    """

    def __and__(self, other: AccExpr) -> AccExpr:
        match (self, other):
            case (Literal(False), _) | (_, Literal(False)):
                return Literal(False)
            case (Literal(True), expr) | (expr, Literal(True)):
                return expr
            case (And(lhs), And(rhs)):
                return And(lhs + rhs)
            case (And(args), expr) | (expr, And(args)):
                return And(args + [expr])
            case _:
                return And([self, other])

    def __or__(self, other: AccExpr) -> AccExpr:
        match (self, other):
            case (Literal(True), _) | (_, Literal(True)):
                return Literal(True)
            case (Literal(False), expr) | (expr, Literal(False)):
                return expr
            case (Or(lhs), Or(rhs)):
                return Or(lhs + rhs)
            case (Or(args), expr) | (expr, Or(args)):
                return Or(args + [expr])
            case _:
                return Or([self, other])

    @abstractmethod
    def dual(self) -> AccExpr: ...


@dataclass(frozen=True, slots=True, eq=True)
class And(AccExpr):
    """Logical conjunction of acceptance expressions."""

    args: list[AccExpr]
    """Subexpressions to conjoin"""

    @override
    def dual(self) -> AccExpr:
        """Dual of conjunction (becomes disjunction by De Morgan's law)."""
        return Or([e.dual() for e in self.args])

    @override
    def __str__(self) -> str:
        return "(" + " & ".join(str(arg) for arg in self.args) + ")"


@dataclass(frozen=True, slots=True, eq=True)
class Or(AccExpr):
    """Logical disjunction of acceptance expressions."""

    args: list[AccExpr]
    """Subexpressions to disjoin"""

    @override
    def dual(self) -> AccExpr:
        """Dual of disjunction (becomes conjunction by De Morgan's law)."""
        return And([e.dual() for e in self.args])

    @override
    def __str__(self) -> str:
        return "(" + " | ".join(str(arg) for arg in self.args) + ")"


@dataclass(frozen=True, slots=True, eq=True)
class Fin(AccExpr):
    """Finitely often acceptance (set visited finitely many times)."""

    arg: int
    """Acceptance set index"""
    invert: bool = False
    """Invert the acceptance set"""

    @override
    def dual(self) -> AccExpr:
        """Dual is infinitely often."""
        return Inf(self.arg, self.invert)

    @override
    def __str__(self) -> str:
        if self.invert:
            return f"Fin(!{self.arg})"
        return f"Fin({self.arg})"


@dataclass(frozen=True, slots=True, eq=True)
class Inf(AccExpr):
    """Infinitely often acceptance (set visited infinitely many times)."""

    arg: int
    """Acceptance set index"""
    invert: bool = False
    """Invert the acceptance set"""

    @override
    def dual(self) -> AccExpr:
        """Dual is finitely often."""
        return Fin(self.arg, self.invert)

    @override
    def __str__(self) -> str:
        if self.invert:
            return f"Inf(!{self.arg})"
        return f"Inf({self.arg})"


@dataclass(frozen=True, slots=True, eq=True)
class Final(AccExpr):
    """Final state acceptance (for finite-word automata).

    This is a morphata-specific extension to the HOA v1 format, not part of the
    standard specification. It provides a natural way to express finite-word
    acceptance conditions in HOA syntax.

    Unlike Inf/Fin which are for omega-automata (infinite words), Final is used
    for finite-word automata where acceptance means reaching a state marked as
    final. The acceptance set index identifies which states are final.

    Semantics:
        - Inf(n): Run visits acceptance set n infinitely often (omega-regular)
        - Fin(n): Run visits acceptance set n finitely often (omega-regular)
        - Final(n): Run ends in a state marked with acceptance set n (regular)

    Example:
        Acceptance: 1 Final(0)
        State: 1 {0}  # State 1 is marked as final
    """

    arg: int
    """Acceptance set index"""

    @override
    def dual(self) -> AccExpr:
        """Dual is always reject (no defined dual for finite acceptance)."""
        return Literal(False)

    @override
    def __str__(self) -> str:
        return f"Final({self.arg})"


@dataclass(frozen=True, slots=True, eq=True)
class Literal(AccExpr):
    """Boolean literal (always accept or never accept)."""

    value: bool
    """True for always accept, False for never accept"""

    @override
    def dual(self) -> AccExpr:
        """Negation of the literal."""
        return Literal(not self.value)

    @override
    def __str__(self) -> str:
        return "t" if self.value else "f"
