"""Acceptance condition expressions and implementations.

Provides abstract and concrete acceptance condition classes for omega-automata
and finite-word automata. This module extends the standard HOA v1 format with
a Final(n) operator for expressing finite-word acceptance conditions.

Extensions to HOA v1:
    - Final(n): For finite-word automata (not part of standard HOA specification)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field

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


class AcceptanceCondition(ABC):
    """Omega-regular acceptance conditions

    Some classical acceptance conditions include:

    - Buchi
    - generalized-Buchi
    - co-Buchi
    - generalized-co-Buchi
    - Streett
    - Rabin
    - generalized-Rabin
    - parity
    - all
    - none
    """

    @abstractmethod
    def to_expr(self) -> AccExpr: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @staticmethod
    def from_name(name: str, *args: bool | int | str) -> "AcceptanceCondition":
        props = args
        match name:
            case "Buchi":
                return Buchi()
            case "Finite":
                return Finite()
            case "generalized-Buchi":
                assert len(props) == 1 and isinstance(props[0], int) and props[0] >= 0, (
                    "Generalized Buchi condition needs one integer property"
                )
                return GeneralizedBuchi(props[0])
            case "co-Buchi":
                return CoBuchi()
            case "generalized-co-Buchi":
                assert len(props) == 1 and isinstance(props[0], int) and props[0] >= 0, (
                    "Generalized Co-Buchi condition needs one integer property"
                )
                return GeneralizedCoBuchi(props[0])
            case "Streett":
                assert len(props) == 1 and isinstance(props[0], int) and props[0] >= 0, (
                    "Streett condition needs one integer property"
                )
                return Streett(props[0])
            case "Rabin":
                assert len(props) == 1 and isinstance(props[0], int) and props[0] >= 0, (
                    "Rabin condition needs one integer property"
                )
                return Rabin(props[0])

            case "parity":
                assert (
                    len(props) == 3 and isinstance(props[0], str) and isinstance(props[1], str) and isinstance(props[2], int)
                ), "Parity condition needs 3 properties of (str, str, int)"
                return Parity(props[2], max=props[0] == "max", odd=props[1] == "odd")
            case _:
                raise ValueError(f"Unknown/unsupported named acceptance condition: {name} {props}")


@dataclass(frozen=True)
class GenericCondition(AcceptanceCondition):
    num_sets: int
    expr: AccExpr

    @override
    def __len__(self) -> int:
        return self.num_sets

    @override
    def to_expr(self) -> AccExpr:
        return self.expr


@dataclass(frozen=True)
class Buchi(AcceptanceCondition):
    @override
    def __len__(self) -> int:
        return 1

    @override
    def to_expr(self) -> AccExpr:
        return Inf(0)


@dataclass(frozen=True)
class GeneralizedBuchi(AcceptanceCondition):
    num_sets: int

    @override
    def __len__(self) -> int:
        return self.num_sets

    @override
    def to_expr(self) -> AccExpr:
        return And([Inf(i) for i in range(self.num_sets)])


@dataclass(frozen=True)
class CoBuchi(AcceptanceCondition):
    @override
    def __len__(self) -> int:
        return 1

    @override
    def to_expr(self) -> AccExpr:
        return Fin(0)


@dataclass(frozen=True)
class Finite(AcceptanceCondition):
    """Finite-word acceptance condition (morphata HOA extension).

    This acceptance condition is used with the extended HOA format's Final(n)
    operator to express finite-word automata. It is not part of the standard
    HOA v1 specification.

    For finite automata, acceptance means reaching a final state. This uses
    a single acceptance set (set 0) to mark final states.

    Usage in HOA format:
        acc-name: Finite
        Acceptance: 1 Final(0)
    """

    @override
    def __len__(self) -> int:
        return 1

    @override
    def to_expr(self) -> AccExpr:
        return Final(0)


@dataclass(frozen=True)
class GeneralizedCoBuchi(AcceptanceCondition):
    num_sets: int

    @override
    def __len__(self) -> int:
        return self.num_sets

    @override
    def to_expr(self) -> AccExpr:
        return Or([Fin(i) for i in range(self.num_sets)])


@dataclass(frozen=True)
class Streett(AcceptanceCondition):
    num_pairs: int

    @override
    def __len__(self) -> int:
        return 2 * self.num_pairs

    @override
    def to_expr(self) -> AccExpr:
        """Return the Streett condition as an expression

        Here, for an Streett condition with `n` pairs and an index `i in
        range(0,n)`, the pair `(B_i, G_i)` correspond to the `2 * i` and `2*i
        + 1` sets in the expression.
        """
        terms = [Fin(2 * i) | Inf(2 * i + 1) for i in range(0, self.num_pairs)]
        match len(terms):
            case 0:
                return Literal(False)
            case 1:
                return terms[0]
            case _:
                return And(terms)


@dataclass(frozen=True)
class Rabin(AcceptanceCondition):
    num_pairs: int

    def index(self) -> int:
        return self.num_pairs

    @override
    def __len__(self) -> int:
        return 2 * self.num_pairs

    @override
    def to_expr(self) -> AccExpr:
        """Return the Rabin condition as an expression

        Here, for an Rabin condition with `n` pairs and  `i in
        range(0,n)`, the pair `(B_i, G_i)` correspond to the `2 * i` and `2*i
        + 1` sets in the expression.
        """
        terms = [Fin(2 * i) & Inf(2 * i + 1) for i in range(0, self.num_pairs)]
        match len(terms):
            case 0:
                return Literal(True)
            case 1:
                return terms[0]
            case _:
                return Or(terms)


@dataclass(frozen=True)
class Parity(AcceptanceCondition):
    num_sets: int
    max: bool = field(kw_only=True)
    odd: bool = field(kw_only=True)

    @override
    def __len__(self) -> int:
        return self.num_sets

    @override
    def to_expr(self) -> AccExpr:
        res: AccExpr
        if self.max:
            res = Literal((self.num_sets & 1) == self.odd)
        else:
            res = Literal(self.odd)

        if self.num_sets == 0:
            return res
        # When you look at something like
        #    acc-name: parity min even 5
        #    Acceptance: 5 Inf(0) | (Fin(1) & (Inf(2) | (Fin(3) & Inf(4))))
        # remember that we build it from right to left.

        iterator: Iterable[int]
        iterator = range(0, self.num_sets)
        if not self.max:
            iterator = reversed(iterator)
        for i in iterator:
            if (i & 1) == self.odd:
                res = res | Inf(i)
            else:
                res = res & Fin(i)

        return res
