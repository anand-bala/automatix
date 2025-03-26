# pyright: reportMissingParameterType=false
from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import override


class LabelExpr(ABC):
    """
    The `LabelExpr` is used to label transitions in an automata.
    The expression is a boolean expression over the atomic predicates (referred by their
    index) or over Boolean literals `True` or `False`.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None: ...  # noqa: ANN002, ANN003

    def __invert__(self) -> "LabelExpr":
        match self:
            case Not(arg):
                return arg
            case _:
                return Not(self)

    def __and__(self, other: "LabelExpr") -> "LabelExpr":
        match (self, other):
            case (Literal(False), _) | (_, Literal(False)):
                return Literal(False)
            case (Literal(True), expr) | (expr, Literal(True)):
                return expr
            case (And(lhs), And(rhs)):
                return And(lhs + rhs)
            case (And(args), expr) | (expr, And(args)):
                return And(args + [expr])
            case (lhs, rhs):
                return And([lhs, rhs])

    def __or__(self, other: "LabelExpr") -> "LabelExpr":
        match (self, other):
            case (Literal(True), _) | (_, Literal(True)):
                return Literal(True)
            case (Literal(False), expr) | (expr, Literal(False)):
                return expr
            case (Or(lhs), Or(rhs)):
                return Or(lhs + rhs)
            case (Or(args), expr) | (expr, Or(args)):
                return Or(args + [expr])
            case (lhs, rhs):
                return Or([lhs, rhs])


@dataclass(frozen=True, slots=True, eq=True)
class And(LabelExpr):
    args: list[LabelExpr]

    @override
    def __str__(self) -> str:
        return "(" + " & ".join(str(arg) for arg in self.args) + ")"


@dataclass(frozen=True, slots=True, eq=True)
class Or(LabelExpr):
    args: list[LabelExpr]

    @override
    def __str__(self) -> str:
        return "(" + " | ".join(str(arg) for arg in self.args) + ")"


@dataclass(frozen=True, slots=True, eq=True)
class Not(LabelExpr):
    arg: LabelExpr

    @override
    def __str__(self) -> str:
        return f"!{str(self.arg)}"


@dataclass(frozen=True, slots=True, eq=True)
class Predicate(LabelExpr):
    idx: int

    @override
    def __str__(self) -> str:
        return str(self.idx)


@dataclass(frozen=True, slots=True, eq=True)
class Literal(LabelExpr):
    value: bool

    @override
    def __str__(self) -> str:
        return "t" if self.value else "f"

    @override
    def __invert__(self) -> "LabelExpr":
        return Literal(not self.value)
