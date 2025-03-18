from abc import ABC, abstractmethod
from dataclasses import dataclass


class LabelExpr(ABC):
    """
    The `LabelExpr` is used to label transitions in an automata.
    The expression is a boolean expression over the atomic predicates (referred by their
    index) or over Boolean literals `True` or `False`.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None: ...  # noqa: ANN002, ANN003

    def __and__(self, other: "LabelExpr") -> "LabelExpr":
        match (self, other):
            case (And(lhs), And(rhs)):
                return And(lhs + rhs)
            case (And(args), expr) | (expr, And(args)):
                return And(args + [expr])
            case (lhs, rhs):
                return And([lhs, rhs])

    def __or__(self, other: "LabelExpr") -> "LabelExpr":
        match (self, other):
            case (Or(lhs), Or(rhs)):
                return Or(lhs + rhs)
            case (Or(args), expr) | (expr, Or(args)):
                return Or(args + [expr])
            case (lhs, rhs):
                return Or([lhs, rhs])


@dataclass(frozen=True, slots=True, eq=True)
class And(LabelExpr):
    args: list[LabelExpr]


@dataclass(frozen=True, slots=True, eq=True)
class Or(LabelExpr):
    args: list[LabelExpr]


@dataclass(frozen=True, slots=True, eq=True)
class Not(LabelExpr):
    arg: LabelExpr


@dataclass(frozen=True, slots=True, eq=True)
class Predicate(LabelExpr):
    idx: int


@dataclass(frozen=True, slots=True, eq=True)
class Literal(LabelExpr):
    value: bool
