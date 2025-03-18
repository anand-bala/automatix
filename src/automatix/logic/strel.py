import itertools
import math
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lark import Lark, Transformer, ast_utils
from typing_extensions import override

from .ltl import (
    AndOp,
    Constant,
    EventuallyOp,
    Expr,
    GloballyOp,
    Identifier,
    NextOp,
    NotOp,
    OrOp,
    TimeInterval,
    UntilOp,
    _Ast,
    _TransformTerminals,
    false,
    true,
)

STREL_GRAMMAR_FILE = Path(__file__).parent / "strel.lark"


@dataclass(eq=True, frozen=True, slots=True)
class DistanceInterval(_Ast):
    start: Optional[float]
    end: Optional[float]

    def __str__(self) -> str:
        return f"[{self.start or 0.0:g}, {self.end or math.inf:g}]"

    def __post_init__(self) -> None:
        if self.start is None:
            object.__setattr__(self, "start", 0.0)
        if self.end is None:
            object.__setattr__(self, "end", math.inf)
        match (self.start, self.end):
            case (float(start), float(end)) if start < 0 or end < 0:
                raise ValueError("Distane cannot be less than 0")
            case (float(start), float(end)) if start >= end:
                raise ValueError(f"Distance interval cannot have `start` >= `end` ({start} >= {end})")


@dataclass(eq=True, frozen=True, slots=True)
class EverywhereOp(Expr):
    interval: DistanceInterval
    arg: Expr

    def __str__(self) -> str:
        return f"(everywhere{self.interval} {self.arg})"

    @override
    def expand_intervals(self) -> "Expr":
        return EverywhereOp(self.interval, self.arg.expand_intervals())


@dataclass(eq=True, frozen=True, slots=True)
class SomewhereOp(Expr):
    interval: DistanceInterval
    arg: Expr

    def __str__(self) -> str:
        return f"(somewhere{self.interval} {self.arg})"

    @override
    def expand_intervals(self) -> "Expr":
        return SomewhereOp(self.interval, self.arg.expand_intervals())


@dataclass(eq=True, frozen=True, slots=True)
class EscapeOp(Expr):
    interval: DistanceInterval
    arg: Expr

    def __str__(self) -> str:
        return f"(escape{self.interval} {self.arg})"

    @override
    def expand_intervals(self) -> "Expr":
        return EscapeOp(self.interval, self.arg.expand_intervals())


@dataclass(eq=True, frozen=True, slots=True)
class ReachOp(Expr):
    lhs: Expr
    interval: DistanceInterval
    rhs: Expr

    def __str__(self) -> str:
        return f"({self.lhs} reach{self.interval} {self.rhs})"

    @override
    def expand_intervals(self) -> "Expr":
        return ReachOp(
            interval=self.interval,
            lhs=self.lhs.expand_intervals(),
            rhs=self.rhs.expand_intervals(),
        )


with open(STREL_GRAMMAR_FILE, "r") as grammar:
    STREL_GRAMMAR = Lark(
        grammar,
        start="phi",
        strict=True,
    )


def _to_ast_transformer() -> Transformer:
    ast = types.ModuleType("ast")
    for c in itertools.chain(
        [TimeInterval, DistanceInterval],
        Expr.__subclasses__(),
    ):
        ast.__dict__[c.__name__] = c
    return ast_utils.create_transformer(ast, _TransformTerminals())


TO_AST_TRANSFORMER = _to_ast_transformer()


def parse(expr: str) -> Expr:
    tree = STREL_GRAMMAR.parse(expr)
    return TO_AST_TRANSFORMER.transform(tree)


__all__ = [
    "parse",
    "Expr",
    "UntilOp",
    "GloballyOp",
    "EventuallyOp",
    "NextOp",
    "AndOp",
    "OrOp",
    "NotOp",
    "Identifier",
    "Constant",
    "true",
    "false",
    "TimeInterval",
    "DistanceInterval",
    "ReachOp",
    "SomewhereOp",
    "EverywhereOp",
    "EscapeOp",
]
