# pyright: reportExplicitAny=false

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lark import Lark, ParseTree, Token, Transformer, v_args
from typing_extensions import override

import automatix.hoa.label_expr as guard
from automatix.hoa.label_expr import LabelExpr
from automatix.omega import (
    AcceptanceCondition,
    AccExpr,
    Fin,
    GenericCondition,
    Inf,
    Literal,
)


class HoaSyntaxError(Exception):
    def __init__(self, label: str = "") -> None:
        super().__init__()
        self.label: str = label

    @override
    def __str__(self) -> str:
        return f"{self.label}"


@dataclass
class IncorrectVersionError(HoaSyntaxError):
    label: str = "automatix only supports v1"


class DuplicateHeaderError(HoaSyntaxError):
    def __init__(self, header: str) -> None:
        super().__init__(f"Header field `{header}` already defined")


class DuplicateAliasError(HoaSyntaxError):
    def __init__(self, alias: str) -> None:
        super().__init__(f"Duplicate alias definition: `{alias}`")


class MissingHeaderError(HoaSyntaxError):
    def __init__(self, header: str) -> None:
        super().__init__(f"Missing mandatory field `{header}`")


class UndefinedAliasError(HoaSyntaxError):
    def __init__(self, alias: str) -> None:
        super().__init__(f"Undefined alias present in expression: `{alias}`")


@dataclass(frozen=True, eq=True, kw_only=True)
class Header:
    acc: AcceptanceCondition
    name: str | None = None
    num_states: int | None = None
    initial: list[list[int]] = field(default_factory=list)
    predicates: list[str] = field(default_factory=list)
    aliases: dict[str, LabelExpr] = field(default_factory=dict)
    properties: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class State:
    idx: int
    label: LabelExpr | None = None
    acc_set: list[int] | None = None
    description: str | None = None

    @override
    def __hash__(self) -> int:
        return hash(self.idx)


@dataclass(frozen=True)
class Transition:
    dst: list[int]
    label: LabelExpr | None = None
    acc_set: list[int] | None = None


@dataclass(frozen=True)
class ParsedAutomaton:
    header: Header
    body: dict[State, list[Transition]]


class AstTransformer(Transformer):  # pyright: ignore[reportMissingTypeArgument]
    def __init__(self, visit_tokens: bool = True) -> None:
        super().__init__(visit_tokens)
        self._aliases: dict[str, LabelExpr] = dict()
        self._num_states: int | None = None
        self._initial_states: list[list[int]] = []
        self._predicates: list[str] = []

        self._num_accept_sets: int
        self._acc: AccExpr

        self._acc_name: tuple[str, list[bool | int | str] | None] | None = None
        self._name: str | None = None

    @v_args(inline=True)
    def automaton(self, header: Header, body: dict[State, list[Transition]]) -> ParsedAutomaton:
        aut = ParsedAutomaton(header, body)

        return aut

    def header(self, _: list[Any]) -> Header:
        if not hasattr(self, "_acc") or not hasattr(self, "_num_accept_sets"):
            raise MissingHeaderError("Acceptance")
        if self._acc_name is not None:
            acc_name, acc_props = self._acc_name
            acc = AcceptanceCondition.from_name(acc_name, acc_props)
            assert acc.to_expr() == self._acc
        else:
            acc = GenericCondition(self._num_accept_sets, self._acc)
        return Header(
            acc=acc,
            name=self._name,
            num_states=self._num_states,
            initial=self._initial_states,
            predicates=self._predicates,
            aliases=self._aliases,
        )

    @v_args(inline=True)
    def format_version(self, version: str) -> None:
        if version != "v1":
            raise IncorrectVersionError

    def num_states(self, value: int) -> None:
        assert isinstance(value, int) and value > 0
        if self._num_states is not None:
            raise DuplicateHeaderError("States")
        else:
            self._num_states = value

    @v_args(inline=True)
    def initial_states(self, children: list[int]) -> None:
        assert isinstance(children, list) and len(children) > 0
        assert all(map(lambda s: isinstance(s, int) and s >= 0, children))
        self._initial_states.append(children)

    @v_args(inline=True)
    def predicates(self, num_predicates: int, *predicates: str) -> None:
        if len(self._predicates) > 0:
            raise DuplicateHeaderError("AP")
        assert len(predicates) == num_predicates, "Number of predicates does not match defined predicates"
        self._predicates = list(predicates)

    @v_args(inline=True)
    def alias(self, name: str, target: LabelExpr) -> None:
        if name in self._aliases:
            raise DuplicateAliasError(name)
        self._aliases[name] = target

    @v_args(inline=True)
    def automaton_acc(self, num_sets: int, condition: AccExpr) -> None:
        if hasattr(self, "_acc") or hasattr(self, "_num_accept_sets"):
            raise DuplicateHeaderError("Acceptance")
        self._num_accept_sets = num_sets
        self._acc = condition

    @v_args(inline=True)
    def acc_name(self, name: str, props: list[bool | int | str] | None) -> None:
        if self._acc_name is not None:
            raise DuplicateHeaderError("acc-name")
        self._acc_name = (name, props)

    def name(self, name: str) -> None:
        if self._name is not None:
            raise DuplicateHeaderError("name")
        self._name = name

    @v_args(inline=True)
    def body(self, *transitions: tuple[State, list[Transition]]) -> dict[State, list[Transition]]:
        if len(transitions) == 0:
            return dict()
        return dict(transitions)

    @v_args(inline=True)
    def transitions(self, state: State, *edges: Transition) -> tuple[State, list[Transition]]:
        if len(edges) == 0:
            ret_edges = []
        else:
            ret_edges = list(edges)
        return (state, ret_edges)

    @v_args(inline=True)
    def state_name(
        self,
        label: LabelExpr | None,
        idx: int,
        description: str | None,
        acc_sig: list[int] | None,
    ) -> State:
        return State(idx, label, acc_sig, description)

    @v_args(inline=True)
    def edge(
        self,
        label: LabelExpr | None,
        state_conj: list[int],
        acc_sig: list[int] | None,
    ) -> Transition:
        return Transition(state_conj, label, acc_sig)

    @v_args(inline=True)
    def acc_sig(self, *sets: int) -> list[int]:
        return list(sets)

    @v_args(inline=True)
    def acc_props(self, *props: bool | int | str) -> list[bool | int | str]:
        return list(props)

    @v_args(inline=True)
    def label_atom(self, val: bool | int | str) -> guard.LabelExpr:
        match val:
            case bool(v):
                return guard.Literal(v)
            case int(v):
                return guard.Predicate(v)
            case str(alias):
                if alias not in self._aliases:
                    raise UndefinedAliasError(alias)
                return self._aliases[alias]

    @v_args(inline=True)
    def label_not(self, arg: LabelExpr) -> LabelExpr:
        return guard.Not(arg)

    @v_args(inline=True)
    def label_and(self, lhs: LabelExpr, rhs: LabelExpr) -> LabelExpr:
        return lhs & rhs

    @v_args(inline=True)
    def label_or(self, lhs: LabelExpr, rhs: LabelExpr) -> LabelExpr:
        return lhs | rhs

    @v_args(inline=True)
    def state_conj(self, *children: int) -> list[int]:
        assert len(children) > 0
        return list(children)

    def acc_bool(self, arg: bool) -> AccExpr:
        assert isinstance(arg, bool)
        return Literal(arg)

    @v_args(inline=True)
    def acc_set(self, invert: str | None, label: int) -> tuple[bool, int]:
        assert isinstance(invert, str | None)
        assert isinstance(label, int)
        return (invert is not None and invert == "!", label)

    @v_args(inline=True)
    def acc_fin(self, acc_set: tuple[bool, int]) -> Fin:
        invert, arg_set = acc_set
        return Fin(arg_set, invert)

    @v_args(inline=True)
    def acc_inf(self, acc_set: tuple[bool, int]) -> Inf:
        invert, arg_set = acc_set
        return Inf(arg_set, invert)

    @v_args(inline=True)
    def acc_and(self, lhs: AccExpr, rhs: AccExpr) -> AccExpr:
        return lhs & rhs

    @v_args(inline=True)
    def acc_or(self, lhs: AccExpr, rhs: AccExpr) -> AccExpr:
        return lhs | rhs

    def INT(self, tok: Token) -> int:  # noqa: N802
        return int(tok)

    def ESCAPED_STRING(self, s: Token) -> str:  # noqa: N802
        # Remove quotation marks
        return s[1:-1]

    def BOOLEAN(self, s: Token) -> bool:  # noqa: N802
        val = str(s)
        assert val in ["t", "f"]
        return val == "t"

    def IDENTIFIER(self, s: Token) -> str:  # noqa: N802
        return str(s)

    def ANAME(self, s: Token) -> str:  # noqa: N802
        return str(s)

    def HEADERNAME(self, s: Token) -> str:  # noqa: N802
        # remove the : at the end
        return str(s[:-1])


HOA_GRAMMAR_FILE = Path(__file__).parent / "hoa.lark"
with open(HOA_GRAMMAR_FILE, "r") as grammar:
    HOA_GRAMMAR = Lark(
        grammar,
        start="automaton",
        strict=True,
        maybe_placeholders=True,
    )


def parse(expr: str) -> ParseTree:
    tree = HOA_GRAMMAR.parse(expr)
    return AstTransformer().transform(tree)
