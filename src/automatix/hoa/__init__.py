from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from lark import Lark, ParseTree, Token, Transformer, v_args

import automatix.hoa.acceptance_cond as acc_expr
import automatix.hoa.label_expr as guard
from automatix.hoa.label_expr import LabelExpr


class HoaSyntaxError(Exception):
    def __init__(self, label: str = "") -> None:
        self.label: str = label

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
    num_accepting_sets: int
    acc_cond: acc_expr.AcceptanceFormula
    acc_name: tuple[str, list[bool | int | str] | None] | None = None
    name: str | None = None
    num_states: int | None = None
    initial: list[int] | None = None
    predicates: list[str] = field(default_factory=list)
    aliases: dict[str, LabelExpr] = field(default_factory=dict)
    properties: list[str] = field(default_factory=list)


@dataclass(frozen=True, eq=True)
class State:
    idx: int
    label: LabelExpr | None = None
    acc_set: list[int] | None = None
    description: str | None = None


@dataclass(frozen=True, eq=True)
class Transition:
    dst: list[int]
    label: LabelExpr | None = None
    acc_set: list[int] | None = None


@dataclass(frozen=True)
class ParsedAutomaton:
    header: Header
    body: dict[State, list[Transition]]


class AstTransformer(Transformer):
    def __init__(self, visit_tokens: bool = True) -> None:
        super().__init__(visit_tokens)
        self._aliases: dict[str, LabelExpr] = dict()
        self._num_states: int | None = None
        self._initial_states: list[int] = []
        self._predicates: list[str] = []

        self._num_accept_sets: int
        self._acc: acc_expr.AcceptanceFormula

        self._acc_name: tuple[str, list[bool | int | str] | None] | None = None
        self._name: str | None = None

    @v_args(inline=True)
    def automaton(self, header: Header, body: dict[State, list[Transition]]) -> ParsedAutomaton:
        aut = ParsedAutomaton(header, body)

        return aut

    def header(self, _: list[Any]) -> Header:
        if not hasattr(self, "_acc") or not hasattr(self, "_num_accept_sets"):
            raise MissingHeaderError("Acceptance")
        return Header(
            name=self._name,
            num_states=self._num_states,
            initial=self._initial_states,
            predicates=self._predicates,
            aliases=self._aliases,
            num_accepting_sets=self._num_accept_sets,
            acc_cond=self._acc,
            acc_name=self._acc_name,
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
        if len(self._initial_states) > 0:
            raise DuplicateHeaderError("Start")
        assert isinstance(children, list) and len(children) > 0
        assert all(map(lambda s: isinstance(s, int) and s >= 0, children))
        self._initial_states = children

    @v_args(inline=True)
    def predicates(self, num_predicates: int, *predicates: str) -> None:
        if len(self._predicates) > 0:
            raise DuplicateHeaderError("Start")
        assert len(predicates) == num_predicates, "Number of predicates does not match defined predicates"
        self._predicates = list(predicates)

    @v_args(inline=True)
    def alias(self, name: str, target: LabelExpr) -> None:
        if name in self._aliases:
            raise DuplicateAliasError(name)
        self._aliases[name] = target

    @v_args(inline=True)
    def automaton_acc(self, num_sets: int, condition: acc_expr.AcceptanceFormula) -> None:
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
        if transitions is None or len(transitions) == 0:
            return dict()
        return dict(transitions)

    @v_args(inline=True)
    def transitions(self, state: State, *edges: Transition) -> tuple[State, list[Transition]]:
        if edges is None or len(edges) == 0:
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
    def acc_sig(self, *sets: int) -> list[int] | None:
        if sets is None:
            return None
        return list(sets)

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
            case _:
                raise TypeError(f"Unexpected type of LabelExpr atom: `{type(val)}`")

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
    def state_conj(self, children: int | list[int]) -> list[int]:
        if isinstance(children, int):
            return [children]
        return children

    def acc_bool(self, arg: bool) -> acc_expr.AcceptanceFormula:
        assert isinstance(arg, bool)
        return acc_expr.Literal(arg)

    @v_args(inline=True)
    def acc_set(self, invert: str | None, label: int) -> tuple[bool, int]:
        assert isinstance(invert, Union[str, None])
        assert isinstance(label, int)
        return (invert is not None and invert == "!", label)

    @v_args(inline=True)
    def acc_fin(self, acc_set: tuple[bool, int]) -> acc_expr.Fin:
        invert, arg_set = acc_set
        return acc_expr.Fin(invert, arg_set)

    @v_args(inline=True)
    def acc_inf(self, acc_set: tuple[bool, int]) -> acc_expr.Inf:
        invert, arg_set = acc_set
        return acc_expr.Inf(invert, arg_set)

    @v_args(inline=True)
    def acc_and(self, lhs: acc_expr.AcceptanceFormula, rhs: acc_expr.AcceptanceFormula) -> acc_expr.AcceptanceFormula:
        return lhs & rhs

    @v_args(inline=True)
    def acc_or(self, lhs: acc_expr.AcceptanceFormula, rhs: acc_expr.AcceptanceFormula) -> acc_expr.AcceptanceFormula:
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
