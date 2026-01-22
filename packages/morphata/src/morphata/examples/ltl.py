"""Implement a simple LTL to alternating automaton procedure.

The translation supports
- LTL to Alternating Büchi Word Automata
- LTLf to Alternating Finite Automata
"""

from __future__ import annotations

import dataclasses
import functools
import typing as ty
from collections import deque
from collections.abc import Collection, Hashable, Iterable, Iterator, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

import logic_asts as logic
import logic_asts.ltl as ltl
from typing_extensions import override

import morphata
from morphata.acceptance import Buchi, Finite

AP = ty.TypeVar("AP", bound=Hashable)
type Input[AP] = AbstractSet[AP]
type BoolExpr[Sym: Hashable] = logic.base.BaseExpr[Sym]
type LTLExpr[AP] = ltl.LTLExpr[AP]


def ltl_to_automaton(formula: LTLExpr[AP], *, finite: bool = False) -> morphata.Automaton[int, Input[AP]]:
    """Convert an LTL expression into an alternating automaton."""

    formula = ty.cast(LTLExpr[AP], formula.expand())
    atomic_predicates: frozenset[AP] = frozenset(
        ty.cast(AP, e.name) for e in formula.iter_subtree() if isinstance(e, ltl.Variable)
    )

    transitions: dict[int, Mapping[Input[AP], BoolExpr[int]]] = dict()
    initial_node = formula
    final_states: set[int] = set()

    # Now, we want to remap to integer nodes such that:
    # 1. 0 is the initial node
    # 2. the rest of the nodes are contiguous integers
    # 3. Unreachable nodes not there
    mappings: dict[logic.Variable[LTLExpr[AP]] | logic.Not, int] = dict()

    expr_remap_cache: dict[BoolExpr[LTLExpr[AP]], BoolExpr[int]] = dict()

    def _remap_bool_expr(node: BoolExpr[LTLExpr[AP]]) -> BoolExpr[int]:
        if node in expr_remap_cache:
            return expr_remap_cache[node]
        match node:
            case logic.Literal():
                return node
            case logic.Variable():
                return logic.Variable(mappings.setdefault(node, len(mappings)))
            case logic.Not(arg):
                # Assumes that the input node was coverted to NNF
                assert isinstance(arg, logic.Variable)
                # Make a new variable for Not(var)
                return logic.Variable(mappings.setdefault(node, len(mappings)))
            case logic.And(args) | logic.Or(args) as nary_node:
                return dataclasses.replace(
                    nary_node, args=tuple(_remap_bool_expr(ty.cast(BoolExpr[LTLExpr[AP]], arg)) for arg in args)
                )
            case _:
                raise RuntimeError("unreachable")

    # Queue containing list of reachable nodes that need to be remapped.
    # The order of insertion in the queue is the reachability order
    queue: deque[logic.Variable[LTLExpr[AP]] | logic.Not] = deque()
    queue.append(logic.Variable(formula))
    # We will build the remap dict first
    while len(queue) > 0:
        # Pop a node_id
        node = queue.popleft()
        # Remap it
        node_id = mappings.setdefault(node, len(mappings))
        if node_id in transitions:
            continue
        # Check if the node is accepting/final or not.
        if isinstance(node, logic.Variable):
            node_is_final = _is_accepting_node_expr(node.name)
        else:
            assert isinstance(node, logic.Not) and isinstance(node.arg, logic.Variable)
            node_is_final = _is_accepting_node_expr(ty.cast(LTLExpr[AP], node.arg.name))
        if node_is_final:
            final_states.add(node_id)
        # Add the reachable nodes in the next step (Variables/Not(Variable)) if they are not already mapped
        reachable: dict[Input[AP], BoolExpr[int]] = dict()
        sym: Input[AP]
        for sym in _powerset(atomic_predicates):
            # Compute the possible successor polynomial
            successor = _aut_expansion_rule(node, sym)  # pyrefly: ignore[bad-argument-type]
            # We want to add the NNF leaf nodes to the queue, i.e., Not(var) is a leaf node, iff they are not already visited
            queue.extend((e for e in successor.atomic_predicates(assume_nnf=True) if e not in mappings))
            # pyrefly: ignore[bad-argument-type]
            reachable[sym] = _remap_bool_expr(successor)
        transitions[node_id] = reachable

    assert mappings[logic.Variable(initial_node)] == 0
    assert 0 in transitions

    domain = LTLDomain(
        frozenset(transitions.keys()),
        atomic_predicates,
    )
    transition_fn = morphata.AlternatingTransitionRelation(transitions)
    acceptance: Buchi[int] | Finite[int]
    if finite:
        acceptance = Finite(frozenset(final_states))
    else:
        acceptance = Buchi(frozenset(final_states))
    return morphata.Automaton(
        domain=domain, initial=mappings[logic.Variable(initial_node)], delta=transition_fn, acceptance=acceptance
    )


def _is_accepting_node_expr(expr: LTLExpr[AP]) -> bool:
    """
    Nodes that are of the form ~(a U b) are accepting, along with the literal True obviously.
    """
    if isinstance(expr, ltl.Literal):
        return expr.value

    if isinstance(expr, ltl.Not) and isinstance(expr.arg, ltl.Until):
        return True

    # G a = ~ F ~a = ~(T U ~a)
    if isinstance(expr, ltl.Always):
        return True
    return False


def _aut_expansion_rule(expr: LTLExpr[AP], symbol: Input[AP]) -> BoolExpr[LTLExpr[AP]]:
    """Symbolic expansion of the automaton state corresponding to a given expression"""

    def _recurse(_state: LTLExpr[AP]) -> BoolExpr[LTLExpr[AP]]:
        return _aut_expansion_rule(_state, symbol)

    def _as_poly(arg: logic.Expr) -> BoolExpr[LTLExpr[AP]]:
        return ty.cast(BoolExpr[LTLExpr[AP]], arg)

    def var(arg: LTLExpr[AP]) -> logic.Variable[LTLExpr[AP]]:
        return logic.Variable(arg)

    match expr:
        case ltl.Literal(value):
            return logic.Literal(value)
        case ltl.Variable(name):
            return logic.Literal(name in symbol)
        case ltl.Not(arg):
            return logic.Not(_recurse(_as_ltl(arg)))
        case ltl.And(args):
            return functools.reduce(lambda a, b: a & b, map(lambda a: _recurse(_as_ltl(a)), args))
        case ltl.Or(args):
            return functools.reduce(lambda a, b: a | b, map(lambda a: _recurse(_as_ltl(a)), args))
        case ltl.Next(arg, steps):
            assert steps is None or steps == 1, (
                "should expand all intervals using `expand` before using automaton expansion rules"
            )
            return var(_as_ltl(arg))
        case ltl.Always(arg, interval):
            assert interval is None or interval.is_untimed(), (
                "should expand all intervals using `expand` before using automaton expansion rules"
            )
            # Expand untimed always to: arg & X G arg -> arg & X expr
            return _as_poly(_recurse(_as_ltl(arg)) & var(expr))
        case ltl.Eventually(arg, interval):
            assert interval is None or interval.is_untimed(), (
                "should expand all intervals using `expand` before using automaton expansion rules"
            )
            # Expand untimed always to: arg | X F arg
            return _as_poly(_recurse(_as_ltl(arg)) | var(expr))
        case ltl.Until(lhs, rhs, interval):
            assert interval is None or interval.is_untimed(), (
                "should expand all intervals using `expand` before using automaton expansion rules"
            )
            # Expand to: rhs | (lhs & X expr )
            lhs = _as_ltl(lhs)
            rhs = _as_ltl(rhs)
            return _as_poly(_recurse(rhs) | (_recurse(lhs) & var(expr)))
        case _:
            raise TypeError(f"Unknown expression type {type(expr)}")
    raise TypeError(f"Unknown expression type {type(expr)}")


def _powerset[T](elements: Collection[T]) -> Iterable[AbstractSet[T]]:
    from itertools import chain, combinations

    return chain.from_iterable(map(frozenset, combinations(elements, r)) for r in range(len(elements) + 1))


@dataclass
class LTLDomain(morphata.Domain[int, Input[AP]]):
    _states: AbstractSet[int]
    atomic_predicates: AbstractSet[AP]

    @property
    @override
    def states(self) -> Iterable[int] | None:
        yield from self._states

    @property
    @override
    def symbols(self) -> Iterable[Input[AP]] | None:
        """Return a powerset of the atomic predicats"""
        yield from _powerset(self.atomic_predicates)


def _iter_expr_nnf[Var: Hashable](expr: BoolExpr[Var]) -> Iterator[BoolExpr[Var]]:
    """Iterate over a propositional logic expression after converting it to NNF form. If
    there is a `Not(var)` in the expression, it will not yield `var`.

    Otherwise, it has the same guarantees as `Expr.iter_subtree`
    """
    expr_nnf = expr.to_nnf()
    assert logic.is_propositional_logic(expr_nnf)
    stack: deque[BoolExpr[Var]] = deque([expr_nnf])
    visited: set[BoolExpr[Var]] = set()

    while stack:
        subexpr = stack[-1]
        if isinstance(subexpr, logic.Not):
            # Don't put in the arguments of NOT
            assert isinstance(subexpr.arg, logic.Variable), "Expected to receive NNF expression"
            need_to_visit_children = set()
        else:
            need_to_visit_children = {
                ty.cast(BoolExpr[Var], child)
                for child in subexpr.children()  # We need to visit `child`
                if child not in visited  # if it hasn't already been visited
            }

        if visited.issuperset(need_to_visit_children):
            # subexpr is a leaf (the set is empty) or it's children have been
            # yielded get rid of it from the stack
            stack.pop()
            # Add subexpr to visited
            visited.add(subexpr)
            # post-order return it
            yield subexpr
        else:
            # mid-level node
            # Add relevant children to stack
            stack.extend(need_to_visit_children)
    # Yield the remaining nodes in the stack in reverse order
    yield from reversed(stack)


def _as_ltl[AP: Hashable](expr: logic.Expr) -> LTLExpr[AP]:
    """Cast logical expression to LTL expression."""
    return ty.cast(LTLExpr[AP], expr)
