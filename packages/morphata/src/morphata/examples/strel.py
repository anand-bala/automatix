"""Alternating finite automaton for Spatio-Temporal Reach Escape Logic (STREL)."""

from __future__ import annotations

import functools
import math
import typing
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING

import attrs
import logic_asts as logic
import logic_asts.strel as strel
import networkx as nx
from logic_asts.base import BaseExpr as PosBoolExpr
from logic_asts.strel import STRELExpr
from typing_extensions import overload, override

import morphata
from morphata.acceptance import Finite

type Q = tuple[int, int]
type State = PosBoolExpr[Q]
type Location = int

if TYPE_CHECKING:
    type Input = nx.Graph[int]
else:
    type Input = nx.Graph


class STRELDomain(morphata.Domain[Q, Input]):
    """Domain for STREL automata with symbolic states and graph inputs."""

    @property
    def states(self) -> Iterable[Q] | None:
        """States are symbolic (not enumerable)."""
        return None

    @property
    def symbols(self) -> Iterable[Input] | None:
        """Symbols are graphs (not enumerable)."""
        return None


def strel_to_automata[AP: Hashable](
    expr: STRELExpr[AP],
    dist_attr: str,
    label_fn: Callable[[Input, Location, AP], bool],
    num_locations: int = 0,
    ego_location: int | Sequence[int] | None = None,
) -> morphata.Automaton[Q, Input]:
    """Convert a STREL expression into an automaton"""

    transition_fn = STRELAutomaton(expr, dist_attr, label_fn, num_locations, ego_location)

    return morphata.Automaton[Q, Input](
        domain=STRELDomain(),
        initial=transition_fn.initial_state,
        delta=transition_fn,
        acceptance=transition_fn.acceptance_condition,
    )


def _as_strel[AP: Hashable](expr: logic.Expr) -> STRELExpr[AP]:
    """Cast logical expression to STREL expression."""
    return typing.cast(STRELExpr[AP], expr)


def _as_poly[AP: Hashable](expr: logic.Expr) -> State:
    """Cast logical expression to polynomial state."""
    return typing.cast(State, expr)


class _NodeMap[AP: Hashable](MutableMapping[STRELExpr[AP], int]):
    """Bidirectional mapping between STREL expressions and integer indices."""

    def __init__(self) -> None:
        super().__init__()
        self.forward: dict[STRELExpr[AP], int] = dict()
        self.backward: dict[int, STRELExpr[AP]] = dict()
        self._last_idx: int = -1  # The largest index in backward

    def add_node(self, key: STRELExpr[AP]) -> int:
        """Add node to the map, and return the index. If the node already exists, returns existing key"""
        idx = self.forward.setdefault(key, self._last_idx + 1)
        self.backward[idx] = key
        self._last_idx = max(self._last_idx, idx)
        return idx

    @overload
    def get_expr[_T](self, key: int, default: None = None) -> None | STRELExpr[AP]: ...
    @overload
    def get_expr[_T](self, key: int, default: _T) -> STRELExpr[AP] | _T: ...

    def get_expr[_T](self, key: int, default: None | _T = None) -> None | STRELExpr[AP] | _T:
        return self.backward.get(key, default)

    @override
    def __getitem__(self, key: STRELExpr[AP]) -> int:
        return self.forward[key]

    @override
    def __setitem__(self, key: STRELExpr[AP], value: int) -> None:
        self.backward[value] = key
        self.forward[key] = value
        self._last_idx = max(value, self._last_idx)

    @override
    def __delitem__(self, key: STRELExpr[AP]) -> None:
        back_idx = self.forward[key]
        if back_idx == self._last_idx:
            self._last_idx = back_idx - 1
        del self.backward[back_idx]
        del self.forward[key]

    @override
    def __iter__(self) -> Iterator[STRELExpr[AP]]:
        return iter(self.forward)

    @override
    def __len__(self) -> int:
        assert len(self.backward) == len(self.forward)
        return len(self.forward)


class STRELAutomaton[AP: Hashable](morphata.AlternatingTransitions[Q, Input]):
    """An alternating finite automaton for STREL with acceptance semantics.

    This is an example implementation showing how to build alternating automata
    for spatio-temporal specifications.
    """

    def __init__(
        self,
        expr: STRELExpr[AP],
        dist_attr: str,
        label_fn: Callable[[Input, Location, AP], bool],
        num_locations: int = 0,
        ego_location: int | Sequence[int] | None = None,
    ) -> None:
        """Initialize STREL automaton from a spatio-temporal expression."""
        self._expr = expr
        self._expr_map = _NodeMap[AP]()
        self._expr_map.add_node(self._expr)

        self.label_fn = label_fn
        self.dist_attr = dist_attr

        self._num_locs = num_locations
        self._ego_locs: frozenset[int] = frozenset()
        self.set_ego_locations(ego_location)

        self._rewriter = _SpLTLRewrite[AP]()

    def set_ego_locations(self, value: int | Sequence[int] | None) -> None:
        """Set the ego locations for spatial constraints."""
        if value is None:
            # All locations conjunction
            self._ego_locs = frozenset(range(self._num_locs))
        elif isinstance(value, int):
            self._ego_locs = frozenset({value})
        elif isinstance(value, Sequence):
            self._ego_locs = frozenset(value)
        else:
            raise TypeError(f"Incorrect type for ego_location: `{type(value)}`")

    def get_ego_locations(self) -> frozenset[int]:
        """Get the ego locations for spatial constraints."""
        return self._ego_locs

    ego_locations = property(get_ego_locations, set_ego_locations)
    """Property for getting/setting ego locations."""

    @property
    def initial_state(self) -> State:
        """The initial state is the conjunction of the root expression node at all ego locations"""
        i = 0
        assert self._expr_map.get_expr(i) == self._expr, "0 should always be the root node for STRELAutomaton"
        variables = tuple(strel.Variable((i, loc)) for loc in self.ego_locations)
        if len(variables) == 1:
            return variables[0]
        return strel.And(variables)

    @property
    def acceptance_condition(self) -> Finite[Q]:
        """Acceptance condition based on STREL semantics."""
        return Finite(
            accepting=frozenset(
                {
                    (i, loc)
                    for loc in self.ego_locations
                    for expr, i in self._expr_map.items()
                    if (
                        isinstance(expr, strel.Not)
                        and isinstance(expr.arg, (strel.Until, strel.Eventually))
                        and (expr.arg.interval is None or expr.arg.interval.is_untimed())
                    )
                    or expr == self._expr
                }
            )
        )

    def delta(self, input_symbol: Input, expr: STRELExpr[AP], loc: Location) -> State:
        """Get the successor state representation for this specific state `Q`"""

        _recurse = functools.partial(self.delta, input_symbol)

        def var(expr: logic.Expr, loc: Location) -> logic.Variable[Q]:
            expr_idx = self._expr_map.add_node(_as_strel(expr))
            return logic.Variable((expr_idx, loc))

        match expr:
            case strel.Literal(value):
                return logic.Literal(value)
            case strel.Variable(name):
                return logic.Literal(self.label_fn(input_symbol, loc, name))
            case strel.Not(arg):
                return _as_poly(~_recurse(_as_strel(arg), loc))
            case strel.And(args):
                return functools.reduce(lambda a, b: a & b, map(lambda a: _recurse(_as_strel(a), loc), args))
            case strel.Or(args):
                return functools.reduce(lambda a, b: a | b, map(lambda a: _recurse(_as_strel(a), loc), args))
            case strel.Everywhere():
                return _recurse(self._rewriter(expr), loc)
            case strel.Somewhere():
                return _recurse(self._rewriter(expr), loc)
            case strel.Reach():
                return self._expand_reach(input_symbol, expr, loc)
            case strel.Next(arg, steps):
                if steps is not None and steps > 1:
                    # If steps > 1, we return the variable for X[steps - 1] arg
                    return var(strel.Next(arg, steps - 1), loc)
                else:
                    assert steps is None or steps == 1
                    # Otherwise, return the variable for arg
                    return var(_as_strel(arg), loc)
            case strel.Always(arg, interval):
                if interval is None or interval.is_untimed():
                    # Expand untimed always to: arg & X G arg
                    return _as_poly(_recurse(_as_strel(arg), loc) & var(expr, loc))
                else:
                    # rewrite the expression
                    return _recurse(self._rewriter(expr), loc)
            case strel.Eventually(arg, interval):
                # dual of Always
                if interval is None or interval.is_untimed():
                    # Expand untimed always to: arg | X F arg
                    return _as_poly(_recurse(_as_strel(arg), loc) | var(expr, loc))
                else:
                    # rewrite the expression
                    return _recurse(self._rewriter(expr), loc)
            case strel.Until(lhs, rhs, interval):
                if interval is None or interval.is_untimed():
                    # Expand to: rhs | (lhs & X expr )
                    lhs = _as_strel(lhs)
                    rhs = _as_strel(rhs)
                    return _as_poly(_recurse(rhs, loc) | (_recurse(lhs, loc) & var(expr, loc)))
                else:
                    # Rewrite and expand
                    return _recurse(self._rewriter(expr), loc)
            case _:
                raise TypeError(f"Unknown expression type {type(expr)}")
        raise TypeError(f"Unknown expression type {type(expr)}")

    def __call__(self, state: Q, symbol: Input) -> State:
        expr_idx, loc = state
        expr = self._expr_map.get_expr(expr_idx)
        assert expr is not None
        return self.delta(symbol, expr, loc)

    def _expand_reach(self, input: Input, phi: strel.Reach, loc: Location) -> State:
        d1 = phi.interval.start or 0.0
        d2 = phi.interval.end or math.inf
        # use a modified version of networkx's all_simple_paths algorithm to generate all simple paths
        # constrained by the distance intervals.
        # Then, make the symbolic expressions for each path, with the terminal one being for the rhs
        lhs = typing.cast(STRELExpr[AP], phi.lhs)
        rhs = typing.cast(STRELExpr[AP], phi.rhs)
        expr: State = logic.Literal(False)
        for edge_path in _all_bounded_simple_paths(input, loc, d1, d2, self.dist_attr):
            path = [loc] + [e[1] for e in edge_path]
            # print(f"{path=}")
            # Path expr checks if last node satisfies rhs and all others satisfy lhs
            path_expr = self.delta(input, rhs, path[-1])
            for l_p in reversed(path[:-1]):  # pyrefly: ignore
                path_expr &= self.delta(input, lhs, l_p)  # pyrefly: ignore
            expr |= path_expr  # type: ignore[assignment]
            # Break early if TOP/True
            if expr == logic.Literal(True):
                return expr
        return expr

    def _expand_escape(self, input: Input, phi: strel.Escape, loc: Location) -> State:
        d1 = phi.interval.start or 0.0
        d2 = phi.interval.end or math.inf

        # get a list of target locations that meet the distance constraint
        shortest_lengths: Mapping[Location, int] = nx.shortest_path_length(input, source=loc, weight=None)  # type: ignore
        assert isinstance(shortest_lengths, Mapping)
        targets = {d for d, dist in shortest_lengths.items() if d1 <= dist <= d2}
        # Make the symbolic expressions for each path, with the terminal one being for the rhs
        arg = typing.cast(STRELExpr[AP], phi.arg)
        expr: State
        for path in nx.all_simple_paths(
            input,  # ty:ignore[invalid-argument-type]
            source=loc,  # ty:ignore[invalid-argument-type]
            target=targets,  # ty:ignore[invalid-argument-type]
        ):  # pyrefly: ignore
            # print(f"{path=}")
            # Path expr checks if all locations satisfy arg
            expr = logic.Literal(False)
            for lp in path:
                expr = typing.cast(State, expr & self.delta(input, arg, lp))
            # Break early if TOP/True
            if expr == logic.Literal(True):
                return expr
        return logic.Literal(False)


@attrs.define
class _SpLTLRewrite[AP: Hashable]:
    """Rewrites STREL expressions using automaton expansion rules."""

    _cache: dict[int, STRELExpr[AP]] = attrs.field(factory=dict)
    """Maintains a cache of rewritten expressions"""

    def __call__(self, expr: STRELExpr[AP]) -> STRELExpr[AP]:
        """Rewrite a STREL expression to automaton form."""
        expr = typing.cast(STRELExpr[AP], expr.expand().to_nnf())

        expr_idx = hash(expr)
        if expr_idx not in self._cache:
            self._cache[expr_idx] = self._rewrite(expr)
        return self._cache[expr_idx]

    def _rewrite(self, expr: STRELExpr[AP]) -> STRELExpr[AP]:
        # Don't recursively rewrite. We will do it lazily in the transition dynamics.

        match expr:
            case strel.Literal() | strel.Variable() | strel.Not() | strel.And() | strel.Or():
                return expr
            case strel.Everywhere(arg, interval):
                return typing.cast(STRELExpr[AP], ~strel.Somewhere(~arg, interval))
            case strel.Somewhere(arg, interval):
                return strel.Reach(strel.Literal(True), arg, interval)
            case strel.Escape():
                return attrs.evolve(expr, arg=(expr.arg))
            case strel.Reach():
                return attrs.evolve(expr, lhs=(expr.lhs), rhs=(expr.rhs))
            case strel.Next(arg, steps):
                return self._expand_next(_as_strel(arg), steps)
            case strel.Always(arg, interval):
                # G[a, b] phi = X X ... X (phi & X (phi & X( ... & X f)))
                #              ^^^^^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^
                #               a times                 b-a times
                #            = X[a] (phi & X (phi & X( ... & X f)))
                #                          ^^^^^^^^^^^^^^^^^^^^^^^
                #                                  b-a times
                return self._expand_always(_as_strel(arg), interval)
            case strel.Eventually(arg, interval):
                # F[a, b] phi = X X ... X (phi | X (phi | X( ... | X f)))
                #              ^^^^^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^
                #               a times                 b-a times
                #            = X[a] (phi | X (phi | X( ... | X f)))
                #                          ^^^^^^^^^^^^^^^^^^^^^^^
                #                                  b-a times
                return self._expand_eventually(_as_strel(arg), interval)
            case strel.Until(lhs, rhs, interval):
                return self._expand_until(_as_strel(lhs), _as_strel(rhs), interval)
            case _:
                raise TypeError(f"Unknown expression type {type(expr)}")

    def _expand_next(self, arg: STRELExpr[AP], steps: int | None = None) -> STRELExpr[AP]:
        steps = steps if steps is not None else 1
        assert steps >= 0
        # If steps is not None, we expand the nesting
        ret: STRELExpr[AP] = arg
        for _ in range(steps):
            ret = strel.Next(ret)
        return ret

    def _expand_until(self, lhs: STRELExpr[AP], rhs: STRELExpr[AP], interval: strel.TimeInterval) -> STRELExpr[AP]:
        # lhs U[t1, t2] rhs = (F[t1,t2] rhs) & (lhs U[t1,] rhs)
        # lhs U[t1,  ] rhs = ~F[0,t1] ~(lhs U rhs)

        start, end = interval.start, interval.end

        unbounded_unt = strel.Until(lhs, rhs)
        self._cache[hash(unbounded_unt)] = unbounded_unt

        match (start, end):
            case (None, None):
                # phi = lhs U rhs
                return strel.Until(lhs, rhs)
            case (int(t1), None):
                # phi = lhs U[t1,] rhs = G[0,t1] (lhs U rhs)
                return self._expand_always(unbounded_unt, interval)
            case (int(t1), int()):
                # phi = lhs U[t1,t2] rhs = (F[t1,t2] rhs) & (lhs U[t1,] rhs) = (F[t1,t2] rhs) & (G[0,t1] (lhs U rhs))
                # phi1 = F[t1,t2] rhs
                # phi2 = G[0,t1] (lhs U rhs)
                phi2 = self._expand_always(unbounded_unt, strel.TimeInterval(0, t1))
                phi1 = self._expand_eventually(rhs, interval)
                return _as_strel(phi1 & phi2)
        raise RuntimeError("unreachable")

    def _expand_always(self, arg: STRELExpr[AP], interval: strel.TimeInterval) -> STRELExpr[AP]:
        # G[a, b] phi = X X ... X (phi & X (phi & X( ... & X f)))
        #              ^^^^^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^
        #               a times                 b-a times
        #            = X[a] (phi & X (phi & X( ... & X f)))
        #                          ^^^^^^^^^^^^^^^^^^^^^^^
        #                                  b-a times

        start, end = interval.start, interval.end

        match (start, end):
            case (None, None):
                # G phi = ~F ~phi
                return _as_strel(~strel.Eventually(~arg))
            case (int(t1), None):
                # phi = G[t1,] arg = X[t1] G arg
                assert t1 > 0
                # if steps=0, it should return unbounded eventually as is
                return self._expand_next(strel.Always(arg), t1)
            case (int(t1), int(t2)):
                # Iteratively expand these
                # 1. phi1 = G[t1, t2] arg = X[t1] G[0, t2 - t1] arg
                # 2. phi2 = G[0, b] arg = arg & X G[0, b-1] arg
                assert t1 >= 0 and t2 > 0
                # First build the (2) for the duration (inclusive)
                duration = t2 - t1
                phi2: STRELExpr[AP] = arg  # b = 0
                for _ in range(0, duration):
                    phi2 = _as_strel(arg & strel.Next(phi2))

                # now expand phi1
                # if steps=0, it should return phi2 as is
                return self._expand_next(phi2, steps=t1)
        raise RuntimeError("unreachable")

    def _expand_eventually(self, arg: STRELExpr[AP], interval: strel.TimeInterval) -> STRELExpr[AP]:
        # F[a, b] phi = X X ... X (phi | X (phi | X( ... | X f)))
        #              ^^^^^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^
        #               a times                 b-a times
        #            = X[a] (phi | X (phi | X( ... | X f)))
        #                          ^^^^^^^^^^^^^^^^^^^^^^^
        #                                  b-a times

        start, end = interval.start, interval.end

        match (start, end):
            case (None, None):
                # Return F arg as is
                return strel.Eventually(arg)
            case (int(t1), None):
                # phi = F[t1,] arg = X[t1] F arg
                assert t1 > 0
                # if steps=0, it should return unbounded eventually as is
                return self._expand_next(strel.Eventually(arg), t1)
            case (int(t1), int(t2)):
                # Iteratively expand these
                # 1. phi1 = F[t1, t2] arg = X[t1] F[0, t2 - t1] arg
                # 2. phi2 = F[0, b] arg = arg | X F[0, b-1] arg
                assert t1 >= 0 and t2 > 0
                # First build the (2) for the duration (inclusive)
                duration = t2 - t1
                phi2: STRELExpr[AP] = arg  # b = 0
                for _ in range(0, duration):
                    phi2 = _as_strel(arg | strel.Next(phi2))

                # now expand phi1
                # if steps=0, it should return phi2 as is
                return self._expand_next(phi2, steps=t1)
        raise RuntimeError("unreachable")


def _all_bounded_simple_paths(
    graph: Input, loc: Location, d1: float, d2: float, dist_attr: str
) -> Iterator[list[tuple[Location, Location, float]]]:
    """Return all edge paths for reachable nodes. The path lengths are always between `d1` and `d2` (inclusive)"""

    # This adapts networkx's all_simple_edge_paths code.
    #
    # Citations:
    #
    # 1. https://xlinux.nist.gov/dads/HTML/allSimplePaths.html
    # 2. https://networkx.org/documentation/stable/_modules/networkx/algorithms/simple_paths.html#all_simple_paths
    def get_edges(node: Location) -> Iterable[tuple[Location, Location, float]]:
        return graph.edges(node, data=dist_attr, default=1.0)

    # The current_path is a dictionary that maps nodes in the path to the edge that was
    # used to enter that node (instead of a list of edges) because we want both a fast
    # membership test for nodes in the path and the preservation of insertion order.
    # Edit: It also keeps track of the cumulative distance of the path.
    current_path: dict[Location | None, None | tuple[None | Location, Location, float]] = {None: None}

    # We simulate recursion with a stack, keeping the current path being explored
    # and the outgoing edge iterators at each point in the stack.
    # To avoid unnecessary checks, the loop is structured in a way such that a path
    # is considered for yielding only after a new node/edge is added.
    # We bootstrap the search by adding a dummy iterator to the stack that only yields
    # a dummy edge to source (so that the trivial path has a chance of being included).
    stack: deque[Iterator[tuple[None | Location, Location, float]]] = deque([iter([(None, loc, 0.0)])])

    # Note that the target is every other reachable node in the graph.
    targets = graph.nodes

    while len(stack) > 0:
        # 1. Try to extend the current path.
        #
        # Checks if node already visited.
        next_edge = next((e for e in stack[-1] if e[1] not in current_path), None)
        if next_edge is None:
            # All edges of the last node in the current path have been explored.
            stack.pop()
            current_path.popitem()
            continue
        previous_node, next_node, next_dist = next_edge

        if previous_node is not None:
            assert current_path[previous_node] is not None
            prev_path_len = (current_path[previous_node] or (None, None, 0.0))[2]
            new_path_len = prev_path_len + next_dist
        else:
            new_path_len = 0.0

        # 2. Check if we've reached a target (if adding the next_edge puts us in the distance range).
        if d1 <= new_path_len <= d2:
            # Yield the current path, removing the initial dummy edges [None, (None, source)]
            ret: list[tuple[Location, Location, float]] = (list(current_path.values()) + [next_edge])[2:]  # type: ignore  # ty:ignore[unused-ignore-comment]
            yield ret

        # 3. Only expand the search through the next node if it makes sense.
        #
        # Check if the current cumulative distance (using previous_node) + new_dist is in the range.
        # Also check if all targets are explored.
        if new_path_len <= d2 and (targets - current_path.keys() - {next_node}):
            # Change next_edge to contain the cumulative distance
            update_edge = next_edge[:-1] + (new_path_len,)
            current_path[next_node] = update_edge  # pyrefly: ignore
            stack.append(iter(get_edges(next_node)))
            pass
