"""Transform STREL parse tree to an AFA."""

import functools
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Collection, Iterable, Iterator, Mapping, TypeAlias, TypeVar

import networkx as nx

import automatix.logic.strel as strel
from automatix.afa.automaton import AFA, AbstractTransition
from automatix.algebra.abc import AbstractPolynomial, PolynomialManager
from automatix.algebra.polynomials.boolean import BooleanPolyCtx

K = TypeVar("K")

Location: TypeAlias = int

Alph: TypeAlias = "nx.Graph[Location]"
"""Input alphabet is a graph over location vertices, with distance edge weights and vertex labels corresponding to semiring
values for each predicate"""

Q: TypeAlias = tuple[strel.Expr, Location]
"""Each state in the automaton represents a subformula in the specification and an ego location.
"""

Poly: TypeAlias = AbstractPolynomial[K]
Manager: TypeAlias = PolynomialManager[Poly[K], K]
LabellingFn: TypeAlias = Callable[[Alph, Location, str], K]


@dataclass
class Transitions(AbstractTransition[Alph, Q, K]):
    manager: Manager[K]
    label_fn: LabellingFn[K]
    dist_attr: str
    const_mapping: dict[Q, Poly[K]] = field(default_factory=dict)
    var_node_map: dict[str, Q] = field(default_factory=dict)
    aliases: dict[strel.Expr, strel.Expr] = field(default_factory=dict)

    def __call__(self, input: Alph, state: Q) -> Poly[K]:
        if state[0] in self.aliases:
            state = (self.aliases[state[0]], state[1])
        expr, loc = state
        # If expr is temporal, the visitor should add the argument variables to the const_mapping dict
        match expr:
            case strel.Constant(value):
                if value:
                    return self.manager.top
                else:
                    return self.manager.bottom
            case strel.Identifier(name):
                return self.manager.const(self.label_fn(input, loc, name))
            case strel.NotOp(arg):
                return self(input, (arg, loc)).negate()
            case strel.AndOp(lhs, rhs):
                return self(input, (lhs, loc)) * self(input, (rhs, loc))
            case strel.OrOp(lhs, rhs):
                return self(input, (lhs, loc)) + self(input, (rhs, loc))
            case strel.EverywhereOp(interval, arg):
                alias = ~strel.SomewhereOp(interval, ~arg)
                self._add_expr_alias(expr, alias)
                return self(input, (alias, loc))
            case strel.SomewhereOp(interval, arg):
                alias = strel.ReachOp(strel.true, interval, arg)
                self._add_expr_alias(expr, alias)
                return self(input, (alias, loc))
            case strel.EscapeOp():
                return self._expand_escape(input, expr, loc)
            case strel.ReachOp():
                return self._expand_reach(input, expr, loc)
            case strel.NextOp(steps, arg):
                if steps is not None and steps > 1:
                    # If steps > 1, we return the variable for X[steps - 1] arg
                    return self.var((strel.NextOp(steps - 1, arg), loc))
                else:
                    assert steps is None or steps == 1
                    # Otherwise, return the variable for arg
                    return self.var((arg, loc))
            case strel.GloballyOp(interval, arg):
                # Create an alias to ~F[a,b] ~arg
                alias = ~strel.EventuallyOp(interval, ~arg)
                self._add_expr_alias(expr, alias)
                return self(input, (alias, loc))
            case strel.EventuallyOp():
                return self._expand_eventually(input, expr, loc)
            case strel.UntilOp():
                return self._expand_until(input, expr, loc)
            case e:
                raise TypeError(f"Unknown expression type {type(e)}")

    def var(self, state: Q) -> Poly[K]:
        """Get polynomial variable for the given state, adding a new one if needed"""
        try:
            return self.get_var(state)
        except KeyError:
            # add the variable because it isn't already added
            pass
        self.const_mapping[state] = self.manager.declare(_make_q_str(state))
        self.var_node_map[_make_q_str(state)] = state
        return self.const_mapping[state]

    def get_var(self, state: Q) -> Poly[K]:
        if state[0] in self.aliases:
            state = (self.aliases[state[0]], state[1])
        if isinstance(state[0], strel.Constant):
            if state[0].value:
                return self.manager.top
            else:
                return self.manager.bottom
        return self.const_mapping[state]

    def _add_expr_alias(self, phi: strel.Expr, alias: strel.Expr) -> None:
        self.aliases.setdefault(phi, alias)

    def _expand_reach(self, input: Alph, phi: strel.ReachOp, loc: Location) -> Poly[K]:
        d1 = phi.interval.start or 0.0
        d2 = phi.interval.end or math.inf
        # use a modified version of networkx's all_simple_paths algorithm to generate all simple paths
        # constrained by the distance intervals.
        # Then, make the symbolic expressions for each path, with the terminal one being for the rhs
        expr = self.manager.bottom
        for edge_path in _all_bounded_simple_paths(input, loc, d1, d2, self.dist_attr):
            path = [loc] + [e[1] for e in edge_path]
            # print(f"{path=}")
            # Path expr checks if last node satisfies rhs and all others satisfy lhs
            path_expr = self(input, (phi.rhs, path[-1]))
            for l_p in reversed(path[:-1]):
                path_expr *= self(input, (phi.lhs, l_p))
            expr += path_expr
            # Break early if TOP/True
            if expr.is_top():
                return expr
        return expr

    def _expand_escape(self, input: Alph, phi: strel.EscapeOp, loc: Location) -> Poly[K]:
        def delta(expr: strel.Expr, loc: Location) -> Poly[K]:
            return self(input, (expr, loc))

        d1 = phi.interval.start or 0.0
        d2 = phi.interval.end or math.inf

        # get a list of target locations that meet the distance constraint
        shortest_lengths: Mapping[Location, int] = nx.shortest_path_length(input, source=loc, weight=None)
        assert isinstance(shortest_lengths, Mapping)
        targets = {d for d, dist in shortest_lengths.items() if d1 <= dist <= d2}
        # Make the symbolic expressions for each path, with the terminal one being for the rhs
        expr = self.manager.bottom
        for path in nx.all_simple_paths(input, source=loc, target=targets):  # type: ignore
            # print(f"{path=}")
            # Path expr checks if all locations satisfy arg
            init = delta(phi.arg, path[0])
            expr = functools.reduce(lambda acc, loc: acc * delta(phi.arg, loc), path, init)
            # Break early if TOP/True
            if expr.is_top():
                return expr
        return expr

    def _expand_eventually(self, input: Alph, phi: strel.EventuallyOp, loc: Location) -> Poly[K]:
        # F[a,b] phi = X X ... X (phi | X (phi | X( ... | X f)))
        #              ^^^^^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^
        #               a times                 b-a times
        #            = X[a] (phi | X (phi | X( ... | X f)))
        #                          ^^^^^^^^^^^^^^^^^^^^^^^
        #                                  b-a times
        def delta(expr: strel.Expr) -> Poly[K]:
            return self(input, (expr, loc))

        if phi.interval is None:
            start, end = 0, None
        else:
            start, end = phi.interval.start or 0, phi.interval.end

        expr: strel.Expr
        match (start, end):
            case (0, None):
                # phi = F arg
                # Expand as F arg = arg | X F arg
                return delta(phi.arg) + self.var((phi, loc))
            case (0, int(t2)):
                # phi = F[0, t2] arg
                # Expand as F[0, t2] arg = arg | X F[0, t2-1] arg
                next_step: strel.Expr
                if t2 > 1:
                    next_step = strel.EventuallyOp(strel.TimeInterval(0, t2 - 1), phi.arg)
                else:
                    next_step = phi.arg
                return delta(phi.arg) + self.var((next_step, loc))

            case (int(t1), None):
                # phi = F[t1,] arg = X[t1] F arg
                expr = strel.NextOp(t1, strel.EventuallyOp(None, phi.arg))
                self._add_expr_alias(phi, expr)
                return delta(expr)

            case (int(t1), int(t2)):
                # phi = F[t1, t2] arg = X[t1] F[0, t2 - t1] arg
                expr = strel.NextOp(
                    t1,
                    strel.EventuallyOp(
                        strel.TimeInterval(0, t2 - t1),
                        phi.arg,
                    ),
                )
                self._add_expr_alias(phi, expr)
                return delta(expr)
        raise RuntimeError(f"Unknown [start, end] interval {(start, end)}")

    def _expand_until(self, input: Alph, phi: strel.UntilOp, loc: Location) -> Poly[K]:
        # lhs U[t1,t2] rhs = (F[t1,t2] rhs) & (lhs U[t1,] rhs)
        # lhs U[t1,  ] rhs = ~F[0,t1] ~(lhs U rhs)
        def delta(expr: strel.Expr) -> Poly[K]:
            return self(input, (expr, loc))

        if phi.interval is None:
            start, end = 0, None
        else:
            start, end = phi.interval.start or 0, phi.interval.end

        expr: strel.Expr
        match (start, end):
            case (0, None):
                # phi = lhs U rhs
                # Expand as phi = lhs U rhs = rhs | (lhs & X phi)
                return delta(phi.rhs) + (delta(phi.lhs) * self.var((phi, loc)))
            case (t1, None):
                # phi = lhs U[t1,] rhs = ~F[0,t1] ~(lhs U rhs)
                expr = ~strel.EventuallyOp(
                    strel.TimeInterval(0, t1),
                    ~strel.UntilOp(phi.lhs, None, phi.rhs),
                )
                self._add_expr_alias(phi, expr)
            case (t1, int()):
                # phi = lhs U[t1,t2] rhs = (F[t1,t2] rhs) & (lhs U[t1,] rhs)
                expr = strel.EventuallyOp(phi.interval, phi.rhs) & strel.UntilOp(
                    interval=strel.TimeInterval(t1, None),
                    lhs=phi.lhs,
                    rhs=phi.rhs,
                )
        self._add_expr_alias(phi, expr)
        return delta(expr)


def make_bool_automaton(phi: strel.Expr, label_fn: LabellingFn[bool], dist_attr: str = "hop") -> "StrelAutomaton[bool]":
    """Make a Boolean/qualitative Alternating Automaton for STREL monitoring.

    **Parameters:**

    - `phi`: STREL expression
    - `label_fn`: A labelling function that takes as input a graph of signals at each location, a specific location
      (`int`), and the name of the predicate and outputs the value of the predicate.
    - `max_locs`: Maximum number of locations in the automaton.
    - `dist_attr`: The distance attribute over edges in the `nx.Graph`.
    """
    return StrelAutomaton[bool].from_strel_expr(
        phi,
        label_fn,
        # this doesn't work while HKTs are not supported in Python
        BooleanPolyCtx(),  # type: ignore
        dist_attr,
    )


class StrelAutomaton(AFA[Alph, Q, K]):
    """(Weighted) Automaton for STREL"""

    def __init__(
        self,
        initial_expr: strel.Expr,
        transitions: Transitions[K],
    ) -> None:
        super().__init__(transitions)

        self._transitions = transitions
        self.initial_expr = initial_expr
        self.var_node_map = self._transitions.var_node_map
        self._manager = self._transitions.manager

    def _is_accepting(self, expr: strel.Expr) -> bool:
        return (
            isinstance(expr, strel.NotOp)
            and isinstance(expr.arg, (strel.UntilOp, strel.EventuallyOp))
            and (expr.arg.interval is None or expr.arg.interval.is_untimed())
        ) or expr == self.initial_expr

    def initial_at(self, loc: Location) -> Poly[K]:
        """Return the polynomial representation of the initial state"""
        return self._transitions.var((self.initial_expr, loc))

    def final_weight(self, current: Poly[K]) -> K:
        """Return the final weight given the current state polynomial"""
        return current.eval(
            {
                var: (
                    self._manager.top.eval({})
                    if self._is_accepting(self.var_node_map[var][0])
                    else self._manager.bottom.eval({})
                )
                for var in current.support
            }
        )

    @property
    def states(self) -> Collection[Q]:
        return self._transitions.const_mapping.keys()

    def next(self, input: Alph, current: Poly[K]) -> Poly[K]:
        """Get the polynomial after transitions by evaluating the current polynomial with the transition function."""

        transitions = {var: self.transitions(input, self.var_node_map[var]) for var in current.support}
        new_state = current.let(transitions)
        return new_state

    @classmethod
    def from_strel_expr(
        cls,
        phi: strel.Expr,
        label_fn: LabellingFn[K],
        manager: Manager[K],
        dist_attr: str = "hop",
    ) -> "StrelAutomaton":
        """Convert a STREL expression to an AFA with the given alphabet"""

        aut = cls(
            phi,
            Transitions(
                manager=manager,
                label_fn=label_fn,
                dist_attr=dist_attr,
            ),
        )

        return aut

    def check_run(self, ego_location: Location, trace: Iterable[Alph]) -> K:
        """Generate the weight of the trace with respect to the automaton"""
        trace = list(trace)
        state = self.initial_at(ego_location)
        for input in trace:
            state = self.next(input, state)
        ret = self.final_weight(state)
        return ret


def _all_bounded_simple_paths(
    graph: Alph, loc: Location, d1: float, d2: float, dist_attr: str
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
            ret: list[tuple[Location, Location, float]] = (list(current_path.values()) + [next_edge])[2:]  # type: ignore
            yield ret

        # 3. Only expand the search through the next node if it makes sense.
        #
        # Check if the current cumulative distance (using previous_node) + new_dist is in the range.
        # Also check if all targets are explored.
        if new_path_len <= d2 and (targets - current_path.keys() - {next_node}):
            # Change next_edge to contain the cumulative distance
            update_edge = next_edge[:-1] + (new_path_len,)
            current_path[next_node] = update_edge
            stack.append(iter(get_edges(next_node)))
            pass


def _make_q_str(state: Q) -> str:
    phi, loc = state
    return str((str(phi), loc))
