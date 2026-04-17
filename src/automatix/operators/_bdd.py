"""Private BDD helpers for the symbolic polynomial operator.

All ``dd`` internals are hidden behind :class:`BDDDag`; the rest of
automatix never imports from ``dd`` directly.

The backend is selected once at import time:

* ``dd.cudd`` — wraps the CUDD C library (much faster, supports automatic
  variable reordering).
* ``dd.autoref`` — pure-Python fallback.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any

import logic_asts as logic
from morphata.spec import BoolExpr

try:
    import dd.cudd as _dd_backend  # type: ignore[import-not-found]
except ImportError:
    import dd.autoref as _dd_backend


@dataclass(frozen=True)
class BDDDagNode:
    """A single node in the extracted BDD DAG.

    Terminal nodes have ``var_index``, ``low_id``, and ``high_id`` set to
    ``None``.  Internal nodes have all fields populated.
    """

    id: int
    var_index: int | None
    low_id: int | None
    high_id: int | None


@dataclass(frozen=True)
class BDDDag:
    """An extracted, reduced BDD as a plain dataclass.

    Node IDs are dense integers starting from 0.  Terminals always have
    IDs 0 (false) and 1 (true).  ``topo_order`` lists node IDs with children
    before parents (suitable for bottom-up DP).
    """

    nodes: tuple[BDDDagNode, ...]
    root_id: int
    false_id: int
    true_id: int
    num_vars: int
    var_order: tuple[int, ...]
    topo_order: tuple[int, ...]


def boolexpr_to_bdd(
    expr: BoolExpr[int],
    num_vars: int,
    *,
    var_order: Sequence[int] | None = None,
) -> BDDDag:
    """Convert a positive boolean expression over integer state variables to a BDD DAG.

    Parameters
    ----------
    expr :
        Boolean expression with integer variable names (state indices).
        Must be in positive normal form — ``Not`` nodes raise :class:`ValueError`.
    num_vars :
        Total number of state variables.
    var_order :
        Optional permutation of ``range(num_vars)`` specifying the BDD variable
        order.  Defaults to natural order ``0, 1, …, num_vars - 1``.

    Returns
    -------
    BDDDag
        Extracted reduced BDD with dense node IDs and topological ordering.

    Raises
    ------
    ValueError
        If *expr* contains a ``Not`` node, an out-of-range state index, or
        *var_order* is not a valid permutation.
    """
    resolved_order = _resolve_var_order(num_vars, var_order)
    mgr = _dd_backend.BDD()
    var_names = [f"q{i}" for i in resolved_order]
    mgr.declare(*var_names)
    # We supply the variable order explicitly; disable CUDD's dynamic
    # reordering so that node references remain stable throughout BDD
    # construction and DAG extraction.
    mgr.configure(reordering=False)

    # Build BDD bottom-up using the post-order traversal from logic_asts.
    # Variable nodes are stored as their name (string) rather than a node
    # object so that they can always be resolved fresh via mgr.var().
    # Compound (And/Or) results are stored as node objects.
    bdd_cache: dict[BoolExpr[int], Any] = {}

    for subexpr in logic.bool_expr_iter(expr):
        bdd_cache[subexpr] = _convert_node(subexpr, num_vars, mgr, bdd_cache)

    root = _resolve(bdd_cache[expr], mgr)
    return _extract_bdd_dag(root, mgr, num_vars, resolved_order)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_var_order(num_vars: int, var_order: Sequence[int] | None) -> tuple[int, ...]:
    if var_order is None:
        return tuple(range(num_vars))
    order = tuple(var_order)
    if sorted(order) != list(range(num_vars)):
        raise ValueError(f"var_order must be a permutation of range({num_vars}), got {order}")
    return order


def _resolve(cached: Any, mgr: Any) -> Any:
    """Return a fresh node reference from the manager.

    Variable nodes are stored in ``bdd_cache`` as their variable name string
    (e.g. ``"q3"``) and must be looked up via ``mgr.var()`` to get a live
    reference.  Compound nodes (terminals, And/Or results) are already live
    node objects and are returned as-is.
    """
    if isinstance(cached, str):
        return mgr.var(cached)
    return cached


def _convert_node(
    e: BoolExpr[int],
    num_vars: int,
    mgr: Any,
    cache: dict[BoolExpr[int], Any],
) -> Any:
    match e:
        case logic.Literal(val):
            return mgr.true if val else mgr.false
        case logic.Variable(q):
            if not isinstance(q, int):
                raise ValueError(f"Invalid state variable: {q!r}, expected int")
            if not (0 <= q < num_vars):
                raise ValueError(f"Invalid state variable: {q}, expected 0..{num_vars - 1}")
            # Store the variable name, not the node object, so callers always
            # resolve a fresh reference via mgr.var().
            return f"q{q}"
        case logic.And(args):
            nodes = [_resolve(cache[a], mgr) for a in args]  # type: ignore[index]
            return reduce(lambda a, b: mgr.apply("and", a, b), nodes)
        case logic.Or(args):
            nodes = [_resolve(cache[a], mgr) for a in args]  # type: ignore[index]
            return reduce(lambda a, b: mgr.apply("or", a, b), nodes)
        case logic.Not():
            raise ValueError(
                f"Not operator encountered in AFA expression: {e}. "
                "AFAs must be in positive normal form. "
                "Use logic_asts.to_nnf() to normalise if needed."
            )
        case _:
            raise TypeError(f"Unsupported BoolExpr type: {type(e).__name__}")


def _extract_bdd_dag(
    root: Any,
    mgr: Any,
    num_vars: int,
    var_order: tuple[int, ...],
) -> BDDDag:
    """Extract a reduced BDD DAG from a ``dd`` manager and root node.

    Terminal nodes always receive dense IDs 0 (false) and 1 (true) so that
    the tensorisation step can pre-populate its cache unconditionally.

    Node objects are used directly as dict keys (both backends implement
    value-based ``__hash__`` / ``__eq__``).  Variable nodes are re-fetched by
    name via ``mgr.var()`` whenever a stable reference is needed.
    """
    false_dense = 0
    true_dense = 1
    node_to_dense: dict[Any, int] = {
        mgr.false: false_dense,
        mgr.true: true_dense,
    }
    dag_nodes: list[BDDDagNode] = [
        BDDDagNode(id=false_dense, var_index=None, low_id=None, high_id=None),
        BDDDagNode(id=true_dense, var_index=None, low_id=None, high_id=None),
    ]
    topo_order: list[int] = [false_dense, true_dense]
    visited: set[Any] = {mgr.false, mgr.true}

    # Iterative post-order DFS using a (node, processed) marker.
    stack: list[tuple[Any, bool]] = []
    if root not in visited:
        stack.append((root, False))

    while stack:
        node, processed = stack.pop()
        if processed:
            dense_id = len(dag_nodes)
            node_to_dense[node] = dense_id
            var_name: str = node.var  # e.g. "q3"
            state_idx = int(var_name[1:])
            # Re-fetch children by their live node reference.
            low_node = node.low
            high_node = node.high
            low_dense = node_to_dense[low_node]
            high_dense = node_to_dense[high_node]
            dag_nodes.append(
                BDDDagNode(
                    id=dense_id,
                    var_index=state_idx,
                    low_id=low_dense,
                    high_id=high_dense,
                )
            )
            topo_order.append(dense_id)
            continue

        if node in visited:
            continue
        visited.add(node)
        stack.append((node, True))
        stack.append((node.high, False))
        stack.append((node.low, False))

    root_dense = node_to_dense[root]
    return BDDDag(
        nodes=tuple(dag_nodes),
        root_id=root_dense,
        false_id=false_dense,
        true_id=true_dense,
        num_vars=num_vars,
        var_order=var_order,
        topo_order=tuple(topo_order),
    )
