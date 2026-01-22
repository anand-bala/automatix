"""Sanity tests for STREL automaton construction.

These tests verify basic properties of the strel_to_automata function
with a focus on reach and escape operators using real-world specifications.
"""

from __future__ import annotations

import typing

import logic_asts
import morphata
import networkx as nx
from morphata.acceptance import Finite
from morphata.examples.strel import strel_to_automata

if typing.TYPE_CHECKING:
    type Input = nx.Graph[int]
else:
    type Input = nx.Graph
type BoolExpr[Var] = logic_asts.base.BaseExpr[Var]


def label_fn(graph: Input, loc: int, pred: str) -> bool:
    """Label function for testing that checks node attributes."""
    return typing.cast(bool, graph.nodes[loc].get(pred, False))


def create_simple_graph() -> Input:
    """Create a simple graph for testing with 5 nodes."""
    g: Input = nx.Graph()
    for i in range(5):
        g.add_node(i, drone=False, groundstation=False, obstacle=False, goal=False)
    # Create a chain: 0 -- 1 -- 2 -- 3 -- 4
    g.add_edge(0, 1, hop=1)
    g.add_edge(1, 2, hop=1)
    g.add_edge(2, 3, hop=1)
    g.add_edge(3, 4, hop=1)
    return g


def test_reach_avoid_spec_parsing() -> None:
    """Test that the reach-avoid specification from swarm-monitoring can be parsed."""
    spec = r"(G ! obstacle) & ((somewhere[0,2] groundstation) U goal)"
    expr = logic_asts.parse_expr(spec, syntax="strel")

    aut = strel_to_automata(
        expr,
        dist_attr="hop",
        label_fn=label_fn,
        num_locations=5,
        ego_location=0,
    )

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Finite)


def test_establish_comms_spec_parsing() -> None:
    """Test that the establish-comms specification from swarm-monitoring can be parsed."""
    spec = r"G( (somewhere[1,2] drone) | (F[0, 100] somewhere[1,2] (drone | groundstation)) )"
    expr = logic_asts.parse_expr(spec, syntax="strel")

    aut = strel_to_automata(
        expr,
        dist_attr="hop",
        label_fn=label_fn,
        num_locations=5,
        ego_location=0,
    )

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Finite)


def test_somewhere_rewritten_to_reach() -> None:
    """Test that somewhere is rewritten to reach internally."""
    # somewhere[a,b](p) should be rewritten to reach[a,b](True, p)
    spec = r"somewhere[0,2] goal"
    expr = logic_asts.parse_expr(spec, syntax="strel")

    aut = strel_to_automata(
        expr,
        dist_attr="hop",
        label_fn=label_fn,
        num_locations=5,
        ego_location=0,
    )

    assert aut.domain is not None
    assert aut.initial is not None
    assert isinstance(aut.acceptance, Finite)


def test_reach_in_until_formula() -> None:
    """Test reach operator within until: (somewhere[0,2] groundstation) U goal."""
    spec = r"(somewhere[0,2] groundstation) U goal"
    expr = logic_asts.parse_expr(spec, syntax="strel")

    aut = strel_to_automata(
        expr,
        dist_attr="hop",
        label_fn=label_fn,
        num_locations=5,
        ego_location=0,
    )

    assert aut.domain is not None
    assert aut.initial is not None
    assert isinstance(aut.acceptance, Finite)


def test_reach_delta_evaluation() -> None:
    """Test that delta can be evaluated on a graph with reach operator."""
    spec = r"somewhere[0,2] goal"
    expr = logic_asts.parse_expr(spec, syntax="strel")

    aut: morphata.Automaton[tuple[int, int], Input] = strel_to_automata(
        expr,
        dist_attr="hop",
        label_fn=label_fn,
        num_locations=5,
        ego_location=0,
    )
    delta = aut.delta
    initial_state = aut.initial
    assert isinstance(delta, morphata.AlternatingTransitions)
    assert logic_asts.is_propositional_logic(initial_state, tuple[int, int])

    # Create a graph with goal at node 2
    graph = create_simple_graph()
    graph.nodes[2]["goal"] = True

    # Test that we can call delta
    next_state = delta.step_run(initial_state, graph)  # ty:ignore[invalid-argument-type]

    assert next_state is not None


def test_reach_with_multiple_ego_locations() -> None:
    """Test reach operator with multiple ego locations."""
    spec = r"somewhere[0,1] groundstation"
    expr = logic_asts.parse_expr(spec, syntax="strel")

    aut = strel_to_automata(
        expr,
        dist_attr="hop",
        label_fn=label_fn,
        num_locations=5,
        ego_location=[0, 1],
    )

    assert aut.domain is not None
    assert aut.initial is not None
    assert isinstance(aut.acceptance, Finite)


def test_nested_reach_operators() -> None:
    """Test formula with nested spatial operators."""
    spec = r"F (somewhere[0,1] goal)"
    expr = logic_asts.parse_expr(spec, syntax="strel")

    aut = strel_to_automata(
        expr,
        dist_attr="hop",
        label_fn=label_fn,
        num_locations=5,
        ego_location=0,
    )

    assert aut.domain is not None
    assert aut.initial is not None
    assert isinstance(aut.acceptance, Finite)


def test_reach_avoid_full_evaluation() -> None:
    """Test full reach-avoid spec can be evaluated on a concrete graph."""
    spec = r"(G ! obstacle) & ((somewhere[0,2] groundstation) U goal)"
    expr = logic_asts.parse_expr(spec, syntax="strel")

    aut = strel_to_automata(
        expr,
        dist_attr="hop",
        label_fn=label_fn,
        num_locations=5,
        ego_location=0,
    )
    # Create graph with groundstation at node 1 and goal at node 4
    graph = create_simple_graph()
    graph.nodes[1]["groundstation"] = True
    graph.nodes[4]["goal"] = True

    # Evaluate initial state
    delta = aut.delta
    initial_state = aut.initial
    assert isinstance(delta, morphata.AlternatingTransitions)
    assert logic_asts.is_propositional_logic(initial_state, tuple[int, int])
    next_state = delta.step_run(initial_state, graph)  # ty:ignore[invalid-argument-type]

    assert next_state is not None
