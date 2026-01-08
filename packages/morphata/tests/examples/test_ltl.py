"""Test cases for LTL to alternating automaton conversion.

These tests verify the ltl_to_automaton function with various LTL formulas
including basic propositional logic, temporal operators, and bounded intervals.
"""

import logic_asts
from morphata.acceptance import Buchi, Finite
from morphata.examples.ltl import ltl_to_automaton


def test_literal_true() -> None:
    """Test automaton construction from literal True."""
    expr = logic_asts.parse_expr("True", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Finite)


def test_literal_false() -> None:
    """Test automaton construction from literal False."""
    expr = logic_asts.parse_expr("False", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Finite)


def test_atomic_proposition() -> None:
    """Test automaton construction from atomic proposition."""
    expr = logic_asts.parse_expr("p", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Finite)


def test_negation() -> None:
    """Test automaton construction with negation."""
    expr = logic_asts.parse_expr("! p", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Finite)


def test_conjunction() -> None:
    """Test automaton construction with conjunction."""
    expr = logic_asts.parse_expr("p & q", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Finite)


def test_disjunction() -> None:
    """Test automaton construction with disjunction."""
    expr = logic_asts.parse_expr("p | q", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Finite)


def test_next_operator() -> None:
    """Test automaton construction with Next operator."""
    expr = logic_asts.parse_expr("X p", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Finite)


def test_eventually_operator_finite() -> None:
    """Test automaton construction with Eventually operator (finite)."""
    expr = logic_asts.parse_expr("F p", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Finite)


def test_eventually_operator_buchi() -> None:
    """Test automaton construction with Eventually operator (Buchi)."""
    expr = logic_asts.parse_expr("F p", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=False)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Buchi)


def test_always_operator_finite() -> None:
    """Test automaton construction with Always operator (finite)."""
    expr = logic_asts.parse_expr("G p", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Finite)


def test_always_operator_buchi() -> None:
    """Test automaton construction with Always operator (Buchi)."""
    expr = logic_asts.parse_expr("G p", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=False)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Buchi)


def test_until_operator_finite() -> None:
    """Test automaton construction with Until operator (finite)."""
    expr = logic_asts.parse_expr("p U q", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Finite)


def test_until_operator_buchi() -> None:
    """Test automaton construction with Until operator (Buchi)."""
    expr = logic_asts.parse_expr("p U q", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=False)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Buchi)


def test_bounded_next() -> None:
    """Test automaton construction with bounded Next operator."""
    expr = logic_asts.parse_expr("X[3] p", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Finite)


def test_bounded_eventually() -> None:
    """Test automaton construction with bounded Eventually operator."""
    expr = logic_asts.parse_expr("F[0, 3] p", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Finite)


def test_bounded_always() -> None:
    """Test automaton construction with bounded Always operator."""
    expr = logic_asts.parse_expr("G[0, 3] p", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Finite)


def test_bounded_until() -> None:
    """Test automaton construction with bounded Until operator."""
    expr = logic_asts.parse_expr("p U[0, 3] q", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Finite)


def test_complex_formula_finite() -> None:
    """Test automaton construction with complex nested formula (finite)."""
    expr = logic_asts.parse_expr("(p & X q) | F r", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Finite)


def test_complex_formula_buchi() -> None:
    """Test automaton construction with complex nested formula (Buchi)."""
    expr = logic_asts.parse_expr("G (p -> F q)", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=False)

    assert aut.domain is not None
    assert aut.initial is not None
    assert aut.delta is not None
    assert isinstance(aut.acceptance, Buchi)


def test_domain_states_enumerable() -> None:
    """Test that domain states are enumerable."""
    expr = logic_asts.parse_expr("F p", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    states = list(aut.domain.states) if aut.domain.states is not None else []
    assert len(states) > 0
    assert all(isinstance(s, int) for s in states)


def test_domain_symbols_enumerable() -> None:
    """Test that domain symbols are enumerable (powerset of atomic propositions)."""
    expr = logic_asts.parse_expr("p & q", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    symbols = list(aut.domain.symbols) if aut.domain.symbols is not None else []
    # For 2 atomic propositions (p, q), powerset has 2^2 = 4 elements
    assert len(symbols) == 4


def test_initial_state_is_integer() -> None:
    """Test that initial state is an integer."""
    expr = logic_asts.parse_expr("F p", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    assert isinstance(aut.initial, int)
    assert aut.initial == 0  # Initial state should be mapped to 0


def test_alternating_transition_callable() -> None:
    """Test that delta can be called with state and symbol."""
    expr = logic_asts.parse_expr("p", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    # Get a valid state from the domain
    states = list(aut.domain.states) if aut.domain.states is not None else []
    assert len(states) > 0

    # Get a valid symbol from the domain
    symbols = list(aut.domain.symbols) if aut.domain.symbols is not None else []
    assert len(symbols) > 0

    # Test transition function call
    state = states[0]
    symbol = symbols[0]
    result = aut.delta(state, symbol)

    # Result should be a boolean expression
    assert result is not None


def test_step_run_evaluation() -> None:
    """Test step_run for symbolic evaluation."""
    expr = logic_asts.parse_expr("F p", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    # Get initial symbolic state
    initial_state = logic_asts.Variable(aut.initial)

    # Get a valid symbol
    symbols = list(aut.domain.symbols) if aut.domain.symbols is not None else []
    assert len(symbols) > 0

    # Test step_run
    next_state = aut.delta.step_run(initial_state, symbols[0])
    assert next_state is not None


def test_safety_property() -> None:
    """Test automaton for safety property: G !bad."""
    expr = logic_asts.parse_expr("G ! bad", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=False)

    assert aut.domain is not None
    assert isinstance(aut.acceptance, Buchi)


def test_liveness_property() -> None:
    """Test automaton for liveness property: G F good."""
    expr = logic_asts.parse_expr("G F good", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=False)

    assert aut.domain is not None
    assert isinstance(aut.acceptance, Buchi)


def test_response_property() -> None:
    """Test automaton for response property: G (req -> F ack)."""
    expr = logic_asts.parse_expr("G (req -> F ack)", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=False)

    assert aut.domain is not None
    assert isinstance(aut.acceptance, Buchi)


def test_multiple_atomic_propositions() -> None:
    """Test automaton with multiple atomic propositions."""
    expr = logic_asts.parse_expr("(p & q) | (r & s)", syntax="ltl")
    aut = ltl_to_automaton(expr, finite=True)

    symbols = list(aut.domain.symbols) if aut.domain.symbols is not None else []
    # For 4 atomic propositions, powerset has 2^4 = 16 elements
    assert len(symbols) == 16
