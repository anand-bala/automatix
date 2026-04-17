"""Test cases for LTL to alternating automaton conversion.

These tests verify the ltl_to_automaton function with various LTL formulas
including basic propositional logic, temporal operators, and bounded intervals.
"""

from collections.abc import Set as AbstractSet

import logic_asts
from morphata import AlternatingTransitions
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

    delta = aut.delta
    assert isinstance(aut.initial, int)
    assert aut.initial == 0
    initial_state = logic_asts.Variable(aut.initial)

    assert isinstance(delta, AlternatingTransitions)
    assert logic_asts.is_propositional_logic(initial_state, var_type=int)

    # Get a valid symbol
    symbols: list[AbstractSet[str]] = []
    if aut.domain.symbols is not None:
        symbols = [sym for sym in aut.domain.symbols]
    assert len(symbols) > 0

    # Test step_run
    next_state = delta.step_run(initial_state, symbols[0])  # ty:ignore[invalid-argument-type]
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


# ---------------------------------------------------------------------------
# Semantic acceptance tests via step_run
# ---------------------------------------------------------------------------


def _evaluate_word(
    aut: logic_asts.ltl.LTLExpr[str],
    word: list[frozenset[str]],
    *,
    finite: bool = True,
) -> bool:
    """Evaluate whether *aut* (as AFA) accepts *word* via step_run.

    For finite acceptance: a word is accepted when the run tree formula,
    after processing the entire word, simplifies to True when all
    accepting states are True and non-accepting states are False.
    """
    automaton = ltl_to_automaton(aut, finite=finite)
    assert isinstance(automaton.initial, int)
    assert isinstance(automaton.acceptance, Finite)

    accepting = automaton.acceptance.accepting
    run: logic_asts.base.BaseExpr[int] = logic_asts.Variable(automaton.initial)

    delta = automaton.delta
    assert isinstance(delta, AlternatingTransitions)
    for sym in word:
        run = delta.step_run(run, sym)

    # Evaluate: substitute True for accepting states, False for non-accepting
    def _eval_at_accepting(expr: logic_asts.base.BaseExpr[int]) -> bool:
        match expr:
            case logic_asts.Literal(val):
                return val
            case logic_asts.Variable(q):
                return q in accepting
            case logic_asts.And(args):
                return all(_eval_at_accepting(a) for a in args)  # type: ignore[arg-type]
            case logic_asts.Or(args):
                return any(_eval_at_accepting(a) for a in args)  # type: ignore[arg-type]
            case _:
                raise TypeError(f"Unexpected expression type: {type(expr)}")

    return _eval_at_accepting(run)


class TestLtlSemantics:
    """Semantic acceptance tests for the LTL-to-AFA translation."""

    def test_eventually_accepts_immediate(self) -> None:
        """F(p): word [{p}] is accepted."""
        f = logic_asts.parse_expr("F p", syntax="ltl")
        assert _evaluate_word(f, [frozenset({"p"})])

    def test_eventually_accepts_delayed(self) -> None:
        """F(p): word [{}, {}, {p}] is accepted."""
        f = logic_asts.parse_expr("F p", syntax="ltl")
        assert _evaluate_word(f, [frozenset(), frozenset(), frozenset({"p"})])

    def test_eventually_rejects_never(self) -> None:
        """F(p): word [{}, {}] is rejected."""
        f = logic_asts.parse_expr("F p", syntax="ltl")
        assert not _evaluate_word(f, [frozenset(), frozenset()])

    def test_always_accepts_all(self) -> None:
        """G(p): word [{p}, {p}, {p}] is accepted."""
        f = logic_asts.parse_expr("G p", syntax="ltl")
        assert _evaluate_word(f, [frozenset({"p"})] * 3)

    def test_always_rejects_gap(self) -> None:
        """G(p): word [{p}, {}, {p}] is rejected."""
        f = logic_asts.parse_expr("G p", syntax="ltl")
        assert not _evaluate_word(f, [frozenset({"p"}), frozenset(), frozenset({"p"})])

    def test_always_accepts_empty(self) -> None:
        """G(p): empty word is vacuously true."""
        f = logic_asts.parse_expr("G p", syntax="ltl")
        assert _evaluate_word(f, [])

    def test_until_immediate(self) -> None:
        """p U q: word [{q}] is accepted."""
        f = logic_asts.parse_expr("p U q", syntax="ltl")
        assert _evaluate_word(f, [frozenset({"q"})])

    def test_until_delayed(self) -> None:
        """p U q: word [{p}, {p}, {q}] is accepted."""
        f = logic_asts.parse_expr("p U q", syntax="ltl")
        assert _evaluate_word(f, [frozenset({"p"}), frozenset({"p"}), frozenset({"q"})])

    def test_until_no_q(self) -> None:
        """p U q: word [{p}, {p}] is rejected (q never appears)."""
        f = logic_asts.parse_expr("p U q", syntax="ltl")
        assert not _evaluate_word(f, [frozenset({"p"}), frozenset({"p"})])

    def test_next_accepts(self) -> None:
        """X(p): word [{}, {p}] is accepted."""
        f = logic_asts.parse_expr("X p", syntax="ltl")
        assert _evaluate_word(f, [frozenset(), frozenset({"p"})])

    def test_next_rejects_immediate(self) -> None:
        """X(p): word [{p}] is rejected."""
        f = logic_asts.parse_expr("X p", syntax="ltl")
        assert not _evaluate_word(f, [frozenset({"p"})])

    def test_negation_semantics(self) -> None:
        """!p: word [{p}] is rejected, word [{}] is accepted."""
        f = logic_asts.parse_expr("! p", syntax="ltl")
        assert not _evaluate_word(f, [frozenset({"p"})])
        assert _evaluate_word(f, [frozenset()])
