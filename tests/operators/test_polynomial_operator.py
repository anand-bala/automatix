"""Unit tests for polynomial operator implementation."""

from __future__ import annotations

import itertools
import typing
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import algebraic
import logic_asts as logic
import logic_asts.ltl as ltl
import morphata
import numpy as np
import pytest
from algebraic.polynomials import RankDecomposition
from algebraic.semirings import boolean_algebra
from algebraic.utils.testing import assert_close
from morphata.acceptance import Buchi, Finite
from morphata.spec import BoolExpr
from typing_extensions import override

from automatix.operators.polynomial import (
    PolynomialOperator,
    _extract_accepting_states,
    _infer_num_states,
    boolexpr_to_polynomial,
)

K = typing.TypeVar("K", bound=algebraic.BoundedDistributiveLattice)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def sym(*aps: str) -> frozenset[str]:
    """Build an input symbol (set of atomic propositions present)."""
    return frozenset(aps)


def assert_accepts(
    op: PolynomialOperator,
    word: Sequence[frozenset[str]],
    expected: bool,
    algebra: algebraic.BooleanAlgebra,
) -> None:
    """Assert that *op* accepts/rejects *word*."""
    result = op.accepts(word)
    expected_val = algebra.one if expected else algebra.zero
    assert_close(result, expected_val)


def assert_poly_equivalent(
    actual: RankDecomposition,
    expected: RankDecomposition,
    algebra: algebraic.BooleanAlgebra,
    num_vars: int,
) -> None:
    """Compare two polynomials by evaluating on all boolean points."""
    for bits in itertools.product([False, True], repeat=num_vars):
        point = np.array([algebra.one if b else algebra.zero for b in bits])
        actual_val = actual.evaluate(point)
        expected_val = expected.evaluate(point)
        a_scalar = np.asarray(actual_val.factors.data[0, 0, 0]).flat[0]
        e_scalar = np.asarray(expected_val.factors.data[0, 0, 0]).flat[0]
        np.testing.assert_allclose(a_scalar, e_scalar, err_msg=f"Mismatch at point {bits}")


@dataclass
class SimpleDomain(morphata.Domain[int, str]):
    """Minimal domain for handcrafted AFA tests."""

    _states: frozenset[int]
    _symbols: tuple[str, ...]

    @property
    @override
    def states(self) -> Iterable[int] | None:
        return iter(self._states)

    @property
    @override
    def symbols(self) -> Iterable[str] | None:
        return iter(self._symbols)


def extract_scalar(poly_result: RankDecomposition, algebra: algebraic.BoundedDistributiveLattice) -> np.generic:
    """Extract scalar value from polynomial evaluation result.

    RankDecomposition.evaluate() returns a constant RankDecomposition.
    Extract the actual scalar value from factors[0, 0, 0].
    """
    assert poly_result.algebra == algebra
    val: np.generic = np.asarray(poly_result.factors.data[0, 0, 0]).flat[0]
    return val


class TestBoolExprConversion:
    """Test boolean expression to polynomial conversion."""

    @pytest.fixture
    def algebra(self) -> algebraic.BooleanAlgebra:
        """Boolean algebra for tests."""
        return boolean_algebra()

    def test_literal_true(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Literal(True) -> one polynomial."""
        expr = logic.Literal(True)
        poly = boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra, backend="numpy")

        point = np.array([algebra.zero, algebra.zero, algebra.zero])
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert np.allclose(scalar, np.asarray(algebra.one))

    def test_literal_false(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Literal(False) -> zero polynomial."""
        expr = logic.Literal(False)
        poly = boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra, backend="numpy")

        point = np.array([algebra.one, algebra.one, algebra.one])
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert np.allclose(scalar, np.asarray(algebra.zero))

    def test_variable(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Variable(q) -> x_q polynomial."""
        expr = logic.Variable(2)
        poly = boolexpr_to_polynomial(expr, num_vars=5, algebra=algebra, backend="numpy")

        point = np.array([algebra.zero, algebra.zero, algebra.one, algebra.zero, algebra.zero])
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert np.allclose(scalar, np.asarray(algebra.one))

        point2 = np.array([algebra.zero, algebra.zero, algebra.zero, algebra.zero, algebra.zero])
        result = poly.evaluate(point2)
        scalar = extract_scalar(result, algebra)
        assert np.allclose(scalar, np.asarray(algebra.zero))

    def test_and_operation(self, algebra: algebraic.BooleanAlgebra) -> None:
        """And(x, y) -> x * y."""
        expr: logic.BoolExpr[int] = logic.And(tuple(logic.Variable(i) for i in [0, 1]))
        poly = boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra, backend="numpy")

        test_cases = [
            ({0: False, 1: False}, False),
            ({0: False, 1: True}, False),
            ({0: True, 1: False}, False),
            ({0: True, 1: True}, True),
        ]

        for point_dict, expected in test_cases:
            point = np.array([algebra.one if point_dict.get(i, False) else algebra.zero for i in range(3)])
            result = poly.evaluate(point)
            scalar = extract_scalar(result, algebra)
            expected_val = algebra.one if expected else algebra.zero
            assert np.allclose(scalar, np.asarray(expected_val)), f"Failed for {point_dict}"

    def test_or_operation(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Or(x, y) -> x + y."""
        expr: logic.BoolExpr[int] = logic.Or(tuple(logic.Variable(i) for i in [0, 1]))
        poly = boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra, backend="numpy")

        test_cases = [
            ({0: False, 1: False}, False),
            ({0: False, 1: True}, True),
            ({0: True, 1: False}, True),
            ({0: True, 1: True}, True),
        ]

        for point_dict, expected in test_cases:
            point = np.array([algebra.one if point_dict.get(i, False) else algebra.zero for i in range(3)])
            result = poly.evaluate(point)
            scalar = extract_scalar(result, algebra)
            expected_val = algebra.one if expected else algebra.zero
            assert np.allclose(scalar, np.asarray(expected_val)), f"Failed for {point_dict}"

    def test_not_operator_raises_assertion(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Not operator should raise ValueError."""
        expr: logic.BoolExpr[int] = logic.Not(logic.Variable(0))

        with pytest.raises(ValueError, match="Not operator encountered"):
            boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra, backend="numpy")

    def test_complex_expression(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Test complex expression: (x_0 AND x_1) OR x_2."""
        x0 = logic.Variable(0)
        x1 = logic.Variable(1)
        x2 = logic.Variable(2)
        expr = typing.cast(logic.BoolExpr[int], (x0 & x1) | x2)
        poly = boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra, backend="numpy")

        point = np.array([algebra.zero, algebra.zero, algebra.zero])
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert np.allclose(scalar, np.asarray(algebra.zero))

        point = np.array([algebra.one, algebra.one, algebra.zero])
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert np.allclose(scalar, np.asarray(algebra.one))

        point = np.array([algebra.zero, algebra.zero, algebra.one])
        result = poly.evaluate(point)
        scalar = extract_scalar(result, algebra)
        assert np.allclose(scalar, np.asarray(algebra.one))

    def test_invalid_state_index(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Invalid state index should raise ValueError."""
        expr = logic.Variable(10)

        with pytest.raises(ValueError, match="Invalid state variable"):
            boolexpr_to_polynomial(expr, num_vars=3, algebra=algebra, backend="numpy")


# ---------------------------------------------------------------------------
# PolynomialOperator – construction
# ---------------------------------------------------------------------------


class TestFromLtl:
    """Test PolynomialOperator.from_ltl construction."""

    @pytest.fixture
    def algebra(self) -> algebraic.BooleanAlgebra:
        return boolean_algebra()

    def test_from_ltl_eventually(self, algebra: algebraic.BooleanAlgebra) -> None:
        """from_ltl(F(a)) produces a valid operator."""
        formula = ltl.Eventually(ltl.Variable("a"))
        op = PolynomialOperator.from_ltl(formula, algebra, backend="numpy", finite=True)
        assert op.num_states >= 1
        assert op.algebra is algebra

    def test_from_ltl_always(self, algebra: algebraic.BooleanAlgebra) -> None:
        """from_ltl(G(a)) creates a valid operator."""
        formula = ltl.Always(ltl.Variable("a"))
        op = PolynomialOperator.from_ltl(formula, algebra, backend="numpy", finite=True)
        assert op.num_states >= 1

    def test_from_ltl_until(self, algebra: algebraic.BooleanAlgebra) -> None:
        """from_ltl(a U b) creates a valid operator."""
        formula = ltl.Until(ltl.Variable("a"), ltl.Variable("b"))
        op = PolynomialOperator.from_ltl(formula, algebra, backend="numpy", finite=True)
        assert op.num_states >= 1

    def test_from_ltl_next(self, algebra: algebraic.BooleanAlgebra) -> None:
        """from_ltl(X(a)) creates a valid operator."""
        formula = ltl.Next(ltl.Variable("a"))
        op = PolynomialOperator.from_ltl(formula, algebra, backend="numpy", finite=True)
        assert op.num_states >= 1


class TestFromAfa:
    """Test PolynomialOperator.from_afa construction and error paths."""

    @pytest.fixture
    def algebra(self) -> algebraic.BooleanAlgebra:
        return boolean_algebra()

    def _make_simple_afa(
        self,
        transitions: dict[int, dict[str, BoolExpr[int]]],
        accepting: frozenset[int],
        symbols: tuple[str, ...],
        initial: int | set[int] | BoolExpr[int] = 0,
    ) -> morphata.Automaton[int, str]:
        """Create a minimal AFA from explicit transitions."""
        states = frozenset(transitions.keys())
        domain = SimpleDomain(_states=states, _symbols=symbols)
        delta = morphata.AlternatingTransitionRelation(transitions)
        acceptance = Finite(accepting)
        return morphata.Automaton(domain=domain, initial=initial, delta=delta, acceptance=acceptance)

    def test_from_afa_basic(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Basic from_afa construction with a 2-state AFA."""
        # State 0 --a--> state 1, state 1 is accepting and self-loops
        transitions: dict[int, dict[str, BoolExpr[int]]] = {
            0: {"a": logic.Variable(1), "b": logic.Literal(False)},
            1: {"a": logic.Variable(1), "b": logic.Variable(1)},
        }
        aut = self._make_simple_afa(transitions, frozenset({1}), ("a", "b"))
        op = PolynomialOperator.from_afa(aut, algebra, backend="numpy")
        assert op.num_states == 2
        assert op.accepting_states == frozenset({1})

    def test_from_afa_initial_set(self, algebra: algebraic.BooleanAlgebra) -> None:
        """from_afa with set-valued initial states."""
        transitions: dict[int, dict[str, BoolExpr[int]]] = {
            0: {"a": logic.Literal(True)},
            1: {"a": logic.Literal(True)},
        }
        aut = self._make_simple_afa(transitions, frozenset({0, 1}), ("a",), initial={0, 1})
        op = PolynomialOperator.from_afa(aut, algebra, backend="numpy")
        assert op.num_states == 2

    def test_from_afa_initial_singleton_set(self, algebra: algebraic.BooleanAlgebra) -> None:
        """from_afa with a singleton set initial."""
        transitions: dict[int, dict[str, BoolExpr[int]]] = {
            0: {"a": logic.Literal(True)},
        }
        aut = self._make_simple_afa(transitions, frozenset({0}), ("a",), initial={0})
        op = PolynomialOperator.from_afa(aut, algebra, backend="numpy")
        assert op.num_states == 1

    def test_from_afa_initial_boolexpr(self, algebra: algebraic.BooleanAlgebra) -> None:
        """from_afa with BoolExpr initial configuration."""
        transitions: dict[int, dict[str, BoolExpr[int]]] = {
            0: {"a": logic.Literal(True)},
            1: {"a": logic.Literal(True)},
        }
        initial_expr: BoolExpr[int] = logic.And((logic.Variable(0), logic.Variable(1)))
        aut = self._make_simple_afa(transitions, frozenset({0, 1}), ("a",), initial=initial_expr)
        op = PolynomialOperator.from_afa(aut, algebra, backend="numpy")
        assert op.num_states == 2

    def test_from_afa_empty_initial_set_raises(self, algebra: algebraic.BooleanAlgebra) -> None:
        """from_afa with empty initial set raises ValueError."""
        transitions: dict[int, dict[str, BoolExpr[int]]] = {
            0: {"a": logic.Literal(True)},
        }
        aut = self._make_simple_afa(transitions, frozenset({0}), ("a",), initial=set())
        with pytest.raises(ValueError, match="Cannot have empty initial set"):
            PolynomialOperator.from_afa(aut, algebra, backend="numpy")

    def test_from_afa_non_alternating_raises(self, algebra: algebraic.BooleanAlgebra) -> None:
        """from_afa with non-alternating transitions raises TypeError."""
        domain = SimpleDomain(_states=frozenset({0}), _symbols=("a",))
        delta = morphata.NonDeterministicTransitionRelation({0: {"a": frozenset({0})}})
        acceptance = Finite(frozenset({0}))
        aut = morphata.Automaton(domain=domain, initial=0, delta=delta, acceptance=acceptance)
        with pytest.raises(TypeError, match="AlternatingTransitions"):
            PolynomialOperator.from_afa(aut, algebra, backend="numpy")

    def test_from_afa_non_finite_acceptance_raises(self, algebra: algebraic.BooleanAlgebra) -> None:
        """from_afa with non-Finite acceptance raises NotImplementedError."""
        transitions: dict[int, dict[str, BoolExpr[int]]] = {
            0: {"a": logic.Variable(0)},
        }
        domain = SimpleDomain(_states=frozenset({0}), _symbols=("a",))
        delta = morphata.AlternatingTransitionRelation(transitions)
        acceptance = Buchi(frozenset({0}))
        aut = morphata.Automaton(domain=domain, initial=0, delta=delta, acceptance=acceptance)
        with pytest.raises(NotImplementedError, match="Finite acceptance"):
            PolynomialOperator.from_afa(aut, algebra, backend="numpy")

    def test_from_afa_no_cache(self, algebra: algebraic.BooleanAlgebra) -> None:
        """from_afa with cache_transitions=False produces empty cache."""
        transitions: dict[int, dict[str, BoolExpr[int]]] = {
            0: {"a": logic.Literal(True)},
        }
        aut = self._make_simple_afa(transitions, frozenset({0}), ("a",))
        op = PolynomialOperator.from_afa(aut, algebra, backend="numpy", cache_transitions=False)
        assert op.num_states == 1
        # Cache is empty — step should raise KeyError
        with pytest.raises(KeyError):
            op.step(op.initial_poly, "a")


# ---------------------------------------------------------------------------
# PolynomialOperator – acceptance (end-to-end via from_ltl)
# ---------------------------------------------------------------------------


class TestAccepts:
    """Test PolynomialOperator.accepts with various LTL formulas."""

    @pytest.fixture
    def algebra(self) -> algebraic.BooleanAlgebra:
        return boolean_algebra()

    # -- F(a): eventually a ------------------------------------------------

    def test_eventually_accepts_immediate(self, algebra: algebraic.BooleanAlgebra) -> None:
        """F(a): word [a] is accepted."""
        op = PolynomialOperator.from_ltl(ltl.Eventually(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        assert_accepts(op, [sym("a")], True, algebra)

    def test_eventually_accepts_delayed(self, algebra: algebraic.BooleanAlgebra) -> None:
        """F(a): word [∅, ∅, a] is accepted."""
        op = PolynomialOperator.from_ltl(ltl.Eventually(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        assert_accepts(op, [sym(), sym(), sym("a")], True, algebra)

    def test_eventually_rejects_never(self, algebra: algebraic.BooleanAlgebra) -> None:
        """F(a): word [∅, ∅] is rejected."""
        op = PolynomialOperator.from_ltl(ltl.Eventually(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        assert_accepts(op, [sym(), sym()], False, algebra)

    def test_eventually_rejects_empty_word(self, algebra: algebraic.BooleanAlgebra) -> None:
        """F(a): empty word is rejected."""
        op = PolynomialOperator.from_ltl(ltl.Eventually(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        assert_accepts(op, [], False, algebra)

    # -- G(a): always a ----------------------------------------------------

    def test_always_accepts_all_a(self, algebra: algebraic.BooleanAlgebra) -> None:
        """G(a): word [a, a, a] is accepted."""
        op = PolynomialOperator.from_ltl(ltl.Always(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        assert_accepts(op, [sym("a"), sym("a"), sym("a")], True, algebra)

    def test_always_rejects_missing_a(self, algebra: algebraic.BooleanAlgebra) -> None:
        """G(a): word [a, ∅, a] is rejected."""
        op = PolynomialOperator.from_ltl(ltl.Always(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        assert_accepts(op, [sym("a"), sym(), sym("a")], False, algebra)

    def test_always_accepts_empty_word(self, algebra: algebraic.BooleanAlgebra) -> None:
        """G(a): empty word is vacuously true."""
        op = PolynomialOperator.from_ltl(ltl.Always(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        assert_accepts(op, [], True, algebra)

    # -- a U b: until -------------------------------------------------------

    def test_until_accepts_immediate_b(self, algebra: algebraic.BooleanAlgebra) -> None:
        """a U b: word [b] is accepted."""
        op = PolynomialOperator.from_ltl(
            ltl.Until(ltl.Variable("a"), ltl.Variable("b")), algebra, backend="numpy", finite=True
        )
        assert_accepts(op, [sym("b")], True, algebra)

    def test_until_accepts_a_then_b(self, algebra: algebraic.BooleanAlgebra) -> None:
        """a U b: word [a, a, b] is accepted."""
        op = PolynomialOperator.from_ltl(
            ltl.Until(ltl.Variable("a"), ltl.Variable("b")), algebra, backend="numpy", finite=True
        )
        assert_accepts(op, [sym("a"), sym("a"), sym("b")], True, algebra)

    def test_until_rejects_no_b(self, algebra: algebraic.BooleanAlgebra) -> None:
        """a U b: word [a, a, a] is rejected (b never appears)."""
        op = PolynomialOperator.from_ltl(
            ltl.Until(ltl.Variable("a"), ltl.Variable("b")), algebra, backend="numpy", finite=True
        )
        assert_accepts(op, [sym("a"), sym("a"), sym("a")], False, algebra)

    # -- X(a): next ----------------------------------------------------------

    def test_next_accepts(self, algebra: algebraic.BooleanAlgebra) -> None:
        """X(a): word [∅, a] is accepted."""
        op = PolynomialOperator.from_ltl(ltl.Next(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        assert_accepts(op, [sym(), sym("a")], True, algebra)

    def test_next_rejects_immediate(self, algebra: algebraic.BooleanAlgebra) -> None:
        """X(a): word [a] is rejected (a must be at second position)."""
        op = PolynomialOperator.from_ltl(ltl.Next(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        assert_accepts(op, [sym("a")], False, algebra)

    # -- Conjunctions / disjunctions ----------------------------------------

    def test_conjunction(self, algebra: algebraic.BooleanAlgebra) -> None:
        """F(a) & F(b): word [a, b] is accepted."""
        f = typing.cast(
            logic.LTLExpr[str],
            logic.And((ltl.Eventually(ltl.Variable("a")), ltl.Eventually(ltl.Variable("b")))),
        )
        op = PolynomialOperator.from_ltl(f, algebra, backend="numpy", finite=True)
        assert_accepts(op, [sym("a"), sym("b")], True, algebra)

    def test_conjunction_rejects_missing(self, algebra: algebraic.BooleanAlgebra) -> None:
        """F(a) & F(b): word [a] is rejected (b never appears)."""
        f = typing.cast(
            logic.LTLExpr[str],
            logic.And((ltl.Eventually(ltl.Variable("a")), ltl.Eventually(ltl.Variable("b")))),
        )
        op = PolynomialOperator.from_ltl(f, algebra, backend="numpy", finite=True)
        assert_accepts(op, [sym("a")], False, algebra)

    def test_disjunction(self, algebra: algebraic.BooleanAlgebra) -> None:
        """F(a) | F(b): word [b] is accepted."""
        f = typing.cast(
            logic.LTLExpr[str],
            logic.Or((ltl.Eventually(ltl.Variable("a")), ltl.Eventually(ltl.Variable("b")))),
        )
        op = PolynomialOperator.from_ltl(f, algebra, backend="numpy", finite=True)
        assert_accepts(op, [sym("b")], True, algebra)

    # -- Literal formulas ----------------------------------------------------

    def test_literal_true_empty_word(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Literal True accepts the empty word."""
        op = PolynomialOperator.from_ltl(ltl.Literal(True), algebra, backend="numpy", finite=True)
        assert_accepts(op, [], True, algebra)

    def test_literal_false_empty_word(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Literal False rejects the empty word."""
        op = PolynomialOperator.from_ltl(ltl.Literal(False), algebra, backend="numpy", finite=True)
        assert_accepts(op, [], False, algebra)


# ---------------------------------------------------------------------------
# PolynomialOperator – run_polynomial
# ---------------------------------------------------------------------------


class TestRunPolynomial:
    """Test PolynomialOperator.run_polynomial intermediate computation."""

    @pytest.fixture
    def algebra(self) -> algebraic.BooleanAlgebra:
        return boolean_algebra()

    def test_empty_word_returns_initial(self, algebra: algebraic.BooleanAlgebra) -> None:
        """run_polynomial([]) should return the initial polynomial."""
        op = PolynomialOperator.from_ltl(ltl.Eventually(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        run_poly = op.run_polynomial([])
        assert_poly_equivalent(run_poly, op.initial_poly, algebra, op.num_states)

    def test_single_step_produces_rank_decomposition(self, algebra: algebraic.BooleanAlgebra) -> None:
        """run_polynomial on a single symbol returns a RankDecomposition."""
        op = PolynomialOperator.from_ltl(ltl.Eventually(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        run_poly = op.run_polynomial([sym("a")])
        assert isinstance(run_poly, RankDecomposition)

    def test_run_poly_consistent_with_accepts(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Evaluating run_polynomial at accepting states matches accepts()."""
        op = PolynomialOperator.from_ltl(ltl.Eventually(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        word: list[frozenset[str]] = [sym(), sym("a")]
        run_poly = op.run_polynomial(word)
        eval_result = op.evaluate_at_accepting(run_poly)
        direct_result = op.accepts(word)
        assert_close(eval_result, direct_result)


# ---------------------------------------------------------------------------
# PolynomialOperator – step
# ---------------------------------------------------------------------------


class TestStep:
    """Test PolynomialOperator.step single-step transitions."""

    @pytest.fixture
    def algebra(self) -> algebraic.BooleanAlgebra:
        return boolean_algebra()

    def test_step_produces_rank_decomposition(self, algebra: algebraic.BooleanAlgebra) -> None:
        """step() returns a RankDecomposition."""
        op = PolynomialOperator.from_ltl(ltl.Eventually(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        result = op.step(op.initial_poly, sym("a"))
        assert isinstance(result, RankDecomposition)

    def test_step_sequence_equals_run_polynomial(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Manually stepping through a word matches run_polynomial."""
        op = PolynomialOperator.from_ltl(ltl.Eventually(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        word = [sym(), sym("a")]
        # Manual stepping
        current = op.initial_poly
        for s in word:
            current = op.step(current, s)
        run_poly = op.run_polynomial(word)
        assert_poly_equivalent(current, run_poly, algebra, op.num_states)

    def test_step_cache_miss_raises_key_error(self, algebra: algebraic.BooleanAlgebra) -> None:
        """step() raises KeyError with helpful message when symbol not in cache."""
        transitions: dict[int, dict[str, BoolExpr[int]]] = {
            0: {"a": logic.Literal(True)},
        }
        domain = SimpleDomain(_states=frozenset({0}), _symbols=("a",))
        delta = morphata.AlternatingTransitionRelation(transitions)
        acceptance = Finite(frozenset({0}))
        aut = morphata.Automaton(domain=domain, initial=0, delta=delta, acceptance=acceptance)
        op = PolynomialOperator.from_afa(aut, algebra, backend="numpy")

        with pytest.raises(KeyError, match="Transition polynomial not found"):
            op.step(op.initial_poly, "unknown_symbol")


# ---------------------------------------------------------------------------
# PolynomialOperator – evaluate_at_accepting
# ---------------------------------------------------------------------------


class TestEvaluateAtAccepting:
    """Test PolynomialOperator.evaluate_at_accepting."""

    @pytest.fixture
    def algebra(self) -> algebraic.BooleanAlgebra:
        return boolean_algebra()

    def test_accepting_state_gives_one(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Polynomial = x_q where q is accepting → evaluates to one."""
        transitions: dict[int, dict[str, BoolExpr[int]]] = {
            0: {"a": logic.Literal(True)},
        }
        domain = SimpleDomain(_states=frozenset({0}), _symbols=("a",))
        delta = morphata.AlternatingTransitionRelation(transitions)
        acceptance = Finite(frozenset({0}))
        aut = morphata.Automaton(domain=domain, initial=0, delta=delta, acceptance=acceptance)
        op = PolynomialOperator.from_afa(aut, algebra, backend="numpy")

        # initial_poly represents x_0, and state 0 is accepting
        result = op.evaluate_at_accepting(op.initial_poly)
        assert_close(result, algebra.one)

    def test_non_accepting_state_gives_zero(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Polynomial = x_q where q is NOT accepting → evaluates to zero."""
        transitions: dict[int, dict[str, BoolExpr[int]]] = {
            0: {"a": logic.Variable(1)},
            1: {"a": logic.Literal(True)},
        }
        domain = SimpleDomain(_states=frozenset({0, 1}), _symbols=("a",))
        delta = morphata.AlternatingTransitionRelation(transitions)
        # Only state 1 is accepting; state 0 is not
        acceptance = Finite(frozenset({1}))
        aut = morphata.Automaton(domain=domain, initial=0, delta=delta, acceptance=acceptance)
        op = PolynomialOperator.from_afa(aut, algebra, backend="numpy")

        # initial_poly = x_0, but only state 1 is accepting → zero
        result = op.evaluate_at_accepting(op.initial_poly)
        assert_close(result, algebra.zero)

    def test_constant_one_poly(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Constant one polynomial evaluates to one regardless of accepting states."""
        op = PolynomialOperator.from_ltl(ltl.Eventually(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        one_poly = RankDecomposition.one(op.num_states, algebra, backend="numpy")
        result = op.evaluate_at_accepting(one_poly)
        assert_close(result, algebra.one)

    def test_constant_zero_poly(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Constant zero polynomial evaluates to zero."""
        op = PolynomialOperator.from_ltl(ltl.Eventually(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        zero_poly = RankDecomposition.zero(op.num_states, algebra, backend="numpy")
        result = op.evaluate_at_accepting(zero_poly)
        assert_close(result, algebra.zero)


# ---------------------------------------------------------------------------
# tree_flatten / tree_unflatten round-trip
# ---------------------------------------------------------------------------


class TestPyTreeRoundTrip:
    """Test AlgebraicPyTree interface."""

    @pytest.fixture
    def algebra(self) -> algebraic.BooleanAlgebra:
        return boolean_algebra()

    def test_round_trip_preserves_fields(self, algebra: algebraic.BooleanAlgebra) -> None:
        """tree_unflatten(tree_flatten(...)) preserves all fields."""
        op = PolynomialOperator.from_ltl(ltl.Eventually(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        children, aux = op.tree_flatten()
        restored = PolynomialOperator.tree_unflatten(aux, children)

        assert restored.num_states == op.num_states
        assert restored.accepting_states == op.accepting_states
        assert restored.algebra is op.algebra
        assert_poly_equivalent(restored.initial_poly, op.initial_poly, algebra, op.num_states)

    def test_round_trip_preserves_acceptance(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Round-tripped operator produces identical acceptance results."""
        op = PolynomialOperator.from_ltl(ltl.Eventually(ltl.Variable("a")), algebra, backend="numpy", finite=True)
        children, aux = op.tree_flatten()
        restored = PolynomialOperator.tree_unflatten(aux, children)

        word_accept = [sym("a")]
        word_reject = [sym()]
        assert_close(restored.accepts(word_accept), op.accepts(word_accept))
        assert_close(restored.accepts(word_reject), op.accepts(word_reject))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class TestInferNumStates:
    """Test _infer_num_states helper."""

    def test_contiguous_states(self) -> None:
        """Contiguous states 0..n-1 work correctly."""
        domain = SimpleDomain(_states=frozenset({0, 1, 2}), _symbols=())
        delta: morphata.AlternatingTransitionRelation[int, object] = morphata.AlternatingTransitionRelation({})
        acceptance = Finite(frozenset({0}))
        aut = morphata.Automaton(domain=domain, initial=0, delta=delta, acceptance=acceptance)
        assert _infer_num_states(aut) == 3

    def test_non_contiguous_states_raises(self) -> None:
        """Non-contiguous states raise ValueError."""
        domain = SimpleDomain(_states=frozenset({0, 2}), _symbols=())
        delta: morphata.AlternatingTransitionRelation[int, object] = morphata.AlternatingTransitionRelation({})
        acceptance = Finite(frozenset({0}))
        aut = morphata.Automaton(domain=domain, initial=0, delta=delta, acceptance=acceptance)
        with pytest.raises(ValueError, match="contiguous"):
            _infer_num_states(aut)

    def test_empty_states_raises(self) -> None:
        """Empty state set raises ValueError."""
        domain = SimpleDomain(_states=frozenset(), _symbols=())
        delta: morphata.AlternatingTransitionRelation[int, object] = morphata.AlternatingTransitionRelation({})
        acceptance: Finite[int] = Finite(frozenset())
        aut = morphata.Automaton(domain=domain, initial=frozenset(), delta=delta, acceptance=acceptance)
        with pytest.raises(ValueError, match="no states"):
            _infer_num_states(aut)


class TestExtractAcceptingStates:
    """Test _extract_accepting_states helper."""

    def test_finite_acceptance(self) -> None:
        """Finite acceptance returns the accepting set."""
        acc = Finite(frozenset({1, 3}))
        result = _extract_accepting_states(acc)
        assert set(result) == {1, 3}

    def test_non_finite_raises(self) -> None:
        """Non-Finite acceptance raises NotImplementedError."""
        acc = Buchi(frozenset({0}))
        with pytest.raises(NotImplementedError, match="Unsupported acceptance"):
            _extract_accepting_states(acc)
