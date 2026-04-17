"""Unit tests for the symbolic polynomial operator (BDD-based construction)."""

from __future__ import annotations

import itertools
import typing
from collections.abc import Iterable
from dataclasses import dataclass

import algebraic
import logic_asts as logic
import logic_asts.ltl as ltl
import morphata
import numpy as np
import pytest
from algebraic.polynomials import RankDecomposition
from algebraic.polynomials.rank_decomp import LowRankFactors
from algebraic.semirings import boolean_algebra
from algebraic.utils.testing import assert_close
from morphata.acceptance import Buchi, Finite
from morphata.spec import BoolExpr
from typing_extensions import override

from automatix.operators._bdd import boolexpr_to_bdd
from automatix.operators.polynomial import boolexpr_to_polynomial
from automatix.operators.symbolic_polynomial import (
    SymbolicPolynomialOperator,
    bdd_to_rank_decomposition,
    boolexpr_to_symbolic_polynomial,
)

K = typing.TypeVar("K", bound=algebraic.BoundedDistributiveLattice)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def sym(*aps: str) -> frozenset[str]:
    """Build an input symbol (set of atomic propositions present)."""
    return frozenset(aps)


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


def extract_scalar(poly_result: RankDecomposition, algebra: algebraic.BoundedDistributiveLattice) -> np.generic:
    assert poly_result.algebra == algebra
    val: np.generic = np.asarray(poly_result.factors.data[0, 0, 0]).flat[0]
    return val


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


# ---------------------------------------------------------------------------
# TestBoolExprToBDD
# ---------------------------------------------------------------------------


class TestBoolExprToBDD:
    """Unit tests for boolexpr_to_bdd."""

    def test_literal_true(self) -> None:
        """Literal(True) -> BDD whose root is the true terminal."""
        expr = logic.Literal(True)
        bdd = boolexpr_to_bdd(expr, num_vars=2)
        assert bdd.root_id == bdd.true_id

    def test_literal_false(self) -> None:
        """Literal(False) -> BDD whose root is the false terminal."""
        expr = logic.Literal(False)
        bdd = boolexpr_to_bdd(expr, num_vars=2)
        assert bdd.root_id == bdd.false_id

    def test_single_variable(self) -> None:
        """Variable(i) -> BDD with one internal node."""
        expr = logic.Variable(0)
        bdd = boolexpr_to_bdd(expr, num_vars=2)
        # Root is an internal node (not a terminal)
        root = bdd.nodes[bdd.root_id]
        assert root.var_index == 0
        assert root.low_id == bdd.false_id
        assert root.high_id == bdd.true_id

    def test_and_expression(self) -> None:
        """And(x0, x1) produces a valid BDD."""
        expr = logic.And((logic.Variable(0), logic.Variable(1)))
        bdd = boolexpr_to_bdd(expr, num_vars=2)
        # Root must be an internal node
        root = bdd.nodes[bdd.root_id]
        assert root.var_index is not None

    def test_or_expression(self) -> None:
        """Or(x0, x1) produces a valid BDD."""
        expr = logic.Or((logic.Variable(0), logic.Variable(1)))
        bdd = boolexpr_to_bdd(expr, num_vars=2)
        root = bdd.nodes[bdd.root_id]
        assert root.var_index is not None

    def test_not_raises(self) -> None:
        """Not operator must raise ValueError."""
        expr = logic.Not(logic.Variable(0))
        with pytest.raises(ValueError, match="Not operator"):
            boolexpr_to_bdd(expr, num_vars=2)

    def test_out_of_range_state_raises(self) -> None:
        """Variable index outside 0..num_vars-1 must raise ValueError."""
        expr = logic.Variable(5)
        with pytest.raises(ValueError, match="Invalid state variable"):
            boolexpr_to_bdd(expr, num_vars=3)

    def test_invalid_var_order_raises(self) -> None:
        """var_order that is not a permutation of range(num_vars) raises."""
        expr = logic.Variable(0)
        with pytest.raises(ValueError, match="permutation"):
            boolexpr_to_bdd(expr, num_vars=3, var_order=[0, 0, 1])

    def test_topo_order_children_before_parents(self) -> None:
        """Every node in topo_order has its children listed before it."""
        expr = typing.cast(
            logic.BoolExpr[int],
            (logic.Variable(0) & logic.Variable(1)) | logic.Variable(2),
        )
        bdd = boolexpr_to_bdd(expr, num_vars=3)
        seen: set[int] = set()
        for node_id in bdd.topo_order:
            node = bdd.nodes[node_id]
            if node.low_id is not None:
                assert node.low_id in seen, "low child must appear before parent"
            if node.high_id is not None:
                assert node.high_id in seen, "high child must appear before parent"
            seen.add(node_id)

    def test_structural_sharing(self) -> None:
        """BDD has fewer internal nodes than raw AST subtree count for shared expr."""
        # (a & b) | (a & c) | (a & b & d)  — 'a' appears in every branch
        a, b, c, d = (logic.Variable(i) for i in range(4))
        expr = typing.cast(
            logic.BoolExpr[int],
            (a & b) | (a & c) | (a & b & d),
        )
        bdd = boolexpr_to_bdd(expr, num_vars=4)
        # Raw AST has >4 variable occurrences; the ROBDD should have at most 4
        # internal nodes (one per variable in the worst case).
        internal_nodes = [n for n in bdd.nodes if n.var_index is not None]
        assert len(internal_nodes) <= 4


# ---------------------------------------------------------------------------
# TestBddToRankDecomposition
# ---------------------------------------------------------------------------


class TestBddToRankDecomposition:
    """Unit tests for bdd_to_rank_decomposition."""

    @pytest.fixture
    def algebra(self) -> algebraic.BooleanAlgebra:
        return boolean_algebra()

    @pytest.mark.parametrize(
        "expr",
        [
            logic.Literal(True),
            logic.Literal(False),
            logic.Variable(0),
            logic.Variable(2),
            typing.cast(logic.BoolExpr[int], logic.And((logic.Variable(0), logic.Variable(1)))),
            typing.cast(logic.BoolExpr[int], logic.Or((logic.Variable(0), logic.Variable(2)))),
            typing.cast(
                logic.BoolExpr[int],
                (logic.Variable(0) & logic.Variable(1)) | logic.Variable(2),
            ),
        ],
    )
    def test_equivalence_with_boolexpr_to_polynomial(
        self,
        expr: logic.BoolExpr[int],
        algebra: algebraic.BooleanAlgebra,
    ) -> None:
        """bdd_to_rank_decomposition must match boolexpr_to_polynomial on all points."""
        num_vars = 4
        bdd = boolexpr_to_bdd(expr, num_vars=num_vars)
        rd_bdd = bdd_to_rank_decomposition(bdd, num_vars, algebra, backend="numpy")
        rd_direct = boolexpr_to_polynomial(expr, num_vars, algebra, backend="numpy")
        assert_poly_equivalent(rd_bdd, rd_direct, algebra, num_vars)


# ---------------------------------------------------------------------------
# TestBoolExprToSymbolicPolynomial
# ---------------------------------------------------------------------------


class TestBoolExprToSymbolicPolynomial:
    """Tests for the boolexpr_to_symbolic_polynomial pipeline."""

    @pytest.fixture
    def algebra(self) -> algebraic.BooleanAlgebra:
        return boolean_algebra()

    @pytest.mark.parametrize("num_vars", [2, 3, 4, 5])
    def test_equivalence_small_formulas(
        self,
        num_vars: int,
        algebra: algebraic.BooleanAlgebra,
    ) -> None:
        """Result equals boolexpr_to_polynomial on all boolean assignments."""
        x = [logic.Variable(i) for i in range(num_vars)]
        expr = typing.cast(
            logic.BoolExpr[int],
            x[0] | (x[1] & x[num_vars - 1]),
        )
        symbolic = boolexpr_to_symbolic_polynomial(expr, num_vars, algebra, backend="numpy")
        direct = boolexpr_to_polynomial(expr, num_vars, algebra, backend="numpy")
        assert isinstance(symbolic, RankDecomposition)
        assert_poly_equivalent(symbolic, direct, algebra, num_vars)

    def test_output_mode_low_rank_factors(self, algebra: algebraic.BooleanAlgebra) -> None:
        """output='low_rank_factors' returns a LowRankFactors instance."""
        expr = typing.cast(
            logic.BoolExpr[int],
            logic.And((logic.Variable(0), logic.Variable(1))),
        )
        result = boolexpr_to_symbolic_polynomial(expr, num_vars=3, algebra=algebra, backend="numpy", output="low_rank_factors")
        assert isinstance(result, LowRankFactors)

    def test_output_mode_equivalence(self, algebra: algebraic.BooleanAlgebra) -> None:
        """RankDecomposition and LowRankFactors evaluate identically."""
        num_vars = 3
        expr = typing.cast(
            logic.BoolExpr[int],
            (logic.Variable(0) & logic.Variable(1)) | logic.Variable(2),
        )
        rd = boolexpr_to_symbolic_polynomial(
            expr, num_vars=num_vars, algebra=algebra, backend="numpy", output="rank_decomposition"
        )
        lrf = boolexpr_to_symbolic_polynomial(
            expr, num_vars=num_vars, algebra=algebra, backend="numpy", output="low_rank_factors"
        )
        assert isinstance(rd, RankDecomposition)
        assert isinstance(lrf, LowRankFactors)

        for bits in itertools.product([False, True], repeat=num_vars):
            point = np.array([algebra.one if b else algebra.zero for b in bits])
            rd_val = np.asarray(rd.evaluate(point).factors.data[0, 0, 0]).flat[0]
            lrf_val = np.asarray(lrf.evaluate(point).to_rank_decomposition().factors.data[0, 0, 0]).flat[0]
            np.testing.assert_allclose(rd_val, lrf_val, err_msg=f"Mismatch at {bits}")

    def test_custom_var_order(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Custom var_order still produces a polynomial equivalent to the direct path."""
        num_vars = 3
        expr = typing.cast(
            logic.BoolExpr[int],
            logic.Or((logic.Variable(0), logic.Variable(2))),
        )
        rd_natural = boolexpr_to_symbolic_polynomial(expr, num_vars=num_vars, algebra=algebra, backend="numpy")
        rd_reversed = boolexpr_to_symbolic_polynomial(
            expr, num_vars=num_vars, algebra=algebra, backend="numpy", var_order=[2, 1, 0]
        )
        assert isinstance(rd_natural, RankDecomposition)
        assert isinstance(rd_reversed, RankDecomposition)
        assert_poly_equivalent(rd_natural, rd_reversed, algebra, num_vars)


# ---------------------------------------------------------------------------
# TestSymbolicPolynomialOperatorFromAfa
# ---------------------------------------------------------------------------


class TestSymbolicPolynomialOperatorFromAfa:
    """Test SymbolicPolynomialOperator.from_afa construction and error paths."""

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
        """Build a minimal AFA for testing."""
        states = frozenset(transitions.keys())
        domain: SimpleDomain = SimpleDomain(_states=states, _symbols=symbols)
        delta = morphata.AlternatingTransitionRelation(
            {s: {sym: expr for sym, expr in sym_map.items()} for s, sym_map in transitions.items()}
        )
        acceptance = Finite(frozenset(accepting))
        return morphata.Automaton(
            domain=domain,
            initial=initial,
            delta=delta,
            acceptance=acceptance,
        )

    def test_basic_two_state_afa(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Constructs successfully for a minimal 2-state AFA."""
        transitions: dict[int, dict[str, BoolExpr[int]]] = {
            0: {"a": logic.Variable(1), "b": logic.Literal(False)},
            1: {"a": logic.Literal(True), "b": logic.Literal(False)},
        }
        aut = self._make_simple_afa(transitions, frozenset({1}), ("a", "b"))
        op = SymbolicPolynomialOperator.from_afa(aut, algebra, backend="numpy")
        assert op.num_states == 2
        assert op.accepting_states == frozenset({1})
        assert op.algebra is algebra

    def test_non_alternating_transitions_raises(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Non-alternating automaton raises TypeError."""
        from morphata.examples.nfa import NFA

        nfa: NFA[str] = NFA()
        nfa.add_location(0, initial=True)
        nfa.add_location(1, final=True)
        nfa.add_transition(0, 1, guard=logic.Variable("a"))
        aut = nfa.to_automaton()
        with pytest.raises(TypeError, match="AlternatingTransitions"):
            SymbolicPolynomialOperator.from_afa(aut, algebra, backend="numpy")

    def test_non_finite_acceptance_raises(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Non-Finite acceptance condition raises NotImplementedError."""
        states = frozenset({0})
        domain: SimpleDomain = SimpleDomain(_states=states, _symbols=("a",))
        delta = morphata.AlternatingTransitionRelation({0: {"a": logic.Variable(0)}})
        acceptance = Buchi(frozenset({0}))
        aut = morphata.Automaton(domain=domain, initial=0, delta=delta, acceptance=acceptance)
        with pytest.raises(NotImplementedError, match="Finite"):
            SymbolicPolynomialOperator.from_afa(aut, algebra, backend="numpy")

    def test_cache_transitions_false(self, algebra: algebraic.BooleanAlgebra) -> None:
        """cache_transitions=False yields an empty transition cache."""
        transitions: dict[int, dict[str, BoolExpr[int]]] = {
            0: {"a": logic.Literal(True)},
        }
        aut = self._make_simple_afa(transitions, frozenset({0}), ("a",))
        op = SymbolicPolynomialOperator.from_afa(aut, algebra, backend="numpy", cache_transitions=False)
        assert len(op._transition_cache) == 0

    def test_output_low_rank_factors(self, algebra: algebraic.BooleanAlgebra) -> None:
        """output='low_rank_factors' stores LowRankFactors in initial_poly."""
        transitions: dict[int, dict[str, BoolExpr[int]]] = {
            0: {"a": logic.Literal(True)},
        }
        aut = self._make_simple_afa(transitions, frozenset({0}), ("a",))
        op = SymbolicPolynomialOperator.from_afa(aut, algebra, backend="numpy", output="low_rank_factors")
        assert isinstance(op.initial_poly, LowRankFactors)

    def test_from_ltl_constructs(self, algebra: algebraic.BooleanAlgebra) -> None:
        """from_ltl convenience constructor works end-to-end."""
        formula = ltl.Eventually(ltl.Variable("a"))
        op = SymbolicPolynomialOperator.from_ltl(formula, algebra, backend="numpy", finite=True)
        assert op.num_states >= 1
        assert op.algebra is algebra


# ---------------------------------------------------------------------------
# TestSymbolicPolynomialOperatorAccepts
# ---------------------------------------------------------------------------


class TestSymbolicPolynomialOperatorAccepts:
    """Operator-level equivalence: SymbolicPolynomialOperator vs PolynomialOperator."""

    @pytest.fixture
    def algebra(self) -> algebraic.BooleanAlgebra:
        return boolean_algebra()

    @pytest.mark.parametrize(
        "formula, word, expected",
        [
            # F(a): eventually a
            (ltl.Eventually(ltl.Variable("a")), [sym(), sym("a")], True),
            (ltl.Eventually(ltl.Variable("a")), [sym(), sym()], False),
            # G(a): always a
            (ltl.Always(ltl.Variable("a")), [sym("a"), sym("a")], True),
            (ltl.Always(ltl.Variable("a")), [sym("a"), sym()], False),
            # X(a): next a
            (ltl.Next(ltl.Variable("a")), [sym(), sym("a")], True),
            (ltl.Next(ltl.Variable("a")), [sym("a"), sym()], False),
        ],
    )
    def test_accepts_matches_polynomial_operator(
        self,
        formula: ltl.LTLExpr[str],
        word: list[frozenset[str]],
        expected: bool,
        algebra: algebraic.BooleanAlgebra,
    ) -> None:
        """SymbolicPolynomialOperator.accepts agrees with PolynomialOperator."""
        from automatix.operators.polynomial import PolynomialOperator

        sym_op = SymbolicPolynomialOperator.from_ltl(formula, algebra, backend="numpy", finite=True)
        poly_op = PolynomialOperator.from_ltl(formula, algebra, backend="numpy", finite=True)

        sym_result = sym_op.accepts(word)
        poly_result = poly_op.accepts(word)

        expected_val = algebra.one if expected else algebra.zero
        assert_close(sym_result, expected_val)
        assert_close(poly_result, expected_val)
        # Both must agree
        assert_close(sym_result, poly_result)

    def test_run_polynomial_type(self, algebra: algebraic.BooleanAlgebra) -> None:
        """run_polynomial returns RankDecomposition by default."""
        formula = ltl.Eventually(ltl.Variable("a"))
        op = SymbolicPolynomialOperator.from_ltl(formula, algebra, backend="numpy", finite=True)
        word = [sym(), sym("a")]
        result = op.run_polynomial(word)
        assert isinstance(result, RankDecomposition)

    def test_run_polynomial_type_low_rank(self, algebra: algebraic.BooleanAlgebra) -> None:
        """run_polynomial returns LowRankFactors when output='low_rank_factors'."""
        formula = ltl.Eventually(ltl.Variable("a"))
        op = SymbolicPolynomialOperator.from_ltl(formula, algebra, backend="numpy", finite=True, output="low_rank_factors")
        word = [sym(), sym("a")]
        result = op.run_polynomial(word)
        assert isinstance(result, LowRankFactors)

    def test_accepts_low_rank_matches_rank_decomposition(self, algebra: algebraic.BooleanAlgebra) -> None:
        """Both output modes accept/reject identically."""
        formula = ltl.Until(ltl.Variable("a"), ltl.Variable("b"))
        op_rd = SymbolicPolynomialOperator.from_ltl(
            formula, algebra, backend="numpy", finite=True, output="rank_decomposition"
        )
        op_lrf = SymbolicPolynomialOperator.from_ltl(formula, algebra, backend="numpy", finite=True, output="low_rank_factors")
        words: list[list[frozenset[str]]] = [
            [sym("a"), sym("b")],
            [sym("b")],
            [sym("a"), sym("a")],
        ]
        for word in words:
            assert_close(op_rd.accepts(word), op_lrf.accepts(word))


class TestSymbolicPolynomialOperatorStep:
    """Regression tests for step() degree-blowup bug (fixed 2026-04).

    Before the fix, ``SymbolicPolynomialOperator.step()`` called
    ``compose()`` which could grow the CP degree to ``1 + D_old * D_trans``.
    On step 2 this reached degree 10 for the ``F(a)&F(b)`` formula, causing
    ``normalize()`` (via ``to_sparse``) to loop ``4^10 ≈ 1 M`` times and hang.
    """

    @pytest.fixture
    def algebra(self) -> algebraic.BooleanAlgebra:
        return boolean_algebra()

    @pytest.mark.parametrize("output", ["rank_decomposition", "low_rank_factors"])
    @pytest.mark.timeout(10)
    def test_step_two_eventualities_terminates(
        self,
        output: str,
        algebra: algebraic.BooleanAlgebra,
    ) -> None:
        """step() must terminate for F(a)&F(b) across at least two steps.

        This is the minimal reproducer from tests/repro_step_hang.py.
        Both output modes exercise the same to_sparse path via normalize().

        Regression: the second step inflates the CP degree to 10, triggering
        normalize() -> to_sparse() with 4^10 ≈ 1 M Python iterations (~100 s).
        The timeout catches that regression within 10 s.
        """
        formula = ltl.Eventually(ltl.Variable("a")) & ltl.Eventually(ltl.Variable("b"))
        op = SymbolicPolynomialOperator.from_ltl(
            formula,
            algebra,
            backend="numpy",
            finite=True,
            output=output,  # type: ignore[arg-type]
        )
        sigma = frozenset()  # empty symbol — the case that previously hung

        poly1 = op.step(op.initial_poly, sigma)
        poly2 = op.step(poly1, sigma)  # this was the hanging call

        # Both results must have bounded degree
        assert poly1.degree <= poly1.max_degree
        assert poly2.degree <= poly2.max_degree

    @pytest.mark.timeout(10)
    def test_step_second_level_agrees_with_run_polynomial(
        self,
        algebra: algebraic.BooleanAlgebra,
    ) -> None:
        """step() called twice must agree with run_polynomial on a 2-symbol word.

        Regression: same degree-10 path as test_step_two_eventualities_terminates.
        """
        formula = ltl.Eventually(ltl.Variable("a")) & ltl.Eventually(ltl.Variable("b"))
        op = SymbolicPolynomialOperator.from_ltl(
            formula, algebra, backend="numpy", finite=True
        )
        sigma = frozenset()

        # Manual two-step
        poly_via_step = op.step(op.step(op.initial_poly, sigma), sigma)

        # run_polynomial on a 2-symbol word
        poly_via_run = op.run_polynomial([sigma, sigma])

        # Both must evaluate identically at all boolean points
        num_vars = op.num_states
        for bits in itertools.product([False, True], repeat=num_vars):
            point = np.array([algebra.one if b else algebra.zero for b in bits])
            step_val = np.asarray(poly_via_step.evaluate(point).factors.data[0, 0, 0]).flat[0]
            run_val = np.asarray(poly_via_run.evaluate(point).factors.data[0, 0, 0]).flat[0]
            np.testing.assert_allclose(step_val, run_val, err_msg=f"Mismatch at {bits}")
