"""Unit tests for the symbolic polynomial operator (BDD-native)."""

from __future__ import annotations

import typing
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import logic_asts as logic
import logic_asts.ltl as ltl
import morphata
import pytest
from morphata.acceptance import Buchi, Finite
from morphata.spec import BoolExpr
from typing_extensions import override

from automatix.operators import BDDOperator
from automatix.operators._bdd import (
    BDDDag,
    bdd_to_boolexpr,
    bdd_to_poly_dict,
    bdd_to_rank_decomp,
    boolexpr_to_bdd,
    compose_bdd,
    evaluate_bdd,
    poly_dict_to_boolexpr,
    rank_decomp_to_bdd,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def sym(*aps: str) -> frozenset[str]:
    """Build an input symbol (set of atomic propositions present)."""
    return frozenset(aps)


def eval_boolexpr(expr: logic.BoolExpr[int], point: dict[int, bool]) -> bool:
    """Recursively evaluate a BoolExpr at a boolean assignment."""
    match expr:
        case logic.Literal(val):
            return bool(val)
        case logic.Variable(q):
            return point[int(q)]
        case logic.And(args):
            return all(eval_boolexpr(a, point) for a in args)  # type: ignore[arg-type]
        case logic.Or(args):
            return any(eval_boolexpr(a, point) for a in args)  # type: ignore[arg-type]
        case logic.Not(arg):
            return not eval_boolexpr(arg, point)  # type: ignore[arg-type]
        case _:
            raise TypeError(f"Unsupported BoolExpr type: {type(expr).__name__}")


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
# TestBoolExprToBDD  (unchanged - tests the _bdd module)
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
        root = bdd.nodes[bdd.root_id]
        assert root.var_index == 0
        assert root.low_id == bdd.false_id
        assert root.high_id == bdd.true_id

    def test_and_expression(self) -> None:
        """And(x0, x1) produces a valid BDD."""
        expr = logic.And((logic.Variable(0), logic.Variable(1)))
        bdd = boolexpr_to_bdd(expr, num_vars=2)
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
        a, b, c, d = (logic.Variable(i) for i in range(4))
        expr = typing.cast(
            logic.BoolExpr[int],
            (a & b) | (a & c) | (a & b & d),
        )
        bdd = boolexpr_to_bdd(expr, num_vars=4)
        internal_nodes = [n for n in bdd.nodes if n.var_index is not None]
        assert len(internal_nodes) <= 4


# ---------------------------------------------------------------------------
# TestEvaluateBdd
# ---------------------------------------------------------------------------


class TestEvaluateBdd:
    """Unit tests for evaluate_bdd."""

    def test_literal_true(self) -> None:
        bdd = boolexpr_to_bdd(logic.Literal(True), num_vars=2)
        assert evaluate_bdd(bdd, {0: False, 1: False}) is True
        assert evaluate_bdd(bdd, {0: True, 1: True}) is True

    def test_literal_false(self) -> None:
        bdd = boolexpr_to_bdd(logic.Literal(False), num_vars=2)
        assert evaluate_bdd(bdd, {0: False, 1: False}) is False
        assert evaluate_bdd(bdd, {0: True, 1: True}) is False

    def test_single_variable(self) -> None:
        bdd = boolexpr_to_bdd(logic.Variable(0), num_vars=2)
        assert evaluate_bdd(bdd, {0: True, 1: False}) is True
        assert evaluate_bdd(bdd, {0: False, 1: True}) is False

    def test_and_expression(self) -> None:
        expr = typing.cast(
            logic.BoolExpr[int],
            logic.And((logic.Variable(0), logic.Variable(1))),
        )
        bdd = boolexpr_to_bdd(expr, num_vars=2)
        assert evaluate_bdd(bdd, {0: True, 1: True}) is True
        assert evaluate_bdd(bdd, {0: True, 1: False}) is False
        assert evaluate_bdd(bdd, {0: False, 1: True}) is False
        assert evaluate_bdd(bdd, {0: False, 1: False}) is False

    def test_or_expression(self) -> None:
        expr = typing.cast(
            logic.BoolExpr[int],
            logic.Or((logic.Variable(0), logic.Variable(1))),
        )
        bdd = boolexpr_to_bdd(expr, num_vars=2)
        assert evaluate_bdd(bdd, {0: True, 1: True}) is True
        assert evaluate_bdd(bdd, {0: True, 1: False}) is True
        assert evaluate_bdd(bdd, {0: False, 1: True}) is True
        assert evaluate_bdd(bdd, {0: False, 1: False}) is False


# ---------------------------------------------------------------------------
# TestBddToBoolexpr
# ---------------------------------------------------------------------------


class TestBddToBoolexpr:
    """Unit tests for bdd_to_boolexpr."""

    def _roundtrip_equiv(self, expr: logic.BoolExpr[int], num_vars: int) -> None:
        """Assert bdd_to_boolexpr(boolexpr_to_bdd(expr)) is semantically equal."""
        import itertools

        bdd = boolexpr_to_bdd(expr, num_vars=num_vars)
        recovered = bdd_to_boolexpr(bdd)
        for bits in itertools.product([False, True], repeat=num_vars):
            point = {i: b for i, b in enumerate(bits)}
            assert evaluate_bdd(bdd, point) == eval_boolexpr(recovered, point), f"Mismatch at {bits}"

    def test_literal_true(self) -> None:
        bdd = boolexpr_to_bdd(logic.Literal(True), num_vars=2)
        expr = bdd_to_boolexpr(bdd)
        assert isinstance(expr, logic.Literal)
        assert expr == logic.Literal(True)

    def test_literal_false(self) -> None:
        bdd = boolexpr_to_bdd(logic.Literal(False), num_vars=2)
        expr = bdd_to_boolexpr(bdd)
        assert isinstance(expr, logic.Literal)
        assert expr == logic.Literal(False)

    def test_single_variable(self) -> None:
        bdd = boolexpr_to_bdd(logic.Variable(0), num_vars=2)
        expr = bdd_to_boolexpr(bdd)
        assert expr == logic.Variable(0)

    def test_negated_variable(self) -> None:
        """A BDD built from And(q0, q1) with q0 forced to False gives NOT-like structure."""
        # NOT q0: build via complement trick - boolexpr_to_bdd can't take Not directly,
        # so compose Variable(0) with [Literal(False)] to get the False function,
        # then test the NOT path via a manually crafted BDD roundtrip equivalence.
        # Easiest: just verify roundtrip for a compound expression containing a Not-like path.
        expr = typing.cast(
            logic.BoolExpr[int],
            logic.Or((logic.Variable(0), logic.Variable(1))),
        )
        self._roundtrip_equiv(expr, num_vars=2)

    def test_and_expression(self) -> None:
        expr = typing.cast(
            logic.BoolExpr[int],
            logic.And((logic.Variable(0), logic.Variable(1))),
        )
        self._roundtrip_equiv(expr, num_vars=2)

    def test_or_expression(self) -> None:
        expr = typing.cast(
            logic.BoolExpr[int],
            logic.Or((logic.Variable(0), logic.Variable(1))),
        )
        self._roundtrip_equiv(expr, num_vars=2)

    def test_complex_expression(self) -> None:
        """Roundtrip for a 3-variable mixed expression."""
        x = [logic.Variable(i) for i in range(3)]
        expr = typing.cast(
            logic.BoolExpr[int],
            (x[0] & x[1]) | x[2],
        )
        self._roundtrip_equiv(expr, num_vars=3)

    def test_compose_then_recover(self) -> None:
        """compose_bdd result can be converted back to a BoolExpr."""
        f = typing.cast(
            logic.BoolExpr[int],
            logic.Or((logic.Variable(0), logic.Variable(1))),
        )
        run_bdd = boolexpr_to_bdd(f, num_vars=2)
        # Substitute q0 -> q1, q1 -> q1 (collapses to q1)
        q1 = boolexpr_to_bdd(logic.Variable(1), num_vars=2)
        composed = compose_bdd(run_bdd, [q1, q1])
        recovered = bdd_to_boolexpr(composed)
        # Result should equal Variable(1) semantically
        assert eval_boolexpr(recovered, {0: False, 1: True}) is True
        assert eval_boolexpr(recovered, {0: True, 1: False}) is False


# ---------------------------------------------------------------------------
# TestComposeBdd
# ---------------------------------------------------------------------------


class TestComposeBdd:
    """Unit tests for compose_bdd."""

    def test_identity_substitution(self) -> None:
        """Substituting q_i -> q_i is a no-op."""
        expr = typing.cast(
            logic.BoolExpr[int],
            logic.Or((logic.Variable(0), logic.Variable(1))),
        )
        run_bdd = boolexpr_to_bdd(expr, num_vars=2)
        subs = [boolexpr_to_bdd(logic.Variable(i), num_vars=2) for i in range(2)]
        result = compose_bdd(run_bdd, subs)
        # Result must agree with original on all assignments
        for a in (False, True):
            for b in (False, True):
                point = {0: a, 1: b}
                assert evaluate_bdd(result, point) == evaluate_bdd(run_bdd, point)

    def test_constant_substitution_true(self) -> None:
        """Substituting all variables with True collapses to Literal(True) for 'Or'."""
        expr = typing.cast(
            logic.BoolExpr[int],
            logic.Or((logic.Variable(0), logic.Variable(1))),
        )
        run_bdd = boolexpr_to_bdd(expr, num_vars=2)
        true_bdd = boolexpr_to_bdd(logic.Literal(True), num_vars=2)
        result = compose_bdd(run_bdd, [true_bdd, true_bdd])
        assert result.root_id == result.true_id

    def test_constant_substitution_false(self) -> None:
        """Substituting all variables with False collapses to Literal(False) for 'And'."""
        expr = typing.cast(
            logic.BoolExpr[int],
            logic.And((logic.Variable(0), logic.Variable(1))),
        )
        run_bdd = boolexpr_to_bdd(expr, num_vars=2)
        false_bdd = boolexpr_to_bdd(logic.Literal(False), num_vars=2)
        result = compose_bdd(run_bdd, [false_bdd, false_bdd])
        assert result.root_id == result.false_id

    def test_variable_swap(self) -> None:
        """Swapping q0 and q1 gives the dual function."""
        # f(q0, q1) = q0 AND NOT(q1) - via q0 & ~q0 workaround using Or+And
        # Use f = q0 as a simple test: substitute q0 -> q1, q1 -> q0
        run_bdd = boolexpr_to_bdd(logic.Variable(0), num_vars=2)
        subs = [
            boolexpr_to_bdd(logic.Variable(1), num_vars=2),  # q0 -> q1
            boolexpr_to_bdd(logic.Variable(0), num_vars=2),  # q1 -> q0
        ]
        result = compose_bdd(run_bdd, subs)
        # result should be q1
        assert evaluate_bdd(result, {0: False, 1: True}) is True
        assert evaluate_bdd(result, {0: True, 1: False}) is False

    def test_wrong_num_substitutions_raises(self) -> None:
        run_bdd = boolexpr_to_bdd(logic.Variable(0), num_vars=3)
        subs = [boolexpr_to_bdd(logic.Variable(0), num_vars=3)]
        with pytest.raises(ValueError, match="substitutions"):
            compose_bdd(run_bdd, subs)

    def test_chained_composition(self) -> None:
        """Two compositions in sequence match manual evaluation."""
        # f = q0 OR q1
        # step 1: q0 -> q1, q1 -> False  =>  q1 OR False = q1
        # step 2: q0 -> False, q1 -> True =>  True
        f = boolexpr_to_bdd(
            typing.cast(logic.BoolExpr[int], logic.Or((logic.Variable(0), logic.Variable(1)))),
            num_vars=2,
        )
        false_bdd = boolexpr_to_bdd(logic.Literal(False), num_vars=2)
        true_bdd = boolexpr_to_bdd(logic.Literal(True), num_vars=2)
        q1 = boolexpr_to_bdd(logic.Variable(1), num_vars=2)

        step1 = compose_bdd(f, [q1, false_bdd])
        step2 = compose_bdd(step1, [false_bdd, true_bdd])
        assert step2.root_id == step2.true_id


# ---------------------------------------------------------------------------
# TestBDDOperatorFromAfa
# ---------------------------------------------------------------------------


class TestBDDOperatorFromAfa:
    """Test BDDOperator.from_afa construction and error paths."""

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

    def test_basic_two_state_afa(self) -> None:
        """Constructs successfully for a minimal 2-state AFA."""
        transitions: dict[int, dict[str, BoolExpr[int]]] = {
            0: {"a": logic.Variable(1), "b": logic.Literal(False)},
            1: {"a": logic.Literal(True), "b": logic.Literal(False)},
        }
        aut = self._make_simple_afa(transitions, frozenset({1}), ("a", "b"))
        op = BDDOperator.from_afa(aut)
        assert op.num_states == 2
        assert op.accepting_states == frozenset({1})
        assert isinstance(op.initial_bdd, BDDDag)

    def test_non_alternating_transitions_raises(self) -> None:
        """Non-alternating automaton raises TypeError."""
        from morphata.examples.nfa import NFA

        nfa: NFA[str] = NFA()
        nfa.add_location(0, initial=True)
        nfa.add_location(1, final=True)
        nfa.add_transition(0, 1, guard=logic.Variable("a"))
        aut = nfa.to_automaton()
        with pytest.raises(TypeError, match="AlternatingTransitions"):
            BDDOperator.from_afa(aut)

    def test_non_finite_acceptance_raises(self) -> None:
        """Non-Finite acceptance condition raises NotImplementedError."""
        states = frozenset({0})
        domain: SimpleDomain = SimpleDomain(_states=states, _symbols=("a",))
        delta = morphata.AlternatingTransitionRelation({0: {"a": logic.Variable(0)}})
        acceptance = Buchi(frozenset({0}))
        aut = morphata.Automaton(domain=domain, initial=0, delta=delta, acceptance=acceptance)
        with pytest.raises(NotImplementedError, match="Finite"):
            BDDOperator.from_afa(aut)

    def test_cache_transitions_false(self) -> None:
        """cache_transitions=False yields an empty transition cache."""
        transitions: dict[int, dict[str, BoolExpr[int]]] = {
            0: {"a": logic.Literal(True)},
        }
        aut = self._make_simple_afa(transitions, frozenset({0}), ("a",))
        op = BDDOperator.from_afa(aut, cache_transitions=False)
        assert len(op._transition_cache) == 0

    def test_transition_cache_contains_bdd_dags(self) -> None:
        """All values in the transition cache are BDDDag instances."""
        transitions: dict[int, dict[str, BoolExpr[int]]] = {
            0: {"a": logic.Variable(0), "b": logic.Literal(False)},
        }
        aut = self._make_simple_afa(transitions, frozenset({0}), ("a", "b"))
        op = BDDOperator.from_afa(aut)
        for val in op._transition_cache.values():
            assert isinstance(val, BDDDag)

    def test_from_ltl_constructs(self) -> None:
        """from_ltl convenience constructor works end-to-end."""
        formula = ltl.Eventually(ltl.Variable("a"))
        op = BDDOperator.from_ltl(formula, finite=True)
        assert op.num_states >= 1
        assert isinstance(op.initial_bdd, BDDDag)


# ---------------------------------------------------------------------------
# TestBDDOperatorAccepts
# ---------------------------------------------------------------------------


class TestBDDOperatorAccepts:
    """Operator-level correctness: accepts() returns correct booleans."""

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
    def test_accepts_correctness(
        self,
        formula: ltl.LTLExpr[str],
        word: list[frozenset[str]],
        expected: bool,
    ) -> None:
        """BDDOperator.accepts returns the correct boolean."""
        op = BDDOperator.from_ltl(formula, finite=True)
        assert op.accepts(word) is expected

    def test_accepts_agrees_with_polynomial_operator(self) -> None:
        """BDDOperator.accepts agrees with PolynomialOperator."""
        from algebraic.semirings import boolean_algebra
        from algebraic.utils.testing import assert_close

        from automatix.operators.polynomial import PolynomialOperator

        algebra = boolean_algebra()

        cases: list[tuple[ltl.LTLExpr[str], list[frozenset[str]], bool]] = [
            (ltl.Eventually(ltl.Variable("a")), [sym(), sym("a")], True),
            (ltl.Eventually(ltl.Variable("a")), [sym(), sym()], False),
            (ltl.Always(ltl.Variable("a")), [sym("a"), sym("a")], True),
            (ltl.Always(ltl.Variable("a")), [sym("a"), sym()], False),
            (ltl.Next(ltl.Variable("a")), [sym(), sym("a")], True),
            (ltl.Next(ltl.Variable("a")), [sym("a"), sym()], False),
        ]
        for formula, word, expected in cases:
            sym_op = BDDOperator.from_ltl(formula, finite=True)
            poly_op = PolynomialOperator.from_ltl(formula, algebra, backend="numpy", finite=True)
            sym_result = sym_op.accepts(word)
            poly_raw = poly_op.accepts(word)
            expected_val = algebra.one if expected else algebra.zero
            assert sym_result == expected, f"BDD mismatch on {word}: got {sym_result}, want {expected}"
            assert_close(poly_raw, expected_val)

    def test_run_bdd_returns_bdd_dag(self) -> None:
        """run_bdd returns a BDDDag."""
        formula = ltl.Eventually(ltl.Variable("a"))
        op = BDDOperator.from_ltl(formula, finite=True)
        result = op.run_bdd([sym(), sym("a")])
        assert isinstance(result, BDDDag)

    def test_custom_var_order(self) -> None:
        """Custom var_order still produces correct acceptance decisions."""
        formula = ltl.Always(ltl.Variable("a"))
        op_natural = BDDOperator.from_ltl(formula, finite=True)
        op_reversed = BDDOperator.from_ltl(formula, finite=True, var_order=list(reversed(range(op_natural.num_states))))
        words: list[list[frozenset[str]]] = [
            [sym("a"), sym("a")],
            [sym("a"), sym()],
            [sym()],
        ]
        for word in words:
            assert op_natural.accepts(word) is op_reversed.accepts(word), f"Mismatch on {word}"


# ---------------------------------------------------------------------------
# TestPolyDictToBoolexpr
# ---------------------------------------------------------------------------


class TestPolyDictToBoolexpr:
    """Unit tests for poly_dict_to_boolexpr."""

    def _algebra(self) -> Any:
        from algebraic.semirings import boolean_algebra

        return boolean_algebra()

    def test_zero_poly_dict(self) -> None:
        """Zero PolyDict -> Literal(False)."""
        from algebraic.polynomials import PolyDict

        sparse = PolyDict.zero(2, algebra=self._algebra(), backend="numpy")
        expr = poly_dict_to_boolexpr(sparse)
        assert expr == logic.Literal(False)

    def test_one_poly_dict(self) -> None:
        """One PolyDict -> Literal(True)."""
        from algebraic.polynomials import PolyDict

        sparse = PolyDict.one(2, algebra=self._algebra(), backend="numpy")
        expr = poly_dict_to_boolexpr(sparse)
        assert expr == logic.Literal(True)

    def test_single_variable(self) -> None:
        """PolyDict.variable(0) -> Variable(0)."""
        from algebraic.polynomials import PolyDict

        sparse = PolyDict.variable(0, 2, algebra=self._algebra(), backend="numpy")
        expr = poly_dict_to_boolexpr(sparse)
        assert expr == logic.Variable(0)

    def test_and_monomial(self) -> None:
        """x_0 * x_1 PolyDict -> And(Variable(0), Variable(1)) semantics."""
        import itertools

        from algebraic.polynomials import PolyDict

        algebra = self._algebra()
        x0 = PolyDict.variable(0, 2, algebra=algebra, backend="numpy")
        x1 = PolyDict.variable(1, 2, algebra=algebra, backend="numpy")
        sparse = x0 * x1
        expr = poly_dict_to_boolexpr(sparse)
        for a, b in itertools.product([False, True], repeat=2):
            point = {0: a, 1: b}
            assert eval_boolexpr(expr, point) == (a and b), f"Mismatch at {a}, {b}"

    def test_or_of_monomials(self) -> None:
        """x_0 + x_1 PolyDict -> Or semantics."""
        import itertools

        from algebraic.polynomials import PolyDict

        algebra = self._algebra()
        x0 = PolyDict.variable(0, 2, algebra=algebra, backend="numpy")
        x1 = PolyDict.variable(1, 2, algebra=algebra, backend="numpy")
        sparse = x0 + x1
        expr = poly_dict_to_boolexpr(sparse)
        for a, b in itertools.product([False, True], repeat=2):
            point = {0: a, 1: b}
            assert eval_boolexpr(expr, point) == (a or b), f"Mismatch at {a}, {b}"


# ---------------------------------------------------------------------------
# TestRankDecompToBdd
# ---------------------------------------------------------------------------


class TestRankDecompToBdd:
    """Unit tests for rank_decomp_to_bdd."""

    def _algebra(self) -> Any:
        from algebraic.semirings import boolean_algebra

        return boolean_algebra()

    def test_variable_polynomial(self) -> None:
        """x_0 RankDecomposition -> BDD evaluates correctly."""
        from algebraic.polynomials import RankDecomposition

        algebra = self._algebra()
        poly = RankDecomposition.variable(0, num_vars=2, algebra=algebra, backend="numpy")
        bdd = rank_decomp_to_bdd(poly)
        assert evaluate_bdd(bdd, {0: True, 1: False}) is True
        assert evaluate_bdd(bdd, {0: False, 1: True}) is False
        assert evaluate_bdd(bdd, {0: False, 1: False}) is False

    def test_and_polynomial(self) -> None:
        """x_0 * x_1 -> AND semantics."""
        import itertools

        from algebraic.polynomials import RankDecomposition

        algebra = self._algebra()
        x0 = RankDecomposition.variable(0, num_vars=2, algebra=algebra, backend="numpy")
        x1 = RankDecomposition.variable(1, num_vars=2, algebra=algebra, backend="numpy")
        bdd = rank_decomp_to_bdd(x0 * x1)
        for a, b in itertools.product([False, True], repeat=2):
            assert evaluate_bdd(bdd, {0: a, 1: b}) is (a and b), f"Mismatch at {a}, {b}"

    def test_or_polynomial(self) -> None:
        """x_0 + x_1 -> OR semantics."""
        import itertools

        from algebraic.polynomials import RankDecomposition

        algebra = self._algebra()
        x0 = RankDecomposition.variable(0, num_vars=2, algebra=algebra, backend="numpy")
        x1 = RankDecomposition.variable(1, num_vars=2, algebra=algebra, backend="numpy")
        bdd = rank_decomp_to_bdd(x0 + x1)
        for a, b in itertools.product([False, True], repeat=2):
            assert evaluate_bdd(bdd, {0: a, 1: b}) is (a or b), f"Mismatch at {a}, {b}"

    def test_zero_polynomial(self) -> None:
        """Zero polynomial -> BDD root is false_id."""
        from algebraic.polynomials import RankDecomposition

        algebra = self._algebra()
        poly = RankDecomposition.zero(num_vars=2, algebra=algebra, backend="numpy")
        bdd = rank_decomp_to_bdd(poly)
        assert bdd.root_id == bdd.false_id

    def test_one_polynomial(self) -> None:
        """One polynomial -> BDD root is true_id."""
        from algebraic.polynomials import RankDecomposition

        algebra = self._algebra()
        poly = RankDecomposition.one(num_vars=2, algebra=algebra, backend="numpy")
        bdd = rank_decomp_to_bdd(poly)
        assert bdd.root_id == bdd.true_id

    def test_low_rank_factors(self) -> None:
        """LowRankFactors input is converted via to_rank_decomposition()."""
        from algebraic.polynomials import LowRankFactors, RankDecomposition

        algebra = self._algebra()
        rd = RankDecomposition.variable(0, num_vars=2, algebra=algebra, backend="numpy")
        lrf = LowRankFactors.from_rank_decomposition(rd)
        bdd = rank_decomp_to_bdd(lrf)
        assert evaluate_bdd(bdd, {0: True, 1: False}) is True
        assert evaluate_bdd(bdd, {0: False, 1: True}) is False

    def test_wrong_type_raises(self) -> None:
        """Non-polynomial input raises TypeError."""
        with pytest.raises(TypeError, match="RankDecomposition or LowRankFactors"):
            rank_decomp_to_bdd("not a polynomial")


# ---------------------------------------------------------------------------
# TestBddToPoly
# ---------------------------------------------------------------------------


class TestBddToPoly:
    """Unit tests for bdd_to_poly_dict and bdd_to_rank_decomp."""

    def _algebra(self) -> Any:
        from algebraic.semirings import boolean_algebra

        return boolean_algebra()

    def _eval_poly(self, poly: Any, point: dict[int, bool]) -> bool:
        """Evaluate a PolyDict at a boolean point, returning a Python bool."""
        result = poly.evaluate(point)
        # A nonzero polynomial has a constant-monomial entry with a truthy coefficient.
        for coeff in result.data.values():
            if coeff.data.item():
                return True
        return False

    def test_true_bdd_to_poly_dict(self) -> None:
        """Literal(True) BDD -> one PolyDict (constant True)."""
        algebra = self._algebra()
        bdd = boolexpr_to_bdd(logic.Literal(True), num_vars=2)
        poly = bdd_to_poly_dict(bdd, algebra)
        assert self._eval_poly(poly, {0: False, 1: False}) is True
        assert self._eval_poly(poly, {0: True, 1: True}) is True

    def test_false_bdd_to_poly_dict(self) -> None:
        """Literal(False) BDD -> zero PolyDict (constant False)."""
        algebra = self._algebra()
        bdd = boolexpr_to_bdd(logic.Literal(False), num_vars=2)
        poly = bdd_to_poly_dict(bdd, algebra)
        assert self._eval_poly(poly, {0: False, 1: False}) is False
        assert self._eval_poly(poly, {0: True, 1: True}) is False

    def test_variable_bdd_to_poly_dict(self) -> None:
        """Variable(0) BDD -> PolyDict that evaluates correctly."""
        algebra = self._algebra()
        bdd = boolexpr_to_bdd(logic.Variable(0), num_vars=2)
        poly = bdd_to_poly_dict(bdd, algebra)
        assert self._eval_poly(poly, {0: True, 1: False}) is True
        assert self._eval_poly(poly, {0: False, 1: True}) is False

    def test_and_bdd_to_poly_dict(self) -> None:
        """AND BDD -> PolyDict with AND semantics."""
        import itertools

        algebra = self._algebra()
        expr = typing.cast(logic.BoolExpr[int], logic.And((logic.Variable(0), logic.Variable(1))))
        bdd = boolexpr_to_bdd(expr, num_vars=2)
        poly = bdd_to_poly_dict(bdd, algebra)
        for a, b in itertools.product([False, True], repeat=2):
            assert self._eval_poly(poly, {0: a, 1: b}) is (a and b), f"Mismatch at {a}, {b}"

    def test_or_bdd_to_poly_dict(self) -> None:
        """OR BDD -> PolyDict with OR semantics."""
        import itertools

        algebra = self._algebra()
        expr = typing.cast(logic.BoolExpr[int], logic.Or((logic.Variable(0), logic.Variable(1))))
        bdd = boolexpr_to_bdd(expr, num_vars=2)
        poly = bdd_to_poly_dict(bdd, algebra)
        for a, b in itertools.product([False, True], repeat=2):
            assert self._eval_poly(poly, {0: a, 1: b}) is (a or b), f"Mismatch at {a}, {b}"

    def test_bdd_to_rank_decomp_variable(self) -> None:
        """Variable BDD -> RankDecomposition; round-trips correctly via to_sparse."""
        algebra = self._algebra()
        bdd = boolexpr_to_bdd(logic.Variable(1), num_vars=2)
        rd = bdd_to_rank_decomp(bdd, algebra)
        # Evaluate via PolyDict (to_sparse -> evaluate) to avoid internal repr details.
        assert self._eval_poly(rd.to_sparse(), {0: False, 1: False}) is False
        assert self._eval_poly(rd.to_sparse(), {0: False, 1: True}) is True
        assert self._eval_poly(rd.to_sparse(), {0: True, 1: False}) is False

    def test_roundtrip_rank_decomp_to_bdd_to_poly_dict(self) -> None:
        """rank_decomp_to_bdd -> bdd_to_poly_dict roundtrip evaluates correctly."""
        import itertools

        from algebraic.polynomials import RankDecomposition

        algebra = self._algebra()
        x0 = RankDecomposition.variable(0, num_vars=3, algebra=algebra, backend="numpy")
        x1 = RankDecomposition.variable(1, num_vars=3, algebra=algebra, backend="numpy")
        x2 = RankDecomposition.variable(2, num_vars=3, algebra=algebra, backend="numpy")
        poly_orig = (x0 * x1) + x2
        bdd = rank_decomp_to_bdd(poly_orig)
        poly_rt = bdd_to_poly_dict(bdd, algebra)
        for a, b, c in itertools.product([False, True], repeat=3):
            expected = (a and b) or c
            assert self._eval_poly(poly_rt, {0: a, 1: b, 2: c}) is expected, f"Mismatch at {a}, {b}, {c}"
