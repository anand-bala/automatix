"""Polynomial-based operator for Alternating Finite Automata.

This module provides PolynomialOperator for representing AFA transitions
and runs as multilinear polynomials over boolean algebra.
"""

from __future__ import annotations

import typing
from collections.abc import Hashable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Generic, TypeVar

import algebraic
import equinox as eqx
import logic_asts as logic
import morphata
from algebraic import BoundedDistributiveLattice as Lattice
from algebraic.polynomials.rank_decomp import RankDecomposition
from jaxtyping import Array, Scalar
from morphata.spec import BoolExpr

Symbol = TypeVar("Symbol")
K = TypeVar("K", bound=Lattice)
AP = TypeVar("AP", bound=Hashable)


def boolexpr_to_polynomial(
    expr: BoolExpr[int],
    num_vars: int,
    algebra: K,
) -> RankDecomposition[K]:
    """Convert boolean expression over state variables to polynomial.

    Args:
        expr: Boolean expression with integer variable names (state indices)
        num_vars: Total number of variables (states) in polynomial
        algebra: Bounded distributive lattice for coefficients

    Returns:
        RankDecomposition representing the boolean expression

    Raises:
        ValueError: If expression contains Not operator or invalid state index
        TypeError: If expression contains unsupported BoolExpr type

    Notes:
        AFAs from LTL are assumed to be in positive normal form (no Not operators).
        If a Not operator is encountered, an assertion error will be raised.
    """
    # Cache for subexpressions (using id() for hash)
    cache: dict[BoolExpr[int], RankDecomposition[K]] = dict()

    def convert(e: BoolExpr[int]) -> RankDecomposition[K]:
        result: RankDecomposition[K]
        match e:
            case logic.Literal(val):
                return RankDecomposition[K].one(num_vars, algebra) if val else RankDecomposition[K].zero(num_vars, algebra)
            case logic.Variable(q):
                if not isinstance(q, int):
                    raise ValueError(f"Invalid state variable: {q}, expected integer")
                if not (0 <= q < num_vars):
                    raise ValueError(f"Invalid state variable: {q}, expected 0..{num_vars - 1}")
                return RankDecomposition[K].variable(q, num_vars, algebra)
            case logic.And(args):
                result, *tail = (cache[typing.cast(BoolExpr[int], arg)] for arg in args)
                for arg in tail:
                    result = result * arg
                return result
            case logic.Or(args):
                result, *tail = (cache[typing.cast(BoolExpr[int], arg)] for arg in args)
                for arg in tail:
                    result = result + arg
                return result
            case logic.Not():
                # Not should never appear in AFA expressions (positive normal form)
                raise ValueError(
                    f"Not operator encountered in AFA expression: {e}. "
                    "AFAs from LTL should be in positive normal form. "
                    "If needed, use logic_asts.to_nnf() to normalize."
                )
            case _:
                raise TypeError(f"Unsupported BoolExpr type: {type(e).__name__}")

    for subexpr in logic.bool_expr_iter(expr):
        cache[subexpr] = convert(subexpr)

    return cache[expr]


class PolynomialOperator(eqx.Module, Generic[Symbol, K]):
    """Polynomial-based operator for Alternating Finite Automata.

    Represents AFA transitions and runs as multilinear polynomials over
    a bounded distributive lattice (typically Boolean algebra).

    Attributes:
        initial_poly: Polynomial representing initial state set
        accepting_states: Frozenset of accepting state indices
        num_states: Number of states in the automaton
        algebra: Bounded distributive lattice for polynomial coefficients
        _transition_cache: Cached polynomials for each (state, symbol) pair
    """

    initial_poly: RankDecomposition[K]
    accepting_states: frozenset[int]
    num_states: int = eqx.field(static=True)
    algebra: K = eqx.field(default_factory=lambda: algebraic.semirings.boolean_algebra(), kw_only=True)
    _transition_cache: Mapping[tuple[int, Symbol], RankDecomposition[K]] = eqx.field(kw_only=True)

    def accepts(self, word: Sequence[Symbol]) -> Array:
        """Check if the automaton accepts the given word.

        Args:
            word: List of input symbols

        Returns:
            Boolean value from the algebra (one for accept, zero for reject)
        """
        run_poly = self.run_polynomial(word)
        return self.evaluate_at_accepting(run_poly)

    def run_polynomial(self, word: Sequence[Symbol]) -> RankDecomposition[K]:
        """Compute the polynomial representing all accepting runs on the word.

        Args:
            word: List of input symbols

        Returns:
            Polynomial over state variables representing the run tree
            after processing the entire word
        """
        current = self.initial_poly

        for symbol in word:
            current = self.step(current, symbol)

        return current

    def step(self, current: RankDecomposition[K], symbol: Symbol) -> RankDecomposition[K]:
        """Single-step transition: advance run polynomial by one symbol.

        Args:
            current: Polynomial representing current run configuration
            symbol: Input symbol to process

        Returns:
            Polynomial representing successor configuration
        """
        # Build substitution map for all state variables
        substitutions: dict[int, RankDecomposition[K]] = dict()

        for state_idx in range(self.num_states):
            # Check cache first
            cache_key = (state_idx, symbol)
            try:
                substitutions[state_idx] = self._transition_cache[cache_key]
            except KeyError:
                # For on-demand computation, would need automaton reference
                # For now, assume all transitions are cached
                raise KeyError(
                    f"Transition polynomial not found in cache for state {state_idx}, symbol {symbol}. "
                    "Use cache_transitions=True in from_afa() or provide some on-demand computation model that follows the Mapping protocol."
                ) from None

        # Compose: replace each variable x_i with its transition polynomial
        return current.compose(substitutions)

    def evaluate_at_accepting(self, poly: RankDecomposition[K]) -> Scalar:
        """Evaluate polynomial at the characteristic point for accepting states.

        Args:
            poly: Polynomial over state variables

        Returns:
            Boolean value: True if any accepting state is reachable
        """
        # Build characteristic point: 1 for accepting states, 0 for others
        point = {i: self.algebra.one if i in self.accepting_states else self.algebra.zero for i in range(self.num_states)}

        ret = poly.evaluate(point)
        return ret.factors.materialise()

    @classmethod
    def from_afa(
        cls,
        aut: morphata.Automaton[int, Symbol],
        algebra: K,
        *,
        cache_transitions: bool = True,
    ) -> PolynomialOperator[Symbol, K]:
        """Construct PolynomialOperator from alternating finite automaton.

        Args:
            aut: Automaton with integer states (0..n-1) and alternating transitions
            algebra: Bounded distributive lattice (typically BooleanAlgebra)
            cache_transitions: Whether to precompute all transition polynomials

        Returns:
            PolynomialOperator ready for evaluation

        Raises:
            TypeError: If automaton doesn't use AlternatingTransitions
            NotImplementedError: If acceptance condition is not Finite
            ValueError: If states are not contiguous integers 0..n-1
        """
        from morphata.acceptance import Finite
        from morphata.spec import AlternatingTransitions

        # Validation: check transitions protocol
        delta = aut.delta
        if not isinstance(delta, AlternatingTransitions):
            raise TypeError(f"Automaton must use AlternatingTransitions, got {type(aut.delta)}")

        # Validation: check acceptance condition
        if not isinstance(aut.acceptance, Finite):
            raise NotImplementedError(f"Only Finite acceptance supported, got {type(aut.acceptance)}")

        # Extract number of states
        num_states = _infer_num_states(aut)

        # Extract accepting states
        accepting_states = _extract_accepting_states(aut.acceptance)

        # Convert initial state to Boolean expression
        initial: BoolExpr[int]
        if isinstance(aut.initial, int):
            initial = logic.Variable(aut.initial)
        elif isinstance(aut.initial, AbstractSet):
            # Interpret this as a disjunction of states, conjunction should be a BoolExpr
            if len(aut.initial) >= 2:
                initial = logic.Or(tuple(logic.Variable(q) for q in aut.initial))
            elif len(aut.initial) == 1:
                initial, *_ = (logic.Variable(q) for q in aut.initial)
            else:
                raise ValueError("Cannot have empty initial set")
        else:
            assert logic.is_propositional_logic(aut.initial, var_type=int)
            initial = aut.initial

        # Convert initial state expression to polynomial
        initial_poly = boolexpr_to_polynomial(initial, num_states, algebra)

        # Optionally precompute transition polynomials
        transition_cache: Mapping[tuple[int, Symbol], RankDecomposition[K]] = dict()
        if cache_transitions:
            transition_cache = _build_transition_cache(delta, num_states, algebra, aut.domain)  # ty:ignore[invalid-argument-type]

        return cls(
            initial_poly=initial_poly,
            accepting_states=frozenset(accepting_states),
            num_states=num_states,
            algebra=algebra,
            _transition_cache=transition_cache,
        )

    @staticmethod
    def from_ltl(
        formula: logic.LTLExpr[AP], algebra: K, *, finite: bool = True, cache_transitions: bool = True
    ) -> PolynomialOperator[AbstractSet[AP], K]:
        """Convenience constructor: build from LTL formula directly.

        Args:
            formula: LTL/LTLf formula
            algebra: Boolean algebra instance
            finite: Whether to produce finite-word automaton
            cache_transitions: Whether to precompute transitions

        Returns:
            PolynomialOperator representing the formula
        """
        from morphata.examples.ltl import ltl_to_automaton

        aut = ltl_to_automaton(formula, finite=finite)
        return PolynomialOperator.from_afa(aut, algebra, cache_transitions=cache_transitions)


def _infer_num_states(aut: morphata.Automaton[int, Symbol]) -> int:
    """Infer number of states from automaton domain.

    Assumes states are integers 0..n-1.
    """
    if aut.domain.states is not None:
        # Domain has enumerable states
        states_list = list(aut.domain.states)
        if not all(isinstance(s, int) for s in states_list):
            raise ValueError("All states must be integers")

        if not states_list:
            raise ValueError("Automaton has no states")

        min_state, max_state = min(states_list), max(states_list)
        if min_state != 0 or max_state != len(states_list) - 1:
            raise ValueError(f"States must be contiguous integers 0..n-1, got {min_state}..{max_state}")

        return len(states_list)
    else:
        # Try to infer from initial state and acceptance
        # This is a fallback for symbolic domains
        raise NotImplementedError(
            "Cannot infer number of states from non-enumerable domain. Please provide num_states explicitly."
        )


def _extract_accepting_states(acceptance: morphata.AcceptanceCondition[int]) -> AbstractSet[int]:
    """Extract set of accepting state indices from acceptance condition."""
    from morphata.acceptance import Finite

    if isinstance(acceptance, Finite):
        # Finite acceptance has final_states attribute
        return acceptance.accepting  # ty:ignore[invalid-return-type]
    else:
        raise NotImplementedError(f"Unsupported acceptance condition: {type(acceptance)}")


def _build_transition_cache(
    transitions: morphata.AlternatingTransitions[int, Symbol],
    num_states: int,
    algebra: K,
    domain: morphata.Domain[int, Symbol],
) -> Mapping[tuple[int, Symbol], RankDecomposition[K]]:
    """Precompute transition polynomials for all (state, symbol) pairs.

    Note: Only feasible if alphabet is small and enumerable.
    For large/infinite alphabets, returns empty cache (use on-demand computation).
    """
    cache: dict[tuple[int, Symbol], RankDecomposition[K]] = dict()

    # Check if domain has enumerable symbols
    if domain.symbols is None:
        # Cannot enumerate alphabet: return empty cache
        return cache

    try:
        symbols = list(domain.symbols)
    except (TypeError, OverflowError):
        # Domain too large or infinite: skip caching
        return cache

    # Precompute for all (state, symbol) pairs
    for state in range(num_states):
        for symbol in symbols:
            trans_expr = transitions(state, symbol)
            trans_poly = boolexpr_to_polynomial(trans_expr, num_states, algebra)
            cache[(state, symbol)] = trans_poly

    return cache
