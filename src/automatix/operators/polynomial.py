"""Polynomial-based operator for Alternating Finite Automata.

This module provides PolynomialOperator for representing AFA transitions
and runs as multilinear polynomials over boolean algebra.
"""

from __future__ import annotations

from typing import Generic, TypeVar

import equinox as eqx
import logic_asts as logic
from algebraic.polynomials.rank_decomp import RankDecomposition
from algebraic.spec import BoundedDistributiveLattice
from jaxtyping import Array
from morphata.spec import BoolExpr

Symbol = TypeVar("Symbol")


def boolexpr_to_polynomial(
    expr: BoolExpr[int],
    num_vars: int,
    algebra: BoundedDistributiveLattice,
) -> RankDecomposition:
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
    cache: dict[int, RankDecomposition] = {}

    def convert(e: BoolExpr[int]) -> RankDecomposition:
        """Recursively convert boolean expression to polynomial."""
        # Check cache first
        expr_id = id(e)
        if expr_id in cache:
            return cache[expr_id]

        result: RankDecomposition

        if isinstance(e, logic.Literal):
            # Literal(True) -> one, Literal(False) -> zero
            if e.value:
                result = RankDecomposition.one(num_vars, algebra)
            else:
                result = RankDecomposition.zero(num_vars, algebra)

        elif isinstance(e, logic.Variable):
            # Variable(q) -> x_q
            q = e.name
            # Handle both integer and string variable names (e.g., "x_2" -> 2)
            if isinstance(q, str) and q.startswith("x_"):
                try:
                    q = int(q.split("_")[1])
                except (IndexError, ValueError):
                    raise ValueError(f"Invalid state variable name: {e.name}, expected 'x_i' or integer")
            elif not isinstance(q, int):
                raise ValueError(f"Invalid state variable: {q}, expected integer or 'x_i' format")

            if not (0 <= q < num_vars):
                raise ValueError(f"Invalid state variable: {q}, expected 0..{num_vars - 1}")
            result = RankDecomposition.variable(q, num_vars, algebra)

        elif isinstance(e, logic.And):
            # And(args) -> product of all args
            # logic_asts.And stores arguments in .args tuple
            if not hasattr(e, "args") or len(e.args) < 2:
                raise ValueError(f"And expression must have at least 2 arguments, got {e}")
            result = convert(e.args[0])
            for arg in e.args[1:]:
                result = result * convert(arg)

        elif isinstance(e, logic.Or):
            # Or(args) -> sum of all args
            # logic_asts.Or stores arguments in .args tuple
            if not hasattr(e, "args") or len(e.args) < 2:
                raise ValueError(f"Or expression must have at least 2 arguments, got {e}")
            result = convert(e.args[0])
            for arg in e.args[1:]:
                result = result + convert(arg)

        elif isinstance(e, logic.Not):
            # Not should never appear in AFA expressions (positive normal form)
            raise AssertionError(
                f"Not operator encountered in AFA expression: {e}. "
                "AFAs from LTL should be in positive normal form. "
                "If needed, use logic_asts.to_nnf() to normalize."
            )

        else:
            raise TypeError(f"Unsupported BoolExpr type: {type(e).__name__}")

        cache[expr_id] = result
        return result

    return convert(expr)


class PolynomialOperator(eqx.Module, Generic[Symbol]):
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

    initial_poly: RankDecomposition
    accepting_states: frozenset[int]
    num_states: int
    algebra: BoundedDistributiveLattice
    _transition_cache: dict[tuple[int, Symbol], RankDecomposition] = eqx.field(static=True)

    def accepts(self, word: list[Symbol]) -> Array:
        """Check if the automaton accepts the given word.

        Args:
            word: List of input symbols

        Returns:
            Boolean value from the algebra (one for accept, zero for reject)
        """
        run_poly = self.run_polynomial(word)
        return self.evaluate_at_accepting(run_poly)

    def run_polynomial(self, word: list[Symbol]) -> RankDecomposition:
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

    def step(self, current: RankDecomposition, symbol: Symbol) -> RankDecomposition:
        """Single-step transition: advance run polynomial by one symbol.

        Args:
            current: Polynomial representing current run configuration
            symbol: Input symbol to process

        Returns:
            Polynomial representing successor configuration
        """
        # Build substitution map for all state variables
        substitutions: dict[int, RankDecomposition] = {}

        for state_idx in range(self.num_states):
            # Check cache first
            cache_key = (state_idx, symbol)
            if cache_key in self._transition_cache:
                substitutions[state_idx] = self._transition_cache[cache_key]
            else:
                # For on-demand computation, would need automaton reference
                # For now, assume all transitions are cached
                raise KeyError(
                    f"Transition polynomial not found in cache for state {state_idx}, symbol {symbol}. "
                    "Use cache_transitions=True in from_afa() or provide automaton for on-demand computation."
                )

        # Compose: replace each variable x_i with its transition polynomial
        return current.compose(substitutions)

    def evaluate_at_accepting(self, poly: RankDecomposition) -> Array:
        """Evaluate polynomial at the characteristic point for accepting states.

        Args:
            poly: Polynomial over state variables

        Returns:
            Boolean value: True if any accepting state is reachable
        """
        # Build characteristic point: 1 for accepting states, 0 for others
        point = {i: self.algebra.one if i in self.accepting_states else self.algebra.zero for i in range(self.num_states)}

        return poly.evaluate(point)


def from_afa(
    aut,  # morphata.Automaton[int, Input[AP]]
    algebra: BoundedDistributiveLattice,
    *,
    cache_transitions: bool = True,
) -> PolynomialOperator:
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
    if not isinstance(aut.delta, AlternatingTransitions):
        raise TypeError(f"Automaton must use AlternatingTransitions, got {type(aut.delta)}")

    # Validation: check acceptance condition
    if not isinstance(aut.acceptance, Finite):
        raise NotImplementedError(f"Only Finite acceptance supported, got {type(aut.acceptance)}")

    # Extract number of states
    num_states = _infer_num_states(aut)

    # Extract accepting states
    accepting_states = _extract_accepting_states(aut.acceptance)

    # Convert initial state expression to polynomial
    initial_poly = boolexpr_to_polynomial(aut.initial, num_states, algebra)

    # Optionally precompute transition polynomials
    transition_cache: dict = {}
    if cache_transitions:
        transition_cache = _build_transition_cache(aut.delta, num_states, algebra, aut.domain)

    return PolynomialOperator(
        initial_poly=initial_poly,
        accepting_states=frozenset(accepting_states),
        num_states=num_states,
        algebra=algebra,
        _transition_cache=transition_cache,
    )


def from_ltl(
    formula,  # LTL formula type
    algebra: BoundedDistributiveLattice,
    *,
    finite: bool = True,
    cache_transitions: bool = True,
) -> PolynomialOperator:
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
    return from_afa(aut, algebra, cache_transitions=cache_transitions)


def _infer_num_states(aut) -> int:
    """Infer number of states from automaton domain.

    Assumes states are integers 0..n-1.
    """
    if hasattr(aut.domain, "states") and aut.domain.states is not None:
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


def _extract_accepting_states(acceptance) -> set[int]:
    """Extract set of accepting state indices from acceptance condition."""
    from morphata.acceptance import Finite

    if isinstance(acceptance, Finite):
        # Finite acceptance has final_states attribute
        if hasattr(acceptance, "final_states"):
            return set(acceptance.final_states)
        else:
            raise ValueError(f"Cannot extract accepting states from {type(acceptance)}")
    else:
        raise NotImplementedError(f"Unsupported acceptance condition: {type(acceptance)}")


def _build_transition_cache(
    transitions,  # AlternatingTransitions
    num_states: int,
    algebra: BoundedDistributiveLattice,
    domain,
) -> dict:
    """Precompute transition polynomials for all (state, symbol) pairs.

    Note: Only feasible if alphabet is small and enumerable.
    For large/infinite alphabets, returns empty cache (use on-demand computation).
    """
    cache = {}

    # Check if domain has enumerable symbols
    if not hasattr(domain, "symbols") or domain.symbols is None:
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
