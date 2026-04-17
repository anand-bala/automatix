"""Polynomial-based operator for Alternating Finite Automata.

Provides :class:`PolynomialOperator` for representing AFA transitions and runs
as multilinear polynomials over a bounded distributive lattice (typically
Boolean algebra).

The operator implements :class:`~algebraic.types.AlgebraicPyTree`. Use
``algebraic.utils.jax.jaxify()`` or ``algebraic.utils.torch.torchify()``
for backend-specific integration.

Usage
-----
Construct via :py:meth:`PolynomialOperator.from_afa` or
:py:meth:`PolynomialOperator.from_ltl`.
"""

from __future__ import annotations

import typing
from collections.abc import Hashable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any

import algebraic
import logic_asts as logic
import morphata
from algebraic import BoundedDistributiveLattice as Lattice
from algebraic.polynomials.rank_decomp import RankDecomposition
from algebraic.types import AnyPyTree, Backend
from morphata.spec import BoolExpr
from typing_extensions import Self

from automatix._backend import _StaticAux, resolve_backend

AP = typing.TypeVar("AP", bound=Hashable)
K = typing.TypeVar("K", bound=Lattice)
Symbol = typing.TypeVar("Symbol")


def boolexpr_to_polynomial(
    expr: BoolExpr[int],
    num_vars: int,
    algebra: Lattice,
    *,
    backend: str | Backend = Backend.NUMPY,
) -> RankDecomposition:
    """Convert a boolean expression over state variables to a polynomial.

    Parameters
    ----------
    expr :
        Boolean expression with integer variable names (state indices).
    num_vars :
        Total number of variables (states) in the polynomial.
    algebra :
        Bounded distributive lattice for coefficients.
    backend :
        Which backend to use for the resulting polynomial.

    Returns
    -------
    RankDecomposition
        Representing the boolean expression.

    Raises
    ------
    ValueError
        If the expression contains a :class:`~logic_asts.Not` operator or an
        out-of-range state index.
    TypeError
        If the expression contains an unsupported :class:`~morphata.spec.BoolExpr` type.

    Notes
    -----
    AFAs from LTL are assumed to be in positive normal form (no ``Not``
    operators).
    """
    cache: dict[BoolExpr[int], RankDecomposition] = {}

    def convert(e: BoolExpr[int]) -> RankDecomposition:
        result: RankDecomposition
        match e:
            case logic.Literal(val):
                return (
                    RankDecomposition.one(num_vars, algebra, backend=backend)
                    if val
                    else RankDecomposition.zero(num_vars, algebra, backend=backend)
                )
            case logic.Variable(q):
                if not isinstance(q, int):
                    raise ValueError(f"Invalid state variable: {q}, expected integer")
                if not (0 <= q < num_vars):
                    raise ValueError(f"Invalid state variable: {q}, expected 0..{num_vars - 1}")
                return RankDecomposition.variable(q, num_vars, algebra, backend=backend)
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


@dataclass
class PolynomialOperator:
    """AFA polynomial operator implementing :class:`~algebraic.types.AlgebraicPyTree`.

    Represents AFA transitions and runs as multilinear polynomials over a
    bounded distributive lattice (typically Boolean algebra).

    Construct via :py:meth:`from_afa` or :py:meth:`from_ltl`.

    Fields
    ------
    initial_poly : RankDecomposition
        Polynomial representing the initial state set.
    accepting_states : frozenset[int]
        Indices of accepting states.
    num_states : int
        Number of states in the automaton.
    algebra : Lattice
        Bounded distributive lattice for polynomial coefficients.
    _transition_cache : Mapping
        ``{(state_idx, symbol): RankDecomposition}`` - pre-computed transition
        polynomials.
    """

    initial_poly: RankDecomposition
    accepting_states: frozenset[int]
    num_states: int
    algebra: Lattice
    _transition_cache: Mapping[tuple[int, Any], RankDecomposition] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # AlgebraicPyTree
    # ------------------------------------------------------------------

    def tree_flatten(self) -> tuple[list[RankDecomposition], tuple[Any, ...]]:
        return [self.initial_poly], (
            self.accepting_states,
            self.num_states,
            self.algebra,
            _StaticAux(self._transition_cache),
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[Any, ...],
        children: Sequence[AnyPyTree],
    ) -> Self:
        accepting_states, num_states, algebra, cache_wrapped = aux_data
        (initial_poly,) = children
        assert isinstance(initial_poly, RankDecomposition)
        return cls(
            initial_poly=initial_poly,
            accepting_states=accepting_states,
            num_states=num_states,
            algebra=algebra,
            _transition_cache=cache_wrapped.value,
        )

    # ------------------------------------------------------------------
    # Shared computation
    # ------------------------------------------------------------------

    def accepts(self, word: Sequence[Any]) -> object:
        """Check if the automaton accepts *word*.

        Parameters
        ----------
        word :
            Sequence of input symbols.

        Returns
        -------
        object
            Algebra one (accept) or zero (reject).
        """
        run_poly = self.run_polynomial(word)
        return self.evaluate_at_accepting(run_poly)

    def run_polynomial(self, word: Sequence[Any]) -> RankDecomposition:
        """Compute the polynomial representing all accepting runs on *word*.

        Parameters
        ----------
        word :
            Sequence of input symbols.

        Returns
        -------
        RankDecomposition
            Polynomial over state variables representing the run tree after
            processing the entire word.
        """
        current: RankDecomposition = self.initial_poly
        for symbol in word:
            current = self.step(current, symbol)
        return current

    def step(self, current: RankDecomposition, symbol: Any) -> RankDecomposition:  # noqa: ANN401
        """Single-step transition: advance the run polynomial by one symbol.

        Parameters
        ----------
        current :
            Polynomial representing the current run configuration.
        symbol :
            Input symbol to process.

        Returns
        -------
        RankDecomposition
            Polynomial representing the successor configuration.
        """
        substitutions: list[RankDecomposition] = []
        for state_idx in range(self.num_states):
            cache_key = (state_idx, symbol)
            try:
                substitutions.append(self._transition_cache[cache_key])
            except KeyError:
                raise KeyError(
                    f"Transition polynomial not found in cache for state {state_idx}, "
                    f"symbol {symbol}. Use cache_transitions=True in from_afa() or "
                    "provide a Mapping that covers all (state, symbol) pairs."
                ) from None
        return current.compose(substitutions)

    def evaluate_at_accepting(self, poly: RankDecomposition) -> algebraic.AlgebraicArray:
        """Evaluate *poly* at the characteristic point for accepting states.

        Parameters
        ----------
        poly :
            Polynomial over state variables.

        Returns
        -------
        object
            Algebra value: one if any accepting state is reachable, zero otherwise.
        """
        import numpy as np

        algebra: Lattice = self.algebra
        accepting: list[int] = list(self.accepting_states)
        point = np.array([algebra.one if i in accepting else algebra.zero for i in range(self.num_states)])
        ret = poly.evaluate(point)
        factors = ret.factors
        item = factors[0, 0, 0]
        assert len(item.shape) == 0
        return item

    @staticmethod
    def from_afa(
        aut: morphata.Automaton[int, Any],
        algebra: Lattice,
        *,
        backend: str | Backend,
        cache_transitions: bool = True,
    ) -> PolynomialOperator:
        """Construct a :class:`PolynomialOperator` from an alternating finite automaton.

        Parameters
        ----------
        aut :
            Automaton with integer states ``0..n-1`` and alternating transitions.
        algebra :
            Bounded distributive lattice (typically :func:`~algebraic.semirings.boolean_algebra`).
        backend :
            Which backend to use: ``'numpy'``, ``'jax'``, or ``'torch'``.
        cache_transitions :
            Whether to pre-compute all ``(state, symbol)`` transition polynomials.

        Returns
        -------
        PolynomialOperator
            The constructed operator. Use ``jaxify()`` or ``torchify()``
            from ``algebraic.utils`` for backend-specific integration.

        Raises
        ------
        TypeError
            If the automaton does not use :class:`~morphata.spec.AlternatingTransitions`.
        NotImplementedError
            If the acceptance condition is not :class:`~morphata.acceptance.Finite`.
        ValueError
            If states are not contiguous integers ``0..n-1``.
        """
        from morphata.acceptance import Finite
        from morphata.spec import AlternatingTransitions

        delta = aut.delta
        if not isinstance(delta, AlternatingTransitions):
            raise TypeError(f"Automaton must use AlternatingTransitions, got {type(aut.delta)}")
        if not isinstance(aut.acceptance, Finite):
            raise NotImplementedError(f"Only Finite acceptance supported, got {type(aut.acceptance)}")

        num_states = _infer_num_states(aut)
        accepting_states = _extract_accepting_states(aut.acceptance)

        initial: BoolExpr[int]
        if isinstance(aut.initial, int):
            initial = logic.Variable(aut.initial)
        elif isinstance(aut.initial, AbstractSet):
            if len(aut.initial) >= 2:
                initial = logic.Or(tuple(logic.Variable(q) for q in aut.initial))
            elif len(aut.initial) == 1:
                initial, *_ = (logic.Variable(q) for q in aut.initial)
            else:
                raise ValueError("Cannot have empty initial set")
        else:
            assert logic.is_propositional_logic(aut.initial, var_type=int)
            initial = aut.initial

        resolved = resolve_backend(backend)

        initial_poly = boolexpr_to_polynomial(initial, num_states, algebra, backend=resolved)

        transition_cache: Mapping[tuple[int, Any], RankDecomposition] = {}
        if cache_transitions:
            transition_cache = _build_transition_cache(
                delta,  # ty:ignore[invalid-argument-type]
                num_states,
                algebra,
                aut.domain,
                backend=resolved,
            )

        return PolynomialOperator(
            initial_poly=initial_poly,
            accepting_states=frozenset(accepting_states),
            num_states=num_states,
            algebra=algebra,
            _transition_cache=transition_cache,
        )

    @staticmethod
    def from_ltl(
        formula: logic.LTLExpr[Any],
        algebra: Lattice,
        *,
        backend: str | Backend,
        finite: bool = True,
        cache_transitions: bool = True,
    ) -> PolynomialOperator:
        """Convenience constructor: build from an LTL formula directly.

        Parameters
        ----------
        formula :
            LTL/LTLf formula.
        algebra :
            Boolean algebra instance.
        backend :
            Which backend to use: ``'numpy'``, ``'jax'``, or ``'torch'``.
        finite :
            Whether to produce a finite-word automaton.
        cache_transitions :
            Whether to pre-compute transitions.

        Returns
        -------
        PolynomialOperator
            Representing the formula.
        """
        from morphata.examples.ltl import ltl_to_automaton

        aut = ltl_to_automaton(formula, finite=finite)
        return PolynomialOperator.from_afa(aut, algebra, backend=backend, cache_transitions=cache_transitions)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _infer_num_states(aut: morphata.Automaton[int, Any]) -> int:
    if aut.domain.states is not None:
        states_list = list(aut.domain.states)
        if not all(isinstance(s, int) for s in states_list):
            raise ValueError("All states must be integers")
        if not states_list:
            raise ValueError("Automaton has no states")
        min_state, max_state = min(states_list), max(states_list)
        if min_state != 0 or max_state != len(states_list) - 1:
            raise ValueError(f"States must be contiguous integers 0..n-1, got {min_state}..{max_state}")
        return len(states_list)
    raise NotImplementedError("Cannot infer number of states from non-enumerable domain.")


def _extract_accepting_states(acceptance: morphata.AcceptanceCondition[int]) -> AbstractSet[int]:
    from morphata.acceptance import Finite

    if isinstance(acceptance, Finite):
        return typing.cast(AbstractSet[int], acceptance.accepting)
    raise NotImplementedError(f"Unsupported acceptance condition: {type(acceptance)}")


def _build_transition_cache(
    transitions: morphata.AlternatingTransitions[int, Any],
    num_states: int,
    algebra: Lattice,
    domain: morphata.Domain[int, Any],
    *,
    backend: str | Backend = Backend.NUMPY,
) -> Mapping[tuple[int, Any], RankDecomposition]:
    cache: dict[tuple[int, Any], RankDecomposition] = {}
    if domain.symbols is None:
        return cache
    try:
        symbols = list(domain.symbols)
    except (TypeError, OverflowError):
        return cache
    for state in range(num_states):
        for symbol in symbols:
            trans_expr = transitions(state, symbol)
            trans_poly = boolexpr_to_polynomial(trans_expr, num_states, algebra, backend=backend)
            cache[(state, symbol)] = trans_poly
    return cache
