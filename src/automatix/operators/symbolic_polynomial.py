"""Symbolic polynomial operator for Alternating Finite Automata.

Provides :class:`SymbolicPolynomialOperator` as an alternative construction
path for AFA transition polynomials via BDD canonicalisation:

.. code-block:: text

    morphata.Automaton (AlternatingTransitions)
        → BoolExpr[int]
        → reduced ordered BDD (dd)
        → tensorised polynomial
        → RankDecomposition or LowRankFactors

The BDD step (via :mod:`automatix.operators._bdd`) canonicalises each
transition formula before tensorisation, so structurally shared sub-functions
are tensorised exactly once. The runtime API mirrors
:class:`~automatix.operators.polynomial.PolynomialOperator`.
"""

from __future__ import annotations

import typing
from collections.abc import Hashable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any, Literal

import algebraic
import logic_asts as logic
import morphata
import numpy as np
from algebraic import BoundedDistributiveLattice as Lattice
from algebraic.polynomials.rank_decomp import LowRankFactors, RankDecomposition
from algebraic.types import AnyPyTree, Backend
from morphata.spec import BoolExpr
from typing_extensions import Self

from automatix._backend import _StaticAux, resolve_backend
from automatix.operators._bdd import BDDDag, boolexpr_to_bdd

AP = typing.TypeVar("AP", bound=Hashable)
Symbol = typing.TypeVar("Symbol")

Poly = RankDecomposition | LowRankFactors


def bdd_to_rank_decomposition(
    bdd: BDDDag,
    num_vars: int,
    algebra: Lattice,
    *,
    backend: str | Backend = Backend.NUMPY,
) -> RankDecomposition:
    """Tensorise a BDD DAG into a :class:`~algebraic.polynomials.RankDecomposition`.

    Uses the Shannon-expansion recurrence bottom-up over ``bdd.topo_order``:

    .. code-block:: text

        P_false = zero,  P_true = one
        P_v = P_low(v) + X_i * P_high(v)

    Parameters
    ----------
    bdd :
        Extracted BDD DAG (children before parents in ``topo_order``).
    num_vars :
        Number of polynomial variables.
    algebra :
        Bounded distributive lattice for polynomial coefficients.
    backend :
        Array backend: ``'numpy'``, ``'jax'``, or ``'torch'``.

    Returns
    -------
    RankDecomposition
        Multilinear polynomial equivalent to the boolean function encoded by
        the BDD.
    """
    resolved = resolve_backend(backend)
    cache: dict[int, RankDecomposition] = {
        bdd.false_id: RankDecomposition.zero(num_vars, algebra, backend=resolved),
        bdd.true_id: RankDecomposition.one(num_vars, algebra, backend=resolved),
    }
    var_cache: dict[int, RankDecomposition] = {
        i: RankDecomposition.variable(i, num_vars, algebra, backend=resolved) for i in range(num_vars)
    }

    for node_id in bdd.topo_order:
        if node_id in cache:
            continue
        node = bdd.nodes[node_id]
        assert node.var_index is not None
        assert node.low_id is not None
        assert node.high_id is not None
        low = cache[node.low_id]
        high = cache[node.high_id]
        x_i = var_cache[node.var_index]
        cache[node_id] = low + (x_i * high)

    return cache[bdd.root_id]


def boolexpr_to_symbolic_polynomial(
    expr: BoolExpr[int],
    num_vars: int,
    algebra: Lattice,
    *,
    backend: str | Backend = Backend.NUMPY,
    output: Literal["rank_decomposition", "low_rank_factors"] = "rank_decomposition",
    var_order: Sequence[int] | None = None,
) -> RankDecomposition | LowRankFactors:
    """Convert a boolean expression to a polynomial via BDD canonicalisation.

    Parameters
    ----------
    expr :
        Boolean expression with integer variable names (state indices).
    num_vars :
        Total number of variables (states).
    algebra :
        Bounded distributive lattice for coefficients.
    backend :
        Array backend: ``'numpy'``, ``'jax'``, or ``'torch'``.
    output :
        Return type — ``"rank_decomposition"`` (default) or
        ``"low_rank_factors"``.
    var_order :
        Optional BDD variable order (permutation of ``range(num_vars)``).

    Returns
    -------
    RankDecomposition or LowRankFactors
        Polynomial equivalent to *expr*.
    """
    bdd = boolexpr_to_bdd(expr, num_vars, var_order=var_order)
    rd = bdd_to_rank_decomposition(bdd, num_vars, algebra, backend=backend)
    if output == "low_rank_factors":
        return LowRankFactors.from_rank_decomposition(rd)
    return rd


@dataclass
class SymbolicPolynomialOperator:
    """AFA polynomial operator built via BDD canonicalisation.

    Mirrors the API of :class:`~automatix.operators.polynomial.PolynomialOperator`
    but routes each boolean transition formula through a reduced ordered BDD
    before tensorisation, ensuring that shared sub-functions are tensorised
    exactly once.

    Construct via :py:meth:`from_afa` or :py:meth:`from_ltl`.

    Fields
    ------
    initial_poly : RankDecomposition or LowRankFactors
        Polynomial representing the initial state set.
    accepting_states : frozenset[int]
        Indices of accepting states.
    num_states : int
        Number of states in the automaton.
    algebra : Lattice
        Bounded distributive lattice for polynomial coefficients.
    _transition_cache : Mapping
        ``{(state_idx, symbol): polynomial}`` — pre-computed transition
        polynomials.
    """

    initial_poly: RankDecomposition | LowRankFactors
    accepting_states: frozenset[int]
    num_states: int
    algebra: Lattice
    _transition_cache: Mapping[tuple[int, Any], RankDecomposition | LowRankFactors] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # AlgebraicPyTree
    # ------------------------------------------------------------------

    def tree_flatten(
        self,
    ) -> tuple[list[RankDecomposition | LowRankFactors], tuple[Any, ...]]:
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
        assert isinstance(initial_poly, (RankDecomposition, LowRankFactors))
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

    def run_polynomial(self, word: Sequence[Any]) -> RankDecomposition | LowRankFactors:
        """Compute the polynomial representing all accepting runs on *word*.

        Parameters
        ----------
        word :
            Sequence of input symbols.

        Returns
        -------
        RankDecomposition or LowRankFactors
            Polynomial over state variables representing the run tree after
            processing the entire word.
        """
        current: RankDecomposition | LowRankFactors = self.initial_poly
        for symbol in word:
            current = self.step(current, symbol)
        return current

    def step(
        self,
        current: RankDecomposition | LowRankFactors,
        symbol: Any,  # noqa: ANN401
    ) -> RankDecomposition | LowRankFactors:
        """Single-step transition: advance the run polynomial by one symbol.

        Parameters
        ----------
        current :
            Polynomial representing the current run configuration.
        symbol :
            Input symbol to process.

        Returns
        -------
        RankDecomposition or LowRankFactors
            Polynomial representing the successor configuration.
        """
        rd_subs: list[RankDecomposition] = []
        lrf_subs: list[LowRankFactors] = []
        for state_idx in range(self.num_states):
            cache_key = (state_idx, symbol)
            try:
                poly = self._transition_cache[cache_key]
            except KeyError:
                raise KeyError(
                    f"Transition polynomial not found in cache for state {state_idx}, "
                    f"symbol {symbol}. Use cache_transitions=True in from_afa() or "
                    "provide a Mapping that covers all (state, symbol) pairs."
                ) from None
            if isinstance(poly, LowRankFactors):
                lrf_subs.append(poly)
            else:
                rd_subs.append(poly)

        if isinstance(current, LowRankFactors):
            return current.compose(lrf_subs)
        return current.compose(rd_subs)

    def evaluate_at_accepting(self, poly: RankDecomposition | LowRankFactors) -> algebraic.AlgebraicArray:
        """Evaluate *poly* at the characteristic point for accepting states.

        Parameters
        ----------
        poly :
            Polynomial over state variables.

        Returns
        -------
        AlgebraicArray
            Scalar algebra value: one if any accepting state is reachable,
            zero otherwise.
        """
        algebra: Lattice = self.algebra
        accepting: list[int] = list(self.accepting_states)
        point = np.array([algebra.one if i in accepting else algebra.zero for i in range(self.num_states)])
        ret = poly.evaluate(point)
        # Both types expose to_rank_decomposition() for uniform scalar extraction.
        rd = ret if isinstance(ret, RankDecomposition) else ret.to_rank_decomposition()
        factors = rd.factors
        item = factors[0, 0, 0]
        assert len(item.shape) == 0
        return item

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @staticmethod
    def from_afa(
        aut: morphata.Automaton[int, Any],
        algebra: Lattice,
        *,
        backend: str | Backend,
        cache_transitions: bool = True,
        output: Literal["rank_decomposition", "low_rank_factors"] = "rank_decomposition",
        var_order: Sequence[int] | None = None,
    ) -> SymbolicPolynomialOperator:
        """Construct from an alternating finite automaton.

        Parameters
        ----------
        aut :
            Automaton with integer states ``0..n-1`` and alternating transitions.
        algebra :
            Bounded distributive lattice (typically
            :func:`~algebraic.semirings.boolean_algebra`).
        backend :
            Which backend to use: ``'numpy'``, ``'jax'``, or ``'torch'``.
        cache_transitions :
            Whether to pre-compute all ``(state, symbol)`` transition polynomials.
        output :
            Polynomial representation — ``"rank_decomposition"`` (default) or
            ``"low_rank_factors"``.
        var_order :
            Optional BDD variable order (permutation of ``range(num_states)``).

        Returns
        -------
        SymbolicPolynomialOperator

        Raises
        ------
        TypeError
            If the automaton does not use
            :class:`~morphata.spec.AlternatingTransitions`.
        NotImplementedError
            If the acceptance condition is not
            :class:`~morphata.acceptance.Finite`.
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

        initial_poly = boolexpr_to_symbolic_polynomial(
            initial, num_states, algebra, backend=resolved, output=output, var_order=var_order
        )

        transition_cache: Mapping[tuple[int, Any], RankDecomposition | LowRankFactors] = {}
        if cache_transitions:
            transition_cache = _build_transition_cache(
                delta,  # ty:ignore[invalid-argument-type]
                num_states,
                algebra,
                aut.domain,
                backend=resolved,
                output=output,
                var_order=var_order,
            )

        return SymbolicPolynomialOperator(
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
        output: Literal["rank_decomposition", "low_rank_factors"] = "rank_decomposition",
        var_order: Sequence[int] | None = None,
    ) -> SymbolicPolynomialOperator:
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
        output :
            Polynomial representation — ``"rank_decomposition"`` or
            ``"low_rank_factors"``.
        var_order :
            Optional BDD variable order.

        Returns
        -------
        SymbolicPolynomialOperator
            Representing the formula.
        """
        from morphata.examples.ltl import ltl_to_automaton

        aut = ltl_to_automaton(formula, finite=finite)
        return SymbolicPolynomialOperator.from_afa(
            aut,
            algebra,
            backend=backend,
            cache_transitions=cache_transitions,
            output=output,
            var_order=var_order,
        )


# ---------------------------------------------------------------------------
# Private helpers (mirrored from polynomial.py)
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


def _extract_accepting_states(
    acceptance: morphata.AcceptanceCondition[int],
) -> AbstractSet[int]:
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
    output: Literal["rank_decomposition", "low_rank_factors"] = "rank_decomposition",
    var_order: Sequence[int] | None = None,
) -> Mapping[tuple[int, Any], RankDecomposition | LowRankFactors]:
    cache: dict[tuple[int, Any], RankDecomposition | LowRankFactors] = {}
    if domain.symbols is None:
        return cache
    try:
        symbols = list(domain.symbols)
    except (TypeError, OverflowError):
        return cache
    for state in range(num_states):
        for symbol in symbols:
            trans_expr = transitions(state, symbol)
            cache[(state, symbol)] = boolexpr_to_symbolic_polynomial(
                trans_expr,
                num_states,
                algebra,
                backend=backend,
                output=output,
                var_order=var_order,
            )
    return cache
