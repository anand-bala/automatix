"""Symbolic BDD-based operator for Alternating Finite Automata.

Provides :class:`BDDOperator` as a BDD-native AFA operator.
Both the initial state set and all transition formulas are represented as
:class:`~automatix.operators._bdd.BDDDag` objects throughout execution:

.. code-block:: text

    morphata.Automaton (AlternatingTransitions)
        -> BoolExpr[int]
        -> reduced ordered BDD (BDDDag)
        -> BDD composition per step
        -> boolean evaluation at accepting states

Each :meth:`~BDDOperator.step` substitutes state variables in
the current run BDD with the transition BDDs, performing full BDD composition
via :func:`~automatix.operators._bdd.compose_bdd`.  No tensor polynomials or
semiring algebra are involved.
"""

from __future__ import annotations

import typing
from collections.abc import Hashable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any

import logic_asts as logic
import morphata
from morphata.spec import BoolExpr

from automatix.operators._bdd import BDDDag, boolexpr_to_bdd, compose_bdd, evaluate_bdd

AP = typing.TypeVar("AP", bound=Hashable)
Symbol = typing.TypeVar("Symbol")


@dataclass
class BDDOperator:
    """AFA operator that keeps the run state as a BDD throughout execution.

    Construct via :py:meth:`from_afa` or :py:meth:`from_ltl`.

    Fields
    ------
    initial_bdd : BDDDag
        BDD representing the initial state set.
    accepting_states : frozenset[int]
        Indices of accepting states.
    num_states : int
        Number of states in the automaton.
    _transition_cache : Mapping
        ``{(state_idx, symbol): BDDDag}`` -- pre-computed transition BDDs.
    """

    initial_bdd: BDDDag
    accepting_states: frozenset[int]
    num_states: int
    _transition_cache: Mapping[tuple[int, Any], BDDDag] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Shared computation
    # ------------------------------------------------------------------

    def accepts(self, word: Sequence[Any]) -> bool:
        """Check if the automaton accepts *word*.

        Parameters
        ----------
        word :
            Sequence of input symbols.

        Returns
        -------
        bool
            ``True`` if the automaton accepts, ``False`` otherwise.
        """
        return self.evaluate_at_accepting(self.run_bdd(word))

    def run_bdd(self, word: Sequence[Any]) -> BDDDag:
        """Compute the BDD representing all accepting runs on *word*.

        Parameters
        ----------
        word :
            Sequence of input symbols.

        Returns
        -------
        BDDDag
            BDD over state variables representing the run tree after
            processing the entire word.
        """
        current: BDDDag = self.initial_bdd
        for symbol in word:
            current = self.step(current, symbol)
        return current

    def step(self, current: BDDDag, symbol: Any) -> BDDDag:  # noqa: ANN401
        """Single-step transition: advance the run BDD by one symbol.

        Substitutes each state variable :math:`q_i` in *current* with the
        transition BDD for state *i* under *symbol*.

        Parameters
        ----------
        current :
            BDD representing the current run configuration.
        symbol :
            Input symbol to process.

        Returns
        -------
        BDDDag
            BDD representing the successor configuration.

        Raises
        ------
        KeyError
            If any ``(state_idx, symbol)`` pair is absent from the cache.
        """
        subs: list[BDDDag] = []
        for state_idx in range(self.num_states):
            cache_key = (state_idx, symbol)
            try:
                subs.append(self._transition_cache[cache_key])
            except KeyError:
                raise KeyError(
                    f"Transition BDD not found in cache for state {state_idx}, "
                    f"symbol {symbol!r}. Use cache_transitions=True in from_afa() or "
                    "provide a Mapping that covers all (state, symbol) pairs."
                ) from None
        return compose_bdd(current, subs)

    def evaluate_at_accepting(self, bdd: BDDDag) -> bool:
        """Evaluate *bdd* at the characteristic point for accepting states.

        Parameters
        ----------
        bdd :
            BDD over state variables.

        Returns
        -------
        bool
            ``True`` if the boolean function is satisfied when accepting
            states are set to ``True`` and all others to ``False``.
        """
        point = {i: (i in self.accepting_states) for i in range(self.num_states)}
        return evaluate_bdd(bdd, point)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @staticmethod
    def from_afa(
        aut: morphata.Automaton[int, Any],
        *,
        cache_transitions: bool = True,
        var_order: Sequence[int] | None = None,
    ) -> BDDOperator:
        """Construct from an alternating finite automaton.

        Parameters
        ----------
        aut :
            Automaton with integer states ``0..n-1`` and alternating
            transitions.
        cache_transitions :
            Whether to pre-compute all ``(state, symbol)`` transition BDDs.
        var_order :
            Optional BDD variable order (permutation of
            ``range(num_states)``).

        Returns
        -------
        BDDOperator

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

        initial_bdd = boolexpr_to_bdd(initial, num_states, var_order=var_order)

        transition_cache: Mapping[tuple[int, Any], BDDDag] = {}
        if cache_transitions:
            transition_cache = _build_transition_cache(
                delta,  # ty:ignore[invalid-argument-type]
                num_states,
                aut.domain,
                var_order=var_order,
            )

        return BDDOperator(
            initial_bdd=initial_bdd,
            accepting_states=frozenset(accepting_states),
            num_states=num_states,
            _transition_cache=transition_cache,
        )

    @staticmethod
    def from_ltl(
        formula: logic.LTLExpr[Any],
        *,
        finite: bool = True,
        cache_transitions: bool = True,
        var_order: Sequence[int] | None = None,
    ) -> BDDOperator:
        """Convenience constructor: build from an LTL formula directly.

        Parameters
        ----------
        formula :
            LTL/LTLf formula.
        finite :
            Whether to produce a finite-word automaton.
        cache_transitions :
            Whether to pre-compute transitions.
        var_order :
            Optional BDD variable order.

        Returns
        -------
        BDDOperator
            Representing the formula.
        """
        from morphata.examples.ltl import ltl_to_automaton

        aut = ltl_to_automaton(formula, finite=finite)
        return BDDOperator.from_afa(
            aut,
            cache_transitions=cache_transitions,
            var_order=var_order,
        )


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
    domain: morphata.Domain[int, Any],
    *,
    var_order: Sequence[int] | None = None,
) -> Mapping[tuple[int, Any], BDDDag]:
    cache: dict[tuple[int, Any], BDDDag] = {}
    if domain.symbols is None:
        return cache
    try:
        symbols = list(domain.symbols)
    except (TypeError, OverflowError):
        return cache
    for state in range(num_states):
        for symbol in symbols:
            trans_expr = transitions(state, symbol)
            cache[(state, symbol)] = boolexpr_to_bdd(trans_expr, num_states, var_order=var_order)
    return cache
