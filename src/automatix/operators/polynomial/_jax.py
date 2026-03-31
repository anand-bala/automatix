from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import Any, ClassVar

import equinox as eqx
from algebraic import BoundedDistributiveLattice as Lattice
from algebraic.polynomials.rank_decomp import RankDecomposition
from algebraic.types import Backend

from ._base import PolynomialOperator


class JaxPolynomialOperator(eqx.Module, PolynomialOperator):
    """JAX-backend AFA polynomial operator.

    An :class:`equinox.Module` and therefore a JAX PyTree.

    Learnable fields (dynamic PyTree leaves)
    ----------------------------------------
    * ``initial_poly`` — the initial state polynomial (a
        :class:`~algebraic.polynomials.RankDecomposition`, itself an
        :class:`equinox.Module`)

    Static fields (not traced)
    --------------------------
    * ``accepting_states``, ``num_states``, ``algebra``,
        ``_transition_cache``

    .. note::
        ``_transition_cache`` is marked ``static``, so JAX retraces when
        the automaton structure changes. In practice this never happens at
        runtime. For very large automata the static hash comparison at JIT
        cache lookup may be slow; this is a known limitation.
    """

    initial_poly: RankDecomposition
    accepting_states: frozenset[int] = eqx.field(static=True)
    num_states: int = eqx.field(static=True)
    algebra: Lattice = eqx.field(static=True)
    _transition_cache: Mapping[tuple[int, Any], RankDecomposition] = eqx.field(static=True)
    backend: ClassVar[Backend] = Backend.JAX

    @staticmethod
    def _make(
        initial_poly: RankDecomposition,
        accepting_states: AbstractSet[int],
        num_states: int,
        algebra: Lattice,
        transition_cache: Mapping[tuple[int, Any], RankDecomposition],
    ) -> PolynomialOperator:

        return JaxPolynomialOperator(  # type: ignore[return-value]
            initial_poly=initial_poly,
            accepting_states=frozenset(accepting_states),
            num_states=num_states,
            algebra=algebra,
            _transition_cache=transition_cache,
        )
