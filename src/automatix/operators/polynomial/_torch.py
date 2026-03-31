from __future__ import annotations

import typing
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import Any, ClassVar

import torch.nn as nn
from algebraic import BoundedDistributiveLattice as Lattice
from algebraic.polynomials.rank_decomp import RankDecomposition
from algebraic.types import Backend

if typing.TYPE_CHECKING:
    from ._base import PolynomialOperator


class TorchPolynomialOperator(nn.Module, PolynomialOperator):
    """Torch-backend AFA polynomial operator.

    A :class:`torch.nn.Module`. ``initial_poly`` is registered as a
    submodule if it is itself an :class:`~torch.nn.Module` (which it will
    be when constructed with ``backend='torch'``).
    """

    backend: ClassVar[Backend] = Backend.TORCH

    def __init__(
        self,
        initial_poly: RankDecomposition,
        accepting_states: frozenset[int],
        num_states: int,
        algebra: Lattice,
        transition_cache: Mapping[tuple[int, Any], RankDecomposition],
    ) -> None:
        nn.Module.__init__(self)
        self.initial_poly = initial_poly
        self.accepting_states = accepting_states
        self.num_states = num_states
        self.algebra = algebra
        self._transition_cache = transition_cache

    @staticmethod
    def _make(
        initial_poly: RankDecomposition,
        accepting_states: AbstractSet[int],
        num_states: int,
        algebra: Lattice,
        transition_cache: Mapping[tuple[int, Any], RankDecomposition],
    ) -> PolynomialOperator:

        return TorchPolynomialOperator(  # type: ignore[return-value]
            initial_poly=initial_poly,
            accepting_states=frozenset(accepting_states),
            num_states=num_states,
            algebra=algebra,
            transition_cache=transition_cache,
        )
