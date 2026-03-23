"""JAX backend implementation of ``RankDecomposition``."""

from __future__ import annotations

import typing

import equinox as eqx
from typing_extensions import Self, override

from algebraic.array import AlgebraicArray
from algebraic.array._jax import JaxAlgebraicArray
from algebraic.polynomials.rank_decomp.base import RankDecomposition
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import Backend
from algebraic.utils.jax import EqxMeta


class JaxRankDecomposition(eqx.Module, RankDecomposition, metaclass=EqxMeta):
    """JAX backend for ``RankDecomposition``."""

    factors: JaxAlgebraicArray
    algebra: Lattice = eqx.field(static=True)
    max_rank: int = eqx.field(static=True)
    max_degree: int = eqx.field(static=True)
    max_replacement_degree: int = eqx.field(static=True)

    backend: typing.ClassVar[Backend] = Backend.JAX

    def __init__(
        self,
        factors: AlgebraicArray,
        algebra: Lattice,
        max_rank: int,
        max_degree: int,
        max_replacement_degree: int,
    ) -> None:
        super().__init__()
        self.factors = typing.cast(JaxAlgebraicArray, factors)
        self.algebra = algebra
        self.max_rank = max_rank
        self.max_degree = max_degree
        self.max_replacement_degree = max_replacement_degree

    @override
    def _replace_factors(self, factors: AlgebraicArray) -> Self:
        return typing.cast(Self, eqx.tree_at(lambda t: t.factors, self, factors))
