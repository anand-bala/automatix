"""Torch backend implementation of ``RankDecomposition``."""

from __future__ import annotations

import typing

import torch.nn as nn
from typing_extensions import Self, override

from algebraic.array import AlgebraicArray
from algebraic.array._torch import TorchAlgebraicArray
from algebraic.polynomials.rank_decomp.base import RankDecomposition
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import Backend


class TorchRankDecomposition(nn.Module, RankDecomposition):
    """Torch backend for ``RankDecomposition``."""

    factors: TorchAlgebraicArray
    algebra: Lattice
    max_rank: int
    max_degree: int
    max_replacement_degree: int

    backend: typing.ClassVar[Backend] = Backend.TORCH

    def __init__(
        self,
        factors: AlgebraicArray,
        algebra: Lattice,
        max_rank: int,
        max_degree: int,
        max_replacement_degree: int,
    ) -> None:
        super().__init__()
        self.factors = typing.cast(TorchAlgebraicArray, factors)
        self.algebra = algebra
        self.max_rank = max_rank
        self.max_degree = max_degree
        self.max_replacement_degree = max_replacement_degree

    @override
    def _replace_factors(self, factors: AlgebraicArray) -> Self:
        return type(self)(factors, self.algebra, self.max_rank, self.max_degree, self.max_replacement_degree)
