"""Torch backend implementation of ``RankDecomposition``."""

from __future__ import annotations

import typing

import torch.nn as nn

from algebraic._backend_mixins import TorchReplaceMixin
from algebraic.array import AlgebraicArray
from algebraic.array._torch import TorchAlgebraicArray
from algebraic.polynomials.rank_decomp.base import RankDecomposition
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import Backend


class TorchRankDecomposition(nn.Module, TorchReplaceMixin, RankDecomposition):
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
