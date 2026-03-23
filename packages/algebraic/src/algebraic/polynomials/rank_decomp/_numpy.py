"""NumPy backend implementation of ``RankDecomposition``."""

from __future__ import annotations

import typing

from algebraic._better_abc import better_dataclass as dataclass
from algebraic.array._numpy import NumpyAlgebraicArray
from algebraic.polynomials.rank_decomp.base import RankDecomposition
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import Backend


@dataclass
class NumpyRankDecomposition(RankDecomposition):
    """NumPy backend for ``RankDecomposition``."""

    factors: NumpyAlgebraicArray
    algebra: Lattice
    max_rank: int
    max_degree: int
    max_replacement_degree: int

    backend: typing.ClassVar[Backend] = Backend.NUMPY
