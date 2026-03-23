"""NumPy backend implementation of `PolyDict`."""

from __future__ import annotations

import typing
from collections.abc import Mapping

from bitarray import frozenbitarray

from algebraic._better_abc import better_dataclass as dataclass
from algebraic.array._numpy import NumpyAlgebraicArray
from algebraic.polynomials.dok.base import PolyDict
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import Backend


@dataclass
class NumpyPolyDict(PolyDict):
    algebra: Lattice
    num_vars: int
    data: Mapping[frozenbitarray, NumpyAlgebraicArray]

    backend: typing.ClassVar[Backend] = Backend.NUMPY
