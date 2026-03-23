"""NumPy backend implementation of ``MonomialBasis``."""

from __future__ import annotations

import typing

from algebraic._better_abc import better_dataclass as dataclass
from algebraic.array._numpy import NumpyAlgebraicArray
from algebraic.polynomials.monomial_basis.base import MonomialBasis
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import Backend


@dataclass
class NumpyMonomialBasis(MonomialBasis):
    """NumPy backend for ``MonomialBasis``."""

    coeffs: NumpyAlgebraicArray
    algebra: Lattice

    backend: typing.ClassVar[Backend] = Backend.NUMPY
