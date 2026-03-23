"""JAX backend implementation of ``MonomialBasis``."""

from __future__ import annotations

import typing

import equinox as eqx
from typing_extensions import Self, override

from algebraic.array import AlgebraicArray
from algebraic.array._jax import JaxAlgebraicArray
from algebraic.polynomials.monomial_basis.base import MonomialBasis
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import Backend
from algebraic.utils.jax import EqxMeta


class JaxMonomialBasis(eqx.Module, MonomialBasis, metaclass=EqxMeta):
    """JAX backend for ``MonomialBasis``."""

    coeffs: JaxAlgebraicArray
    algebra: Lattice = eqx.field(static=True)

    backend: typing.ClassVar[Backend] = Backend.JAX

    def __init__(self, coeffs: AlgebraicArray, algebra: Lattice) -> None:
        super().__init__()
        self.coeffs = typing.cast(JaxAlgebraicArray, coeffs)
        self.algebra = algebra

    @override
    def _replace_coeffs(self, coeffs: AlgebraicArray) -> Self:
        return typing.cast(Self, eqx.tree_at(lambda t: t.coeffs, self, coeffs))
