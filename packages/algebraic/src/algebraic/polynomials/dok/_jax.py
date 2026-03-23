"""JAX backend implementation of `PolyDict`."""

from __future__ import annotations

import typing
from collections.abc import Mapping

import equinox as eqx
from bitarray import frozenbitarray
from typing_extensions import Self, override

from algebraic.array import AlgebraicArray
from algebraic.array._jax import JaxAlgebraicArray
from algebraic.polynomials.dok.base import PolyDict
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import Backend
from algebraic.utils.jax import EqxMeta


class JaxPolyDict(eqx.Module, PolyDict, metaclass=EqxMeta):
    algebra: Lattice
    num_vars: int
    data: dict[frozenbitarray, JaxAlgebraicArray]

    backend: typing.ClassVar[Backend] = Backend.JAX

    def __init__(self, algebra: Lattice, num_vars: int, data: Mapping[frozenbitarray, JaxAlgebraicArray]) -> None:
        super().__init__()

        self.algebra = algebra
        self.num_vars = num_vars
        self.data = dict(data)

    @override
    def _wrap(self, data: Mapping[frozenbitarray, AlgebraicArray]) -> Self:
        return typing.cast(Self, eqx.tree_at(lambda t: t.data, self, data))
