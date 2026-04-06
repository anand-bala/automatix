"""Torch backend implementation of ``MonomialBasis``."""

from __future__ import annotations

import typing

import torch.nn as nn

from algebraic._backend_mixins import TorchReplaceMixin
from algebraic.array import AlgebraicArray
from algebraic.array._torch import TorchAlgebraicArray
from algebraic.polynomials.monomial_basis.base import MonomialBasis
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import Backend


class TorchMonomialBasis(nn.Module, TorchReplaceMixin, MonomialBasis):
    """Torch backend for ``MonomialBasis``."""

    coeffs: TorchAlgebraicArray
    algebra: Lattice

    backend: typing.ClassVar[Backend] = Backend.TORCH

    def __init__(self, coeffs: AlgebraicArray, algebra: Lattice) -> None:
        super().__init__()
        self.coeffs = typing.cast(TorchAlgebraicArray, coeffs)
        self.algebra = algebra
