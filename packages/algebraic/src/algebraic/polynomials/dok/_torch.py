"""Torch backend implementation of `PolyDict`."""

from __future__ import annotations

import typing
from collections.abc import Mapping

import torch.nn as nn
from bitarray import frozenbitarray

from algebraic._backend_mixins import TorchReplaceMixin
from algebraic.array._torch import TorchAlgebraicArray
from algebraic.polynomials.dok.base import PolyDict
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import Backend


class TorchPolyDict(nn.Module, TorchReplaceMixin, PolyDict):
    algebra: Lattice
    num_vars: int
    data: dict[frozenbitarray, TorchAlgebraicArray]

    backend: typing.ClassVar[Backend] = Backend.TORCH

    def __init__(self, algebra: Lattice, num_vars: int, data: Mapping[frozenbitarray, TorchAlgebraicArray]) -> None:
        super().__init__()

        self.algebra = algebra
        self.num_vars = num_vars
        self.data = dict(data)
        # register the TorchAlgebraicArray as modules
        for k, v in self.data.items():
            self.add_module(k.to01(sep=""), v)
