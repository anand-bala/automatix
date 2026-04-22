"""PyTorch integration for algebraic types.

Provides:
- PyTree registration with ``optree`` for ``torch.func`` transforms.
- ``PyTreeModule[T]``: an ``nn.Module`` that stores tensor leaves as
  ``nn.Parameter`` instances while reconstructing the full algebraic object
  (with all ops) in ``forward()``.
- ``torchify(obj)`` factory with ``@overload`` signatures.
"""

from __future__ import annotations

import typing
from typing import Generic, TypeVar, overload

import optree
import torch
import torch.nn as nn

from algebraic.array.base import AlgebraicArray
from algebraic.polynomials.dok import PolyDict
from algebraic.polynomials.monomial_basis import MonomialBasis
from algebraic.polynomials.rank_decomp import LowRankFactors, RankDecomposition
from algebraic.types import AlgebraicPyTree, AnyPyTree
from algebraic.utils import pytree

T = TypeVar("T", bound=AnyPyTree)


class PyTreeModule(nn.Module, Generic[T]):
    """``nn.Module`` wrapper that stores algebraic pytree leaves as ``nn.Parameter``.

    On :meth:`forward`, the original algebraic object is reconstructed via
    ``pytree.unflatten`` so the caller gets back a fully functional
    instance (``RankDecomposition``, ``LowRankFactors``, ``AlgebraicArray``, …)
    with all algebraic operations available and autograd flowing through the
    parameters.

    Parameters
    ----------
    obj : T
        The algebraic pytree to wrap. Must be backed by torch tensors.
    """

    def __init__(self, obj: T) -> None:
        super().__init__()
        leaves, self._spec = pytree.flatten(typing.cast(optree.PyTree[torch.Tensor], obj))
        for i, tensor in enumerate(leaves):
            self.register_parameter(f"leaf_{i}", nn.Parameter(tensor))

    def forward(self) -> T:
        """Reconstruct the algebraic object from the current parameter values."""
        params = list(self.parameters())
        return typing.cast(T, pytree.unflatten(self._spec, params))


@overload
def torchify(obj: AlgebraicArray) -> PyTreeModule[AlgebraicArray]: ...


@overload
def torchify(obj: RankDecomposition) -> PyTreeModule[RankDecomposition]: ...


@overload
def torchify(obj: LowRankFactors) -> PyTreeModule[LowRankFactors]: ...


@overload
def torchify(obj: PolyDict) -> PyTreeModule[PolyDict]: ...


@overload
def torchify(obj: MonomialBasis) -> PyTreeModule[MonomialBasis]: ...


@overload
def torchify(obj: AlgebraicPyTree) -> PyTreeModule[AlgebraicPyTree]: ...


def torchify(obj: AnyPyTree) -> PyTreeModule[typing.Any]:
    """Wrap an algebraic pytree in a :class:`PyTreeModule`.

    The returned module stores all tensor leaves as ``nn.Parameter`` instances
    (for optimizers, ``state_dict``, ``model.to(device)``, DDP, etc.) and
    reconstructs the full algebraic object in ``forward()``.
    """
    return PyTreeModule(obj)
