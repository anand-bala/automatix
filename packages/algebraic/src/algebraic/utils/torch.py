from collections.abc import Mapping, Sequence

import torch.nn as nn

from algebraic.array import AlgebraicArray
from algebraic.types import AlgebraicPyTree, AnyPyTree


class TorchWrapper(nn.Module):
    def __init__(self, wrapped: AnyPyTree) -> None:
        super().__init__()
        self.wrapped = wrapped
        self._register(wrapped)

    def _register(self, wrapped: AnyPyTree, prefix: str | None = None) -> None:
        if prefix is None:
            prefix = ""
        if isinstance(wrapped, AlgebraicArray):
            self.register_parameter(prefix + "algebraic_array", nn.Parameter(wrapped.data))
        elif isinstance(wrapped, AlgebraicPyTree):
            for i, c in enumerate(wrapped.tree_flatten[0]):
                self._register(c, f"{prefix}{i}.")
        elif isinstance(wrapped, Mapping):
            for k, v in wrapped.items():
                self._register(v, prefix=f"{prefix}{str(k)}.")
        elif isinstance(wrapped, Sequence):
            for i, c in enumerate(wrapped):
                self._register(c, f"{prefix}{i}.")


def torchify(obj: AnyPyTree) -> TorchWrapper:
    return TorchWrapper(obj)
