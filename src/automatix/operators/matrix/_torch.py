import warnings
from typing import Any, ClassVar

import torch.nn as nn
from algebraic import AlgebraicArray, Semiring
from algebraic.types import Backend

from automatix.spec import Guard, WeightFunction

from ._base import MatrixOperator


class TorchMatrixOperator(nn.Module, MatrixOperator):
    """Torch-backend weighted NFA operator.

    A :class:`torch.nn.Module`. If *weight_function* is itself an
    :class:`~torch.nn.Module`, it is registered as a submodule and its
    parameters appear in :py:meth:`~torch.nn.Module.parameters`.

    .. warning::
        If *weight_function* is not an :class:`~torch.nn.Module`, its
        internal parameters (if any) will **not** appear in
        :py:meth:`~torch.nn.Module.parameters` and will not be updated by
        a standard PyTorch optimizer. Pass an :class:`~torch.nn.Module`
        weight function to enable end-to-end training.
    """

    backend: ClassVar[Backend] = Backend.TORCH

    def __init__(
        self,
        initial_weights: AlgebraicArray,
        final_weights: AlgebraicArray,
        weight_function: WeightFunction,
        semiring: Semiring,
        transition_graph: tuple[tuple[int, int, Guard[Any]], ...],
    ) -> None:
        nn.Module.__init__(self)
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.semiring = semiring
        self._transition_graph = transition_graph

        if isinstance(weight_function, nn.Module):
            self.weight_function = weight_function  # auto-registered as submodule
        else:
            warnings.warn(
                "weight_function is not a torch.nn.Module. Its internal parameters "
                "(if any) will not appear in model.parameters() and will not be "
                "updated by a PyTorch optimizer. Wrap it in an nn.Module to enable "
                "end-to-end training.",
                UserWarning,
                stacklevel=2,
            )
            self.weight_function = weight_function

    @classmethod
    def _make(
        cls,
        initial_weights: AlgebraicArray,
        final_weights: AlgebraicArray,
        weight_function: WeightFunction,
        semiring: Semiring,
        transition_graph: tuple[tuple[int, int, Guard[Any]], ...],
    ) -> MatrixOperator:
        return TorchMatrixOperator(  # type: ignore[return-value]
            initial_weights=initial_weights,
            final_weights=final_weights,
            weight_function=weight_function,
            semiring=semiring,
            transition_graph=transition_graph,
        )
