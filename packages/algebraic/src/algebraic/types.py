from __future__ import annotations

import enum
import typing
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import array_api_compat
from jaxtyping import Shaped
from typing_extensions import TypeIs

if typing.TYPE_CHECKING:
    import jax
    import numpy.typing as npt
    import torch
    from _typeshed import Incomplete

    Array = npt.NDArray[typing.Any] | jax.Array | torch.Tensor
    type DType = Incomplete

else:
    type DType = typing.Any

    @runtime_checkable
    class Array(Protocol):
        """Opaque jaxtyping-compatible array placeholder."""

        shape: tuple[int, ...]
        dtype: DType


class Backend(enum.StrEnum):
    """Enum of supported backends"""

    NUMPY = "numpy"
    TORCH = "torch"
    JAX = "jax"

    @classmethod
    def from_array(cls, data: object) -> "Backend":
        """Detect backend from an array instance."""
        if array_api_compat.is_jax_array(data):
            return cls.JAX
        if array_api_compat.is_torch_array(data):
            return cls.TORCH
        if array_api_compat.is_numpy_array(data) or isinstance(data, Number):
            # NOTE: Use numpy as default backend for scalars.
            return cls.NUMPY
        raise TypeError(f"Cannot detect backend for type {type(data).__name__!r}")


Number = float | int | bool
type Scalar = Number | Shaped[Array, ""]

type Axis = int | Sequence[int]
type MaybeAxis = None | Axis
type Shape = tuple[int, ...]


type UnaryOp = Callable[[Number | Array], Number | Array]
type BinaryOp = Callable[[Number | Array, Number | Array], Number | Array]
type VdotFn = Callable[[Shaped[Array, " n"], Shaped[Array, " n"]], Shaped[Array, ""]]
type MatmulFn = Callable[[Shaped[Array, "n k"], Shaped[Array, "k m"]], Shaped[Array, "n m"]]


@runtime_checkable
class ReductionOp(Protocol):
    def __call__(self, a: Array, axis: MaybeAxis = None) -> Array: ...


class AccumulationFn[Acc, X](Protocol):
    def __call__(self, acc: Acc, x: X) -> Acc: ...


class ScanFn[Carry, X, Y](Protocol):
    def __call__(self, carry: Carry, acc: X) -> tuple[Carry, Y]: ...


def is_scalar(x: object) -> TypeIs[Scalar]:
    """Test if an object is a scalar"""
    if isinstance(x, Number):
        return True
    elif is_array(x):
        return len(x.shape) == 0
    else:
        return False


def is_array(x: object) -> TypeIs[Array]:
    """Test if an object is an array"""
    return array_api_compat.is_numpy_array(x) or array_api_compat.is_jax_array(x) or array_api_compat.is_torch_array(x)
