from __future__ import annotations

import enum
import typing
from collections.abc import Callable, Sequence
from numbers import Number
from typing import Protocol, runtime_checkable

import array_api_compat
from jaxtyping import Num, Shaped
from typing_extensions import TypeIs

if typing.TYPE_CHECKING:
    import jax
    import numpy as np
    import numpy.typing as npt
    import torch

    Array = npt.NDArray[typing.Any] | jax.Array | torch.Tensor

    ArrayType = typing.TypeVar("ArrayType", np.ndarray, jax.Array, torch.Tensor)
else:

    @runtime_checkable
    class Array(Protocol):
        """Opaque jaxtyping-compatible array placeholder."""

        shape: tuple[int, ...]
        dtype: typing.Any

    ArrayType = typing.TypeVar("ArrayType")


class Backend(enum.StrEnum):
    """Enum of supported backends"""

    NUMPY = "numpy"
    TORCH = "torch"
    JAX = "jax"


type Scalar = Number | Shaped[Array, ""]

type Axis = int | Sequence[int]
type MaybeAxis = None | Axis
type Shape = tuple[int, ...]

type UnaryOp = Callable[[Scalar | Array], Array]
type BinaryOp = Callable[[Scalar | Array, Scalar | Array], Array]
type VdotFn = Callable[[Num[Array, " n"], Num[Array, " n"]], Num[Array, ""]]
type MatmulFn = Callable[[Num[Array, "n k"], Num[Array, "k m"]], Num[Array, "n m"]]


@runtime_checkable
class IdentityFn(Protocol):
    def __call__(self, shape: Shape) -> Shaped[Array, " {shape}"]: ...


@runtime_checkable
class ReductionOp(Protocol):
    def __call__(self, a: Array, axis: MaybeAxis = None) -> Array: ...


class AccumulationFn(Protocol):
    def __call__(self, acc: Array, x: Array) -> Array: ...


class ScanFn(Protocol):
    def __call__(self, carry: Array, acc: Array) -> tuple[Array, Array]: ...


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
