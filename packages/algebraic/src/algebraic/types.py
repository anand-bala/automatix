"""Type definitions for the algebraic package.

This module defines the core type aliases and protocols used throughout the
package, including array types, backend selection, and callable protocols.
"""

from __future__ import annotations

import enum
import sys
import typing
from collections.abc import Callable, Hashable, Sequence
from types import ModuleType
from typing import Protocol, runtime_checkable

import array_api_compat
import array_api_compat.common._helpers as array_helpers
from jaxtyping import Shaped
from typing_extensions import Self, TypeAlias, TypeIs

if typing.TYPE_CHECKING:
    import jax
    import numpy.typing as npt
    import torch
    from _typeshed import Incomplete

    from algebraic import AlgebraicArray

    Array: TypeAlias = npt.NDArray[typing.Any] | jax.Array | torch.Tensor
    Device: TypeAlias = jax.Device | torch.device | str
    type DType = Incomplete

else:
    type DType = typing.Any

    @runtime_checkable
    class Array(Protocol):
        """Opaque jaxtyping-compatible array placeholder."""

        shape: tuple[int, ...]
        dtype: DType


Number = float | int
type Scalar = Number | Shaped[Array, ""]

type Axis = int | Sequence[int]
type MaybeAxis = None | Axis
type Shape = tuple[int, ...]


type UnaryOp = Callable[[Number | Array], Number | Array]
type BinaryOp = Callable[[Number | Array, Number | Array], Number | Array]
type VdotFn = Callable[[Shaped[Array, " n"], Shaped[Array, " n"]], Shaped[Array, ""]]
type MatmulFn = Callable[[Shaped[Array, "n k"], Shaped[Array, "k m"]], Shaped[Array, "n m"]]


def is_scalar(x: object) -> TypeIs[Scalar]:
    """Test if an object is a scalar"""
    if isinstance(x, Number):
        return True
    elif is_array(x):
        return len(x.shape) == 0  # ty: ignore[unresolved-attribute]
    else:
        return False


def is_numpy_array(x: object) -> TypeIs[npt.NDArray[typing.Any]]:
    return array_api_compat.is_numpy_array(x)


def is_numpy_device(x: object) -> TypeIs[typing.Literal["cpu"]]:
    return isinstance(x, str) and x == "cpu"


def is_torch_array(x: object) -> TypeIs[torch.Tensor]:
    return array_api_compat.is_torch_array(x)


def is_torch_device(dev: object) -> TypeIs[torch.device]:
    dev_type = typing.cast(Hashable, type(dev))
    return array_helpers._issubclass_fast(dev_type, "torch", "device")


def is_jax_array(x: object) -> TypeIs[jax.Array]:
    return (
        array_api_compat.is_jax_array(x)
        # Need to perform a custom override for `jaxlib._jax.ArrayImpl`
        or array_helpers._issubclass_fast(typing.cast(Hashable, type(x)), "jaxlib._jax", "ArrayImpl")
    )


def is_jax_device(dev: object) -> TypeIs[jax.Device]:
    dev_type = typing.cast(Hashable, type(dev))
    return array_helpers._issubclass_fast(dev_type, "jax", "Device")


def is_array(x: object) -> TypeIs[Array]:
    """Test if an object is an array"""
    return is_numpy_array(x) or is_jax_array(x) or is_torch_array(x)


def is_a_device(dev: object) -> TypeIs[Device]:
    if is_jax_device(dev) or is_torch_device(dev):
        return True
    if not isinstance(dev, str):
        return False
    if dev in ["cpu", "cuda"]:
        return True
    if "jax" in sys.modules:
        import jax

        try:
            return is_jax_device(jax.devices(dev)[0])
        except Exception:
            pass
    if "torch" in sys.modules:
        import torch

        try:
            return is_torch_device(torch.device(dev))
        except Exception:
            pass
    return False


def is_cpu_device(dev: Device) -> bool:
    """Return ``True`` if *dev* represents a CPU device (string or native)."""
    return str(dev).lower().startswith("cpu")


class Backend(enum.StrEnum):
    """Enum of supported backends"""

    NUMPY = "numpy"
    TORCH = "torch"
    JAX = "jax"

    @classmethod
    def from_array(cls, data: Array | Number) -> "Backend":
        """Detect backend from an array instance."""
        if array_api_compat.is_jax_array(data):
            return cls.JAX
        if array_api_compat.is_torch_array(data):
            return cls.TORCH
        if array_api_compat.is_numpy_array(data) or isinstance(data, Number):
            # NOTE: Use numpy as default backend for scalars.
            return cls.NUMPY
        raise TypeError(f"Cannot detect backend for type {type(data).__name__!r}")

    def get_array_namespace(self) -> ModuleType:
        match self:
            case Backend.NUMPY:
                import array_api_compat.numpy as np

                return np
            case Backend.JAX:
                import jax.numpy as jnp

                return jnp
            case Backend.TORCH:
                import array_api_compat.torch as torch

                return torch
            case _:
                raise ValueError(f"Unsupported backend {self}")


@runtime_checkable
class ReductionOp(Protocol):
    def __call__(self, a: Array, axis: MaybeAxis = None) -> Array: ...


class AccumulationFn[Acc, X](Protocol):
    def __call__(self, acc: Acc, x: X) -> Acc: ...


class ScanFn[Carry, X, Y](Protocol):
    def __call__(self, carry: Carry, acc: X) -> tuple[Carry, Y]: ...


type AnyPyTree = (
    Array | AlgebraicArray | AlgebraicPyTree | tuple[AnyPyTree, ...] | list[AnyPyTree] | dict[typing.Any, AnyPyTree]
)


@runtime_checkable
class AlgebraicPyTree(Protocol):
    def tree_flatten(self) -> tuple[Sequence[AnyPyTree], typing.Any]: ...

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: typing.Any,  # noqa: ANN401
        children: Sequence[AnyPyTree],
    ) -> Self: ...
