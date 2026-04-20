"""Base algebraic array interfaces

This module defines an abstract `AlgebraicArray` that defines the interface for backend-specific algebraic array implementations.
"""

import math
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import array_api_compat
from typing_extensions import Self

from algebraic.spec import Semiring, has_complement, is_ring
from algebraic.types import AnyPyTree, Array, DType, MatmulFn, Number, Scalar, VdotFn, is_array, is_torch_array

if typing.TYPE_CHECKING:
    from algebraic.utils.indexing import _IndexUpdateHelper


def _resolve_device(dev_a: object, dev_b: object) -> object:
    """Return the higher-precedence device.

    Precedence: explicit (e.g. ``cuda``) > ``'cpu'`` > ``None``.
    """
    if dev_a is None:
        return dev_b
    if dev_b is None:
        return dev_a
    if str(dev_a).lower() == "cpu":
        return dev_b
    if str(dev_b).lower() == "cpu":
        return dev_a
    return dev_a


@dataclass
class AlgebraicArray:
    """A multidimensional array with elements from a semiring.

    This array overrides multiplication and addition to be defined with respect
    to the corresponding semiring.

    Attributes
    ----------
    data : Array
        The underlying backend array (``numpy.ndarray``, ``jax.Array``, or
        ``torch.Tensor``).
    semiring : Semiring
        The :class:`~algebraic.spec.Semiring` governing arithmetic operations.
    """

    data: Array
    semiring: Semiring

    _vdot: VdotFn | None = None
    _matmul: MatmulFn | None = None

    def _wrap(self, data: Array | Number) -> "AlgebraicArray":
        """Create a new instance with the given data, preserving all other attributes."""
        if is_torch_array(self.data):
            import torch

            data = torch.as_tensor(data)
        if is_torch_array(data):
            import torch

            data = typing.cast(Array, data.clone())
        else:
            array_ns = array_api_compat.array_namespace(self.data)
            data = typing.cast(Array, array_ns.asarray(data))
        return AlgebraicArray(data, self.semiring, self._vdot, self._matmul)

    def _coerce_other_device(self, other_data: "Array | Number") -> "tuple[Array, Array | Number]":
        """Return ``(self_data, other_data)`` both on the higher-precedence device.

        Scalars (non-arrays) are returned unchanged alongside ``self.data``.
        """
        if not is_array(other_data):
            return self.data, other_data
        other_arr = typing.cast(Array, other_data)
        self_dev = array_api_compat.device(self.data)
        other_dev = array_api_compat.device(other_arr)
        target = _resolve_device(self_dev, other_dev)
        if target is None:
            return self.data, other_arr
        self_out = self.data if str(self_dev) == str(target) else array_api_compat.to_device(self.data, target)
        other_out = other_arr if str(other_dev) == str(target) else array_api_compat.to_device(other_arr, target)
        return self_out, other_out

    def __add__(self, other: Self | Scalar) -> "AlgebraicArray":
        other_data = other.data if isinstance(other, AlgebraicArray) else other
        self_data, other_data = self._coerce_other_device(other_data)
        return self._wrap(self.semiring.add(self_data, other_data))

    def __mul__(self, other: Self | Scalar) -> "AlgebraicArray":
        other_data = other.data if isinstance(other, AlgebraicArray) else other
        self_data, other_data = self._coerce_other_device(other_data)
        return self._wrap(self.semiring.mul(self_data, other_data))

    def __sub__(self, other: Self | Scalar) -> "AlgebraicArray":
        other_data, _other_semiring = (
            (other.data, other.semiring) if isinstance(other, AlgebraicArray) else (other, self.semiring)
        )

        if not is_ring(self.semiring):
            raise NotImplementedError(
                f"Subtraction requires a Ring with additive_inverse. "
                f"Semiring {type(self.semiring).__name__} does not support subtraction."
            )
        self_data, other_data = self._coerce_other_device(other_data)
        neg_rhs = self.semiring.additive_inverse(other_data)
        return self._wrap(self.semiring.add(self_data, neg_rhs))

    def __neg__(self) -> "AlgebraicArray":
        semiring = self.semiring

        # Try additive_inverse first (for Rings)
        if is_ring(semiring):
            result_data = semiring.additive_inverse(self.data)
            return self._wrap(result_data)

        # Try complement (for Boolean/DeMorgan/Heyting/Stone algebras)
        if has_complement(semiring):
            result_data = semiring.complement(self.data)
            return self._wrap(result_data)

        raise NotImplementedError(
            f"Negation requires either additive_inverse (Ring) or complement (Boolean algebra). "
            f"Semiring {type(semiring).__name__} has neither."
        )

    def __matmul__(self, other: Self) -> "AlgebraicArray":
        """Matrix multiplication using semiring operations.

        Delegates to ``dot_general`` with dimension numbers determined by ``ndim``:

        - 2D x 2D: standard matrix multiply (contract last of lhs, first of rhs).

        - ND (batched): contract ``(ndim-1,)`` of lhs with ``(ndim-2,)`` of rhs; all leading
          dimensions are treated as batch dimensions.
        """
        from algebraic.ops._semiring_ops import dot_general

        self_data, other_arr_data = self._coerce_other_device(other.data)
        lhs: AlgebraicArray = (
            self if self_data is self.data else AlgebraicArray(self_data, self.semiring, self._vdot, self._matmul)
        )
        rhs: AlgebraicArray = (
            other
            if other_arr_data is other.data
            else AlgebraicArray(typing.cast(Array, other_arr_data), other.semiring, other._vdot, other._matmul)
        )

        ndim = lhs.ndim
        if ndim == 2:
            dimension_numbers: tuple[
                tuple[tuple[int, ...], tuple[int, ...]],
                tuple[tuple[int, ...], tuple[int, ...]],
            ] = (((1,), (0,)), ((), ()))
        else:
            batch = tuple(range(ndim - 2))
            dimension_numbers = (((ndim - 1,), (ndim - 2,)), (batch, batch))
        return typing.cast(Self, dot_general(lhs, rhs, dimension_numbers))

    def __eq__(self, other: object) -> Array:  # type: ignore[override]
        """Element-wise equality; returns a raw bool array (not wrapped)."""
        if isinstance(other, AlgebraicArray):
            self_data, other_data = self._coerce_other_device(other.data)
        else:
            self_data, other_data = self.data, other
        xp = array_api_compat.array_namespace(self_data)
        result: Array = xp.equal(self_data, other_data)
        return result

    def __ne__(self, other: object) -> Array:  # type: ignore[override]
        """Element-wise inequality; returns a raw bool array (not wrapped)."""
        if isinstance(other, AlgebraicArray):
            self_data, other_data = self._coerce_other_device(other.data)
        else:
            self_data, other_data = self.data, other
        xp = array_api_compat.array_namespace(self_data)
        result: Array = xp.not_equal(self_data, other_data)
        return result

    # Python sets __hash__ = None when __eq__ is defined; declare it explicitly
    # to make the intent clear to type checkers and documentation readers.
    __hash__: typing.ClassVar[None] = None  # type: ignore[assignment]

    def __getitem__(self, key: Any) -> "AlgebraicArray":  # noqa: ANN401
        """Index into the array, forwarding to the underlying data."""
        return self._wrap(self.data[key])

    def __setitem__(self, key: Any, value: Any) -> None:  # noqa: ANN401
        """Raise :class:`NotImplementedError` as ``AlgebraicArray`` is immutable.

        Use the functional ``.at[...].set(...)`` pattern instead.

        Raises
        ------
        NotImplementedError
            Always, since in-place updates are not supported.
        """
        raise NotImplementedError(
            "AlgebraicArray does not support in-place index updates. "
            "Use the `.at[...].set(...)` syntax for functional index updates."
        )

    @property
    def at(self) -> "_IndexUpdateHelper":
        """Return a helper for functional index updates.

        Example:
            result = arr.at[idx].set(value)
        """
        from algebraic.utils.indexing import _IndexUpdateHelper

        return _IndexUpdateHelper(self)

    def __pos__(self) -> "AlgebraicArray":
        """Unary positive; is a no-op."""
        return self

    def to_device(self, device: object, /, *, stream: int | None = None) -> "AlgebraicArray":
        """Return a copy of this array on the specified device.

        Parameters
        ----------
        device : object
            Target device for the returned array.
        stream : int or None, optional
            Device stream for async transfers.
        """
        # array_api_compat.to_device is the portable entry point across backends.
        result: Array = array_api_compat.to_device(self.data, device, stream=stream)
        return self._wrap(result)

    @property
    def dtype(self) -> DType:
        """Element data type of the underlying array."""
        return self.data.dtype

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        return len(self.data.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array as a tuple of ints."""
        return typing.cast(tuple[int, ...], self.data.shape)

    @property
    def size(self) -> int:
        """Total number of elements (product of all dimension sizes)."""
        return math.prod(self.shape)

    @property
    def device(self) -> object:
        """Device on which the underlying array resides."""
        # array_api_compat.device is the portable entry point across backends.
        return array_api_compat.device(self.data)

    @property
    def T(self) -> "AlgebraicArray":  # noqa: N802
        """Transpose of a 2-D matrix (swap last two axes)."""
        xp = array_api_compat.array_namespace(self.data)
        return self._wrap(xp.linalg.matrix_transpose(self.data))

    @property
    def mT(self) -> "AlgebraicArray":  # noqa: N802
        """Batch matrix transpose (same as `T`; alias for Array-API compatibility)."""
        return self.T

    def tree_flatten(self) -> tuple[list[Array], tuple[typing.Any, ...]]:
        return [self.data], (self.semiring, self._vdot, self._matmul)

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[typing.Any, ...], children: Sequence[AnyPyTree]) -> "AlgebraicArray":
        semiring, _vdot, _matmul = aux_data
        data = children[0]
        assert is_array(data)
        return cls(data, semiring, _vdot, _matmul)

    def clone(self) -> "AlgebraicArray":
        if is_torch_array(self.data):
            return AlgebraicArray(self.data.clone(), self.semiring, self._vdot, self._matmul)
        else:
            return self
