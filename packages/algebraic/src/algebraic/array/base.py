"""Base algebraic array interfaces

This module defines an abstract `AlgebraicArray` that defines the interface for backend-specific algebraic array implementations.
"""

# NOTE: Do NOT add `from __future__ import annotations` to this module.
# The `AbstractVar` annotations from `algebraic._better_abc` are processed at class
# definition time and must not be stringified (PEP 563 lazy evaluation would break them).

import copy
import math
import typing
from abc import abstractmethod
from typing import Any

import array_api_compat
from typing_extensions import Self

from algebraic._better_abc import AbstractVar
from algebraic._better_abc import BetterABCMeta as ABCMeta
from algebraic.spec import Semiring, has_complement, is_ring
from algebraic.types import Array, DType, MatmulFn, Number, Scalar, VdotFn

if typing.TYPE_CHECKING:
    from algebraic.array._index_update import _IndexUpdateHelper


class AlgebraicArray(metaclass=ABCMeta):
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

    data: AbstractVar[Array]
    semiring: AbstractVar[Semiring]

    _vdot: VdotFn | None = None
    _matmul: MatmulFn | None = None

    def _wrap(self, data: Array | Number) -> Self:
        """Create a new instance with the given data, preserving all other attributes."""
        clone = copy.copy(self)
        array_ns = array_api_compat.array_namespace(self.data)
        data = typing.cast(Array, array_ns.asarray(data))
        object.__setattr__(clone, "data", data)
        return clone

    def __add__(self, other: Self | Scalar) -> Self:
        other_data = other.data if isinstance(other, AlgebraicArray) else other
        return self._wrap(self.semiring.add(self.data, other_data))

    def __mul__(self, other: Self | Scalar) -> Self:
        other_data = other.data if isinstance(other, AlgebraicArray) else other
        return self._wrap(self.semiring.mul(self.data, other_data))

    def __sub__(self, other: Self | Scalar) -> Self:
        other_data, _other_semiring = (
            (other.data, other.semiring) if isinstance(other, AlgebraicArray) else (other, self.semiring)
        )

        if not is_ring(self.semiring):
            raise NotImplementedError(
                f"Subtraction requires a Ring with additive_inverse. "
                f"Semiring {type(self.semiring).__name__} does not support subtraction."
            )
        neg_rhs = self.semiring.additive_inverse(other_data)
        return self._wrap(self.semiring.add(self.data, neg_rhs))

    def __neg__(self) -> Self:
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

    def __matmul__(self, other: Self) -> Self:
        """Matrix multiplication using semiring operations.

        Delegates to ``dot_general`` with dimension numbers determined by ``ndim``:

        - 2D x 2D: standard matrix multiply (contract last of lhs, first of rhs).

        - ND (batched): contract ``(ndim-1,)`` of lhs with ``(ndim-2,)`` of rhs; all leading
          dimensions are treated as batch dimensions.
        """
        ndim = self.ndim
        if ndim == 2:
            dimension_numbers: tuple[
                tuple[tuple[int, ...], tuple[int, ...]],
                tuple[tuple[int, ...], tuple[int, ...]],
            ] = (((1,), (0,)), ((), ()))
        else:
            batch = tuple(range(ndim - 2))
            dimension_numbers = (((ndim - 1,), (ndim - 2,)), (batch, batch))
        return self.dot_general(other, dimension_numbers)

    def __eq__(self, other: object) -> Array:  # type: ignore[override]
        """Element-wise equality; returns a raw bool array (not wrapped)."""
        xp = array_api_compat.array_namespace(self.data)
        other_data: object = other.data if isinstance(other, AlgebraicArray) else other
        return xp.equal(self.data, other_data)  # type: ignore[no-any-return]

    def __ne__(self, other: object) -> Array:  # type: ignore[override]
        """Element-wise inequality; returns a raw bool array (not wrapped)."""
        xp = array_api_compat.array_namespace(self.data)
        other_data: object = other.data if isinstance(other, AlgebraicArray) else other
        return xp.not_equal(self.data, other_data)  # type: ignore[no-any-return]

    # Python sets __hash__ = None when __eq__ is defined; declare it explicitly
    # to make the intent clear to type checkers and documentation readers.
    __hash__: typing.ClassVar[None] = None  # type: ignore[assignment]

    def __getitem__(self, key: Any) -> Self:  # noqa: ANN401
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
        from algebraic.array._index_update import _IndexUpdateHelper

        return _IndexUpdateHelper(self)

    def __pos__(self) -> Self:
        """Unary positive; returns a shallow copy."""
        return copy.copy(self)

    def to_device(self, device: object, /, *, stream: int | None = None) -> Self:
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
        return self.data.shape

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
    def T(self) -> Self:  # noqa: N802
        """Transpose of a 2-D matrix (swap last two axes)."""
        xp = array_api_compat.array_namespace(self.data)
        return self._wrap(xp.linalg.matrix_transpose(self.data))

    @property
    def mT(self) -> Self:  # noqa: N802
        """Batch matrix transpose (same as `T`; alias for Array-API compatibility)."""
        return self.T

    @abstractmethod
    def dot_general(
        self,
        other: Self,
        dimension_numbers: tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]],
    ) -> Self:
        """Compute generalized dot product using semiring operations.

        This implements ``dot_general`` for :class:`AlgebraicArray`, using the
        semiring's multiplication and addition instead of standard arithmetic.

        Parameters
        ----------
        other : AlgebraicArray
            Right-hand side array.
        dimension_numbers : tuple
            Nested tuple of ``((contracting_dims), (batch_dims))`` for each
            operand, following the same layout as ``jax.lax.dot_general``.
        """
