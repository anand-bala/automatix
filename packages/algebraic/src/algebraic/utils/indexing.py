"""Index update functionality for AlgebraicArray using semiring operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import array_api_compat

from algebraic.spec import is_ring

if TYPE_CHECKING:
    import torch

    from algebraic.array.base import AlgebraicArray
    from algebraic.types import Array, Number


def _torchy_index(idx: Any, shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:  # noqa: ANN401
    """Convert an index to a tuple of long tensors for use with ``index_put``."""
    import torch

    if isinstance(idx, int):
        return (torch.tensor(idx).long(),)

    if isinstance(idx, slice):
        return (torch.arange(*idx.indices(shape[0])).long(),)

    if isinstance(idx, torch.Tensor):
        return (idx.long(),)

    if isinstance(idx, tuple):
        result = []
        for dim, i in enumerate(idx):
            if isinstance(i, slice):
                result.append(torch.arange(*i.indices(shape[dim])).long())
            elif isinstance(i, int):
                result.append(torch.tensor(i).long())
            else:
                result.append(torch.as_tensor(i).long())
        return tuple(result)

    raise TypeError(f"Unsupported index type for torch: {type(idx)}")


def _set_at_index(
    data: Any,  # noqa: ANN401
    idx: Any,  # noqa: ANN401
    value: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Set a value at an index, handling JAX immutability and mutable backends."""
    if array_api_compat.is_jax_array(data):
        return data.at[idx].set(value)
    if array_api_compat.is_torch_array(data):
        import torch

        torchy = _torchy_index(idx, tuple(data.shape))
        value = torch.as_tensor(value, dtype=data.dtype, device=data.device)
        broadcast_shape = torch.broadcast_shapes(*[t.shape for t in torchy])  # type: ignore[no-untyped-call]
        value = value.reshape(broadcast_shape)
        return data.index_put(torchy, value)
    new_data = data.copy()
    new_data[idx] = value
    return new_data


class _IndexUpdateRef:
    """Helper class for functional index updates with semiring operations.

    This class provides methods like set, add, multiply that return a new
    AlgebraicArray with the indexed elements updated using semiring operations.

    This is a transient builder object that is not a PyTree - it only exists
    to provide the .at[idx].set() syntax and is consumed within a single expression.
    """

    def __init__(self, array: AlgebraicArray, indices: Any) -> None:  # noqa: ANN401
        self.array: AlgebraicArray = array
        self.indices: Any = indices

    def set(self, values: AlgebraicArray | Array | Number) -> AlgebraicArray:  # noqa: ANN401
        """Set the indexed elements to the given values.

        Args:
            values: Values to set. Can be a scalar or array.

        Returns:
            New AlgebraicArray with updated values.
        """
        from algebraic.array.base import AlgebraicArray as BaseAlgebraicArray

        values_data = values.data if isinstance(values, BaseAlgebraicArray) else values
        new_data = _set_at_index(self.array.data, self.indices, values_data)
        return self.array._wrap(new_data)

    def add(self, values: AlgebraicArray | Array | Number) -> AlgebraicArray:  # noqa: ANN401
        """Add values to the indexed elements using semiring addition.

        Args:
            values: Values to add using semiring addition.

        Returns:
            New AlgebraicArray with updated values.
        """
        from algebraic.array.base import AlgebraicArray as BaseAlgebraicArray

        values_data = values.data if isinstance(values, BaseAlgebraicArray) else values
        xp = array_api_compat.array_namespace(self.array.data)
        values_data = xp.asarray(values_data)
        current = self.array.data[self.indices]
        updated = self.array.semiring.add(current, values_data)
        new_data = _set_at_index(self.array.data, self.indices, updated)
        return self.array._wrap(new_data)

    def multiply(self, values: AlgebraicArray | Array | Number) -> AlgebraicArray:  # noqa: ANN401
        """Multiply indexed elements by values using semiring multiplication.

        Args:
            values: Values to multiply using semiring multiplication.

        Returns:
            New AlgebraicArray with updated values.
        """
        from algebraic.array.base import AlgebraicArray as BaseAlgebraicArray

        values_data = values.data if isinstance(values, BaseAlgebraicArray) else values
        xp = array_api_compat.array_namespace(self.array.data)
        values_data = xp.asarray(values_data)
        current = self.array.data[self.indices]
        updated = self.array.semiring.mul(current, values_data)
        new_data = _set_at_index(self.array.data, self.indices, updated)
        return self.array._wrap(new_data)

    def subtract(self, values: AlgebraicArray | Array | Number) -> AlgebraicArray:  # noqa: ANN401
        """Subtract values from indexed elements (only for Rings).

        Args:
            values: Values to subtract using additive inverse.

        Returns:
            New AlgebraicArray with updated values.

        Raises:
            TypeError: If the semiring doesn't support subtraction.
        """
        semiring = self.array.semiring
        if not is_ring(semiring):
            raise TypeError(
                f"Subtraction requires a Ring with additive_inverse. "
                f"Semiring {type(semiring).__name__} does not support subtraction."
            )

        from algebraic.array.base import AlgebraicArray as BaseAlgebraicArray

        values_data = values.data if isinstance(values, BaseAlgebraicArray) else values
        xp = array_api_compat.array_namespace(self.array.data)
        values_data = xp.asarray(values_data)
        current = self.array.data[self.indices]
        neg_values = semiring.additive_inverse(values_data)
        updated = semiring.add(current, neg_values)
        new_data = _set_at_index(self.array.data, self.indices, updated)
        return self.array._wrap(new_data)

    def get(self) -> AlgebraicArray:
        """Get the indexed elements.

        Returns:
            AlgebraicArray containing the indexed elements.
        """
        return self.array[self.indices]

    def apply(self, func: Any) -> AlgebraicArray:  # noqa: ANN401
        """Apply a function to the indexed elements.

        Args:
            func: Function to apply to the indexed elements.
                 The function should work with the underlying backend arrays.

        Returns:
            New AlgebraicArray with updated values.
        """
        current = self.array.data[self.indices]
        updated = func(current)
        new_data = _set_at_index(self.array.data, self.indices, updated)
        return self.array._wrap(new_data)


class _IndexUpdateHelper:
    """Helper class to provide the .at[idx] syntax for AlgebraicArray.

    This is a transient builder object that is not a PyTree - it only exists
    to provide the .at[idx] syntax and is consumed within a single expression.
    Similar to JAX's native array.at[idx] which is also not a PyTree.
    """

    array: AlgebraicArray

    def __init__(self, array: AlgebraicArray) -> None:
        self.array = array

    def __getitem__(self, indices: Any) -> _IndexUpdateRef:  # noqa: ANN401
        """Return an _IndexUpdateRef for the given indices.

        Args:
            indices: Indices to select.

        Returns:
            _IndexUpdateRef object with methods for functional updates.
        """
        return _IndexUpdateRef(self.array, indices)
