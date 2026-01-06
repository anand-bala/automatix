# mypy: disable-error-code="no-any-return,no-untyped-call"
"""Index update functionality for AlgebraicArray using semiring operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp

if TYPE_CHECKING:
    from algebraic.array.core import AlgebraicArray, Semiring


class _IndexUpdateRef[K: Semiring](eqx.Module):
    """Helper class for functional index updates with semiring operations.

    This class provides methods like set, add, multiply that return a new
    AlgebraicArray with the indexed elements updated using semiring operations.
    """

    array: AlgebraicArray[K]
    indices: Any = eqx.field(static=True)  # Indices don't participate in differentiation

    def set(self, values: Any) -> AlgebraicArray[K]:  # noqa: ANN401
        """Set the indexed elements to the given values.

        Args:
            values: Values to set. Can be a scalar or array.

        Returns:
            New AlgebraicArray with updated values.
        """
        # Extract data if values is an AlgebraicArray
        if isinstance(values, type(self.array)):
            values_data = values.data
        else:
            values_data = values

        # Use JAX's at[].set() on the underlying data
        new_data = self.array.data.at[self.indices].set(values_data)
        return eqx.tree_at(lambda arr: arr.data, self.array, new_data)

    def add(self, values: Any) -> AlgebraicArray[K]:  # noqa: ANN401
        """Add values to the indexed elements using semiring addition.

        Args:
            values: Values to add using semiring addition.

        Returns:
            New AlgebraicArray with updated values.
        """
        # Extract data if values is an AlgebraicArray
        if isinstance(values, type(self.array)):
            values_data = values.data
        else:
            values_data = jnp.asarray(values)

        # Get current values at indices
        current = self.array.data[self.indices]

        # Add using semiring
        updated = self.array.semiring.add(current, values_data)

        # Use scatter to update
        new_data = self.array.data.at[self.indices].set(updated)
        return eqx.tree_at(lambda arr: arr.data, self.array, new_data)

    def multiply(self, values: Any) -> AlgebraicArray[K]:  # noqa: ANN401
        """Multiply indexed elements by values using semiring multiplication.

        Args:
            values: Values to multiply using semiring multiplication.

        Returns:
            New AlgebraicArray with updated values.
        """
        # Extract data if values is an AlgebraicArray
        if isinstance(values, type(self.array)):
            values_data = values.data
        else:
            values_data = jnp.asarray(values)

        # Get current values at indices
        current = self.array.data[self.indices]

        # Multiply using semiring
        updated = self.array.semiring.mul(current, values_data)

        # Use scatter to update
        new_data = self.array.data.at[self.indices].set(updated)
        return eqx.tree_at(lambda arr: arr.data, self.array, new_data)

    def subtract(self, values: Any) -> AlgebraicArray[K]:  # noqa: ANN401
        """Subtract values from indexed elements (only for Rings).

        Args:
            values: Values to subtract using additive inverse.

        Returns:
            New AlgebraicArray with updated values.

        Raises:
            TypeError: If the semiring doesn't support subtraction.
        """
        # Check if semiring has additive inverse
        if not hasattr(self.array.semiring, "additive_inverse"):
            raise TypeError(
                f"Subtraction requires a Ring with additive_inverse. "
                f"Semiring {type(self.array.semiring).__name__} does not support subtraction."
            )

        # Extract data if values is an AlgebraicArray
        if isinstance(values, type(self.array)):
            values_data = values.data
        else:
            values_data = jnp.asarray(values)

        # Get current values at indices
        current = self.array.data[self.indices]

        # Compute: current + (-values)
        neg_values = self.array.semiring.additive_inverse(values_data)
        updated = self.array.semiring.add(current, neg_values)

        # Use scatter to update
        new_data = self.array.data.at[self.indices].set(updated)
        return eqx.tree_at(lambda arr: arr.data, self.array, new_data)

    def get(self) -> AlgebraicArray[K]:
        """Get the indexed elements.

        Returns:
            AlgebraicArray containing the indexed elements.
        """
        return self.array[self.indices]

    def apply(self, func: Any) -> AlgebraicArray[K]:  # noqa: ANN401
        """Apply a function to the indexed elements.

        Args:
            func: Function to apply to the indexed elements.
                 The function should work with the underlying JAX arrays.

        Returns:
            New AlgebraicArray with updated values.
        """
        current = self.array.data[self.indices]
        updated = func(current)
        new_data = self.array.data.at[self.indices].set(updated)
        return eqx.tree_at(lambda arr: arr.data, self.array, new_data)


class _IndexUpdateHelper[K: Semiring](eqx.Module):
    """Helper class to provide the .at[idx] syntax for AlgebraicArray.

    This class is registered as an equinox Module to ensure it works
    properly with JAX transformations like jit, vmap, and grad.
    """

    array: AlgebraicArray[K]

    def __getitem__(self, indices: Any) -> _IndexUpdateRef[K]:  # noqa: ANN401
        """Return an _IndexUpdateRef for the given indices.

        Args:
            indices: Indices to select.

        Returns:
            _IndexUpdateRef object with methods for functional updates.
        """
        return _IndexUpdateRef(self.array, indices)
