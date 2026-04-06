"""Shared reduction/scan helpers for NumPy and Torch backends.

JAX uses ``jax.lax.reduce`` / ``jax.lax.associative_scan`` and does not
share this code path.
"""

# mypy: disable-error-code="no-any-return,no-untyped-call"

from __future__ import annotations

from typing import Any

import array_api_compat

from algebraic.types import Array


def reduce_axes(
    data: Array,
    semiring_op: Any,  # noqa: ANN401
    identity: Array,
    dims: tuple[int, ...],
) -> Array:
    """Reduce *data* along *dims* using a binary semiring operation.

    Parameters
    ----------
    data : Array
        Input array (NumPy or Torch).
    semiring_op : callable
        Binary operation (e.g. ``semiring.add`` or ``semiring.mul``).
    identity : Array
        Identity element for the operation (e.g. ``zero`` or ``one``).
    dims : tuple of int
        Sorted, non-negative axis indices to reduce over.

    Returns
    -------
    Array
        Reduced array.
    """
    xp = array_api_compat.array_namespace(data)
    result = xp.asarray(data)
    for dim in sorted(dims, reverse=True):
        n = result.shape[dim]
        slices = [_take(result, i, dim) for i in range(n)]
        acc = _broadcast_copy(identity, slices[0].shape)
        for s in slices:
            acc = xp.asarray(semiring_op(acc, s))
        result = acc
    return result


def scan_axis(
    data: Array,
    semiring_op: Any,  # noqa: ANN401
    identity: Array,
    axis: int,
    include_initial: bool,
) -> Array:
    """Inclusive prefix scan along *axis* using a binary semiring operation.

    Parameters
    ----------
    data : Array
        Input array (NumPy or Torch).
    semiring_op : callable
        Binary operation (e.g. ``semiring.add`` or ``semiring.mul``).
    identity : Array
        Identity element for the operation.
    axis : int
        Non-negative axis along which to scan.
    include_initial : bool
        When ``True``, prepend an identity slice so that the output has
        one extra element along *axis*.

    Returns
    -------
    Array
        Scanned array.
    """
    xp = array_api_compat.array_namespace(data)
    n = data.shape[axis]

    slices = [_take(data, i, axis) for i in range(n)]
    outputs: list[Any] = []
    acc = _broadcast_copy(identity, slices[0].shape) if n > 0 else identity
    for s in slices:
        acc = xp.asarray(semiring_op(acc, s))
        outputs.append(acc)

    scanned = xp.stack(outputs, axis=axis) if outputs else data

    if include_initial:
        shape = list(scanned.shape)
        shape[axis] = 1
        id_slice = _full(tuple(shape), identity, data)
        scanned = _concat(id_slice, scanned, axis)
    return scanned


# -- Private helpers ----------------------------------------------------------


def _take(data: Array, index: int, axis: int) -> Array:
    """Extract a single slice along *axis*."""
    if array_api_compat.is_torch_array(data):
        return data.select(axis, index)
    import numpy as np

    return np.take(data, index, axis=axis)


def _broadcast_copy(identity: Array, shape: tuple[int, ...]) -> Array:
    """Broadcast *identity* to *shape* and copy to make mutable."""
    if array_api_compat.is_torch_array(identity):
        return identity.expand(shape).clone()
    import numpy as np

    return np.broadcast_to(identity, shape).copy()


def _full(shape: tuple[int, ...], fill_value: Array, reference: Array) -> Array:
    """Create a filled array matching the dtype of *reference*."""
    if array_api_compat.is_torch_array(reference):
        import torch

        return torch.full(shape, fill_value, dtype=reference.dtype)
    import numpy as np

    return np.full(shape, fill_value, dtype=reference.dtype)


def _concat(a: Array, b: Array, axis: int) -> Array:
    """Concatenate two arrays along *axis*."""
    if array_api_compat.is_torch_array(a):
        import torch

        return torch.cat([a, b], dim=axis)
    import numpy as np

    return np.concatenate([a, b], axis=axis)
