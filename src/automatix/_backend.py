"""Backend resolution utilities for automatix operators."""

from __future__ import annotations

from algebraic import AlgebraicArray
from algebraic.types import Backend


def resolve_backend(
    explicit: str | Backend | None,
    *arrays: AlgebraicArray | None,
) -> Backend:
    """Resolve the backend to use for an operator.

    Parameters
    ----------
    explicit : str | Backend | None
        Explicitly requested backend. Takes priority over inference.
    *arrays : AlgebraicArray | None
        Existing arrays to infer the backend from, tried left to right.

    Returns
    -------
    Backend
        The resolved backend.

    Raises
    ------
    ValueError
        If no backend can be determined (neither explicit nor inferable).
    TypeError
        If the explicit backend string is not a recognised backend name.
    """
    if explicit is not None:
        return Backend(explicit)
    for arr in arrays:
        if arr is not None:
            return Backend.from_array(arr.data)
    raise ValueError(
        "backend= must be provided, or inferable from initial_weights / final_weights. "
        "Pass backend='numpy', 'jax', or 'torch'."
    )
