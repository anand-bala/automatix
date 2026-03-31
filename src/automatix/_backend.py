"""Backend resolution utilities for automatix operators."""

from __future__ import annotations

from algebraic import AlgebraicArray
from algebraic.types import Backend


class _StaticAux:
    """Wrapper making arbitrary objects usable as JAX pytree aux_data.

    Uses ``hash()`` when available, falls back to ``id()`` for unhashable
    objects.  This mirrors equinox's behaviour for static fields.
    """

    __slots__ = ("value",)

    def __init__(self, value: object) -> None:
        self.value = value

    def __hash__(self) -> int:
        try:
            return hash(self.value)
        except TypeError:
            return id(self.value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _StaticAux):
            try:
                return bool(self.value == other.value)
            except Exception:
                return self.value is other.value
        return NotImplemented


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
