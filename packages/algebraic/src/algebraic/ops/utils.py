"""Utility functions for defining custom array operations"""

from __future__ import annotations

from collections.abc import Sequence

from plum import Dispatcher

dispatch = Dispatcher()


def normalize_axes(axis: int | Sequence[int] | None, ndim: int) -> tuple[int, ...]:
    """Return a sorted tuple of non-negative axis indices.

    Args:
        axis: `None` (all axes), a single `int`, or a sequence of `int`s.
            Negative values are resolved modulo *ndim*.
        ndim: Number of dimensions of the array being operated on.
    """
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        return (axis % ndim,)
    return tuple(sorted(a % ndim for a in axis))
