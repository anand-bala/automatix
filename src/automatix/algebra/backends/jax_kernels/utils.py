"""Shared utilities for JAX kernel implementations.

This module contains helper functions and decorators for implementing optimized
forward and backward passes for various semirings.
"""

from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def kernel(func: F) -> F:
    """Decorator to mark a function as an optimized JAX kernel.

    This decorator serves as documentation and can be extended in the future
    to add additional metadata or validation.

    Parameters
    ----------
    func : Callable
        The kernel function to decorate.

    Returns
    -------
    Callable
        The decorated function (unchanged).

    Examples
    --------
    >>> @kernel
    ... def my_custom_forward(x, y):
    ...     # Custom optimized implementation
    ...     return x + y
    """
    return func


__all__ = ["kernel"]
