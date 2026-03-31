"""Backend-agnostic transformations for algebraic arrays.

This module provides wrapped versions of common array transformations (``jit``,
``vmap``) that work across JAX, PyTorch, and NumPy backends.

Examples
--------
>>> from algebraic._jax_wrappers import jit, vmap
>>>
>>> @jit(backend="jax")
... def compute(x):
...     return x + x
>>>
>>> @vmap(backend="torch")
... def batch_compute(xs):
...     return xs * xs
"""

from __future__ import annotations

import functools
import typing
from collections.abc import Callable, Hashable, Sequence

from algebraic.types import Backend

_FnParams = typing.ParamSpec("_FnParams")
_ReturnType = typing.TypeVar("_ReturnType")


def jit(
    fun: Callable[_FnParams, _ReturnType] | None = None,
    *,
    backend: str | Backend,
) -> Callable[_FnParams, _ReturnType] | Callable[[Callable[_FnParams, _ReturnType]], Callable[_FnParams, _ReturnType]]:
    """JIT compilation with backend selection.

    Parameters
    ----------
    fun : callable or None
        Function to compile. If ``None``, returns a decorator.
    backend : str or Backend
        Backend to use (``"jax"``, ``"torch"``, or ``"numpy"``).

    Returns
    -------
    callable
        Compiled function, or a decorator if *fun* is ``None``.
    """
    b = Backend(backend)

    def decorator(f: Callable[_FnParams, _ReturnType]) -> Callable[_FnParams, _ReturnType]:
        if b == Backend.JAX:
            import equinox as eqx

            return eqx.filter_jit(f)
        elif b == Backend.TORCH:
            import torch

            compiled = torch.compile(f)

            @functools.wraps(f)
            def wrapper(*args: _FnParams.args, **kwargs: _FnParams.kwargs) -> _ReturnType:
                return compiled(*args, **kwargs)  # type: ignore[no-any-return]

            return wrapper
        else:
            return f

    if fun is None:
        return decorator
    return decorator(fun)
