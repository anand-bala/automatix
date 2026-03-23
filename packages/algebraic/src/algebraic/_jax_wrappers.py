# mypy: disable-error-code="misc"
"""Backend-agnostic transformations for algebraic arrays.

This module provides wrapped versions of common array transformations (jit, vmap) that
work across JAX, PyTorch, and NumPy backends.

Example:
    from algebraic._jax_wrappers import jit, vmap

    @jit(backend="jax")
    def compute(x):
        return x + x

    @vmap(backend="torch")
    def batch_compute(xs):
        return xs * xs
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
) -> (
    Callable[_FnParams, _ReturnType]
    | Callable[[Callable[_FnParams, _ReturnType]], Callable[_FnParams, _ReturnType]]
):
    """JIT compilation with backend selection.

    Args:
        fun: Function to compile. If ``None``, returns a decorator.
        backend: Backend to use (``"jax"``, ``"torch"``, or ``"numpy"``).

    Returns:
        Compiled function, or a decorator if *fun* is ``None``.
    """
    b = Backend(backend)

    def decorator(f: Callable[_FnParams, _ReturnType]) -> Callable[_FnParams, _ReturnType]:
        if b == Backend.JAX:
            import equinox as eqx

            return eqx.filter_jit(f)  # type: ignore[return-value]
        elif b == Backend.TORCH:
            import torch

            compiled = torch.compile(f)

            @functools.wraps(f)
            def wrapper(*args: _FnParams.args, **kwargs: _FnParams.kwargs) -> _ReturnType:
                return compiled(*args, **kwargs)

            return wrapper
        else:
            return f

    if fun is None:
        return decorator
    return decorator(fun)


def vmap(
    fun: Callable[_FnParams, _ReturnType] | None = None,
    *,
    backend: str | Backend,
    in_axes: int | None | Sequence[int | None] = 0,
    out_axes: int | None | Sequence[int | None] = 0,
    axis_name: Hashable | None = None,
    axis_size: int | None = None,
) -> (
    Callable[_FnParams, _ReturnType]
    | Callable[[Callable[_FnParams, _ReturnType]], Callable[_FnParams, _ReturnType]]
):
    """Vectorizing map with backend selection.

    Args:
        fun: Function to vectorize. If ``None``, returns a decorator.
        backend: Backend to use (``"jax"``, ``"torch"``, or ``"numpy"``).
        in_axes: Input axis specifications (JAX/Torch).
        out_axes: Output axis specifications (JAX/Torch).
        axis_name: Axis name for collective operations (JAX only).
        axis_size: Override for axis size (JAX only).

    Returns:
        Vectorized function, or a decorator if *fun* is ``None``.

    Raises:
        NotImplementedError: If *backend* is ``"numpy"``.
    """
    b = Backend(backend)

    def decorator(f: Callable[_FnParams, _ReturnType]) -> Callable[_FnParams, _ReturnType]:
        if b == Backend.JAX:
            import equinox as eqx

            return eqx.filter_vmap(  # type: ignore[return-value]
                f,
                in_axes=in_axes,
                out_axes=out_axes,
                axis_name=axis_name,
                axis_size=axis_size,
            )
        elif b == Backend.TORCH:
            import torch

            vmapped = torch.vmap(f, in_dims=in_axes, out_dims=out_axes)  # type: ignore[arg-type]

            @functools.wraps(f)
            def wrapper(*args: _FnParams.args, **kwargs: _FnParams.kwargs) -> _ReturnType:
                return vmapped(*args, **kwargs)

            return wrapper
        else:
            raise NotImplementedError("vmap is not supported for the NumPy backend.")

    if fun is None:
        return decorator
    return decorator(fun)
