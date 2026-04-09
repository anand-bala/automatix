"""Backend-agnostic transformations for algebraic arrays.


Examples
--------
>>> from algebraic import vmap
>>>
>>> @vmap(backend="torch")
... def batch_compute(xs):
...     return xs * xs
"""

from __future__ import annotations

import typing
from collections.abc import Callable, Hashable, Sequence

from algebraic.types import Backend

_FnParams = typing.ParamSpec("_FnParams")
_ReturnType = typing.TypeVar("_ReturnType")


def vmap(
    fun: Callable[_FnParams, _ReturnType] | None = None,
    *,
    backend: str | Backend,
    in_axes: int | None | Sequence[int | None] = 0,
    out_axes: int | None | Sequence[int | None] = 0,
    axis_name: Hashable | None = None,
    axis_size: int | None = None,
) -> Callable[_FnParams, _ReturnType] | Callable[[Callable[_FnParams, _ReturnType]], Callable[_FnParams, _ReturnType]]:
    """Vectorizing map with backend selection.

    Parameters
    ----------
    fun : callable or None
        Function to vectorize. If ``None``, returns a decorator.
    backend : str or Backend
        Backend to use (``"jax"``, ``"torch"``, or ``"numpy"``).
    in_axes : int or None or sequence, default 0
        Input axis specifications (JAX/Torch).
    out_axes : int or None or sequence, default 0
        Output axis specifications (JAX/Torch).
    axis_name : Hashable or None, optional
        Axis name for collective operations (JAX only).
    axis_size : int or None, optional
        Override for axis size (JAX only).

    Returns
    -------
    callable
        Vectorized function, or a decorator if *fun* is ``None``.

    Raises
    ------
    NotImplementedError
        If *backend* is ``"numpy"``.
    """
    b = Backend(backend)

    def decorator(f: Callable[_FnParams, _ReturnType]) -> Callable[_FnParams, _ReturnType]:
        if b == Backend.JAX:
            import jax

            return typing.cast(
                Callable[_FnParams, _ReturnType],
                jax.vmap(
                    f,
                    in_axes=in_axes,
                    out_axes=out_axes,
                    axis_name=axis_name,
                    axis_size=axis_size,
                ),
            )
        elif b == Backend.TORCH:
            import functools

            import torch

            from algebraic.array import AlgebraicArray

            @functools.wraps(f)
            def wrapper(
                *args: _FnParams.args, **kwargs: _FnParams.kwargs
            ) -> _ReturnType:
                semiring = None
                for a in args:
                    if isinstance(a, AlgebraicArray):
                        semiring = a.semiring
                        break

                raw_args = tuple(
                    a.data if isinstance(a, AlgebraicArray) else a for a in args
                )

                def raw_fn(
                    *raw: torch.Tensor,
                ) -> torch.Tensor:
                    rebuilt = tuple(
                        AlgebraicArray(data=r, semiring=semiring)
                        if isinstance(orig, AlgebraicArray)
                        else r
                        for r, orig in zip(raw, args)
                    )
                    out = f(*rebuilt, **kwargs)
                    if isinstance(out, AlgebraicArray):
                        return out.data
                    return out

                result = torch.vmap(raw_fn, in_dims=in_axes, out_dims=out_axes)(
                    *raw_args
                )
                if semiring is not None:
                    return typing.cast(
                        _ReturnType, AlgebraicArray(data=result, semiring=semiring)
                    )
                return typing.cast(_ReturnType, result)

            return typing.cast(Callable[_FnParams, _ReturnType], wrapper)

        else:
            raise NotImplementedError("vmap is not supported for the NumPy backend.")

    if fun is None:
        return decorator
    return decorator(fun)

