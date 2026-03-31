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
            import torch

            return torch.vmap(f, in_dims=in_axes, out_dims=out_axes)  # type: ignore[arg-type]

        else:
            raise NotImplementedError("vmap is not supported for the NumPy backend.")

    if fun is None:
        return decorator
    return decorator(fun)


def _numpy_vmap(
    fun: Callable[_FnParams, _ReturnType] | None = None,
    *,
    in_axes: int | None | Sequence[int | None] = 0,
    out_axes: int | None | Sequence[int | None] = 0,
) -> Callable[_FnParams, _ReturnType] | Callable[[Callable[_FnParams, _ReturnType]], Callable[_FnParams, _ReturnType]]:
    """
    A braindead NumPy implementation of jax.vmap.

    Args:
        fun: function to map
        in_axes: int, None, or tuple matching args
        out_axes: int, or tuple matching outputs

    Returns:
        vectorized function
    """
    import numpy as np

    def normalize_in_axes(in_axes, args):
        if isinstance(in_axes, tuple):
            assert len(in_axes) == len(args)
            return in_axes
        else:
            return (in_axes,) * len(args)

    def move_axis_to_front(x, axis):
        if axis is None:
            return x
        return np.moveaxis(x, axis, 0)

    def move_axis_from_front(x, axis):
        if axis is None:
            return x
        return np.moveaxis(x, 0, axis)

    def vmapped(*args):
        axes = normalize_in_axes(in_axes, args)

        # Move mapped axes to front
        moved_args = []
        sizes = []

        for arg, ax in zip(args, axes):
            if ax is None:
                moved_args.append(arg)
            else:
                moved = move_axis_to_front(arg, ax)
                moved_args.append(moved)
                sizes.append(moved.shape[0])

        # Determine batch size
        if sizes:
            batch_size = sizes[0]
            assert all(s == batch_size for s in sizes), "Mismatched batch sizes"
        else:
            # No mapped axes: just call function
            return fun(*args)

        outputs = []

        for i in range(batch_size):
            sliced_args = []
            for arg, ax in zip(moved_args, axes):
                if ax is None:
                    sliced_args.append(arg)
                else:
                    sliced_args.append(arg[i])
            out = fun(*sliced_args)
            outputs.append(out)

        # Stack outputs
        def stack_outputs(outputs, out_axes):
            if isinstance(outputs[0], tuple):
                # multiple outputs
                transposed = list(zip(*outputs))
                if isinstance(out_axes, tuple):
                    return tuple(move_axis_from_front(np.stack(o, axis=0), ax) for o, ax in zip(transposed, out_axes))
                else:
                    return tuple(move_axis_from_front(np.stack(o, axis=0), out_axes) for o in transposed)
            else:
                return move_axis_from_front(np.stack(outputs, axis=0), out_axes)

        return stack_outputs(outputs, out_axes)

    return vmapped
