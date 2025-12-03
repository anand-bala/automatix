from collections.abc import Iterable

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Shaped
from typing_extensions import overload


@overload
def normalize_axis(axis: None, ndim: int) -> None: ...
@overload
def normalize_axis(axis: int, ndim: int) -> int: ...
@overload
def normalize_axis(axis: Iterable[int], ndim: int) -> Iterable[int]: ...


def normalize_axis(axis: int | Iterable[int] | None, ndim: int) -> int | Iterable[int] | None:
    """
    Normalize negative axis indices to their positive counterpart for a given
    number of dimensions.
    """
    if axis is None:
        return None

    if isinstance(axis, int):
        if axis < 0:
            axis += ndim

        if axis >= ndim or axis < 0:
            raise ValueError(f"Invalid axis index {axis} for {ndim=}")

        return axis

    if isinstance(axis, Iterable):
        # Strings are iterable but not valid axis specifications
        if isinstance(axis, str):
            raise ValueError(f"axis {axis} not understood")
        # Recursive call will validate each element is an int
        return tuple(normalize_axis(a, ndim) for a in axis)

    raise ValueError(f"axis {axis} not understood")


def equivalent(x: Shaped[ArrayLike, "#*n"], y: Shaped[ArrayLike, "#*n"], /, loose: bool = False) -> Bool[Array, "#*n"]:
    """The element-wise comparison of where two arrays are equivalent.

    Checks the equivalence of two scalars or arrays with broadcasting. Assumes
    a consistent dtype.

    Examples
    --------
    >>> equivalent(1, 1)
    np.True_
    >>> equivalent(jnp.nan, jnp.nan + 1)
    np.True_
    >>> equivalent(1, 2)
    np.False_
    >>> equivalent(jnp.inf, jnp.inf)
    np.True_
    >>> equivalent(jnp.float64(0.0), jnp.float64(-0.0))
    np.False_
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    # Can't contain NaNs
    dt = jnp.result_type(x.dtype, y.dtype)
    if not any(jnp.issubdtype(dt, t) for t in [jnp.floating, jnp.complexfloating]):
        return x == y

    if loose:
        if jnp.issubdtype(dt, jnp.complexfloating):
            return equivalent(x.real, y.real, loose=True) & equivalent(x.imag, y.imag, loose=True)

        # TODO: Rec array handling
        return (x == y) | ((x != x) & (y != y))

    if x.size == 0 or y.size == 0:
        shape = jnp.broadcast_shapes(x.shape, y.shape)
        return jnp.empty(shape, dtype=jnp.bool_)
    x, y = jnp.broadcast_arrays(x[..., None], y[..., None])
    return (x.astype(dt).view(jnp.uint8) == y.astype(dt).view(jnp.uint8)).all(axis=-1)
