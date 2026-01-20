"""A `jax.numpy` interface for `algebraic`.

This module provides a drop-in replacement for `jax.numpy` that automatically
wraps functions with `quax.quaxify` to ensure that `AlgebraicArray`s are correctly
handled. Users can import this module instead of `jax.numpy` and all operations
will work seamlessly with `AlgebraicArray` without needing explicit `quax.quaxify`
calls.

Example:
    import algebraic.numpy as anp
    from algebraic.array.core import zeros

    a = zeros((3, 3), semiring)
    b = zeros((3, 3), semiring)
    c = anp.matmul(a, b)  # No quax.quaxify needed!

Note:
    Array creation functions (zeros, ones, etc.) are not available in this module.
    Use `algebraic.array.core` for creating AlgebraicArray instances, as they
    require a `semiring` argument.
"""
# ruff: noqa: F822

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

import jax.numpy as jnp
import quax
from jaxtyping import ArrayLike, Shaped
from typing_extensions import overload

from algebraic.spec import Semiring, Shape

if TYPE_CHECKING:
    from algebraic import AlgebraicArray


def array[K: Semiring](data: Shaped[ArrayLike | AlgebraicArray[K], "..."], dtype: K) -> Shaped[AlgebraicArray[K], "..."]:
    from algebraic import AlgebraicArray

    return AlgebraicArray(data, dtype)


def asarray[K: Semiring](data: Shaped[ArrayLike | AlgebraicArray[K], "..."], dtype: K) -> Shaped[AlgebraicArray[K], "..."]:
    return array(data, dtype)


@overload
def zeros[K: Semiring](shape: int, dtype: K) -> Shaped[AlgebraicArray[K], " {shape}"]: ...
@overload
def zeros[K: Semiring](shape: tuple[int, ...], dtype: K) -> Shaped[AlgebraicArray[K], " {*shape}"]: ...


def zeros[K: Semiring](shape: Shape, dtype: K) -> Shaped[AlgebraicArray[K], "*shape"]:
    """Return an array of given shape filled with the additive identity (zero)"""
    from algebraic import AlgebraicArray

    return AlgebraicArray(jnp.full(shape, dtype.zero), dtype)


@overload
def ones[K: Semiring](shape: int, dtype: K) -> Shaped[AlgebraicArray[K], " {shape}"]: ...
@overload
def ones[K: Semiring](shape: tuple[int, ...], dtype: K) -> Shaped[AlgebraicArray[K], " {*shape}"]: ...


def ones[K: Semiring](shape: Shape, dtype: K) -> Shaped[AlgebraicArray[K], "*shape"]:
    """Return an array of given shape filled with the additive identity (one)"""
    from algebraic import AlgebraicArray

    return AlgebraicArray(jnp.full(shape, dtype.one), dtype)


def eye[K: Semiring](N: int, M: int | None = None, k: int | ArrayLike = 0, *, dtype: K) -> AlgebraicArray[K]:  # noqa: N803
    """Create a square or rectangular identity matrix with 0 and 1 from the semiring"""
    if M is None:
        M = N  # noqa: N806
    ret = zeros((N, M), dtype)
    mask = jnp.eye(N, M, k=k, dtype=jnp.bool_)
    ret = ret.at[mask].set(dtype.one)
    return ret


# Whitelist of methods that should be wrapped.
_JAX_NUMPY_WHITELIST: tuple[str, ...] = (
    "add",
    "all",
    "allclose",
    "any",
    "append",
    "apply_along_axis",
    "apply_over_axes",
    "argsort",
    "argwhere",
    "array_equal",
    "array_equiv",
    "array_repr",
    "array_split",
    "array_str",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "block",
    "broadcast_arrays",
    "broadcast_shapes",
    "broadcast_to",
    "choose",
    "column_stack",
    "compress",
    "concat",
    "concatenate",
    "copy",
    "copysign",
    "cross",
    "cumprod",
    "cumsum",
    "cumulative_prod",
    "cumulative_sum",
    "delete",
    "diag",
    "diag_indices",
    "diag_indices_from",
    "diagflat",
    "diagonal",
    "dot",
    "dsplit",
    "dstack",
    "einsum",
    "einsum_path",
    "equal",
    "expand_dims",
    "extract",
    "fill_diagonal",
    "flip",
    "fliplr",
    "flipud",
    "hsplit",
    "hstack",
    "inner",
    "insert",
    "isclose",
    "kron",
    "matmul",
    "matrix_transpose",
    "matvec",
    "moveaxis",
    "multiply",
    "nan_to_num",
    "nancumprod",
    "nancumsum",
    "nanprod",
    "nansum",
    "not_equal",
    "outer",
    "packbits",
    "pad",
    "partition",
    "permute_dims",
    "piecewise",
    "place",
    "prod",
    "put",
    "put_along_axis",
    "ravel",
    "ravel_multi_index",
    "repeat",
    "reshape",
    "resize",
    "right_shift",
    "roll",
    "rollaxis",
    "select",
    "split",
    "squeeze",
    "stack",
    "subtract",
    "sum",
    "swapaxes",
    "take",
    "take_along_axis",
    "tensordot",
    "tile",
    "transpose",
    "unique",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "unpackbits",
    "unravel_index",
    "unstack",
    "vdot",
    "vecdot",
    "vecmat",
    "vsplit",
    "vstack",
    "where",
)

# Specialize some where you just want the arguments quaxed, but not the return value
_RET_PASSTHROUGH = {
    "array_equal",
    "array_equiv",
    "allclose",
    "argsort",
    "argwhere",
    "ravel_multi_index",
    "unravel_index",
    "diag_indices",
    "diag_indices_from",
}

# Cache for wrapped functions to avoid re-wrapping on every call
_wrapped_cache: dict[str, object] = {}


def __getattr__(attr: str) -> object:
    """Dynamically wrap jax.numpy attributes with quax.quaxify.

    This function is called when an attribute is accessed that isn't explicitly
    defined in this module. It retrieves the attribute from jax.numpy and, if
    it's callable, wraps it with quax.quaxify to handle AlgebraicArray instances.

    Array creation functions are explicitly excluded and will raise AttributeError
    with a helpful message pointing to algebraic.array.core.

    Args:
        attr: The name of the attribute to retrieve.

    Returns:
        The wrapped function (if callable) or the attribute as-is (if not callable).

    Raises:
        AttributeError: If the attribute doesn't exist in jax.numpy or is excluded.
    """
    # Check if this is an excluded array creation function
    if attr not in _JAX_NUMPY_WHITELIST:
        raise AttributeError(f"'{attr}' is not available in algebraic.numpy.")

    # Check cache first for performance
    if attr in _wrapped_cache:
        return _wrapped_cache[attr]

    # Get the attribute from jax.numpy
    jnp_attr = getattr(jnp, attr)
    # If it's callable, wrap it with quaxify for AlgebraicArray support
    if callable(jnp_attr):
        if attr in _RET_PASSTHROUGH:
            _wrapped_cache[attr] = _quax_ret_passthrough(jnp_attr)
        else:
            _wrapped_cache[attr] = quax.quaxify(jnp_attr)
    else:
        # For constants/non-callables, return as-is
        _wrapped_cache[attr] = jnp_attr
    return _wrapped_cache[attr]


def _quax_ret_passthrough(fn: Callable[..., object]) -> object:
    from algebraic import AlgebraicArray

    wrapped = quax.quaxify(fn)

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> object:
        r = wrapped(*args, **kwargs)
        if isinstance(r, tuple):
            return tuple(r.data if isinstance(r, AlgebraicArray) else r)
        if isinstance(r, AlgebraicArray):
            return r.data
        return r

    return wrapper


__all__: tuple[str, ...] = _JAX_NUMPY_WHITELIST + (
    "AlgebraicArray",
    "zeros",
    "ones",
    "eye",
)
