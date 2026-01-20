"""JAX transformations wrapper for algebraic arrays.

This module provides wrapped versions of common JAX transformations (jit, vmap) that
automatically handle AlgebraicArray instances via quax.quaxify.
Users can import these functions instead of the JAX versions and avoid manual quaxify
calls.

Example:
    from algebraic._jax_wrappers import jit, vmap
    from algebraic.array.core import zeros
    from algebraic.semirings import counting_semiring

    @jit
    def compute(x):
        return x + x

    @vmap
    def batch_compute(xs):
        return xs * xs

    semiring = counting_semiring()
    x = zeros((3, 3), semiring)
    result = compute(x)  # Works seamlessly with AlgebraicArray
"""
# mypy: disable-error-code="misc"

import functools
import typing
from collections.abc import Callable, Hashable, Iterable, Sequence

import jax
import jaxlib.xla_client as xc
import quax
from jax.sharding import Sharding
from jaxtyping import PyTree
from typing_extensions import overload

_FnParams = typing.ParamSpec("_FnParams")
_ReturnType = typing.TypeVar("_ReturnType")


@overload
def jit(
    fun: Callable[_FnParams, _ReturnType],
    /,
    in_shardings: None | Sharding | PyTree[Sharding] = ...,
    out_shardings: None | Sharding | PyTree[Sharding] = ...,
    static_argnums: int | Sequence[int] | None = ...,
    static_argnames: str | Iterable[str] | None = ...,
    donate_argnums: int | Sequence[int] | None = ...,
    donate_argnames: str | Iterable[str] | None = ...,
    keep_unused: bool = ...,
    device: xc.Device | None = ...,
    backend: str | None = ...,
    inline: bool = ...,
    compiler_options: dict[str, typing.Any] | None = ...,
) -> Callable[_FnParams, _ReturnType]: ...


@overload
def jit(
    *,
    in_shardings: None | Sharding | PyTree[Sharding] = ...,
    out_shardings: None | Sharding | PyTree[Sharding] = ...,
    static_argnums: int | Sequence[int] | None = ...,
    static_argnames: str | Iterable[str] | None = ...,
    donate_argnums: int | Sequence[int] | None = ...,
    donate_argnames: str | Iterable[str] | None = ...,
    keep_unused: bool = ...,
    device: xc.Device | None = ...,
    backend: str | None = ...,
    inline: bool = ...,
    compiler_options: dict[str, typing.Any] | None = ...,
) -> Callable[[Callable[_FnParams, _ReturnType]], Callable[_FnParams, _ReturnType]]: ...


def jit(
    fun: Callable[_FnParams, _ReturnType] | None = None,
    *,
    in_shardings: None | Sharding | PyTree[Sharding] = None,
    out_shardings: None | Sharding | PyTree[Sharding] = None,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    keep_unused: bool = False,
    device: xc.Device | None = None,
    backend: str | None = None,
    inline: bool = False,
    compiler_options: dict[str, typing.Any] | None = None,
) -> Callable[_FnParams, _ReturnType] | Callable[[Callable[_FnParams, _ReturnType]], Callable[_FnParams, _ReturnType]]:
    """JIT compilation with automatic AlgebraicArray support.

    This is a wrapped version of jax.jit that automatically handles AlgebraicArray
    instances via quax.quaxify.

    See `jax.jit` documentation for parameter details.
    """

    def decorator(f: Callable[_FnParams, _ReturnType]) -> Callable[_FnParams, _ReturnType]:
        jitted = jax.jit(
            f,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            donate_argnums=donate_argnums,
            donate_argnames=donate_argnames,
            keep_unused=keep_unused,
            device=device,
            backend=backend,
            inline=inline,
            compiler_options=compiler_options,
        )
        wrapped = quax.quaxify(jitted)

        @functools.wraps(f)
        def wrapper(*args: _FnParams.args, **kwargs: _FnParams.kwargs) -> _ReturnType:
            return typing.cast(_ReturnType, wrapped(*args, **kwargs))

        return wrapper

    if fun is None:
        return decorator
    return decorator(fun)


@overload
def vmap(
    fun: Callable[_FnParams, _ReturnType],
    /,
    in_axes: int | None | Sequence[PyTree[int | None]] = ...,
    out_axes: int | None | Sequence[PyTree[int | None]] = ...,
    axis_name: Hashable | None = ...,
    axis_size: int | None = ...,
    spmd_axis_name: Hashable | tuple[Hashable, ...] | None = ...,
) -> Callable[_FnParams, _ReturnType]: ...


@overload
def vmap(
    *,
    in_axes: int | None | Sequence[PyTree[int | None]] = ...,
    out_axes: int | None | Sequence[PyTree[int | None]] = ...,
    axis_name: Hashable | None = ...,
    axis_size: int | None = ...,
    spmd_axis_name: Hashable | tuple[Hashable, ...] | None = ...,
) -> Callable[[Callable[_FnParams, _ReturnType]], Callable[_FnParams, _ReturnType]]: ...


def vmap(
    fun: Callable[_FnParams, _ReturnType] | None = None,
    *,
    in_axes: int | None | Sequence[PyTree[int | None]] = 0,
    out_axes: int | None | Sequence[PyTree[int | None]] = 0,
    axis_name: Hashable | None = None,
    axis_size: int | None = None,
    spmd_axis_name: Hashable | tuple[Hashable, ...] | None = None,
) -> Callable[_FnParams, _ReturnType] | Callable[[Callable[_FnParams, _ReturnType]], Callable[_FnParams, _ReturnType]]:
    """Vectorizing map with automatic AlgebraicArray support.

    This is a wrapped version of jax.vmap that automatically handles AlgebraicArray
    instances via quax.quaxify.

    see `jax.vmap` documentation for parameter details.
    """

    def decorator(f: Callable[_FnParams, _ReturnType]) -> Callable[_FnParams, _ReturnType]:
        vmapped = jax.vmap(
            f,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
            spmd_axis_name=spmd_axis_name,
        )
        wrapped = quax.quaxify(vmapped)

        @functools.wraps(f)
        def wrapper(*args: _FnParams.args, **kwargs: _FnParams.kwargs) -> _ReturnType:
            return wrapped(*args, **kwargs)

        return wrapper

    if fun is None:
        return decorator
    return decorator(fun)
