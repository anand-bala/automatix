"""JAX-specific kernel implementations.

Provides a numerically stable ``logaddexp`` with ``custom_vjp`` as the
fundamental primitive, plus ``logsumexp`` built as a reduction over it.
Also includes smooth max/min, sigmoid-based boolean ops, and
boolean-sum reductions with custom backward passes.
"""
# mypy: disable-error-code="no-untyped-call, no-any-return"

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, Num
from typing_extensions import TypeAlias

_Axis: TypeAlias = None | int | Sequence[int]
_Array: TypeAlias = Num[Array, "..."]


@jax.custom_vjp
def logaddexp(x: _Array, y: _Array) -> _Array:
    r"""Numerically stable ``log(exp(x) + exp(y))`` with correct gradients
    when both arguments are :math:`-\infty`.

    ``jnp.logaddexp`` produces NaN gradients in the both-``-inf`` case;
    this implementation returns zero gradients instead.
    """
    return _logaddexp_fwd(x, y)[0]


def _logaddexp_fwd(x: _Array, y: _Array) -> tuple[_Array, tuple[_Array, _Array]]:
    result = jnp.logaddexp(x, y)
    return result, (x, y)


def _logaddexp_bwd(res: tuple[_Array, _Array], g: _Array) -> tuple[_Array, _Array]:
    x, y = res
    # softmax weights: exp(x) / (exp(x) + exp(y)) and exp(y) / (exp(x) + exp(y))
    # When both are -inf, both weights should be 0.
    diff_xy = x - y
    diff_yx = y - x
    # sigmoid(-inf) = 0, sigmoid(inf) = 1, sigmoid(nan) would be nan
    # jnp.where guards against nan from inf - inf
    wx = jnp.where(jnp.isfinite(diff_xy), jax.nn.sigmoid(diff_xy), jnp.where(x > y, 1.0, 0.0))
    wy = jnp.where(jnp.isfinite(diff_yx), jax.nn.sigmoid(diff_yx), jnp.where(y > x, 1.0, 0.0))
    return (g * wx, g * wy)


logaddexp.defvjp(_logaddexp_fwd, _logaddexp_bwd)


def logsumexp(a: _Array, axis: _Axis = None) -> _Array:
    r"""Numerically stable ``logsumexp`` built as a reduction of :func:`logaddexp`."""
    a = jnp.asarray(a)
    return jax.lax.reduce(a, jnp.array(-jnp.inf, dtype=a.dtype), logaddexp, dimensions=_normalize_reduce_axes(axis, a.ndim))


def _normalize_reduce_axes(axis: _Axis, ndim: int) -> tuple[int, ...]:
    """Convert *axis* to a tuple of non-negative dimension indices."""
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        return (axis % ndim,)
    return tuple(ax % ndim for ax in axis)


def smooth_boolean_and(
    x: _Array,
    y: _Array,
    temperature: float = 1.0,
) -> _Array:
    """Smooth Boolean AND using sigmoid approximation.

    Formula: ``smooth_and(x,y) = sigmoid(temperature * (x + y - 1))``
    """
    return jax.nn.sigmoid(temperature * (x + y - 1))


def smooth_boolean_or(
    x: _Array,
    y: _Array,
    temperature: float = 1.0,
) -> _Array:
    """Smooth Boolean OR using sigmoid approximation.

    Formula: ``smooth_or(x,y) = sigmoid(temperature * (x + y))``
    """
    return jax.nn.sigmoid(temperature * (x + y))


def smooth_boolean_not(
    x: _Array,
    temperature: float = 1.0,
) -> _Array:
    """Smooth Boolean negation using sigmoid approximation.

    Formula: ``smooth_not(x) = sigmoid(temperature * (0.5 - x))``
    """
    return jax.nn.sigmoid(temperature * (0.5 - x))


def smooth_maximum(x: _Array, y: _Array, temperature: float = 1.0) -> _Array:
    r"""Smooth approximation of ``max(x, y)`` using :func:`logaddexp`.

    Formula: ``smooth_max(x, y) = (1/T) * logaddexp(T*x, T*y)``
    """
    return logaddexp(temperature * x, temperature * y) / temperature


def smooth_max(x: _Array, axis: _Axis = None, temperature: float = 1.0) -> _Array:
    r"""Smooth max reduction using logsumexp scaled by temperature."""
    x = jnp.asarray(x)
    return logsumexp(temperature * x, axis=axis) / temperature


def smooth_minimum(x: _Array, y: _Array, temperature: float = 1.0) -> _Array:
    r"""Smooth approximation of ``min(x, y)`` using negated :func:`logaddexp`."""
    return -logaddexp(-temperature * x, -temperature * y) / temperature


def smooth_min(x: _Array, axis: _Axis = None, temperature: float = 1.0) -> _Array:
    r"""Smooth min reduction using negated logsumexp."""
    x = jnp.asarray(x)
    return -logsumexp(-temperature * x, axis=axis) / temperature
