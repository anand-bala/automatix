import functools
from typing import Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Num
from typing_extensions import TypeAlias

_Axis: TypeAlias = Union[None, int, tuple[int, ...]]
_ArrayLike: TypeAlias = Num[ArrayLike, "..."]
_Array: TypeAlias = Num[Array, "..."]


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def logsumexp(a: _ArrayLike, axis: _Axis = None) -> _Array:
    r"""Implementation of `logsumexp` that generates correct gradients for arguments containing all $-\infty$.

    Derived from [this comment in `jax` issue #6811](https://github.com/google/jax/issues/6811#issuecomment-986265534).
    """
    return _logsumexp_fwd(a, axis)[0]


def _logsumexp_fwd(a: _ArrayLike, axis: _Axis) -> tuple[_Array, tuple[_Array, _Array]]:
    c = jnp.max(a, axis=axis, keepdims=True)
    safe = jnp.isfinite(c)
    c = jnp.where(safe, c, 0)
    e = jnp.exp(a - c)
    z = jnp.sum(e, axis=axis, keepdims=True)
    r = jnp.squeeze(c, axis=axis) + jnp.log(jnp.squeeze(z, axis=axis))
    return r, (e, z)


def _logsumexp_bwd(axis: _Axis, res: tuple[_ArrayLike, _ArrayLike], g: _ArrayLike) -> tuple[_Array]:
    e = jnp.asarray(res[0])
    z = jnp.asarray(res[1])
    g = jnp.asarray(g)
    safe = z != 0
    z = jnp.where(safe, z, 1)
    if axis is not None:
        g = jnp.expand_dims(g, axis=axis)
    return (g / z * e,)


logsumexp.defvjp(_logsumexp_fwd, _logsumexp_bwd)
