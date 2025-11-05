"""JAX-based semiring implementations.

This module contains all semiring implementations using JAX arrays and operations.
All semirings here are designed to work with jax.jit, jax.vmap, and automatic
differentiation.
"""

from typing import Union

import jax.numpy as jnp
from jaxtyping import Array, Num
from typing_extensions import TypeAlias, override

from automatix.algebra.backends.jax_kernels import logsumexp
from automatix.algebra.spec import AbstractDeMorganAlgebra, AbstractSemiring

Axis: TypeAlias = Union[None, int, tuple[int, ...]]
Shape: TypeAlias = Union[int, tuple[int, ...]]


class CountingSemiring(AbstractSemiring):
    r"""Implementation of the counting semiring (R, +, *, 0, 1)."""

    @override
    @staticmethod
    def zeros(shape: Shape) -> Num[Array, "..."]:
        return jnp.zeros(shape)

    @override
    @staticmethod
    def ones(shape: Shape) -> Num[Array, "..."]:
        return jnp.ones(shape)

    @override
    @classmethod
    def add(cls, x1: Num[Array, " n"], x2: Num[Array, " n"]) -> Num[Array, " n"]:
        return jnp.add(x1, x2)

    @override
    @classmethod
    def multiply(cls, x1: Num[Array, " n"], x2: Num[Array, " n"]) -> Num[Array, " n"]:
        return jnp.multiply(x1, x2)

    @override
    @classmethod
    def sum(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        return jnp.sum(a, axis=axis)

    @override
    @classmethod
    def prod(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        return jnp.prod(a, axis=axis)


class MaxMinSemiring(AbstractSemiring):
    r"""Implementation of the min-max semiring on reals (R cup {-inf, inf}, max, min, -inf, inf)."""

    @override
    @staticmethod
    def zeros(shape: Shape) -> Num[Array, "..."]:
        return jnp.full(shape, fill_value=-jnp.inf)

    @override
    @staticmethod
    def ones(shape: Shape) -> Num[Array, "..."]:
        return jnp.full(shape, fill_value=jnp.inf)

    @override
    @classmethod
    def add(cls, x1: Num[Array, " n"], x2: Num[Array, " n"]) -> Num[Array, " n"]:
        return jnp.maximum(x1, x2)

    @override
    @classmethod
    def multiply(cls, x1: Num[Array, " n"], x2: Num[Array, " n"]) -> Num[Array, " n"]:
        return jnp.minimum(x1, x2)

    @override
    @classmethod
    def sum(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        return jnp.amax(a, axis=axis)

    @override
    @classmethod
    def prod(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        return jnp.amin(a, axis=axis)

    is_additively_idempotent = True
    is_multiplicatively_idempotent = True
    is_commutative = True
    is_simple = True


class LeftMaxMinSemiring(MaxMinSemiring):
    r"""Implementation of the min-max semiring on negative reals (R_<=0 cup {-inf, inf}, max, min, -inf, 0)."""

    @override
    @staticmethod
    def zeros(shape: Shape) -> Num[Array, "..."]:
        return jnp.full(shape, fill_value=-jnp.inf)

    @override
    @staticmethod
    def ones(shape: Shape) -> Num[Array, "..."]:
        return jnp.full(shape, fill_value=-0.0)


class RightMaxMinSemiring(MaxMinSemiring):
    r"""Implementation of the min-max semiring on positive reals (R_>=0 cup {-inf, inf}, max, min, 0, inf)."""

    @override
    @staticmethod
    def zeros(shape: Shape) -> Num[Array, "..."]:
        return jnp.zeros(shape)

    @override
    @staticmethod
    def ones(shape: Shape) -> Num[Array, "..."]:
        return jnp.full(shape, fill_value=jnp.inf)


class MaxMinAlgebra(MaxMinSemiring, AbstractDeMorganAlgebra):
    """De Morgan algebra based on max-min semiring."""

    @override
    @classmethod
    def negate(cls, x: Num[Array, "*size"]) -> Num[Array, "*size"]:
        return -x


class LSEMaxMinSemiring(MaxMinSemiring):
    r"""Implementation of the smooth min-max semiring using logsumexp (R cup {-inf, inf}, logsumexp, -logsumexp, -inf, inf)."""

    @override
    @classmethod
    def add(cls, x1: Num[Array, " n"], x2: Num[Array, " n"]) -> Num[Array, " n"]:
        return logsumexp(jnp.stack([x1, x2], axis=-1), axis=-1)

    @override
    @classmethod
    def multiply(cls, x1: Num[Array, " n"], x2: Num[Array, " n"]) -> Num[Array, " n"]:
        return -logsumexp(jnp.stack([-x1, -x2], axis=-1), axis=-1)

    @override
    @classmethod
    def sum(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        return logsumexp(a, axis=axis)

    @override
    @classmethod
    def prod(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        return -logsumexp(-a, axis=axis)


class LeftLSEMaxMinSemiring(LSEMaxMinSemiring):
    """LSE variant for negative reals."""

    @override
    @staticmethod
    def zeros(shape: Shape) -> Num[Array, "..."]:
        return jnp.full(shape, fill_value=-jnp.inf)

    @override
    @staticmethod
    def ones(shape: Shape) -> Num[Array, "..."]:
        return jnp.full(shape, fill_value=-0.0)


class RightLSEMaxMinSemiring(LSEMaxMinSemiring):
    """LSE variant for positive reals."""

    @override
    @staticmethod
    def zeros(shape: Shape) -> Num[Array, "..."]:
        return jnp.zeros(shape)

    @override
    @staticmethod
    def ones(shape: Shape) -> Num[Array, "..."]:
        return jnp.full(shape, fill_value=jnp.inf)


class MaxPlusSemiring(AbstractSemiring):
    r"""Implementation of the max-plus tropical semiring (R_<=0 cup {-inf, inf}, max, +, -inf, 0)."""

    @override
    @staticmethod
    def zeros(shape: Shape) -> Num[Array, "..."]:
        return jnp.full(shape, fill_value=-jnp.inf)

    @override
    @staticmethod
    def ones(shape: Shape) -> Num[Array, "..."]:
        return jnp.full(shape, fill_value=-0.0)

    @override
    @classmethod
    def add(cls, x1: Num[Array, " n"], x2: Num[Array, " n"]) -> Num[Array, " n"]:
        return jnp.maximum(x1, x2)

    @override
    @classmethod
    def multiply(cls, x1: Num[Array, " n"], x2: Num[Array, " n"]) -> Num[Array, " n"]:
        return jnp.add(x1, x2)

    @override
    @classmethod
    def sum(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        return jnp.amax(a, axis=axis)

    @override
    @classmethod
    def prod(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        return jnp.sum(a, axis=axis)

    is_additively_idempotent = True
    is_commutative = True
    is_simple = True


class LogSemiring(AbstractSemiring):
    r"""Implementation of the log semiring (R_<=0 cup {-inf, inf}, logsumexp, +, -inf, 0)."""

    @override
    @staticmethod
    def zeros(shape: Shape) -> Num[Array, "..."]:
        return jnp.full(shape, fill_value=-jnp.inf)

    @override
    @staticmethod
    def ones(shape: Shape) -> Num[Array, "..."]:
        return jnp.full(shape, fill_value=-0.0)

    @override
    @classmethod
    def add(cls, x1: Num[Array, " n"], x2: Num[Array, " n"]) -> Num[Array, " n"]:
        return logsumexp(jnp.stack([x1, x2], axis=-1), axis=-1)

    @override
    @classmethod
    def multiply(cls, x1: Num[Array, " n"], x2: Num[Array, " n"]) -> Num[Array, " n"]:
        return jnp.add(x1, x2)

    @override
    @classmethod
    def sum(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        return logsumexp(a, axis=axis)

    @override
    @classmethod
    def prod(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        return jnp.sum(a, axis=axis)


class LatticeAlgebra(AbstractDeMorganAlgebra):
    """A simple lattice algebra on (1, 0, max, min, 1 - x)."""

    @override
    @staticmethod
    def zeros(shape: Shape) -> Num[Array, "..."]:
        return jnp.zeros(shape)

    @override
    @staticmethod
    def ones(shape: Shape) -> Num[Array, "..."]:
        return jnp.ones(shape)

    @override
    @classmethod
    def add(cls, x1: Num[Array, " n"], x2: Num[Array, " n"]) -> Num[Array, " n"]:
        return jnp.maximum(x1, x2)

    @override
    @classmethod
    def multiply(cls, x1: Num[Array, " n"], x2: Num[Array, " n"]) -> Num[Array, " n"]:
        return jnp.minimum(x1, x2)

    @override
    @classmethod
    def sum(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        return jnp.amax(a, axis=axis)

    @override
    @classmethod
    def prod(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        return jnp.amin(a, axis=axis)

    @override
    @classmethod
    def negate(cls, x: Num[Array, "*size"]) -> Num[Array, "*size"]:
        return 1 - x


__all__ = [
    "CountingSemiring",
    "MaxMinSemiring",
    "LeftMaxMinSemiring",
    "RightMaxMinSemiring",
    "MaxMinAlgebra",
    "LSEMaxMinSemiring",
    "LeftLSEMaxMinSemiring",
    "RightLSEMaxMinSemiring",
    "MaxPlusSemiring",
    "LogSemiring",
    "LatticeAlgebra",
]
