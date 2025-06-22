from abc import abstractmethod
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Num
from typing_extensions import TypeAlias, override

from .utils.logsumexp import logsumexp

Axis: TypeAlias = Union[None, int, tuple[int, ...]]
Shape: TypeAlias = Union[int, tuple[int, ...]]

# pyright: reportIncompatibleMethodOverride=false


class AbstractSemiring(eqx.Module, strict=True):
    """Base semiring class."""

    @staticmethod
    @abstractmethod
    def zeros(shape: Shape) -> Num[Array, "..."]:
        """Return a new array of given shape and type, filled with abstract zeros defined by the semiring.

        **Parameters:**

        - `shape`: Shape of the new array, e.g., `(2, 3)` or `2`.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def ones(shape: Shape) -> Num[Array, "..."]:
        """Return a new array of given shape and type, filled with abstract ones defined by the semiring.

        **Parameters**

        - `shape`: Shape of the new array, e.g., `(2, 3)` or `2`.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def add(cls, x1: Num[Array, " n"], x2: Num[Array, " n"]) -> Num[Array, " n"]:
        """Add arguments element-wise using semiring addition.

        **Parameters:**

        - `x1`:  The arrays to be added. If `x1.shape != x2.shape`, they must be broadcastable to a common shape (which becomes the shape of the output).
        - `x2`:  The arrays to be added. If `x1.shape != x2.shape`, they must be broadcastable to a common shape (which becomes the shape of the output).
        """
        ...

    @classmethod
    @abstractmethod
    def multiply(cls, x1: Num[Array, " n"], x2: Num[Array, " n"]) -> Num[Array, " n"]:
        """Multiply arguments element-wise using semiring multiply.

        **Parameters:**

        - `x1`:  The arrays to be multiplied. If `x1.shape != x2.shape`, they must be broadcastable to a common shape (which becomes the shape of the output).
        - `x2`:  The arrays to be multiplied. If `x1.shape != x2.shape`, they must be broadcastable to a common shape (which becomes the shape of the output).
        """
        ...

    @classmethod
    @abstractmethod
    def sum(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        """
        Returns the sum w.r.t semiring addition over a given axis.

        **Parameters:**

        - `a`: Elements to sum
        - `axis`: Axis or axes along which a sum is performed. The default, `axis=None`, will sum all of the elements of the input array. If axis is negative it counts from the last to the first axis.
        """
        ...

    @classmethod
    @abstractmethod
    def prod(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        """
        Returns the product w.r.t semiring addition over a given axis.

        **Parameters:**

        - `a`: Input array
        - `axis`: Axis or axes along which a product is performed. The default, `axis=None`, will product all of the elements of the input array. If axis is negative it counts from the last to the first axis.
        """
        ...

    @classmethod
    def vdot(cls, a: Num[Array, " n"], b: Num[Array, " n"]) -> Num[Array, ""]:
        """Computes the dot product of two 1D tensors using the semiring.

        **Parameters:**

        - `a`, `b`: input arrays
        """
        return cls.sum(cls.multiply(a, b))

    @classmethod
    def matmul(cls, a: Num[Array, "n k"], b: Num[Array, "k m"]) -> Num[Array, "n m"]:
        """Matrix-semiring product of two arrays."""
        mv = jax.vmap(cls.vdot, (0, None), 0)
        mm = jax.vmap(mv, (None, 1), 1)
        c: Num[Array, "n m"] = jax.jit(mm)(a, b)
        return c


class AbstractNegation(eqx.Module, strict=True):
    """A negation function on `S` is an involution on `S`"""

    @classmethod
    @abstractmethod
    def negate(cls, x: Num[Array, "*size"]) -> Num[Array, "*size"]:
        """An involution in the algebra"""


class CountingSemiring(AbstractSemiring):
    r"""Implementation of the counting semiring $(\mathbb{R}, +, \times, 0, 1)$."""

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
    r"""Implementation of the min-max semiring $(\mathbb{R}_{\leq 0} \cup \{-\infty, \infty\}, \max, \min, -\infty, 0)$."""

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
        return jnp.minimum(x1, x2)

    @override
    @classmethod
    def sum(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        return jnp.amax(a, axis=axis)

    @override
    @classmethod
    def prod(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        return jnp.amin(a, axis=axis)


class LSEMaxMinSemiring(AbstractSemiring):
    r"""Implementation of the smooth min-max semiring using `logsumexp` $(\mathbb{R}_{\leq 0} \cup \{-\infty, \infty\}, logsumexp, -logsumexp, -\infty, 0)$."""

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
        return -logsumexp(jnp.stack([-x1, -x2], axis=-1), axis=-1)

    @override
    @classmethod
    def sum(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        return logsumexp(a, axis=axis)

    @override
    @classmethod
    def prod(cls, a: Num[Array, " ..."], axis: Axis = None) -> Num[Array, " ..."]:
        return -logsumexp(-a, axis=axis)


class MaxPlusSemiring(AbstractSemiring):
    r"""Implementation of the max-plus tropical semiring $(\mathbb{R}_{\leq 0} \cup \{-\infty,\infty\}, \max, +, -\infty, 0)$."""

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


class LogSemiring(AbstractSemiring):
    r"""Implementation of the log semiring $(\mathbb{R}_{\leq 0} \cup \{-\infty, \infty\}, logsumexp, +, -\infty, 0)$."""

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


class LatticeAlgebra(AbstractSemiring, AbstractNegation):
    """A simple lattice algebra on (1, 0, max, min, 1- x)"""

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
