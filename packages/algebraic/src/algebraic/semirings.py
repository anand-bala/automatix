# mypy: disable-error-code="no-any-return"
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Shaped
from typing_extensions import overload

import algebraic.kernels as kernels
from algebraic.spec import BinaryOp, BooleanAlgebra, DeMorganAlgebra, MaybeAxis, Semiring
from algebraic.spec import BoundedDistributiveLattice as Lattice


def counting_semiring() -> Semiring:
    r"""Implementation of the counting semiring (R, +, *, 0, 1)."""

    def add(x1: Shaped[Array, "*#n"], x2: Shaped[Array, "*#n"]) -> Shaped[Array, "*#n"]:
        return x1 + x2

    def multiply(x1: Shaped[Array, "*#n"], x2: Shaped[Array, "*#n"]) -> Shaped[Array, "*#n"]:
        return x1 * x2

    return Semiring(
        add=add,
        mul=multiply,
        zero=jnp.asarray(0.0),
        one=jnp.asarray(1.0),
    )


@overload
def max_min_algebra(
    *,
    smooth: bool = False,
    only: None = None,
    temperature: float = 1.0,
) -> DeMorganAlgebra: ...


@overload
def max_min_algebra(
    *,
    smooth: bool,
    only: Literal["negative", "positive"],
    temperature: float,
) -> Lattice: ...


def max_min_algebra(
    *,
    smooth: bool = False,
    only: None | Literal["negative", "positive"] = None,
    temperature: float = 1.0,
) -> Lattice | DeMorganAlgebra:
    """Implementation of the min-max semiring on reals (R cup {-inf, inf}, max, min, -inf, inf).

    Parameters
    ----------
    smooth : bool
        If `True`, use the logsumexp approximation of max and min.
    only : "negative", "positive", None (default)
        Restrict the semiring to either the negative or positive extended reals. If
        `None`, returns a full complemented max-min algebra (with negation).
    temperature : float, default 1.0
        Temperature closer to infinity is closer to true max/min

    """
    add_kernel: BinaryOp
    mul_kernel: BinaryOp

    if smooth:

        def add_kernel(a: Array, b: Array) -> Array:
            return kernels.smooth_maximum(a, b, temperature=temperature)

        def mul_kernel(a: Array, b: Array) -> Array:
            return kernels.smooth_minimum(a, b, temperature=temperature)

    else:
        add_kernel = jnp.maximum
        mul_kernel = jnp.minimum

    zero = jnp.asarray(0.0 if only == "positive" else -jnp.inf)
    one = jnp.asarray(-0.0 if only == "negative" else jnp.inf)

    def add(x1: Shaped[Array, "*#n"], x2: Shaped[Array, "*#n"]) -> Shaped[Array, "*#n"]:
        return add_kernel(x1, x2)

    def multiply(x1: Shaped[Array, "*#n"], x2: Shaped[Array, "*#n"]) -> Shaped[Array, "*#n"]:
        return mul_kernel(x1, x2)

    def complement(x: Shaped[Array, " ..."]) -> Shaped[Array, " ..."]:
        return -x

    if only is None:
        # We can return complemented algebra
        return DeMorganAlgebra(
            add=add,
            mul=multiply,
            zero=zero,
            one=one,
            complement=complement,
        )
    else:
        return Lattice(
            add=add,
            mul=multiply,
            zero=zero,
            one=one,
        )


def tropical_semiring(*, minplus: bool = True, smooth: bool = False, temperature: float = 1.0) -> Semiring:
    """The min-plus tropical semiring

    The choice of `minplus` determines if the output is the min-plus semiring (R_>=0 cup
    {-inf, inf}, min, +, inf, 0) or the max-plus tropical semiring (R_<=0 cup {-inf,
    inf}, max, +, -inf, 0).

    Parameters
    ----------
    minplus: bool
        If `True`, returns the min-plus tropical semiring. Else, the maxplus semiring.
    smooth : bool
        If `True`, use the logsumexp approximation of max and min.
    only : "negative", "positive", None (default)
        Restrict the semiring to either the negative or positive extended reals. If
        `None`, returns a full complemented max-min algebra (with negation).
    temperature : float, default 1.0
        Temperature for the smooth approximation; closer to infinity is closer to true max/min
    """
    add_kernel: BinaryOp
    # sum_kernel: ReductionOp
    if smooth:
        if minplus:

            def add_kernel(a: Array, b: Array) -> Array:
                return kernels.smooth_minimum(a, b, temperature=temperature)

            def sum_kernel(a: Array, axis: MaybeAxis) -> Array:
                return kernels.smooth_min(a, axis, temperature)
        else:

            def add_kernel(a: Array, b: Array) -> Array:
                return kernels.smooth_maximum(a, b, temperature=temperature)

            def sum_kernel(a: Array, axis: MaybeAxis) -> Array:
                return kernels.smooth_max(a, axis, temperature)
    else:
        if minplus:
            add_kernel = jnp.minimum
            sum_kernel = jnp.min
        else:
            add_kernel = jnp.maximum
            sum_kernel = jnp.max

    if minplus:
        zero = jnp.asarray(jnp.inf)
        one = jnp.asarray(0.0)
    else:
        zero = jnp.asarray(-jnp.inf)
        one = jnp.asarray(-0.0)

    def add(x1: Shaped[Array, "*#n"], x2: Shaped[Array, "*#n"]) -> Shaped[Array, "*#n"]:
        return add_kernel(x1, x2)

    def sum(a: Shaped[Array, " ..."], axis: MaybeAxis = None) -> Shaped[Array, " ..."]:
        return sum_kernel(a, axis)

    def multiply(x1: Shaped[Array, "*#n"], x2: Shaped[Array, "*#n"]) -> Shaped[Array, "*#n"]:
        return x1 + x2

    def prod(a: Shaped[Array, " ..."], axis: MaybeAxis = None) -> Shaped[Array, " ..."]:
        return jnp.sum(a, axis=axis)

    return Semiring(
        add=add,
        mul=multiply,
        zero=zero,
        one=one,
        properties={"idempotent_add", "commutative", "simple"},
    )


def boolean_algebra(
    mode: Literal["logic", "soft", "smooth", "ste"] = "soft",
    temperature: float = 1.0,
) -> BooleanAlgebra:
    """Create a differentiable Boolean kernel.

    Parameters
    ----------
    mode : {"logic", "soft", "smooth", "ste"}
        Differentiation mode:
        - "logic": non-differentiable
        - "soft": Soft Boolean using multiplication and addition (fastest, smoothest)
        - "smooth": Smooth Boolean using sigmoid with temperature
        - "ste": Straight-Through Estimator (biased gradients, but works generally)
    temperature : float, optional
        Temperature parameter for "smooth" mode (default: 1.0)


    Notes
    -----
    The differentiable modes work best with inputs in [0,1] closer to the boundaries.
    """

    zero = jnp.asarray(0.0)
    one = jnp.asarray(1.0)

    match mode:
        case "logic":
            add = jnp.logical_or
            mul = jnp.logical_and
            neg = jnp.logical_not
        case "soft":
            add = kernels.soft_boolean_or
            mul = kernels.soft_boolean_and
            neg = kernels.soft_boolean_not
        case "smooth":

            def add(a: Array, b: Array) -> Array:
                return kernels.smooth_boolean_or(a, b, temperature=temperature)

            def mul(a: Array, b: Array) -> Array:
                return kernels.smooth_boolean_and(a, b, temperature=temperature)

            def neg(a: Array) -> Array:
                return kernels.smooth_boolean_not(a, temperature)

        case "ste":
            add = jnp.max
            mul = jnp.min

            def neg(a: Array) -> Array:
                return 1 - a
        case _:
            raise ValueError(f"Unknown mode: {mode}. Use 'logic', 'soft', 'smooth', or 'ste'.")
    return BooleanAlgebra(
        zero=zero,
        one=one,
        add=add,
        mul=mul,
        complement=neg,
    )
