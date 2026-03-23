# ty: ignore[unsupported-operator]
from __future__ import annotations

import math
import operator
from collections.abc import Callable
from typing import Literal

import array_api_compat
from typing_extensions import Unpack, overload

import algebraic.kernels as kernels
from algebraic.spec import BooleanAlgebra, DeMorganAlgebra, Semiring
from algebraic.spec import BoundedDistributiveLattice as Lattice
from algebraic.types import Array, Number

type _OneOrMore[T] = tuple[T, *tuple[T, ...]]

type _ScalarFn = Callable[[Unpack[_OneOrMore[Number]]], Number]


def _try_array_else_scalar(
    first: Array | Number, /, *rest: Array | Number, array_fn: str, scalar_fn: _ScalarFn
) -> Array | Number:
    xs = (first,) + rest
    try:
        array_ns = array_api_compat.array_namespace(*xs)
    except TypeError:
        # All are scalar
        assert all(isinstance(x, Number) for x in xs)
        return scalar_fn(*xs)
    else:
        # We have an array namespace
        fn = getattr(array_ns, array_fn)
        # We need to get the common dtype
        dtype = array_ns.result_type(*xs)
        result: Array = array_ns.astype(fn(*xs), dtype, copy=False)

        return result


def counting_semiring() -> Semiring:
    r"""Implementation of the counting semiring (R, +, *, 0, 1)."""

    def add(x1: Number | Array, x2: Number | Array) -> Number | Array:
        return x1 + x2

    def multiply(x1: Number | Array, x2: Number | Array) -> Number | Array:
        return x1 * x2

    zero = 0.0

    one = 1.0

    return Semiring(
        add=add,
        mul=multiply,
        zero=zero,
        one=one,
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
    smooth: bool = False,
    only: Literal["negative", "positive"],
    temperature: float = 1.0,
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
        If `True`, use the logaddexp approximation of max and min.
    only : "negative", "positive", None (default)
        Restrict the semiring to either the negative or positive extended reals. If
        `None`, returns a full complemented max-min algebra (with negation).
    temperature : float, default 1.0
        Temperature closer to infinity is closer to true max/min

    """

    if smooth:

        def add(a: Array | Number, b: Array | Number) -> Array | Number:
            return kernels.smooth_maximum(a, b, temperature=temperature)

        def mul(a: Array | Number, b: Array | Number) -> Array | Number:
            return kernels.smooth_minimum(a, b, temperature=temperature)

    else:

        def add(a: Array | Number, b: Array | Number) -> Array | Number:
            return _try_array_else_scalar(a, b, array_fn="maximum", scalar_fn=max)

        def mul(a: Array | Number, b: Array | Number) -> Array | Number:
            return _try_array_else_scalar(a, b, array_fn="minimum", scalar_fn=min)

    zero = 0.0 if only == "positive" else -math.inf
    one = -0.0 if only == "negative" else math.inf

    def complement(x: Array | Number) -> Array | Number:
        return -x

    if only is None:
        # We can return complemented algebra
        return DeMorganAlgebra(
            add=add,
            mul=mul,
            zero=zero,
            one=one,
            complement=complement,
        )
    else:
        return Lattice(
            add=add,
            mul=mul,
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
        If `True`, use the logaddexp approximation of max and min.
    only : "negative", "positive", None (default)
        Restrict the semiring to either the negative or positive extended reals. If
        `None`, returns a full complemented max-min algebra (with negation).
    temperature : float, default 1.0
        Temperature for the smooth approximation; closer to infinity is closer to true max/min
    """
    if smooth:
        if minplus:

            def add(a: Array | Number, b: Array | Number) -> Array | Number:
                return kernels.smooth_minimum(a, b, temperature=temperature)

        else:

            def add(a: Array | Number, b: Array | Number) -> Array | Number:
                return kernels.smooth_maximum(a, b, temperature=temperature)
    else:
        if minplus:

            def add(a: Array | Number, b: Array | Number) -> Array | Number:
                return _try_array_else_scalar(a, b, array_fn="minimum", scalar_fn=min)
        else:

            def add(a: Array | Number, b: Array | Number) -> Array | Number:
                return _try_array_else_scalar(a, b, array_fn="maximum", scalar_fn=max)

    if minplus:
        zero = math.inf
        one = 0.0
    else:
        zero = -math.inf
        one = -0.0

    def multiply(x1: Array | Number, x2: Array | Number) -> Array | Number:
        return x1 + x2

    return Semiring(
        add=add,
        mul=multiply,
        zero=zero,
        one=one,
        properties={"idempotent_add", "commutative", "simple"},
    )


def boolean_algebra(
    mode: Literal["logic", "soft", "smooth", "ste", "std-fuzzy"] = "soft",
    temperature: float = 1.0,
) -> BooleanAlgebra:
    """Create a differentiable Boolean kernel.

    Parameters
    ----------
    mode : {"logic", "soft", "smooth", "ste", "std-fuzzy"}
        Differentiation mode:
        - "logic": non-differentiable
        - "soft": Soft Boolean using multiplication and addition (fastest, smoothest)
        - "smooth": Smooth Boolean using sigmoid with temperature
        - "ste"|"std-fuzzy": Straight-Through Estimator or, equivalently, the standard fuzzy algebra
    temperature : float, optional
        Temperature parameter for "smooth" mode (default: 1.0)


    Notes
    -----
    The differentiable modes work best with inputs in [0,1] closer to the boundaries.
    """

    zero = 0.0
    one = 1.0

    match mode:
        case "logic":

            def add(a: Array | Number, b: Array | Number) -> Array | Number:
                return _try_array_else_scalar(a, b, array_fn="logical_or", scalar_fn=operator.__or__)

            def mul(a: Array | Number, b: Array | Number) -> Array | Number:
                return _try_array_else_scalar(a, b, array_fn="logical_and", scalar_fn=operator.__and__)

            def neg(a: Array | Number) -> Array | Number:
                return _try_array_else_scalar(a, array_fn="logical_not", scalar_fn=operator.__not__)
        case "soft":

            def add(a: Array | Number, b: Array | Number) -> Array | Number:
                return kernels.soft_boolean_or(a, b)

            def mul(a: Array | Number, b: Array | Number) -> Array | Number:
                return kernels.soft_boolean_and(a, b)

            def neg(a: Array | Number) -> Array | Number:
                return kernels.soft_boolean_not(a)

        case "smooth":

            def add(a: Array | Number, b: Array | Number) -> Array | Number:
                return kernels.smooth_boolean_or(a, b, temperature=temperature)

            def mul(a: Array | Number, b: Array | Number) -> Array | Number:
                return kernels.smooth_boolean_and(a, b, temperature=temperature)

            def neg(a: Array | Number) -> Array | Number:
                return kernels.smooth_boolean_not(a, temperature)

        case "ste" | "std-fuzzy":

            def add(a: Array | Number, b: Array | Number) -> Array | Number:
                return _try_array_else_scalar(a, b, array_fn="maximum", scalar_fn=max)

            def mul(a: Array | Number, b: Array | Number) -> Array | Number:
                return _try_array_else_scalar(a, b, array_fn="minimum", scalar_fn=min)

            def neg(a: Array | Number) -> Array | Number:
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
