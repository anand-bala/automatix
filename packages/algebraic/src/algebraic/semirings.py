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
        return scalar_fn(*xs)  # type: ignore[arg-type]
    else:
        # We have an array namespace
        fn = getattr(array_ns, array_fn)
        # We need to get the common dtype
        dtype = array_ns.result_type(*xs)
        result: Array = array_ns.astype(fn(*xs), dtype, copy=False)

        return result


def counting_semiring() -> Semiring:
    r"""Create the counting semiring :math:`(\mathbb{R}, +, \times, 0, 1)`.

    The counting semiring uses standard addition and multiplication, and is
    useful for counting paths in graphs.

    Returns
    -------
    Semiring
        A :class:`~algebraic.spec.Semiring` with standard ``+`` and ``*``.

    Examples
    --------
    >>> from algebraic.semirings import counting_semiring
    >>> sr = counting_semiring()
    >>> sr.add(2.0, 3.0)
    5.0
    >>> sr.mul(2.0, 3.0)
    6.0
    """

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
    r"""Create a max-min algebra on the extended reals.

    The max-min algebra :math:`(\mathbb{R} \cup \{-\infty, \infty\}, \max, \min, -\infty, \infty)`
    is useful for robustness semantics and Signal Temporal Logic (STL).

    Parameters
    ----------
    smooth : bool
        If ``True``, use the logaddexp approximation of max and min.
    only : {"negative", "positive", None}
        Restrict the semiring to either the negative or positive extended reals. If
        ``None`` (default), returns a full complemented max-min algebra (with negation).
    temperature : float, default 1.0
        Temperature closer to infinity is closer to true max/min.

    Returns
    -------
    DeMorganAlgebra or BoundedDistributiveLattice
        A :class:`~algebraic.spec.DeMorganAlgebra` when *only* is ``None``,
        otherwise a :class:`~algebraic.spec.BoundedDistributiveLattice`.

    Examples
    --------
    >>> from algebraic.semirings import max_min_algebra
    >>> mm = max_min_algebra()
    >>> mm.add(-0.5, 0.2)
    0.2
    >>> mm.mul(-0.5, 0.2)
    -0.5
    """

    if smooth:

        def add(a: Array | Number, b: Array | Number) -> Array | Number:
            return kernels.smooth_maximum(a, b, temperature=temperature)

        def mul(a: Array | Number, b: Array | Number) -> Array | Number:
            return kernels.smooth_minimum(a, b, temperature=temperature)

    else:

        def add(a: Array | Number, b: Array | Number) -> Array | Number:
            return _try_array_else_scalar(a, b, array_fn="maximum", scalar_fn=max)  # type: ignore[arg-type]

        def mul(a: Array | Number, b: Array | Number) -> Array | Number:
            return _try_array_else_scalar(a, b, array_fn="minimum", scalar_fn=min)  # type: ignore[arg-type]

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
    r"""Create a tropical semiring.

    When *minplus* is ``True``, returns the min-plus semiring
    :math:`(\mathbb{R}_{\ge 0} \cup \{\infty\}, \min, +, \infty, 0)`.
    Otherwise, returns the max-plus semiring
    :math:`(\mathbb{R}_{\le 0} \cup \{-\infty\}, \max, +, -\infty, 0)`.

    Parameters
    ----------
    minplus : bool
        If ``True``, returns the min-plus tropical semiring. Otherwise, the
        max-plus semiring.
    smooth : bool
        If ``True``, use the logaddexp approximation of max and min.
    temperature : float, default 1.0
        Temperature for the smooth approximation; closer to infinity is closer
        to true max/min.

    Returns
    -------
    Semiring
        A :class:`~algebraic.spec.Semiring` with tropical operations.

    Examples
    --------
    >>> from algebraic.semirings import tropical_semiring
    >>> tp = tropical_semiring(minplus=True)
    >>> tp.add(2.0, 3.0)
    2.0
    >>> tp.mul(2.0, 3.0)
    5.0
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
                return _try_array_else_scalar(a, b, array_fn="minimum", scalar_fn=min)  # type: ignore[arg-type]
        else:

            def add(a: Array | Number, b: Array | Number) -> Array | Number:
                return _try_array_else_scalar(a, b, array_fn="maximum", scalar_fn=max)  # type: ignore[arg-type]

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
    """Create a Boolean algebra with configurable differentiation mode.

    Parameters
    ----------
    mode : {"logic", "soft", "smooth", "ste", "std-fuzzy"}
        Differentiation mode:

        - ``"logic"``: non-differentiable exact Boolean operations.
        - ``"soft"``: Soft Boolean using multiplication and addition (fastest,
          smoothest).
        - ``"smooth"``: Smooth Boolean using sigmoid with *temperature*.
        - ``"ste"`` | ``"std-fuzzy"``: Straight-Through Estimator or,
          equivalently, the standard fuzzy algebra.
    temperature : float, default 1.0
        Temperature parameter for ``"smooth"`` mode.

    Returns
    -------
    BooleanAlgebra
        A :class:`~algebraic.spec.BooleanAlgebra` instance.

    Notes
    -----
    The differentiable modes work best with inputs in [0, 1] closer to the
    boundaries.

    Examples
    --------
    >>> from algebraic.semirings import boolean_algebra
    >>> ba = boolean_algebra(mode="logic")
    >>> ba.add(True, False)
    True
    >>> ba.mul(True, False)
    False
    """

    zero = 0.0
    one = 1.0

    match mode:
        case "logic":

            def add(a: Array | Number, b: Array | Number) -> Array | Number:
                return _try_array_else_scalar(a, b, array_fn="logical_or", scalar_fn=operator.__or__)  # type: ignore[arg-type]

            def mul(a: Array | Number, b: Array | Number) -> Array | Number:
                return _try_array_else_scalar(a, b, array_fn="logical_and", scalar_fn=operator.__and__)  # type: ignore[arg-type]

            def neg(a: Array | Number) -> Array | Number:
                return _try_array_else_scalar(a, array_fn="logical_not", scalar_fn=operator.__not__)  # type: ignore[arg-type]
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
                return _try_array_else_scalar(a, b, array_fn="maximum", scalar_fn=max)  # type: ignore[arg-type]

            def mul(a: Array | Number, b: Array | Number) -> Array | Number:
                return _try_array_else_scalar(a, b, array_fn="minimum", scalar_fn=min)  # type: ignore[arg-type]

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
