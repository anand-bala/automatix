"""Generic kernel implementations that work on all backends.

These operations use only basic arithmetic (+, -, *) so they are
backend-agnostic and need no dispatch.
"""

from __future__ import annotations

from algebraic.types import Array, Number


def soft_boolean_and(x: Array | Number, y: Array | Number) -> Array | Number:
    """Soft Boolean AND (multiplicative relaxation).

    For x, y in [0,1]: ``soft_and(x,y) = x * y``

    This is smooth and differentiable everywhere, approximating AND.
    When x,y are close to 0 or 1, this matches Boolean AND semantics.

    In LaTeX: :math:`x \\wedge y \\approx x \\cdot y`
    """
    return x * y  # type: ignore[operator]


def soft_boolean_or(x: Array | Number, y: Array | Number) -> Array | Number:
    """Soft Boolean OR (probabilistic relaxation).

    For x, y in [0,1]: ``soft_or(x,y) = x + y - x*y``

    This is the complement of De Morgan law: ``OR(x,y) = NOT(AND(NOT(x), NOT(y)))``
    ``= 1 - (1-x)*(1-y) = x + y - x*y``

    In LaTeX: :math:`x \\vee y \\approx x + y - xy`
    """
    return x + y - x * y  # type: ignore[operator]


def soft_boolean_not(x: Array | Number) -> Array | Number:
    """Soft Boolean negation.

    For x in [0,1]: ``soft_not(x) = 1 - x``

    Perfect relaxation of Boolean negation with smooth gradients.

    In LaTeX: :math:`\\neg x = 1 - x`
    """
    return 1 - x
