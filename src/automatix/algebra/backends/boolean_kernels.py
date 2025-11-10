"""Differentiable kernels for Boolean algebra.

This module provides multiple approaches to making Boolean operations
differentiable for gradient-based optimization:

1. Soft Boolean: Fast, smooth, restricted to [0,1] inputs
2. Smooth Boolean: Temperature-controlled sigmoid-based approximation
3. Straight-Through Estimator: Fast, works with any domain, biased gradients

Each approach trades off between smoothness, speed, and generality.
"""

from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Num

from automatix.algebra.kernels import AlgebraicStructure


def soft_and(x: Num[Array, "..."], y: Num[Array, "..."]) -> Num[Array, "..."]:
    """Soft Boolean AND (multiplicative relaxation).

    For x, y in [0,1]: soft_and(x,y) = x * y

    This is smooth and differentiable everywhere, approximating AND.
    When x,y are close to 0 or 1, this matches Boolean AND semantics.

    In LaTeX: x wedge y approx x cdot y
    """
    return x * y


def soft_or(x: Num[Array, "..."], y: Num[Array, "..."]) -> Num[Array, "..."]:
    """Soft Boolean OR (probabilistic relaxation).

    For x, y in [0,1]: soft_or(x,y) = x + y - x*y

    This is the complement of De Morgan law: OR(x,y) = NOT(AND(NOT(x), NOT(y)))
    = 1 - (1-x)*(1-y) = x + y - x*y

    In LaTeX: x vee y approx x + y - xy
    """
    return x + y - x * y


def soft_negate(x: Num[Array, "..."]) -> Num[Array, "..."]:
    """Soft Boolean negation.

    For x in [0,1]: soft_negate(x) = 1 - x

    Perfect relaxation of Boolean negation with smooth gradients.

    In LaTeX: neg x = 1 - x
    """
    return 1 - x


def smooth_and(
    x: Num[Array, "..."],
    y: Num[Array, "..."],
    temperature: float = 1.0,
) -> Num[Array, "..."]:
    """Smooth Boolean AND using sigmoid approximation.

    For sharp transitions, use high temperature (e.g., temperature=10).
    For gradual transitions, use low temperature (e.g., temperature=0.1).

    Formula: smooth_and(x,y) = sigmoid(temperature * (x + y - 1))

    In LaTeX: text{smooth_and}(x,y) = sigma(T(x + y - 1))
    where sigma is sigmoid and T is temperature
    """
    return jax.nn.sigmoid(temperature * (x + y - 1))


def smooth_or(
    x: Num[Array, "..."],
    y: Num[Array, "..."],
    temperature: float = 1.0,
) -> Num[Array, "..."]:
    """Smooth Boolean OR using sigmoid approximation.

    Formula: smooth_or(x,y) = sigmoid(temperature * (x + y))

    In LaTeX: text{smooth_or}(x,y) = sigma(T(x + y))
    """
    return jax.nn.sigmoid(temperature * (x + y))


def smooth_negate(
    x: Num[Array, "..."],
    temperature: float = 1.0,
) -> Num[Array, "..."]:
    """Smooth Boolean negation using sigmoid approximation.

    Formula: smooth_negate(x) = sigmoid(temperature * (0.5 - x))

    In LaTeX: text{smooth_negate}(x) = sigma(T(0.5 - x))
    """
    return jax.nn.sigmoid(temperature * (0.5 - x))


def create_boolean_kernel(
    mode: Literal["soft", "smooth", "ste"] = "soft",
    temperature: float = 1.0,
) -> AlgebraicStructure:
    """Create a differentiable Boolean kernel.

    Parameters
    ----------
    mode : {"soft", "smooth", "ste"}
        Differentiation mode:
        - "soft": Soft Boolean using multiplication and addition (fastest, smoothest)
        - "smooth": Smooth Boolean using sigmoid with temperature
        - "ste": Straight-Through Estimator (biased gradients, but works generally)
    temperature : float, optional
        Temperature parameter for "smooth" mode (default: 1.0)

    Returns
    -------
    AlgebraicStructure
        A Boolean kernel with specified differentiation mode.

    Notes
    -----
    Soft and smooth modes work best with inputs in [0,1].
    STE mode works with any input domain but provides biased gradients.
    """
    if mode == "soft":
        return AlgebraicStructure(
            add=soft_or,
            mul=soft_and,
            zero=0.0,
            one=1.0,
            negate=soft_negate,
            properties=frozenset(["idempotent_add", "idempotent_mul", "commutative", "simple", "has_negation"]),
        )
    elif mode == "smooth":
        add_fn = lambda x, y: smooth_or(x, y, temperature=temperature)  # noqa: E731
        mul_fn = lambda x, y: smooth_and(x, y, temperature=temperature)  # noqa: E731
        neg_fn = lambda x: smooth_negate(x, temperature=temperature)  # noqa: E731
        return AlgebraicStructure(
            add=add_fn,
            mul=mul_fn,
            zero=0.0,
            one=1.0,
            negate=neg_fn,
            properties=frozenset(["commutative", "has_negation"]),
        )
    elif mode == "ste":
        return AlgebraicStructure(
            add=lambda x, y: jnp.logical_or(x > 0.5, y > 0.5).astype(x.dtype),
            mul=lambda x, y: jnp.logical_and(x > 0.5, y > 0.5).astype(x.dtype),
            zero=0.0,
            one=1.0,
            negate=lambda x: 1.0 - x,
            properties=frozenset(["idempotent_add", "idempotent_mul", "commutative", "simple", "has_negation"]),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'soft', 'smooth', or 'ste'.")
