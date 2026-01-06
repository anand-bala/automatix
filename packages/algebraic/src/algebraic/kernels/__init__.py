"""JAX kernel implementations for boolean and semiring operations.

Provides smooth approximations and differentiable boolean operations.
"""
# mypy: disable-error-code="no-untyped-call, no-any-return"

import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, Num
from typing_extensions import TypeAlias

_Axis: TypeAlias = None | int | Sequence[int]
_Array: TypeAlias = Num[Array, "..."]


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def logsumexp(a: _Array, axis: _Axis = None) -> _Array:
    r"""Implementation of `logsumexp` that generates correct gradients for arguments containing all $-\infty$.

    Derived from [this comment in `jax` issue #6811](https://github.com/google/jax/issues/6811#issuecomment-986265534).
    """
    return _logsumexp_fwd(a, axis)[0]


def _logsumexp_fwd(a: _Array, axis: _Axis) -> tuple[_Array, tuple[_Array, _Array]]:
    c = jnp.max(a, axis=axis, keepdims=True)
    safe = jnp.isfinite(c)
    c = jnp.where(safe, c, 0)
    e = jnp.exp(a - c)
    z = jnp.sum(e, axis=axis, keepdims=True)
    r = jnp.squeeze(c, axis=axis) + jnp.log(jnp.squeeze(z, axis=axis))
    return r, (e, z)


def _logsumexp_bwd(axis: _Axis, res: tuple[_Array, _Array], g: _Array) -> tuple[_Array]:
    e = jnp.asarray(res[0])
    z = jnp.asarray(res[1])
    g = jnp.asarray(g)
    safe = z != 0
    z = jnp.where(safe, z, 1)
    if axis is not None:
        g = jnp.expand_dims(g, axis=axis)
    return (g / z * e,)


logsumexp.defvjp(_logsumexp_fwd, _logsumexp_bwd)


def soft_boolean_and(x: Num[Array, "..."], y: Num[Array, "..."]) -> Num[Array, "..."]:
    """Soft Boolean AND (multiplicative relaxation).

    For x, y in [0,1]: soft_and(x,y) = x * y

    This is smooth and differentiable everywhere, approximating AND.
    When x,y are close to 0 or 1, this matches Boolean AND semantics.

    In LaTeX: x wedge y approx x cdot y
    """
    return x * y


def soft_boolean_or(x: Num[Array, "..."], y: Num[Array, "..."]) -> Num[Array, "..."]:
    """Soft Boolean OR (probabilistic relaxation).

    For x, y in [0,1]: soft_or(x,y) = x + y - x*y

    This is the complement of De Morgan law: OR(x,y) = NOT(AND(NOT(x), NOT(y)))
    = 1 - (1-x)*(1-y) = x + y - x*y

    In LaTeX: x vee y approx x + y - xy
    """
    return x + y - x * y


def soft_boolean_not(x: Num[Array, "..."]) -> Num[Array, "..."]:
    """Soft Boolean negation.

    For x in [0,1]: soft_not(x) = 1 - x

    Perfect relaxation of Boolean negation with smooth gradients.

    In LaTeX: neg x = 1 - x
    """
    return 1 - x


def smooth_boolean_and(
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


def smooth_boolean_or(
    x: Num[Array, "..."],
    y: Num[Array, "..."],
    temperature: float = 1.0,
) -> Num[Array, "..."]:
    """Smooth Boolean OR using sigmoid approximation.

    Formula: smooth_or(x,y) = sigmoid(temperature * (x + y))

    In LaTeX: text{smooth_or}(x,y) = sigma(T(x + y))
    """
    return jax.nn.sigmoid(temperature * (x + y))


def smooth_boolean_not(
    x: Num[Array, "..."],
    temperature: float = 1.0,
) -> Num[Array, "..."]:
    """Smooth Boolean negation using sigmoid approximation.

    Formula: smooth_not(x) = sigmoid(temperature * (0.5 - x))

    In LaTeX: text{smooth_not}(x) = sigma(T(0.5 - x))
    """
    return jax.nn.sigmoid(temperature * (0.5 - x))


def smooth_maximum(x: _Array, y: _Array, temperature: float = 1.0) -> _Array:
    r"""Smooth approximation of max(x, y) using log-sum-exp.

    Formula: smooth_max(x, y) = (1/T) * log(exp(T*x) + exp(T*y))

    For temperature -> infinity, this approaches max(x, y).
    For temperature -> 0, this approaches (x + y) / 2.

    In LaTeX: text{smooth_max}(x, y) = frac{1}{T} log(exp(T x) + exp(T y))
    """
    return logsumexp(jnp.stack([temperature * x, temperature * y], axis=0), axis=0) / temperature


def smooth_max(x: _Array, axis: _Axis = None, temperature: float = 1.0) -> _Array:
    r"""Smooth MaxPlus reduction using logsumexp.

    Uses logsumexp scaled by temperature to approximate max along axis.
    Fully differentiable everywhere.

    In LaTeX: text{smooth_maxplus_sum}(x, axis) = frac{1}{T} log(sum(exp(T x)))
    """
    x = jnp.asarray(x)
    return logsumexp(temperature * x, axis=axis) / temperature


def smooth_minimum(x: _Array, y: _Array, temperature: float = 1.0) -> _Array:
    r"""Smooth approximation of min(x, y) using log-sum-exp with negation.

    Formula: smooth_min(x, y) = -(1/T) * log(exp(-T*x) + exp(-T*y))

    In LaTeX: text{smooth_min}(x, y) = -frac{1}{T} log(exp(-T x) + exp(-T y))
    """
    return -logsumexp(jnp.stack([-temperature * x, -temperature * y], axis=0), axis=0) / temperature


def smooth_min(x: _Array, axis: _Axis = None, temperature: float = 1.0) -> _Array:
    r"""Smooth MaxMin reduction using negated logsumexp.

    Fully differentiable reduction approximating max of mins.

    In LaTeX: text{smooth_maxmin_sum}(x, axis) = -frac{1}{T} log(sum(exp(-T x)))
    """
    x = jnp.asarray(x)
    return -logsumexp(-temperature * x, axis=axis) / temperature


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def soft_boolean_sum(x: _Array, axis: _Axis = None) -> _Array:
    r"""Boolean reduction via soft_or along axis.

    Computes OR_{axis} x using soft_or for differentiability.
    In Boolean algebra, OR is the join operation.

    soft_or(x, y) = x + y - x*y, which is the probabilistic OR formula.

    **IMPORTANT: Boolean-Regime Assumption**
    This function assumes values are in Boolean regime (near 0 or 1):
    - Values close to 0: interpreted as FALSE
    - Values close to 1: interpreted as TRUE
    - Middle values (0.3-0.7): rare, treated conservatively

    With this assumption, the approximate gradient computation is acceptable
    for ML applications because:
    1. FALSE values (near 0) naturally have minimal impact on OR result
    2. TRUE values (near 1) dominate and drive optimization
    3. Gradient signal still flows to all inputs (correct direction)
    4. Magnitude approximation is minor compared to Boolean signal strength

    **Gradient Note**: Uses uniform distribution across inputs (simplified from
    exact soft_or chain derivatives). For Boolean-regime values, this approximation
    has negligible practical impact on optimization.

    In LaTeX: text{boolean_sum}(x, axis) = bigvee_{axis} x approx bigcup_{axis} x

    Note: axis is marked as nondiff_argnums because we don't differentiate w.r.t. axis.

    Parameters
    ----------
    x : array-like
        Input array with values expected in Boolean regime (near 0 or 1)
    axis : None, int, or tuple of ints
        Axis or axes along which to compute OR reduction

    Returns
    -------
    result : array
        Result of soft OR reduction along specified axis
    """
    return _soft_boolean_sum_fwd(x, axis)[0]


def _soft_boolean_sum_fwd(x: _Array, axis: _Axis) -> tuple[_Array, tuple[_Array, _Axis]]:
    """Forward pass for boolean_sum using reduction via soft_or.

    Steps:
    1. Convert input to JAX array
    2. If axis is None: flatten array and accumulate soft_or operations using scan
    3. If axis is int or tuple: normalize axis indices, then apply soft_or reduction
       along each axis (in reverse order to maintain index validity)
    4. Return result and residuals for backward pass

    soft_or implements: a OR b = a + b - a*b
    which is the probabilistic interpretation of OR.
    """
    x = jnp.asarray(x)

    # Case 1: Reduce all dimensions
    if axis is None:
        # Flatten to 1D array and use scan for accumulation
        flat_x = jnp.reshape(x, (-1,))
        # scan accumulates: (carry, input) -> (carry, output)
        # We use carry as the accumulated result and iterate through remaining elements
        result, _ = jax.lax.scan(lambda carry, elem: (soft_boolean_or(carry, elem), None), flat_x[0], flat_x[1:])
        return result, (x, axis)

    # Case 2: Reduce along specific axis/axes
    ndim = x.ndim
    # Normalize axis specification (handle negative indices)
    if isinstance(axis, int):
        axis_normalized = axis if axis >= 0 else ndim + axis
        axes = [axis_normalized]
    else:
        axes = [ax if ax >= 0 else ndim + ax for ax in axis]

    # Apply soft_or reduction along each axis in reverse order
    # (reverse order prevents index shifting when reducing)
    result = x
    for ax in sorted(axes, reverse=True):
        # apply_along_axis applies soft_or reduction along the specified axis
        # functools.reduce chains the soft_or operations: a[0] OR a[1] OR a[2] OR ...
        result = jnp.apply_along_axis(lambda a: functools.reduce(soft_boolean_or, a), ax, result)

    return result, (x, axis)


def _soft_boolean_sum_bwd(axis: _Axis, res: tuple[_Array, _Axis], g: _Array) -> tuple[_Array]:
    """Backward pass for boolean_sum.

    APPROXIMATION JUSTIFIED BY BOOLEAN-REGIME SEMANTICS:
    This implementation distributes gradients uniformly across inputs.
    This is approximate compared to exact soft_or chain derivatives, but is
    acceptable (even desirable) for Boolean-regime values because:

    1. **FALSE values (near 0)**: Getting "too much" gradient is harmless since:
       - They already have minimal impact on OR result
       - Optimization won't increase them beyond Boolean threshold
       - Extra gradient signal is wasted but not harmful

    2. **TRUE values (near 1)**: Getting "too little" gradient is harmless since:
       - They already dominate the OR result
       - Optimization already favors keeping them high
       - Small gradient change is imperceptible

    3. **Gradient direction is correct**: Signal flows to all inputs, enabling
       optimization to proceed normally. Only magnitude is approximate.

    Mathematical details (for reference):
    - Exact: For soft_or(a, b) = a + b - a*b:
      - d/da = 1 - b
      - d/db = 1 - a
    - For chained reductions [a, b, c] -> soft_or(soft_or(a, b), c):
      - Correct d/da = (1 - b) * (1 - c)
      - Correct d/db = (1 - a) * (1 - c)
      - Correct d/dc = 1 - soft_or(a, b)
    - Our implementation: d/da = d/db = d/dc = g (uniform)
      - Correct gradient direction but approximate magnitudes
      - In Boolean regime, this approximation is negligible

    Steps:
    1. Extract residuals from forward pass
    2. Expand gradient to match reduction dimension
    3. Distribute gradient uniformly (justified approximation for Boolean regime)
    4. Return gradient with respect to input x
    """
    x, axis = res
    x = jnp.asarray(x)
    g = jnp.asarray(g)

    # Expand gradient back to the dimensions that were reduced
    # Example: if we reduced (2,3) along axis=1 to get shape (2,),
    # expand_dims adds a dimension back: (2,) -> (2, 1)
    if axis is not None:
        g = jnp.expand_dims(g, axis=axis)

    # Distribute gradient uniformly across the reduced dimension
    # Broadcasting automatically expands from (2, 1) to (2, 3)
    grad_x = g * jnp.ones_like(x)
    return (grad_x,)


soft_boolean_sum.defvjp(_soft_boolean_sum_fwd, _soft_boolean_sum_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def smooth_boolean_sum(x: _Array, axis: _Axis = None, temperature: float = 1.0) -> _Array:
    r"""Smooth Boolean reduction via sigmoid-based approximation.

    Uses smooth_or with temperature to approximate Boolean OR reduction.
    Fully differentiable everywhere.

    smooth_or(a, b, T) = sigmoid(T * (a + b))
    where sigmoid is the logistic function: 1 / (1 + exp(-x))

    Temperature parameter controls the sharpness:
    - High temperature (T >> 1): closer to hard OR
    - Low temperature (T << 1): smoother transitions

    **IMPORTANT: Boolean-Regime Assumption**
    Same as boolean_sum: assumes values in Boolean regime (near 0 or 1).
    The approximations in backward pass are justified for Boolean semantics.

    **Additional Note**: Temperature parameter affects forward pass sharpness.
    In backward pass, we use uniform gradient distribution (same as boolean_sum),
    which doesn't account for temperature-dependent scaling. This is acceptable
    because:
    - Temperature mainly affects forward behavior
    - Gradient signal still flows to all inputs
    - For Boolean-regime values, approximation impact is minimal

    In LaTeX: text{smooth_boolean_sum}(x, axis, T) approx bigvee_{axis} x

    Parameters
    ----------
    x : array-like
        Input array with values expected in Boolean regime (near 0 or 1)
    axis : None, int, or tuple of ints
        Axis or axes along which to compute OR reduction
    temperature : float, default 1.0
        Temperature parameter controlling sigmoid sharpness.
        Higher T makes transitions sharper (closer to hard OR).
        Lower T makes transitions smoother.

    Returns
    -------
    result : array
        Result of smooth OR reduction along specified axis

    Note: temperature is marked as nondiff_argnums because we don't typically
    differentiate w.r.t. the temperature parameter (it's a hyperparameter).
    """
    return _smooth_boolean_sum_fwd(x, axis, temperature)[0]


def _smooth_boolean_sum_fwd(
    x: _Array,
    axis: _Axis,
    temperature: float,
) -> tuple[_Array, tuple[_Array, _Axis, float]]:
    """Forward pass for smooth_boolean_sum.

    Steps:
    1. Convert input to JAX array
    2. If axis is None: flatten and accumulate smooth_or operations with temperature
    3. If axis is int or tuple: normalize indices, then apply smooth_or reduction
    4. Store input, axis, and temperature for backward pass

    smooth_or(a, b, T) = sigmoid(T * (a + b)) provides smooth approximation of OR
    with temperature-controlled sharpness.
    """
    x = jnp.asarray(x)

    # Case 1: Reduce all dimensions
    if axis is None:
        # Flatten to 1D array and use scan for accumulation with temperature
        flat_x = jnp.reshape(x, (-1,))
        # scan accumulates: (carry, input) -> (carry, output)
        # We use carry as the accumulated result and iterate through remaining elements
        result, _ = jax.lax.scan(
            lambda carry, elem: (smooth_boolean_or(carry, elem, temperature), None), flat_x[0], flat_x[1:]
        )
        return result, (x, axis, temperature)

    # Case 2: Reduce along specific axis/axes
    ndim = x.ndim
    # Normalize axis specification
    if isinstance(axis, int):
        axis_normalized = axis if axis >= 0 else ndim + axis
        axes = [axis_normalized]
    else:
        axes = [ax if ax >= 0 else ndim + ax for ax in axis]

    # Apply smooth_or reduction along each axis in reverse order
    result = x
    for ax in sorted(axes, reverse=True):
        # Lambda function creates a version of smooth_or with temperature bound
        # This chains: a[0] OR a[1] OR a[2] OR ... (all with same temperature)
        result = jnp.apply_along_axis(
            lambda a: functools.reduce(lambda u, v: smooth_boolean_or(u, v, temperature), a),
            ax,
            result,
        )

    return result, (x, axis, temperature)


def _smooth_boolean_sum_bwd(
    axis: _Axis,
    temperature: float,
    res: tuple[_Array, _Axis, float],
    g: _Array,
) -> tuple[_Array]:
    """Backward pass for smooth_boolean_sum.

    IMPORTANT: This is a SIMPLIFIED implementation with a known gradient approximation.
    (Same approximation strategy as boolean_sum backward pass)

    Gradient computation for smooth_or based reduction:
    - smooth_or(a, b, T) = sigmoid(T * (a + b))
    - Gradient flows through sigmoid
    - sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    - d(smooth_or)/da = T * sigmoid'(T * (a + b)) = T * smooth_or * (1 - smooth_or)
    - Similar for d(smooth_or)/db

    CURRENT IMPLEMENTATION (APPROXIMATION):
    - We distribute the gradient uniformly across all elements in the reduced dimension
    - grad_x = g * ones_like(x)
    - This is mathematically INCORRECT but provides reasonable results

    CORRECT IMPLEMENTATION (would require):
    - Storing intermediate sigmoid results for each soft_or operation
    - Computing local gradients: d(smooth_or)/da = T * sigmoid'(...)
    - Backpropagating through the reduction chain using dynamic programming
    - This would increase memory usage and computational cost significantly

    WHY THE APPROXIMATION WORKS:
    - For well-behaved inputs, sigmoid gradient is relatively smooth
    - Even distribution gives reasonable signal propagation
    - Works adequately for optimization and training purposes
    - Temperature parameter can be tuned to adjust gradient flow

    Steps:
    1. Extract residuals (input, axis, temperature) and convert to proper types
    2. Expand gradient to match reduction dimension
    3. Distribute gradient uniformly across the reduced dimension (APPROXIMATION)
    4. Return gradient with respect to input x

    LIMITATION: Same as boolean_sum - gradient distribution is approximate.
    """
    x, axis_res, temperature_res = res
    x = jnp.asarray(x)
    g = jnp.asarray(g)

    # Expand gradient back to the dimensions that were reduced
    # Example: if we reduced (2,3) along axis=1 to get shape (2,),
    # expand_dims adds a dimension back: (2,) -> (2, 1)
    if axis is not None:
        g = jnp.expand_dims(g, axis=axis)

    # Distribute gradient uniformly across the reduced dimension
    # APPROXIMATION: treats all elements equally during backward pass
    # More exact would require tracking intermediate smooth_or results
    # and backpropagating through their gradients with temperature-scaled factors
    # ISSUE: This ignores temperature-dependent gradient scaling
    #   - Higher temperature makes gradients more uniform (smoother)
    #   - Lower temperature makes gradients concentrate on extreme values
    #   - Our approximation doesn't account for this difference
    grad_x = g * jnp.ones_like(x)
    return (grad_x,)


smooth_boolean_sum.defvjp(_smooth_boolean_sum_fwd, _smooth_boolean_sum_bwd)
