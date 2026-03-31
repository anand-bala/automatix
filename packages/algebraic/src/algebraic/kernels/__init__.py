"""Kernel implementations for boolean and semiring operations.

Public facade that routes to backend-specific implementations.
Generic operations (soft boolean ops) work on any backend via basic
arithmetic. Smooth/sigmoid operations dispatch to the appropriate
backend at call time using lazy imports.

Backend detection uses ``array_api_compat.is_*_array`` checks.
Scalar inputs fall back to the NumPy implementation by default.
"""

from __future__ import annotations

import typing

from array_api_compat import is_jax_array, is_torch_array

from algebraic.kernels._generic import soft_boolean_and as soft_boolean_and
from algebraic.kernels._generic import soft_boolean_not as soft_boolean_not
from algebraic.kernels._generic import soft_boolean_or as soft_boolean_or
from algebraic.types import Array, MaybeAxis, Number
from algebraic.utils import normalize_axes


def logaddexp(x: Array | Number, y: Array | Number) -> Array | Number:
    r"""Numerically stable ``log(exp(x) + exp(y))``.

    Routes to JAX (with ``custom_vjp`` for safe ``-inf`` gradients),
    torch, or numpy.
    """
    if any(is_torch_array(arg) for arg in (x, y)):
        import torch

        from algebraic.kernels._torch import logaddexp as _torch_logaddexp

        return _torch_logaddexp(torch.as_tensor(x), torch.as_tensor(y))
    if any(is_jax_array(arg) for arg in (x, y)):
        from algebraic.kernels._jax import logaddexp as _jax_logaddexp

        return _jax_logaddexp(x, y)
    import numpy as np

    return typing.cast(np.ndarray, np.logaddexp(np.asarray(x), np.asarray(y)))


def logsumexp(a: Array | Number, axis: MaybeAxis = None) -> Array | Number:
    r"""Numerically stable ``logsumexp``.

    Routes to JAX (with ``custom_vjp``), torch, or numpy implementation.
    """
    if is_torch_array(a):
        import torch

        a = torch.as_tensor(a)
        axis = normalize_axes(axis, a.ndim)  # type: ignore[union-attr]

        result = torch.logsumexp(a, dim=axis)
        return result
    if is_jax_array(a):
        from algebraic.kernels._jax import logsumexp as _jax_logsumexp

        return _jax_logsumexp(a, axis)
    import numpy as np
    import scipy.special

    return scipy.special.logsumexp(np.asarray(a), axis=axis)


def smooth_boolean_and(
    x: Array | Number,
    y: Array | Number,
    temperature: float = 1.0,
) -> Array | Number:
    """Smooth Boolean AND: ``sigmoid(T * (x + y - 1))``."""
    if any(is_torch_array(arg) for arg in (x, y)):
        import torch

        return torch.sigmoid(temperature * (torch.as_tensor(x) + torch.as_tensor(y) - 1))
    if any(is_jax_array(arg) for arg in (x, y)):
        import jax.numpy as jnp

        import algebraic.kernels._jax as _jax

        x = jnp.asarray(x)
        y = jnp.asarray(y)

        return _jax.smooth_boolean_and(x, y, temperature)
    # assume numpy / scalar
    import numpy as np
    from scipy.special import expit

    return typing.cast(np.ndarray, expit(temperature * (np.asarray(x) + np.asarray(y) - 1)))


def smooth_boolean_or(
    x: Array | Number,
    y: Array | Number,
    temperature: float = 1.0,
) -> Array | Number:
    """Smooth Boolean OR: ``sigmoid(T * (x + y))``."""
    if any(is_torch_array(arg) for arg in (x, y)):
        import torch

        return torch.sigmoid(temperature * (torch.as_tensor(x) + torch.as_tensor(y)))
    if any(is_jax_array(arg) for arg in (x, y)):
        import jax.numpy as jnp

        import algebraic.kernels._jax as _jax

        x = jnp.asarray(x)
        y = jnp.asarray(y)

        return _jax.smooth_boolean_or(x, y, temperature)
    # Otherwise, assume numpy
    import numpy as np
    from scipy.special import expit

    return typing.cast(np.ndarray, expit(temperature * (np.asarray(x) + np.asarray(y))))


def smooth_boolean_not(
    x: Array | Number,
    temperature: float = 1.0,
) -> Array | Number:
    """Smooth Boolean NOT: ``sigmoid(T * (0.5 - x))``."""
    if is_torch_array(x):
        import torch

        return torch.sigmoid(temperature * (0.5 - torch.as_tensor(x)))
    if is_jax_array(x):
        from algebraic.kernels._jax import smooth_boolean_not as _jax

        return _jax(x, temperature)
    import numpy as np
    from scipy.special import expit

    return typing.cast(np.ndarray, expit(temperature * (0.5 - np.asarray(x))))


def smooth_maximum(x: Array | Number, y: Array | Number, temperature: float = 1.0) -> Array | Number:
    r"""Smooth approximation of ``max(x, y)`` using ``logaddexp``."""
    if any(is_torch_array(arg) for arg in (x, y)):
        import torch

        from algebraic.kernels._torch import logaddexp as _torch_logaddexp

        return _torch_logaddexp(temperature * torch.as_tensor(x), temperature * torch.as_tensor(y)) / temperature
    if any(is_jax_array(arg) for arg in (x, y)):
        import jax.numpy as jnp

        import algebraic.kernels._jax as _jax

        x = jnp.asarray(x)
        y = jnp.asarray(y)

        return _jax.smooth_maximum(x, y, temperature)
    import numpy as np

    return np.logaddexp(temperature * np.asarray(x), temperature * np.asarray(y)) / temperature


def smooth_minimum(x: Array | Number, y: Array | Number, temperature: float = 1.0) -> Array | Number:
    r"""Smooth approximation of ``min(x, y)`` using negated ``logaddexp``."""
    if any(is_torch_array(arg) for arg in (x, y)):
        import torch

        from algebraic.kernels._torch import logaddexp as _torch_logaddexp

        return -_torch_logaddexp(-temperature * torch.as_tensor(x), -temperature * torch.as_tensor(y)) / temperature
    if any(is_jax_array(arg) for arg in (x, y)):
        import jax.numpy as jnp

        import algebraic.kernels._jax as _jax

        x = jnp.asarray(x)
        y = jnp.asarray(y)

        return _jax.smooth_minimum(x, y, temperature)
    import numpy as np

    return -np.logaddexp(-temperature * np.asarray(x), -temperature * np.asarray(y)) / temperature


def smooth_max(x: Array | Number, axis: MaybeAxis = None, temperature: float = 1.0) -> Array | Number:
    r"""Smooth max reduction using logsumexp scaled by temperature."""
    if is_torch_array(x):
        import torch

        tx = torch.as_tensor(x)
        axis = normalize_axes(axis, tx.ndim)
        return torch.logsumexp(temperature * tx, dim=axis) / temperature
    if is_jax_array(x):
        from algebraic.kernels._jax import smooth_max as _jax

        return _jax(x, axis, temperature)
    import numpy as np
    from scipy.special import logsumexp as _scipy_logsumexp

    return _scipy_logsumexp(temperature * np.asarray(x), axis=axis) / temperature


def smooth_min(x: Array | Number, axis: MaybeAxis = None, temperature: float = 1.0) -> Array | Number:
    r"""Smooth min reduction using negated logsumexp."""
    if is_torch_array(x):
        import torch

        tx = torch.as_tensor(x)
        axis = normalize_axes(axis, tx.ndim)
        return -torch.logsumexp(-temperature * tx, dim=axis) / temperature
    if is_jax_array(x):
        from algebraic.kernels._jax import smooth_min as _jax

        return _jax(x, axis, temperature)
    import numpy as np
    from scipy.special import logsumexp as _scipy_logsumexp

    return -_scipy_logsumexp(-temperature * np.asarray(x), axis=axis) / temperature
