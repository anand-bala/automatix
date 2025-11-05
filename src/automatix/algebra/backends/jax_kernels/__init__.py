"""Optimized JAX kernels for semiring operations.

This package contains custom forward and backward passes for various semirings,
optimized for performance and numerical stability.

Current kernels:
- logsumexp: Custom logsumexp with proper gradient handling

Planned kernels (v0.6.0 or later):
- maxplus: MaxPlus-specific optimizations
- log_semiring: LogSemiring-specific optimizations

Usage:
    from automatix.algebra.backends.jax_kernels import logsumexp

    # Use in semiring implementations
    result = logsumexp(array, axis=-1)
"""

from automatix.algebra.backends.jax_kernels.logsumexp import logsumexp
from automatix.algebra.backends.jax_kernels.utils import kernel

__all__ = ["logsumexp", "kernel"]
