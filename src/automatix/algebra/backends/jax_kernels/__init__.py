"""Optimized JAX kernels for semiring operations.

This package contains custom forward and backward passes for various semirings,
optimized for performance and numerical stability.

Current kernels:
- logsumexp: Custom logsumexp with proper gradient handling
- batch_operations: Batch semiring operations for polynomial coefficient accumulation (v0.6.0)

Planned kernels:
- scalar_ops: Scalar-optimized semiring operations for tight loops (v0.6.0)
- maxplus: MaxPlus-specific optimizations
- log_semiring: LogSemiring-specific optimizations

Usage:
    from automatix.algebra.backends.jax_kernels import logsumexp, batch_accumulate_coefficients

    # Use in semiring implementations
    result = logsumexp(array, axis=-1)

    # Use in polynomial evaluation
    coeffs = batch_accumulate_coefficients(semiring, current_coeffs, indices, values)
"""

from automatix.algebra.backends.jax_kernels.batch_operations import (
    batch_accumulate_coefficients,
    batch_accumulate_with_multiplication,
    batch_evaluate_monomials,
)
from automatix.algebra.backends.jax_kernels.logsumexp import logsumexp
from automatix.algebra.backends.jax_kernels.utils import kernel

__all__ = [
    "logsumexp",
    "kernel",
    "batch_accumulate_coefficients",
    "batch_accumulate_with_multiplication",
    "batch_evaluate_monomials",
]
