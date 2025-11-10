"""Automatix algebra module for semiring operations and polynomial management.

This module provides:
1. Pure interface definitions (spec.py)
2. Semiring registry and factory (registry.py)
3. Backend-specific implementations (backends/)
4. Polynomial abstractions (abstract/)
5. Backward compatibility layer (old imports still work)

Quick start:
    from automatix.algebra import get_semiring

    # Get a semiring by name
    MaxPlus = get_semiring("MaxPlus", backend="jax")

    # Create arrays
    weights = MaxPlus.zeros((3, 3))
    result = MaxPlus.matmul(weights, weights)
"""

# JAX semirings (currently the primary backend)
# Compatibility layer
from automatix.algebra._compat import normalize_semiring

# Boolean kernels
from automatix.algebra.backends.boolean_kernels import create_boolean_kernel
from automatix.algebra.backends.jax_ import (
    CountingSemiring,
    LatticeAlgebra,
    LeftLSEMaxMinSemiring,
    LeftMaxMinSemiring,
    LogSemiring,
    LSEMaxMinSemiring,
    MaxMinAlgebra,
    MaxMinSemiring,
    MaxPlusSemiring,
    MinPlusSemiring,
    RightLSEMaxMinSemiring,
    RightMaxMinSemiring,
)

# GPU-optimized kernels
from automatix.algebra.kernels import AlgebraicStructure

# Polynomial implementations
from automatix.algebra.polynomials.boolean import (
    BooleanPolyCtx,
    BooleanPolynomial,
)

# Registry and factory
from automatix.algebra.registry import (
    get_available_backends,
    get_kernel,
    get_semiring,
    list_kernels,
    list_semirings,
    register,
    register_kernel,
    unregister,
    unregister_kernel,
)

# Core abstractions
from automatix.algebra.spec import (
    AbstractDeMorganAlgebra,
    AbstractNegation,
    AbstractPolynomial,
    AbstractSemiring,
    PolynomialManager,
)

__all__ = [
    # Abstractions
    "AbstractSemiring",
    "AbstractNegation",
    "AbstractDeMorganAlgebra",
    "AbstractPolynomial",
    "PolynomialManager",
    # GPU kernels
    "AlgebraicStructure",
    "normalize_semiring",
    "create_boolean_kernel",
    # Registry
    "get_semiring",
    "register",
    "unregister",
    "list_semirings",
    "get_available_backends",
    "get_kernel",
    "register_kernel",
    "unregister_kernel",
    "list_kernels",
    # JAX semirings
    "CountingSemiring",
    "MaxMinSemiring",
    "LeftMaxMinSemiring",
    "RightMaxMinSemiring",
    "MaxMinAlgebra",
    "LSEMaxMinSemiring",
    "LeftLSEMaxMinSemiring",
    "RightLSEMaxMinSemiring",
    "MaxPlusSemiring",
    "MinPlusSemiring",
    "LogSemiring",
    "LatticeAlgebra",
    # Polynomials
    "BooleanPolynomial",
    "BooleanPolyCtx",
]
