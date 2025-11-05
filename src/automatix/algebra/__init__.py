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

# Core abstractions
from automatix.algebra.abstract import (
    AbstractDeMorganAlgebra,
    AbstractNegation,
    AbstractPolynomial,
    AbstractSemiring,
    PolynomialManager,
)

# JAX semirings (currently the primary backend)
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
    RightLSEMaxMinSemiring,
    RightMaxMinSemiring,
)

# Polynomial implementations
from automatix.algebra.polynomials.boolean import (
    BooleanPolyCtx,
    BooleanPolynomial,
)

# Registry and factory
from automatix.algebra.registry import (
    get_available_backends,
    get_semiring,
    list_semirings,
    register,
    unregister,
)

__all__ = [
    # Abstractions
    "AbstractSemiring",
    "AbstractNegation",
    "AbstractDeMorganAlgebra",
    "AbstractPolynomial",
    "PolynomialManager",
    # Registry
    "get_semiring",
    "register",
    "unregister",
    "list_semirings",
    "get_available_backends",
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
    "LogSemiring",
    "LatticeAlgebra",
    # Polynomials
    "BooleanPolynomial",
    "BooleanPolyCtx",
]
