"""Backward compatibility layer for jax_backend imports.

DEPRECATED: This module is deprecated. Use the new imports instead:
    from automatix.algebra.backends.jax_ import MaxPlusSemiring, ...
"""

# Re-export everything from new location
from automatix.algebra.backends.jax_ import *  # noqa: F401, F403

__all__ = [
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
    "AbstractSemiring",
    "AbstractNegation",
    "AbstractDeMorganAlgebra",
]
