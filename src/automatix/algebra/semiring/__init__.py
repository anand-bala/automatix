"""Backward compatibility layer for semiring imports.

DEPRECATED: This module is deprecated. Use the new imports instead:
    from automatix.algebra import get_semiring, MaxPlusSemiring, ...
    # or
    from automatix.algebra.backends.jax_ import MaxPlusSemiring
"""

# Re-export JAX semirings from new location for backward compatibility
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
]
