"""Optimized kernels for MaxPlus semiring.

This module contains forward and backward pass implementations for MaxPlus
operations, designed for numerical stability and performance.

Planned kernels (v0.6.0 or later):
- forward_maxplus_matmul: Optimized matrix multiplication
- backward_maxplus_matmul: Gradient computation for matrix multiplication
- forward_maxplus_sum: Optimized reduction operation
- backward_maxplus_sum: Gradient for reduction
"""

__all__: list[str] = []
