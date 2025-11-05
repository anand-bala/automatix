"""Optimized kernels for LogSemiring.

This module contains forward and backward pass implementations for LogSemiring
operations, optimized for numerical stability in log-domain arithmetic.

Planned kernels (v0.6.0 or later):
- forward_log_add: Numerically stable logsumexp addition
- backward_log_add: Gradient computation for log addition
- forward_log_matmul: Optimized log-domain matrix multiplication
- backward_log_matmul: Gradient for log matrix multiplication
"""

__all__: list[str] = []
