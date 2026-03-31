"""Torch-specific kernel implementations.

Provides a numerically stable ``logaddexp`` with a custom backward pass that
returns zero gradients (instead of NaN) when both arguments are ``-inf``.
"""
# mypy: disable-error-code="no-untyped-call, no-any-return, attr-defined"

from __future__ import annotations

import torch


class _LogAddExp(torch.autograd.Function):  # type: ignore[misc]
    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.logaddexp(x, y)

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx, inputs: tuple[torch.Tensor, torch.Tensor], output: torch.Tensor
    ) -> None:  # noqa: ARG004
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = ctx.saved_tensors
        # softmax weights: exp(x)/(exp(x)+exp(y)) = sigmoid(x-y)
        # When both are -inf, diff is nan -> isfinite guard -> fallback to 0
        diff_xy = x - y
        diff_yx = y - x
        wx = torch.where(
            torch.isfinite(diff_xy), torch.sigmoid(diff_xy), torch.where(x > y, torch.ones_like(x), torch.zeros_like(x))
        )
        wy = torch.where(
            torch.isfinite(diff_yx), torch.sigmoid(diff_yx), torch.where(y > x, torch.ones_like(y), torch.zeros_like(y))
        )
        return g * wx, g * wy


def logaddexp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""Numerically stable ``log(exp(x) + exp(y))`` with safe gradients.

    Identical to ``torch.logaddexp`` in the forward pass, but the backward
    pass returns zero gradients when both *x* and *y* are ``-inf`` instead
    of NaN (which ``torch.logaddexp`` produces).
    """
    return _LogAddExp.apply(x, y)
