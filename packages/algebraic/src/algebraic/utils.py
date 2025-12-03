"""Helper functions and types for `algebraic`"""
# Allow the use of Any
# ruff: noqa: ANN401

import functools
import typing as ty

from jaxtyping import Array

from algebraic.spec import BinaryOp, ReductionOp, UnaryOp

# Versions of functools.partial that specifically wrap Unary, Binary, and Reduction operations for type checking.


def wrap_unary_op[**P](fn: ty.Any, /, *args: P.args, **kwargs: P.kwargs) -> UnaryOp[Array]:
    assert callable(fn)
    return ty.cast(UnaryOp[Array], functools.partial(fn, *args, **kwargs))


def wrap_binary_op[**P](fn: ty.Any, /, *args: P.args, **kwargs: P.kwargs) -> BinaryOp[Array]:
    assert callable(fn)
    return ty.cast(BinaryOp[Array], functools.partial(fn, *args, **kwargs))


def wrap_reduction_op[**P](fn: ty.Any, /, *args: P.args, **kwargs: P.kwargs) -> ReductionOp:
    assert callable(fn)
    return ty.cast(ReductionOp, functools.partial(fn, *args, **kwargs))
