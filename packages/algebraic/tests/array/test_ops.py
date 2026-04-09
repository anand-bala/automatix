"""Tests for algebraic ops module."""

from __future__ import annotations

import typing

import algebraic
import pytest
from algebraic import AlgebraicArray, Semiring


@pytest.fixture(
    params=[
        ("counting_semiring", dict()),
        ("max_min_algebra", {"smooth": True}),
        ("tropical_semiring", {"minplus": True}),
        ("boolean_algebra", {"mode": "soft"}),
        ("boolean_algebra", {"mode": "logic"}),
        ("boolean_algebra", {"mode": "std-fuzzy"}),
    ],
)
def semiring(request: pytest.FixtureRequest) -> Semiring:
    name, kwargs = request.param
    make_semiring: typing.Callable[..., Semiring] = getattr(algebraic.semirings, name)
    return make_semiring(**kwargs)


class TestOpsBasics:
    """Test basic functionality of algebraic operations."""

    def test_add(self, backend: str, semiring: Semiring) -> None:
        a = algebraic.zeros((3, 3), semiring=semiring, backend=backend)
        b = algebraic.zeros((3, 3), semiring=semiring, backend=backend)
        result = algebraic.add(a, b)
        assert isinstance(result, AlgebraicArray)
        assert result.semiring is semiring

    def test_matmul(self, backend: str, semiring: Semiring) -> None:

        a = algebraic.zeros((3, 3), semiring=semiring, backend=backend)
        b = algebraic.zeros((3, 3), semiring=semiring, backend=backend)
        result = algebraic.matmul(a, b)
        assert isinstance(result, AlgebraicArray)
        assert result.semiring is semiring

    def test_sum(self, backend: str, semiring: Semiring) -> None:

        a = algebraic.zeros((3, 3), semiring=semiring, backend=backend)
        result = algebraic.sum(a)
        assert isinstance(result, AlgebraicArray)
        assert result.semiring is semiring

    def test_matrix_transpose(self, backend: str, semiring: Semiring) -> None:

        a = algebraic.zeros((3, 4), semiring=semiring, backend=backend)
        result = algebraic.matrix_transpose(a)
        assert isinstance(result, AlgebraicArray)
        assert result.shape == (4, 3)
        assert result.semiring is semiring

    def test_reshape(self, backend: str, semiring: Semiring) -> None:

        a = algebraic.zeros((3, 4), semiring=semiring, backend=backend)
        result = algebraic.reshape(a, (12,))
        assert isinstance(result, AlgebraicArray)
        assert result.shape == (12,)
        assert result.semiring is semiring

    def test_multiply(self, backend: str, semiring: Semiring) -> None:

        a = algebraic.zeros((3, 3), semiring=semiring, backend=backend)
        b = algebraic.zeros((3, 3), semiring=semiring, backend=backend)
        result = algebraic.multiply(a, b)
        assert isinstance(result, AlgebraicArray)
        assert result.semiring is semiring

    def test_cumulative_sum(self, backend: str, semiring: Semiring) -> None:

        a = algebraic.zeros((5,), semiring=semiring, backend=backend)
        result = algebraic.cumulative_sum(a)
        assert isinstance(result, AlgebraicArray)
        assert result.shape == (5,)
        assert result.semiring is semiring
