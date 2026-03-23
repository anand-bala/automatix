"""Shared fixtures and utilities for polynomial tests."""

# ruff: noqa: ANN001, ANN201, ANN202, ANN204
# mypy: disable-error-code="no-untyped-call,no-untyped-def,import-not-found"

from __future__ import annotations

import algebraic
import pytest


@pytest.fixture
def bool_algebra() -> algebraic.BooleanAlgebra:
    """Boolean algebra for tests."""

    return algebraic.semirings.boolean_algebra(mode="logic")


@pytest.fixture
def tropical_minplus_algebra() -> algebraic.Semiring:
    """Max-min algebra (restricted to negative reals) - similar to tropical min-plus."""

    # This gives a lattice with: add=max, mul=min, zero=-inf, one=inf
    # While not exactly tropical min-plus, it's a lattice that tests similar properties
    return algebraic.semirings.tropical_semiring(minplus=True)


@pytest.fixture
def tropical_maxplus_algebra() -> algebraic.Semiring:
    """Max-min algebra (restricted to positive reals) - similar to tropical max-plus."""

    # This gives a lattice with: add=max, mul=min, zero=0, one=inf
    return algebraic.semirings.tropical_semiring(minplus=False)


@pytest.fixture
def maxmin_algebra() -> algebraic.DeMorganAlgebra:
    """Max-min algebra (full De Morgan algebra with complement)."""

    return algebraic.semirings.max_min_algebra()
