"""Tests for torchify wrapping of algebraic polynomial types."""
# ruff: noqa: ANN201, ANN001
# mypy: disable-error-code="no-untyped-call,no-untyped-def,import-not-found,arg-type"

from __future__ import annotations

import algebraic
import torch
from algebraic.polynomials import MonomialBasis, PolyDict, RankDecomposition
from algebraic.semirings import boolean_algebra, max_min_algebra
from algebraic.types import is_torch_array
from algebraic.utils.torch import TorchWrapper, torchify


class TestTorchifyRankDecomposition:
    """Test torchify wrapping of RankDecomposition."""

    def test_returns_torch_wrapper(self) -> None:
        alg = max_min_algebra()
        x0 = RankDecomposition.variable(0, 2, alg, backend="torch")
        wrapped = torchify(x0)
        assert isinstance(wrapped, TorchWrapper)

    def test_exposes_factors_as_parameter(self) -> None:
        """The underlying factors tensor should be a registered nn.Parameter."""
        alg = max_min_algebra()
        x0 = RankDecomposition.variable(0, 2, alg, backend="torch")
        wrapped = torchify(x0)

        params = dict(wrapped.named_parameters())
        assert len(params) == 1
        assert next(iter(params.values())).shape == x0.factors.data.shape

    def test_parameter_values_match(self) -> None:
        """Registered parameter data should match the original factors."""
        alg = max_min_algebra()
        x0 = RankDecomposition.variable(0, 2, alg, backend="torch")
        wrapped = torchify(x0)

        param = next(iter(wrapped.parameters()))
        factors = x0.factors.data
        assert is_torch_array(factors)
        assert torch.allclose(param, factors, equal_nan=True)


class TestTorchifyPolyDict:
    """Test torchify wrapping of PolyDict."""

    def test_returns_torch_wrapper(self) -> None:
        alg = max_min_algebra()
        x0 = PolyDict.variable(0, 2, algebra=alg, backend="torch")
        wrapped = torchify(x0)
        assert isinstance(wrapped, TorchWrapper)

    def test_exposes_coefficients_as_parameters(self) -> None:
        """Each monomial coefficient should be a registered nn.Parameter."""
        alg = max_min_algebra()
        x0 = PolyDict.variable(0, 2, algebra=alg, backend="torch")
        x1 = PolyDict.variable(1, 2, algebra=alg, backend="torch")
        p = x0 + x1
        wrapped = torchify(p)

        # p has two monomials, so two AlgebraicArray leaves → two parameters
        assert len(list(wrapped.parameters())) == len(p)


class TestTorchifyMonomialBasis:
    """Test torchify wrapping of MonomialBasis."""

    def test_returns_torch_wrapper(self) -> None:
        alg = boolean_algebra(mode="logic")
        x0 = MonomialBasis.variable(0, 2, algebra=alg, backend="torch")
        wrapped = torchify(x0)
        assert isinstance(wrapped, TorchWrapper)

    def test_exposes_coeffs_as_parameter(self) -> None:
        """The coefficients tensor should be a registered nn.Parameter."""
        alg = boolean_algebra(mode="logic")
        x0 = MonomialBasis.variable(0, 2, algebra=alg, backend="torch")
        wrapped = torchify(x0)

        params = list(wrapped.parameters())
        assert len(params) == 1
        assert params[0].shape == x0.coeffs.data.shape


class TestTorchifyAlgebraicArray:
    """Test torchify wrapping of plain AlgebraicArray."""

    def test_returns_torch_wrapper(self) -> None:
        alg = max_min_algebra()
        arr = algebraic.array([1.0, 2.0, 3.0], semiring=alg, backend="torch")
        wrapped = torchify(arr)
        assert isinstance(wrapped, TorchWrapper)

    def test_exposes_data_as_parameter(self) -> None:
        """The underlying ``.data`` tensor should be a registered nn.Parameter."""
        alg = max_min_algebra()
        arr = algebraic.array([1.0, 2.0, 3.0], semiring=alg, backend="torch")
        wrapped = torchify(arr)

        params = dict(wrapped.named_parameters())
        assert len(params) == 1
        assert next(iter(params.values())).shape == arr.data.shape

    def test_parameter_values_match(self) -> None:
        """Registered parameter data should match the original ``.data``."""
        alg = max_min_algebra()
        arr = algebraic.array([1.0, 2.0, 3.0], semiring=alg, backend="torch")
        assert is_torch_array(arr.data)
        wrapped = torchify(arr)

        param = next(iter(wrapped.parameters()))
        assert torch.allclose(param, arr.data)

    def test_mapping_of_arrays(self) -> None:
        """A mapping of AlgebraicArrays should register one parameter per entry."""
        alg = boolean_algebra(mode="logic")
        arrays = {
            "a": algebraic.array([1, 0], semiring=alg, backend="torch"),
            "b": algebraic.array([0, 1], semiring=alg, backend="torch"),
        }
        wrapped = torchify(arrays)

        assert len(list(wrapped.parameters())) == 2

    def test_sequence_of_arrays(self) -> None:
        """A sequence of AlgebraicArrays should register one param per element."""
        alg = max_min_algebra()
        arrays = [
            algebraic.array([1.0, 2.0], semiring=alg, backend="torch"),
            algebraic.array([3.0, 4.0], semiring=alg, backend="torch"),
            algebraic.array([5.0, 6.0], semiring=alg, backend="torch"),
        ]
        wrapped = torchify(arrays)

        assert len(list(wrapped.parameters())) == 3
