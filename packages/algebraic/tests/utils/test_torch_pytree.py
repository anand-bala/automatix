"""Tests for PyTreeModule wrapping of algebraic polynomial types."""
# ruff: noqa: ANN201, ANN001
# mypy: disable-error-code="no-untyped-call,no-untyped-def,import-not-found,arg-type"

from __future__ import annotations

import algebraic
import torch
from algebraic.polynomials import MonomialBasis, PolyDict, RankDecomposition
from algebraic.polynomials.rank_decomp import LowRankFactors
from algebraic.semirings import boolean_algebra, max_min_algebra
from algebraic.types import is_torch_array
from algebraic.utils.torch import PyTreeModule, torchify


class TestTorchifyRankDecomposition:
    """Test torchify wrapping of RankDecomposition."""

    def test_returns_pytree_module(self) -> None:
        alg = max_min_algebra()
        x0 = RankDecomposition.variable(0, 2, alg, backend="torch")
        wrapped = torchify(x0)
        assert isinstance(wrapped, PyTreeModule)

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

    def test_forward_returns_rank_decomposition(self) -> None:
        """forward() should return a fully functional RankDecomposition."""
        alg = max_min_algebra()
        x0 = RankDecomposition.variable(0, 2, alg, backend="torch")
        wrapped = torchify(x0)

        reconstructed = wrapped()
        assert isinstance(reconstructed, RankDecomposition)
        assert reconstructed.rank == x0.rank
        assert reconstructed.num_vars == x0.num_vars
        assert torch.allclose(
            torch.as_tensor(reconstructed.factors.data),
            torch.as_tensor(x0.factors.data),
            equal_nan=True,
        )


class TestTorchifyLowRankFactors:
    """Test torchify wrapping of LowRankFactors."""

    def test_returns_pytree_module(self) -> None:
        alg = max_min_algebra()
        lrf = LowRankFactors.variable(0, 2, alg, backend="torch")
        wrapped = torchify(lrf)
        assert isinstance(wrapped, PyTreeModule)

    def test_exposes_weights_and_bias_as_parameters(self) -> None:
        """Should register two parameters: one for weights, one for bias."""
        alg = max_min_algebra()
        lrf = LowRankFactors.variable(0, 2, alg, backend="torch")
        wrapped = torchify(lrf)

        params = list(wrapped.parameters())
        assert len(params) == 2

    def test_forward_returns_low_rank_factors(self) -> None:
        """forward() should return a fully functional LowRankFactors."""
        alg = max_min_algebra()
        lrf = LowRankFactors.variable(0, 2, alg, backend="torch")
        wrapped = torchify(lrf)

        reconstructed = wrapped()
        assert isinstance(reconstructed, LowRankFactors)
        assert reconstructed.num_vars == lrf.num_vars


class TestTorchifyPolyDict:
    """Test torchify wrapping of PolyDict."""

    def test_returns_pytree_module(self) -> None:
        alg = max_min_algebra()
        x0 = PolyDict.variable(0, 2, algebra=alg, backend="torch")
        wrapped = torchify(x0)
        assert isinstance(wrapped, PyTreeModule)

    def test_exposes_coefficients_as_parameters(self) -> None:
        """Each monomial coefficient should be a registered nn.Parameter."""
        alg = max_min_algebra()
        x0 = PolyDict.variable(0, 2, algebra=alg, backend="torch")
        x1 = PolyDict.variable(1, 2, algebra=alg, backend="torch")
        p = x0 + x1
        wrapped = torchify(p)

        # p has two monomials, so two AlgebraicArray leaves -> two parameters
        assert len(list(wrapped.parameters())) == len(p)


class TestTorchifyMonomialBasis:
    """Test torchify wrapping of MonomialBasis."""

    def test_returns_pytree_module(self) -> None:
        alg = boolean_algebra(mode="logic")
        x0 = MonomialBasis.variable(0, 2, algebra=alg, backend="torch")
        wrapped = torchify(x0)
        assert isinstance(wrapped, PyTreeModule)

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

    def test_returns_pytree_module(self) -> None:
        alg = max_min_algebra()
        arr = algebraic.array([1.0, 2.0, 3.0], semiring=alg, backend="torch")
        wrapped = torchify(arr)
        assert isinstance(wrapped, PyTreeModule)

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

    def test_forward_returns_algebraic_array(self) -> None:
        """forward() should return a fully functional AlgebraicArray."""
        alg = max_min_algebra()
        arr = algebraic.array([1.0, 2.0, 3.0], semiring=alg, backend="torch")
        wrapped = torchify(arr)

        reconstructed = wrapped()
        assert isinstance(reconstructed, algebraic.AlgebraicArray)
        assert torch.allclose(
            torch.as_tensor(reconstructed.data),
            torch.as_tensor(arr.data),
        )
