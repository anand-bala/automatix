"""Tests for kernel compatibility layer and backward compatibility."""

import jax.numpy as jnp
import pytest
from automatix.algebra._compat import normalize_semiring
from automatix.algebra.backends.jax_ import (
    CountingSemiring,
    MaxMinSemiring,
    MaxPlusSemiring,
    MinPlusSemiring,
)
from automatix.algebra.kernels import AlgebraicStructure
from automatix.algebra.registry import get_kernel


class TestNormalizeSemiring:
    """Test normalize_semiring adapter function."""

    def test_normalize_semiring_class(self) -> None:
        """Test that normalize_semiring converts class to kernel."""
        kernel = normalize_semiring(MaxPlusSemiring)
        assert isinstance(kernel, AlgebraicStructure)
        assert kernel.zero == float("-inf")
        assert kernel.one == 0.0

    def test_normalize_semiring_kernel(self) -> None:
        """Test that normalize_semiring passes through kernel instances."""
        kernel1 = MaxPlusSemiring.to_kernel()
        kernel2 = normalize_semiring(kernel1)
        assert kernel2 is kernel1

    def test_normalize_semiring_invalid_type(self) -> None:
        """Test that normalize_semiring raises TypeError for invalid input."""
        with pytest.raises(TypeError):
            normalize_semiring("invalid")  # type: ignore[arg-type]

    def test_normalize_all_semirings(self) -> None:
        """Test normalization works for all JAX semirings."""
        semirings = [
            MaxPlusSemiring,
            MinPlusSemiring,
            CountingSemiring,
            MaxMinSemiring,
        ]
        for semiring in semirings:
            kernel = normalize_semiring(semiring)
            assert isinstance(kernel, AlgebraicStructure)


class TestKernelProperties:
    """Test AlgebraicStructure kernel properties."""

    def test_kernel_has_required_operations(self) -> None:
        """Test that kernel has all required operations."""
        kernel = MaxPlusSemiring.to_kernel()
        assert callable(kernel.add)
        assert callable(kernel.mul)
        assert kernel.zero is not None
        assert kernel.one is not None

    def test_kernel_properties_tracking(self) -> None:
        """Test that properties are correctly tracked."""
        kernel = MaxMinSemiring.to_kernel()
        assert kernel.is_idempotent_add()
        assert kernel.is_idempotent_mul()
        assert kernel.is_commutative()

    def test_kernel_creates_zeros_ones(self) -> None:
        """Test that kernel can create zero and one arrays."""
        kernel = MaxPlusSemiring.to_kernel()
        zeros = kernel.zeros((2, 3))
        ones = kernel.ones((2, 3))
        assert zeros.shape == (2, 3)
        assert ones.shape == (2, 3)
        assert jnp.all(zeros == float("-inf"))
        assert jnp.all(ones == 0.0)


class TestAndOrCompatibility:
    """Test that And/Or predicates work with both class and kernel APIs."""

    def test_and_with_semiring_class(self) -> None:
        """Test And predicate with semiring class."""
        from automatix.predicates import And, Predicate

        pred1 = Predicate(lambda x: jnp.asarray(1.0))
        pred2 = Predicate(lambda x: jnp.asarray(2.0))
        and_pred = And(args=[pred1, pred2], semiring=MaxPlusSemiring)

        # After __post_init__, should be normalized to kernel
        assert isinstance(and_pred.semiring, AlgebraicStructure)

    def test_and_with_kernel(self) -> None:
        """Test And predicate with kernel directly."""
        from automatix.predicates import And, Predicate

        pred1 = Predicate(lambda x: jnp.asarray(1.0))
        pred2 = Predicate(lambda x: jnp.asarray(2.0))
        kernel = MaxPlusSemiring.to_kernel()
        and_pred = And(args=[pred1, pred2], semiring=kernel)

        assert isinstance(and_pred.semiring, AlgebraicStructure)

    def test_or_with_semiring_class(self) -> None:
        """Test Or predicate with semiring class."""
        from automatix.predicates import Or, Predicate

        pred1 = Predicate(lambda x: jnp.asarray(1.0))
        pred2 = Predicate(lambda x: jnp.asarray(2.0))
        or_pred = Or(args=[pred1, pred2], semiring=MinPlusSemiring)

        # After __post_init__, should be normalized to kernel
        assert isinstance(or_pred.semiring, AlgebraicStructure)

    def test_or_with_kernel(self) -> None:
        """Test Or predicate with kernel directly."""
        from automatix.predicates import Or, Predicate

        pred1 = Predicate(lambda x: jnp.asarray(1.0))
        pred2 = Predicate(lambda x: jnp.asarray(2.0))
        kernel = MinPlusSemiring.to_kernel()
        or_pred = Or(args=[pred1, pred2], semiring=kernel)

        assert isinstance(or_pred.semiring, AlgebraicStructure)


class TestAutomatonOperatorCompatibility:
    """Test that automaton operators work with both class and kernel APIs."""

    def test_predicate_with_semiring_class(self) -> None:
        """Test that predicates properly normalize semiring classes."""
        from automatix.predicates import Predicate

        pred = Predicate(lambda x: x > 0)
        # Create And with class-based semiring
        from automatix.predicates import And

        and_pred = And(args=[pred, pred], semiring=MaxPlusSemiring)

        # After __post_init__, should be normalized to kernel
        assert isinstance(and_pred.semiring, AlgebraicStructure)

    def test_predicate_with_kernel(self) -> None:
        """Test that predicates work with kernel directly."""
        from automatix.predicates import And, Predicate

        pred = Predicate(lambda x: x > 0)
        kernel = MaxPlusSemiring.to_kernel()
        and_pred = And(args=[pred, pred], semiring=kernel)

        assert isinstance(and_pred.semiring, AlgebraicStructure)
        assert and_pred.semiring is kernel


class TestRegistryKernelLookup:
    """Test kernel registry functions."""

    def test_get_kernel_from_registry(self) -> None:
        """Test getting kernel from registry."""
        kernel = get_kernel("MaxPlus", "jax")
        assert isinstance(kernel, AlgebraicStructure)
        assert kernel.zero == float("-inf")

    def test_list_registered_kernels(self) -> None:
        """Test listing registered kernels."""
        from automatix.algebra.registry import list_kernels

        kernels = list_kernels("jax")
        assert isinstance(kernels, list)
        assert len(kernels) > 0
        assert "MaxPlus" in kernels

    def test_kernel_properties_match_semiring(self) -> None:
        """Test that kernel properties match semiring properties."""
        semiring = MaxMinSemiring
        kernel = get_kernel("MaxMin", "jax")

        assert kernel.is_idempotent_add() == semiring.is_additively_idempotent
        assert kernel.is_idempotent_mul() == semiring.is_multiplicatively_idempotent
        assert kernel.is_commutative() == semiring.is_commutative
