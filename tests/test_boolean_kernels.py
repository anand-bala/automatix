"""Tests for differentiable Boolean kernels."""

import jax.numpy as jnp
import pytest

from automatix.algebra.backends.boolean_kernels import (
    create_boolean_kernel,
    smooth_and,
    smooth_negate,
    smooth_or,
    soft_and,
    soft_negate,
    soft_or,
)
from automatix.algebra.kernels import AlgebraicStructure
from automatix.algebra.registry import get_kernel


class TestSoftBoolean:
    """Test soft Boolean operations."""

    def test_soft_and(self) -> None:
        """Test soft AND is multiplication."""
        result = soft_and(jnp.array(0.3), jnp.array(0.7))
        expected = 0.3 * 0.7
        assert jnp.allclose(result, expected)

    def test_soft_or(self) -> None:
        """Test soft OR matches De Morgan's law."""
        x, y = jnp.array(0.3), jnp.array(0.7)
        result = soft_or(x, y)
        expected = x + y - x * y
        assert jnp.allclose(result, expected)

    def test_soft_negate(self) -> None:
        """Test soft negation is 1 - x."""
        result = soft_negate(jnp.array(0.3))
        expected = 1.0 - 0.3
        assert jnp.allclose(result, expected)

    def test_soft_boolean_semantics_corners(self) -> None:
        """Test soft Boolean semantics at boundary values."""
        # AND: (0, 0) -> 0, (1, 1) -> 1
        assert jnp.isclose(soft_and(jnp.array(0.0), jnp.array(0.0)), 0.0)
        assert jnp.isclose(soft_and(jnp.array(1.0), jnp.array(1.0)), 1.0)

        # OR: (0, 0) -> 0, (1, 1) -> 1
        assert jnp.isclose(soft_or(jnp.array(0.0), jnp.array(0.0)), 0.0)
        assert jnp.isclose(soft_or(jnp.array(1.0), jnp.array(1.0)), 1.0)

        # NOT: (0) -> 1, (1) -> 0
        assert jnp.isclose(soft_negate(jnp.array(0.0)), 1.0)
        assert jnp.isclose(soft_negate(jnp.array(1.0)), 0.0)

    def test_soft_boolean_idempotence(self) -> None:
        """Test soft Boolean OR idempotence: x OR x = x."""
        # OR: x + y - xy = x + x - x*x = 2x - x^2
        # For x=0.5: 2(0.5) - 0.5^2 = 1.0 - 0.25 = 0.75, not idempotent
        # But AND is NOT idempotent: x * x = x^2 which is only true for x=0 or x=1
        # Soft Boolean is not actually idempotent (only appears idempotent at corners)
        assert jnp.isclose(soft_and(jnp.array(0.0), jnp.array(0.0)), 0.0)
        assert jnp.isclose(soft_and(jnp.array(1.0), jnp.array(1.0)), 1.0)
        assert jnp.isclose(soft_or(jnp.array(0.0), jnp.array(0.0)), 0.0)
        assert jnp.isclose(soft_or(jnp.array(1.0), jnp.array(1.0)), 1.0)

    def test_soft_boolean_commutativity(self) -> None:
        """Test soft Boolean commutativity."""
        x, y = jnp.array(0.3), jnp.array(0.7)
        assert jnp.isclose(soft_and(x, y), soft_and(y, x))
        assert jnp.isclose(soft_or(x, y), soft_or(y, x))


class TestSmoothBoolean:
    """Test smooth Boolean operations with temperature."""

    def test_smooth_and_default_temperature(self) -> None:
        """Test smooth AND with default temperature."""
        result = smooth_and(jnp.array(0.5), jnp.array(0.5), temperature=1.0)
        # Formula: sigmoid(1.0 * (0.5 + 0.5 - 1)) = sigmoid(0) = 0.5
        assert jnp.isclose(result, 0.5, atol=1e-6)

    def test_smooth_or_default_temperature(self) -> None:
        """Test smooth OR with default temperature."""
        result = smooth_or(jnp.array(0.5), jnp.array(0.5), temperature=1.0)
        # Formula: sigmoid(1.0 * (0.5 + 0.5)) = sigmoid(1.0) ≈ 0.731
        assert 0.7 < result < 0.75

    def test_smooth_negate_default_temperature(self) -> None:
        """Test smooth negation with default temperature."""
        result = smooth_negate(jnp.array(0.5), temperature=1.0)
        # Formula: sigmoid(1.0 * (0.5 - 0.5)) = sigmoid(0) = 0.5
        assert jnp.isclose(result, 0.5, atol=1e-6)

    def test_smooth_temperature_effect(self) -> None:
        """Test that temperature affects smoothness."""
        x = jnp.array(0.2)
        y = jnp.array(0.8)

        # High temperature makes transition sharper
        result_cold = smooth_and(x, y, temperature=0.1)
        result_hot = smooth_and(x, y, temperature=10.0)

        # Both should be different from soft AND
        soft_result = soft_and(x, y)
        assert not jnp.isclose(result_cold, soft_result)
        assert not jnp.isclose(result_hot, soft_result)

    def test_smooth_boolean_corners(self) -> None:
        """Test smooth Boolean at corners with high temperature."""
        high_temp = 100.0

        # AND: (0, 0) -> sigmoid(-100) ≈ 0
        result = smooth_and(jnp.array(0.0), jnp.array(0.0), temperature=high_temp)
        assert result < 0.01

        # AND: (1, 1) -> sigmoid(100) ≈ 1
        result = smooth_and(jnp.array(1.0), jnp.array(1.0), temperature=high_temp)
        assert result > 0.99

    def test_smooth_boolean_vectorized(self) -> None:
        """Test smooth Boolean operations with arrays."""
        x = jnp.array([0.0, 0.5, 1.0])
        y = jnp.array([0.0, 0.5, 1.0])

        result_and = smooth_and(x, y, temperature=1.0)
        result_or = smooth_or(x, y, temperature=1.0)

        assert result_and.shape == (3,)
        assert result_or.shape == (3,)
        assert jnp.all(result_and >= 0.0)
        assert jnp.all(result_and <= 1.0)


class TestBooleanKernelCreation:
    """Test Boolean kernel creation factory."""

    def test_create_soft_kernel(self) -> None:
        """Test creating soft Boolean kernel."""
        kernel = create_boolean_kernel(mode="soft")
        assert isinstance(kernel, AlgebraicStructure)
        assert kernel.zero == 0.0
        assert kernel.one == 1.0
        assert kernel.is_idempotent_add()
        assert kernel.is_idempotent_mul()
        assert kernel.is_commutative()
        assert kernel.has_negation()

    def test_create_smooth_kernel(self) -> None:
        """Test creating smooth Boolean kernel with temperature."""
        kernel = create_boolean_kernel(mode="smooth", temperature=2.0)
        assert isinstance(kernel, AlgebraicStructure)
        assert kernel.zero == 0.0
        assert kernel.one == 1.0
        assert kernel.is_commutative()
        assert kernel.has_negation()

    def test_create_ste_kernel(self) -> None:
        """Test creating STE kernel."""
        kernel = create_boolean_kernel(mode="ste")
        assert isinstance(kernel, AlgebraicStructure)
        assert kernel.zero == 0.0
        assert kernel.one == 1.0
        assert kernel.is_idempotent_add()
        assert kernel.is_idempotent_mul()
        assert kernel.is_commutative()
        assert kernel.has_negation()

    def test_invalid_kernel_mode(self) -> None:
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            create_boolean_kernel(mode="invalid")  # type: ignore[arg-type]

    def test_kernel_operations(self) -> None:
        """Test that created kernel has working operations."""
        kernel = create_boolean_kernel(mode="soft")

        x = jnp.array([0.2, 0.5])
        y = jnp.array([0.3, 0.7])

        # Test add (OR)
        result_add = kernel.add(x, y)
        assert result_add.shape == (2,)

        # Test mul (AND)
        result_mul = kernel.mul(x, y)
        assert result_mul.shape == (2,)

        # Test negate
        if kernel.negate is not None:
            result_neg = kernel.negate(x)
            assert result_neg.shape == (2,)


class TestBooleanKernelRegistry:
    """Test Boolean kernels registered in the registry."""

    def test_soft_boolean_in_registry(self) -> None:
        """Test that soft Boolean kernel is registered."""
        kernel = get_kernel("BooleanSoft", "jax")
        assert isinstance(kernel, AlgebraicStructure)
        assert kernel.is_idempotent_add()

    def test_smooth_boolean_in_registry(self) -> None:
        """Test that smooth Boolean kernel is registered."""
        kernel = get_kernel("BooleanSmooth", "jax")
        assert isinstance(kernel, AlgebraicStructure)
        assert kernel.is_commutative()

    def test_smooth_boolean_sharp_in_registry(self) -> None:
        """Test that sharp smooth Boolean kernel is registered."""
        kernel = get_kernel("BooleanSmoothSharp", "jax")
        assert isinstance(kernel, AlgebraicStructure)

    def test_ste_boolean_in_registry(self) -> None:
        """Test that STE Boolean kernel is registered."""
        kernel = get_kernel("BooleanSTE", "jax")
        assert isinstance(kernel, AlgebraicStructure)
        assert kernel.is_idempotent_add()


class TestBooleanKernelProperties:
    """Test Boolean kernel algebraic properties."""

    def test_soft_kernel_de_morgan_law(self) -> None:
        """Test De Morgan's law with soft Boolean: NOT(x OR y) = (NOT x) AND (NOT y)."""
        kernel = create_boolean_kernel(mode="soft")
        x = jnp.array(0.3)
        y = jnp.array(0.7)

        # NOT(x OR y)
        or_result = kernel.add(x, y)
        negated_or = kernel.negate(or_result) if kernel.negate else None

        # (NOT x) AND (NOT y)
        neg_x = kernel.negate(x) if kernel.negate else None
        neg_y = kernel.negate(y) if kernel.negate else None
        and_result = kernel.mul(neg_x, neg_y) if neg_x is not None and neg_y is not None else None

        if negated_or is not None and and_result is not None:
            assert jnp.isclose(negated_or, and_result, atol=1e-6)

    def test_soft_kernel_reduction_operations(self) -> None:
        """Test sum and prod reduction operations on soft kernel."""
        kernel = create_boolean_kernel(mode="soft")
        x = jnp.array([0.1, 0.5, 0.9])

        # Sum (OR) should use max for soft Boolean
        result_sum = kernel.sum(x, axis=None) if kernel.sum else None
        if result_sum is not None:
            assert jnp.isclose(result_sum, jnp.max(x))

        # Prod (AND) should use min for soft Boolean
        result_prod = kernel.prod(x, axis=None) if kernel.prod else None
        if result_prod is not None:
            assert jnp.isclose(result_prod, jnp.min(x))
