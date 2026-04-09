# mypy: disable-error-code="arg-type,operator,return-value"
import jax
import jax.numpy as jnp
import pytest
from algebraic.kernels import (
    smooth_boolean_and,
    smooth_boolean_or,
    smooth_max,
    smooth_maximum,
    soft_boolean_and,
    soft_boolean_or,
)
from beartype import beartype as typechecker
from jaxtyping import Array, Num, jaxtyped

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
@jaxtyped(typechecker=typechecker)
def simple_vector() -> Num[Array, " 3"]:
    """Simple 1D test vector."""
    return jnp.array([1.0, 2.0, 3.0])


@pytest.fixture
@jaxtyped(typechecker=typechecker)
def simple_matrix() -> Num[Array, "2 3"]:
    """Simple 2D test matrix."""
    return jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
@jaxtyped(typechecker=typechecker)
def simple_3d_array() -> Num[Array, "2 2 2"]:
    """Simple 3D test array."""
    return jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])


@pytest.fixture
@jaxtyped(typechecker=typechecker)
def boolean_vector() -> Num[Array, " 3"]:
    """Vector with values in [0, 1] for boolean operations."""
    return jnp.array([0.1, 0.5, 0.9])


@pytest.fixture
@jaxtyped(typechecker=typechecker)
def boolean_matrix() -> Num[Array, "2 3"]:
    """Matrix with values in [0, 1] for boolean operations."""
    return jnp.array([[0.1, 0.5, 0.9], [0.2, 0.3, 0.7]])


@pytest.fixture
@jaxtyped(typechecker=typechecker)
def matmul_pair() -> tuple[Num[Array, "2 2"], Num[Array, "2 2"]]:
    """Pair of matrices for matrix multiplication tests."""
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = jnp.array([[2.0, 1.0], [1.0, 2.0]])
    return a, b


# ============================================================================
# MaxPlus Sum Tests
# ============================================================================


class TestSmoothMaxplusSum:
    """Tests for smooth_maxplus_sum: smooth max approximation."""

    def test_forward_approximates_max(self, simple_vector: Num[Array, " 3"]) -> None:
        """smooth_maxplus_sum approximates max with high temperature."""
        exact = jnp.max(simple_vector, axis=None)
        smooth = smooth_max(simple_vector, axis=None, temperature=100.0)
        # High temperature should give close approximation to max
        assert jnp.allclose(exact, smooth, atol=0.1)

    def test_temperature_effect(self, simple_vector: Num[Array, " 3"]) -> None:
        """Higher temperature makes smooth function closer to hard max."""
        results = [smooth_max(simple_vector, axis=None, temperature=T) for T in [0.1, 1.0, 10.0]]
        exact = jnp.max(simple_vector, axis=None)

        # Results should monotonically approach exact max as T increases
        diffs = [abs(r - exact) for r in results]
        assert diffs[0] > diffs[1] > diffs[2]

    def test_backward_differentiable(self, simple_vector: Num[Array, " 3"]) -> None:
        """smooth_maxplus_sum gradients are nonzero for all elements."""

        def f(x: Array) -> Array:
            return smooth_max(x, axis=None, temperature=1.0)

        grads = jax.grad(f)(simple_vector)

        # Unlike hard max, smooth version gives nonzero gradients everywhere
        assert not jnp.any(grads == 0.0)
        assert grads.shape == simple_vector.shape


@pytest.mark.parametrize(
    "x,y,temperature",
    [
        (jnp.array(2.0), jnp.array(3.0), 1.0),
        (jnp.array([1.0, 2.0]), jnp.array([2.0, 1.0]), 2.0),
    ],
)
class TestSmoothMaxplusAddition:
    """Tests for smooth_maxplus_addition: smooth max approximation."""

    def test_forward(self, x: Array, y: Array, temperature: float) -> None:
        """smooth_maxplus_addition approximates max."""
        result = smooth_maximum(x, y, temperature=temperature)
        exact = jnp.maximum(x, y)
        # With T=1.0, should be reasonably close
        if temperature >= 1.0:
            assert jnp.allclose(result, exact, atol=0.5)

    def test_differentiable(self, x: Array, y: Array, temperature: float) -> None:
        """smooth_maxplus_addition has nonzero gradients."""

        @jaxtyped(typechecker=typechecker)
        def f(x: Num[Array, "..."]) -> Num[Array, ""]:
            return jnp.sum(smooth_maximum(x, y, temperature=temperature))

        grads = jax.grad(f)(x)
        # Should have gradients everywhere
        assert not jnp.all(grads == 0.0)


class TestBooleanOr:
    """Tests for soft_boolean_or: soft OR operation."""

    def test_soft_or_formula(self) -> None:
        """boolean_sum uses correct soft_or formula: a + b - a*b."""
        a = jnp.array(0.3)
        b = jnp.array(0.7)
        # Manual application: 0.3 + 0.7 - 0.3*0.7 = 1.0 - 0.21 = 0.79
        expected = 0.3 + 0.7 - 0.3 * 0.7
        result = soft_boolean_or(a, b)
        assert jnp.allclose(result, expected)


# ============================================================================
# Basic Soft/Smooth Boolean Operations Tests
# ============================================================================


class TestSoftBoolean:
    """Tests for soft Boolean operations: AND, OR, NOT."""

    def test_soft_and(self) -> None:
        """soft_and: x * y (multiplicative relaxation)."""
        x, y = jnp.array(0.3), jnp.array(0.7)
        result = soft_boolean_and(x, y)
        expected = 0.3 * 0.7
        assert jnp.allclose(result, expected)

    def test_soft_or(self) -> None:
        """soft_or: x + y - x*y (probabilistic OR)."""
        x, y = jnp.array(0.3), jnp.array(0.7)
        result = soft_boolean_or(x, y)
        expected = 0.3 + 0.7 - 0.3 * 0.7
        assert jnp.allclose(result, expected)

    def test_smooth_and(self) -> None:
        """smooth_and: sigmoid(T * (x + y - 1))."""
        x, y = jnp.array(0.3), jnp.array(0.7)
        result = smooth_boolean_and(x, y, temperature=1.0)
        # Should be between 0 and 1
        assert 0.0 <= result <= 1.0

    def test_smooth_or(self) -> None:
        """smooth_or: sigmoid(T * (x + y))."""
        x, y = jnp.array(0.3), jnp.array(0.7)
        result = smooth_boolean_or(x, y, temperature=1.0)
        # For high values, sigmoid should give close to 1
        assert result > 0.5


# ============================================================================
# Shape and Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    pass


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Tests for numerical stability of implementations."""

    def test_smooth_maxplus_large_values(self) -> None:
        """smooth_maxplus_sum handles large values via logsumexp."""
        x = jnp.array([1e3, 1e3 + 1.0, 1e3 - 1.0])
        result = smooth_max(x, axis=None, temperature=1.0)
        # Should not overflow or produce NaN
        assert jnp.isfinite(result)

    pass
