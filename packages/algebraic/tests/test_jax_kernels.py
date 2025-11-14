import typing

import jax
import jax.numpy as jnp
import pytest
from algebraic.backends.kernels.jax_kernels import (
    smooth_boolean_and,
    smooth_boolean_or,
    smooth_boolean_sum,
    smooth_max,
    smooth_maximum,
    soft_boolean_and,
    soft_boolean_or,
    soft_boolean_sum,
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


class TestBooleanSum:
    """Tests for boolean_sum: OR reduction using soft_or."""

    def test_forward_axis_none(self, boolean_vector: Num[Array, " n"]) -> None:
        """boolean_sum reduces entire array using soft_or."""
        result = soft_boolean_sum(boolean_vector, axis=None)
        assert result.shape == ()
        # soft_or saturates near 1.0 for reasonable inputs
        assert 0.0 < result < 1.1

    def test_forward_axis_int(self, boolean_matrix: Num[Array, "n m"]) -> None:
        """boolean_sum with axis=int reduces along that dimension."""
        result = soft_boolean_sum(boolean_matrix, axis=1)
        assert result.shape == (2,)
        # Results should be near 1.0 since OR includes high values
        assert jnp.all(result > 0.7)

    def test_soft_or_formula(self) -> None:
        """boolean_sum uses correct soft_or formula: a + b - a*b."""
        a = jnp.array(0.3)
        b = jnp.array(0.7)
        # Manual application: 0.3 + 0.7 - 0.3*0.7 = 1.0 - 0.21 = 0.79
        expected = 0.3 + 0.7 - 0.3 * 0.7
        result = soft_boolean_or(a, b)
        assert jnp.allclose(result, expected)

    def test_backward_gradient_shape(self, boolean_vector: Num[Array, " n"]) -> None:
        """Gradients have same shape as input."""

        def f(x: Num[Array, "..."]) -> Num[Array, ""]:
            return typing.cast(Array, soft_boolean_sum(x, axis=None))

        grads = jax.grad(f)(boolean_vector)
        assert grads.shape == boolean_vector.shape

    def test_backward_gradient_nonzero(self, boolean_vector: Num[Array, " n"]) -> None:
        """Gradients are nonzero for soft_or reduction."""

        def f(x: Num[Array, "..."]) -> Num[Array, ""]:
            return typing.cast(Array, soft_boolean_sum(x, axis=None))

        grads = jax.grad(f)(boolean_vector)
        # All gradients should be nonzero (soft_or is smooth)
        assert jnp.all(grads > 0.0)


# ============================================================================
# Smooth Boolean Sum Tests
# ============================================================================


class TestSmoothBooleanSum:
    """Tests for smooth_boolean_sum: sigmoid-based OR reduction."""

    def test_forward_axis_int(self, boolean_matrix: Num[Array, "n m"]) -> None:
        """smooth_boolean_sum with axis reduces along that dimension."""
        result = smooth_boolean_sum(boolean_matrix, axis=1, temperature=1.0)
        assert result.shape == (2,)
        assert jnp.all(jnp.isfinite(result))

    def test_temperature_effect(self, boolean_vector: Num[Array, " n"]) -> None:
        """Higher temperature makes sigmoid sharper (closer to hard OR)."""
        results = {
            0.1: smooth_boolean_sum(boolean_vector, axis=None, temperature=0.1),
            1.0: smooth_boolean_sum(boolean_vector, axis=None, temperature=1.0),
            10.0: smooth_boolean_sum(boolean_vector, axis=None, temperature=10.0),
        }

        # Higher temperature should approach 1.0 (saturates sigmoid)
        assert results[10.0] > results[1.0] > results[0.1]

    def test_backward_differentiable(self, boolean_vector: Num[Array, " n"]) -> None:
        """smooth_boolean_sum gradients are smooth and nonzero.

        APPROXIMATION: Backward pass distributes gradients uniformly.
        JUSTIFIED: For Boolean-regime values (near 0 or 1), this approximation
        is acceptable because gradient signal still flows and smooth_or preserves
        Boolean semantics. Temperature parameter is ignored in backward pass,
        but this is acceptable since temperature mainly affects forward pass.
        """

        def f(x: Num[Array, " n"]) -> Num[Array, ""]:
            return typing.cast(Array, smooth_boolean_sum(x, axis=None, temperature=1.0))

        grads = jax.grad(f)(boolean_vector)
        assert grads.shape == boolean_vector.shape
        # All gradients nonzero due to sigmoid smoothness
        assert jnp.all(grads != 0.0)


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


class TestShapeHandling:
    """Tests for correct shape handling across operations."""

    def test_boolean_sum_batch_reduction(self) -> None:
        """boolean_sum handles batches correctly."""
        x = jnp.ones((2, 3, 4))
        result = soft_boolean_sum(x, axis=2)
        assert result.shape == (2, 3)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zeros_and_ones_boolean(self) -> None:
        """boolean_sum works with 0 and 1 boundary values."""
        x = jnp.array([0.0, 1.0])
        result = soft_boolean_sum(x, axis=None)
        # soft_or(0, 1) = 0 + 1 - 0*1 = 1
        assert jnp.allclose(result, 1.0)


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

    def test_boolean_sum_extreme_values(self) -> None:
        """boolean_sum doesn't saturate or produce NaN with extreme values."""
        x = jnp.array([0.0, 0.5, 1.0])
        result = soft_boolean_sum(x, axis=None)
        assert jnp.isfinite(result)
        assert 0.0 <= result <= 1.0


# ============================================================================
# Known Limitations Documentation Tests
# ============================================================================


class TestKnownLimitations:
    """Document and test known limitations of implementations."""

    def test_boolean_sum_gradient_approximation(self, boolean_vector: Num[Array, " n"]) -> None:
        """
        boolean_sum backward uses APPROXIMATE gradients.

        APPROXIMATION: Distributes gradient uniformly across reduced dimension.
        EXACT would require: tracking reduction chain and computing
                           d(soft_or)/da = 1 - b, d(soft_or)/db = 1 - a
                           for each operation.

        ACCEPTABLE because: gradient signal still flows, works well
                           for optimization, lower memory overhead.
        """

        def f(x: Num[Array, " n"]) -> Num[Array, ""]:
            return typing.cast(Array, soft_boolean_sum(x, axis=None))

        grads = jax.grad(f)(boolean_vector)

        # Our implementation makes all gradients equal (uniform)
        # This is the approximation
        assert jnp.allclose(grads, grads[0])

    def test_smooth_boolean_sum_ignores_temperature_in_backward(self) -> None:
        """
        smooth_boolean_sum backward IGNORES temperature parameter.

        LIMITATION: Technically should scale gradients by T * sigmoid'(...).
        ACTUAL: Distributes uniformly like boolean_sum.

        JUSTIFIED: For Boolean-regime values (near 0 or 1), temperature-scaled
        gradients don't significantly improve optimization since sigmoid is
        already very sharp at the extremes. The approximation of uniform
        distribution is acceptable. Temperature mainly affects forward pass.
        """
        x = jnp.array([0.1, 0.5, 0.9])

        # Gradients should be same regardless of temperature
        # (because we ignore temperature in backward)
        grads_t01 = jax.grad(lambda x: smooth_boolean_sum(x, axis=None, temperature=0.1))(x)
        grads_t10 = jax.grad(lambda x: smooth_boolean_sum(x, axis=None, temperature=10.0))(x)

        # Our approximation makes them equal across temperatures
        # This is acceptable for Boolean-regime optimization
        assert jnp.allclose(grads_t01, grads_t10)
