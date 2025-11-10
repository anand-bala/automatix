"""Tests for polynomial evaluation algorithms.

Tests cover:
1. Algorithm 1 (monomial basis enumeration)
2. Batch evaluation via vmap
3. Evaluation on different semirings
"""

import jax.numpy as jnp
import pytest

from automatix.algebra.backends.jax_ import LatticeAlgebra, MaxMinSemiring
from automatix.algebra.polynomials import MultilinearPolynomial
from automatix.algebra.polynomials.tensor_encoding import (
    eval_algorithm_1,
    eval_algorithm_1_batch,
)


class TestEvalAlgorithm1Basic:
    """Test Algorithm 1 with simple polynomials."""

    def test_constant_polynomial(self) -> None:
        """Test evaluation of constant polynomial (degree 0)."""
        poly = MultilinearPolynomial.ones(LatticeAlgebra, num_states=2)
        values = {0: jnp.array(0.5), 1: jnp.array(0.3)}

        result = eval_algorithm_1(poly, values)

        # Constant polynomial: 1 (coefficient at index 0 only)
        # Result should be 1 * 1 = 1 (identity for multiplication)
        assert jnp.allclose(result, 1.0)

    def test_single_variable_polynomial(self) -> None:
        """Test evaluation of single-variable polynomial (degree 1)."""
        # Create polynomial x_0
        poly = MultilinearPolynomial.from_monomial(LatticeAlgebra, 2, (1, 0), jnp.asarray(1.0))
        values = {0: jnp.array(0.5), 1: jnp.array(0.3)}

        result = eval_algorithm_1(poly, values)

        # x_0 = 0.5 (for MaxMin semiring, result is just the value)
        assert jnp.allclose(result, 0.5)

    def test_two_variable_polynomial(self) -> None:
        """Test evaluation of two-variable polynomial (degree 2)."""
        # Create polynomial x_0 * x_1
        poly = MultilinearPolynomial.from_monomial(LatticeAlgebra, 2, (1, 1), jnp.asarray(1.0))
        values = {0: jnp.array(0.5), 1: jnp.array(0.3)}

        result = eval_algorithm_1(poly, values)

        # x_0 * x_1 = min(0.5, 0.3) = 0.3 (for MaxMin, multiply is min)
        assert jnp.allclose(result, 0.3)

    def test_sum_of_variables(self) -> None:
        """Test evaluation of x_0 + x_1."""
        # Create polynomial: x_0 + x_1
        poly = MultilinearPolynomial.zeros(LatticeAlgebra, num_states=2)
        poly = poly.set_monomial((1, 0), jnp.asarray(1.0))  # x_0
        poly = poly.set_monomial((0, 1), jnp.asarray(1.0))  # x_1

        values = {0: jnp.array(0.5), 1: jnp.array(0.3)}

        result = eval_algorithm_1(poly, values)

        # x_0 + x_1 = max(0.5, 0.3) = 0.5 (for MaxMin, add is max)
        assert jnp.allclose(result, 0.5)

    def test_complex_polynomial_maxmin(self) -> None:
        """Test evaluation of more complex polynomial: x_0 + x_1*x_2."""
        poly = MultilinearPolynomial.zeros(LatticeAlgebra, num_states=3)
        poly = poly.set_monomial((1, 0, 0), jnp.asarray(1.0))  # x_0 with coeff 1
        poly = poly.set_monomial((0, 1, 1), jnp.asarray(1.0))  # x_1*x_2 with coeff 1

        # Evaluate at x_0=0.4, x_1=0.6, x_2=0.8
        values = {0: jnp.array(0.4), 1: jnp.array(0.6), 2: jnp.array(0.8)}

        result = eval_algorithm_1(poly, values)

        # Result: max(0.4, min(0.6, 0.8)) = max(0.4, 0.6) = 0.6
        assert jnp.allclose(result, 0.6)

    def test_weighted_sum(self) -> None:
        """Test polynomial with non-unit coefficients."""
        poly = MultilinearPolynomial.zeros(LatticeAlgebra, num_states=2)
        poly = poly.set_monomial((1, 0), jnp.asarray(2.0))  # 2*x_0
        poly = poly.set_monomial((0, 1), jnp.asarray(3.0))  # 3*x_1

        values = {0: jnp.array(0.5), 1: jnp.array(0.4)}

        result = eval_algorithm_1(poly, values)
        # For MaxMin: max(min(2,0.5), min(3,0.4)) = max(0.5, 0.4) = 0.5
        assert result.item() == 0.5


class TestEvalAlgorithm1Validation:
    """Test input validation and error handling."""

    def test_missing_value_raises_error(self) -> None:
        """Test that missing state values raise ValueError."""
        poly = MultilinearPolynomial.zeros(LatticeAlgebra, num_states=3)
        values = {0: jnp.array(0.5), 1: jnp.array(0.3)}  # Missing state 2

        with pytest.raises(ValueError, match="Missing value for state"):
            eval_algorithm_1(poly, values)

    def test_extra_values_ignored(self) -> None:
        """Test that extra values in dictionary are ignored."""
        poly = MultilinearPolynomial.ones(LatticeAlgebra, num_states=2)
        values = {
            0: jnp.array(0.5),
            1: jnp.array(0.3),
            2: jnp.array(0.9),  # Extra
        }

        # Should not raise
        result = eval_algorithm_1(poly, values)
        assert jnp.allclose(result, 1.0)


class TestEvalAlgorithm1Batch:
    """Test batch evaluation functionality."""

    def test_batch_single_point(self) -> None:
        """Test batch evaluation with single point."""
        poly = MultilinearPolynomial.from_monomial(LatticeAlgebra, 2, (1, 0), jnp.asarray(1.0))
        evaluation_points = jnp.array([[0.5, 0.3]])  # Shape (1, 2)

        results = eval_algorithm_1_batch(poly, evaluation_points)

        assert results.shape == (1,)
        assert jnp.allclose(results[0], 0.5)

    def test_batch_multiple_points(self) -> None:
        """Test batch evaluation with multiple points."""
        poly = MultilinearPolynomial.from_monomial(LatticeAlgebra, 2, (1, 0), jnp.asarray(1.0))
        evaluation_points = jnp.array(
            [
                [0.5, 0.3],
                [0.2, 0.8],
                [0.9, 0.1],
            ]
        )

        results = eval_algorithm_1_batch(poly, evaluation_points)

        assert results.shape == (3,)
        assert jnp.allclose(results[0], 0.5)  # x_0 at [0.5, 0.3]
        assert jnp.allclose(results[1], 0.2)  # x_0 at [0.2, 0.8]
        assert jnp.allclose(results[2], 0.9)  # x_0 at [0.9, 0.1]

    def test_batch_evaluation_consistency(self) -> None:
        """Test that batch evaluation matches sequential evaluation."""
        poly = MultilinearPolynomial.zeros(LatticeAlgebra, num_states=3)
        poly = poly.set_monomial((1, 1, 0), jnp.asarray(1.0))

        evaluation_points = jnp.array(
            [
                [0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7],
                [0.8, 0.1, 0.9],
            ]
        )

        # Batch evaluation
        batch_results = eval_algorithm_1_batch(poly, evaluation_points)

        # Sequential evaluation
        sequential_results = []
        for point in evaluation_points:
            values = {i: point[i] for i in range(3)}
            result = eval_algorithm_1(poly, values)
            sequential_results.append(result)

        # Results should match
        sequential_results_array = jnp.array(sequential_results)
        assert jnp.allclose(batch_results, sequential_results_array)


class TestEvalAlgorithm1EdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_state_polynomial(self) -> None:
        """Test polynomial with single state."""
        poly = MultilinearPolynomial.from_monomial(LatticeAlgebra, 1, (1,), jnp.asarray(1.0))
        values = {0: jnp.array(0.7)}

        result = eval_algorithm_1(poly, values)

        assert jnp.allclose(result, 0.7)

    def test_all_states_in_monomial(self) -> None:
        """Test monomial with all states present."""
        poly = MultilinearPolynomial.from_monomial(LatticeAlgebra, 4, (1, 1, 1, 1), jnp.asarray(1.0))
        values = {0: jnp.array(0.2), 1: jnp.array(0.3), 2: jnp.array(0.4), 3: jnp.array(0.5)}

        result = eval_algorithm_1(poly, values)

        # Result: min(0.2, 0.3, 0.4, 0.5) = 0.2 (for MaxMin multiply is min)
        assert jnp.allclose(result, 0.2)

    def test_zero_coefficient(self) -> None:
        """Test polynomial with zero coefficient."""
        poly = MultilinearPolynomial.zeros(LatticeAlgebra, num_states=2)
        poly = poly.set_monomial((1, 0), jnp.asarray(0.0))  # Zero coefficient
        poly = poly.set_monomial((0, 1), jnp.asarray(1.0))

        values = {0: jnp.array(0.9), 1: jnp.array(0.1)}

        result = eval_algorithm_1(poly, values)

        # Zero from first term: 0*0.9 = 0
        # Second term: 1*0.1 = 0.1
        # Result: max(0, 0.1) = 0.1
        assert jnp.allclose(result, 0.1)

    def test_unit_coefficient(self) -> None:
        """Test polynomial with unit coefficient (1.0)."""
        poly = MultilinearPolynomial.from_monomial(LatticeAlgebra, 2, (0, 0), jnp.asarray(1.0))  # Constant 1
        values = {0: jnp.array(0.5), 1: jnp.array(0.3)}

        result = eval_algorithm_1(poly, values)

        # Constant term: 1 * 1 = 1
        assert jnp.allclose(result, 1.0)


class TestEvalAlgorithm1Semirings:
    """Test evaluation on different semiring types."""

    def test_maxmin_semiring(self) -> None:
        """Test evaluation with explicit MaxMin semiring."""
        poly = MultilinearPolynomial.zeros(MaxMinSemiring, num_states=2)
        poly = poly.set_monomial((1, 0), jnp.asarray(1.0))

        values = {0: jnp.array(0.7), 1: jnp.array(0.4)}

        result = eval_algorithm_1(poly, values)

        # MaxMin: multiply is min, so 1 * 0.7 = min(1, 0.7) = 0.7
        assert jnp.allclose(result, 0.7)
